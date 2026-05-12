"""
Repeated in-context LLM prediction pipeline for tabular data + text notes.

GitHub-ready refactor of an in-context few-shot LLM predictor:
- no hard-coded paths or credentials
- OpenAI-compatible endpoint via CLI/env vars
- configurable data/text/label files
- repeated trials with different few-shot exemplars
- excludes the target case from its own exemplars
- saves per-repeat predictions, case-level pooled predictions, metrics, and variance
- optional resume mode

Example:
    python scripts/llm_predictor_incontext_repeated.py \
        --input-csv data/context.csv \
        --label-csv data/labels.csv \
        --text-dir data/notes \
        --output-dir artifacts/llm_incontext \
        --id-column encounter_id \
        --label-column fever \
        --prediction-name persistent_fever_48_72h \
        --n-fewshot 20 \
        --n-repeats 5 \
        --model "$LLM_MODEL"

Environment variables supported:
    OPENAI_BASE_URL
    OPENAI_API_KEY
    LLM_MODEL
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMInContextConfig:
    prediction_name: str
    positive_label: str = "yes"
    negative_label: str = "no"
    n_fewshot: int = 20
    n_repeats: int = 5
    seed: int = 42
    max_exemplar_note_chars: int = 1200
    max_target_note_chars: int = 12000
    max_structured_value_chars: int = 200
    max_tokens: int = 1400
    temperature: float = 0.0
    bootstrap_iterations: int = 2000


# -----------------------------------------------------------------------------
# Logging / I/O helpers
# -----------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_short_id(value: Any) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def build_text_file_index(text_dir: Path, *, file_prefix: str, file_suffix: str) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in text_dir.glob(f"*{file_suffix}"):
        stem = path.stem
        if file_prefix:
            if not stem.startswith(file_prefix):
                continue
            stem = stem[len(file_prefix):]
        entity_id = stem.split("_", 1)[0]
        if entity_id:
            index[entity_id].append(path)
    return dict(index)


def read_text_files(paths: Iterable[Path], encodings: Sequence[str]) -> str:
    texts: List[str] = []
    for path in paths:
        content = None
        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                LOGGER.warning("Could not read %s: %s", path, exc)
                break
        if content:
            texts.append(content)
    return "\n\n".join(texts)


def read_csv(path: Path, sep: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep=sep)


# -----------------------------------------------------------------------------
# JSON parsing and value parsing
# -----------------------------------------------------------------------------

def extract_first_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def normalize_binary_label(value: Any, *, positive_label: str = "yes", negative_label: str = "no") -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return positive_label if value else negative_label

    text = str(value).strip().lower()
    positive_values = {"1", "1.0", positive_label.lower(), "yes", "true", "positive", "pos"}
    negative_values = {"0", "0.0", negative_label.lower(), "no", "false", "negative", "neg"}

    if text in positive_values:
        return positive_label
    if text in negative_values:
        return negative_label
    return None


def parse_binary_prediction(value: Any, *, positive_label: str, negative_label: str) -> float:
    normalized = normalize_binary_label(value, positive_label=positive_label, negative_label=negative_label)
    if normalized == positive_label:
        return 1.0
    if normalized == negative_label:
        return 0.0
    return np.nan


def parse_float01(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return np.nan
    return min(max(parsed, 0.0), 1.0)


def join_evidence(field_obj: Any, max_items: int = 3) -> Any:
    if not isinstance(field_obj, dict):
        return np.nan
    evidence = field_obj.get("evidence", [])
    if isinstance(evidence, list):
        return " | ".join(str(item) for item in evidence[:max_items])
    return np.nan


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

def summarize_structured_row(
    row: pd.Series,
    *,
    exclude_columns: set[str],
    max_value_chars: int,
) -> str:
    lines: List[str] = []
    for column, value in row.items():
        if column in exclude_columns:
            continue
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if len(text) > max_value_chars:
            text = text[:max_value_chars] + "..."
        lines.append(f"- {column}: {text}")
    return "\n".join(lines) if lines else "(no structured variables available)"


def build_fewshot_block(
    *,
    current_entity_id: str,
    fewshot_pool: pd.DataFrame,
    text_index: Dict[str, List[Path]],
    encodings: Sequence[str],
    rng: np.random.Generator,
    n_fewshot: int,
    label_column_norm: str,
    prediction_name: str,
    positive_label: str,
    negative_label: str,
    exclude_columns: set[str],
    max_exemplar_note_chars: int,
    max_structured_value_chars: int,
) -> Tuple[str, List[str]]:
    pool = fewshot_pool.loc[fewshot_pool["entity_short_id"].astype(str) != str(current_entity_id)].copy()
    if pool.empty:
        return "No labeled examples available.", []

    n = min(n_fewshot, len(pool))
    sampled_indices = rng.choice(pool.index.to_numpy(), size=n, replace=False)
    sampled = pool.loc[sampled_indices]

    blocks: List[str] = []
    exemplar_ids: List[str] = []

    for j, (_, ex_row) in enumerate(sampled.iterrows(), start=1):
        ex_id = str(ex_row["entity_short_id"])
        ex_label = ex_row[label_column_norm]
        exemplar_ids.append(ex_id)

        structured = summarize_structured_row(
            ex_row,
            exclude_columns=exclude_columns,
            max_value_chars=max_structured_value_chars,
        )
        text = read_text_files(text_index.get(ex_id, []), encodings=encodings).strip()
        if len(text) > max_exemplar_note_chars:
            text = text[:max_exemplar_note_chars] + "\n[TRUNCATED]"

        blocks.append(
            f"""Example {j}
Structured variables:
{structured}

Clinical notes:
{text}

Correct label:
{{
  "{prediction_name}": {{
    "value": "{ex_label}"
  }}
}}"""
        )

    return "\n\n".join(blocks), exemplar_ids


def build_prompt(
    *,
    entity_id: str,
    prediction_name: str,
    task_description: str,
    definition_positive: str,
    definition_negative: str,
    fewshot_block: str,
    structured_summary: str,
    text: str,
    positive_label: str,
    negative_label: str,
    extra_instructions: str,
) -> str:
    return f"""
You are an expert annotator making one direct binary prediction for one case.

Case ID: {entity_id}

Task:
{task_description}

Use BOTH:
1. structured variables
2. unstructured notes

You are given labeled examples first. Use them only as in-context examples of the task.

Important rules:
- Base your decision only on the information provided below.
- Return exactly one prediction: "{positive_label}" or "{negative_label}".
- If the evidence is uncertain, still choose the more likely option.
- Keep evidence snippets short.
- Return only valid JSON. Do not include markdown or explanations.
{extra_instructions}

Definition:
- "{positive_label}" = {definition_positive}
- "{negative_label}" = {definition_negative}

Labeled examples:
{fewshot_block}

Now predict for the following target case.

Structured variables:
{structured_summary}

Clinical notes:
{text}

Output schema:
{{
  "{prediction_name}": {{
    "value": "{positive_label}|{negative_label}",
    "confidence": 0.0,
    "evidence": ["..."]
  }}
}}
""".strip()


# -----------------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------------

def query_llm(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": "Make a direct binary prediction and return strict JSON only."},
        {"role": "user", "content": prompt},
    ]

    for use_json_mode in (True, False):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            parsed = extract_first_json_object(raw)
            if parsed is not None:
                return parsed
            LOGGER.warning("Could not parse JSON. Raw: %s", raw[:500])
        except Exception as exc:
            mode = "JSON mode" if use_json_mode else "fallback mode"
            LOGGER.warning("LLM call failed in %s: %s", mode, exc)
    return None


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-9)
    sensitivity = tp / (tp + fn + 1e-9)
    npv = tn / (tn + fn + 1e-9)

    if len(np.unique(y_true)) < 2:
        auc_value = np.nan
        auprc_value = np.nan
    else:
        auc_value = float(roc_auc_score(y_true, y_prob))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        auprc_value = float(auc(recall_curve, precision_curve))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": auc_value,
        "auprc": auprc_value,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, *, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        rows.append(compute_binary_metrics(y_true[idx], y_pred[idx], y_prob[idx]))
    boot = pd.DataFrame(rows)
    summary = []
    for metric in boot.columns:
        mean, low, high = ci(boot[metric])
        summary.append({"metric": metric, "mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(summary)


def evaluate_predictions(predictions: pd.DataFrame, *, probability_col: str, pred_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    valid = predictions[["y_true", probability_col, pred_col]].dropna().copy()
    y_true = valid["y_true"].astype(int).to_numpy()
    y_pred = valid[pred_col].astype(int).to_numpy()
    y_prob = valid[probability_col].astype(float).to_numpy()
    return bootstrap_metrics(y_true, y_pred, y_prob, n_boot=n_boot, seed=seed)


def evaluate_per_repeat(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat, group in predictions.dropna(subset=["y_true", "pred", "probability"]).groupby("repeat"):
        metrics = compute_binary_metrics(
            group["y_true"].astype(int).to_numpy(),
            group["pred"].astype(int).to_numpy(),
            group["probability"].astype(float).to_numpy(),
        )
        rows.append({"repeat": int(repeat), **metrics})
    return pd.DataFrame(rows)


def summarize_repeat_variance(per_repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in per_repeat_metrics.columns if col != "repeat"]
    rows = []
    for metric in metric_cols:
        values = pd.to_numeric(per_repeat_metrics[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "metric": metric,
            "mean_across_repeats": float(values.mean()),
            "std_across_repeats": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min_across_repeats": float(values.min()),
            "max_across_repeats": float(values.max()),
            "n_repeats": int(values.shape[0]),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Prediction aggregation
# -----------------------------------------------------------------------------

def make_case_level_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions.dropna(subset=["pred"]).copy()
    pooled = valid.groupby("entity_short_id", as_index=False).agg(
        y_true=("y_true", "first"),
        pred_mean=("pred", "mean"),
        pred_std=("pred", "std"),
        confidence_mean=("confidence", "mean"),
        confidence_std=("confidence", "std"),
        n_repeats_observed=("repeat", "nunique"),
    )
    pooled["probability"] = pooled["pred_mean"]
    pooled["pred_majority"] = (pooled["pred_mean"] >= 0.5).astype(int)
    pooled["prediction_label_majority"] = np.where(pooled["pred_majority"] == 1, "yes", "no")
    pooled["FP_pred_majority"] = pooled["y_true"].eq(0) & pooled["pred_majority"].eq(1)
    pooled["FN_pred_majority"] = pooled["y_true"].eq(1) & pooled["pred_majority"].eq(0)
    return pooled


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def prepare_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, List[Path]], pd.DataFrame]:
    data = read_csv(Path(args.input_csv), args.input_sep)
    labels = read_csv(Path(args.label_csv), args.label_sep)

    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in input CSV.")
    if args.id_column not in labels.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in label CSV.")
    if args.label_column not in labels.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in label CSV.")

    data = data.copy()
    data["entity_short_id"] = data[args.id_column].map(make_short_id)

    labels = labels.copy()
    labels["entity_short_id"] = labels[args.id_column].map(make_short_id)
    label_subset = labels[["entity_short_id", args.label_column]].drop_duplicates("entity_short_id")
    label_subset["gold_label_norm"] = label_subset[args.label_column].apply(
        lambda x: normalize_binary_label(x, positive_label=args.positive_label, negative_label=args.negative_label)
    )
    label_subset["y_true"] = label_subset["gold_label_norm"].map({args.positive_label: 1, args.negative_label: 0})

    text_index = build_text_file_index(Path(args.text_dir), file_prefix=args.file_prefix, file_suffix=args.file_suffix)
    ids_with_text = set(text_index.keys())

    unique_cases = (
        data.drop_duplicates(subset=[args.id_column])
        .loc[lambda df: df["entity_short_id"].astype(str).isin(ids_with_text)]
        .merge(label_subset, on="entity_short_id", how="left")
        .copy()
    )

    labeled_pool = unique_cases.loc[unique_cases["gold_label_norm"].isin([args.positive_label, args.negative_label])].copy()

    LOGGER.info("Text files indexed: %d", sum(len(v) for v in text_index.values()))
    LOGGER.info("Eligible cases with text: %d", len(unique_cases))
    LOGGER.info("Few-shot labeled exemplar pool size: %d", len(labeled_pool))

    return unique_cases, text_index, labeled_pool


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = args.model or os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("Model must be supplied via --model or LLM_MODEL.")

    client = OpenAI(
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
    )

    config = LLMInContextConfig(
        prediction_name=args.prediction_name,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        n_fewshot=args.n_fewshot,
        n_repeats=args.n_repeats,
        seed=args.seed,
        max_exemplar_note_chars=args.max_exemplar_note_chars,
        max_target_note_chars=args.max_target_note_chars,
        max_structured_value_chars=args.max_structured_value_chars,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        bootstrap_iterations=args.bootstrap_iterations,
    )

    cases, text_index, fewshot_pool = prepare_data(args)
    encodings = [enc.strip() for enc in args.text_encodings.split(",")]

    prediction_col = f"{args.prediction_name}_pred"
    exclude_columns = {
        args.id_column,
        "entity_short_id",
        args.label_column,
        "gold_label_norm",
        "y_true",
        prediction_col,
        f"{args.prediction_name}_pred_label",
        f"{args.prediction_name}_pred_conf",
        f"{args.prediction_name}_pred_evidence",
    }
    exclude_columns.update(args.exclude_columns or [])

    if args.resume and (output_dir / "llm_incontext_predictions_all_repeats.csv").exists():
        existing = pd.read_csv(output_dir / "llm_incontext_predictions_all_repeats.csv")
        done_pairs = set(zip(existing["entity_short_id"].astype(str), existing["repeat"].astype(int)))
    else:
        done_pairs = set()

    records: List[Dict[str, Any]] = []

    for repeat in range(1, args.n_repeats + 1):
        rng = np.random.default_rng(args.seed + repeat - 1)
        LOGGER.info("Starting repeat %d/%d", repeat, args.n_repeats)

        for _, row in tqdm(cases.iterrows(), total=len(cases), desc=f"repeat {repeat}"):
            entity_id = str(row["entity_short_id"])
            if (entity_id, repeat) in done_pairs:
                continue

            text = read_text_files(text_index.get(entity_id, []), encodings=encodings).strip()
            if not text:
                LOGGER.warning("No text for case %s", entity_id)
                continue
            if len(text) > config.max_target_note_chars:
                text = text[:config.max_target_note_chars] + "\n\n[TRUNCATED]"

            fewshot_block, exemplar_ids = build_fewshot_block(
                current_entity_id=entity_id,
                fewshot_pool=fewshot_pool,
                text_index=text_index,
                encodings=encodings,
                rng=rng,
                n_fewshot=config.n_fewshot,
                label_column_norm="gold_label_norm",
                prediction_name=config.prediction_name,
                positive_label=config.positive_label,
                negative_label=config.negative_label,
                exclude_columns=exclude_columns,
                max_exemplar_note_chars=config.max_exemplar_note_chars,
                max_structured_value_chars=config.max_structured_value_chars,
            )

            structured_summary = summarize_structured_row(
                row,
                exclude_columns=exclude_columns,
                max_value_chars=config.max_structured_value_chars,
            )
            prompt = build_prompt(
                entity_id=entity_id,
                prediction_name=config.prediction_name,
                task_description=args.task_description,
                definition_positive=args.definition_positive,
                definition_negative=args.definition_negative,
                fewshot_block=fewshot_block,
                structured_summary=structured_summary,
                text=text,
                positive_label=config.positive_label,
                negative_label=config.negative_label,
                extra_instructions=args.extra_instructions,
            )

            result = query_llm(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )

            field_obj = result.get(config.prediction_name, {}) if result else {}
            raw_label = field_obj.get("value") if isinstance(field_obj, dict) else field_obj
            pred = parse_binary_prediction(raw_label, positive_label=config.positive_label, negative_label=config.negative_label)
            confidence = parse_float01(field_obj.get("confidence") if isinstance(field_obj, dict) else np.nan)

            records.append({
                "entity_short_id": entity_id,
                "repeat": repeat,
                "y_true": row.get("y_true", np.nan),
                "gold_label": row.get("gold_label_norm", np.nan),
                "prediction_label": str(raw_label).strip().lower() if raw_label is not None else np.nan,
                "pred": pred,
                "probability": pred if not pd.isna(pred) else np.nan,
                "confidence": confidence,
                "evidence": join_evidence(field_obj),
                "exemplar_ids": ",".join(exemplar_ids),
            })

        partial = pd.DataFrame(records)
        if args.save_partial and not partial.empty:
            if done_pairs:
                existing = pd.read_csv(output_dir / "llm_incontext_predictions_all_repeats.csv")
                pd.concat([existing, partial], ignore_index=True).drop_duplicates(["entity_short_id", "repeat"], keep="last").to_csv(
                    output_dir / "llm_incontext_predictions_all_repeats.csv", index=False
                )
            else:
                partial.to_csv(output_dir / "llm_incontext_predictions_all_repeats.csv", index=False)

    predictions = pd.DataFrame(records)
    if args.resume and (output_dir / "llm_incontext_predictions_all_repeats.csv").exists():
        previous = pd.read_csv(output_dir / "llm_incontext_predictions_all_repeats.csv")
        predictions = pd.concat([previous, predictions], ignore_index=True).drop_duplicates(["entity_short_id", "repeat"], keep="last")

    predictions.to_csv(output_dir / "llm_incontext_predictions_all_repeats.csv", index=False)
    case_level = make_case_level_predictions(predictions)
    case_level.to_csv(output_dir / "llm_incontext_predictions_case_level_pooled.csv", index=False)

    valid_repeat = predictions.dropna(subset=["y_true", "pred", "probability"])
    if not valid_repeat.empty and valid_repeat["y_true"].nunique() >= 2:
        pooled_metrics = evaluate_predictions(
            valid_repeat,
            probability_col="probability",
            pred_col="pred",
            n_boot=config.bootstrap_iterations,
            seed=config.seed,
        )
        pooled_metrics.insert(0, "aggregation", "all_repeat_predictions")
        pooled_metrics.to_csv(output_dir / "llm_incontext_metrics_pooled_all_repeats.csv", index=False)

        per_repeat = evaluate_per_repeat(valid_repeat)
        per_repeat.to_csv(output_dir / "llm_incontext_metrics_per_repeat.csv", index=False)
        summarize_repeat_variance(per_repeat).to_csv(output_dir / "llm_incontext_metric_variance_across_repeats.csv", index=False)

    valid_case = case_level.dropna(subset=["y_true", "pred_majority", "probability"])
    if not valid_case.empty and valid_case["y_true"].nunique() >= 2:
        case_metrics = evaluate_predictions(
            valid_case,
            probability_col="probability",
            pred_col="pred_majority",
            n_boot=config.bootstrap_iterations,
            seed=config.seed,
        )
        case_metrics.insert(0, "aggregation", "case_level_majority_vote")
        case_metrics.to_csv(output_dir / "llm_incontext_metrics_case_level.csv", index=False)

    metadata = {
        "config": asdict(config),
        "model": model,
        "input_csv": args.input_csv,
        "label_csv": args.label_csv,
        "text_dir": args.text_dir,
        "n_cases": int(len(cases)),
        "n_predictions": int(len(predictions)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Done. Outputs saved to %s", output_dir.resolve())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated in-context LLM prediction pipeline.")

    parser.add_argument("--input-csv", required=True, help="Input tabular CSV with case-level rows.")
    parser.add_argument("--input-sep", default=";", help="Input CSV separator.")
    parser.add_argument("--label-csv", required=True, help="CSV containing gold labels for exemplars/evaluation.")
    parser.add_argument("--label-sep", default=";", help="Label CSV separator.")
    parser.add_argument("--text-dir", required=True, help="Directory containing text files.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")

    parser.add_argument("--id-column", default="encounter_id")
    parser.add_argument("--label-column", default="fever")
    parser.add_argument("--prediction-name", default="persistent_fever_48_72h")
    parser.add_argument("--positive-label", default="yes")
    parser.add_argument("--negative-label", default="no")
    parser.add_argument("--file-prefix", default="Encounter_")
    parser.add_argument("--file-suffix", default=".txt")
    parser.add_argument("--text-encodings", default="utf-8,ISO-8859-1")

    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Can also use OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Can also use OPENAI_API_KEY.")
    parser.add_argument("--model", default=None, help="Model name. Can also use LLM_MODEL.")

    parser.add_argument("--n-fewshot", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-exemplar-note-chars", type=int, default=1200)
    parser.add_argument("--max-target-note-chars", type=int, default=12000)
    parser.add_argument("--max-structured-value-chars", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)

    parser.add_argument(
        "--task-description",
        default="Predict whether the target outcome will occur for this case.",
    )
    parser.add_argument(
        "--definition-positive",
        default="the target outcome is likely to occur.",
    )
    parser.add_argument(
        "--definition-negative",
        default="the target outcome is unlikely to occur.",
    )
    parser.add_argument("--extra-instructions", default="")
    parser.add_argument("--exclude-columns", nargs="*", default=[])

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-partial", action="store_true", help="Save predictions after each repeat.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    run_pipeline(args)
