#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repeated-inference stability testing for direct LLM yes/no prediction.

Adapted from the direct LLM prediction workflow:
- structured variables + German clinical notes
- direct prediction of persistent fever 48–72h after antibiotic initiation
- strict JSON output
- repeated inference on a random subset of encounters

Outputs:
- llm_prediction_stability_runs_long.csv
- llm_prediction_stability_per_encounter.csv
- llm_prediction_stability_summary.json
- llm_prediction_stability_sampled_encounters.csv

Configuration:
    Set the following environment variables before running:
    - OPENAI_BASE_URL: base URL of the OpenAI-compatible inference server
    - OPENAI_API_KEY: API credential for that server

    Do not hard-code or commit credentials. Use only endpoints approved for the
    data being processed. Clinical text and derived outputs may contain sensitive
    information and should remain outside version control. Sampled encounter IDs are
    written for reproducibility; evidence snippets and raw model outputs are opt-in.

Example:
    python llm_predictor_repeated_inference_stability.py \
        --input-csv ./data/cohort.csv \
        --text-dir ./data/notes \
        --output-dir ./results/llm_prediction_stability \
        --sample-size 100 \
        --n-runs 3 \
        --model YOUR_EXACT_MODEL_ID \
        --temperature 0.0 \
        --top-p 1.0 \
        --max-tokens 1200
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def extract_message_text(message: Any) -> str:
    """
    Some vLLM/Qwen endpoints may return JSON in message.content, whereas others
    may expose it in message.reasoning. This checks both.
    """
    content = getattr(message, "content", None)
    if content:
        return str(content)

    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return str(reasoning)

    try:
        dumped = message.model_dump()
    except Exception:
        dumped = message if isinstance(message, dict) else {}

    if isinstance(dumped, dict):
        for key in ("content", "reasoning"):
            value = dumped.get(key)
            if value:
                return str(value)

    return ""


def extract_first_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object from a model response."""
    if raw is None:
        return None

    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None

    return None


def parse_yes_no(value: Any) -> float:
    """Convert yes/no-like model output to 1/0/NaN."""
    if value is None:
        return np.nan
    s = str(value).strip().lower()
    if s in {"yes", "ja", "true", "1"}:
        return 1.0
    if s in {"no", "nein", "false", "0"}:
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
        return " | ".join(str(x) for x in evidence[:max_items])
    return np.nan


def make_short_id(value: Any) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def build_text_file_index(text_dir: Path, file_prefix: str, file_suffix: str) -> Dict[str, List[Path]]:
    id_to_files: Dict[str, List[Path]] = defaultdict(list)

    for p in text_dir.glob(f"*{file_suffix}"):
        stem = p.stem
        if file_prefix:
            if not stem.startswith(file_prefix):
                continue
            stem = stem[len(file_prefix):]

        enc_id = stem.split("_", 1)[0]
        if enc_id:
            id_to_files[enc_id].append(p)

    return dict(id_to_files)


def read_text_files(paths: List[Path], encodings: List[str]) -> str:
    texts = []
    for path in paths:
        content = None
        for encoding in encodings:
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                print(f"[TXT Error] {path}: {exc}")
                break

        if content:
            texts.append(content)

    return "\n\n".join(texts)


def summarize_structured_row(row: pd.Series, id_col: str, short_id_col: str, label_col: str) -> str:
    exclude_cols = {
        id_col,
        short_id_col,
        label_col,
        "persistent_fever_48_72h_pred",
        "persistent_fever_48_72h_pred_label",
        "persistent_fever_48_72h_pred_conf",
        "persistent_fever_48_72h_pred_evidence",
    }

    lines = []
    for col, val in row.items():
        if col in exclude_cols:
            continue
        if pd.isna(val):
            continue

        sval = str(val).strip()
        if not sval:
            continue
        if len(sval) > 200:
            sval = sval[:200] + "..."

        lines.append(f"- {col}: {sval}")

    return "\n".join(lines) if lines else "(no structured variables available)"


def sample_encounters(df: pd.DataFrame, sample_size: int, seed: int, label_col: str, stratified: bool) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy()

    if stratified and label_col in df.columns and df[label_col].nunique(dropna=True) == 2:
        sampled_parts = []
        for _, group in df.groupby(label_col, dropna=True):
            n_group = max(1, int(round(sample_size * len(group) / len(df))))
            n_group = min(n_group, len(group))
            sampled_parts.append(group.sample(n=n_group, random_state=seed))
        sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=seed)
        return sampled.head(sample_size).copy()

    return df.sample(n=sample_size, random_state=seed).copy()


def build_prediction_prompt(structured_summary: str, text: str, encounter_id: str) -> str:
    return f"""
You are a clinical expert reading German hospital notes for ONE hospital stay (encounter ID: {encounter_id}).

Task:
Predict whether this patient will have persistent fever during the time window 48 to 72 hours after start of antibiotic therapy.

Use BOTH:
1. structured clinical variables
2. unstructured clinical notes

Important rules:
- Base your decision only on the information provided below.
- Do NOT use information that clearly refers to time points after 72 hours after antibiotic start.
- You MUST return exactly one prediction: "yes" or "no".
- If the evidence is uncertain, still choose the more likely option.
- This is a direct yes/no prediction.
- Keep evidence snippets short (max 20 words each).
- Return ONLY valid JSON (no markdown, no explanation).

Definition:
- "yes" = fever is likely to still be present at any point in the 48 to 72 hour window after antibiotic start.
- "no" = fever is unlikely to persist into that 48 to 72 hour window.

Structured variables:
{structured_summary}

Clinical notes:
{text}

Output schema:
{{
  "persistent_fever_48_72h": {{
    "value": "yes|no",
    "confidence": 0.0,
    "evidence": ["..."]
  }}
}}
""".strip()


def query_llm_prediction(
    client: OpenAI,
    model: str,
    structured_summary: str,
    text: str,
    encounter_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = build_prediction_prompt(structured_summary, text, encounter_id)

    messages = [
        {
            "role": "system",
            "content": (
                "You make direct clinical predictions from structured variables and German clinical notes. "
                "Return strict JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    raw = ""
    parsed = None
    error = None
    used_json_mode = False

    for use_json_mode in (True, False):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            raw = extract_message_text(response.choices[0].message)
            parsed = extract_first_json_object(raw)

            if parsed is not None:
                used_json_mode = use_json_mode
                break

            error = "JSON parse failed"
        except Exception as exc:
            error = str(exc)
            raw = ""
            parsed = None

    field_obj = parsed.get("persistent_fever_48_72h", {}) if isinstance(parsed, dict) else {}
    raw_label = field_obj.get("value") if isinstance(field_obj, dict) else field_obj

    pred = parse_yes_no(raw_label)
    conf = parse_float01(field_obj.get("confidence") if isinstance(field_obj, dict) else np.nan)
    evidence = join_evidence(field_obj)

    return {
        "parse_success": parsed is not None and np.isfinite(pred),
        "prediction": pred,
        "prediction_label": str(raw_label).strip().lower() if raw_label is not None else np.nan,
        "confidence": conf,
        "evidence": evidence,
        "raw": raw,
        "used_json_mode": used_json_mode,
        "error": error,
    }


def pairwise_agreement(values: List[Any]) -> float:
    vals = [v for v in values if pd.notna(v)]
    if len(vals) < 2:
        return np.nan

    pairs = list(itertools.combinations(vals, 2))
    return float(np.mean([a == b for a, b in pairs]))


def summarize_stability(runs: pd.DataFrame, n_runs: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    per_encounter_rows = []

    for enc_id, grp in runs.groupby("encounter_short_id"):
        preds = grp["prediction"].tolist()
        valid_preds = [p for p in preds if pd.notna(p)]
        parse_successes = int(grp["parse_success"].sum())

        all_requested_exact = (
            parse_successes == n_runs
            and len(valid_preds) == n_runs
            and len(set(valid_preds)) == 1
        )

        successful_only_exact = len(valid_preds) >= 2 and len(set(valid_preds)) == 1
        pred_flip = len(set(valid_preds)) > 1 if len(valid_preds) >= 2 else np.nan

        conf_values = grp.loc[grp["confidence"].notna(), "confidence"].astype(float).values
        conf_sd = float(np.std(conf_values, ddof=1)) if len(conf_values) >= 2 else np.nan
        conf_range = float(np.max(conf_values) - np.min(conf_values)) if len(conf_values) >= 2 else np.nan

        per_encounter_rows.append({
            "encounter_short_id": enc_id,
            "n_calls": len(grp),
            "n_parse_success": parse_successes,
            "all_requested_runs_exact_agreement": all_requested_exact,
            "successful_runs_exact_agreement": successful_only_exact,
            "pairwise_prediction_agreement": pairwise_agreement(preds),
            "prediction_flip": pred_flip,
            "n_valid_predictions": len(valid_preds),
            "first_valid_prediction": valid_preds[0] if valid_preds else np.nan,
            "confidence_sd": conf_sd,
            "confidence_range": conf_range,
        })

    per_encounter = pd.DataFrame(per_encounter_rows)

    summary = {
        "n_encounters": int(per_encounter.shape[0]),
        "n_runs_requested": int(n_runs),
        "n_total_calls": int(runs.shape[0]),
        "overall_parse_success_rate": float(runs["parse_success"].mean()) if len(runs) else np.nan,
        "prediction_exact_agreement_all_requested_runs": float(per_encounter["all_requested_runs_exact_agreement"].mean()) if len(per_encounter) else np.nan,
        "prediction_exact_agreement_successful_runs": float(per_encounter["successful_runs_exact_agreement"].mean()) if len(per_encounter) else np.nan,
        "mean_pairwise_prediction_agreement": float(per_encounter["pairwise_prediction_agreement"].mean()) if len(per_encounter) else np.nan,
        "prediction_flip_rate_among_encounters_with_at_least_two_valid_runs": float(per_encounter["prediction_flip"].mean()) if len(per_encounter) else np.nan,
        "mean_confidence_sd": float(per_encounter["confidence_sd"].mean()) if len(per_encounter) else np.nan,
        "mean_confidence_range": float(per_encounter["confidence_range"].mean()) if len(per_encounter) else np.nan,
    }

    return per_encounter, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated-inference stability testing for prompt-based LLM yes/no predictions."
    )

    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--input-sep", default=";")
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="encounter_id")
    parser.add_argument("--label-column", default="persistent_fever_48_72h")
    parser.add_argument("--file-prefix", default="Encounter_")
    parser.add_argument("--file-suffix", default=".txt")
    parser.add_argument("--text-encodings", default="utf-8,ISO-8859-1")

    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint. If omitted, OPENAI_BASE_URL must be set.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact model identifier used for inference; required for reproducibility.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1200)

    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified-sample", action="store_true")
    parser.add_argument("--max-text-chars", type=int, default=0, help="Optional truncation. 0 means no truncation.")
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        help="Save model-provided evidence snippets. WARNING: evidence may contain sensitive clinical text.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model outputs in the long CSV. WARNING: raw outputs may contain sensitive clinical text.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Privacy reminder: input notes and generated CSV files may contain sensitive "
        "clinical information. Keep data/results outside version control."
    )

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    if not base_url:
        raise ValueError(
            "No API base URL configured. Provide --base-url or set OPENAI_BASE_URL. "
            "The script intentionally does not fall back to a default provider."
        )
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Credentials must be supplied via the environment; "
            "command-line API keys are intentionally unsupported."
        )

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    data = pd.read_csv(args.input_csv, sep=args.input_sep)
    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found.")

    short_id_col = "encounter_short_id"
    data[short_id_col] = data[args.id_column].map(make_short_id)

    text_dir = Path(args.text_dir)
    id_to_files = build_text_file_index(
        text_dir=text_dir,
        file_prefix=args.file_prefix,
        file_suffix=args.file_suffix,
    )

    eligible = (
        data.drop_duplicates(subset=[args.id_column])
        .loc[lambda df: df[short_id_col].astype(str).isin(set(id_to_files.keys()))]
        .copy()
    )

    sampled = sample_encounters(
        eligible,
        sample_size=args.sample_size,
        seed=args.seed,
        label_col=args.label_column,
        stratified=args.stratified_sample,
    )

    sampled[[args.id_column, short_id_col]].to_csv(
        output_dir / "llm_prediction_stability_sampled_encounters.csv",
        sep=";",
        index=False,
    )

    encodings = [x.strip() for x in args.text_encodings.split(",")]
    long_rows: List[Dict[str, Any]] = []

    print(f"Eligible encounters with TXT: {len(eligible)}")
    print(f"Sampled encounters: {len(sampled)}")
    print(f"Repeated runs per encounter: {args.n_runs}")

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Encounters"):
        enc_short_id = str(row[short_id_col])
        text = read_text_files(id_to_files.get(enc_short_id, []), encodings=encodings)

        if args.max_text_chars and len(text) > args.max_text_chars:
            text = text[:args.max_text_chars]

        structured_summary = summarize_structured_row(
            row=row,
            id_col=args.id_column,
            short_id_col=short_id_col,
            label_col=args.label_column,
        )

        for run_id in range(1, args.n_runs + 1):
            result = query_llm_prediction(
                client=client,
                model=args.model,
                structured_summary=structured_summary,
                text=text,
                encounter_id=enc_short_id,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

            row_out = {
                "encounter_short_id": enc_short_id,
                "run_id": run_id,
                "parse_success": result["parse_success"],
                "prediction": result["prediction"],
                "prediction_label": result["prediction_label"],
                "confidence": result["confidence"],
                "used_json_mode": result["used_json_mode"],
                "error": result["error"],
            }

            if args.label_column in row.index:
                row_out["y_true"] = row[args.label_column]

            if args.save_evidence:
                row_out["evidence"] = result["evidence"]

            if args.save_raw:
                row_out["raw"] = result["raw"]

            long_rows.append(row_out)

    runs = pd.DataFrame(long_rows)
    runs.to_csv(output_dir / "llm_prediction_stability_runs_long.csv", sep=";", index=False)

    per_encounter, summary = summarize_stability(runs, n_runs=args.n_runs)
    per_encounter.to_csv(output_dir / "llm_prediction_stability_per_encounter.csv", sep=";", index=False)

    with open(output_dir / "llm_prediction_stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Prediction stability summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()
