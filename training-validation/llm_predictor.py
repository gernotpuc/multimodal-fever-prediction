"""
Generic LLM-based prediction pipeline for tabular data + per-entity text files.

This script is designed to be reproducible and GitHub-ready:
- no hard-coded cohort paths
- configurable model endpoint and task definition
- generic ID/file matching
- resume mode
- structured logging
- robust JSON parsing

Example:
    python generic_llm_prediction_pipeline.py \
        --input-csv data/input.csv \
        --text-dir data/notes \
        --output-csv outputs/predictions.csv \
        --id-column encounter_id \
        --file-prefix Encounter_ \
        --prediction-name persistent_fever_48_72h \
        --task-description "Predict whether persistent fever is present 48-72h after antibiotic start." \
        --positive-label "yes" \
        --negative-label "no" \
        --base-url "$OPENAI_BASE_URL" \
        --api-key "$OPENAI_API_KEY" \
        --model "$LLM_MODEL"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Generic parsing utilities
# -----------------------------------------------------------------------------

def extract_first_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first JSON object from an LLM response.

    Handles:
    - plain JSON
    - fenced JSON blocks
    - extra text before/after JSON
    """
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
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def parse_binary_label(value: Any, positive_label: str, negative_label: str) -> float:
    """Map configured positive/negative string labels to 1/0."""
    if value is None or pd.isna(value):
        return np.nan

    normalized = str(value).strip().lower()
    if normalized == positive_label.lower():
        return 1.0
    if normalized == negative_label.lower():
        return 0.0
    return np.nan


def parse_float_in_unit_interval(value: Any) -> float:
    """Parse confidence scores and clamp to [0, 1]."""
    try:
        parsed = float(value)
    except Exception:
        return np.nan
    return min(max(parsed, 0.0), 1.0)


def join_evidence(field_obj: Any, max_items: int = 3) -> float | str:
    """Join short evidence snippets from the model output."""
    if not isinstance(field_obj, dict):
        return np.nan

    evidence = field_obj.get("evidence", [])
    if not isinstance(evidence, list):
        return np.nan

    return " | ".join(str(item) for item in evidence[:max_items])


# -----------------------------------------------------------------------------
# File and data handling
# -----------------------------------------------------------------------------

def make_short_id(value: Any) -> str:
    """Use the final URL/path segment as the short ID."""
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def build_text_file_index(
    text_dir: Path,
    file_prefix: str = "",
    file_suffix: str = ".txt",
) -> Dict[str, List[Path]]:
    """
    Build an index from entity ID to text files.

    Expected filename pattern by default:
        <file_prefix><entity_id>*.txt

    Example:
        Encounter_123_note1.txt -> entity ID 123 if file_prefix='Encounter_'
        123_note1.txt           -> entity ID 123 if file_prefix=''
    """
    index: Dict[str, List[Path]] = defaultdict(list)

    for path in text_dir.glob(f"*{file_suffix}"):
        stem = path.stem

        if file_prefix:
            if not stem.startswith(file_prefix):
                continue
            stem = stem[len(file_prefix) :]

        entity_id = stem.split("_", 1)[0]
        if entity_id:
            index[entity_id].append(path)

    return dict(index)


def read_text_files(paths: Iterable[Path], encodings: List[str]) -> str:
    """Read and concatenate text files, trying multiple encodings."""
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
                logging.warning("Could not read %s: %s", path, exc)
                break

        if content:
            texts.append(content)

    return "\n\n".join(texts)


def summarize_structured_row(
    row: pd.Series,
    exclude_columns: set[str],
    max_value_chars: int = 200,
) -> str:
    """Convert structured tabular variables into a readable text block."""
    lines: List[str] = []

    for column, value in row.items():
        if column in exclude_columns:
            continue
        if pd.isna(value):
            continue

        text_value = str(value).strip()
        if not text_value:
            continue

        if len(text_value) > max_value_chars:
            text_value = text_value[:max_value_chars] + "..."

        lines.append(f"- {column}: {text_value}")

    return "\n".join(lines) if lines else "(no structured variables available)"


def get_unprocessed_entities(
    entity_ids: set[str],
    output_csv: Path,
    entity_short_id_column: str,
    prediction_column: str,
    csv_sep: str,
) -> set[str]:
    """
    Resume mode: return IDs that do not yet have a prediction in an existing output CSV.
    """
    if not output_csv.exists():
        return entity_ids

    existing = pd.read_csv(output_csv, sep=csv_sep)
    if entity_short_id_column not in existing.columns:
        logging.warning(
            "Existing output has no '%s' column. Processing all eligible entities.",
            entity_short_id_column,
        )
        return entity_ids

    if prediction_column not in existing.columns:
        logging.info("Existing output has no prediction column. Processing all eligible entities.")
        return entity_ids

    status = (
        existing.groupby(entity_short_id_column, dropna=False)[prediction_column]
        .apply(lambda s: s.notna().any())
        .reset_index(name="has_prediction")
    )

    processed = set(status.loc[status["has_prediction"], entity_short_id_column].astype(str))
    return entity_ids - processed


# -----------------------------------------------------------------------------
# Prompting and LLM call
# -----------------------------------------------------------------------------

def build_prompt(
    entity_id: str,
    prediction_name: str,
    task_description: str,
    structured_summary: str,
    unstructured_text: str,
    positive_label: str,
    negative_label: str,
    domain_context: str,
    extra_instructions: str,
) -> str:
    """Build a generic prediction prompt with a stable JSON schema."""
    return f"""
You are an expert annotator making one prediction for one entity.

Domain context:
{domain_context}

Entity ID: {entity_id}

Task:
{task_description}

Rules:
- Use only the information provided below.
- Return exactly one prediction: "{positive_label}" or "{negative_label}".
- If evidence is uncertain, choose the more likely option.
- Keep evidence snippets short.
- Return only valid JSON. Do not include markdown or explanations.
{extra_instructions}

Structured variables:
{structured_summary}

Unstructured text:
{unstructured_text}

Output schema:
{{
  "{prediction_name}": {{
    "value": "{positive_label}|{negative_label}",
    "confidence": 0.0,
    "evidence": ["..."]
  }}
}}
""".strip()


def query_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    prediction_name: str,
    max_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    """Call an OpenAI-compatible chat completion endpoint."""
    messages = [
        {
            "role": "system",
            "content": "Return strict JSON only. Do not include markdown or prose.",
        },
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

            if parsed is not None and prediction_name in parsed:
                return parsed

            logging.warning("Could not parse valid JSON for '%s'. Raw: %s", prediction_name, raw[:500])
        except Exception as exc:
            mode = "JSON mode" if use_json_mode else "fallback mode"
            logging.warning("LLM call failed in %s: %s", mode, exc)

    return None


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    text_dir = Path(args.text_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
    )

    data = pd.read_csv(input_csv, sep=args.input_sep)
    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in input CSV.")

    short_id_col = args.short_id_column
    data[short_id_col] = data[args.id_column].map(make_short_id)

    file_index = build_text_file_index(
        text_dir=text_dir,
        file_prefix=args.file_prefix,
        file_suffix=args.file_suffix,
    )
    ids_with_text = set(file_index.keys())

    eligible = (
        data.drop_duplicates(subset=[args.id_column])
        .loc[lambda df: df[short_id_col].astype(str).isin(ids_with_text)]
        .copy()
    )

    prediction_col = f"{args.prediction_name}_pred"
    label_col = f"{args.prediction_name}_pred_label"
    confidence_col = f"{args.prediction_name}_pred_confidence"
    evidence_col = f"{args.prediction_name}_pred_evidence"

    if args.resume:
        unprocessed_ids = get_unprocessed_entities(
            entity_ids=set(eligible[short_id_col].astype(str)),
            output_csv=output_csv,
            entity_short_id_column=short_id_col,
            prediction_column=prediction_col,
            csv_sep=args.output_sep,
        )
        eligible = eligible.loc[eligible[short_id_col].astype(str).isin(unprocessed_ids)].copy()

    logging.info("Text files indexed: %d", sum(len(v) for v in file_index.values()))
    logging.info("Entities with text files: %d", len(ids_with_text))
    logging.info("Eligible entities this run: %d", len(eligible))

    exclude_columns = set(args.exclude_columns or [])
    exclude_columns.update(
        {
            args.id_column,
            short_id_col,
            prediction_col,
            label_col,
            confidence_col,
            evidence_col,
        }
    )

    records: List[Dict[str, Any]] = []
    encodings = [encoding.strip() for encoding in args.text_encodings.split(",")]

    for _, row in tqdm(eligible.iterrows(), total=len(eligible)):
        entity_short_id = str(row[short_id_col])
        text = read_text_files(file_index.get(entity_short_id, []), encodings=encodings)

        if not text.strip():
            logging.warning("No readable text for entity %s", entity_short_id)
            continue

        structured_summary = summarize_structured_row(
            row=row,
            exclude_columns=exclude_columns,
            max_value_chars=args.max_structured_value_chars,
        )

        if args.max_text_chars and len(text) > args.max_text_chars:
            text = text[: args.max_text_chars]

        prompt = build_prompt(
            entity_id=entity_short_id,
            prediction_name=args.prediction_name,
            task_description=args.task_description,
            structured_summary=structured_summary,
            unstructured_text=text,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            domain_context=args.domain_context,
            extra_instructions=args.extra_instructions,
        )

        result = query_llm(
            client=client,
            model=args.model,
            prompt=prompt,
            prediction_name=args.prediction_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        if result is None:
            continue

        field_obj = result.get(args.prediction_name, {})
        raw_label = field_obj.get("value") if isinstance(field_obj, dict) else field_obj

        records.append(
            {
                short_id_col: entity_short_id,
                prediction_col: parse_binary_label(
                    raw_label,
                    positive_label=args.positive_label,
                    negative_label=args.negative_label,
                ),
                label_col: str(raw_label).strip().lower() if raw_label is not None else np.nan,
                confidence_col: parse_float_in_unit_interval(
                    field_obj.get("confidence") if isinstance(field_obj, dict) else np.nan
                ),
                evidence_col: join_evidence(field_obj, max_items=args.max_evidence_items),
            }
        )

    predictions = pd.DataFrame(
        records,
        columns=[short_id_col, prediction_col, label_col, confidence_col, evidence_col],
    )

    if predictions.empty:
        logging.warning("No predictions were generated.")
        for col in [prediction_col, label_col, confidence_col, evidence_col]:
            if col not in data.columns:
                data[col] = np.nan
    else:
        data = data.merge(predictions, on=short_id_col, how="left")

    if args.drop_short_id and short_id_col in data.columns:
        data = data.drop(columns=[short_id_col])

    data.to_csv(output_csv, sep=args.output_sep, index=False)

    n_positive = int((data[prediction_col] == 1).sum()) if prediction_col in data.columns else 0
    n_negative = int((data[prediction_col] == 0).sum()) if prediction_col in data.columns else 0
    n_missing = int(data[prediction_col].isna().sum()) if prediction_col in data.columns else 0

    logging.info("Saved output to: %s", output_csv)
    logging.info("Predicted entities this run: %d", len(predictions))
    logging.info(
        "Prediction counts: %s=%d, %s=%d, missing=%d",
        args.positive_label,
        n_positive,
        args.negative_label,
        n_negative,
        n_missing,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic LLM-based binary prediction from tabular data and text files."
    )

    parser.add_argument("--input-csv", required=True, help="Path to input tabular CSV.")
    parser.add_argument("--input-sep", default=",", help="Input CSV separator.")
    parser.add_argument("--text-dir", required=True, help="Directory containing text files.")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV.")
    parser.add_argument("--output-sep", default=";", help="Output CSV separator.")

    parser.add_argument("--id-column", default="encounter_id", help="Entity ID column in input CSV.")
    parser.add_argument("--short-id-column", default="entity_short_id", help="Temporary short-ID column name.")
    parser.add_argument("--file-prefix", default="Encounter_", help="Prefix before entity ID in text filenames.")
    parser.add_argument("--file-suffix", default=".txt", help="Text file suffix.")
    parser.add_argument(
        "--text-encodings",
        default="utf-8,ISO-8859-1",
        help="Comma-separated list of encodings to try when reading text files.",
    )

    parser.add_argument("--prediction-name", required=True, help="Name of prediction field in JSON output.")
    parser.add_argument("--task-description", required=True, help="Prediction task given to the LLM.")
    parser.add_argument("--positive-label", default="yes", help="Positive class label expected from model.")
    parser.add_argument("--negative-label", default="no", help="Negative class label expected from model.")
    parser.add_argument("--domain-context", default="General prediction task.", help="Optional domain context.")
    parser.add_argument("--extra-instructions", default="", help="Additional task-specific instructions.")
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=[],
        help="Columns to exclude from the structured prompt.",
    )

    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Can also use OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Can also use OPENAI_API_KEY.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum output tokens.")

    parser.add_argument("--max-text-chars", type=int, default=0, help="Optional text truncation length. 0 means no truncation.")
    parser.add_argument("--max-structured-value-chars", type=int, default=200)
    parser.add_argument("--max-evidence-items", type=int, default=3)

    parser.add_argument("--resume", action="store_true", help="Skip entities already labeled in output CSV.")
    parser.add_argument("--drop-short-id", action="store_true", help="Drop temporary short-ID column before saving.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    setup_logging(parsed_args.verbose)
    run_pipeline(parsed_args)
