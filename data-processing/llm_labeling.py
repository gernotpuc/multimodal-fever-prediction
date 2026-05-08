"""
Generic LLM-based labeling pipeline for extracting structured labels from text files.

This script is designed for reproducible GitHub use:
- no hard-coded cohort paths
- configurable OpenAI-compatible endpoint
- configurable entity/file matching
- configurable label schema via JSON
- resume mode
- robust JSON parsing
- confidence/evidence columns for QA

Example:
    python generic_llm_labeling_pipeline.py \
        --input-csv data/context.csv \
        --text-dir data/notes \
        --output-csv outputs/llm_labels.csv \
        --schema-json schemas/clinical_phenotypes.json \
        --id-column encounter_id \
        --file-prefix Encounter_ \
        --model "$LLM_MODEL" \
        --resume

Expected schema JSON format:
{
  "domain_context": "Clinical phenotype extraction from German hospital notes.",
  "task_description": "Extract structured phenotypes from the documents.",
  "global_rules": [
    "Use only information explicitly documented in the notes.",
    "If a fact is not documented or unclear, return unknown.",
    "Return only valid JSON."
  ],
  "fields": [
    {
      "name": "documented_infection",
      "type": "boolish",
      "description": "Whether infection is documented or strongly suspected.",
      "allowed_values": ["true", "false", "unknown"]
    },
    {
      "name": "clinical_impression",
      "type": "ordinal_int",
      "description": "Global clinical status during the relevant period.",
      "allowed_values": [1, 2]
    },
    {
      "name": "infection_focus",
      "type": "categorical",
      "description": "Main documented or suspected infection source.",
      "allowed_values": ["pneumonia", "uti", "intraabdominal", "line", "ssti", "cns", "endocarditis", "unknown", "other"]
    }
  ]
}
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
# Schema defaults
# -----------------------------------------------------------------------------

DEFAULT_CLINICAL_SCHEMA: Dict[str, Any] = {
    "domain_context": "Clinical phenotype extraction from German hospital notes.",
    "task_description": "Extract structured phenotypes from the documents to support downstream prediction or analysis.",
    "global_rules": [
        "Use only information explicitly documented in the notes.",
        "If a fact is not documented or unclear, return unknown; do not guess.",
        "Keep evidence snippets short, ideally fewer than 20 words each.",
        "Return only valid JSON. Do not include markdown or explanations.",
    ],
    "fields": [
        {
            "name": "documented_infection",
            "type": "boolish",
            "description": "Bacterial, fungal, or viral infection is documented, diagnosed, or strongly suspected.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "documented_resistance",
            "type": "boolish",
            "description": "Microbiology or clinical documentation mentions antimicrobial resistance or reduced susceptibility, such as MRSA, ESBL, VRE, or resistant organism.",
            "allowed_values": ["true", "false", "unknown"],
            "aliases": ["documented_resistence"],
        },
        {
            "name": "clinical_impression",
            "type": "ordinal_int",
            "description": "Global clinical status during the relevant period. 1 = fairly okay/stable; 2 = clinically ill, deteriorating, or severe condition.",
            "allowed_values": [1, 2],
        },
        {
            "name": "probability_of_persisting_fever",
            "type": "ordinal_int",
            "description": "Note-based estimate of whether fever will persist beyond 48h after antibiotic start. 1 = low likelihood; 2 = high likelihood.",
            "allowed_values": [1, 2],
        },
        {
            "name": "probability_icu",
            "type": "ordinal_int",
            "description": "Note-based estimate of ICU transfer or escalation likelihood. 1 = low likelihood; 2 = high likelihood.",
            "allowed_values": [1, 2],
        },
        {
            "name": "ab_change",
            "type": "boolish",
            "description": "Any documented antibiotic regimen change, including start, stop, switch, escalation, de-escalation, or replacement.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "clinical_trajectory_0_48h",
            "type": "categorical",
            "description": "Trend in clinical condition in the first 48h after antibiotic start, if inferable.",
            "allowed_values": ["improving", "stable", "worsening", "unknown"],
        },
        {
            "name": "source_control_needed",
            "type": "boolish",
            "description": "Notes suggest infection source likely requires procedural or surgical source control.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "source_control_performed",
            "type": "boolish",
            "description": "Source control procedure was documented as performed, such as drainage, debridement, catheter removal, or biliary intervention.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "pathogen_identified",
            "type": "boolish",
            "description": "Specific pathogen organism is documented from culture, PCR, or another microbiology source.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "persistent_positive_cultures",
            "type": "boolish",
            "description": "Repeat cultures remain positive or persistent bacteremia/microbiological positivity is documented.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "mdro_suspected_or_confirmed",
            "type": "boolish",
            "description": "Multidrug-resistant organism is suspected or confirmed, such as MRSA, VRE, ESBL, or carbapenem-resistant organism.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "empiric_abx_adequate",
            "type": "categorical",
            "description": "Clinical judgment from notes whether empiric antibiotics likely covered the suspected pathogen or source.",
            "allowed_values": ["likely", "unlikely", "uncertain", "unknown"],
        },
        {
            "name": "abx_escalation_due_to_failure",
            "type": "boolish",
            "description": "Antibiotics were escalated or switched due to persistent fever, clinical worsening, treatment failure, or microbiology results.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "sepsis_or_shock",
            "type": "categorical",
            "description": "Severity of sepsis/shock documentation.",
            "allowed_values": ["none", "sepsis", "septic_shock", "unknown"],
        },
        {
            "name": "neutropenia",
            "type": "boolish",
            "description": "Neutropenia is documented or suspected, including febrile neutropenia context.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "profound_immunosuppression",
            "type": "boolish",
            "description": "Strong immunosuppression is documented, such as intensive chemotherapy, high-dose steroids, transplant, or severe immune suppression.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "noninfectious_fever_suspected",
            "type": "boolish",
            "description": "Notes favor or consider a noninfectious fever cause, such as tumor fever, drug fever, thromboembolism, or inflammatory cause.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "diagnostic_uncertainty_high",
            "type": "boolish",
            "description": "Clinicians document unclear source or etiology, broad differential, ongoing search, or uncertain diagnosis.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "infection_focus",
            "type": "categorical",
            "description": "Main documented or suspected infection source.",
            "allowed_values": ["pneumonia", "uti", "intraabdominal", "line", "ssti", "cns", "endocarditis", "unknown", "other"],
        },
    ],
}


# -----------------------------------------------------------------------------
# Generic parsing utilities
# -----------------------------------------------------------------------------

def extract_first_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first JSON object from an LLM response.

    Handles plain JSON, fenced JSON blocks, and extra text before/after JSON.
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


def get_nested(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def parse_boolish(value: Any) -> Any:
    """Convert true/false/unknown-like values to True/False/NaN."""
    if isinstance(value, bool):
        return value
    if value is None:
        return np.nan

    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return np.nan


def parse_int_or_nan(value: Any, allowed: Optional[set[int]] = None) -> Any:
    if value is None:
        return np.nan
    try:
        parsed = int(value)
    except Exception:
        return np.nan

    if allowed is not None and parsed not in allowed:
        return np.nan
    return parsed


def parse_float_in_unit_interval(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return np.nan
    return min(max(parsed, 0.0), 1.0)


def parse_categorical(value: Any, allowed_values: Optional[List[Any]] = None) -> Any:
    if value is None:
        return np.nan

    parsed = str(value).strip().lower()
    if allowed_values is not None:
        allowed_normalized = {str(v).strip().lower() for v in allowed_values}
        if parsed not in allowed_normalized:
            return np.nan
    return parsed


def join_evidence(field_obj: Any, max_items: int = 3) -> Any:
    evidence = get_nested(field_obj, "evidence", [])
    if isinstance(evidence, list):
        return " | ".join(str(item) for item in evidence[:max_items])
    return np.nan


def parse_field_value(field_obj: Any, field: Dict[str, Any]) -> Any:
    value = get_nested(field_obj, "value", field_obj)
    field_type = field.get("type", "categorical")
    allowed_values = field.get("allowed_values")

    if field_type == "boolish":
        return parse_boolish(value)
    if field_type == "ordinal_int":
        allowed = set(int(v) for v in allowed_values) if allowed_values else None
        return parse_int_or_nan(value, allowed=allowed)
    if field_type == "float01":
        return parse_float_in_unit_interval(value)

    return parse_categorical(value, allowed_values=allowed_values)


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

    Expected filename pattern:
        <file_prefix><entity_id>*.txt

    Examples:
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


def get_unprocessed_entities(
    entity_ids: set[str],
    output_csv: Path,
    short_id_column: str,
    label_check_columns: List[str],
    csv_sep: str,
) -> set[str]:
    """Resume mode: return entity IDs without any existing label values."""
    if not output_csv.exists():
        return entity_ids

    existing = pd.read_csv(output_csv, sep=csv_sep)
    if short_id_column not in existing.columns:
        if "encounter_id" in existing.columns:
            existing[short_id_column] = existing["encounter_id"].map(make_short_id)
        else:
            logging.warning(
                "Existing output has no '%s' or encounter_id column. Processing all eligible entities.",
                short_id_column,
            )
            return entity_ids

    for column in label_check_columns:
        if column not in existing.columns:
            existing[column] = np.nan

    existing["has_any_llm_label"] = existing[label_check_columns].notna().any(axis=1)

    status = (
        existing.groupby(short_id_column, dropna=False)["has_any_llm_label"]
        .any()
        .reset_index()
    )

    processed = set(status.loc[status["has_any_llm_label"], short_id_column].astype(str))
    return entity_ids - processed


# -----------------------------------------------------------------------------
# Schema and prompt building
# -----------------------------------------------------------------------------

def load_schema(schema_json: Optional[str]) -> Dict[str, Any]:
    if schema_json is None:
        return DEFAULT_CLINICAL_SCHEMA

    with Path(schema_json).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    required = {"fields", "task_description"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"Schema is missing required keys: {sorted(missing)}")

    if not isinstance(schema["fields"], list) or not schema["fields"]:
        raise ValueError("Schema must contain a non-empty 'fields' list.")

    return schema


def format_allowed_values(values: List[Any]) -> str:
    return "|".join(str(value) for value in values)


def build_field_definitions(fields: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for field in fields:
        name = field["name"]
        description = field.get("description", "")
        allowed = field.get("allowed_values", [])
        allowed_text = f" Allowed values: {format_allowed_values(allowed)}." if allowed else ""
        lines.append(f"- {name}: {description}{allowed_text}")
    return "\n".join(lines)


def build_output_schema(fields: List[Dict[str, Any]]) -> str:
    schema: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        allowed_values = field.get("allowed_values", ["unknown"])
        example_value = allowed_values[0] if allowed_values else "unknown"
        schema[field["name"]] = {
            "value": example_value,
            "confidence": 0.0,
            "evidence": ["..."],
        }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_prompt(entity_id: str, text: str, schema: Dict[str, Any]) -> str:
    global_rules = schema.get("global_rules", [])
    rule_text = "\n".join(f"- {rule}" for rule in global_rules)
    field_definitions = build_field_definitions(schema["fields"])
    output_schema = build_output_schema(schema["fields"])

    return f"""
You are an expert annotator extracting structured labels for one entity.

Domain context:
{schema.get("domain_context", "General information extraction.")}

Entity ID: {entity_id}

Task:
{schema["task_description"]}

Rules:
{rule_text}

Field definitions:
{field_definitions}

Output schema, using exact keys:
{output_schema}

Documents:
{text}
""".strip()


# -----------------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------------

def query_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": "Extract structured labels and return strict JSON only.",
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

            if parsed is not None:
                return parsed

            logging.warning("Could not parse JSON. Raw: %s", raw[:500])
        except Exception as exc:
            mode = "JSON mode" if use_json_mode else "fallback mode"
            logging.warning("LLM call failed in %s: %s", mode, exc)

    return None


# -----------------------------------------------------------------------------
# Result flattening
# -----------------------------------------------------------------------------

def get_field_object(result: Dict[str, Any], field: Dict[str, Any]) -> Any:
    name = field["name"]
    if name in result:
        return result[name]

    for alias in field.get("aliases", []):
        if alias in result:
            return result[alias]

    return {}


def flatten_result(
    result: Dict[str, Any],
    entity_short_id: str,
    short_id_column: str,
    fields: List[Dict[str, Any]],
    max_evidence_items: int,
    include_confidence: bool,
    include_evidence: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {short_id_column: entity_short_id}

    for field in fields:
        name = field["name"]
        field_obj = get_field_object(result, field)

        output_name = field.get("output_name", name)
        record[output_name] = parse_field_value(field_obj, field)

        if include_confidence:
            record[f"{output_name}_conf"] = parse_float_in_unit_interval(
                get_nested(field_obj, "confidence")
            )

        if include_evidence:
            record[f"{output_name}_evidence"] = join_evidence(
                field_obj,
                max_items=max_evidence_items,
            )

    return record


def expected_output_columns(
    short_id_column: str,
    fields: List[Dict[str, Any]],
    include_confidence: bool,
    include_evidence: bool,
) -> List[str]:
    columns = [short_id_column]
    for field in fields:
        output_name = field.get("output_name", field["name"])
        columns.append(output_name)
        if include_confidence:
            columns.append(f"{output_name}_conf")
        if include_evidence:
            columns.append(f"{output_name}_evidence")
    return columns


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    text_dir = Path(args.text_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    schema = load_schema(args.schema_json)
    fields = schema["fields"]

    client = OpenAI(
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
    )

    data = pd.read_csv(input_csv, sep=args.input_sep)
    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in input CSV.")

    short_id_column = args.short_id_column
    data[short_id_column] = data[args.id_column].map(make_short_id)

    file_index = build_text_file_index(
        text_dir=text_dir,
        file_prefix=args.file_prefix,
        file_suffix=args.file_suffix,
    )
    ids_with_text = set(file_index.keys())

    entity_subset = (
        data.drop_duplicates(subset=[args.id_column])
        .loc[lambda df: df[short_id_column].astype(str).isin(ids_with_text)]
        .copy()
    )

    output_columns = expected_output_columns(
        short_id_column=short_id_column,
        fields=fields,
        include_confidence=args.include_confidence,
        include_evidence=args.include_evidence,
    )

    label_check_columns = args.label_check_columns or [
        field.get("output_name", field["name"]) for field in fields[: min(6, len(fields))]
    ]

    if args.resume:
        unprocessed_ids = get_unprocessed_entities(
            entity_ids=set(entity_subset[short_id_column].astype(str)),
            output_csv=output_csv,
            short_id_column=short_id_column,
            label_check_columns=label_check_columns,
            csv_sep=args.output_sep,
        )
        entity_subset = entity_subset.loc[
            entity_subset[short_id_column].astype(str).isin(unprocessed_ids)
        ].copy()

    logging.info("Text files indexed: %d", sum(len(v) for v in file_index.values()))
    logging.info("Entities with text files: %d", len(ids_with_text))
    logging.info("Eligible entities this run: %d", len(entity_subset))

    encodings = [encoding.strip() for encoding in args.text_encodings.split(",")]
    records: List[Dict[str, Any]] = []

    for _, row in tqdm(entity_subset.iterrows(), total=len(entity_subset)):
        entity_short_id = str(row[short_id_column])
        text = read_text_files(file_index.get(entity_short_id, []), encodings=encodings)

        if not text.strip():
            logging.warning("No readable text for entity %s", entity_short_id)
            continue

        if args.max_text_chars and len(text) > args.max_text_chars:
            text = text[: args.max_text_chars]

        prompt = build_prompt(entity_short_id, text, schema)
        result = query_llm(
            client=client,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        if result is None:
            continue

        records.append(
            flatten_result(
                result=result,
                entity_short_id=entity_short_id,
                short_id_column=short_id_column,
                fields=fields,
                max_evidence_items=args.max_evidence_items,
                include_confidence=args.include_confidence,
                include_evidence=args.include_evidence,
            )
        )

    labels = pd.DataFrame(records, columns=output_columns)

    if labels.empty:
        logging.warning("No labels were generated.")
        for column in output_columns[1:]:
            if column not in data.columns:
                data[column] = np.nan
    else:
        data = data.merge(labels, on=short_id_column, how="left")

    if args.drop_short_id and short_id_column in data.columns:
        data = data.drop(columns=[short_id_column])

    data.to_csv(output_csv, sep=args.output_sep, index=False)
    logging.info("Saved output to: %s", output_csv)
    logging.info("LLM-labeled entities this run: %d", len(labels))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic LLM-based labeling pipeline for text-associated tabular entities."
    )

    parser.add_argument("--input-csv", required=True, help="Path to input tabular CSV.")
    parser.add_argument("--input-sep", default=";", help="Input CSV separator.")
    parser.add_argument("--text-dir", required=True, help="Directory containing text files.")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV.")
    parser.add_argument("--output-sep", default=";", help="Output CSV separator.")

    parser.add_argument("--schema-json", default=None, help="Optional JSON schema defining extraction fields.")
    parser.add_argument("--id-column", default="encounter_id", help="Entity ID column in input CSV.")
    parser.add_argument("--short-id-column", default="entity_short_id", help="Temporary short-ID column name.")
    parser.add_argument("--file-prefix", default="Encounter_", help="Prefix before entity ID in text filenames.")
    parser.add_argument("--file-suffix", default=".txt", help="Text file suffix.")
    parser.add_argument(
        "--text-encodings",
        default="utf-8,ISO-8859-1",
        help="Comma-separated encodings to try when reading text files.",
    )

    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Can also use OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Can also use OPENAI_API_KEY.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum output tokens.")
    parser.add_argument("--max-text-chars", type=int, default=0, help="Optional text truncation length. 0 means no truncation.")

    parser.add_argument("--resume", action="store_true", help="Skip entities already labeled in output CSV.")
    parser.add_argument(
        "--label-check-columns",
        nargs="*",
        default=None,
        help="Columns used to decide whether an entity was already labeled in resume mode.",
    )
    parser.add_argument("--include-confidence", action="store_true", default=True, help="Write confidence columns.")
    parser.add_argument("--no-confidence", dest="include_confidence", action="store_false", help="Do not write confidence columns.")
    parser.add_argument("--include-evidence", action="store_true", default=True, help="Write evidence columns.")
    parser.add_argument("--no-evidence", dest="include_evidence", action="store_false", help="Do not write evidence columns.")
    parser.add_argument("--max-evidence-items", type=int, default=3, help="Maximum evidence snippets per field.")
    parser.add_argument("--drop-short-id", action="store_true", help="Drop temporary short-ID column before saving.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    setup_logging(parsed_args.verbose)
    run_pipeline(parsed_args)
