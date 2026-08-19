#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repeated-inference stability assessment for LLM-derived clinical phenotype extraction.

Purpose
-------
This script reruns the same fixed prompt, same input text, same model, and same
client-side decoding settings multiple times on a random sample of encounters.
It then quantifies output stability across repeated runs.

Recommended reviewer-facing use
-------------------------------
- Use after final prompts/settings have been fixed.
- Use a random sample, e.g. 50-100 encounters, not the full cohort.
- Do not use the stability results to modify prompts or select settings unless
  prompt development is explicitly restricted to training data.

Example
-------
Set the API credential in the ``OPENAI_API_KEY`` environment variable (or your
secret manager) rather than passing it on the command line. Configure the endpoint
with ``OPENAI_BASE_URL`` or ``--base-url``. For example:

    export OPENAI_BASE_URL="https://your-openai-compatible-server.example/v1"

Then run:

    python llm_labeling_repeated_inference_stability.py \
      --input-csv ./data/cohort.csv \
      --text-dir ./data/notes \
      --output-dir ./results/llm_stability \
      --sample-size 100 \
      --n-runs 3 \
      --model YOUR_MODEL_NAME \
      --temperature 0.0 \
      --top-p 1.0 \
      --max-tokens 4000

Environment variables
---------------------
OPENAI_API_KEY must contain the API credential. For a trusted local/OpenAI-compatible
server that does not require authentication, set it explicitly to a non-secret placeholder
such as "not-required" if required by the client library.

OPENAI_BASE_URL may be used instead of --base-url. No endpoint is hard-coded in this script.

Privacy note
------------
Input notes and generated outputs may contain protected health information (PHI). Keep data
and result directories outside version control. Raw model responses are saved only when
--save-raw-json is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Feature schema
# -----------------------------------------------------------------------------

FIELDS: List[Dict[str, Any]] = [
    {
        "name": "documented_infection",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = a bacterial/fungal/viral infection is documented or strongly clinically diagnosed/suspected in the notes.
  false = notes explicitly suggest no infection / no evidence of infection.
  unknown = not enough information.""",
    },
    {
        "name": "documented_resistance",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "aliases": ["documented_resistence"],
        "definition": """true = microbiology or clinical documentation mentions antimicrobial resistance / reduced susceptibility (e.g., MRSA, ESBL, VRE, resistant organism).
  false = no resistance documented.
  unknown = no microbiology or unclear.""",
    },
    {
        "name": "clinical_impression",
        "type": "ordinal_int",
        "allowed_values": [1, 2],
        "definition": """Global clinical status during the relevant period (0-48h after antibiotic start, if inferable).
  1 = fairly okay / clinically stable / not severely ill.
  2 = bad / clinically ill / concerning deterioration or severe condition.
  Use physician/nursing overall impression, not only one lab value.""",
    },
    {
        "name": "probability_of_persisting_fever",
        "type": "ordinal_int",
        "allowed_values": [1, 2],
        "definition": """Your note-based estimate of whether fever will still persist >48h after antibiotic start.
  1 = low likelihood
  2 = high likelihood
  This is an ordinal risk rating, NOT a numeric probability.""",
    },
    {
        "name": "probability_icu",
        "type": "ordinal_int",
        "allowed_values": [1, 2],
        "definition": """Your note-based estimate of likelihood of ICU transfer/escalation.
  1 = low likelihood
  2 = high likelihood""",
    },
    {
        "name": "ab_change",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """Any documented change in antibiotic regimen during the encounter/relevant period, including start, stop, switch, escalation, de-escalation, or replacement.
  true = any such change documented
  false = no antibiotic change documented
  unknown = unclear / not enough information""",
    },
    {
        "name": "clinical_trajectory_0_48h",
        "type": "categorical",
        "allowed_values": ["improving", "stable", "worsening", "unknown"],
        "definition": """Trend in clinical condition in the first 48h after antibiotic start (if inferable).
  improving = symptoms/signs/labs/overall status improving
  stable = no clear improvement or worsening
  worsening = deterioration, escalation, increasing instability, worsening infection concern
  unknown = timing unclear or trajectory not documented""",
    },
    {
        "name": "source_control_needed",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = notes suggest infection source likely requires procedural/surgical control (e.g., drainage, debridement, line removal, biliary intervention)
  false = no such source control need documented
  unknown = unclear""",
    },
    {
        "name": "source_control_performed",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = source control procedure was documented as performed (e.g., abscess drained, catheter removed)
  false = not performed / not documented as done
  unknown = unclear whether performed""",
    },
    {
        "name": "pathogen_identified",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = a specific pathogen organism is documented from culture/PCR/etc. (not just \"infection\")
  false = no pathogen identified/documented
  unknown = microbiology not available/unclear""",
    },
    {
        "name": "persistent_positive_cultures",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = repeat cultures remain positive / persistent bacteremia or ongoing microbiological positivity documented
  false = no persistent positivity documented (includes clearance or no evidence of persistence)
  unknown = no repeat culture information / unclear""",
    },
    {
        "name": "mdro_suspected_or_confirmed",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = multidrug-resistant organism suspected or confirmed (e.g., MRSA, VRE, ESBL, carbapenem-resistant organism)
  false = no MDRO concern documented
  unknown = unclear""",
    },
    {
        "name": "empiric_abx_adequate",
        "type": "categorical",
        "allowed_values": ["likely", "unlikely", "uncertain", "unknown"],
        "definition": """Clinical judgment from notes whether empiric antibiotics likely covered the suspected pathogen/source.
  likely = notes suggest coverage appropriate/effective
  unlikely = notes suggest mismatch, insufficient coverage, or ineffective regimen
  uncertain = mixed signals / explicitly uncertain
  unknown = no basis in notes""",
    },
    {
        "name": "abx_escalation_due_to_failure",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = antibiotics were escalated/switched specifically due to persistent fever, clinical worsening, treatment failure, or microbiology results
  false = no such failure-driven escalation documented
  unknown = reason for change unclear or no information""",
    },
    {
        "name": "sepsis_or_shock",
        "type": "categorical",
        "allowed_values": ["none", "sepsis", "septic_shock", "unknown"],
        "definition": """none = no sepsis/shock documented or suggested
  sepsis = sepsis documented/suspected without shock
  septic_shock = septic shock documented/suspected (e.g., vasopressors, severe hemodynamic instability)
  unknown = unclear""",
    },
    {
        "name": "neutropenia",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = neutropenia documented/suspected (including febrile neutropenia context)
  false = neutropenia not documented
  unknown = no information""",
    },
    {
        "name": "profound_immunosuppression",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = strong immunosuppression documented (e.g., intensive chemotherapy, high-dose steroids, transplant, severe immune suppression)
  false = no strong immunosuppression documented
  unknown = unclear""",
    },
    {
        "name": "noninfectious_fever_suspected",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = notes favor/consider noninfectious fever cause (e.g., tumor fever, drug fever, thromboembolism, inflammatory cause)
  false = no noninfectious cause suspected/documented
  unknown = unclear""",
    },
    {
        "name": "diagnostic_uncertainty_high",
        "type": "boolish",
        "allowed_values": ["true", "false", "unknown"],
        "definition": """true = clinicians document unclear source/etiology, broad differential, ongoing search, uncertain diagnosis
  false = diagnosis/source appears reasonably clear in notes
  unknown = unclear""",
    },
    {
        "name": "infection_focus",
        "type": "categorical",
        "allowed_values": ["pneumonia", "uti", "intraabdominal", "line", "ssti", "cns", "endocarditis", "unknown", "other"],
        "definition": """Main documented/suspected infection source.
  pneumonia = pulmonary infection/pneumonia
  uti = urinary tract infection / pyelonephritis
  intraabdominal = intra-abdominal source (abscess, biliary, GI etc.)
  line = catheter/line-associated infection
  ssti = skin/soft tissue infection
  cns = CNS infection
  endocarditis = endocarditis suspected/confirmed
  unknown = no clear focus
  other = infectious focus exists but not covered above""",
    },
]

VALUE_FIELDS = [field["name"] for field in FIELDS]
CONF_FIELDS = [f"{name}_confidence" for name in VALUE_FIELDS]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def make_short_id(value: Any) -> str:
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def build_text_index(text_dir: Path, file_prefix: str, file_suffix: str) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for p in text_dir.glob(f"*{file_suffix}"):
        stem = p.stem
        if file_prefix:
            if not stem.startswith(file_prefix):
                continue
            stem = stem[len(file_prefix):]
        entity_id = stem.split("_", 1)[0]
        if entity_id:
            index[entity_id].append(p)
    return dict(index)


def read_text_files(paths: List[Path], encodings: List[str]) -> str:
    texts: List[str] = []
    for path in paths:
        content = None
        for enc in encodings:
            try:
                content = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                print(f"[TXT Error] {path}: {exc}")
                break
        if content:
            texts.append(content)
    return "\n\n".join(texts)


def extract_first_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    s = str(raw).strip()
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


def get_nested(dct: Any, key: str, default: Any = None) -> Any:
    if not isinstance(dct, dict):
        return default
    return dct.get(key, default)


def normalize_value(value: Any, field: Dict[str, Any]) -> str:
    """Normalize output for stability comparison while preserving unknown as a value."""
    allowed = [str(v).lower() for v in field.get("allowed_values", [])]
    if value is None:
        return "unknown"

    if field.get("type") == "ordinal_int":
        try:
            normalized = str(int(value))
        except Exception:
            return "unknown"
        return normalized if normalized in allowed else "unknown"

    if isinstance(value, bool):
        normalized = "true" if value else "false"
    else:
        normalized = str(value).strip().lower()

    return normalized if normalized in allowed else "unknown"


def parse_float01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return np.nan
    return min(max(v, 0.0), 1.0)


def get_field_object(result: Dict[str, Any], field: Dict[str, Any]) -> Any:
    if field["name"] in result:
        return result[field["name"]]
    for alias in field.get("aliases", []):
        if alias in result:
            return result[alias]
    return {}


def flatten_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten parsed JSON into normalized values and confidences."""
    row: Dict[str, Any] = {}
    if result is None:
        for field in FIELDS:
            row[field["name"]] = np.nan
            row[f"{field['name']}_confidence"] = np.nan
        return row

    for field in FIELDS:
        name = field["name"]
        obj = get_field_object(result, field)
        value = get_nested(obj, "value", obj)
        row[name] = normalize_value(value, field)
        row[f"{name}_confidence"] = parse_float01(get_nested(obj, "confidence"))
    return row


# -----------------------------------------------------------------------------
# Prompt and LLM call
# -----------------------------------------------------------------------------

def build_output_schema() -> str:
    schema = {}
    for field in FIELDS:
        name = field["name"]
        allowed = field.get("allowed_values", ["unknown"])
        if field["type"] == "ordinal_int":
            example_value = allowed[0]
        else:
            example_value = "|".join(str(v) for v in allowed)
        schema[name] = {
            "value": example_value,
            "confidence": 0.0,
            "evidence": ["..."]
        }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_prompt(text: str, encounter_id: str) -> str:
    field_blocks = []
    for field in FIELDS:
        field_blocks.append(f"- {field['name']}:\n  {field['definition']}")
    field_definitions = "\n\n".join(field_blocks)

    return f"""
You are a clinical expert reading German hospital notes for ONE hospital stay (encounter ID: {encounter_id}).

Task:
Extract structured phenotypes from the documents to help predict whether fever will persist >48h after start of antibiotic therapy.

Important rules:
- Use ONLY information explicitly documented in the notes.
- If a fact is not documented or unclear, return "unknown" (do not guess).
- Keep evidence snippets short (max 20 words each), copied/paraphrased from the notes.
- Return ONLY valid JSON (no markdown, no explanation).

Field definitions (use these meanings exactly):
{field_definitions}

Output schema (exact keys):
{build_output_schema()}

Documents:
{text}
""".strip()


def extract_raw_message(response: Any) -> str:
    """Get content from OpenAI SDK response, including vLLM reasoning field fallback."""
    msg = response.choices[0].message
    raw = getattr(msg, "content", None)
    if raw:
        return raw
    raw = getattr(msg, "reasoning", None)
    if raw:
        return raw
    extra = getattr(msg, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get("reasoning") or extra.get("content") or ""
    return ""


def query_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    sleep_seconds: float = 0.0,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "You extract structured clinical phenotypes from German notes and return strict JSON only."
        },
        {"role": "user", "content": prompt},
    ]

    last_error = None
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
            raw = extract_raw_message(response)
            parsed = extract_first_json_object(raw)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            return {
                "parsed": parsed,
                "raw": raw,
                "json_mode": use_json_mode,
                "parse_success": parsed is not None,
                "error": "",
            }
        except Exception as exc:
            last_error = str(exc)
            continue

    return {
        "parsed": None,
        "raw": "",
        "json_mode": False,
        "parse_success": False,
        "error": last_error or "unknown_error",
    }


# -----------------------------------------------------------------------------
# Stability metrics
# -----------------------------------------------------------------------------

def values_agree(values: pd.Series) -> bool:
    cleaned = values.fillna("__PARSE_FAILURE__").astype(str)
    return cleaned.nunique(dropna=False) <= 1


def pairwise_agreement(values: pd.Series) -> float:
    cleaned = values.fillna("__PARSE_FAILURE__").astype(str).tolist()
    n = len(cleaned)
    if n < 2:
        return np.nan
    total = 0
    agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            agree += int(cleaned[i] == cleaned[j])
    return agree / total if total else np.nan


def compute_stability_tables(runs_df: pd.DataFrame, output_dir: Path) -> None:
    value_fields = VALUE_FIELDS

    per_encounter_rows = []
    for enc_id, group in runs_df.groupby("encounter_short_id"):
        row = {
            "encounter_short_id": enc_id,
            "n_runs": len(group),
            "all_runs_parse_success": bool(group["parse_success"].all()),
            "parse_success_rate": float(group["parse_success"].mean()),
        }
        for feature in value_fields:
            row[f"{feature}_all_runs_agree"] = values_agree(group[feature])
            row[f"{feature}_pairwise_agreement"] = pairwise_agreement(group[feature])
        row["all_value_fields_identical"] = all(row[f"{feature}_all_runs_agree"] for feature in value_fields)
        row["mean_pairwise_value_agreement"] = float(np.nanmean([row[f"{feature}_pairwise_agreement"] for feature in value_fields]))
        per_encounter_rows.append(row)

    per_encounter = pd.DataFrame(per_encounter_rows)
    per_encounter.to_csv(output_dir / "llm_stability_per_encounter_agreement.csv", sep=";", index=False)

    feature_rows = []
    for feature in value_fields:
        agreement_col = f"{feature}_all_runs_agree"
        pairwise_col = f"{feature}_pairwise_agreement"
        feature_rows.append({
            "feature": feature,
            "n_encounters": int(per_encounter.shape[0]),
            "all_runs_exact_agreement_rate": float(per_encounter[agreement_col].mean()),
            "mean_pairwise_agreement": float(per_encounter[pairwise_col].mean()),
        })

    per_feature = pd.DataFrame(feature_rows).sort_values("all_runs_exact_agreement_rate", ascending=True)
    per_feature.to_csv(output_dir / "llm_stability_per_feature_agreement.csv", sep=";", index=False)

    summary = {
        "n_encounters": int(per_encounter.shape[0]),
        "n_runs_requested": int(runs_df["run"].nunique()),
        "n_total_calls": int(runs_df.shape[0]),
        "overall_parse_success_rate": float(runs_df["parse_success"].mean()),
        "encounter_level_all_features_exact_agreement_rate": float(per_encounter["all_value_fields_identical"].mean()),
        "mean_feature_exact_agreement_rate": float(per_feature["all_runs_exact_agreement_rate"].mean()),
        "mean_pairwise_feature_agreement": float(per_feature["mean_pairwise_agreement"].mean()),
    }
    with open(output_dir / "llm_stability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Stability summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nLeast stable features:")
    print(per_feature.head(10).to_string(index=False))


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def select_sample(df: pd.DataFrame, sample_size: int, seed: int, stratify_column: Optional[str]) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).copy()

    if stratify_column and stratify_column in df.columns and df[stratify_column].notna().nunique() > 1:
        # Proportional stratified sample with at least one from each stratum where possible.
        sampled_parts = []
        rng = np.random.default_rng(seed)
        counts = df[stratify_column].value_counts(dropna=True)
        total = counts.sum()
        remaining = sample_size
        for i, (level, count) in enumerate(counts.items()):
            group = df[df[stratify_column] == level]
            if i == len(counts) - 1:
                n = min(len(group), remaining)
            else:
                n = max(1, int(round(sample_size * count / total)))
                n = min(n, len(group), remaining)
            if n > 0:
                sampled_parts.append(group.sample(n=n, random_state=int(rng.integers(0, 1_000_000))))
                remaining -= n
        sampled = pd.concat(sampled_parts, axis=0)
        if len(sampled) < sample_size:
            extras = df.drop(index=sampled.index).sample(n=min(sample_size - len(sampled), len(df) - len(sampled)), random_state=seed)
            sampled = pd.concat([sampled, extras], axis=0)
        return sampled.sample(frac=1.0, random_state=seed).copy()

    return df.sample(n=sample_size, random_state=seed).copy()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated-inference stability assessment for LLM phenotype extraction.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--input-sep", default=";")
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="encounter_id")
    parser.add_argument("--file-prefix", default="Encounter_")
    parser.add_argument("--file-suffix", default=".txt")
    parser.add_argument("--text-encodings", default="utf-8,ISO-8859-1")
    parser.add_argument("--max-text-chars", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-column", default=None, help="Optional column for stratified sampling, e.g. outcome label.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL. May also be set with OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Exact model identifier used for inference; required for reproducibility.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional pause between API calls.")
    parser.add_argument("--save-raw-json", action="store_true", help="Save raw model outputs; can be large and may contain PHI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise ValueError(
            "No API base URL configured. Set OPENAI_BASE_URL or pass --base-url explicitly. "
            "The script intentionally has no default endpoint to avoid sending clinical text "
            "to an unintended service."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Store API credentials in the environment rather than "
            "passing them on the command line. For a trusted server that does not require "
            "authentication, set OPENAI_API_KEY to an explicit non-secret placeholder if "
            "required by your OpenAI-compatible client/server setup."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)

    data = pd.read_csv(args.input_csv, sep=args.input_sep)
    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in input CSV.")

    data = data.drop_duplicates(subset=[args.id_column]).copy()
    data["encounter_short_id"] = data[args.id_column].map(make_short_id)

    text_index = build_text_index(
        text_dir=Path(args.text_dir),
        file_prefix=args.file_prefix,
        file_suffix=args.file_suffix,
    )
    ids_with_text = set(text_index.keys())
    eligible = data[data["encounter_short_id"].astype(str).isin(ids_with_text)].copy()

    if eligible.empty:
        raise ValueError("No eligible encounters with matching text files found.")

    sample = select_sample(eligible, args.sample_size, args.seed, args.stratify_column)
    sample_ids = sample["encounter_short_id"].astype(str).tolist()
    sample[[args.id_column, "encounter_short_id"]].to_csv(output_dir / "llm_stability_sampled_encounters.csv", sep=";", index=False)

    print(f"Eligible encounters with text: {len(eligible)}")
    print(f"Sampled encounters: {len(sample_ids)}")
    print(f"Repeated runs per encounter: {args.n_runs}")

    encodings = [x.strip() for x in args.text_encodings.split(",")]
    rows = []
    raw_rows = []

    for enc_id in tqdm(sample_ids, desc="Encounters"):
        text = read_text_files(text_index.get(enc_id, []), encodings=encodings)
        if args.max_text_chars and len(text) > args.max_text_chars:
            text = text[: args.max_text_chars]
        prompt = build_prompt(text=text, encounter_id=enc_id)

        for run_idx in range(1, args.n_runs + 1):
            result = query_llm(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                sleep_seconds=args.sleep_seconds,
            )
            flat = flatten_result(result["parsed"])
            row = {
                "encounter_short_id": enc_id,
                "run": run_idx,
                "parse_success": result["parse_success"],
                "json_mode": result["json_mode"],
                "error": result["error"],
            }
            row.update(flat)
            rows.append(row)

            if args.save_raw_json:
                raw_rows.append({
                    "encounter_short_id": enc_id,
                    "run": run_idx,
                    "raw": result["raw"],
                })

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(output_dir / "llm_stability_runs_long.csv", sep=";", index=False)

    if args.save_raw_json:
        pd.DataFrame(raw_rows).to_csv(output_dir / "llm_stability_raw_outputs.csv", sep=";", index=False)

    compute_stability_tables(runs_df, output_dir)
    print(f"\nSaved stability outputs to: {output_dir}")


if __name__ == "__main__":
    main()
