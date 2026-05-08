"""
Generic VLM-based image labeling pipeline for DICOM studies or image files.

This script is designed for reproducible GitHub use:
- no hard-coded paths or credentials
- OpenAI-compatible vision endpoint
- configurable feature schema via JSON
- DICOM-to-montage preprocessing
- optional PNG/JPEG image support
- resume mode
- robust JSON parsing
- confidence/evidence columns for QA

Example:
    python generic_vlm_image_labeling_pipeline.py \
        --input-csv data/context.csv \
        --image-dir data/images \
        --output-csv outputs/vlm_labels.csv \
        --schema-json schemas/thoracic_ct_features.json \
        --id-column encounter_id \
        --file-prefix Encounter_ \
        --image-extension .dcm \
        --model "$VLM_MODEL" \
        --resume

Environment variables supported:
    OPENAI_BASE_URL
    OPENAI_API_KEY
    VLM_MODEL
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
except ImportError:  # allows non-DICOM image workflows without pydicom installed
    pydicom = None


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
# Default schema
# -----------------------------------------------------------------------------

DEFAULT_VISUAL_SCHEMA: Dict[str, Any] = {
    "domain_context": "Visual feature extraction from medical images.",
    "task_description": "Review the provided image montage and extract the configured structured visual features.",
    "global_rules": [
        "Use only the provided images.",
        "Do not use clinical notes, tabular data, filenames, or outside assumptions.",
        "If a feature is not visible or cannot be determined from the provided images, return unknown.",
        "Be conservative and do not guess.",
        "Return only valid JSON. Do not include markdown or explanations.",
    ],
    "fields": [
        {
            "name": "focal_consolidation",
            "type": "boolish",
            "description": "Focal airspace consolidation is visible.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "ground_glass_opacities",
            "type": "boolish",
            "description": "Ground-glass opacities are visible.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "pulmonary_nodules",
            "type": "boolish",
            "description": "One or more pulmonary nodules are visible.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "halo_or_reversed_halo_sign",
            "type": "boolish",
            "description": "Halo sign or reversed-halo sign is visible.",
            "allowed_values": ["true", "false", "unknown"],
        },
        {
            "name": "pleural_effusion",
            "type": "boolish",
            "description": "Pleural effusion is visible.",
            "allowed_values": ["true", "false", "unknown"],
        },
    ],
}


# -----------------------------------------------------------------------------
# Generic parsing helpers
# -----------------------------------------------------------------------------

def extract_first_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response."""
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
# Data and file indexing
# -----------------------------------------------------------------------------

def make_short_id(value: Any) -> str:
    """Use the final URL/path segment as the short ID."""
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def extract_entity_id_from_filename(path: Path, file_prefix: str = "") -> str:
    """
    Extract entity ID from image filename.

    Pattern:
        <file_prefix><entity_id>_anything.ext
        <file_prefix><entity_id>.ext

    Examples:
        Encounter_12345_0001.dcm -> 12345 with file_prefix='Encounter_'
        12345_0001.dcm           -> 12345 with file_prefix=''
    """
    stem = path.stem

    if file_prefix:
        if not stem.startswith(file_prefix):
            return ""
        stem = stem[len(file_prefix) :]

    return stem.split("_", 1)[0]


def build_image_file_index(
    image_dir: Path,
    image_extension: str,
    file_prefix: str,
    recursive: bool = False,
) -> Dict[str, List[Path]]:
    pattern = f"**/*{image_extension}" if recursive else f"*{image_extension}"
    index: Dict[str, List[Path]] = defaultdict(list)

    for path in image_dir.glob(pattern):
        entity_id = extract_entity_id_from_filename(path, file_prefix=file_prefix)
        if entity_id:
            index[entity_id].append(path)

    return dict(index)


def get_unprocessed_entities(
    entity_ids: set[str],
    output_csv: Path,
    short_id_column: str,
    id_column: str,
    label_check_columns: List[str],
    csv_sep: str,
) -> set[str]:
    if not output_csv.exists():
        return entity_ids

    existing = pd.read_csv(output_csv, sep=csv_sep)

    if short_id_column not in existing.columns:
        if id_column in existing.columns:
            existing[short_id_column] = existing[id_column].map(make_short_id)
        else:
            logging.warning(
                "Existing output has neither '%s' nor '%s'. Processing all eligible entities.",
                short_id_column,
                id_column,
            )
            return entity_ids

    for column in label_check_columns:
        if column not in existing.columns:
            existing[column] = np.nan

    existing["has_any_vlm_label"] = existing[label_check_columns].notna().any(axis=1)

    status = (
        existing.groupby(short_id_column, dropna=False)["has_any_vlm_label"]
        .any()
        .reset_index()
    )

    processed = set(status.loc[status["has_any_vlm_label"], short_id_column].astype(str))
    return entity_ids - processed


# -----------------------------------------------------------------------------
# DICOM and image processing
# -----------------------------------------------------------------------------

def get_slice_position(ds: Any) -> float:
    """Return a numeric position for slice sorting."""
    try:
        if hasattr(ds, "ImagePositionPatient"):
            ipp = ds.ImagePositionPatient
            if len(ipp) >= 3:
                return float(ipp[2])
    except Exception:
        pass

    try:
        if hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
    except Exception:
        pass

    try:
        if hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
    except Exception:
        pass

    return 0.0


def apply_ct_window(
    ds: Any,
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Convert raw CT pixels to 8-bit windowed image."""
    arr = pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept

    low = window_center - window_width / 2
    high = window_center + window_width / 2

    hu = np.clip(hu, low, high)
    hu = (hu - low) / (high - low)
    return (hu * 255.0).astype(np.uint8)


def read_dicom_as_uint8(
    path: Path,
    window_center: float,
    window_width: float,
) -> Tuple[float, Image.Image]:
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM input. Install it with: pip install pydicom")

    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM file has no PixelData")

    pixel_array = ds.pixel_array
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[pixel_array.shape[0] // 2]

    img_np = apply_ct_window(ds, pixel_array, window_center, window_width)
    return get_slice_position(ds), Image.fromarray(img_np).convert("L")


def read_standard_image(path: Path) -> Tuple[float, Image.Image]:
    img = Image.open(path).convert("L")
    return 0.0, img


def resize_and_pad(img: Image.Image, target_size: int) -> Image.Image:
    img = img.convert("L")
    width, height = img.size

    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (target_size, target_size), color=0)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    canvas.paste(img, (x_offset, y_offset))
    return canvas


def choose_evenly_spaced(items: List[Any], n: int) -> List[Any]:
    if len(items) <= n:
        return items
    indices = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in indices]


def build_montage(
    paths: List[Path],
    image_extension: str,
    max_slices: int,
    montage_cols: int,
    cell_size: int,
    window_center: float,
    window_width: float,
) -> Optional[Image.Image]:
    """Build a montage from DICOM slices or standard images."""
    if not paths:
        return None

    images: List[Tuple[float, Image.Image]] = []
    is_dicom = image_extension.lower() in {".dcm", ".dicom"}

    for path in paths:
        try:
            if is_dicom:
                images.append(read_dicom_as_uint8(path, window_center, window_width))
            else:
                images.append(read_standard_image(path))
        except Exception as exc:
            logging.warning("Could not read image %s: %s", path, exc)

    if not images:
        return None

    images.sort(key=lambda item: item[0])
    sampled = choose_evenly_spaced(images, max_slices)

    tiles = [resize_and_pad(img, target_size=cell_size) for _, img in sampled]
    rows = int(np.ceil(len(tiles) / montage_cols))
    montage = Image.new("L", (montage_cols * cell_size, rows * cell_size), color=0)

    for idx, tile in enumerate(tiles):
        row = idx // montage_cols
        col = idx % montage_cols
        montage.paste(tile, (col * cell_size, row * cell_size))

    return montage.convert("RGB")


def pil_to_data_url(img: Image.Image, jpeg_quality: int) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# -----------------------------------------------------------------------------
# Schema and prompt building
# -----------------------------------------------------------------------------

def load_schema(schema_json: Optional[str]) -> Dict[str, Any]:
    if schema_json is None:
        return DEFAULT_VISUAL_SCHEMA

    with Path(schema_json).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    required = {"fields", "task_description"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"Schema missing required keys: {sorted(missing)}")

    if not isinstance(schema["fields"], list) or not schema["fields"]:
        raise ValueError("Schema must contain a non-empty 'fields' list.")

    return schema


def format_allowed_values(values: List[Any]) -> str:
    return "|".join(str(value) for value in values)


def build_field_definitions(fields: List[Dict[str, Any]]) -> str:
    lines = []
    for field in fields:
        allowed = field.get("allowed_values", [])
        allowed_text = f" Allowed values: {format_allowed_values(allowed)}." if allowed else ""
        lines.append(f"- {field['name']}: {field.get('description', '')}{allowed_text}")
    return "\n".join(lines)


def build_output_schema(fields: List[Dict[str, Any]]) -> str:
    schema: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        allowed_values = field.get("allowed_values", ["unknown"])
        schema[field["name"]] = {
            "value": allowed_values[0] if allowed_values else "unknown",
            "confidence": 0.0,
            "evidence": ["short visual evidence"],
        }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_prompt(entity_id: str, schema: Dict[str, Any]) -> str:
    rules = "\n".join(f"- {rule}" for rule in schema.get("global_rules", []))
    field_definitions = build_field_definitions(schema["fields"])
    output_schema = build_output_schema(schema["fields"])

    return f"""
You are an expert visual annotation assistant.

Domain context:
{schema.get("domain_context", "Visual feature extraction.")}

Entity ID: {entity_id}

Task:
{schema["task_description"]}

Rules:
{rules}

Feature definitions:
{field_definitions}

Output schema, using exact keys:
{output_schema}
""".strip()


# -----------------------------------------------------------------------------
# VLM call
# -----------------------------------------------------------------------------

def query_vlm(
    client: OpenAI,
    model: str,
    prompt: str,
    image: Image.Image,
    jpeg_quality: int,
    max_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    image_data_url = pil_to_data_url(image, jpeg_quality=jpeg_quality)

    messages = [
        {
            "role": "system",
            "content": "Extract structured visual labels from image input and return strict JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
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
            logging.warning("VLM call failed in %s: %s", mode, exc)

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
    include_confidence: bool,
    include_evidence: bool,
    max_evidence_items: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {short_id_column: entity_short_id}

    for field in fields:
        field_obj = get_field_object(result, field)
        output_name = field.get("output_name", field["name"])

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
    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    schema = load_schema(args.schema_json)
    fields = schema["fields"]

    model = args.model or os.getenv("VLM_MODEL")
    if not model:
        raise ValueError("Model must be provided via --model or VLM_MODEL.")

    client = OpenAI(
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
    )

    data = pd.read_csv(input_csv, sep=args.input_sep)
    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in input CSV.")

    short_id_column = args.short_id_column
    data[short_id_column] = data[args.id_column].map(make_short_id)

    image_index = build_image_file_index(
        image_dir=image_dir,
        image_extension=args.image_extension,
        file_prefix=args.file_prefix,
        recursive=args.recursive,
    )
    ids_with_images = set(image_index.keys())

    entity_subset = (
        data.drop_duplicates(subset=[args.id_column])
        .loc[lambda df: df[short_id_column].astype(str).isin(ids_with_images)]
        .copy()
    )

    output_columns = expected_output_columns(
        short_id_column=short_id_column,
        fields=fields,
        include_confidence=args.include_confidence,
        include_evidence=args.include_evidence,
    )

    label_check_columns = args.label_check_columns or [
        field.get("output_name", field["name"]) for field in fields[: min(5, len(fields))]
    ]

    if args.resume:
        unprocessed_ids = get_unprocessed_entities(
            entity_ids=set(entity_subset[short_id_column].astype(str)),
            output_csv=output_csv,
            short_id_column=short_id_column,
            id_column=args.id_column,
            label_check_columns=label_check_columns,
            csv_sep=args.output_sep,
        )
        entity_subset = entity_subset.loc[
            entity_subset[short_id_column].astype(str).isin(unprocessed_ids)
        ].copy()

    logging.info("Image files indexed: %d", sum(len(v) for v in image_index.values()))
    logging.info("Entities with image files: %d", len(ids_with_images))
    logging.info("Eligible entities this run: %d", len(entity_subset))

    records: List[Dict[str, Any]] = []

    for _, row in tqdm(entity_subset.iterrows(), total=len(entity_subset)):
        entity_short_id = str(row[short_id_column])

        montage = build_montage(
            paths=image_index.get(entity_short_id, []),
            image_extension=args.image_extension,
            max_slices=args.max_slices,
            montage_cols=args.montage_cols,
            cell_size=args.cell_size,
            window_center=args.window_center,
            window_width=args.window_width,
        )

        if montage is None:
            logging.warning("No readable image montage for entity %s", entity_short_id)
            continue

        prompt = build_prompt(entity_short_id, schema)
        result = query_vlm(
            client=client,
            model=model,
            prompt=prompt,
            image=montage,
            jpeg_quality=args.jpeg_quality,
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
                include_confidence=args.include_confidence,
                include_evidence=args.include_evidence,
                max_evidence_items=args.max_evidence_items,
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
    logging.info("VLM-labeled entities this run: %d", len(labels))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic VLM-based labeling pipeline for DICOM or image files."
    )

    parser.add_argument("--input-csv", required=True, help="Path to input tabular CSV.")
    parser.add_argument("--input-sep", default=";", help="Input CSV separator.")
    parser.add_argument("--image-dir", required=True, help="Directory containing DICOM/image files.")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV.")
    parser.add_argument("--output-sep", default=";", help="Output CSV separator.")

    parser.add_argument("--schema-json", default=None, help="Optional JSON schema defining visual features.")
    parser.add_argument("--id-column", default="encounter_id", help="Entity ID column in input CSV.")
    parser.add_argument("--short-id-column", default="entity_short_id", help="Temporary short-ID column name.")
    parser.add_argument("--file-prefix", default="Encounter_", help="Prefix before entity ID in image filenames.")
    parser.add_argument("--image-extension", default=".dcm", help="Image extension, e.g. .dcm, .png, .jpg.")
    parser.add_argument("--recursive", action="store_true", help="Search image directory recursively.")

    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Can also use OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Can also use OPENAI_API_KEY.")
    parser.add_argument("--model", default=None, help="Vision model name. Can also use VLM_MODEL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum output tokens.")

    parser.add_argument("--max-slices", type=int, default=12, help="Maximum slices/images per entity montage.")
    parser.add_argument("--montage-cols", type=int, default=4, help="Number of columns in montage.")
    parser.add_argument("--cell-size", type=int, default=256, help="Tile size in pixels.")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for API image payload.")
    parser.add_argument("--window-center", type=float, default=-600, help="DICOM CT window center.")
    parser.add_argument("--window-width", type=float, default=1500, help="DICOM CT window width.")

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
