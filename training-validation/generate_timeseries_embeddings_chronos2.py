#!/usr/bin/env python3
"""Generate encounter-level Chronos-2 embeddings from body-temperature series.

This script is a publication-oriented extraction of the time-series embedding
stage used in ``mm_NC_revisions_ume_nested_cv_early_fusion_c_grid-2.ipynb``.

Study implementation preserved
------------------------------
- Model: ``amazon/chronos-2`` via ``Chronos2Pipeline``.
- Input: semicolon-separated long-format CSV with ``encounter_id``,
  ``recorded_time``, and ``value``.
- Encounter IDs are normalized to ``Encounter/<id>``.
- Duplicate timestamps are averaged.
- Series are resampled to a 5-hour grid using the mean.
- Short gaps are time-interpolated with ``limit=2`` and
  ``limit_direction='both'``.
- Series shorter than four values are edge-padded to four values.
- Only the most recent 512 values are retained.
- ``pipeline.embed`` is used; token-level embeddings are mean-pooled to one
  vector per encounter.
- Embeddings are stored as float32 ``.npy`` plus a CSV containing
  ``encounter_id`` and ``embedding_row``.

Example
-------
python generate_timeseries_embeddings_chronos2.py \
    --input-csv data/ts_ume/body_temperature.csv \
    --output-embeddings data/processed/ume_body_temperature_chronos2_embeddings.npy \
    --output-index data/processed/ume_body_temperature_chronos2_index.csv

Raw clinical data and model weights are not distributed with this repository.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger("generate_timeseries_embeddings_chronos2")

ID_COL = "encounter_id"
DEFAULT_MODEL = "amazon/chronos-2"
DEFAULT_ID_COL_RAW = "encounter_id"
DEFAULT_TIME_COL = "recorded_time"
DEFAULT_VALUE_COL = "value"
DEFAULT_RESAMPLE_RULE = "5h"
DEFAULT_MAX_CONTEXT_LENGTH = 512
DEFAULT_MIN_POINTS = 4
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEPARATOR = ";"


def normalize_encounter_reference(value) -> str | float:
    """Normalize an encounter identifier to ``Encounter/<id>``."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower().endswith(".txt"):
        value = value[:-4]

    value = value.replace("\\", "/").strip()

    if value.lower().startswith("encounter_"):
        suffix = value.split("_", 1)[1]
        return f"Encounter/{suffix}"

    if value.lower().startswith("encounter/"):
        suffix = value.split("/", 1)[1]
        return f"Encounter/{suffix}"

    return f"Encounter/{value}"


def load_body_temperature_long_csv(
    path: Path,
    id_col_raw: str = DEFAULT_ID_COL_RAW,
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
    id_col_target: str = ID_COL,
    separator: str = DEFAULT_SEPARATOR,
) -> pd.DataFrame:
    """Load and clean long-format body-temperature measurements."""
    path = Path(path)
    df = pd.read_csv(path, sep=separator)

    required = [id_col_raw, time_col, value_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\nFound columns: {list(df.columns)}"
        )

    df = df[[id_col_raw, time_col, value_col]].copy()
    df = df.rename(
        columns={
            id_col_raw: id_col_target,
            time_col: "recorded_time",
            value_col: "body_temperature",
        }
    )

    df[id_col_target] = df[id_col_target].apply(normalize_encounter_reference)
    df["recorded_time"] = pd.to_datetime(df["recorded_time"], errors="coerce")

    # Preserve support for both decimal dots and decimal commas.
    df["body_temperature"] = (
        df["body_temperature"].astype(str).str.replace(",", ".", regex=False)
    )
    df["body_temperature"] = pd.to_numeric(
        df["body_temperature"], errors="coerce"
    )

    df = df.dropna(
        subset=[id_col_target, "recorded_time", "body_temperature"]
    )
    return df.sort_values([id_col_target, "recorded_time"]).reset_index(drop=True)


def build_time_series_tensors_from_long_df(
    df_long: pd.DataFrame,
    id_col: str = ID_COL,
    time_col: str = "recorded_time",
    value_col: str = "body_temperature",
    resample_rule: str | None = DEFAULT_RESAMPLE_RULE,
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
    min_points: int = DEFAULT_MIN_POINTS,
) -> tuple[list[str], list[torch.Tensor]]:
    """Convert long-format temperatures to one Chronos input tensor per encounter."""
    if max_context_length <= 0:
        raise ValueError("max_context_length must be > 0")
    if min_points <= 0:
        raise ValueError("min_points must be > 0")

    df_long = df_long.copy()
    df_long[id_col] = df_long[id_col].apply(normalize_encounter_reference)
    df_long[time_col] = pd.to_datetime(df_long[time_col], errors="coerce")
    df_long[value_col] = pd.to_numeric(df_long[value_col], errors="coerce")
    df_long = df_long.dropna(subset=[id_col, time_col, value_col])

    encounter_ids: list[str] = []
    series_tensors: list[torch.Tensor] = []
    skipped_no_valid_time = 0
    skipped_empty = 0

    for encounter_reference, group in df_long.groupby(id_col):
        group = group[[time_col, value_col]].copy()
        group = group.dropna(subset=[time_col, value_col])

        if len(group) == 0:
            skipped_empty += 1
            continue

        group[time_col] = pd.to_datetime(group[time_col], errors="coerce")
        group[value_col] = pd.to_numeric(group[value_col], errors="coerce")
        group = group.dropna(subset=[time_col, value_col])

        if len(group) == 0:
            skipped_no_valid_time += 1
            continue

        group = group.sort_values(time_col)

        # Average duplicate timestamps exactly as in the notebook.
        group = (
            group.groupby(time_col, as_index=False)[value_col]
            .mean()
            .sort_values(time_col)
        )

        dt_index = pd.DatetimeIndex(group[time_col])
        series = pd.Series(
            data=group[value_col].astype(float).values,
            index=dt_index,
            name=value_col,
        )
        series = series[~series.index.isna()].sort_index()

        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(
                f"Expected DatetimeIndex, got {type(series.index).__name__}"
            )

        if len(series) == 0:
            skipped_empty += 1
            continue

        if resample_rule is not None:
            try:
                series = series.resample(resample_rule).mean()
                series = series.interpolate(
                    method="time",
                    limit=2,
                    limit_direction="both",
                )
            except Exception as exc:
                LOGGER.warning(
                    "Resampling failed for %s: %s. Using raw ordered sequence instead.",
                    encounter_reference,
                    exc,
                )

        series = series.dropna()
        values = series.values.astype(np.float32)

        if len(values) == 0:
            skipped_empty += 1
            continue

        if len(values) < min_points:
            if len(values) == 1:
                values = np.repeat(values[0], min_points).astype(np.float32)
            else:
                pad_width = min_points - len(values)
                values = np.pad(
                    values,
                    pad_width=(pad_width, 0),
                    mode="edge",
                ).astype(np.float32)

        # Keep the most recent context.
        if len(values) > max_context_length:
            values = values[-max_context_length:]

        encounter_ids.append(encounter_reference)
        series_tensors.append(torch.tensor(values, dtype=torch.float32))

    LOGGER.info("Built time-series tensors.")
    LOGGER.info("Encounters with valid series: %d", len(series_tensors))
    LOGGER.info("Skipped empty: %d", skipped_empty)
    LOGGER.info("Skipped invalid time: %d", skipped_no_valid_time)

    return encounter_ids, series_tensors


def pool_chronos_embeddings(embeddings) -> torch.Tensor:
    """Mean-pool Chronos token embeddings to one vector per encounter.

    The handling of list/tensor return formats mirrors the source notebook to
    support Chronos versions that expose slightly different ``embed`` outputs.
    """
    if isinstance(embeddings, list):
        pooled_list = []

        for embedding in embeddings:
            if not torch.is_tensor(embedding):
                embedding = torch.as_tensor(embedding)

            embedding = embedding.float()

            if embedding.ndim == 3 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            if embedding.ndim == 2:
                pooled = embedding.mean(dim=0)
            elif embedding.ndim == 1:
                pooled = embedding
            else:
                raise ValueError(
                    f"Unexpected Chronos list element shape: {embedding.shape}"
                )

            pooled_list.append(pooled)

        return torch.stack(pooled_list, dim=0)

    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    embeddings = embeddings.float()

    if embeddings.ndim == 3:
        return embeddings.mean(dim=1)
    if embeddings.ndim == 2:
        return embeddings
    if embeddings.ndim == 1:
        return embeddings.unsqueeze(0)

    raise ValueError(f"Unexpected Chronos embedding shape: {embeddings.shape}")


def create_chronos2_time_series_embeddings(
    encounter_ids: list[str],
    series_tensors: list[torch.Tensor],
    output_embedding_path: Path,
    output_index_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_recompute: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create and save one frozen Chronos-2 embedding per encounter."""
    output_embedding_path = Path(output_embedding_path)
    output_index_path = Path(output_index_path)

    if (
        output_embedding_path.exists()
        and output_index_path.exists()
        and not force_recompute
    ):
        LOGGER.info("Existing Chronos-2 embeddings found; loading existing files.")
        embeddings = np.load(output_embedding_path)
        index_df = pd.read_csv(output_index_path)
        index_df[ID_COL] = index_df[ID_COL].apply(normalize_encounter_reference)
        return embeddings, index_df

    if len(series_tensors) == 0:
        raise ValueError("No time series available for Chronos-2 embedding.")

    from chronos import Chronos2Pipeline

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    LOGGER.info("Loading Chronos-2: %s", model_name)
    LOGGER.info("device_map: %s", device_map)
    LOGGER.info("dtype: %s", dtype)

    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
        )
    except TypeError:
        # Compatibility with older transformers/chronos versions.
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )

    all_pooled_embeddings: list[np.ndarray] = []

    for start in range(0, len(series_tensors), batch_size):
        end = min(start + batch_size, len(series_tensors))
        batch_series = series_tensors[start:end]

        with torch.no_grad():
            embed_output = pipeline.embed(batch_series)

        # Chronos commonly returns (embeddings, tokenizer_state), while some
        # versions return only embeddings.
        embeddings_raw = embed_output[0] if isinstance(embed_output, tuple) else embed_output
        pooled = pool_chronos_embeddings(embeddings_raw)

        if pooled.shape[0] != len(batch_series):
            raise ValueError(
                "Batch-size mismatch after pooling Chronos embeddings.\n"
                f"Expected {len(batch_series)} rows, got {pooled.shape[0]}.\n"
                f"Raw output type: {type(embeddings_raw)}"
            )

        all_pooled_embeddings.append(
            pooled.detach().cpu().numpy().astype(np.float32)
        )
        LOGGER.info("Embedded %d/%d series", end, len(series_tensors))

    embedding_matrix = np.concatenate(all_pooled_embeddings, axis=0).astype(
        np.float32
    )

    index_df = pd.DataFrame(
        {
            ID_COL: [normalize_encounter_reference(x) for x in encounter_ids],
            "embedding_row": np.arange(len(encounter_ids)),
        }
    )

    duplicated = index_df[ID_COL].duplicated().sum()
    if duplicated > 0:
        raise ValueError(
            f"Chronos embedding index has {duplicated} duplicated encounter IDs."
        )

    output_embedding_path.parent.mkdir(parents=True, exist_ok=True)
    output_index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_embedding_path, embedding_matrix)
    index_df.to_csv(output_index_path, index=False)

    LOGGER.info("Saved Chronos-2 embeddings: %s", output_embedding_path)
    LOGGER.info("Saved Chronos-2 index: %s", output_index_path)
    LOGGER.info("Embedding shape: %s", embedding_matrix.shape)

    return embedding_matrix, index_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate body-temperature Chronos-2 embeddings using the exact "
            "preprocessing and pooling procedure from the study notebook."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-embeddings", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--separator", default=DEFAULT_SEPARATOR)
    parser.add_argument("--id-column", default=DEFAULT_ID_COL_RAW)
    parser.add_argument("--time-column", default=DEFAULT_TIME_COL)
    parser.add_argument("--value-column", default=DEFAULT_VALUE_COL)
    parser.add_argument("--resample-rule", default=DEFAULT_RESAMPLE_RULE)
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=DEFAULT_MAX_CONTEXT_LENGTH,
    )
    parser.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if both output files already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    long_df = load_body_temperature_long_csv(
        path=args.input_csv,
        id_col_raw=args.id_column,
        time_col=args.time_column,
        value_col=args.value_column,
        separator=args.separator,
    )

    encounter_ids, series_tensors = build_time_series_tensors_from_long_df(
        long_df,
        resample_rule=args.resample_rule,
        max_context_length=args.max_context_length,
        min_points=args.min_points,
    )

    create_chronos2_time_series_embeddings(
        encounter_ids=encounter_ids,
        series_tensors=series_tensors,
        output_embedding_path=args.output_embeddings,
        output_index_path=args.output_index,
        model_name=args.model,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
