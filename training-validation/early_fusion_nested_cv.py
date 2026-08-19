#!/usr/bin/env python3
"""Nested cross-validation for the early-fusion multimodal fever model.

This script is a publication-oriented refactor of the analysis notebook used for
UME nested cross-validation. It implements leakage-safe early fusion of:

- structured/tabular encounter-level variables;
- precomputed clinical-note embeddings;
- precomputed body-temperature time-series embeddings;
- optional precomputed imaging embeddings; and
- optional structured concept variables.

For each outer cross-validation fold, all tabular/concept preprocessing is fitted
on the outer-training data only. Frozen modality embeddings are matched by
encounter ID, unavailable embeddings are zero-filled, and one availability
indicator per active modality is appended. L2-regularized logistic regression is
then fitted to the concatenated early-fusion representation.

The logistic-regression regularization strength (C) and all threshold-dependent
operating points are selected exclusively from inner-CV out-of-fold predictions.
The held-out outer fold is used exactly once for performance estimation.

Expected embedding format
-------------------------
Each embedding modality is supplied as:

1. ``*.npy``: a 2-D array with shape ``(n_encounters, embedding_dim)``;
2. ``*.csv``: an index containing the encounter ID column and ``embedding_row``.

Example
-------
python early_fusion_nested_cv.py \
    --tabular-csv data/ume_adapted.csv \
    --output-dir results/early_fusion_nested_cv \
    --text-embeddings data/ume_note_embeddings.npy \
    --text-index data/ume_note_embedding_index.csv \
    --time-series-embeddings data/ume_body_temp_chronos2_embeddings.npy \
    --time-series-index data/ume_body_temp_chronos2_index.csv

Notes
-----
- Raw patient data are intentionally not bundled with this script.
- The embedding-generation stages from the exploratory notebook are separated
  from model training. This keeps the public training/validation entry point
  deterministic, auditable, and independent of heavyweight foundation-model
  downloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger("early_fusion_nested_cv")


# Columns excluded from the structured feature branch in the source notebook.
# The missing comma between ``Unnamed: 0_y`` and ``encounter_id`` in the
# exploratory notebook is intentionally corrected here.
DEFAULT_EXCLUDE_COLUMNS = [
    "Unnamed: 0_x",
    "Unnamed: 0_y",
    "encounter_id",
    "subject_reference",
    "predictions_max",
    "predictions_min",
    "documented_infection",
    "documented_resistence",
    "clinical_impression",
    "probability_of_persisting_fever",
    "probability_icu",
    "ab_change",
    "documented_infection_conf",
    "documented_resistance_conf",
    "clinical_impression_conf",
    "persisting_fever_prob_conf",
    "probability_icu_conf",
    "ab_change_conf",
    "documented_infection_evidence",
    "documented_resistance_evidence",
    "ab_change_evidence",
    "clinical_trajectory_0_48h",
    "source_control_needed",
    "source_control_performed",
    "pathogen_identified",
    "persistent_positive_cultures",
    "mdro_suspected_or_confirmed",
    "empiric_abx_adequate",
    "abx_escalation_due_to_failure",
    "neutropenia",
    "profound_immunosuppression",
    "noninfectious_fever_suspected",
    "diagnostic_uncertainty_high",
    "source_control_needed_conf",
    "clinical_trajectory_0_48h_conf",
    "neutropenia_conf",
    "diagnostic_uncertainty_high_conf",
    "infection_focus_cns",
    "infection_focus_intraabdominal",
    "infection_focus_line",
    "infection_focus_other",
    "infection_focus_pneumonia",
    "infection_focus_ssti",
    "infection_focus_unknown",
    "infection_focus_uti",
    "sepsis_or_shock_none",
    "sepsis_or_shock_sepsis",
    "sepsis_or_shock_septic_shock",
    "sepsis_or_shock_unknown",
]


@dataclass(frozen=True)
class Config:
    """Runtime configuration for nested cross-validation."""

    tabular_csv: Path
    output_dir: Path
    id_col: str = "encounter_id"
    label_col: str = "fever"
    random_state: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    n_bootstraps: int = 2000
    bootstrap_seed: int = 42
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    c_selection_metric: str = "AUPRC"
    class_weight: str | None = "balanced"
    logistic_max_iter: int = 2000
    selected_threshold_strategy: str = "youden"
    min_sensitivity: float = 0.80
    min_specificity: float = 0.80
    threshold_grid_size: int = 999
    use_tabular: bool = True
    concept_cols: tuple[str, ...] = ()
    exclude_cols: tuple[str, ...] = tuple(DEFAULT_EXCLUDE_COLUMNS)
    text_embeddings: Path | None = None
    text_index: Path | None = None
    time_series_embeddings: Path | None = None
    time_series_index: Path | None = None
    ct_embeddings: Path | None = None
    ct_index: Path | None = None


@dataclass(frozen=True)
class EmbeddingSource:
    """In-memory lookup for one frozen embedding modality."""

    name: str
    lookup: Mapping[str, np.ndarray]
    dim: int


class ConstantProbabilityClassifier:
    """Fallback classifier for the unlikely case of a one-class fit subset."""

    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantProbabilityClassifier":
        del X, y
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.full(X.shape[0], self.probability, dtype=float)
        return np.column_stack([1.0 - p1, p1])


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Leakage-safe nested CV for the early-fusion multimodal fever model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tabular-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-col", default="encounter_id")
    parser.add_argument("--label-col", default="fever")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        help="Candidate inverse L2 regularization strengths for logistic regression.",
    )
    parser.add_argument(
        "--c-selection-metric",
        choices=["AUROC", "AUPRC", "Brier score"],
        default="AUPRC",
    )
    parser.add_argument("--logistic-max-iter", type=int, default=2000)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting used by logistic regression.",
    )
    parser.add_argument(
        "--selected-threshold-strategy",
        default="youden",
        help="Threshold strategy additionally exported as the selected result.",
    )
    parser.add_argument("--min-sensitivity", type=float, default=0.80)
    parser.add_argument("--min-specificity", type=float, default=0.80)
    parser.add_argument("--threshold-grid-size", type=int, default=999)
    parser.add_argument(
        "--no-tabular",
        action="store_true",
        help="Disable the structured tabular modality.",
    )
    parser.add_argument(
        "--concept-cols",
        nargs="*",
        default=[],
        help="Optional structured concept columns modeled as a separate modality.",
    )
    parser.add_argument(
        "--exclude-cols",
        nargs="*",
        default=None,
        help=(
            "Columns excluded from the tabular branch. If supplied, this replaces "
            "the publication defaults rather than extending them."
        ),
    )
    parser.add_argument("--text-embeddings", type=Path)
    parser.add_argument("--text-index", type=Path)
    parser.add_argument("--time-series-embeddings", type=Path)
    parser.add_argument("--time-series-index", type=Path)
    parser.add_argument("--ct-embeddings", type=Path)
    parser.add_argument("--ct-index", type=Path)

    args = parser.parse_args(argv)

    exclude_cols = (
        tuple(DEFAULT_EXCLUDE_COLUMNS)
        if args.exclude_cols is None
        else tuple(args.exclude_cols)
    )
    class_weight = None if args.class_weight == "none" else args.class_weight

    config = Config(
        tabular_csv=args.tabular_csv,
        output_dir=args.output_dir,
        id_col=args.id_col,
        label_col=args.label_col,
        random_state=args.random_state,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_bootstraps=args.n_bootstraps,
        bootstrap_seed=args.bootstrap_seed,
        c_grid=tuple(args.c_grid),
        c_selection_metric=args.c_selection_metric,
        class_weight=class_weight,
        logistic_max_iter=args.logistic_max_iter,
        selected_threshold_strategy=args.selected_threshold_strategy,
        min_sensitivity=args.min_sensitivity,
        min_specificity=args.min_specificity,
        threshold_grid_size=args.threshold_grid_size,
        use_tabular=not args.no_tabular,
        concept_cols=tuple(args.concept_cols),
        exclude_cols=exclude_cols,
        text_embeddings=args.text_embeddings,
        text_index=args.text_index,
        time_series_embeddings=args.time_series_embeddings,
        time_series_index=args.time_series_index,
        ct_embeddings=args.ct_embeddings,
        ct_index=args.ct_index,
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Validate CLI configuration before loading any patient-level data."""
    if config.outer_folds < 2 or config.inner_folds < 2:
        raise ValueError("Both outer_folds and inner_folds must be >= 2.")
    if config.n_bootstraps < 0:
        raise ValueError("n_bootstraps must be >= 0.")
    if config.threshold_grid_size < 2:
        raise ValueError("threshold_grid_size must be >= 2.")
    if not 0.0 <= config.min_sensitivity <= 1.0:
        raise ValueError("min_sensitivity must be between 0 and 1.")
    if not 0.0 <= config.min_specificity <= 1.0:
        raise ValueError("min_specificity must be between 0 and 1.")
    if not config.c_grid or any(c <= 0 for c in config.c_grid):
        raise ValueError("All values in c_grid must be > 0.")

    modality_pairs = [
        ("text", config.text_embeddings, config.text_index),
        ("time_series", config.time_series_embeddings, config.time_series_index),
        ("ct_image", config.ct_embeddings, config.ct_index),
    ]
    for name, embedding_path, index_path in modality_pairs:
        if (embedding_path is None) != (index_path is None):
            raise ValueError(
                f"{name}: provide both embedding and index paths, or neither."
            )

    if not (
        config.use_tabular
        or config.concept_cols
        or config.text_embeddings is not None
        or config.time_series_embeddings is not None
        or config.ct_embeddings is not None
    ):
        raise ValueError("At least one modality must be enabled.")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_encounter_reference(value: object) -> str | float:
    """Normalize encounter identifiers to ``Encounter/<id>``."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text.lower().endswith(".txt"):
        text = text[:-4]
    text = text.replace("\\", "/").strip()

    lower = text.lower()
    if lower.startswith("encounter_"):
        return f"Encounter/{text.split('_', 1)[1]}"
    if lower.startswith("encounter/"):
        return f"Encounter/{text.split('/', 1)[1]}"
    return f"Encounter/{text}"


def detect_separator(path: Path, encoding: str = "utf-8-sig") -> str:
    with path.open("r", encoding=encoding, errors="replace") as handle:
        sample = handle.read(10000)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[";", ",", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        candidates = [";", ",", "\t", "|"]
        return max(candidates, key=first_line.count)


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read clinical CSV exports with common encodings/separators."""
    if not path.exists():
        raise FileNotFoundError(path)

    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "utf-16"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            separator = detect_separator(path, encoding)
            df = pd.read_csv(
                path,
                sep=separator,
                encoding=encoding,
                engine="python",
                on_bad_lines="warn",
            )
            df.columns = (
                df.columns.astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
            )
            if df.shape[1] > 1:
                LOGGER.info(
                    "Loaded %s: shape=%s encoding=%s separator=%r",
                    path,
                    df.shape,
                    encoding,
                    separator,
                )
                return df
        except Exception as exc:  # try the next plausible export encoding
            last_error = exc
    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def load_tabular_data(config: Config) -> pd.DataFrame:
    df = read_csv_robust(config.tabular_csv)

    if config.id_col not in df.columns:
        casefold_map = {column.strip().lower(): column for column in df.columns}
        original = casefold_map.get(config.id_col.lower())
        if original is None:
            raise ValueError(
                f"Expected ID column {config.id_col!r}; available columns: {list(df.columns)}"
            )
        df = df.rename(columns={original: config.id_col})

    if config.label_col not in df.columns:
        raise ValueError(f"Label column {config.label_col!r} not found.")

    if df[config.id_col].isna().any():
        raise ValueError(f"{config.id_col!r} contains missing values.")

    df[config.id_col] = df[config.id_col].map(normalize_encounter_reference)
    duplicate_count = int(df[config.id_col].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicated {config.id_col!r} values; "
            "the encounter-level table must contain one row per encounter."
        )

    if df[config.label_col].isna().any():
        raise ValueError(f"{config.label_col!r} contains missing values.")

    labels = set(pd.unique(df[config.label_col]))
    if not labels.issubset({0, 1, False, True}):
        raise ValueError(
            f"{config.label_col!r} must be binary 0/1; observed values: {sorted(map(str, labels))}"
        )
    df[config.label_col] = df[config.label_col].astype(int)

    LOGGER.info("Outcome distribution: %s", df[config.label_col].value_counts().to_dict())
    return df.reset_index(drop=True)


def load_embedding_source(
    name: str,
    embedding_path: Path,
    index_path: Path,
    id_col: str,
) -> EmbeddingSource:
    """Load and validate a ``.npy`` embedding matrix plus its row-index CSV."""
    if not embedding_path.exists():
        raise FileNotFoundError(embedding_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    embeddings = np.load(embedding_path, allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(
            f"{name} embeddings must be 2-D; got shape {embeddings.shape}."
        )
    if not np.issubdtype(embeddings.dtype, np.number):
        raise TypeError(f"{name} embeddings must be numeric.")

    index_df = pd.read_csv(index_path)
    required = {id_col, "embedding_row"}
    missing = required.difference(index_df.columns)
    if missing:
        raise ValueError(f"{name} index is missing required columns: {sorted(missing)}")

    index_df = index_df[[id_col, "embedding_row"]].copy()
    index_df[id_col] = index_df[id_col].map(normalize_encounter_reference)
    if index_df[id_col].isna().any():
        raise ValueError(f"{name} index contains missing encounter IDs.")
    if index_df[id_col].duplicated().any():
        duplicates = index_df.loc[index_df[id_col].duplicated(), id_col].tolist()[:10]
        raise ValueError(f"{name} index contains duplicate encounter IDs, e.g. {duplicates}")

    rows = pd.to_numeric(index_df["embedding_row"], errors="raise").astype(int)
    if ((rows < 0) | (rows >= embeddings.shape[0])).any():
        raise IndexError(f"{name} index contains embedding_row values outside the .npy matrix.")

    lookup = {
        encounter_id: np.asarray(embeddings[row], dtype=np.float32)
        for encounter_id, row in zip(index_df[id_col], rows)
    }
    LOGGER.info(
        "Loaded %s embeddings: n=%d dim=%d",
        name,
        len(lookup),
        embeddings.shape[1],
    )
    return EmbeddingSource(name=name, lookup=lookup, dim=int(embeddings.shape[1]))


def load_embedding_sources(config: Config) -> dict[str, EmbeddingSource]:
    sources: dict[str, EmbeddingSource] = {}
    specs = [
        ("text", config.text_embeddings, config.text_index),
        ("time_series", config.time_series_embeddings, config.time_series_index),
        ("ct_image", config.ct_embeddings, config.ct_index),
    ]
    for name, embedding_path, index_path in specs:
        if embedding_path is not None and index_path is not None:
            sources[name] = load_embedding_source(
                name=name,
                embedding_path=embedding_path,
                index_path=index_path,
                id_col=config.id_col,
            )
    return sources


def make_one_hot_encoder() -> OneHotEncoder:
    """Construct OneHotEncoder across recent and older scikit-learn releases."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_feature_columns(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    concept_cols: Iterable[str],
    exclude_cols: Iterable[str],
) -> list[str]:
    unavailable = {id_col, label_col, *concept_cols, *exclude_cols}
    return [column for column in df.columns if column not in unavailable]


def ensure_columns_exist(
    df: pd.DataFrame,
    required_cols: Sequence[str],
) -> pd.DataFrame:
    """Add absent evaluation columns as NaN for train-fitted preprocessing."""
    df = df.copy()
    for column in required_cols:
        if column not in df.columns:
            df[column] = np.nan
    return df


def build_preprocessor(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = [
        column for column in feature_cols if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_cols = [column for column in feature_cols if column not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns available for preprocessing.")

    return (
        ColumnTransformer(transformers=transformers, remainder="drop"),
        numeric_cols,
        categorical_cols,
    )


def to_dense_float32(array: object) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float32)


def lookup_matrix_for_ids(
    ids: Iterable[object],
    source: EmbeddingSource,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_ids = [normalize_encounter_reference(value) for value in ids]
    matrix = np.zeros((len(normalized_ids), source.dim), dtype=np.float32)
    mask = np.zeros(len(normalized_ids), dtype=bool)

    for row_idx, encounter_id in enumerate(normalized_ids):
        embedding = source.lookup.get(encounter_id)
        if embedding is not None:
            matrix[row_idx] = embedding
            mask[row_idx] = True
    return matrix, mask


def concatenate_modalities(
    features: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    modalities: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    n_rows: int | None = None

    for modality in modalities:
        block = np.asarray(features[modality], dtype=np.float32)
        if block.ndim != 2:
            raise ValueError(f"{modality} feature block must be 2-D; got {block.shape}.")
        if n_rows is None:
            n_rows = block.shape[0]
        elif block.shape[0] != n_rows:
            raise ValueError("Modality blocks contain different numbers of rows.")
        blocks.append(block)
        feature_names.extend(f"{modality}_{idx}" for idx in range(block.shape[1]))

    if n_rows is None:
        raise RuntimeError("No active modality blocks were constructed.")

    for modality in modalities:
        availability = np.asarray(masks[modality], dtype=np.float32).reshape(-1, 1)
        blocks.append(availability)
        feature_names.append(f"has_{modality}")

    return np.hstack(blocks).astype(np.float32), feature_names


def make_early_fusion_arrays_for_fold(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: Config,
    embedding_sources: Mapping[str, EmbeddingSource],
) -> dict[str, object]:
    """Construct leakage-safe early-fusion matrices for one fit/evaluation split."""
    fit_df = fit_df.copy().reset_index(drop=True)
    eval_df = eval_df.copy().reset_index(drop=True)
    fit_df[config.id_col] = fit_df[config.id_col].map(normalize_encounter_reference)
    eval_df[config.id_col] = eval_df[config.id_col].map(normalize_encounter_reference)

    fit_features: dict[str, np.ndarray] = {}
    eval_features: dict[str, np.ndarray] = {}
    fit_masks: dict[str, np.ndarray] = {}
    eval_masks: dict[str, np.ndarray] = {}
    tabular_feature_cols: list[str] = []

    if config.use_tabular:
        tabular_feature_cols = get_feature_columns(
            fit_df,
            id_col=config.id_col,
            label_col=config.label_col,
            concept_cols=config.concept_cols,
            exclude_cols=config.exclude_cols,
        )
        if not tabular_feature_cols:
            raise ValueError("Tabular modality enabled but no tabular features remain.")
        eval_df = ensure_columns_exist(eval_df, tabular_feature_cols)
        preprocessor, _, _ = build_preprocessor(fit_df, tabular_feature_cols)
        fit_features["tabular"] = to_dense_float32(
            preprocessor.fit_transform(fit_df[tabular_feature_cols])
        )
        eval_features["tabular"] = to_dense_float32(
            preprocessor.transform(eval_df[tabular_feature_cols])
        )
        fit_masks["tabular"] = np.ones(len(fit_df), dtype=bool)
        eval_masks["tabular"] = np.ones(len(eval_df), dtype=bool)

    if config.concept_cols:
        missing_fit = [column for column in config.concept_cols if column not in fit_df.columns]
        if missing_fit:
            raise ValueError(f"Missing concept columns in fit data: {missing_fit}")
        eval_df = ensure_columns_exist(eval_df, config.concept_cols)
        concept_preprocessor, _, _ = build_preprocessor(fit_df, config.concept_cols)
        fit_features["concepts"] = to_dense_float32(
            concept_preprocessor.fit_transform(fit_df[list(config.concept_cols)])
        )
        eval_features["concepts"] = to_dense_float32(
            concept_preprocessor.transform(eval_df[list(config.concept_cols)])
        )
        fit_masks["concepts"] = np.ones(len(fit_df), dtype=bool)
        eval_masks["concepts"] = np.ones(len(eval_df), dtype=bool)

    for modality in ("text", "time_series", "ct_image"):
        source = embedding_sources.get(modality)
        if source is None:
            continue
        fit_matrix, fit_mask = lookup_matrix_for_ids(fit_df[config.id_col], source)
        eval_matrix, eval_mask = lookup_matrix_for_ids(eval_df[config.id_col], source)
        fit_features[modality] = fit_matrix
        eval_features[modality] = eval_matrix
        fit_masks[modality] = fit_mask
        eval_masks[modality] = eval_mask

    modality_order = [
        name
        for name in ("tabular", "text", "concepts", "time_series", "ct_image")
        if name in fit_features
    ]
    if not modality_order:
        raise RuntimeError("No usable early-fusion modality features found.")

    x_fit, feature_names = concatenate_modalities(
        fit_features, fit_masks, modality_order
    )
    x_eval, eval_feature_names = concatenate_modalities(
        eval_features, eval_masks, modality_order
    )
    if feature_names != eval_feature_names:
        raise RuntimeError("Feature definitions differ between fit and evaluation data.")

    return {
        "ids_fit": fit_df[config.id_col].astype(str).to_numpy(),
        "y_fit": fit_df[config.label_col].astype(int).to_numpy(),
        "X_fit": x_fit,
        "ids_eval": eval_df[config.id_col].astype(str).to_numpy(),
        "y_eval": eval_df[config.label_col].astype(int).to_numpy(),
        "X_eval": x_eval,
        "modalities": modality_order,
        "masks_fit": fit_masks,
        "masks_eval": eval_masks,
        "feature_names": feature_names,
        "tabular_feature_cols": tabular_feature_cols,
    }


def make_early_fusion_estimator(config: Config, logistic_c: float) -> Pipeline:
    classifier = LogisticRegression(
        C=float(logistic_c),
        penalty="l2",
        solver="lbfgs",
        max_iter=config.logistic_max_iter,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )
    # The source notebook scales the complete concatenated representation after
    # modality-specific tabular preprocessing; that behavior is preserved here.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def safe_prevalence(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.5
    return float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))


def fit_early_fusion_model(
    x: np.ndarray,
    y: np.ndarray,
    config: Config,
    logistic_c: float,
    stage: str,
) -> tuple[object, pd.DataFrame]:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        model: object = ConstantProbabilityClassifier(safe_prevalence(y))
        status = "constant_fallback"
    else:
        model = make_early_fusion_estimator(config, logistic_c)
        model.fit(x, y)
        status = "fitted"

    info = pd.DataFrame(
        [
            {
                "stage": stage,
                "classifier": "logistic",
                "logistic_c": float(logistic_c),
                "status": status,
                "n_samples": int(len(y)),
                "n_features": int(x.shape[1]),
                "n_positive": int(np.sum(y == 1)),
                "n_negative": int(np.sum(y == 0)),
            }
        ]
    )
    return model, info


def predict_probability(model: object, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)[:, 1]
    return np.clip(probabilities, 1e-6, 1.0 - 1e-6)


def safe_divide(numerator: float, denominator: float) -> float:
    return np.nan if denominator == 0 else numerator / denominator


def logit_np(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope from the prediction logit."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    linear_predictor = logit_np(y_prob).reshape(-1, 1)
    try:
        import statsmodels.api as sm  # optional; numerically convenient here

        design = sm.add_constant(linear_predictor)
        fit = sm.GLM(y_true, design, family=sm.families.Binomial()).fit(disp=0)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        # Compatible fallback for environments where statsmodels is unavailable.
        for penalty in (None, "none"):
            try:
                calibration_model = LogisticRegression(
                    penalty=penalty,
                    solver="lbfgs",
                    max_iter=1000,
                )
                calibration_model.fit(linear_predictor, y_true)
                return (
                    float(calibration_model.intercept_[0]),
                    float(calibration_model.coef_[0, 0]),
                )
            except (TypeError, ValueError):
                continue
        return np.nan, np.nan


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan

    calibration_intercept, calibration_slope = calibration_intercept_slope(
        y_true, y_prob
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Calibration intercept": calibration_intercept,
        "Calibration slope": calibration_slope,
        "Brier score": brier_score_loss(y_true, y_prob),
        "Sensitivity": safe_divide(tp, tp + fn),
        "Specificity": safe_divide(tn, tn + fp),
        "PPV": safe_divide(tp, tp + fp),
        "NPV": safe_divide(tn, tn + fn),
        "F1 score": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def calculate_binary_metrics_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan

    calibration_intercept, calibration_slope = calibration_intercept_slope(
        y_true, y_prob
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Calibration intercept": calibration_intercept,
        "Calibration slope": calibration_slope,
        "Brier score": brier_score_loss(y_true, y_prob),
        "Sensitivity": safe_divide(tp, tp + fn),
        "Specificity": safe_divide(tn, tn + fp),
        "PPV": safe_divide(tp, tp + fp),
        "NPV": safe_divide(tn, tn + fn),
        "F1 score": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


METRIC_ORDER = [
    "AUROC",
    "AUPRC",
    "Calibration intercept",
    "Calibration slope",
    "Brier score",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "F1 score",
    "Accuracy",
]


def summarize_bootstrap(
    point: Mapping[str, float],
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for metric in METRIC_ORDER:
        values = bootstrap_df[metric].dropna().to_numpy(dtype=float)
        if values.size:
            lower, upper = np.percentile(values, [2.5, 97.5])
        else:
            lower, upper = np.nan, np.nan
        estimate = float(point[metric]) if not pd.isna(point[metric]) else np.nan
        estimate_ci = (
            "NA"
            if np.isnan(estimate)
            else f"{estimate:.3f} ({lower:.3f}–{upper:.3f})"
        )
        rows.append(
            {
                "Metric": metric,
                "Estimate": estimate,
                "CI lower": lower,
                "CI upper": upper,
                "Estimate with 95% CI": estimate_ci,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    point = calculate_binary_metrics(y_true, y_prob, threshold)
    if n_bootstraps == 0:
        empty = pd.DataFrame(columns=METRIC_ORDER)
        return point, summarize_bootstrap(point, empty)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    for _ in range(n_bootstraps):
        sample_idx = rng.integers(0, n, size=n)
        rows.append(
            calculate_binary_metrics(
                np.asarray(y_true)[sample_idx],
                np.asarray(y_prob)[sample_idx],
                threshold,
            )
        )
    bootstrap_df = pd.DataFrame(rows)
    return point, summarize_bootstrap(point, bootstrap_df)


def bootstrap_binary_metrics_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    point = calculate_binary_metrics_from_predictions(y_true, y_prob, y_pred)
    if n_bootstraps == 0:
        empty = pd.DataFrame(columns=METRIC_ORDER)
        return point, summarize_bootstrap(point, empty)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    for _ in range(n_bootstraps):
        sample_idx = rng.integers(0, n, size=n)
        rows.append(
            calculate_binary_metrics_from_predictions(
                np.asarray(y_true)[sample_idx],
                np.asarray(y_prob)[sample_idx],
                np.asarray(y_pred)[sample_idx],
            )
        )
    bootstrap_df = pd.DataFrame(rows)
    return point, summarize_bootstrap(point, bootstrap_df)


def threshold_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def threshold_grid_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid_size: int,
) -> pd.DataFrame:
    thresholds = np.linspace(0.001, 0.999, int(grid_size))
    return pd.DataFrame(
        [threshold_metrics_at_threshold(y_true, y_prob, threshold) for threshold in thresholds]
    )


def find_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "sensitivity": tpr,
            "specificity": 1.0 - fpr,
        }
    )
    curve["youden_j"] = curve["sensitivity"] + curve["specificity"] - 1.0
    curve = curve[
        np.isfinite(curve["threshold"])
        & curve["threshold"].between(0.0, 1.0)
    ].copy()
    if curve.empty:
        return np.nan, curve
    best_idx = curve["youden_j"].idxmax()
    return float(curve.loc[best_idx, "threshold"]), curve


def build_candidate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    youden, youden_df = find_youden_threshold(y_true, y_prob)
    grid = threshold_grid_metrics(y_true, y_prob, config.threshold_grid_size)

    max_f1 = float(grid.loc[grid["f1"].idxmax(), "threshold"])

    sensitivity_eligible = grid[grid["sensitivity"] >= config.min_sensitivity]
    sensitivity_threshold = (
        float(
            sensitivity_eligible.loc[
                sensitivity_eligible["specificity"].idxmax(), "threshold"
            ]
        )
        if not sensitivity_eligible.empty
        else np.nan
    )

    specificity_eligible = grid[grid["specificity"] >= config.min_specificity]
    specificity_threshold = (
        float(
            specificity_eligible.loc[
                specificity_eligible["sensitivity"].idxmax(), "threshold"
            ]
        )
        if not specificity_eligible.empty
        else np.nan
    )

    candidates = pd.DataFrame(
        [
            {
                "strategy": "default_0.50",
                "threshold": 0.50,
                "derivation_dataset": "fixed_default",
                "criterion": "fixed threshold of 0.50",
            },
            {
                "strategy": "youden",
                "threshold": youden,
                "derivation_dataset": "inner_cv_predictions",
                "criterion": "max sensitivity + specificity - 1",
            },
            {
                "strategy": "max_f1",
                "threshold": max_f1,
                "derivation_dataset": "inner_cv_predictions",
                "criterion": "max F1 on inner-CV predictions",
            },
            {
                "strategy": f"sensitivity_at_least_{config.min_sensitivity:.2f}",
                "threshold": sensitivity_threshold,
                "derivation_dataset": "inner_cv_predictions",
                "criterion": (
                    "highest specificity with sensitivity >= "
                    f"{config.min_sensitivity:.2f}"
                ),
            },
            {
                "strategy": f"specificity_at_least_{config.min_specificity:.2f}",
                "threshold": specificity_threshold,
                "derivation_dataset": "inner_cv_predictions",
                "criterion": (
                    "highest sensitivity with specificity >= "
                    f"{config.min_specificity:.2f}"
                ),
            },
        ]
    )
    candidates = candidates[np.isfinite(candidates["threshold"])].copy()
    return candidates, grid, youden_df


def score_inner_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "AUROC": roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else np.nan,
        "AUPRC": average_precision_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else np.nan,
        "Brier score": brier_score_loss(y_true, y_prob),
    }


def make_inner_cv_predictions(
    outer_train_df: pd.DataFrame,
    outer_fold: int,
    logistic_c: float,
    config: Config,
    embedding_sources: Mapping[str, EmbeddingSource],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    y_outer = outer_train_df[config.label_col].astype(int).to_numpy()
    counts = np.bincount(y_outer, minlength=2)
    n_splits = min(config.inner_folds, int(counts.min()))
    if n_splits < 2:
        raise RuntimeError(
            f"Not enough examples in both classes for inner CV in outer fold {outer_fold}."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_state + 1000 + outer_fold,
    )
    probabilities = np.full(len(outer_train_df), np.nan, dtype=float)
    ids = (
        outer_train_df[config.id_col]
        .map(normalize_encounter_reference)
        .astype(str)
        .to_numpy()
    )
    info_frames = []

    for inner_fold, (fit_idx, val_idx) in enumerate(
        cv.split(outer_train_df, y_outer), start=1
    ):
        arrays = make_early_fusion_arrays_for_fold(
            outer_train_df.iloc[fit_idx],
            outer_train_df.iloc[val_idx],
            config,
            embedding_sources,
        )
        model, info = fit_early_fusion_model(
            arrays["X_fit"],
            arrays["y_fit"],
            config,
            logistic_c=logistic_c,
            stage="threshold_inner_cv_fit",
        )
        probabilities[val_idx] = predict_probability(model, arrays["X_eval"])
        info.insert(0, "outer_fold", outer_fold)
        info.insert(1, "inner_fold", inner_fold)
        info["n_modalities"] = len(arrays["modalities"])
        info["modalities"] = ",".join(arrays["modalities"])
        info_frames.append(info)

    if np.isnan(probabilities).any():
        raise RuntimeError(f"Inner-CV predictions contain NaN in outer fold {outer_fold}.")

    return ids, y_outer, probabilities, pd.concat(info_frames, ignore_index=True)


def select_logistic_c(
    outer_train_df: pd.DataFrame,
    outer_fold: int,
    config: Config,
    embedding_sources: Mapping[str, EmbeddingSource],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate_rows = []
    payloads: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}

    for c_value in config.c_grid:
        LOGGER.info("Outer fold %d: evaluating logistic C=%g", outer_fold, c_value)
        payload = make_inner_cv_predictions(
            outer_train_df,
            outer_fold,
            logistic_c=float(c_value),
            config=config,
            embedding_sources=embedding_sources,
        )
        ids, y_true, y_prob, info = payload
        candidate_rows.append(
            {
                "outer_fold": outer_fold,
                "classifier": "logistic",
                "logistic_c": float(c_value),
                **score_inner_predictions(y_true, y_prob),
            }
        )
        payloads[float(c_value)] = (ids, y_true, y_prob, info)

    hpo_df = pd.DataFrame(candidate_rows)
    metric = config.c_selection_metric
    if metric == "Brier score":
        best_idx = hpo_df[metric].astype(float).idxmin()
    else:
        best_idx = hpo_df[metric].astype(float).idxmax()
    selected_c = float(hpo_df.loc[best_idx, "logistic_c"])
    hpo_df["selection_metric"] = metric
    hpo_df["selected"] = hpo_df["logistic_c"].eq(selected_c)

    ids, y_true, y_prob, info = payloads[selected_c]
    info = info.copy()
    info["selected_logistic_c"] = selected_c
    LOGGER.info(
        "Outer fold %d: selected C=%g by inner-CV %s",
        outer_fold,
        selected_c,
        metric,
    )
    return selected_c, ids, y_true, y_prob, info, hpo_df


def confusion_matrix_long(
    cohort: str,
    strategy: str,
    threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    outer_fold: int | None = None,
) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    data = pd.DataFrame(
        [
            {"actual": 0, "predicted": 0, "cell": "TN", "n": int(tn)},
            {"actual": 0, "predicted": 1, "cell": "FP", "n": int(fp)},
            {"actual": 1, "predicted": 0, "cell": "FN", "n": int(fn)},
            {"actual": 1, "predicted": 1, "cell": "TP", "n": int(tp)},
        ]
    )
    data.insert(0, "cohort", cohort)
    insert_at = 1
    if outer_fold is not None:
        data.insert(insert_at, "outer_fold", outer_fold)
        insert_at += 1
    data.insert(insert_at, "threshold_strategy", strategy)
    data.insert(insert_at + 1, "threshold", threshold)
    return data


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved %s (%d rows)", path, len(df))


def save_run_metadata(
    config: Config,
    embedding_sources: Mapping[str, EmbeddingSource],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, tuple):
            config_dict[key] = list(value)

    metadata = {
        "config": config_dict,
        "active_embedding_modalities": {
            name: {"n_embeddings": len(source.lookup), "dim": source.dim}
            for name, source in embedding_sources.items()
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def run_nested_cv(
    df: pd.DataFrame,
    config: Config,
    embedding_sources: Mapping[str, EmbeddingSource],
) -> None:
    y_all = df[config.label_col].astype(int).to_numpy()
    class_counts = np.bincount(y_all, minlength=2)
    if class_counts.min() < config.outer_folds:
        raise ValueError(
            "outer_folds exceeds the number of observations in the minority class: "
            f"counts={class_counts.tolist()}, outer_folds={config.outer_folds}."
        )

    outer_cv = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    outer_predictions = []
    outer_point_metrics = []
    outer_bootstrap_summaries = []
    outer_confusions = []
    candidate_thresholds_all = []
    threshold_grids_all = []
    youden_curves_all = []
    inner_predictions_all = []
    training_info_all = []
    c_hpo_all = []
    availability_rows = []
    feature_info_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(df, y_all), start=1
    ):
        LOGGER.info(
            "Outer fold %d/%d: train=%d test=%d",
            outer_fold,
            config.outer_folds,
            len(train_idx),
            len(test_idx),
        )
        outer_train = df.iloc[train_idx].copy().reset_index(drop=True)
        outer_test = df.iloc[test_idx].copy().reset_index(drop=True)

        selected_c, inner_ids, inner_y, inner_prob, inner_info, hpo_df = select_logistic_c(
            outer_train, outer_fold, config, embedding_sources
        )
        c_hpo_all.append(hpo_df)
        training_info_all.append(inner_info)
        inner_predictions_all.append(
            pd.DataFrame(
                {
                    "outer_fold": outer_fold,
                    config.id_col: inner_ids,
                    "y_true": inner_y,
                    "y_prob": inner_prob,
                    "prediction_source": "inner_cv_threshold_derivation",
                    "model": "nested_cv_early_fusion",
                    "classifier": "logistic",
                    "selected_logistic_c": selected_c,
                }
            )
        )

        candidates, threshold_grid, youden_curve = build_candidate_thresholds(
            inner_y, inner_prob, config
        )
        candidates.insert(0, "outer_fold", outer_fold)
        candidates.insert(1, "selected_logistic_c", selected_c)
        threshold_grid.insert(0, "outer_fold", outer_fold)
        threshold_grid.insert(1, "selected_logistic_c", selected_c)
        youden_curve.insert(0, "outer_fold", outer_fold)
        youden_curve.insert(1, "selected_logistic_c", selected_c)
        candidate_thresholds_all.append(candidates)
        threshold_grids_all.append(threshold_grid)
        youden_curves_all.append(youden_curve)

        arrays = make_early_fusion_arrays_for_fold(
            outer_train, outer_test, config, embedding_sources
        )
        modalities: list[str] = arrays["modalities"]
        feature_info_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "nested_cv_early_fusion",
                "classifier": "logistic",
                "selected_logistic_c": selected_c,
                "n_features": int(arrays["X_fit"].shape[1]),
                "n_modalities": len(modalities),
                "modalities": ",".join(modalities),
                "n_tabular_source_columns": len(arrays["tabular_feature_cols"]),
            }
        )

        for cohort, masks in (
            ("outer_train", arrays["masks_fit"]),
            ("outer_test", arrays["masks_eval"]),
        ):
            for modality in modalities:
                mask = np.asarray(masks[modality], dtype=bool)
                availability_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "cohort": cohort,
                        "modality": modality,
                        "n": len(mask),
                        "n_available": int(mask.sum()),
                        "availability": float(mask.mean()),
                    }
                )

        model, fit_info = fit_early_fusion_model(
            arrays["X_fit"],
            arrays["y_fit"],
            config,
            logistic_c=selected_c,
            stage="outer_train_final_fit",
        )
        fit_info.insert(0, "outer_fold", outer_fold)
        fit_info["n_modalities"] = len(modalities)
        fit_info["modalities"] = ",".join(modalities)
        training_info_all.append(fit_info)

        y_prob = predict_probability(model, arrays["X_eval"])
        y_true = np.asarray(arrays["y_eval"], dtype=int)
        fold_predictions = pd.DataFrame(
            {
                "outer_fold": outer_fold,
                config.id_col: arrays["ids_eval"],
                "y_true": y_true,
                "y_prob": y_prob,
                "model": "nested_cv_early_fusion",
                "classifier": "logistic",
                "selected_logistic_c": selected_c,
            }
        )

        for strategy_row in candidates.itertuples(index=False):
            strategy = strategy_row.strategy
            threshold = float(strategy_row.threshold)
            y_pred = (y_prob >= threshold).astype(int)
            fold_predictions[f"threshold_{strategy}"] = threshold
            fold_predictions[f"y_pred_{strategy}"] = y_pred

            point, summary = bootstrap_binary_metrics(
                y_true,
                y_prob,
                threshold,
                n_bootstraps=config.n_bootstraps,
                seed=(
                    config.bootstrap_seed
                    + outer_fold * 1000
                    + len(outer_point_metrics)
                ),
            )
            outer_point_metrics.append(
                {
                    "outer_fold": outer_fold,
                    "threshold_strategy": strategy,
                    "threshold": threshold,
                    "model": "nested_cv_early_fusion",
                    "classifier": "logistic",
                    "selected_logistic_c": selected_c,
                    **point,
                }
            )
            summary.insert(0, "outer_fold", outer_fold)
            summary.insert(1, "threshold_strategy", strategy)
            summary.insert(2, "threshold", threshold)
            summary.insert(3, "model", "nested_cv_early_fusion")
            summary.insert(4, "classifier", "logistic")
            summary.insert(5, "selected_logistic_c", selected_c)
            outer_bootstrap_summaries.append(summary)
            outer_confusions.append(
                confusion_matrix_long(
                    cohort="outer_fold_test",
                    outer_fold=outer_fold,
                    strategy=strategy,
                    threshold=threshold,
                    y_true=y_true,
                    y_pred=y_pred,
                )
            )

        outer_predictions.append(fold_predictions)

    # Fold-level outputs
    predictions_df = pd.concat(outer_predictions, ignore_index=True)
    candidates_df = pd.concat(candidate_thresholds_all, ignore_index=True)
    grids_df = pd.concat(threshold_grids_all, ignore_index=True)
    youden_df = pd.concat(youden_curves_all, ignore_index=True)
    inner_predictions_df = pd.concat(inner_predictions_all, ignore_index=True)
    point_metrics_df = pd.DataFrame(outer_point_metrics)
    bootstrap_metrics_df = pd.concat(outer_bootstrap_summaries, ignore_index=True)
    confusion_df = pd.concat(outer_confusions, ignore_index=True)
    availability_df = pd.DataFrame(availability_rows)
    training_info_df = pd.concat(training_info_all, ignore_index=True)
    c_hpo_df = pd.concat(c_hpo_all, ignore_index=True)
    feature_info_df = pd.DataFrame(feature_info_rows)

    outputs = {
        "nested_cv_early_fusion_modality_availability.csv": availability_df,
        "nested_cv_early_fusion_feature_info.csv": feature_info_df,
        "nested_cv_early_fusion_training_info.csv": training_info_df,
        "nested_cv_early_fusion_logistic_c_grid_search.csv": c_hpo_df,
        "nested_cv_inner_predictions_for_threshold_derivation.csv": inner_predictions_df,
        "nested_cv_candidate_thresholds_all_outer_folds.csv": candidates_df,
        "nested_cv_inner_threshold_grid_metrics_all_outer_folds.csv": grids_df,
        "nested_cv_inner_youden_thresholds_all_outer_folds.csv": youden_df,
        "nested_cv_outer_fold_predictions_early_fusion.csv": predictions_df,
        "nested_cv_outer_fold_point_metrics_by_threshold.csv": point_metrics_df,
        "nested_cv_outer_fold_bootstrap_metrics_by_threshold_long.csv": bootstrap_metrics_df,
        "nested_cv_outer_fold_confusion_matrices_by_threshold_long.csv": confusion_df,
    }
    for filename, frame in outputs.items():
        save_csv(frame, config.output_dir / filename)

    # Pooled outer-fold predictions. Threshold-dependent predictions remain those
    # produced by the fold-specific threshold derived from each patient's training folds.
    pooled_point_rows = []
    pooled_summary_frames = []
    pooled_confusion_frames = []

    for strategy in candidates_df["strategy"].drop_duplicates():
        pred_col = f"y_pred_{strategy}"
        threshold_col = f"threshold_{strategy}"
        if pred_col not in predictions_df:
            continue

        y_true = predictions_df["y_true"].astype(int).to_numpy()
        y_prob = predictions_df["y_prob"].astype(float).to_numpy()
        y_pred = predictions_df[pred_col].astype(int).to_numpy()
        thresholds = predictions_df[threshold_col].astype(float).to_numpy()

        point, summary = bootstrap_binary_metrics_from_predictions(
            y_true,
            y_prob,
            y_pred,
            n_bootstraps=config.n_bootstraps,
            seed=config.bootstrap_seed + 90000 + len(pooled_point_rows),
        )
        pooled_point_rows.append(
            {
                "threshold_strategy": strategy,
                "threshold_mean": float(np.mean(thresholds)),
                "threshold_sd": float(np.std(thresholds)),
                "threshold_min": float(np.min(thresholds)),
                "threshold_max": float(np.max(thresholds)),
                "model": "nested_cv_early_fusion",
                "classifier": "logistic",
                **point,
            }
        )
        summary.insert(0, "threshold_strategy", strategy)
        summary.insert(1, "threshold_mean", float(np.mean(thresholds)))
        summary.insert(2, "threshold_sd", float(np.std(thresholds)))
        summary.insert(3, "model", "nested_cv_early_fusion")
        summary.insert(4, "classifier", "logistic")
        pooled_summary_frames.append(summary)
        pooled_confusion_frames.append(
            confusion_matrix_long(
                cohort="pooled_outer_cv",
                strategy=strategy,
                threshold=float(np.mean(thresholds)),
                y_true=y_true,
                y_pred=y_pred,
            )
        )

    pooled_point_df = pd.DataFrame(pooled_point_rows)
    pooled_bootstrap_df = pd.concat(pooled_summary_frames, ignore_index=True)
    pooled_confusion_df = pd.concat(pooled_confusion_frames, ignore_index=True)

    save_csv(
        pooled_point_df,
        config.output_dir / "nested_cv_pooled_outer_predictions_point_metrics_by_threshold.csv",
    )
    save_csv(
        pooled_bootstrap_df,
        config.output_dir / "nested_cv_pooled_outer_predictions_bootstrap_metrics_by_threshold_long.csv",
    )
    save_csv(
        pooled_confusion_df,
        config.output_dir / "nested_cv_pooled_outer_predictions_confusion_matrices_by_threshold_long.csv",
    )

    selected_strategy = config.selected_threshold_strategy
    if selected_strategy not in set(pooled_point_df["threshold_strategy"]):
        if "youden" in set(pooled_point_df["threshold_strategy"]):
            LOGGER.warning(
                "Threshold strategy %r unavailable; exporting Youden as selected instead.",
                selected_strategy,
            )
            selected_strategy = "youden"
        else:
            raise ValueError(
                f"Selected threshold strategy {selected_strategy!r} is unavailable."
            )

    save_csv(
        pooled_point_df[pooled_point_df["threshold_strategy"] == selected_strategy],
        config.output_dir / f"nested_cv_selected_{selected_strategy}_pooled_point_metrics.csv",
    )
    save_csv(
        pooled_bootstrap_df[
            pooled_bootstrap_df["threshold_strategy"] == selected_strategy
        ],
        config.output_dir
        / f"nested_cv_selected_{selected_strategy}_pooled_bootstrap_metrics_long.csv",
    )
    save_csv(
        pooled_confusion_df[
            pooled_confusion_df["threshold_strategy"] == selected_strategy
        ],
        config.output_dir
        / f"nested_cv_selected_{selected_strategy}_pooled_confusion_matrix_long.csv",
    )

    LOGGER.info("Nested cross-validation complete: %s", config.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    config = parse_args(argv)
    set_global_seed(config.random_state)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_tabular_data(config)
    embedding_sources = load_embedding_sources(config)
    save_run_metadata(config, embedding_sources)

    active_modalities = []
    if config.use_tabular:
        active_modalities.append("tabular")
    if config.concept_cols:
        active_modalities.append("concepts")
    active_modalities.extend(embedding_sources.keys())
    LOGGER.info("Active modalities: %s", ", ".join(active_modalities))

    run_nested_cv(data, config, embedding_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
