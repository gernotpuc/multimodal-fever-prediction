#!/usr/bin/env python3
"""External validation for the late-fusion multimodal fever model.

This script is the external-validation counterpart of ``late_fusion_nested_cv.py``.
It implements the same publication model architecture while preserving a strict
separation between development and external data.

Study architecture
------------------
The model combines five possible modality branches:

- structured/tabular encounter-level variables;
- precomputed Qwen clinical-note embeddings;
- optional structured concept variables;
- precomputed Chronos-2 body-temperature embeddings; and
- precomputed MedImageInsight CT embeddings.

Each enabled modality is first modeled independently by a regularized logistic-
regression base learner. Leakage-free out-of-fold base probabilities from the
UME tuning-training subset are used to train a second-stage TabPFN V2.5 stacker,
along with modality-availability indicators. Final unimodal base learners are
then refitted on the complete tuning-training subset.

External-validation protocol
----------------------------
1. Split the UME development cohort into 80% tuning-training and 20% internal
   validation using a stratified split (default random state 42).
2. Fit all train-dependent preprocessing on tuning-training only.
3. Fit the late-fusion package on tuning-training only. The stacker receives
   leakage-free OOF base probabilities generated within tuning-training.
4. Predict the untouched internal-validation split.
5. Fit logistic recalibration on the internal-validation predictions only.
6. Derive operating thresholds from the recalibrated internal-validation
   predictions only.
7. Apply the frozen preprocessing, unimodal models, TabPFN stacker,
   recalibrator, and thresholds unchanged to the external cohort.

The external cohort is never used for preprocessing fitting, base-model fitting,
stacker fitting, recalibration, or threshold selection.

Expected embedding format
-------------------------
Each frozen embedding modality is supplied separately for development and
external cohorts as:

1. ``*.npy``: a 2-D array of encounter embeddings;
2. ``*.csv``: an index containing ``encounter_id`` and ``embedding_row``.

Example
-------
python late_fusion_external_validation.py \\
    --train-tabular-csv data/ume_adapted.csv \\
    --external-tabular-csv data/josef_adapted.csv \\
    --external-cohort-name josef \\
    --output-dir results/late_fusion_external_josef \\
    --train-text-embeddings data/ume_note_embeddings.npy \\
    --train-text-index data/ume_note_embedding_index.csv \\
    --external-text-embeddings data/josef_note_embeddings.npy \\
    --external-text-index data/josef_note_embedding_index.csv \\
    --train-time-series-embeddings data/ume_body_temp_chronos2_embeddings.npy \\
    --train-time-series-index data/ume_body_temp_chronos2_index.csv \\
    --external-time-series-embeddings data/josef_body_temp_chronos2_embeddings.npy \\
    --external-time-series-index data/josef_body_temp_chronos2_index.csv \\
    --train-ct-embeddings data/ume_ct_medimageinsight_embeddings.npy \\
    --train-ct-index data/ume_ct_medimageinsight_index.csv \\
    --external-ct-embeddings data/josef_ct_medimageinsight_embeddings.npy \\
    --external-ct-index data/josef_ct_medimageinsight_index.csv

Notes
-----
- The publication configuration uses fixed unimodal logistic base learners with
  ``C=1.0`` and a TabPFN V2.5 second-stage stacker.
- The three frozen embedding branches are required in the primary reproduction
  path, consistent with the late-fusion nested-CV script.
- A logistic second-stage stacker is exposed only as a diagnostic alternative.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

# Reuse the exact late-fusion base learners, stacking implementation, encounter
# normalization, embedding loading, preprocessing, thresholds, and metric
# utilities from the nested-CV implementation.
from late_fusion_nested_cv import (
    DEFAULT_EXCLUDE_COLUMNS,
    Config as NestedCVConfig,
    EmbeddingSource,
    bootstrap_binary_metrics_from_predictions,
    build_preprocessor,
    calculate_binary_metrics_from_predictions,
    calibration_intercept_slope,
    confusion_matrix_long,
    effective_splits,
    ensure_columns_exist,
    find_youden_threshold,
    fit_late_fusion_package,
    get_feature_columns,
    load_embedding_source,
    lookup_matrix_for_ids,
    normalize_encounter_reference,
    package_version,
    predict_late_fusion_package,
    read_csv_robust,
    threshold_grid_metrics,
    to_dense_float32,
)

LOGGER = logging.getLogger("late_fusion_external_validation")
MODEL_NAME = "late_fusion_external_validation_tabpfn_v25"


@dataclass(frozen=True)
class ExternalValidationConfig:
    """Runtime configuration for late-fusion external validation."""

    train_tabular_csv: Path
    external_tabular_csv: Path
    output_dir: Path
    external_cohort_name: str = "external"
    id_col: str = "encounter_id"
    label_col: str = "fever"
    random_state: int = 42
    internal_validation_size: float = 0.20
    n_bootstraps: int = 2000
    bootstrap_seed: int = 42

    # Unimodal base learners: fixed study configuration.
    base_oof_folds: int = 5
    base_c: float = 1.0
    class_weight: str | None = "balanced"
    base_max_iter: int = 2000

    # Second-stage stacker.
    stacker: str = "tabpfn"
    stacker_c: float = 1.0
    stacker_max_iter: int = 2000
    tabpfn_model_version: str = "V2_5"
    tabpfn_device: str = "auto"
    tabpfn_ignore_pretraining_limits: bool = True

    selected_threshold_strategy: str = "youden"
    min_sensitivity: float = 0.80
    min_specificity: float = 0.80
    threshold_grid_size: int = 999

    use_tabular: bool = True
    concept_cols: tuple[str, ...] = ()
    exclude_cols: tuple[str, ...] = tuple(DEFAULT_EXCLUDE_COLUMNS)

    train_text_embeddings: Path | None = None
    train_text_index: Path | None = None
    external_text_embeddings: Path | None = None
    external_text_index: Path | None = None

    train_time_series_embeddings: Path | None = None
    train_time_series_index: Path | None = None
    external_time_series_embeddings: Path | None = None
    external_time_series_index: Path | None = None

    train_ct_embeddings: Path | None = None
    train_ct_index: Path | None = None
    external_ct_embeddings: Path | None = None
    external_ct_index: Path | None = None


@dataclass
class LateFusionFeatureSpace:
    """Train-fitted modality definition reused unchanged for evaluation cohorts."""

    tabular_feature_cols: list[str]
    tabular_preprocessor: Any | None
    concept_cols: list[str]
    concept_preprocessor: Any | None
    modalities: list[str]


@dataclass
class LogisticRecalibrator:
    """Logistic recalibration fitted on internal-validation predictions only."""

    enabled: bool
    reason: str
    intercept: float
    slope: float
    model: LogisticRegression | None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    value = value.strip("_.-")
    return value or "external"


def parse_args(argv: Sequence[str] | None = None) -> ExternalValidationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "External validation of the late-fusion multimodal fever model "
            "with fixed unimodal logistic learners and a TabPFN V2.5 stacker."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-tabular-csv", type=Path, required=True)
    parser.add_argument("--external-tabular-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-cohort-name", default="external")
    parser.add_argument("--id-col", default="encounter_id")
    parser.add_argument("--label-col", default="fever")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--internal-validation-size", type=float, default=0.20)
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument("--base-oof-folds", type=int, default=5)
    parser.add_argument("--base-c", type=float, default=1.0)
    parser.add_argument("--base-max-iter", type=int, default=2000)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")

    parser.add_argument("--stacker", choices=["tabpfn", "logistic"], default="tabpfn")
    parser.add_argument("--stacker-c", type=float, default=1.0)
    parser.add_argument("--stacker-max-iter", type=int, default=2000)
    parser.add_argument("--tabpfn-model-version", default="V2_5")
    parser.add_argument(
        "--tabpfn-device",
        default="auto",
        help="TabPFN device. 'auto' selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--respect-tabpfn-pretraining-limits",
        action="store_true",
        help="Do not set ignore_pretraining_limits=True when supported by TabPFN.",
    )

    parser.add_argument("--selected-threshold-strategy", default="youden")
    parser.add_argument("--min-sensitivity", type=float, default=0.80)
    parser.add_argument("--min-specificity", type=float, default=0.80)
    parser.add_argument("--threshold-grid-size", type=int, default=999)
    parser.add_argument(
        "--no-tabular",
        action="store_true",
        help="Disable the structured tabular branch for an ablation run.",
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
        help="Replace the publication-default tabular exclusion list.",
    )

    # Primary study embedding branches: required for development and external data.
    parser.add_argument("--train-text-embeddings", type=Path, required=True)
    parser.add_argument("--train-text-index", type=Path, required=True)
    parser.add_argument("--external-text-embeddings", type=Path, required=True)
    parser.add_argument("--external-text-index", type=Path, required=True)

    parser.add_argument("--train-time-series-embeddings", type=Path, required=True)
    parser.add_argument("--train-time-series-index", type=Path, required=True)
    parser.add_argument("--external-time-series-embeddings", type=Path, required=True)
    parser.add_argument("--external-time-series-index", type=Path, required=True)

    parser.add_argument("--train-ct-embeddings", type=Path, required=True)
    parser.add_argument("--train-ct-index", type=Path, required=True)
    parser.add_argument("--external-ct-embeddings", type=Path, required=True)
    parser.add_argument("--external-ct-index", type=Path, required=True)

    args = parser.parse_args(argv)
    exclude_cols = (
        tuple(DEFAULT_EXCLUDE_COLUMNS)
        if args.exclude_cols is None
        else tuple(args.exclude_cols)
    )
    class_weight = None if args.class_weight == "none" else args.class_weight

    config = ExternalValidationConfig(
        train_tabular_csv=args.train_tabular_csv,
        external_tabular_csv=args.external_tabular_csv,
        output_dir=args.output_dir,
        external_cohort_name=slugify(args.external_cohort_name),
        id_col=args.id_col,
        label_col=args.label_col,
        random_state=args.random_state,
        internal_validation_size=args.internal_validation_size,
        n_bootstraps=args.n_bootstraps,
        bootstrap_seed=args.bootstrap_seed,
        base_oof_folds=args.base_oof_folds,
        base_c=args.base_c,
        class_weight=class_weight,
        base_max_iter=args.base_max_iter,
        stacker=args.stacker,
        stacker_c=args.stacker_c,
        stacker_max_iter=args.stacker_max_iter,
        tabpfn_model_version=args.tabpfn_model_version,
        tabpfn_device=args.tabpfn_device,
        tabpfn_ignore_pretraining_limits=not args.respect_tabpfn_pretraining_limits,
        selected_threshold_strategy=args.selected_threshold_strategy,
        min_sensitivity=args.min_sensitivity,
        min_specificity=args.min_specificity,
        threshold_grid_size=args.threshold_grid_size,
        use_tabular=not args.no_tabular,
        concept_cols=tuple(args.concept_cols),
        exclude_cols=exclude_cols,
        train_text_embeddings=args.train_text_embeddings,
        train_text_index=args.train_text_index,
        external_text_embeddings=args.external_text_embeddings,
        external_text_index=args.external_text_index,
        train_time_series_embeddings=args.train_time_series_embeddings,
        train_time_series_index=args.train_time_series_index,
        external_time_series_embeddings=args.external_time_series_embeddings,
        external_time_series_index=args.external_time_series_index,
        train_ct_embeddings=args.train_ct_embeddings,
        train_ct_index=args.train_ct_index,
        external_ct_embeddings=args.external_ct_embeddings,
        external_ct_index=args.external_ct_index,
    )
    validate_config(config)
    return config


def validate_config(config: ExternalValidationConfig) -> None:
    if not 0.0 < config.internal_validation_size < 1.0:
        raise ValueError("internal_validation_size must be strictly between 0 and 1.")
    if config.n_bootstraps < 0:
        raise ValueError("n_bootstraps must be >= 0.")
    if config.base_oof_folds < 2:
        raise ValueError("base_oof_folds must be >= 2.")
    if config.base_c <= 0:
        raise ValueError("base_c must be > 0.")
    if config.stacker_c <= 0:
        raise ValueError("stacker_c must be > 0.")
    if config.threshold_grid_size < 2:
        raise ValueError("threshold_grid_size must be >= 2.")
    if not 0.0 <= config.min_sensitivity <= 1.0:
        raise ValueError("min_sensitivity must be between 0 and 1.")
    if not 0.0 <= config.min_specificity <= 1.0:
        raise ValueError("min_specificity must be between 0 and 1.")


def load_encounter_table(path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    """Load and validate a one-row-per-encounter binary-outcome table."""
    df = read_csv_robust(path)

    if id_col not in df.columns:
        casefold_map = {column.strip().lower(): column for column in df.columns}
        original = casefold_map.get(id_col.lower())
        if original is None:
            raise ValueError(
                f"Expected ID column {id_col!r} in {path}; "
                f"available columns: {list(df.columns)}"
            )
        df = df.rename(columns={original: id_col})

    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in {path}.")
    if df[id_col].isna().any():
        raise ValueError(f"{path}: {id_col!r} contains missing values.")

    df[id_col] = df[id_col].map(normalize_encounter_reference)
    duplicate_count = int(df[id_col].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"{path}: found {duplicate_count} duplicated {id_col!r} values; "
            "the encounter-level table must contain one row per encounter."
        )

    if df[label_col].isna().any():
        raise ValueError(f"{path}: {label_col!r} contains missing values.")
    labels = set(pd.unique(df[label_col]))
    if not labels.issubset({0, 1, False, True}):
        raise ValueError(
            f"{path}: {label_col!r} must be binary 0/1; "
            f"observed values: {sorted(map(str, labels))}"
        )
    df[label_col] = df[label_col].astype(int)

    LOGGER.info(
        "Loaded %s: shape=%s outcome=%s",
        path,
        df.shape,
        df[label_col].value_counts().to_dict(),
    )
    return df.reset_index(drop=True)


def load_embedding_pair(
    config: ExternalValidationConfig,
) -> tuple[dict[str, EmbeddingSource], dict[str, EmbeddingSource]]:
    """Load development/external embeddings and verify matching dimensions."""
    train_specs = {
        "text": (config.train_text_embeddings, config.train_text_index),
        "time_series": (
            config.train_time_series_embeddings,
            config.train_time_series_index,
        ),
        "ct_image": (config.train_ct_embeddings, config.train_ct_index),
    }
    external_specs = {
        "text": (config.external_text_embeddings, config.external_text_index),
        "time_series": (
            config.external_time_series_embeddings,
            config.external_time_series_index,
        ),
        "ct_image": (config.external_ct_embeddings, config.external_ct_index),
    }

    train_sources: dict[str, EmbeddingSource] = {}
    external_sources: dict[str, EmbeddingSource] = {}
    for modality in ("text", "time_series", "ct_image"):
        train_embeddings, train_index = train_specs[modality]
        external_embeddings, external_index = external_specs[modality]
        assert train_embeddings is not None and train_index is not None
        assert external_embeddings is not None and external_index is not None

        train_sources[modality] = load_embedding_source(
            modality, train_embeddings, train_index, config.id_col
        )
        external_sources[modality] = load_embedding_source(
            modality, external_embeddings, external_index, config.id_col
        )
        if train_sources[modality].dim != external_sources[modality].dim:
            raise ValueError(
                f"{modality} embedding dimension differs between development "
                f"({train_sources[modality].dim}) and external "
                f"({external_sources[modality].dim}) cohorts."
            )

    return train_sources, external_sources


def fit_feature_space(
    fit_df: pd.DataFrame,
    config: ExternalValidationConfig,
    train_sources: Mapping[str, EmbeddingSource],
) -> tuple[
    LateFusionFeatureSpace,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Fit train-dependent preprocessing and construct modality matrices."""
    fit_df = fit_df.copy().reset_index(drop=True)
    fit_df[config.id_col] = fit_df[config.id_col].map(normalize_encounter_reference)

    features: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    tabular_feature_cols: list[str] = []
    tabular_preprocessor: Any | None = None
    concept_preprocessor: Any | None = None

    if config.use_tabular:
        tabular_feature_cols = get_feature_columns(
            fit_df,
            config.id_col,
            config.label_col,
            config.concept_cols,
            config.exclude_cols,
        )
        if not tabular_feature_cols:
            raise ValueError("Tabular modality enabled but no tabular features remain.")
        tabular_preprocessor, _, _ = build_preprocessor(
            fit_df, tabular_feature_cols
        )
        features["tabular"] = to_dense_float32(
            tabular_preprocessor.fit_transform(fit_df[tabular_feature_cols])
        )
        masks["tabular"] = np.ones(len(fit_df), dtype=bool)

    concept_cols = list(config.concept_cols)
    if concept_cols:
        missing = [column for column in concept_cols if column not in fit_df.columns]
        if missing:
            raise ValueError(f"Missing concept columns in tuning-training data: {missing}")
        concept_preprocessor, _, _ = build_preprocessor(fit_df, concept_cols)
        features["concepts"] = to_dense_float32(
            concept_preprocessor.fit_transform(fit_df[concept_cols])
        )
        masks["concepts"] = np.ones(len(fit_df), dtype=bool)

    for modality in ("text", "time_series", "ct_image"):
        matrix, mask = lookup_matrix_for_ids(fit_df[config.id_col], train_sources[modality])
        features[modality] = matrix
        masks[modality] = mask

    modalities = [
        name
        for name in ("tabular", "text", "concepts", "time_series", "ct_image")
        if name in features
    ]
    if not modalities:
        raise RuntimeError("No usable modalities found for late fusion.")

    feature_space = LateFusionFeatureSpace(
        tabular_feature_cols=tabular_feature_cols,
        tabular_preprocessor=tabular_preprocessor,
        concept_cols=concept_cols,
        concept_preprocessor=concept_preprocessor,
        modalities=modalities,
    )
    return feature_space, features, masks


def transform_feature_space(
    df: pd.DataFrame,
    feature_space: LateFusionFeatureSpace,
    config: ExternalValidationConfig,
    embedding_sources: Mapping[str, EmbeddingSource],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Transform a cohort using only tuning-train-fitted preprocessing objects."""
    df = df.copy().reset_index(drop=True)
    df[config.id_col] = df[config.id_col].map(normalize_encounter_reference)

    features: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    if "tabular" in feature_space.modalities:
        assert feature_space.tabular_preprocessor is not None
        df = ensure_columns_exist(df, feature_space.tabular_feature_cols)
        features["tabular"] = to_dense_float32(
            feature_space.tabular_preprocessor.transform(
                df[feature_space.tabular_feature_cols]
            )
        )
        masks["tabular"] = np.ones(len(df), dtype=bool)

    if "concepts" in feature_space.modalities:
        assert feature_space.concept_preprocessor is not None
        df = ensure_columns_exist(df, feature_space.concept_cols)
        features["concepts"] = to_dense_float32(
            feature_space.concept_preprocessor.transform(df[feature_space.concept_cols])
        )
        masks["concepts"] = np.ones(len(df), dtype=bool)

    for modality in ("text", "time_series", "ct_image"):
        if modality not in feature_space.modalities:
            continue
        matrix, mask = lookup_matrix_for_ids(df[config.id_col], embedding_sources[modality])
        features[modality] = matrix
        masks[modality] = mask

    missing_modalities = [m for m in feature_space.modalities if m not in features]
    if missing_modalities:
        raise RuntimeError(f"Missing transformed modalities: {missing_modalities}")
    return features, masks


def valid_stratify_series(y: pd.Series) -> pd.Series | None:
    counts = y.value_counts(dropna=False)
    if y.nunique(dropna=False) <= 20 and counts.min() >= 2:
        return y
    return None


def split_development_cohort(
    df: pd.DataFrame,
    config: ExternalValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_y = valid_stratify_series(df[config.label_col])
    tuning_train, internal_validation = train_test_split(
        df,
        test_size=config.internal_validation_size,
        random_state=config.random_state,
        stratify=stratify_y,
    )
    return tuning_train.reset_index(drop=True), internal_validation.reset_index(drop=True)


def late_cv_compat_config(config: ExternalValidationConfig) -> NestedCVConfig:
    """Create the shared late-fusion model configuration used by the CV module."""
    return NestedCVConfig(
        tabular_csv=config.train_tabular_csv,
        output_dir=config.output_dir,
        id_col=config.id_col,
        label_col=config.label_col,
        random_state=config.random_state,
        outer_folds=5,
        inner_folds=3,
        base_oof_folds=config.base_oof_folds,
        n_bootstraps=config.n_bootstraps,
        bootstrap_seed=config.bootstrap_seed,
        base_c=config.base_c,
        class_weight=config.class_weight,
        base_max_iter=config.base_max_iter,
        stacker=config.stacker,
        stacker_c=config.stacker_c,
        stacker_max_iter=config.stacker_max_iter,
        tabpfn_model_version=config.tabpfn_model_version,
        tabpfn_device=config.tabpfn_device,
        tabpfn_ignore_pretraining_limits=config.tabpfn_ignore_pretraining_limits,
        selected_threshold_strategy=config.selected_threshold_strategy,
        min_sensitivity=config.min_sensitivity,
        min_specificity=config.min_specificity,
        threshold_grid_size=config.threshold_grid_size,
        use_tabular=config.use_tabular,
        concept_cols=config.concept_cols,
        exclude_cols=config.exclude_cols,
        text_embeddings=config.train_text_embeddings,
        text_index=config.train_text_index,
        time_series_embeddings=config.train_time_series_embeddings,
        time_series_index=config.train_time_series_index,
        ct_embeddings=config.train_ct_embeddings,
        ct_index=config.train_ct_index,
    )


def logit_np(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def fit_logistic_recalibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> LogisticRecalibrator:
    """Fit logistic recalibration on internal-validation predictions only."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return LogisticRecalibrator(
            enabled=False,
            reason="one_class_only",
            intercept=0.0,
            slope=1.0,
            model=None,
        )

    x = logit_np(y_prob).reshape(-1, 1)
    model = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
    )
    model.fit(x, y_true)
    return LogisticRecalibrator(
        enabled=True,
        reason="fitted",
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0, 0]),
        model=model,
    )


def apply_logistic_recalibration(
    recalibrator: LogisticRecalibrator,
    y_prob: np.ndarray,
) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    if not recalibrator.enabled or recalibrator.model is None:
        return np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    x = logit_np(y_prob).reshape(-1, 1)
    return np.clip(recalibrator.model.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)


def recalibration_summary(
    recalibrator: LogisticRecalibrator,
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
) -> pd.DataFrame:
    before_intercept, before_slope = calibration_intercept_slope(y_true, y_prob_before)
    after_intercept, after_slope = calibration_intercept_slope(y_true, y_prob_after)
    return pd.DataFrame(
        [
            {
                "dataset_used_for_recalibration": "internal_validation",
                "recalibration_enabled": recalibrator.enabled,
                "recalibration_reason": recalibrator.reason,
                "recalibration_intercept": recalibrator.intercept,
                "recalibration_slope": recalibrator.slope,
                "calibration_intercept_before": before_intercept,
                "calibration_slope_before": before_slope,
                "brier_before": brier_score_loss(y_true, y_prob_before),
                "calibration_intercept_after": after_intercept,
                "calibration_slope_after": after_slope,
                "brier_after": brier_score_loss(y_true, y_prob_after),
            }
        ]
    )


def build_candidate_thresholds_from_internal_validation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: ExternalValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive the same five operating thresholds used by the CV pipeline."""
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
                "derivation_dataset": "internal_validation",
                "criterion": "max sensitivity + specificity - 1",
            },
            {
                "strategy": "max_f1",
                "threshold": max_f1,
                "derivation_dataset": "internal_validation",
                "criterion": "max F1 on internal validation",
            },
            {
                "strategy": f"sensitivity_at_least_{config.min_sensitivity:.2f}",
                "threshold": sensitivity_threshold,
                "derivation_dataset": "internal_validation",
                "criterion": (
                    "highest specificity with sensitivity >= "
                    f"{config.min_sensitivity:.2f}"
                ),
            },
            {
                "strategy": f"specificity_at_least_{config.min_specificity:.2f}",
                "threshold": specificity_threshold,
                "derivation_dataset": "internal_validation",
                "criterion": (
                    "highest sensitivity with specificity >= "
                    f"{config.min_specificity:.2f}"
                ),
            },
        ]
    )
    candidates = candidates[np.isfinite(candidates["threshold"])].copy()
    return candidates, grid, youden_df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved %s (%d rows)", path, len(df))


def availability_table(
    masks_by_cohort: Mapping[str, Mapping[str, np.ndarray]],
    modalities: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for cohort, masks in masks_by_cohort.items():
        for modality in modalities:
            mask = np.asarray(masks[modality], dtype=bool)
            rows.append(
                {
                    "cohort": cohort,
                    "modality": modality,
                    "n": int(mask.size),
                    "n_available": int(mask.sum()),
                    "n_missing": int((~mask).sum()),
                    "availability": float(mask.mean()) if mask.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def serialize_config(config: ExternalValidationConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data


def save_run_metadata(
    config: ExternalValidationConfig,
    train_sources: Mapping[str, EmbeddingSource],
    external_sources: Mapping[str, EmbeddingSource],
) -> None:
    metadata = {
        "model": {
            "name": MODEL_NAME,
            "architecture": (
                "unimodal logistic base learners + probability/availability "
                "stacking + logistic recalibration"
            ),
            "stacker": config.stacker,
            "tabpfn_model_version": (
                config.tabpfn_model_version if config.stacker == "tabpfn" else None
            ),
            "development_protocol": (
                "80/20 development split; OOF base probabilities within tuning-training; "
                "recalibration and thresholds on internal validation only"
            ),
        },
        "config": serialize_config(config),
        "development_embedding_modalities": {
            name: {"n_embeddings": len(source.lookup), "dim": source.dim}
            for name, source in train_sources.items()
        },
        "external_embedding_modalities": {
            name: {"n_embeddings": len(source.lookup), "dim": source.dim}
            for name, source in external_sources.items()
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "tabpfn": package_version("tabpfn"),
            "tabpfn_extensions": package_version("tabpfn-extensions"),
        },
    }
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def add_base_probabilities_and_availability(
    table: pd.DataFrame,
    base_probabilities: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    modalities: Sequence[str],
) -> pd.DataFrame:
    out = table.copy()
    for column in base_probabilities.columns:
        out[column] = base_probabilities[column].to_numpy()
    for modality in modalities:
        out[f"has_{modality}"] = np.asarray(masks[modality], dtype=int)
    return out


def run_external_validation(config: ExternalValidationConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_dir = config.output_dir / "threshold_analysis"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    development_df = load_encounter_table(
        config.train_tabular_csv, config.id_col, config.label_col
    )
    external_df = load_encounter_table(
        config.external_tabular_csv, config.id_col, config.label_col
    )
    train_sources, external_sources = load_embedding_pair(config)
    save_run_metadata(config, train_sources, external_sources)

    tuning_train_df, internal_validation_df = split_development_cohort(
        development_df, config
    )
    LOGGER.info(
        "Development split: full=%d tuning_train=%d internal_validation=%d",
        len(development_df),
        len(tuning_train_df),
        len(internal_validation_df),
    )

    feature_space, train_features, train_masks = fit_feature_space(
        tuning_train_df, config, train_sources
    )
    validation_features, validation_masks = transform_feature_space(
        internal_validation_df, feature_space, config, train_sources
    )
    external_features, external_masks = transform_feature_space(
        external_df, feature_space, config, external_sources
    )

    y_train = tuning_train_df[config.label_col].astype(int).to_numpy()
    y_validation = internal_validation_df[config.label_col].astype(int).to_numpy()
    y_external = external_df[config.label_col].astype(int).to_numpy()

    ids_train = tuning_train_df[config.id_col].astype(str).to_numpy()
    ids_validation = internal_validation_df[config.id_col].astype(str).to_numpy()
    ids_external = external_df[config.id_col].astype(str).to_numpy()

    modalities = feature_space.modalities
    model_config = late_cv_compat_config(config)

    feature_info_rows = []
    for modality in modalities:
        feature_info_rows.append(
            {
                "model": MODEL_NAME,
                "stacker": config.stacker,
                "external_cohort": config.external_cohort_name,
                "modality": modality,
                "feature_dimension": int(train_features[modality].shape[1]),
                "n_development_full": len(development_df),
                "n_tuning_train": len(tuning_train_df),
                "n_internal_validation": len(internal_validation_df),
                "n_external": len(external_df),
            }
        )
    save_csv(
        pd.DataFrame(feature_info_rows),
        config.output_dir / "external_validation_feature_info.csv",
    )
    save_csv(
        pd.DataFrame(
            {
                "stacker_feature_name": [f"p_{m}" for m in modalities]
                + [f"has_{m}" for m in modalities]
            }
        ),
        config.output_dir / "external_validation_stacker_feature_names.csv",
    )
    save_csv(
        availability_table(
            {
                "tuning_train": train_masks,
                "internal_validation": validation_masks,
                config.external_cohort_name: external_masks,
            },
            modalities,
        ),
        config.output_dir / "external_validation_modality_availability.csv",
    )

    base_oof_folds = effective_splits(
        y_train, config.base_oof_folds, "External-validation tuning-train base OOF"
    )
    package = fit_late_fusion_package(
        train_features,
        train_masks,
        y_train,
        modalities,
        model_config,
        n_splits=base_oof_folds,
    )

    training_frames = []
    for key, stage in (
        ("oof_info_df", "tuning_train_oof_base"),
        ("final_info_df", "tuning_train_final_base"),
    ):
        frame = package[key].copy()
        frame.insert(0, "stage", stage)
        frame.insert(1, "model", MODEL_NAME)
        frame.insert(2, "stacker", config.stacker)
        training_frames.append(frame)
    save_csv(
        pd.concat(training_frames, ignore_index=True),
        config.output_dir / "external_validation_training_info.csv",
    )

    y_train_prob_uncalibrated, train_base_prob = predict_late_fusion_package(
        package, train_features, train_masks
    )
    y_validation_prob_uncalibrated, validation_base_prob = predict_late_fusion_package(
        package, validation_features, validation_masks
    )
    y_external_prob_uncalibrated, external_base_prob = predict_late_fusion_package(
        package, external_features, external_masks
    )

    recalibrator = fit_logistic_recalibrator(
        y_validation, y_validation_prob_uncalibrated
    )
    y_train_prob = apply_logistic_recalibration(
        recalibrator, y_train_prob_uncalibrated
    )
    y_validation_prob = apply_logistic_recalibration(
        recalibrator, y_validation_prob_uncalibrated
    )
    y_external_prob = apply_logistic_recalibration(
        recalibrator, y_external_prob_uncalibrated
    )

    recalibration_df = recalibration_summary(
        recalibrator,
        y_validation,
        y_validation_prob_uncalibrated,
        y_validation_prob,
    )
    recalibration_df["model"] = MODEL_NAME
    recalibration_df["stacker"] = config.stacker
    recalibration_df["tabpfn_model_version"] = (
        config.tabpfn_model_version if config.stacker == "tabpfn" else np.nan
    )
    save_csv(
        recalibration_df,
        config.output_dir
        / "external_validation_logistic_recalibration_internal_validation.csv",
    )

    prediction_tables = {
        "tuning_train": add_base_probabilities_and_availability(
            pd.DataFrame(
                {
                    config.id_col: ids_train,
                    "y_true": y_train,
                    "y_prob_uncalibrated": y_train_prob_uncalibrated,
                    "y_prob": y_train_prob,
                    "prediction_source": "final_models_resubstitution",
                }
            ),
            train_base_prob,
            train_masks,
            modalities,
        ),
        "internal_validation": add_base_probabilities_and_availability(
            pd.DataFrame(
                {
                    config.id_col: ids_validation,
                    "y_true": y_validation,
                    "y_prob_uncalibrated": y_validation_prob_uncalibrated,
                    "y_prob": y_validation_prob,
                    "prediction_source": "held_out_internal_validation",
                }
            ),
            validation_base_prob,
            validation_masks,
            modalities,
        ),
        config.external_cohort_name: add_base_probabilities_and_availability(
            pd.DataFrame(
                {
                    config.id_col: ids_external,
                    "y_true": y_external,
                    "y_prob_uncalibrated": y_external_prob_uncalibrated,
                    "y_prob": y_external_prob,
                    "prediction_source": "external_validation",
                }
            ),
            external_base_prob,
            external_masks,
            modalities,
        ),
    }
    for cohort, table in prediction_tables.items():
        table["model"] = MODEL_NAME
        table["stacker"] = config.stacker
        table["base_c"] = config.base_c
        table["tabpfn_model_version"] = (
            config.tabpfn_model_version if config.stacker == "tabpfn" else np.nan
        )
        save_csv(
            table,
            config.output_dir / f"{slugify(cohort)}_predictions_late_fusion.csv",
        )

    candidates, threshold_grid, youden_curve = (
        build_candidate_thresholds_from_internal_validation(
            y_validation, y_validation_prob, config
        )
    )
    candidates["model"] = MODEL_NAME
    candidates["stacker"] = config.stacker
    save_csv(
        candidates,
        threshold_dir / "candidate_thresholds_from_internal_validation.csv",
    )
    save_csv(
        threshold_grid,
        threshold_dir / "internal_validation_threshold_grid_metrics.csv",
    )
    save_csv(
        youden_curve,
        threshold_dir / "internal_validation_youden_thresholds.csv",
    )

    operating_rows = []
    for row in candidates.itertuples(index=False):
        y_pred = (y_validation_prob >= float(row.threshold)).astype(int)
        point = calculate_binary_metrics_from_predictions(
            y_validation, y_validation_prob, y_pred
        )
        operating_rows.append(
            {
                "strategy": row.strategy,
                "threshold": float(row.threshold),
                "derivation_dataset": row.derivation_dataset,
                "criterion": row.criterion,
                **point,
            }
        )
    save_csv(
        pd.DataFrame(operating_rows),
        threshold_dir
        / "candidate_thresholds_internal_validation_operating_metrics.csv",
    )

    strategies = set(candidates["strategy"])
    selected_strategy = config.selected_threshold_strategy
    if selected_strategy not in strategies:
        if "youden" not in strategies:
            raise RuntimeError(
                f"Requested threshold strategy {selected_strategy!r} is unavailable "
                "and Youden threshold could not be derived."
            )
        LOGGER.warning(
            "Requested threshold strategy %r unavailable; falling back to 'youden'.",
            selected_strategy,
        )
        selected_strategy = "youden"

    selected_threshold = float(
        candidates.loc[candidates["strategy"] == selected_strategy, "threshold"].iloc[0]
    )
    LOGGER.info(
        "Selected threshold strategy=%s threshold=%.6f",
        selected_strategy,
        selected_threshold,
    )

    cohort_payloads = {
        "tuning_train": (y_train, y_train_prob),
        "internal_validation": (y_validation, y_validation_prob),
        config.external_cohort_name: (y_external, y_external_prob),
    }

    all_points = []
    all_summaries = []
    all_confusions = []
    for cohort_idx, (cohort, (y_true, y_prob)) in enumerate(cohort_payloads.items()):
        cohort_points = []
        cohort_summaries = []
        cohort_confusions = []

        for strategy_idx, row in enumerate(candidates.itertuples(index=False)):
            strategy = str(row.strategy)
            threshold = float(row.threshold)
            y_pred = (y_prob >= threshold).astype(int)
            point = calculate_binary_metrics_from_predictions(y_true, y_prob, y_pred)
            point_row = {
                "cohort": cohort,
                "threshold_strategy": strategy,
                "threshold": threshold,
                "model": MODEL_NAME,
                "stacker": config.stacker,
                "base_c": config.base_c,
                **point,
            }
            cohort_points.append(point_row)
            all_points.append(point_row)

            _, summary = bootstrap_binary_metrics_from_predictions(
                y_true,
                y_prob,
                y_pred,
                n_bootstraps=config.n_bootstraps,
                seed=(
                    config.bootstrap_seed
                    + cohort_idx * 10000
                    + strategy_idx * 1000
                ),
            )
            summary = summary.copy()
            summary.insert(0, "cohort", cohort)
            summary.insert(1, "threshold_strategy", strategy)
            summary.insert(2, "threshold", threshold)
            summary.insert(3, "model", MODEL_NAME)
            summary.insert(4, "stacker", config.stacker)
            summary.insert(5, "base_c", config.base_c)
            cohort_summaries.append(summary)
            all_summaries.append(summary)

            confusion = confusion_matrix_long(
                cohort,
                strategy,
                threshold,
                y_true,
                y_pred,
            )
            confusion["model"] = MODEL_NAME
            confusion["stacker"] = config.stacker
            confusion["base_c"] = config.base_c
            cohort_confusions.append(confusion)
            all_confusions.append(confusion)

        save_csv(
            pd.DataFrame(cohort_points),
            threshold_dir / f"{slugify(cohort)}_point_metrics_by_threshold.csv",
        )
        save_csv(
            pd.concat(cohort_summaries, ignore_index=True),
            threshold_dir
            / f"{slugify(cohort)}_bootstrap_metrics_by_threshold_long.csv",
        )
        save_csv(
            pd.concat(cohort_confusions, ignore_index=True),
            threshold_dir
            / f"{slugify(cohort)}_confusion_matrices_by_threshold_long.csv",
        )

    all_points_df = pd.DataFrame(all_points)
    all_summaries_df = pd.concat(all_summaries, ignore_index=True)
    all_confusions_df = pd.concat(all_confusions, ignore_index=True)
    save_csv(
        all_points_df,
        threshold_dir / "all_cohorts_point_metrics_by_threshold.csv",
    )
    save_csv(
        all_summaries_df,
        threshold_dir / "all_cohorts_bootstrap_metrics_by_threshold_long.csv",
    )
    save_csv(
        all_confusions_df,
        threshold_dir / "all_cohorts_confusion_matrices_by_threshold_long.csv",
    )

    selected_points = all_points_df[
        all_points_df["threshold_strategy"] == selected_strategy
    ].copy()
    selected_summaries = all_summaries_df[
        all_summaries_df["threshold_strategy"] == selected_strategy
    ].copy()
    selected_confusions = all_confusions_df[
        all_confusions_df["threshold_strategy"] == selected_strategy
    ].copy()
    save_csv(
        selected_points,
        threshold_dir / f"selected_threshold_{selected_strategy}_point_metrics.csv",
    )
    save_csv(
        selected_summaries,
        threshold_dir / f"selected_threshold_{selected_strategy}_metrics_long.csv",
    )
    save_csv(
        selected_confusions,
        threshold_dir
        / f"selected_threshold_{selected_strategy}_confusion_matrices_long.csv",
    )

    model_package = {
        "model_name": MODEL_NAME,
        "architecture": "late_fusion_stacking",
        "stacker": config.stacker,
        "tabpfn_model_version": (
            config.tabpfn_model_version if config.stacker == "tabpfn" else None
        ),
        "base_c": config.base_c,
        "selected_threshold_strategy": selected_strategy,
        "selected_threshold": selected_threshold,
        "late_fusion_package": package,
        "logistic_recalibrator": recalibrator,
        "feature_space": feature_space,
        "candidate_thresholds": candidates,
        "modalities": modalities,
        "stacker_feature_names": package["stacker_feature_names"],
        "config": serialize_config(config),
    }
    model_path = config.output_dir / "late_fusion_external_validation_model.joblib"
    joblib.dump(model_package, model_path)
    LOGGER.info("Saved model package: %s", model_path)

    LOGGER.info(
        "External validation complete: cohort=%s n=%d",
        config.external_cohort_name,
        len(external_df),
    )


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    config = parse_args(argv)
    set_global_seed(config.random_state)
    run_external_validation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
