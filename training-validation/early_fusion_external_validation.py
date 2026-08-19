#!/usr/bin/env python3
"""External validation of the early-fusion multimodal fever model.

Publication-oriented refactor of the external-validation notebook used in the
study. The script intentionally reuses feature/model utilities from
``early_fusion_nested_cv.py`` so that the internal nested-CV and external-
validation pipelines cannot silently drift apart.

Primary early-fusion modalities
-------------------------------
The reported model combines:

- structured/tabular encounter-level variables;
- Qwen clinical-note embeddings;
- Chronos-2 body-temperature embeddings; and
- MedImageInsight CT image embeddings.

All three frozen embedding modalities are required for both the UME development
cohort and the external validation cohort. Missing embeddings for individual
encounters are zero-filled and represented by modality-availability indicators,
matching the nested-CV implementation.

Validation design
-----------------
The procedure preserves the source notebook exactly at the methodological level:

1. Split the UME development cohort into 80% tuning-training and 20% internal
   validation data using a stratified split and a fixed random state.
2. Fit tabular preprocessing only on the UME tuning-training subset.
3. Select the L2-regularized logistic-regression ``C`` on the UME internal
   validation set only.
4. Fit logistic recalibration (Platt-style scaling on the prediction logit) on
   the UME internal-validation predictions only.
5. Derive operating thresholds from the recalibrated UME internal-validation
   probabilities only.
6. Apply the fitted model, recalibrator, and thresholds unchanged to the
   external cohort. The external cohort is never used for fitting, model
   selection, recalibration, or threshold selection.

Embedding format
----------------
Each embedding modality is supplied as a ``.npy`` matrix and a CSV index with
``<id_col>`` and ``embedding_row`` columns, as produced by the feature-extraction
scripts in this repository.

Example
-------
python early_fusion_external_validation.py \\
    --train-tabular-csv data/ume_adapted.csv \\
    --external-tabular-csv data/josef_adapted.csv \\
    --external-cohort-name josef \\
    --output-dir results/early_fusion_external_josef \\
    --train-text-embeddings data/ume_note_embeddings.npy \\
    --train-text-index data/ume_note_embedding_index.csv \\
    --external-text-embeddings data/josef_note_embeddings.npy \\
    --external-text-index data/josef_note_embedding_index.csv \\
    --train-time-series-embeddings data/ume_body_temperature_chronos2_embeddings.npy \\
    --train-time-series-index data/ume_body_temperature_chronos2_index.csv \\
    --external-time-series-embeddings data/josef_body_temperature_chronos2_embeddings.npy \\
    --external-time-series-index data/josef_body_temperature_chronos2_index.csv \\
    --train-ct-embeddings data/ume_ct_medimageinsight_embeddings.npy \\
    --train-ct-index data/ume_ct_medimageinsight_index.csv \\
    --external-ct-embeddings data/josef_ct_medimageinsight_embeddings.npy \\
    --external-ct-index data/josef_ct_medimageinsight_index.csv

Notes
-----
Raw clinical data and pretrained foundation-model weights are intentionally not
bundled with this script. Generate frozen embeddings first with the scripts in
``feature-extraction/``.
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

# Reuse the exact preprocessing, fusion, model, threshold, and metric utilities
# from the nested-CV implementation. Running this file directly works because
# Python adds the script directory to sys.path.
from early_fusion_nested_cv import (
    DEFAULT_EXCLUDE_COLUMNS,
    Config as NestedCVConfig,
    EmbeddingSource,
    bootstrap_binary_metrics_from_predictions,
    build_preprocessor,
    calculate_binary_metrics_from_predictions,
    calibration_intercept_slope,
    concatenate_modalities,
    confusion_matrix_long,
    ensure_columns_exist,
    find_youden_threshold,
    fit_early_fusion_model,
    get_feature_columns,
    load_embedding_source,
    lookup_matrix_for_ids,
    normalize_encounter_reference,
    predict_probability,
    read_csv_robust,
    score_inner_predictions,
    threshold_grid_metrics,
    to_dense_float32,
)

LOGGER = logging.getLogger("early_fusion_external_validation")


@dataclass(frozen=True)
class ExternalValidationConfig:
    """Runtime configuration for external validation."""

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
class EarlyFusionFeatureSpace:
    """Train-fitted feature definition reused unchanged for validation cohorts."""

    tabular_feature_cols: list[str]
    tabular_preprocessor: Any | None
    concept_cols: list[str]
    concept_preprocessor: Any | None
    modalities: list[str]
    feature_names: list[str]


@dataclass
class LogisticRecalibrator:
    """Logistic recalibration fitted on internal-validation predictions."""

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
            "External validation of the publication early-fusion multimodal "
            "fever model with internal UME model selection and recalibration."
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
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1, 1.0],
        help="Candidate inverse L2 regularization strengths.",
    )
    parser.add_argument(
        "--c-selection-metric",
        choices=["AUROC", "AUPRC", "Brier score"],
        default="AUPRC",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
    )
    parser.add_argument("--logistic-max-iter", type=int, default=2000)
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
        help=(
            "Columns excluded from the tabular branch. If supplied, replaces "
            "the publication default exclusion list."
        ),
    )

    # Primary study modalities: required for both development and external data.
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
    if config.threshold_grid_size < 2:
        raise ValueError("threshold_grid_size must be >= 2.")
    if not 0.0 <= config.min_sensitivity <= 1.0:
        raise ValueError("min_sensitivity must be between 0 and 1.")
    if not 0.0 <= config.min_specificity <= 1.0:
        raise ValueError("min_specificity must be between 0 and 1.")
    if not config.c_grid or any(c <= 0 for c in config.c_grid):
        raise ValueError("All values in c_grid must be > 0.")


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
    """Load development/external frozen embeddings and verify matching dimensions."""
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
            modality,
            train_embeddings,
            train_index,
            config.id_col,
        )
        external_sources[modality] = load_embedding_source(
            modality,
            external_embeddings,
            external_index,
            config.id_col,
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
) -> tuple[EarlyFusionFeatureSpace, np.ndarray, dict[str, np.ndarray]]:
    """Fit tabular preprocessing once and construct the tuning-training matrix."""
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
            id_col=config.id_col,
            label_col=config.label_col,
            concept_cols=config.concept_cols,
            exclude_cols=config.exclude_cols,
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
        source = train_sources[modality]
        matrix, mask = lookup_matrix_for_ids(fit_df[config.id_col], source)
        features[modality] = matrix
        masks[modality] = mask

    modalities = [
        name
        for name in ("tabular", "text", "concepts", "time_series", "ct_image")
        if name in features
    ]
    x_fit, feature_names = concatenate_modalities(features, masks, modalities)

    feature_space = EarlyFusionFeatureSpace(
        tabular_feature_cols=tabular_feature_cols,
        tabular_preprocessor=tabular_preprocessor,
        concept_cols=concept_cols,
        concept_preprocessor=concept_preprocessor,
        modalities=modalities,
        feature_names=feature_names,
    )
    return feature_space, x_fit, masks


def transform_feature_space(
    df: pd.DataFrame,
    feature_space: EarlyFusionFeatureSpace,
    config: ExternalValidationConfig,
    embedding_sources: Mapping[str, EmbeddingSource],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Transform a cohort using only train-fitted preprocessing objects."""
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
            feature_space.concept_preprocessor.transform(
                df[feature_space.concept_cols]
            )
        )
        masks["concepts"] = np.ones(len(df), dtype=bool)

    for modality in ("text", "time_series", "ct_image"):
        if modality not in feature_space.modalities:
            continue
        source = embedding_sources[modality]
        matrix, mask = lookup_matrix_for_ids(df[config.id_col], source)
        features[modality] = matrix
        masks[modality] = mask

    x, feature_names = concatenate_modalities(
        features, masks, feature_space.modalities
    )
    if feature_names != feature_space.feature_names:
        raise RuntimeError(
            "Feature definitions differ between tuning-training and evaluation data."
        )
    return x, masks


def valid_stratify_series(y: pd.Series) -> pd.Series | None:
    value_counts = y.value_counts(dropna=False)
    if y.nunique(dropna=False) <= 20 and value_counts.min() >= 2:
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
    return (
        tuning_train.reset_index(drop=True),
        internal_validation.reset_index(drop=True),
    )


def nested_cv_compat_config(config: ExternalValidationConfig) -> NestedCVConfig:
    """Create the shared model-config object used by nested-CV helper functions."""
    return NestedCVConfig(
        tabular_csv=config.train_tabular_csv,
        output_dir=config.output_dir,
        id_col=config.id_col,
        label_col=config.label_col,
        random_state=config.random_state,
        n_bootstraps=config.n_bootstraps,
        bootstrap_seed=config.bootstrap_seed,
        c_grid=config.c_grid,
        c_selection_metric=config.c_selection_metric,
        class_weight=config.class_weight,
        logistic_max_iter=config.logistic_max_iter,
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


def select_logistic_c_on_internal_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: ExternalValidationConfig,
) -> tuple[float, object, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Select logistic C on internal UME validation predictions only."""
    model_config = nested_cv_compat_config(config)
    rows = []
    fitted: dict[float, tuple[object, np.ndarray, pd.DataFrame]] = {}

    for c_value in config.c_grid:
        model, fit_info = fit_early_fusion_model(
            x_train,
            y_train,
            model_config,
            logistic_c=float(c_value),
            stage="tuning_train_candidate_fit",
        )
        y_prob = predict_probability(model, x_validation)
        scores = score_inner_predictions(y_validation, y_prob)
        rows.append(
            {
                "classifier": "logistic",
                "logistic_c": float(c_value),
                **scores,
            }
        )
        fitted[float(c_value)] = (model, y_prob, fit_info)

    hpo_df = pd.DataFrame(rows)
    metric = config.c_selection_metric
    if metric == "Brier score":
        best_idx = hpo_df[metric].astype(float).idxmin()
    else:
        best_idx = hpo_df[metric].astype(float).idxmax()

    selected_c = float(hpo_df.loc[best_idx, "logistic_c"])
    hpo_df["selection_metric"] = metric
    hpo_df["selected"] = hpo_df["logistic_c"].eq(selected_c)
    model, y_prob, fit_info = fitted[selected_c]
    fit_info = fit_info.copy()
    fit_info["selected_logistic_c"] = selected_c
    return selected_c, model, y_prob, hpo_df, fit_info


def logit_np(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(probabilities / (1.0 - probabilities))


def fit_logistic_recalibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> LogisticRecalibrator:
    """Fit the source-notebook recalibration model on internal validation only."""
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
    before_intercept, before_slope = calibration_intercept_slope(
        y_true, y_prob_before
    )
    after_intercept, after_slope = calibration_intercept_slope(
        y_true, y_prob_after
    )
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
    """Derive the same five operating thresholds as the nested-CV pipeline."""
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
        },
    }
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def run_external_validation(config: ExternalValidationConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_dir = config.output_dir / "threshold_analysis"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_encounter_table(
        config.train_tabular_csv, config.id_col, config.label_col
    )
    external_df = load_encounter_table(
        config.external_tabular_csv, config.id_col, config.label_col
    )
    train_sources, external_sources = load_embedding_pair(config)
    save_run_metadata(config, train_sources, external_sources)

    tuning_train_df, internal_validation_df = split_development_cohort(
        train_df, config
    )
    LOGGER.info(
        "Development split: full=%d tuning_train=%d internal_validation=%d",
        len(train_df),
        len(tuning_train_df),
        len(internal_validation_df),
    )

    feature_space, x_train, train_masks = fit_feature_space(
        tuning_train_df, config, train_sources
    )
    x_validation, validation_masks = transform_feature_space(
        internal_validation_df, feature_space, config, train_sources
    )
    x_external, external_masks = transform_feature_space(
        external_df, feature_space, config, external_sources
    )

    y_train = tuning_train_df[config.label_col].astype(int).to_numpy()
    y_validation = internal_validation_df[config.label_col].astype(int).to_numpy()
    y_external = external_df[config.label_col].astype(int).to_numpy()

    ids_train = tuning_train_df[config.id_col].astype(str).to_numpy()
    ids_validation = internal_validation_df[config.id_col].astype(str).to_numpy()
    ids_external = external_df[config.id_col].astype(str).to_numpy()

    feature_info = pd.DataFrame(
        [
            {
                "model": "early_fusion_external_validation",
                "classifier": "logistic",
                "external_cohort": config.external_cohort_name,
                "n_development_full": len(train_df),
                "n_tuning_train": len(tuning_train_df),
                "n_internal_validation": len(internal_validation_df),
                "n_external": len(external_df),
                "n_features": int(x_train.shape[1]),
                "n_modalities": len(feature_space.modalities),
                "modalities": ",".join(feature_space.modalities),
                "n_tabular_source_columns": len(feature_space.tabular_feature_cols),
            }
        ]
    )
    save_csv(feature_info, config.output_dir / "external_validation_feature_info.csv")
    save_csv(
        pd.DataFrame({"feature_name": feature_space.feature_names}),
        config.output_dir / "external_validation_feature_names.csv",
    )
    save_csv(
        availability_table(
            {
                "tuning_train": train_masks,
                "internal_validation": validation_masks,
                config.external_cohort_name: external_masks,
            },
            feature_space.modalities,
        ),
        config.output_dir / "external_validation_modality_availability.csv",
    )

    selected_c, selected_model, y_validation_prob_uncalibrated, hpo_df, fit_info = (
        select_logistic_c_on_internal_validation(
            x_train,
            y_train,
            x_validation,
            y_validation,
            config,
        )
    )
    LOGGER.info("Selected logistic C=%g by %s", selected_c, config.c_selection_metric)
    save_csv(
        hpo_df,
        config.output_dir / "external_validation_logistic_c_grid_search.csv",
    )
    save_csv(fit_info, config.output_dir / "external_validation_training_info.csv")

    y_train_prob_uncalibrated = predict_probability(selected_model, x_train)
    y_external_prob_uncalibrated = predict_probability(selected_model, x_external)

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
    recalibration_df["model"] = "early_fusion_external_validation"
    recalibration_df["classifier"] = "logistic"
    recalibration_df["selected_logistic_c"] = selected_c
    save_csv(
        recalibration_df,
        config.output_dir
        / "external_validation_logistic_recalibration_internal_validation.csv",
    )

    prediction_tables = {
        "tuning_train": pd.DataFrame(
            {
                config.id_col: ids_train,
                "y_true": y_train,
                "y_prob_uncalibrated": y_train_prob_uncalibrated,
                "y_prob": y_train_prob,
            }
        ),
        "internal_validation": pd.DataFrame(
            {
                config.id_col: ids_validation,
                "y_true": y_validation,
                "y_prob_uncalibrated": y_validation_prob_uncalibrated,
                "y_prob": y_validation_prob,
            }
        ),
        config.external_cohort_name: pd.DataFrame(
            {
                config.id_col: ids_external,
                "y_true": y_external,
                "y_prob_uncalibrated": y_external_prob_uncalibrated,
                "y_prob": y_external_prob,
            }
        ),
    }
    for cohort, table in prediction_tables.items():
        table["model"] = "early_fusion_external_validation"
        table["classifier"] = "logistic"
        table["selected_logistic_c"] = selected_c
        save_csv(
            table,
            config.output_dir / f"{slugify(cohort)}_predictions_early_fusion.csv",
        )

    candidates, threshold_grid, youden_curve = (
        build_candidate_thresholds_from_internal_validation(
            y_validation, y_validation_prob, config
        )
    )
    candidates["selected_logistic_c"] = selected_c
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
        point = calculate_binary_metrics_from_predictions(
            y_validation,
            y_validation_prob,
            (y_validation_prob >= float(row.threshold)).astype(int),
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
    for cohort_idx, (cohort, (y_true, y_prob)) in enumerate(
        cohort_payloads.items()
    ):
        cohort_points = []
        cohort_summaries = []
        cohort_confusions = []

        for strategy_idx, row in enumerate(candidates.itertuples(index=False)):
            strategy = row.strategy
            threshold = float(row.threshold)
            y_pred = (y_prob >= threshold).astype(int)
            point = calculate_binary_metrics_from_predictions(y_true, y_prob, y_pred)
            point_row = {
                "cohort": cohort,
                "threshold_strategy": strategy,
                "threshold": threshold,
                "model": "early_fusion_external_validation",
                "classifier": "logistic",
                "selected_logistic_c": selected_c,
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
            summary.insert(3, "model", "early_fusion_external_validation")
            summary.insert(4, "classifier", "logistic")
            summary.insert(5, "selected_logistic_c", selected_c)
            cohort_summaries.append(summary)
            all_summaries.append(summary)

            confusion = confusion_matrix_long(
                cohort,
                strategy,
                threshold,
                y_true,
                y_pred,
            )
            confusion["model"] = "early_fusion_external_validation"
            confusion["classifier"] = "logistic"
            confusion["selected_logistic_c"] = selected_c
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
        "model_name": "early_fusion_external_validation",
        "classifier": "logistic",
        "selected_logistic_c": selected_c,
        "selected_threshold_strategy": selected_strategy,
        "selected_threshold": selected_threshold,
        "model": selected_model,
        "logistic_recalibrator": recalibrator,
        "feature_space": feature_space,
        "candidate_thresholds": candidates,
        "modalities": feature_space.modalities,
        "feature_names": feature_space.feature_names,
        "config": serialize_config(config),
    }
    model_path = config.output_dir / "early_fusion_external_validation_model.joblib"
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
