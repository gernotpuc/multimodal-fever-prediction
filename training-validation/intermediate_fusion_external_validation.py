#!/usr/bin/env python3
"""External validation for the intermediate-fusion multimodal fever model.

This script is the external-validation counterpart of
``intermediate_fusion_nested_cv.py``. It preserves the same representation-level
fusion architecture and model-selection logic while ensuring that the external
cohort is never used for fitting, hyperparameter selection, recalibration, or
threshold selection.

Model architecture
------------------
Frozen modality representations are supplied for:

- clinical notes (Qwen/Qwen3-Embedding-0.6B),
- body-temperature trajectories (Chronos-2),
- CT images (MedImageInsight),
- structured/tabular variables transformed using preprocessing fitted on UME.

Each modality is encoded/projected into a shared token space. A Transformer
encoder fuses the modality tokens together with a learned CLS token, and the
fused CLS representation is passed to the binary prediction head.

External-validation design
--------------------------
The full UME development cohort is treated analogously to an outer-training set
in nested CV:

1. Perform inner stratified CV on UME for hyperparameter selection.
2. Use the selected trial's UME out-of-fold predictions to fit logistic
   recalibration and derive operating thresholds.
3. Determine the final training epoch count from the selected inner-fold best
   epochs (median by default).
4. Fit structured-data preprocessing and one final Transformer model on all UME.
5. Apply the frozen model, recalibrator, and thresholds once to the external
   cohort.

The external cohort is never used for model fitting, HPO, recalibration, or
threshold selection.

Expected embedding format
-------------------------
Each modality is provided separately for UME and the external cohort as:

- ``*.npy``: 2-D embedding matrix;
- ``*.csv``: encounter index containing ``embedding_row``.

Example
-------
python intermediate_fusion_external_validation.py \
    --train-tabular-csv data/ume_adapted.csv \
    --external-tabular-csv data/josef_adapted.csv \
    --external-cohort-name josef \
    --output-dir results/intermediate_fusion_external_josef \
    --train-text-embeddings data/ume_note_embeddings.npy \
    --train-text-index data/ume_note_embedding_index.csv \
    --external-text-embeddings data/josef_note_embeddings.npy \
    --external-text-index data/josef_note_embedding_index.csv \
    --train-time-series-embeddings data/ume_body_temp_chronos2_embeddings.npy \
    --train-time-series-index data/ume_body_temp_chronos2_index.csv \
    --external-time-series-embeddings data/josef_body_temp_chronos2_embeddings.npy \
    --external-time-series-index data/josef_body_temp_chronos2_index.csv \
    --train-ct-embeddings data/ume_ct_medimageinsight_embeddings.npy \
    --train-ct-index data/ume_ct_medimageinsight_index.csv \
    --external-ct-embeddings data/josef_ct_medimageinsight_embeddings.npy \
    --external-ct-index data/josef_ct_medimageinsight_index.csv

Raw foundation-model inference is intentionally separated into the repository's
feature-extraction scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, confusion_matrix
from torch.utils.data import DataLoader

# This script is intended to live beside intermediate_fusion_nested_cv.py.
import intermediate_fusion_nested_cv as cv

LOGGER = logging.getLogger("intermediate_fusion_external_validation")


@dataclass(frozen=True)
class Config:
    train_tabular_csv: Path
    external_tabular_csv: Path
    output_dir: Path
    external_cohort_name: str

    train_text_embeddings: Path
    train_text_index: Path
    external_text_embeddings: Path
    external_text_index: Path

    train_time_series_embeddings: Path
    train_time_series_index: Path
    external_time_series_embeddings: Path
    external_time_series_index: Path

    train_ct_embeddings: Path
    train_ct_index: Path
    external_ct_embeddings: Path
    external_ct_index: Path

    id_col: str = "encounter_id"
    label_col: str = "fever"
    random_state: int = 42
    inner_folds: int = 3
    batch_size: int = 32
    hpo_trials: int = 10
    hpo_max_epochs: int = 40
    hpo_patience: int = 6
    hpo_selection_metric: str = "val_auprc"
    final_epoch_aggregation: str = "median_inner_best_epoch"
    n_bootstraps: int = 2000
    bootstrap_seed: int = 42
    threshold_grid_size: int = 999
    min_sensitivity: float = 0.80
    min_specificity: float = 0.80
    concept_cols: tuple[str, ...] = ()
    exclude_cols: tuple[str, ...] = tuple(cv.DEFAULT_EXCLUDE_COLUMNS)
    device: str = "auto"
    num_workers: int = 0


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="External validation for the intermediate-fusion multimodal fever model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-tabular-csv", type=Path, required=True)
    parser.add_argument("--external-tabular-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-cohort-name", default="external")

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

    parser.add_argument("--id-col", default="encounter_id")
    parser.add_argument("--label-col", default="fever")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hpo-trials", type=int, default=10)
    parser.add_argument("--hpo-max-epochs", type=int, default=40)
    parser.add_argument("--hpo-patience", type=int, default=6)
    parser.add_argument(
        "--hpo-selection-metric",
        choices=["val_loss", "val_auroc", "val_auprc"],
        default="val_auprc",
    )
    parser.add_argument(
        "--final-epoch-aggregation",
        choices=["median_inner_best_epoch", "mean_inner_best_epoch"],
        default="median_inner_best_epoch",
    )
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--threshold-grid-size", type=int, default=999)
    parser.add_argument("--min-sensitivity", type=float, default=0.80)
    parser.add_argument("--min-specificity", type=float, default=0.80)
    parser.add_argument("--concept-cols", nargs="*", default=[])
    parser.add_argument("--exclude-cols", nargs="*", default=cv.DEFAULT_EXCLUDE_COLUMNS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num-workers", type=int, default=0)

    ns = parser.parse_args(argv)
    return Config(
        train_tabular_csv=ns.train_tabular_csv,
        external_tabular_csv=ns.external_tabular_csv,
        output_dir=ns.output_dir,
        external_cohort_name=ns.external_cohort_name,
        train_text_embeddings=ns.train_text_embeddings,
        train_text_index=ns.train_text_index,
        external_text_embeddings=ns.external_text_embeddings,
        external_text_index=ns.external_text_index,
        train_time_series_embeddings=ns.train_time_series_embeddings,
        train_time_series_index=ns.train_time_series_index,
        external_time_series_embeddings=ns.external_time_series_embeddings,
        external_time_series_index=ns.external_time_series_index,
        train_ct_embeddings=ns.train_ct_embeddings,
        train_ct_index=ns.train_ct_index,
        external_ct_embeddings=ns.external_ct_embeddings,
        external_ct_index=ns.external_ct_index,
        id_col=ns.id_col,
        label_col=ns.label_col,
        random_state=ns.random_state,
        inner_folds=ns.inner_folds,
        batch_size=ns.batch_size,
        hpo_trials=ns.hpo_trials,
        hpo_max_epochs=ns.hpo_max_epochs,
        hpo_patience=ns.hpo_patience,
        hpo_selection_metric=ns.hpo_selection_metric,
        final_epoch_aggregation=ns.final_epoch_aggregation,
        n_bootstraps=ns.n_bootstraps,
        bootstrap_seed=ns.bootstrap_seed,
        threshold_grid_size=ns.threshold_grid_size,
        min_sensitivity=ns.min_sensitivity,
        min_specificity=ns.min_specificity,
        concept_cols=tuple(ns.concept_cols),
        exclude_cols=tuple(ns.exclude_cols),
        device=ns.device,
        num_workers=ns.num_workers,
    )


def as_cv_config(config: Config) -> cv.Config:
    """Create the shared CV configuration used by common helper functions."""
    return cv.Config(
        tabular_csv=config.train_tabular_csv,
        output_dir=config.output_dir,
        text_embeddings=config.train_text_embeddings,
        text_index=config.train_text_index,
        time_series_embeddings=config.train_time_series_embeddings,
        time_series_index=config.train_time_series_index,
        ct_embeddings=config.train_ct_embeddings,
        ct_index=config.train_ct_index,
        id_col=config.id_col,
        label_col=config.label_col,
        random_state=config.random_state,
        outer_folds=5,  # unused in this external-validation script
        inner_folds=config.inner_folds,
        batch_size=config.batch_size,
        hpo_trials=config.hpo_trials,
        hpo_max_epochs=config.hpo_max_epochs,
        hpo_patience=config.hpo_patience,
        hpo_selection_metric=config.hpo_selection_metric,
        final_epoch_aggregation=config.final_epoch_aggregation,
        n_bootstraps=config.n_bootstraps,
        bootstrap_seed=config.bootstrap_seed,
        threshold_grid_size=config.threshold_grid_size,
        min_sensitivity=config.min_sensitivity,
        min_specificity=config.min_specificity,
        concept_cols=config.concept_cols,
        exclude_cols=config.exclude_cols,
        device=config.device,
        num_workers=config.num_workers,
    )


def load_dataframe(path: Path, config: Config, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = cv.read_table(path).copy()
    if config.id_col not in df.columns:
        raise ValueError(f"{name}: missing ID column {config.id_col!r}.")
    if config.label_col not in df.columns:
        raise ValueError(f"{name}: missing label column {config.label_col!r}.")
    df[config.id_col] = df[config.id_col].map(cv.normalize_encounter_reference)
    if df[config.id_col].isna().any():
        raise ValueError(f"{name}: missing encounter IDs after normalization.")
    if df[config.id_col].duplicated().any():
        examples = df.loc[df[config.id_col].duplicated(keep=False), config.id_col].head(10).tolist()
        raise ValueError(f"{name}: encounter IDs must be unique. Examples: {examples}")
    df[config.label_col] = pd.to_numeric(df[config.label_col], errors="raise").astype(int)
    labels = sorted(df[config.label_col].dropna().unique().tolist())
    if not set(labels).issubset({0, 1}) or len(labels) < 2:
        raise ValueError(f"{name}: label must be binary with both classes present; found {labels}.")
    return df.reset_index(drop=True)


def load_sources(config: Config) -> tuple[dict[str, cv.EmbeddingSource], dict[str, cv.EmbeddingSource]]:
    train_sources = {
        "text": cv.load_embedding_source(
            "text", config.train_text_embeddings, config.train_text_index, config.id_col
        ),
        "time_series": cv.load_embedding_source(
            "time_series", config.train_time_series_embeddings, config.train_time_series_index, config.id_col
        ),
        "ct_image": cv.load_embedding_source(
            "ct_image", config.train_ct_embeddings, config.train_ct_index, config.id_col
        ),
    }
    external_sources = {
        "text": cv.load_embedding_source(
            "text", config.external_text_embeddings, config.external_text_index, config.id_col
        ),
        "time_series": cv.load_embedding_source(
            "time_series", config.external_time_series_embeddings, config.external_time_series_index, config.id_col
        ),
        "ct_image": cv.load_embedding_source(
            "ct_image", config.external_ct_embeddings, config.external_ct_index, config.id_col
        ),
    }
    for name in train_sources:
        if train_sources[name].dim != external_sources[name].dim:
            raise ValueError(
                f"Embedding dimension mismatch for {name}: "
                f"UME={train_sources[name].dim}, external={external_sources[name].dim}."
            )
    return train_sources, external_sources


def prepare_train_external_data(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    shared_config: cv.Config,
    train_sources: Mapping[str, cv.EmbeddingSource],
    external_sources: Mapping[str, cv.EmbeddingSource],
) -> dict[str, object]:
    """Fit preprocessing on full UME and create UME/external DataLoaders."""
    train_df = train_df.copy().reset_index(drop=True)
    external_df = external_df.copy().reset_index(drop=True)

    feature_cols = cv.get_feature_columns(
        train_df,
        shared_config.id_col,
        shared_config.label_col,
        shared_config.concept_cols,
        shared_config.exclude_cols,
    )
    external_df = cv.ensure_columns_exist(
        external_df,
        [*feature_cols, *shared_config.concept_cols],
        "external cohort",
    )

    tab_preprocessor, _, _ = cv.build_preprocessor(train_df, feature_cols)
    x_train_tab = cv.to_dense_float32(tab_preprocessor.fit_transform(train_df[feature_cols]))
    x_external_tab = cv.to_dense_float32(tab_preprocessor.transform(external_df[feature_cols]))

    if shared_config.concept_cols:
        missing = [c for c in shared_config.concept_cols if c not in train_df.columns]
        if missing:
            raise ValueError(f"Concept columns absent from UME development data: {missing}")
        concept_preprocessor, _, _ = cv.build_preprocessor(train_df, list(shared_config.concept_cols))
        x_train_concepts = cv.to_dense_float32(
            concept_preprocessor.fit_transform(train_df[list(shared_config.concept_cols)])
        )
        x_external_concepts = cv.to_dense_float32(
            concept_preprocessor.transform(external_df[list(shared_config.concept_cols)])
        )
        concept_dim = int(x_train_concepts.shape[1])
    else:
        concept_preprocessor = None
        x_train_concepts = None
        x_external_concepts = None
        concept_dim = None

    train_ds = cv.MultimodalEncounterDataset(
        train_df,
        shared_config.id_col,
        shared_config.label_col,
        x_train_tab,
        train_sources["text"],
        train_sources["time_series"],
        train_sources["ct_image"],
        x_train_concepts,
    )
    external_ds = cv.MultimodalEncounterDataset(
        external_df,
        shared_config.id_col,
        shared_config.label_col,
        x_external_tab,
        external_sources["text"],
        external_sources["time_series"],
        external_sources["ct_image"],
        x_external_concepts,
    )

    loader_kwargs = dict(collate_fn=cv.multimodal_collate_fn, num_workers=shared_config.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=shared_config.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=shared_config.batch_size * 2,
        shuffle=False,
        **loader_kwargs,
    )
    external_loader = DataLoader(
        external_ds,
        batch_size=shared_config.batch_size * 2,
        shuffle=False,
        **loader_kwargs,
    )

    return {
        "train_df": train_df,
        "eval_df": external_df,
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "eval_loader": external_loader,
        "dims": {
            "text_emb_dim": train_sources["text"].dim,
            "tabular_dim": int(x_train_tab.shape[1]),
            "time_series_emb_dim": train_sources["time_series"].dim,
            "ct_image_emb_dim": train_sources["ct_image"].dim,
            "concept_dim": concept_dim,
        },
        "feature_cols": feature_cols,
        "tabular_preprocessor": tab_preprocessor,
        "concept_preprocessor": concept_preprocessor,
    }


def run_hpo_on_ume(
    train_df: pd.DataFrame,
    shared_config: cv.Config,
    train_sources: Mapping[str, cv.EmbeddingSource],
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    """Select hyperparameters using leakage-free inner CV across full UME."""
    inner_cv = cv.make_stratified_kfold(
        train_df[shared_config.label_col], shared_config.inner_folds, shared_config.random_state
    )
    trials = cv.build_trial_hparams(shared_config.hpo_trials, shared_config.random_state + 1000)

    best_mean: float | None = None
    best_trial = 0
    best_hp: dict[str, object] | None = None
    best_preds: pd.DataFrame | None = None
    best_epochs: list[int] | None = None
    hpo_rows: list[dict[str, object]] = []

    hpo_dir = output_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    for trial_idx, hp in enumerate(trials, start=1):
        LOGGER.info("HPO trial %d/%d: %s", trial_idx, len(trials), hp)
        inner_rows: list[dict[str, object]] = []
        pred_frames: list[pd.DataFrame] = []
        history_frames: list[pd.DataFrame] = []

        # Recreate the splitter for every trial so folds are identical across trials.
        trial_cv = cv.make_stratified_kfold(
            train_df[shared_config.label_col], shared_config.inner_folds, shared_config.random_state
        )
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            trial_cv.split(train_df, train_df[shared_config.label_col]), start=1
        ):
            fold_data = cv.prepare_fold_data(
                train_df.iloc[inner_train_idx].reset_index(drop=True),
                train_df.iloc[inner_val_idx].reset_index(drop=True),
                shared_config,
                train_sources["text"],
                train_sources["time_series"],
                train_sources["ct_image"],
            )
            result = cv.train_with_early_stopping(
                hp,
                fold_data,
                device,
                seed=shared_config.random_state + trial_idx * 100 + inner_fold,
                max_epochs=shared_config.hpo_max_epochs,
                patience=shared_config.hpo_patience,
            )
            sel = cv.selection_value(result["best_row"], shared_config.hpo_selection_metric)
            inner_rows.append(
                {
                    "trial": trial_idx,
                    "inner_fold": inner_fold,
                    "best_epoch": result["best_epoch"],
                    "selection_value": sel,
                    **result["best_row"],
                }
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        shared_config.id_col: result["ids"],
                        "y_true": result["y_true"],
                        "y_prob": result["y_prob"],
                        "trial": trial_idx,
                        "inner_fold": inner_fold,
                    }
                )
            )
            history = result["history"].copy()
            history.insert(0, "trial", trial_idx)
            history.insert(1, "inner_fold", inner_fold)
            history_frames.append(history)
            del result["model"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        inner_df = pd.DataFrame(inner_rows)
        mean_sel = float(inner_df["selection_value"].mean())
        hpo_rows.append(
            {
                "trial": trial_idx,
                "selection_metric": shared_config.hpo_selection_metric,
                "mean_selection_value": mean_sel,
                "sd_selection_value": float(inner_df["selection_value"].std(ddof=1)),
                "mean_best_epoch": float(inner_df["best_epoch"].mean()),
                "median_best_epoch": float(inner_df["best_epoch"].median()),
                **cv.hparam_row(hp),
            }
        )
        pd.concat(history_frames, ignore_index=True).to_csv(
            hpo_dir / f"trial_{trial_idx:02d}_inner_training_history.csv", index=False
        )
        trial_preds = pd.concat(pred_frames, ignore_index=True)
        trial_preds.to_csv(hpo_dir / f"trial_{trial_idx:02d}_inner_oof_predictions.csv", index=False)

        if cv.better(mean_sel, best_mean, shared_config.hpo_selection_metric):
            best_mean = mean_sel
            best_trial = trial_idx
            best_hp = dict(hp)
            best_preds = trial_preds.copy()
            best_epochs = inner_df["best_epoch"].astype(int).tolist()

    if best_hp is None or best_preds is None or best_epochs is None:
        raise RuntimeError("No HPO trial was selected.")

    hpo_df = pd.DataFrame(hpo_rows)
    hpo_df["is_selected"] = hpo_df["trial"].eq(best_trial)
    hpo_df.to_csv(output_dir / "hpo_results.csv", index=False)

    return {
        "selected_trial": best_trial,
        "selected_hparams": best_hp,
        "selected_oof_predictions": best_preds,
        "selected_best_epochs": best_epochs,
        "hpo_results": hpo_df,
    }


def aggregate_epochs(best_epochs: Sequence[int], method: str) -> int:
    if method == "mean_inner_best_epoch":
        return max(1, int(round(float(np.mean(best_epochs)))))
    return max(1, int(round(float(np.median(best_epochs)))))


def save_run_config(
    config: Config,
    device: torch.device,
    train_sources: Mapping[str, cv.EmbeddingSource],
    external_sources: Mapping[str, cv.EmbeddingSource],
    selected_trial: int,
    selected_hparams: Mapping[str, object],
    final_epochs: int,
) -> None:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    payload.update(
        {
            "model_type": "intermediate_fusion_transformer",
            "fusion_level": "representation_level",
            "validation_type": "external_validation",
            "external_data_used_for_fitting": False,
            "development_strategy": "full_ume_inner_cv_then_final_full_ume_fit",
            "logistic_recalibration_source": "selected_ume_inner_cv_oof_predictions",
            "threshold_derivation_source": "recalibrated_selected_ume_inner_cv_oof_predictions",
            "ct_image_modality_enabled": True,
            "selected_hpo_trial": selected_trial,
            "selected_hparams": dict(selected_hparams),
            "final_epochs": final_epochs,
            "default_hparams": cv.DEFAULT_HPARAMS,
            "hpo_search_space": cv.HPO_SEARCH_SPACE,
            "device_resolved": str(device),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "embedding_sources": {
                "development": {
                    k: {"dim": v.dim, "n_rows": v.n_rows} for k, v in train_sources.items()
                },
                "external": {
                    k: {"dim": v.dim, "n_rows": v.n_rows} for k, v in external_sources.items()
                },
            },
        }
    )
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def run_external_validation(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shared_config = as_cv_config(config)
    device = cv.resolve_device(config.device)
    cv.set_global_seed(config.random_state)

    LOGGER.info("Using device: %s", device)
    train_df = load_dataframe(config.train_tabular_csv, config, "UME development data")
    external_df = load_dataframe(config.external_tabular_csv, config, config.external_cohort_name)
    train_sources, external_sources = load_sources(config)

    # Transparency: report modality availability separately in development and external data.
    train_availability = cv.modality_availability(
        train_df, shared_config, list(train_sources.values())
    )
    train_availability.insert(0, "cohort", "UME")
    external_availability = cv.modality_availability(
        external_df, shared_config, list(external_sources.values())
    )
    external_availability.insert(0, "cohort", config.external_cohort_name)
    pd.concat([train_availability, external_availability], ignore_index=True).to_csv(
        config.output_dir / "modality_availability.csv", index=False
    )

    # ------------------------------------------------------------------
    # 1. HPO on UME only; selected OOF probabilities remain out-of-sample.
    # ------------------------------------------------------------------
    hpo = run_hpo_on_ume(
        train_df,
        shared_config,
        train_sources,
        device,
        config.output_dir,
    )
    selected_trial = int(hpo["selected_trial"])
    selected_hparams = dict(hpo["selected_hparams"])
    selected_oof = hpo["selected_oof_predictions"].copy()
    selected_best_epochs = list(hpo["selected_best_epochs"])

    # ------------------------------------------------------------------
    # 2. Logistic recalibration on selected UME inner-CV OOF predictions.
    # ------------------------------------------------------------------
    selected_oof["y_prob_uncalibrated"] = selected_oof["y_prob"].astype(float)
    recalibrator = cv.fit_logistic_recalibrator(
        selected_oof["y_true"].to_numpy(),
        selected_oof["y_prob_uncalibrated"].to_numpy(),
    )
    selected_oof["y_prob"] = cv.apply_recalibrator(
        selected_oof["y_prob_uncalibrated"].to_numpy(), recalibrator
    )
    selected_oof.insert(0, "selected_hpo_trial", selected_trial)
    selected_oof.to_csv(
        config.output_dir / "selected_trial_ume_inner_oof_predictions.csv", index=False
    )

    unc_i, unc_s = cv.calibration_intercept_slope(
        selected_oof["y_true"].to_numpy(), selected_oof["y_prob_uncalibrated"].to_numpy()
    )
    rec_i, rec_s = cv.calibration_intercept_slope(
        selected_oof["y_true"].to_numpy(), selected_oof["y_prob"].to_numpy()
    )
    recalibration_row = {
        "selected_hpo_trial": selected_trial,
        **recalibrator,
        "ume_oof_uncalibrated_calibration_intercept": unc_i,
        "ume_oof_uncalibrated_calibration_slope": unc_s,
        "ume_oof_uncalibrated_brier": float(
            brier_score_loss(selected_oof["y_true"], selected_oof["y_prob_uncalibrated"])
        ),
        "ume_oof_recalibrated_calibration_intercept": rec_i,
        "ume_oof_recalibrated_calibration_slope": rec_s,
        "ume_oof_recalibrated_brier": float(
            brier_score_loss(selected_oof["y_true"], selected_oof["y_prob"])
        ),
    }
    pd.DataFrame([recalibration_row]).to_csv(
        config.output_dir / "logistic_recalibration_from_ume_inner_cv.csv", index=False
    )

    # ------------------------------------------------------------------
    # 3. Operating thresholds from recalibrated UME OOF probabilities.
    # ------------------------------------------------------------------
    candidates, threshold_grid, youden = cv.build_candidate_thresholds(
        selected_oof["y_true"].to_numpy(),
        selected_oof["y_prob"].to_numpy(),
        shared_config,
    )
    candidates.to_csv(config.output_dir / "candidate_thresholds_from_ume_inner_cv.csv", index=False)
    threshold_grid.to_csv(config.output_dir / "ume_inner_cv_threshold_grid_metrics.csv", index=False)
    youden.to_csv(config.output_dir / "ume_inner_cv_youden_thresholds.csv", index=False)

    candidate_metrics = []
    for _, row in candidates.iterrows():
        candidate_metrics.append(
            {
                **row.to_dict(),
                **cv.threshold_metrics(
                    selected_oof["y_true"].to_numpy(),
                    selected_oof["y_prob"].to_numpy(),
                    float(row["threshold"]),
                ),
            }
        )
    pd.DataFrame(candidate_metrics).to_csv(
        config.output_dir / "ume_inner_cv_operating_metrics_at_candidate_thresholds.csv",
        index=False,
    )

    final_epochs = aggregate_epochs(selected_best_epochs, config.final_epoch_aggregation)

    # ------------------------------------------------------------------
    # 4. Fit final preprocessing + model on all UME only.
    # ------------------------------------------------------------------
    final_data = prepare_train_external_data(
        train_df,
        external_df,
        shared_config,
        train_sources,
        external_sources,
    )
    final_model, final_history = cv.train_fixed_epochs(
        selected_hparams,
        final_data,
        device,
        seed=config.random_state + 50000,
        epochs=final_epochs,
    )
    final_history.insert(0, "selected_hpo_trial", selected_trial)
    final_history.insert(1, "final_epochs", final_epochs)
    final_history.to_csv(config.output_dir / "final_full_ume_training_history.csv", index=False)

    # Development predictions from the final full-UME model are saved only as
    # descriptive resubstitution diagnostics. Model selection/calibration use OOF predictions above.
    criterion = nn.BCEWithLogitsLoss()
    _, train_ids, train_y, train_p_uncal = cv.evaluate_model(
        final_model, final_data["train_eval_loader"], criterion, device
    )
    train_p = cv.apply_recalibrator(train_p_uncal, recalibrator)
    train_pred = pd.DataFrame(
        {
            config.id_col: train_ids,
            "y_true": train_y,
            "y_prob_uncalibrated": train_p_uncal,
            "y_prob": train_p,
            "prediction_source": "final_full_ume_model_resubstitution",
        }
    )
    train_pred.to_csv(config.output_dir / "ume_final_model_predictions_descriptive.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Apply frozen model + recalibrator to external cohort once.
    # ------------------------------------------------------------------
    _, external_ids, external_y, external_p_uncal = cv.evaluate_model(
        final_model, final_data["eval_loader"], criterion, device
    )
    external_p = cv.apply_recalibrator(external_p_uncal, recalibrator)
    external_pred = pd.DataFrame(
        {
            config.id_col: external_ids,
            "y_true": external_y,
            "y_prob_uncalibrated": external_p_uncal,
            "y_prob": external_p,
        }
    )

    # Save one prediction column per pre-specified threshold strategy.
    for _, row in candidates.iterrows():
        strategy = str(row["strategy"])
        threshold = float(row["threshold"])
        safe_name = strategy.replace(".", "p").replace(" ", "_")
        external_pred[f"y_pred_{safe_name}"] = (external_p >= threshold).astype(int)
    external_pred.to_csv(
        config.output_dir / f"{config.external_cohort_name}_predictions_intermediate_fusion.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 6. External performance for every threshold strategy.
    # ------------------------------------------------------------------
    point_rows = []
    boot_summaries = []
    confusion_rows = []
    for threshold_idx, row in candidates.reset_index(drop=True).iterrows():
        strategy = str(row["strategy"])
        threshold = float(row["threshold"])
        point = cv.binary_metrics(external_y, external_p, threshold)
        point_rows.append(
            {
                "cohort": config.external_cohort_name,
                "threshold_strategy": strategy,
                **point,
            }
        )
        boot = cv.bootstrap_metrics(
            external_y,
            external_p,
            threshold,
            config.n_bootstraps,
            config.bootstrap_seed + threshold_idx * 1000,
        )
        boot.to_csv(
            config.output_dir / f"bootstrap_{config.external_cohort_name}_{strategy}.csv",
            index=False,
        )
        summary = cv.summarize_bootstrap(point, boot)
        summary.insert(0, "cohort", config.external_cohort_name)
        summary.insert(1, "threshold_strategy", strategy)
        summary.insert(2, "threshold", threshold)
        boot_summaries.append(summary)

        yhat = (external_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(external_y, yhat, labels=[0, 1]).ravel()
        confusion_rows.append(
            {
                "cohort": config.external_cohort_name,
                "threshold_strategy": strategy,
                "threshold": threshold,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    pd.DataFrame(point_rows).to_csv(
        config.output_dir / "external_point_metrics_by_threshold.csv", index=False
    )
    pd.concat(boot_summaries, ignore_index=True).to_csv(
        config.output_dir / "external_bootstrap_metrics_by_threshold_long.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        config.output_dir / "external_confusion_matrices_by_threshold.csv", index=False
    )

    # ------------------------------------------------------------------
    # 7. Serialize complete reproduction package.
    # ------------------------------------------------------------------
    model_state_cpu = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
    package = {
        "model_type": "intermediate_fusion_transformer",
        "fusion_level": "representation_level",
        "selected_hpo_trial": selected_trial,
        "selected_hparams": selected_hparams,
        "final_epochs": final_epochs,
        "model_dims": final_data["dims"],
        "model_state_dict": model_state_cpu,
        "feature_columns": final_data["feature_cols"],
        "concept_columns": list(config.concept_cols),
        "tabular_preprocessor": final_data["tabular_preprocessor"],
        "concept_preprocessor": final_data["concept_preprocessor"],
        "recalibrator": recalibrator,
        "candidate_thresholds": candidates,
        "id_col": config.id_col,
        "label_col": config.label_col,
        "embedding_dimensions": {
            "text": train_sources["text"].dim,
            "time_series": train_sources["time_series"].dim,
            "ct_image": train_sources["ct_image"].dim,
        },
    }
    joblib.dump(package, config.output_dir / "intermediate_fusion_external_validation_model.joblib")

    save_run_config(
        config,
        device,
        train_sources,
        external_sources,
        selected_trial,
        selected_hparams,
        final_epochs,
    )

    LOGGER.info("External validation complete. Results saved to %s", config.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    cv.configure_logging()
    config = parse_args(argv)
    run_external_validation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
