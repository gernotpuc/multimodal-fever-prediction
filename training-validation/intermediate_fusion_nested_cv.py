#!/usr/bin/env python3
"""Nested cross-validation for the intermediate-fusion multimodal fever model.

This script is a publication-oriented refactor of the UME nested-CV notebook
``mm_NC_revisions_ume_nested_cv_recalibrated-2.ipynb``.

Fusion architecture
-------------------
The model performs *intermediate / representation-level fusion*:

1. Clinical-note, body-temperature, and CT modalities are supplied as frozen,
   precomputed embeddings.
2. Structured/tabular variables are transformed using preprocessing fitted on
   the current training fold and encoded by a small MLP.
3. Each active modality is projected into a shared ``fusion_dim`` token space.
4. Modality tokens plus a learned CLS token are fused by a Transformer encoder.
5. The CLS representation is passed to a binary prediction head.

Missing frozen embeddings are zero-filled and masked so that unavailable
modalities are excluded from Transformer attention. CT imaging is treated as a
core modality in the publication reproduction path, matching the actual model
run (the ``USE_CT_IMAGE_MODALITY=False`` value in the exploratory notebook was
an erroneous stale switch).

Validation design
-----------------
For each outer fold:

- all structured-data preprocessing is fitted on training data only;
- hyperparameters are selected by inner cross-validation;
- the selected trial's inner-CV out-of-fold predictions are used to fit
  logistic recalibration;
- operating thresholds are derived from the recalibrated inner-CV predictions;
- the final outer-fold model is trained for the median inner best epoch count;
- the recalibrator and thresholds are applied unchanged to the held-out outer
  fold.

Expected embedding format
-------------------------
Each frozen embedding modality is supplied as:

1. ``*.npy``: a 2-D array ``(n_encounters, embedding_dim)``;
2. ``*.csv``: an index containing an encounter ID and ``embedding_row``.

Example
-------
python intermediate_fusion_nested_cv.py \
    --tabular-csv data/ume_adapted.csv \
    --output-dir results/intermediate_fusion_nested_cv \
    --text-embeddings data/ume_note_embeddings.npy \
    --text-index data/ume_note_embedding_index.csv \
    --time-series-embeddings data/ume_body_temp_chronos2_embeddings.npy \
    --time-series-index data/ume_body_temp_chronos2_index.csv \
    --ct-embeddings data/ume_ct_medimageinsight_embeddings.npy \
    --ct-index data/ume_ct_medimageinsight_index.csv

Raw foundation-model inference is intentionally separated into the repository's
feature-extraction scripts.
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
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("intermediate_fusion_nested_cv")


# The malformed missing comma in the exploratory notebook is corrected here.
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

DEFAULT_HPARAMS = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.10,
    "fusion_dim": 256,
    "hidden_dim": 256,
    "num_layers": 2,
    "num_heads": 4,
}

HPO_SEARCH_SPACE = {
    "learning_rate": [3e-5, 1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "dropout": [0.05, 0.10, 0.20, 0.30],
    "fusion_dim": [128, 256, 384],
    "hidden_dim": [128, 256, 512],
    "num_layers": [1, 2, 3],
    "num_heads": [2, 4, 8],
}


@dataclass(frozen=True)
class Config:
    tabular_csv: Path
    output_dir: Path
    text_embeddings: Path
    text_index: Path
    time_series_embeddings: Path
    time_series_index: Path
    ct_embeddings: Path
    ct_index: Path
    id_col: str = "encounter_id"
    label_col: str = "fever"
    random_state: int = 42
    outer_folds: int = 5
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
    exclude_cols: tuple[str, ...] = tuple(DEFAULT_EXCLUDE_COLUMNS)
    device: str = "auto"
    num_workers: int = 0


@dataclass(frozen=True)
class EmbeddingSource:
    name: str
    lookup: Mapping[str, np.ndarray]
    dim: int
    n_rows: int


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Nested CV for the intermediate-fusion multimodal fever model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tabular-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--text-index", type=Path, required=True)
    parser.add_argument("--time-series-embeddings", type=Path, required=True)
    parser.add_argument("--time-series-index", type=Path, required=True)
    parser.add_argument("--ct-embeddings", type=Path, required=True)
    parser.add_argument("--ct-index", type=Path, required=True)
    parser.add_argument("--id-col", default="encounter_id")
    parser.add_argument("--label-col", default="fever")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
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
    parser.add_argument(
        "--concept-cols",
        nargs="*",
        default=[],
        help="Optional structured concept columns. Empty reproduces the study notebook.",
    )
    parser.add_argument(
        "--exclude-cols",
        nargs="*",
        default=DEFAULT_EXCLUDE_COLUMNS,
        help="Columns excluded from the standard tabular branch.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    return Config(
        tabular_csv=args.tabular_csv,
        output_dir=args.output_dir,
        text_embeddings=args.text_embeddings,
        text_index=args.text_index,
        time_series_embeddings=args.time_series_embeddings,
        time_series_index=args.time_series_index,
        ct_embeddings=args.ct_embeddings,
        ct_index=args.ct_index,
        id_col=args.id_col,
        label_col=args.label_col,
        random_state=args.random_state,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        batch_size=args.batch_size,
        hpo_trials=args.hpo_trials,
        hpo_max_epochs=args.hpo_max_epochs,
        hpo_patience=args.hpo_patience,
        hpo_selection_metric=args.hpo_selection_metric,
        final_epoch_aggregation=args.final_epoch_aggregation,
        n_bootstraps=args.n_bootstraps,
        bootstrap_seed=args.bootstrap_seed,
        threshold_grid_size=args.threshold_grid_size,
        min_sensitivity=args.min_sensitivity,
        min_specificity=args.min_specificity,
        concept_cols=tuple(args.concept_cols),
        exclude_cols=tuple(args.exclude_cols),
        device=args.device,
        num_workers=args.num_workers,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    return torch.device(name)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_encounter_reference(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    for suffix in (".txt", ".dcm"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    lower = text.lower()
    if lower.startswith("encounter_"):
        return f"Encounter/{text.split('_', 1)[1]}"
    if lower.startswith("encounter/"):
        return f"Encounter/{text.split('/', 1)[1]}"
    return f"Encounter/{text}"


def detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ";" if sample.count(";") > sample.count(",") else ","


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=detect_separator(path), encoding="utf-8-sig")


def _embedding_id_column(index_df: pd.DataFrame, preferred: str) -> str:
    for candidate in (preferred, "encounter_id", "encounter_reference"):
        if candidate in index_df.columns:
            return candidate
    raise ValueError(
        "Embedding index must contain one of: "
        f"{preferred!r}, 'encounter_id', 'encounter_reference'."
    )


def load_embedding_source(
    name: str,
    embedding_path: Path,
    index_path: Path,
    preferred_id_col: str,
) -> EmbeddingSource:
    if not embedding_path.exists():
        raise FileNotFoundError(embedding_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    matrix = np.load(embedding_path, allow_pickle=False)
    if matrix.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D embedding array, got {matrix.shape}.")
    matrix = np.asarray(matrix, dtype=np.float32)

    index_df = read_table(index_path)
    if "embedding_row" not in index_df.columns:
        raise ValueError(f"{name}: index is missing 'embedding_row'.")
    id_col = _embedding_id_column(index_df, preferred_id_col)
    index_df = index_df[[id_col, "embedding_row"]].copy()
    index_df[id_col] = index_df[id_col].map(normalize_encounter_reference)
    if index_df[id_col].eq("").any():
        raise ValueError(f"{name}: embedding index contains missing encounter IDs.")
    if index_df[id_col].duplicated().any():
        duplicates = index_df.loc[index_df[id_col].duplicated(keep=False), id_col].unique()[:10]
        raise ValueError(f"{name}: duplicate encounter IDs in index: {duplicates.tolist()}")

    rows = pd.to_numeric(index_df["embedding_row"], errors="raise").astype(int).to_numpy()
    if np.any(rows < 0) or np.any(rows >= matrix.shape[0]):
        raise ValueError(f"{name}: embedding_row contains out-of-range values.")

    lookup = {
        encounter_id: matrix[row]
        for encounter_id, row in zip(index_df[id_col].tolist(), rows, strict=True)
    }
    return EmbeddingSource(name=name, lookup=lookup, dim=int(matrix.shape[1]), n_rows=len(lookup))


def load_development_dataframe(config: Config) -> pd.DataFrame:
    df = read_table(config.tabular_csv)
    for col in (config.id_col, config.label_col):
        if col not in df.columns:
            raise ValueError(f"Required column {col!r} not found in {config.tabular_csv}.")
    df = df.copy()
    df[config.id_col] = df[config.id_col].map(normalize_encounter_reference)
    if df[config.id_col].eq("").any():
        raise ValueError("Tabular data contain missing encounter IDs.")
    if df[config.id_col].duplicated().any():
        duplicates = df.loc[df[config.id_col].duplicated(keep=False), config.id_col].unique()[:10]
        raise ValueError(f"Tabular data contain duplicate encounter IDs: {duplicates.tolist()}")
    df[config.label_col] = pd.to_numeric(df[config.label_col], errors="raise").astype(int)
    labels = set(df[config.label_col].unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError(f"Outcome must be binary with both classes present; found {sorted(labels)}.")
    return df.reset_index(drop=True)


def get_feature_columns(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    concept_cols: Sequence[str],
    exclude_cols: Sequence[str],
) -> list[str]:
    excluded = {id_col, label_col, *concept_cols, *exclude_cols}
    return [col for col in df.columns if col not in excluded]


def ensure_columns_exist(df: pd.DataFrame, required_cols: Sequence[str], df_name: str) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        LOGGER.warning("%s is missing %d structured columns; adding them as NaN.", df_name, len(missing))
        for col in missing:
            out[col] = np.nan
    return out


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No structured feature columns remain after exclusions.")
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_cols, categorical_cols


def to_dense_float32(x: object) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


class MultimodalEncounterDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_col: str,
        label_col: str,
        tabular_matrix: np.ndarray,
        text_source: EmbeddingSource,
        time_series_source: EmbeddingSource,
        ct_source: EmbeddingSource,
        concept_matrix: np.ndarray | None = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.id_col = id_col
        self.label_col = label_col
        self.tabular_matrix = tabular_matrix
        self.concept_matrix = concept_matrix
        self.text_source = text_source
        self.time_series_source = time_series_source
        self.ct_source = ct_source
        if len(self.df) != len(self.tabular_matrix):
            raise ValueError("Tabular matrix and dataframe length differ.")
        if concept_matrix is not None and len(self.df) != len(concept_matrix):
            raise ValueError("Concept matrix and dataframe length differ.")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _lookup(source: EmbeddingSource, encounter_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        value = source.lookup.get(encounter_id)
        if value is None:
            return torch.zeros(source.dim, dtype=torch.float32), torch.tensor(False)
        return torch.tensor(value, dtype=torch.float32), torch.tensor(True)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        encounter_id = normalize_encounter_reference(row[self.id_col])
        text_emb, has_text = self._lookup(self.text_source, encounter_id)
        ts_emb, has_ts = self._lookup(self.time_series_source, encounter_id)
        ct_emb, has_ct = self._lookup(self.ct_source, encounter_id)
        concept_x = (
            torch.tensor(self.concept_matrix[idx], dtype=torch.float32)
            if self.concept_matrix is not None
            else None
        )
        return {
            "encounter_id": encounter_id,
            "label": torch.tensor(float(row[self.label_col]), dtype=torch.float32),
            "tabular_x": torch.tensor(self.tabular_matrix[idx], dtype=torch.float32),
            "has_tabular": torch.tensor(True),
            "text_emb": text_emb,
            "has_text": has_text,
            "time_series_emb": ts_emb,
            "has_time_series": has_ts,
            "ct_image_emb": ct_emb,
            "has_ct_image": has_ct,
            "concept_x": concept_x,
            "has_concepts": torch.tensor(concept_x is not None),
        }


def _stack_optional(batch: list[dict[str, object]], key: str) -> torch.Tensor | None:
    if batch[0][key] is None:
        return None
    return torch.stack([item[key] for item in batch])


def multimodal_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "encounter_id": [item["encounter_id"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "tabular_x": _stack_optional(batch, "tabular_x"),
        "has_tabular": torch.stack([item["has_tabular"] for item in batch]),
        "text_emb": _stack_optional(batch, "text_emb"),
        "has_text": torch.stack([item["has_text"] for item in batch]),
        "time_series_emb": _stack_optional(batch, "time_series_emb"),
        "has_time_series": torch.stack([item["has_time_series"] for item in batch]),
        "ct_image_emb": _stack_optional(batch, "ct_image_emb"),
        "has_ct_image": torch.stack([item["has_ct_image"] for item in batch]),
        "concept_x": _stack_optional(batch, "concept_x"),
        "has_concepts": torch.stack([item["has_concepts"] for item in batch]),
    }


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityTokenProjector(nn.Module):
    def __init__(self, in_dim: int, fusion_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossModalFusionTransformer(nn.Module):
    def __init__(self, fusion_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=4 * fusion_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)

    def forward(self, modality_tokens: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        batch_size = modality_tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, modality_tokens], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=modality_tokens.device)
        full_mask = torch.cat([cls_mask, modality_mask], dim=1)
        fused = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return fused[:, 0, :]


class MultimodalPredictionModel(nn.Module):
    """Representation-level multimodal fusion model used in the study notebook."""

    def __init__(
        self,
        text_emb_dim: int,
        tabular_dim: int,
        time_series_emb_dim: int,
        ct_image_emb_dim: int,
        concept_dim: int | None = None,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_concepts = concept_dim is not None
        self.text_projector = ModalityTokenProjector(text_emb_dim, fusion_dim)
        self.tabular_encoder = MLP(tabular_dim, hidden_dim, fusion_dim, dropout)
        self.time_series_projector = ModalityTokenProjector(time_series_emb_dim, fusion_dim)
        self.ct_image_projector = ModalityTokenProjector(ct_image_emb_dim, fusion_dim)
        if self.use_concepts:
            self.concept_encoder = MLP(concept_dim, hidden_dim, fusion_dim, dropout)
        self.fusion = CrossModalFusionTransformer(fusion_dim, num_heads, num_layers, dropout)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        tabular_x: torch.Tensor,
        time_series_emb: torch.Tensor,
        ct_image_emb: torch.Tensor,
        has_text: torch.Tensor,
        has_tabular: torch.Tensor,
        has_time_series: torch.Tensor,
        has_ct_image: torch.Tensor,
        concept_x: torch.Tensor | None = None,
        has_concepts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Preserve the modality-token ordering used in the source notebook:
        # text -> tabular -> concepts (if enabled) -> time series -> CT image.
        tokens = [
            self.text_projector(text_emb),
            self.tabular_encoder(tabular_x),
        ]
        masks = [has_text.bool(), has_tabular.bool()]
        if self.use_concepts:
            if concept_x is None:
                raise ValueError("Concept branch enabled but concept_x is None.")
            tokens.append(self.concept_encoder(concept_x))
            masks.append(
                has_concepts.bool()
                if has_concepts is not None
                else torch.ones(concept_x.size(0), dtype=torch.bool, device=concept_x.device)
            )
        tokens.extend([
            self.time_series_projector(time_series_emb),
            self.ct_image_projector(ct_image_emb),
        ])
        masks.extend([has_time_series.bool(), has_ct_image.bool()])
        modality_tokens = torch.stack(tokens, dim=1)
        modality_mask = torch.stack(masks, dim=1)
        fused = self.fusion(modality_tokens, modality_mask)
        return self.prediction_head(fused)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out: dict[str, object] = {"encounter_id": batch["encounter_id"]}
    for key, value in batch.items():
        if key == "encounter_id":
            continue
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    out["label"] = out["label"].view(-1, 1)
    return out


def model_forward_from_batch(model: nn.Module, batch: dict[str, object]) -> torch.Tensor:
    return model(
        text_emb=batch["text_emb"],
        tabular_x=batch["tabular_x"],
        time_series_emb=batch["time_series_emb"],
        ct_image_emb=batch["ct_image_emb"],
        has_text=batch["has_text"],
        has_tabular=batch["has_tabular"],
        has_time_series=batch["has_time_series"],
        has_ct_image=batch["has_ct_image"],
        concept_x=batch["concept_x"],
        has_concepts=batch["has_concepts"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        b = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model_forward_from_batch(model, b)
        loss = criterion(logits, b["label"])
        loss.backward()
        optimizer.step()
        size = int(b["label"].size(0))
        total_loss += float(loss.item()) * size
        n_samples += size
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    ns: list[int] = []
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    ids: list[str] = []
    for batch in loader:
        b = move_batch_to_device(batch, device)
        logits = model_forward_from_batch(model, b)
        loss = criterion(logits, b["label"])
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        y = b["label"].detach().cpu().numpy().reshape(-1)
        p_all.append(probs)
        y_all.append(y)
        ids.extend(batch["encounter_id"])
        losses.append(float(loss.item()))
        ns.append(len(y))
    y_true = np.concatenate(y_all).astype(int)
    y_prob = np.concatenate(p_all).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "loss": float(np.average(losses, weights=ns)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    return metrics, np.asarray(ids), y_true, y_prob


def validate_hparams(hparams: Mapping[str, object]) -> bool:
    return int(hparams["fusion_dim"]) % int(hparams["num_heads"]) == 0


def sample_hparams(rng: random.Random) -> dict[str, object]:
    while True:
        hp = {key: rng.choice(values) for key, values in HPO_SEARCH_SPACE.items()}
        if validate_hparams(hp):
            return hp


def build_trial_hparams(n_trials: int, seed: int) -> list[dict[str, object]]:
    trials = [DEFAULT_HPARAMS.copy()]
    if n_trials <= 1:
        return trials
    rng = random.Random(seed)
    seen = {tuple(sorted(DEFAULT_HPARAMS.items()))}
    while len(trials) < n_trials:
        hp = sample_hparams(rng)
        key = tuple(sorted(hp.items()))
        if key not in seen:
            seen.add(key)
            trials.append(hp)
    return trials


def make_stratified_kfold(y: Sequence[int], requested_splits: int, seed: int) -> StratifiedKFold:
    counts = pd.Series(y).value_counts()
    n_splits = min(int(requested_splits), int(counts.min()))
    if n_splits < 2:
        raise ValueError(f"Insufficient minority-class samples for CV: {counts.to_dict()}")
    if n_splits != requested_splits:
        LOGGER.warning("Reducing CV folds from %d to %d because of class counts.", requested_splits, n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def prepare_fold_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: Config,
    text_source: EmbeddingSource,
    ts_source: EmbeddingSource,
    ct_source: EmbeddingSource,
    shuffle_train: bool = True,
) -> dict[str, object]:
    train_df = train_df.copy().reset_index(drop=True)
    eval_df = eval_df.copy().reset_index(drop=True)

    feature_cols = get_feature_columns(
        train_df,
        config.id_col,
        config.label_col,
        config.concept_cols,
        config.exclude_cols,
    )
    eval_df = ensure_columns_exist(eval_df, [*feature_cols, *config.concept_cols], "evaluation fold")

    tab_preprocessor, _, _ = build_preprocessor(train_df, feature_cols)
    x_train_tab = to_dense_float32(tab_preprocessor.fit_transform(train_df[feature_cols]))
    x_eval_tab = to_dense_float32(tab_preprocessor.transform(eval_df[feature_cols]))

    if config.concept_cols:
        missing_concepts = [c for c in config.concept_cols if c not in train_df.columns]
        if missing_concepts:
            raise ValueError(f"Concept columns absent from training data: {missing_concepts}")
        concept_preprocessor, _, _ = build_preprocessor(train_df, list(config.concept_cols))
        x_train_concepts = to_dense_float32(concept_preprocessor.fit_transform(train_df[list(config.concept_cols)]))
        x_eval_concepts = to_dense_float32(concept_preprocessor.transform(eval_df[list(config.concept_cols)]))
        concept_dim = int(x_train_concepts.shape[1])
    else:
        concept_preprocessor = None
        x_train_concepts = None
        x_eval_concepts = None
        concept_dim = None

    train_ds = MultimodalEncounterDataset(
        train_df,
        config.id_col,
        config.label_col,
        x_train_tab,
        text_source,
        ts_source,
        ct_source,
        x_train_concepts,
    )
    eval_ds = MultimodalEncounterDataset(
        eval_df,
        config.id_col,
        config.label_col,
        x_eval_tab,
        text_source,
        ts_source,
        ct_source,
        x_eval_concepts,
    )
    loader_kwargs = dict(collate_fn=multimodal_collate_fn, num_workers=config.num_workers)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=shuffle_train, **loader_kwargs)
    train_eval_loader = DataLoader(train_ds, batch_size=config.batch_size * 2, shuffle=False, **loader_kwargs)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size * 2, shuffle=False, **loader_kwargs)
    return {
        "train_df": train_df,
        "eval_df": eval_df,
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "eval_loader": eval_loader,
        "dims": {
            "text_emb_dim": text_source.dim,
            "tabular_dim": int(x_train_tab.shape[1]),
            "time_series_emb_dim": ts_source.dim,
            "ct_image_emb_dim": ct_source.dim,
            "concept_dim": concept_dim,
        },
        "feature_cols": feature_cols,
        "tabular_preprocessor": tab_preprocessor,
        "concept_preprocessor": concept_preprocessor,
    }


def create_model(hparams: Mapping[str, object], dims: Mapping[str, int | None], device: torch.device) -> MultimodalPredictionModel:
    if not validate_hparams(hparams):
        raise ValueError("fusion_dim must be divisible by num_heads.")
    return MultimodalPredictionModel(
        text_emb_dim=int(dims["text_emb_dim"]),
        tabular_dim=int(dims["tabular_dim"]),
        time_series_emb_dim=int(dims["time_series_emb_dim"]),
        ct_image_emb_dim=int(dims["ct_image_emb_dim"]),
        concept_dim=None if dims["concept_dim"] is None else int(dims["concept_dim"]),
        fusion_dim=int(hparams["fusion_dim"]),
        hidden_dim=int(hparams["hidden_dim"]),
        num_heads=int(hparams["num_heads"]),
        num_layers=int(hparams["num_layers"]),
        dropout=float(hparams["dropout"]),
    ).to(device)


def train_with_early_stopping(
    hparams: Mapping[str, object],
    fold_data: Mapping[str, object],
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
) -> dict[str, object]:
    set_global_seed(seed)
    model = create_model(hparams, fold_data["dims"], device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["learning_rate"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    best_loss = np.inf
    best_epoch = 0
    best_state = None
    best_row = None
    patience_counter = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, fold_data["train_loader"], optimizer, criterion, device)
        val_metrics, _, _, _ = evaluate_model(model, fold_data["eval_loader"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_row = row.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state is None or best_row is None:
        raise RuntimeError("No best state recorded during early stopping.")
    model.load_state_dict(best_state)
    model.to(device)
    _, ids, y_true, y_prob = evaluate_model(model, fold_data["eval_loader"], criterion, device)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_row": best_row,
        "history": pd.DataFrame(history),
        "ids": ids,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def train_fixed_epochs(
    hparams: Mapping[str, object],
    fold_data: Mapping[str, object],
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[MultimodalPredictionModel, pd.DataFrame]:
    set_global_seed(seed)
    model = create_model(hparams, fold_data["dims"], device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["learning_rate"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    rows = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, fold_data["train_loader"], optimizer, criterion, device)
        rows.append({"epoch": epoch, "train_loss": loss})
    return model, pd.DataFrame(rows)


def safe_divide(a: float, b: float) -> float:
    return np.nan if b == 0 else float(a / b)


def logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_logistic_recalibrator(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    result: dict[str, object] = {
        "recalibration_source": "selected_inner_cv_out_of_fold",
        "n_samples": int(len(y_true)),
        "n_events": int(np.sum(y_true == 1)),
        "n_nonevents": int(np.sum(y_true == 0)),
        "method": "identity",
        "intercept": 0.0,
        "slope": 1.0,
        "fit_success": False,
        "fit_message": "",
    }
    if len(np.unique(y_true)) < 2:
        result["fit_message"] = "Skipped: only one outcome class."
        return result
    try:
        try:
            clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
        except TypeError:
            clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=5000)
        clf.fit(logit_np(y_prob).reshape(-1, 1), y_true)
        result.update(
            method="logistic_recalibration",
            intercept=float(clf.intercept_[0]),
            slope=float(clf.coef_[0, 0]),
            fit_success=True,
            fit_message="Fitted logistic recalibration on selected inner-CV OOF predictions.",
        )
    except Exception as exc:  # identity fallback mirrors the notebook's defensive behavior
        result["fit_message"] = f"Recalibration failed; identity used: {type(exc).__name__}: {exc}"
    return result


def apply_recalibrator(y_prob: np.ndarray, recalibrator: Mapping[str, object]) -> np.ndarray:
    p = np.asarray(y_prob, dtype=float)
    if recalibrator.get("method") != "logistic_recalibration":
        return p.copy()
    lp = float(recalibrator["intercept"]) + float(recalibrator["slope"]) * logit_np(p)
    lp = np.clip(lp, -50, 50)
    return 1.0 / (1.0 + np.exp(-lp))


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    try:
        try:
            clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
        except TypeError:
            clf = LogisticRegression(penalty="none", solver="lbfgs", max_iter=5000)
        clf.fit(logit_np(y_prob).reshape(-1, 1), y_true)
        return float(clf.intercept_[0]), float(clf.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def threshold_grid(y_true: np.ndarray, y_prob: np.ndarray, size: int) -> pd.DataFrame:
    return pd.DataFrame(
        [threshold_metrics(y_true, y_prob, t) for t in np.linspace(0.001, 0.999, int(size))]
    )


def build_candidate_thresholds(y_true: np.ndarray, y_prob: np.ndarray, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = threshold_grid(y_true, y_prob, config.threshold_grid_size)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = pd.DataFrame(
        {
            "threshold": thresholds,
            "sensitivity": tpr,
            "specificity": 1.0 - fpr,
            "youden_j": tpr - fpr,
        }
    )
    youden = youden[np.isfinite(youden["threshold"]) & youden["threshold"].between(0, 1)]
    t_youden = float(youden.loc[youden["youden_j"].idxmax(), "threshold"])
    t_f1 = float(grid.loc[grid["f1"].idxmax(), "threshold"])
    eligible_sens = grid[grid["sensitivity"] >= config.min_sensitivity]
    t_sens = (
        float(eligible_sens.loc[eligible_sens["specificity"].idxmax(), "threshold"])
        if len(eligible_sens)
        else np.nan
    )
    eligible_spec = grid[grid["specificity"] >= config.min_specificity]
    t_spec = (
        float(eligible_spec.loc[eligible_spec["sensitivity"].idxmax(), "threshold"])
        if len(eligible_spec)
        else np.nan
    )
    candidates = pd.DataFrame(
        [
            ("default_0.50", 0.50, "fixed threshold of 0.50"),
            ("youden", t_youden, "max sensitivity + specificity - 1"),
            ("max_f1", t_f1, "max F1 on inner-CV OOF predictions"),
            (
                f"sensitivity_at_least_{config.min_sensitivity:.2f}",
                t_sens,
                f"highest specificity with sensitivity >= {config.min_sensitivity:.2f}",
            ),
            (
                f"specificity_at_least_{config.min_specificity:.2f}",
                t_spec,
                f"highest sensitivity with specificity >= {config.min_specificity:.2f}",
            ),
        ],
        columns=["strategy", "threshold", "criterion"],
    )
    candidates["derivation_dataset"] = np.where(
        candidates["strategy"].eq("default_0.50"), "fixed_default", "inner_cv_out_of_fold"
    )
    return candidates[np.isfinite(candidates["threshold"])].reset_index(drop=True), grid, youden.reset_index(drop=True)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    result = threshold_metrics(y_true, y_prob, threshold)
    result.update(
        auroc=float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        auprc=float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        brier=float(brier_score_loss(y_true, y_prob)),
    )
    return result


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    for boot in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        row = binary_metrics(yt, yp, threshold)
        row["bootstrap"] = boot + 1
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bootstrap(point: Mapping[str, object], boot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["auroc", "auprc", "brier", "sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"]:
        vals = pd.to_numeric(boot_df[metric], errors="coerce").dropna().to_numpy()
        rows.append(
            {
                "metric": metric,
                "estimate": point.get(metric, np.nan),
                "ci_lower_95": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_upper_95": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "n_valid_bootstraps": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def modality_availability(df: pd.DataFrame, config: Config, sources: Sequence[EmbeddingSource]) -> pd.DataFrame:
    ids = df[config.id_col].map(normalize_encounter_reference)
    rows = []
    for source in sources:
        present = ids.isin(source.lookup.keys())
        rows.append(
            {
                "modality": source.name,
                "n_available": int(present.sum()),
                "n_total": int(len(ids)),
                "availability_fraction": float(present.mean()),
                "embedding_dim": source.dim,
            }
        )
    rows.append(
        {
            "modality": "tabular",
            "n_available": len(ids),
            "n_total": len(ids),
            "availability_fraction": 1.0,
            "embedding_dim": np.nan,
        }
    )
    return pd.DataFrame(rows)


def hparam_row(hp: Mapping[str, object]) -> dict[str, object]:
    return {key: hp[key] for key in DEFAULT_HPARAMS}


def selection_value(best_row: Mapping[str, object], metric: str) -> float:
    return float(best_row.get(metric, np.nan))


def better(candidate: float, incumbent: float | None, metric: str) -> bool:
    if np.isnan(candidate):
        return False
    if incumbent is None or np.isnan(incumbent):
        return True
    return candidate < incumbent if metric == "val_loss" else candidate > incumbent


def run_nested_cv(
    df: pd.DataFrame,
    config: Config,
    text_source: EmbeddingSource,
    ts_source: EmbeddingSource,
    ct_source: EmbeddingSource,
    device: torch.device,
) -> None:
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    outer_cv = make_stratified_kfold(df[config.label_col], config.outer_folds, config.random_state)

    all_hpo = []
    all_recal = []
    all_candidates = []
    all_inner_preds = []
    all_outer_preds = []
    all_point = []
    all_boot_summary = []
    all_confusions = []
    all_histories = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(df, df[config.label_col]), start=1
    ):
        LOGGER.info("Outer fold %d/%d", outer_fold, outer_cv.n_splits)
        fold_dir = out / f"outer_fold_{outer_fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        outer_train = df.iloc[outer_train_idx].copy().reset_index(drop=True)
        outer_test = df.iloc[outer_test_idx].copy().reset_index(drop=True)
        inner_cv = make_stratified_kfold(
            outer_train[config.label_col], config.inner_folds, config.random_state + outer_fold * 100
        )
        trials = build_trial_hparams(config.hpo_trials, config.random_state + outer_fold * 1000)

        best_mean: float | None = None
        best_trial = 0
        best_hp: dict[str, object] | None = None
        best_preds: pd.DataFrame | None = None
        best_epochs: list[int] | None = None
        hpo_rows = []

        for trial_idx, hp in enumerate(trials, start=1):
            inner_rows = []
            pred_frames = []
            history_frames = []
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                inner_cv.split(outer_train, outer_train[config.label_col]), start=1
            ):
                fold_data = prepare_fold_data(
                    outer_train.iloc[inner_train_idx].reset_index(drop=True),
                    outer_train.iloc[inner_val_idx].reset_index(drop=True),
                    config,
                    text_source,
                    ts_source,
                    ct_source,
                )
                result = train_with_early_stopping(
                    hp,
                    fold_data,
                    device,
                    seed=config.random_state + outer_fold * 10000 + trial_idx * 100 + inner_fold,
                    max_epochs=config.hpo_max_epochs,
                    patience=config.hpo_patience,
                )
                sel = selection_value(result["best_row"], config.hpo_selection_metric)
                inner_rows.append(
                    {
                        "inner_fold": inner_fold,
                        "best_epoch": result["best_epoch"],
                        "selection_value": sel,
                        **result["best_row"],
                    }
                )
                pred_frames.append(
                    pd.DataFrame(
                        {
                            config.id_col: result["ids"],
                            "y_true": result["y_true"],
                            "y_prob": result["y_prob"],
                            "inner_fold": inner_fold,
                        }
                    )
                )
                history = result["history"].copy()
                history.insert(0, "inner_fold", inner_fold)
                history_frames.append(history)
                del result["model"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            inner_df = pd.DataFrame(inner_rows)
            mean_sel = float(inner_df["selection_value"].mean())
            hpo_rows.append(
                {
                    "outer_fold": outer_fold,
                    "trial": trial_idx,
                    "selection_metric": config.hpo_selection_metric,
                    "mean_selection_value": mean_sel,
                    "sd_selection_value": float(inner_df["selection_value"].std(ddof=1)),
                    "mean_best_epoch": float(inner_df["best_epoch"].mean()),
                    "median_best_epoch": float(inner_df["best_epoch"].median()),
                    **hparam_row(hp),
                }
            )
            pd.concat(history_frames, ignore_index=True).to_csv(
                fold_dir / f"trial_{trial_idx:02d}_inner_training_history.csv", index=False
            )
            trial_preds = pd.concat(pred_frames, ignore_index=True)
            trial_preds.to_csv(fold_dir / f"trial_{trial_idx:02d}_inner_oof_predictions.csv", index=False)
            if better(mean_sel, best_mean, config.hpo_selection_metric):
                best_mean = mean_sel
                best_trial = trial_idx
                best_hp = dict(hp)
                best_preds = trial_preds.copy()
                best_epochs = inner_df["best_epoch"].astype(int).tolist()

        if best_hp is None or best_preds is None or best_epochs is None:
            raise RuntimeError("No HPO trial was selected.")
        hpo_df = pd.DataFrame(hpo_rows)
        hpo_df["is_selected"] = hpo_df["trial"].eq(best_trial)
        hpo_df.to_csv(fold_dir / "hpo_results.csv", index=False)
        all_hpo.append(hpo_df)

        best_preds.insert(0, "outer_fold", outer_fold)
        best_preds.insert(1, "selected_hpo_trial", best_trial)
        best_preds["y_prob_uncalibrated"] = best_preds["y_prob"].astype(float)
        recalibrator = fit_logistic_recalibrator(
            best_preds["y_true"].to_numpy(), best_preds["y_prob_uncalibrated"].to_numpy()
        )
        best_preds["y_prob"] = apply_recalibrator(best_preds["y_prob_uncalibrated"].to_numpy(), recalibrator)
        unc_i, unc_s = calibration_intercept_slope(
            best_preds["y_true"].to_numpy(), best_preds["y_prob_uncalibrated"].to_numpy()
        )
        rec_i, rec_s = calibration_intercept_slope(
            best_preds["y_true"].to_numpy(), best_preds["y_prob"].to_numpy()
        )
        recal_row = {
            "outer_fold": outer_fold,
            "selected_hpo_trial": best_trial,
            **recalibrator,
            "inner_oof_uncalibrated_calibration_intercept": unc_i,
            "inner_oof_uncalibrated_calibration_slope": unc_s,
            "inner_oof_uncalibrated_brier": float(
                brier_score_loss(best_preds["y_true"], best_preds["y_prob_uncalibrated"])
            ),
            "inner_oof_recalibrated_calibration_intercept": rec_i,
            "inner_oof_recalibrated_calibration_slope": rec_s,
            "inner_oof_recalibrated_brier": float(brier_score_loss(best_preds["y_true"], best_preds["y_prob"])),
        }
        pd.DataFrame([recal_row]).to_csv(fold_dir / "logistic_recalibration_from_inner_cv.csv", index=False)
        all_recal.append(recal_row)
        best_preds.to_csv(fold_dir / "selected_trial_inner_oof_predictions.csv", index=False)
        all_inner_preds.append(best_preds)

        candidates, grid, youden = build_candidate_thresholds(
            best_preds["y_true"].to_numpy(), best_preds["y_prob"].to_numpy(), config
        )
        candidates.insert(0, "outer_fold", outer_fold)
        candidate_metrics = []
        for _, row in candidates.iterrows():
            candidate_metrics.append(
                {**row.to_dict(), **threshold_metrics(best_preds["y_true"].to_numpy(), best_preds["y_prob"].to_numpy(), float(row["threshold"]))}
            )
        pd.DataFrame(candidate_metrics).to_csv(fold_dir / "inner_cv_operating_metrics_at_candidate_thresholds.csv", index=False)
        candidates.to_csv(fold_dir / "candidate_thresholds_from_inner_cv.csv", index=False)
        grid.to_csv(fold_dir / "inner_cv_threshold_grid_metrics.csv", index=False)
        youden.to_csv(fold_dir / "inner_cv_youden_thresholds.csv", index=False)
        all_candidates.append(candidates)

        if config.final_epoch_aggregation == "mean_inner_best_epoch":
            final_epochs = max(1, int(round(np.mean(best_epochs))))
        else:
            final_epochs = max(1, int(round(np.median(best_epochs))))

        outer_data = prepare_fold_data(
            outer_train,
            outer_test,
            config,
            text_source,
            ts_source,
            ct_source,
        )
        final_model, history = train_fixed_epochs(
            best_hp,
            outer_data,
            device,
            seed=config.random_state + outer_fold * 50000,
            epochs=final_epochs,
        )
        history.insert(0, "outer_fold", outer_fold)
        history.insert(1, "selected_hpo_trial", best_trial)
        history.insert(2, "final_epochs", final_epochs)
        history.to_csv(fold_dir / "final_training_history.csv", index=False)
        all_histories.append(history)

        criterion = nn.BCEWithLogitsLoss()
        _, ids, y_true, p_uncal = evaluate_model(final_model, outer_data["eval_loader"], criterion, device)
        p = apply_recalibrator(p_uncal, recalibrator)
        base_pred = pd.DataFrame(
            {
                config.id_col: ids,
                "y_true": y_true,
                "y_prob_uncalibrated": p_uncal,
                "y_prob": p,
                "outer_fold": outer_fold,
                "selected_hpo_trial": best_trial,
                "final_epochs": final_epochs,
            }
        )
        base_pred.to_csv(fold_dir / "heldout_predictions.csv", index=False)
        all_outer_preds.append(base_pred)

        for threshold_idx, row in candidates.reset_index(drop=True).iterrows():
            strategy = str(row["strategy"])
            threshold = float(row["threshold"])
            point = binary_metrics(y_true, p, threshold)
            point_row = {"outer_fold": outer_fold, "threshold_strategy": strategy, **point}
            all_point.append(point_row)
            boot = bootstrap_metrics(
                y_true,
                p,
                threshold,
                config.n_bootstraps,
                config.bootstrap_seed + outer_fold * 10000 + threshold_idx * 1000,
            )
            summary = summarize_bootstrap(point, boot)
            summary.insert(0, "outer_fold", outer_fold)
            summary.insert(1, "threshold_strategy", strategy)
            summary.insert(2, "threshold", threshold)
            all_boot_summary.append(summary)
            y_pred = (p >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            all_confusions.append(
                {
                    "outer_fold": outer_fold,
                    "threshold_strategy": strategy,
                    "threshold": threshold,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )

        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.concat(all_hpo, ignore_index=True).to_csv(out / "nested_cv_hpo_results_all_outer_folds.csv", index=False)
    pd.DataFrame(all_recal).to_csv(out / "nested_cv_logistic_recalibration_by_outer_fold.csv", index=False)
    candidate_all = pd.concat(all_candidates, ignore_index=True)
    candidate_all.to_csv(out / "nested_cv_candidate_thresholds_all_outer_folds.csv", index=False)
    pd.concat(all_inner_preds, ignore_index=True).to_csv(out / "nested_cv_selected_inner_oof_predictions_all_outer_folds.csv", index=False)
    pooled_preds = pd.concat(all_outer_preds, ignore_index=True)
    pooled_preds.to_csv(out / "nested_cv_outer_fold_predictions_base.csv", index=False)
    point_df = pd.DataFrame(all_point)
    point_df.to_csv(out / "nested_cv_outer_fold_point_metrics_by_threshold.csv", index=False)
    metric_cols = [
        "auroc", "auprc", "brier", "sensitivity", "specificity",
        "ppv", "npv", "f1", "accuracy",
    ]
    fold_summary_rows = []
    for strategy, group in point_df.groupby("threshold_strategy", sort=False):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            fold_summary_rows.append({
                "threshold_strategy": strategy,
                "metric": metric,
                "mean_across_outer_folds": float(values.mean()) if len(values) else np.nan,
                "sd_across_outer_folds": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "n_outer_folds": int(len(values)),
            })
    pd.DataFrame(fold_summary_rows).to_csv(
        out / "nested_cv_outer_fold_metric_mean_sd_by_threshold.csv", index=False
    )
    pd.concat(all_boot_summary, ignore_index=True).to_csv(
        out / "nested_cv_outer_fold_bootstrap_metrics_by_threshold_long.csv", index=False
    )
    pd.DataFrame(all_confusions).to_csv(
        out / "nested_cv_outer_fold_confusion_matrices_by_threshold_long.csv", index=False
    )
    pd.concat(all_histories, ignore_index=True).to_csv(out / "nested_cv_final_outer_training_histories.csv", index=False)

    # Pooled outer-fold estimates by threshold strategy use each patient's fold-specific threshold.
    pooled_rows = []
    for strategy in candidate_all["strategy"].unique():
        y_parts = []
        p_parts = []
        pred_parts = []
        for outer_fold in sorted(pooled_preds["outer_fold"].unique()):
            fold_pred = pooled_preds[pooled_preds["outer_fold"] == outer_fold]
            threshold_rows = candidate_all
            threshold_rows = threshold_rows[
                (threshold_rows["outer_fold"] == outer_fold) & (threshold_rows["strategy"] == strategy)
            ]
            if threshold_rows.empty:
                continue
            threshold = float(threshold_rows.iloc[0]["threshold"])
            y_parts.append(fold_pred["y_true"].to_numpy())
            p_parts.append(fold_pred["y_prob"].to_numpy())
            pred_parts.append((fold_pred["y_prob"].to_numpy() >= threshold).astype(int))
        if not y_parts:
            continue
        y_all = np.concatenate(y_parts)
        p_all = np.concatenate(p_parts)
        yhat_all = np.concatenate(pred_parts)
        tn, fp, fn, tp = confusion_matrix(y_all, yhat_all, labels=[0, 1]).ravel()
        pooled_rows.append(
            {
                "threshold_strategy": strategy,
                "n": len(y_all),
                "auroc": float(roc_auc_score(y_all, p_all)),
                "auprc": float(average_precision_score(y_all, p_all)),
                "brier": float(brier_score_loss(y_all, p_all)),
                "sensitivity": safe_divide(tp, tp + fn),
                "specificity": safe_divide(tn, tn + fp),
                "ppv": safe_divide(tp, tp + fp),
                "npv": safe_divide(tn, tn + fn),
                "f1": float(f1_score(y_all, yhat_all, zero_division=0)),
                "accuracy": float(accuracy_score(y_all, yhat_all)),
            }
        )
    pd.DataFrame(pooled_rows).to_csv(out / "nested_cv_pooled_outer_performance_by_threshold.csv", index=False)


def save_run_config(config: Config, device: torch.device, sources: Sequence[EmbeddingSource]) -> None:
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
            "ct_image_modality_enabled": True,
            "default_hparams": DEFAULT_HPARAMS,
            "hpo_search_space": HPO_SEARCH_SPACE,
            "device_resolved": str(device),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "embedding_sources": [
                {"name": s.name, "dim": s.dim, "n_rows": s.n_rows} for s in sources
            ],
        }
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    config = parse_args(argv)
    device = resolve_device(config.device)
    set_global_seed(config.random_state)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Loading UME tabular data: %s", config.tabular_csv)
    df = load_development_dataframe(config)
    text_source = load_embedding_source("text", config.text_embeddings, config.text_index, config.id_col)
    ts_source = load_embedding_source(
        "time_series", config.time_series_embeddings, config.time_series_index, config.id_col
    )
    ct_source = load_embedding_source("ct_image", config.ct_embeddings, config.ct_index, config.id_col)
    sources = [text_source, ts_source, ct_source]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    modality_availability(df, config, sources).to_csv(
        config.output_dir / "modality_availability.csv", index=False
    )
    save_run_config(config, device, sources)
    LOGGER.info(
        "Active modalities: tabular, text, time_series, ct_image%s",
        ", concepts" if config.concept_cols else "",
    )
    LOGGER.info(
        "Embedding dimensions | text=%d | time_series=%d | ct_image=%d",
        text_source.dim,
        ts_source.dim,
        ct_source.dim,
    )
    run_nested_cv(df, config, text_source, ts_source, ct_source, device)
    LOGGER.info("Finished. Results written to %s", config.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
