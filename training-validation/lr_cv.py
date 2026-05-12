"""
Repeated nested cross-validation pipeline for logistic regression.

This script is designed for reproducible GitHub use:
- reads features/target from CSV or Parquet
- performs leakage-safe nested CV with Optuna tuning
- optionally repeats the full nested CV with different seeds
- selects thresholds using outer-train out-of-fold predictions only
- evaluates default, MCC-optimal, and Youden thresholds
- saves per-case predictions, pooled predictions, per-repeat metrics, pooled metrics,
  variance estimates, FP/FN tables, thresholds, best params, and plots

Example:
    python scripts/train_nested_logistic_regression.py \
        --data-path artifacts/preprocessed/data_ml.parquet \
        --target-col fever \
        --id-cols encounter_id subject_reference \
        --output-dir artifacts/models/logistic_regression_repeated_nested_cv \
        --outer-splits 5 \
        --inner-splits 5 \
        --n-repeats 5 \
        --n-trials 50

Optional:
    --feature-cols age elixhauser_score time_lag_1_bt_max ...

If --feature-cols is omitted, all columns except target/id columns are used as features.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NestedCVConfig:
    outer_splits: int = 5
    inner_splits: int = 5
    n_repeats: int = 1
    n_trials: int = 50
    seed: int = 42
    bootstrap_iterations: int = 2000
    optimize_metric: str = "auc"
    class_weight: Optional[str] = "balanced"
    max_iter: int = 5000


@dataclass(frozen=True)
class NestedCVResult:
    indices: np.ndarray
    y_true: np.ndarray
    probabilities: np.ndarray
    pred_default: np.ndarray
    pred_mcc: np.ndarray
    pred_youden: np.ndarray
    repeat: np.ndarray
    fold: np.ndarray
    fold_thresholds: List[Dict[str, float]]
    fold_best_params: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# I/O and preprocessing
# -----------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_table(path: Path, *, sep: str = ";") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=sep)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use CSV or Parquet.")


def infer_feature_columns(
    data: pd.DataFrame,
    *,
    target_col: str,
    id_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    if feature_cols:
        missing = [col for col in feature_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from data: {missing}")
        return list(feature_cols)

    excluded = set(id_cols) | {target_col}
    candidate_cols = [col for col in data.columns if col not in excluded]
    numeric_cols = data[candidate_cols].select_dtypes(include=["number", "bool"]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric feature columns found. Pass --feature-cols explicitly.")
    return numeric_cols


def prepare_xy(
    data: pd.DataFrame,
    *,
    target_col: str,
    id_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    features = infer_feature_columns(data, target_col=target_col, id_cols=id_cols, feature_cols=feature_cols)
    required = list(features) + [target_col]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = data.copy()
    working[target_col] = working[target_col].astype(int)

    X = working[features].copy()
    y = working[target_col].copy()

    # Logistic regression cannot handle missing values. Keep this conservative and explicit.
    before = len(X)
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]
    metadata_cols = [col for col in id_cols if col in working.columns]
    metadata = working.loc[valid_rows, metadata_cols].copy()

    dropped = before - len(X)
    if dropped:
        LOGGER.warning("Dropped %d rows with missing feature/target values.", dropped)

    if y.nunique() < 2:
        raise ValueError(f"Target '{target_col}' must contain at least two classes.")

    return X, y, metadata, features


# -----------------------------------------------------------------------------
# Metrics and confidence intervals
# -----------------------------------------------------------------------------

def safe_roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-9)
    sensitivity = tp / (tp + fn + 1e-9)
    npv = tn / (tn + fn + 1e-9)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": safe_roc_auc_score(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    metrics = {key: [] for key in [
        "accuracy", "precision", "recall", "specificity", "sensitivity",
        "npv", "f1", "mcc", "auc", "auprc", "tn", "fp", "fn", "tp"
    ]}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        values = compute_binary_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for key, value in values.items():
            metrics[key].append(value)

    return metrics


def ci(values: Sequence[float]) -> Tuple[float, float, float]:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.mean(values_array)),
        float(np.percentile(values_array, 2.5)),
        float(np.percentile(values_array, 97.5)),
    )


def summarize_bootstrap_metrics(boot: Dict[str, List[float]]) -> pd.DataFrame:
    rows = []
    for metric, values in boot.items():
        mean, low, high = ci(values)
        rows.append({"metric": metric, "mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def logits_from_probabilities(y_prob: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    return np.log(y_prob + eps) - np.log(1 - y_prob + eps)


def calibration_in_the_large(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    logits = logits_from_probabilities(y_prob)
    model = LogisticRegression(solver="lbfgs")
    model.fit(logits.reshape(-1, 1), y_true)
    return float(model.intercept_[0])


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    logits = logits_from_probabilities(y_prob)
    model = LogisticRegression(solver="lbfgs")
    model.fit(logits.reshape(-1, 1), y_true)
    return float(model.coef_[0][0])


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "citl": calibration_in_the_large(y_true, y_prob),
        "slope": calibration_slope(y_true, y_prob),
    }


def bootstrap_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    results = {"brier": [], "citl": [], "slope": []}

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, pr = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        values = compute_calibration_metrics(yt, pr)
        for key, value in values.items():
            results[key].append(value)

    return results


# -----------------------------------------------------------------------------
# Thresholds
# -----------------------------------------------------------------------------

def optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return {"f1": 0.5, "mcc": 0.5, "youden": 0.5}

    f1_values = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_threshold = float(thresholds[np.argmax(f1_values[:-1])])

    mcc_values = [matthews_corrcoef(y_true, (y_prob >= threshold).astype(int)) for threshold in thresholds]
    mcc_threshold = float(thresholds[int(np.argmax(mcc_values))])

    youden_values = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        youden_values.append(sensitivity + specificity - 1)
    youden_threshold = float(thresholds[int(np.argmax(youden_values))])

    return {"f1": f1_threshold, "mcc": mcc_threshold, "youden": youden_threshold}


# -----------------------------------------------------------------------------
# Model building and Optuna objective
# -----------------------------------------------------------------------------

def build_lr_pipeline(
    params: Dict[str, Any],
    *,
    class_weight: Optional[str] = "balanced",
    max_iter: int = 5000,
) -> Pipeline:
    penalty = params["penalty"]
    solver = "liblinear" if penalty in {"l1", "l2"} else "saga"
    l1_ratio = params.get("l1_ratio") if penalty == "elasticnet" else None

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty=penalty,
                    C=params["C"],
                    solver=solver,
                    l1_ratio=l1_ratio,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    n_jobs=-1 if solver == "saga" else None,
                    random_state=42 if solver == "saga" else None,
                ),
            ),
        ]
    )


def make_objective_lr(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    inner_splits: int,
    seed: int,
    class_weight: Optional[str],
    max_iter: int,
    optimize_metric: str,
):
    def objective(trial: optuna.Trial) -> float:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        params: Dict[str, Any] = {
            "penalty": penalty,
            "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        scores = []

        for train_idx, valid_idx in cv.split(X, y):
            model = build_lr_pipeline(params, class_weight=class_weight, max_iter=max_iter)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X.iloc[valid_idx])[:, 1]

            if optimize_metric == "auprc":
                score = safe_auprc(y.iloc[valid_idx].to_numpy(), prob)
            else:
                score = safe_roc_auc_score(y.iloc[valid_idx].to_numpy(), prob)
            scores.append(score)

        return float(np.nanmean(scores))

    return objective


def oof_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    *,
    n_splits: int,
    seed: int,
    class_weight: Optional[str],
    max_iter: int,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)

    for train_idx, valid_idx in cv.split(X, y):
        model = build_lr_pipeline(params, class_weight=class_weight, max_iter=max_iter)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]

    return oof


# -----------------------------------------------------------------------------
# Nested CV
# -----------------------------------------------------------------------------

def nested_cv_lr(X: pd.DataFrame, y: pd.Series, config: NestedCVConfig) -> NestedCVResult:
    """Run repeated nested CV and pool all outer-test predictions across repeats."""
    all_y_true: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    all_pred_default: List[np.ndarray] = []
    all_pred_mcc: List[np.ndarray] = []
    all_pred_youden: List[np.ndarray] = []
    all_repeats: List[np.ndarray] = []
    all_folds: List[np.ndarray] = []
    fold_thresholds: List[Dict[str, float]] = []
    fold_best_params: List[Dict[str, Any]] = []

    for repeat in range(1, config.n_repeats + 1):
        repeat_seed = config.seed + repeat - 1
        outer_cv = StratifiedKFold(n_splits=config.outer_splits, shuffle=True, random_state=repeat_seed)
        LOGGER.info("Starting repeat %d/%d with seed=%d", repeat, config.n_repeats, repeat_seed)

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            LOGGER.info(
                "Repeat %d/%d | outer fold %d/%d: tuning on outer-training set",
                repeat,
                config.n_repeats,
                fold,
                config.outer_splits,
            )
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            fold_seed = repeat_seed * 1000 + fold
            sampler = optuna.samplers.TPESampler(seed=fold_seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                make_objective_lr(
                    X_train,
                    y_train,
                    inner_splits=config.inner_splits,
                    seed=fold_seed,
                    class_weight=config.class_weight,
                    max_iter=config.max_iter,
                    optimize_metric=config.optimize_metric,
                ),
                n_trials=config.n_trials,
                show_progress_bar=False,
            )

            best_params = study.best_params
            fold_best_params.append({"repeat": repeat, "fold": fold, **best_params})
            LOGGER.info("Repeat %d fold %d best params: %s", repeat, fold, best_params)

            model = build_lr_pipeline(best_params, class_weight=config.class_weight, max_iter=config.max_iter)
            model.fit(X_train, y_train)
            test_prob = model.predict_proba(X_test)[:, 1]

            train_oof_prob = oof_predict_proba(
                X_train,
                y_train,
                best_params,
                n_splits=config.inner_splits,
                seed=fold_seed,
                class_weight=config.class_weight,
                max_iter=config.max_iter,
            )
            thresholds = optimal_thresholds(y_train.to_numpy(), train_oof_prob)
            thresholds_for_fold = {
                "repeat": float(repeat),
                "fold": float(fold),
                "default": 0.5,
                "mcc": thresholds["mcc"],
                "youden": thresholds["youden"],
                "f1": thresholds["f1"],
            }
            fold_thresholds.append(thresholds_for_fold)
            LOGGER.info("Repeat %d fold %d thresholds: %s", repeat, fold, thresholds_for_fold)

            all_y_true.append(y_test.to_numpy())
            all_probabilities.append(test_prob)
            all_indices.append(X_test.index.to_numpy())
            all_pred_default.append((test_prob >= 0.5).astype(int))
            all_pred_mcc.append((test_prob >= thresholds["mcc"]).astype(int))
            all_pred_youden.append((test_prob >= thresholds["youden"]).astype(int))
            all_repeats.append(np.full(len(test_idx), repeat, dtype=int))
            all_folds.append(np.full(len(test_idx), fold, dtype=int))

    return NestedCVResult(
        indices=np.concatenate(all_indices),
        y_true=np.concatenate(all_y_true),
        probabilities=np.concatenate(all_probabilities),
        pred_default=np.concatenate(all_pred_default),
        pred_mcc=np.concatenate(all_pred_mcc),
        pred_youden=np.concatenate(all_pred_youden),
        repeat=np.concatenate(all_repeats),
        fold=np.concatenate(all_folds),
        fold_thresholds=fold_thresholds,
        fold_best_params=fold_best_params,
    )


# -----------------------------------------------------------------------------
# Outputs and plots
# -----------------------------------------------------------------------------

def make_prediction_table(result: NestedCVResult, metadata: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "row_index": result.indices,
            "repeat": result.repeat,
            "fold": result.fold,
            "y_true": result.y_true,
            "p": result.probabilities,
            "pred_0_5": result.pred_default,
            "pred_mcc": result.pred_mcc,
            "pred_youden": result.pred_youden,
        }
    )

    if not metadata.empty:
        meta = metadata.copy()
        meta = meta.reset_index().rename(columns={"index": "row_index"})
        df = df.merge(meta, on="row_index", how="left")

    for pred_col in ["pred_0_5", "pred_mcc", "pred_youden"]:
        df[f"FP_{pred_col}"] = df["y_true"].eq(0) & df[pred_col].eq(1)
        df[f"FN_{pred_col}"] = df["y_true"].eq(1) & df[pred_col].eq(0)

    return df


def make_case_level_pooled_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repeated predictions to one row per original sample.

    For repeated nested CV, each original sample appears once per repeat. The pooled
    probability is the mean of its repeat-level outer-test probabilities. Binary
    predictions are recomputed using threshold 0.5 on the mean probability; MCC/Youden
    repeat-specific binary predictions are summarized by majority vote.
    """
    grouped = predictions.groupby("row_index", as_index=False)
    pooled = grouped.agg(
        y_true=("y_true", "first"),
        p_mean=("p", "mean"),
        p_std=("p", "std"),
        p_min=("p", "min"),
        p_max=("p", "max"),
        n_repeats_observed=("repeat", "nunique"),
    )

    metadata_cols = [
        col for col in predictions.columns
        if col not in {
            "repeat", "fold", "y_true", "p", "pred_0_5", "pred_mcc", "pred_youden",
            "FP_pred_0_5", "FN_pred_0_5", "FP_pred_mcc", "FN_pred_mcc", "FP_pred_youden", "FN_pred_youden"
        }
        and col != "row_index"
    ]
    if metadata_cols:
        meta = predictions[["row_index", *metadata_cols]].drop_duplicates(subset=["row_index"])
        pooled = pooled.merge(meta, on="row_index", how="left")

    vote_summary = predictions.groupby("row_index", as_index=False).agg(
        pred_0_5_vote=("pred_0_5", "mean"),
        pred_mcc_vote=("pred_mcc", "mean"),
        pred_youden_vote=("pred_youden", "mean"),
    )
    pooled = pooled.merge(vote_summary, on="row_index", how="left")

    pooled["pred_0_5"] = (pooled["p_mean"] >= 0.5).astype(int)
    pooled["pred_mcc_majority"] = (pooled["pred_mcc_vote"] >= 0.5).astype(int)
    pooled["pred_youden_majority"] = (pooled["pred_youden_vote"] >= 0.5).astype(int)

    for pred_col in ["pred_0_5", "pred_mcc_majority", "pred_youden_majority"]:
        pooled[f"FP_{pred_col}"] = pooled["y_true"].eq(0) & pooled[pred_col].eq(1)
        pooled[f"FN_{pred_col}"] = pooled["y_true"].eq(1) & pooled[pred_col].eq(0)

    return pooled.sort_values("row_index")


def get_fp_fn(df: pd.DataFrame, pred_col: str, prob_col: str = "p") -> Tuple[pd.DataFrame, pd.DataFrame]:
    fp = df[(df["y_true"] == 0) & (df[pred_col] == 1)].copy().sort_values(prob_col, ascending=False)
    fn = df[(df["y_true"] == 1) & (df[pred_col] == 0)].copy().sort_values(prob_col, ascending=True)
    return fp, fn


def evaluate_all_thresholds(
    result: NestedCVResult,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap metrics across all repeat-fold outer-test predictions."""
    threshold_map = {
        "default_0_5": result.pred_default,
        "mcc_optimal": result.pred_mcc,
        "youden": result.pred_youden,
    }
    rows = []
    for threshold_name, predictions in threshold_map.items():
        boot = bootstrap_metrics(result.y_true, predictions, result.probabilities, n_boot=n_boot, seed=seed)
        summary = summarize_bootstrap_metrics(boot)
        summary.insert(0, "threshold_strategy", threshold_name)
        summary.insert(0, "aggregation", "pooled_repeat_fold_predictions")
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def evaluate_per_repeat(result: NestedCVResult) -> pd.DataFrame:
    """Compute point-estimate metrics separately for each repeat."""
    rows = []
    threshold_map = {
        "default_0_5": result.pred_default,
        "mcc_optimal": result.pred_mcc,
        "youden": result.pred_youden,
    }
    for repeat in sorted(np.unique(result.repeat)):
        mask = result.repeat == repeat
        for threshold_name, predictions in threshold_map.items():
            metrics = compute_binary_metrics(result.y_true[mask], predictions[mask], result.probabilities[mask])
            rows.append({"repeat": int(repeat), "threshold_strategy": threshold_name, **metrics})
    return pd.DataFrame(rows)


def summarize_repeat_variance(per_repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize variability of metrics across repeated nested-CV runs."""
    metric_cols = [
        col for col in per_repeat_metrics.columns
        if col not in {"repeat", "threshold_strategy"}
    ]
    rows = []
    for strategy, group in per_repeat_metrics.groupby("threshold_strategy"):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "threshold_strategy": strategy,
                    "metric": metric,
                    "mean_across_repeats": float(values.mean()),
                    "std_across_repeats": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min_across_repeats": float(values.min()),
                    "max_across_repeats": float(values.max()),
                    "n_repeats": int(values.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def summarize_calibration(result: NestedCVResult, *, n_boot: int, seed: int) -> pd.DataFrame:
    point = compute_calibration_metrics(result.y_true, result.probabilities)
    boot = bootstrap_calibration(result.y_true, result.probabilities, n_boot=n_boot, seed=seed)

    rows = []
    for metric, point_value in point.items():
        mean, low, high = ci(boot[metric])
        rows.append(
            {
                "metric": metric,
                "point_estimate": point_value,
                "bootstrap_mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = safe_roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc_value = safe_auprc(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AUPRC={auprc_value:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, *, n_bins: int = 10) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed outcome frequency")
    plt.title("Calibration Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    config: NestedCVConfig,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tune on full data and fit a final model for deployment/reuse."""
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective_lr(
            X,
            y,
            inner_splits=config.inner_splits,
            seed=config.seed,
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            optimize_metric=config.optimize_metric,
        ),
        n_trials=config.n_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    model = build_lr_pipeline(best_params, class_weight=config.class_weight, max_iter=config.max_iter)
    model.fit(X, y)
    return model, best_params


def save_run_outputs(
    output_dir: Path,
    *,
    result: NestedCVResult,
    predictions: pd.DataFrame,
    case_level_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    per_repeat_metrics: pd.DataFrame,
    repeat_variance: pd.DataFrame,
    calibration: pd.DataFrame,
    features: Sequence[str],
    config: NestedCVConfig,
    save_final_model: bool,
    final_model: Optional[Pipeline] = None,
    final_model_params: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_dir / "nested_cv_predictions_all_repeats.csv", index=False)
    case_level_predictions.to_csv(output_dir / "nested_cv_predictions_case_level_pooled.csv", index=False)
    metrics.to_csv(output_dir / "nested_cv_metrics_pooled.csv", index=False)
    per_repeat_metrics.to_csv(output_dir / "nested_cv_metrics_per_repeat.csv", index=False)
    repeat_variance.to_csv(output_dir / "nested_cv_metric_variance_across_repeats.csv", index=False)
    calibration.to_csv(output_dir / "nested_cv_calibration_pooled.csv", index=False)
    pd.DataFrame(result.fold_thresholds).to_csv(output_dir / "fold_thresholds.csv", index=False)
    pd.DataFrame(result.fold_best_params).to_csv(output_dir / "fold_best_params.csv", index=False)

    for pred_col, name in [("pred_0_5", "default_0_5"), ("pred_mcc", "mcc_optimal"), ("pred_youden", "youden")]:
        fp, fn = get_fp_fn(predictions, pred_col)
        fp.to_csv(output_dir / f"false_positives_all_repeats_{name}.csv", index=False)
        fn.to_csv(output_dir / f"false_negatives_all_repeats_{name}.csv", index=False)

    for pred_col, name in [
        ("pred_0_5", "default_0_5"),
        ("pred_mcc_majority", "mcc_majority_vote"),
        ("pred_youden_majority", "youden_majority_vote"),
    ]:
        fp, fn = get_fp_fn(case_level_predictions.rename(columns={"p_mean": "p"}), pred_col)
        fp.to_csv(output_dir / f"false_positives_case_level_{name}.csv", index=False)
        fn.to_csv(output_dir / f"false_negatives_case_level_{name}.csv", index=False)

    plot_roc_curve(result.y_true, result.probabilities, output_dir / "roc_curve_pooled_repeats.png")
    plot_pr_curve(result.y_true, result.probabilities, output_dir / "precision_recall_curve_pooled_repeats.png")
    plot_calibration_curve(result.y_true, result.probabilities, output_dir / "calibration_curve_pooled_repeats.png")

    run_metadata = {
        "config": asdict(config),
        "features": list(features),
        "n_outer_test_predictions": int(len(result.y_true)),
        "n_unique_samples": int(len(np.unique(result.indices))),
        "positive_rate_outer_predictions": float(np.mean(result.y_true)),
        "final_model_params": final_model_params,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    if save_final_model and final_model is not None:
        joblib.dump(final_model, output_dir / "final_logistic_regression_model.joblib")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested CV logistic regression training/evaluation pipeline.")

    parser.add_argument("--data-path", required=True, help="Input CSV or Parquet containing features and target.")
    parser.add_argument("--csv-sep", default=";", help="CSV separator if input is CSV.")
    parser.add_argument("--target-col", default="fever", help="Binary target column.")
    parser.add_argument("--id-cols", nargs="*", default=["encounter_id", "subject_reference"], help="Metadata columns to carry into outputs.")
    parser.add_argument("--feature-cols", nargs="*", default=None, help="Optional explicit feature columns.")
    parser.add_argument("--output-dir", default="artifacts/logistic_regression_nested_cv", help="Output directory.")

    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1, help="Number of times to repeat the full nested CV with different seeds.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--optimize-metric", choices=["auc", "auprc"], default="auc")
    parser.add_argument("--no-class-weight-balanced", action="store_true", help="Disable class_weight='balanced'.")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--save-final-model", action="store_true", help="Tune and fit a final model on all data, then save it.")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data = read_table(Path(args.data_path), sep=args.csv_sep)
    X, y, metadata, features = prepare_xy(
        data,
        target_col=args.target_col,
        id_cols=args.id_cols,
        feature_cols=args.feature_cols,
    )

    config = NestedCVConfig(
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_repeats=args.n_repeats,
        n_trials=args.n_trials,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
        optimize_metric=args.optimize_metric,
        class_weight=None if args.no_class_weight_balanced else "balanced",
        max_iter=args.max_iter,
    )

    LOGGER.info(
        "Starting repeated nested CV with %d samples, %d features, %d repeats",
        len(X),
        len(features),
        config.n_repeats,
    )
    result = nested_cv_lr(X, y, config)

    predictions = make_prediction_table(result, metadata)
    case_level_predictions = make_case_level_pooled_predictions(predictions)
    metrics = evaluate_all_thresholds(
        result,
        n_boot=config.bootstrap_iterations,
        seed=config.seed,
    )
    per_repeat_metrics = evaluate_per_repeat(result)
    repeat_variance = summarize_repeat_variance(per_repeat_metrics)
    calibration = summarize_calibration(
        result,
        n_boot=config.bootstrap_iterations,
        seed=config.seed,
    )

    final_model = None
    final_model_params = None
    if args.save_final_model:
        LOGGER.info("Tuning/fitting final model on all data")
        final_model, final_model_params = fit_final_model(X, y, config=config)

    save_run_outputs(
        Path(args.output_dir),
        result=result,
        predictions=predictions,
        case_level_predictions=case_level_predictions,
        metrics=metrics,
        per_repeat_metrics=per_repeat_metrics,
        repeat_variance=repeat_variance,
        calibration=calibration,
        features=features,
        config=config,
        save_final_model=args.save_final_model,
        final_model=final_model,
        final_model_params=final_model_params,
    )

    LOGGER.info("Done. Outputs saved to %s", Path(args.output_dir).resolve())
    LOGGER.info("ROC AUC: %.3f", safe_roc_auc_score(result.y_true, result.probabilities))
    LOGGER.info("AUPRC: %.3f", safe_auprc(result.y_true, result.probabilities))


if __name__ == "__main__":
    main()
