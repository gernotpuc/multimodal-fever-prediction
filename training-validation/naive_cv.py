"""
Repeated cross-validation pipeline for a last-observed-value naive baseline.

This script is designed for reproducible GitHub use and to be directly comparable
with repeated nested-CV ML models.

Baseline idea:
- use one numeric feature, e.g. last observed temperature in a prior window
- fit min-max scaling on each outer-training fold only
- apply that scaling to the corresponding outer-test fold to obtain probabilities
- derive MCC/Youden thresholds from outer-training probabilities only
- repeat the full outer CV with different seeds

Example:
    python scripts/train_repeated_cv_naive_baseline.py \
        --data-path artifacts/preprocessed/data_ml.parquet \
        --target-col fever \
        --naive-feature last_bt_fever_lag_1 \
        --id-cols encounter_id subject_reference \
        --output-dir artifacts/models/naive_repeated_cv \
        --outer-splits 5 \
        --n-repeats 5
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NaiveCVConfig:
    outer_splits: int = 5
    n_repeats: int = 1
    seed: int = 42
    bootstrap_iterations: int = 2000
    naive_feature: str = "last_bt_fever_lag_1"
    higher_values_mean_higher_risk: bool = True


@dataclass(frozen=True)
class CVResult:
    indices: np.ndarray
    y_true: np.ndarray
    probabilities: np.ndarray
    pred_default: np.ndarray
    pred_mcc: np.ndarray
    pred_youden: np.ndarray
    repeat: np.ndarray
    fold: np.ndarray
    fold_thresholds: List[Dict[str, float]]
    fold_scalers: List[Dict[str, float]]


# -----------------------------------------------------------------------------
# I/O
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
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, sep=sep)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Unsupported input format. Use CSV or Parquet.")


def prepare_xy(
    data: pd.DataFrame,
    *,
    target_col: str,
    naive_feature: str,
    id_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    missing = [col for col in [target_col, naive_feature] if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = data.copy()
    working[target_col] = working[target_col].astype(int)
    working[naive_feature] = pd.to_numeric(working[naive_feature], errors="coerce")

    valid = working[[target_col, naive_feature]].notna().all(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        LOGGER.warning("Dropped %d rows with missing target or naive feature.", dropped)

    working = working.loc[valid].copy()
    if working[target_col].nunique() < 2:
        raise ValueError(f"Target '{target_col}' must contain at least two classes.")

    X = working[[naive_feature]].copy()
    y = working[target_col].copy()
    metadata_cols = [col for col in id_cols if col in working.columns]
    metadata = working[metadata_cols].copy()
    return X, y, metadata


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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
        "auc": safe_auc(y_true, y_prob),
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
    n_boot: int,
    seed: int,
) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    keys = ["accuracy", "precision", "recall", "specificity", "sensitivity", "npv", "f1", "mcc", "auc", "auprc", "tn", "fp", "fn", "tp"]
    out = {key: [] for key in keys}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        values = compute_binary_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for key, value in values.items():
            out[key].append(value)
    return out


def ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


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


def bootstrap_calibration(y_true: np.ndarray, y_prob: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    out = {"brier": [], "citl": [], "slope": []}
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, pr = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals = compute_calibration_metrics(yt, pr)
        for key, value in vals.items():
            out[key].append(value)
    return out


def summarize_calibration(result: CVResult, *, n_boot: int, seed: int) -> pd.DataFrame:
    point = compute_calibration_metrics(result.y_true, result.probabilities)
    boot = bootstrap_calibration(result.y_true, result.probabilities, n_boot=n_boot, seed=seed)
    rows = []
    for metric, point_value in point.items():
        mean, low, high = ci(boot[metric])
        rows.append({"metric": metric, "point_estimate": point_value, "bootstrap_mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Thresholds and probabilities
# -----------------------------------------------------------------------------

def optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return {"f1": 0.5, "mcc": 0.5, "youden": 0.5}

    f1_values = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_threshold = float(thresholds[np.argmax(f1_values[:-1])])

    mcc_values = [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    mcc_threshold = float(thresholds[int(np.argmax(mcc_values))])

    youden_values = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        youden_values.append(sensitivity + specificity - 1)
    youden_threshold = float(thresholds[int(np.argmax(youden_values))])

    return {"f1": f1_threshold, "mcc": mcc_threshold, "youden": youden_threshold}


def minmax_train_test_probabilities(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    higher_values_mean_higher_risk: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    min_value = float(np.min(x_train))
    max_value = float(np.max(x_train))
    denom = (max_value - min_value) + 1e-12
    p_train = (x_train - min_value) / denom
    p_test = (x_test - min_value) / denom
    p_train = np.clip(p_train, 0.0, 1.0)
    p_test = np.clip(p_test, 0.0, 1.0)

    if not higher_values_mean_higher_risk:
        p_train = 1.0 - p_train
        p_test = 1.0 - p_test

    return p_train, p_test, min_value, max_value


# -----------------------------------------------------------------------------
# Repeated CV
# -----------------------------------------------------------------------------

def repeated_cv_naive(X: pd.DataFrame, y: pd.Series, config: NaiveCVConfig) -> CVResult:
    feature = config.naive_feature
    if feature not in X.columns:
        raise ValueError(f"Naive predictor feature '{feature}' missing.")

    all_idx: List[np.ndarray] = []
    all_y_true: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []
    all_default: List[np.ndarray] = []
    all_mcc: List[np.ndarray] = []
    all_youden: List[np.ndarray] = []
    all_repeats: List[np.ndarray] = []
    all_folds: List[np.ndarray] = []
    fold_thresholds: List[Dict[str, float]] = []
    fold_scalers: List[Dict[str, float]] = []

    x_feature = X[feature].astype(float)

    for repeat in range(1, config.n_repeats + 1):
        repeat_seed = config.seed + repeat - 1
        outer_cv = StratifiedKFold(n_splits=config.outer_splits, shuffle=True, random_state=repeat_seed)
        LOGGER.info("Starting repeat %d/%d with seed=%d", repeat, config.n_repeats, repeat_seed)

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            x_train = x_feature.iloc[train_idx].to_numpy(dtype=float)
            x_test = x_feature.iloc[test_idx].to_numpy(dtype=float)
            y_train = y.iloc[train_idx].to_numpy(dtype=int)
            y_test = y.iloc[test_idx].to_numpy(dtype=int)

            p_train, p_test, min_value, max_value = minmax_train_test_probabilities(
                x_train,
                x_test,
                higher_values_mean_higher_risk=config.higher_values_mean_higher_risk,
            )

            thresholds = optimal_thresholds(y_train, p_train)
            fold_thresholds.append({
                "repeat": float(repeat),
                "fold": float(fold),
                "default": 0.5,
                "mcc": thresholds["mcc"],
                "youden": thresholds["youden"],
                "f1": thresholds["f1"],
            })
            fold_scalers.append({
                "repeat": float(repeat),
                "fold": float(fold),
                "train_min": min_value,
                "train_max": max_value,
                "higher_values_mean_higher_risk": float(config.higher_values_mean_higher_risk),
            })

            all_idx.append(X.iloc[test_idx].index.to_numpy())
            all_y_true.append(y_test)
            all_prob.append(p_test)
            all_default.append((p_test >= 0.5).astype(int))
            all_mcc.append((p_test >= thresholds["mcc"]).astype(int))
            all_youden.append((p_test >= thresholds["youden"]).astype(int))
            all_repeats.append(np.full(len(test_idx), repeat, dtype=int))
            all_folds.append(np.full(len(test_idx), fold, dtype=int))

    return CVResult(
        indices=np.concatenate(all_idx),
        y_true=np.concatenate(all_y_true),
        probabilities=np.concatenate(all_prob),
        pred_default=np.concatenate(all_default),
        pred_mcc=np.concatenate(all_mcc),
        pred_youden=np.concatenate(all_youden),
        repeat=np.concatenate(all_repeats),
        fold=np.concatenate(all_folds),
        fold_thresholds=fold_thresholds,
        fold_scalers=fold_scalers,
    )


# -----------------------------------------------------------------------------
# Output tables
# -----------------------------------------------------------------------------

def make_prediction_table(result: CVResult, metadata: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "row_index": result.indices,
        "repeat": result.repeat,
        "fold": result.fold,
        "y_true": result.y_true,
        "p": result.probabilities,
        "pred_0_5": result.pred_default,
        "pred_mcc": result.pred_mcc,
        "pred_youden": result.pred_youden,
    })

    if not metadata.empty:
        meta = metadata.reset_index().rename(columns={"index": "row_index"})
        df = df.merge(meta, on="row_index", how="left")

    for pred_col in ["pred_0_5", "pred_mcc", "pred_youden"]:
        df[f"FP_{pred_col}"] = df["y_true"].eq(0) & df[pred_col].eq(1)
        df[f"FN_{pred_col}"] = df["y_true"].eq(1) & df[pred_col].eq(0)
    return df


def make_case_level_pooled_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    pooled = predictions.groupby("row_index", as_index=False).agg(
        y_true=("y_true", "first"),
        p_mean=("p", "mean"),
        p_std=("p", "std"),
        p_min=("p", "min"),
        p_max=("p", "max"),
        n_repeats_observed=("repeat", "nunique"),
    )

    excluded = {
        "repeat", "fold", "y_true", "p", "pred_0_5", "pred_mcc", "pred_youden",
        "FP_pred_0_5", "FN_pred_0_5", "FP_pred_mcc", "FN_pred_mcc", "FP_pred_youden", "FN_pred_youden",
    }
    metadata_cols = [col for col in predictions.columns if col not in excluded and col != "row_index"]
    if metadata_cols:
        meta = predictions[["row_index", *metadata_cols]].drop_duplicates(subset=["row_index"])
        pooled = pooled.merge(meta, on="row_index", how="left")

    votes = predictions.groupby("row_index", as_index=False).agg(
        pred_0_5_vote=("pred_0_5", "mean"),
        pred_mcc_vote=("pred_mcc", "mean"),
        pred_youden_vote=("pred_youden", "mean"),
    )
    pooled = pooled.merge(votes, on="row_index", how="left")

    pooled["pred_0_5"] = (pooled["p_mean"] >= 0.5).astype(int)
    pooled["pred_mcc_majority"] = (pooled["pred_mcc_vote"] >= 0.5).astype(int)
    pooled["pred_youden_majority"] = (pooled["pred_youden_vote"] >= 0.5).astype(int)

    for pred_col in ["pred_0_5", "pred_mcc_majority", "pred_youden_majority"]:
        pooled[f"FP_{pred_col}"] = pooled["y_true"].eq(0) & pooled[pred_col].eq(1)
        pooled[f"FN_{pred_col}"] = pooled["y_true"].eq(1) & pooled[pred_col].eq(0)

    return pooled.sort_values("row_index")


def evaluate_all_thresholds(result: CVResult, *, n_boot: int, seed: int) -> pd.DataFrame:
    threshold_map = {
        "default_0_5": result.pred_default,
        "mcc_optimal": result.pred_mcc,
        "youden": result.pred_youden,
    }
    rows = []
    for strategy, predictions in threshold_map.items():
        boot = bootstrap_metrics(result.y_true, predictions, result.probabilities, n_boot=n_boot, seed=seed)
        summary = summarize_bootstrap_metrics(boot)
        summary.insert(0, "threshold_strategy", strategy)
        summary.insert(0, "aggregation", "pooled_repeat_fold_predictions")
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def evaluate_per_repeat(result: CVResult) -> pd.DataFrame:
    rows = []
    threshold_map = {
        "default_0_5": result.pred_default,
        "mcc_optimal": result.pred_mcc,
        "youden": result.pred_youden,
    }
    for repeat in sorted(np.unique(result.repeat)):
        mask = result.repeat == repeat
        for strategy, predictions in threshold_map.items():
            values = compute_binary_metrics(result.y_true[mask], predictions[mask], result.probabilities[mask])
            rows.append({"repeat": int(repeat), "threshold_strategy": strategy, **values})
    return pd.DataFrame(rows)


def summarize_repeat_variance(per_repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in per_repeat_metrics.columns if col not in {"repeat", "threshold_strategy"}]
    rows = []
    for strategy, group in per_repeat_metrics.groupby("threshold_strategy"):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "threshold_strategy": strategy,
                "metric": metric,
                "mean_across_repeats": float(values.mean()),
                "std_across_repeats": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min_across_repeats": float(values.min()),
                "max_across_repeats": float(values.max()),
                "n_repeats": int(values.shape[0]),
            })
    return pd.DataFrame(rows)


def get_fp_fn(df: pd.DataFrame, pred_col: str, prob_col: str = "p") -> Tuple[pd.DataFrame, pd.DataFrame]:
    fp = df[(df["y_true"] == 0) & (df[pred_col] == 1)].copy().sort_values(prob_col, ascending=False)
    fn = df[(df["y_true"] == 1) & (df[pred_col] == 0)].copy().sort_values(prob_col, ascending=True)
    return fp, fn


# -----------------------------------------------------------------------------
# Plots and saving
# -----------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = safe_auc(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve – Naive Baseline")
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
    plt.title("Precision–Recall Curve – Naive Baseline")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, *, n_bins: int = 10) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Naive")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed outcome frequency")
    plt.title("Calibration Curve – Naive Baseline")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_outputs(
    output_dir: Path,
    *,
    result: CVResult,
    predictions: pd.DataFrame,
    case_level_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    per_repeat_metrics: pd.DataFrame,
    repeat_variance: pd.DataFrame,
    calibration: pd.DataFrame,
    config: NaiveCVConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_dir / "naive_predictions_all_repeats.csv", index=False)
    case_level_predictions.to_csv(output_dir / "naive_predictions_case_level_pooled.csv", index=False)
    metrics.to_csv(output_dir / "naive_metrics_pooled.csv", index=False)
    per_repeat_metrics.to_csv(output_dir / "naive_metrics_per_repeat.csv", index=False)
    repeat_variance.to_csv(output_dir / "naive_metric_variance_across_repeats.csv", index=False)
    calibration.to_csv(output_dir / "naive_calibration_pooled.csv", index=False)
    pd.DataFrame(result.fold_thresholds).to_csv(output_dir / "fold_thresholds.csv", index=False)
    pd.DataFrame(result.fold_scalers).to_csv(output_dir / "fold_minmax_scalers.csv", index=False)

    for pred_col, name in [("pred_0_5", "default_0_5"), ("pred_mcc", "mcc_optimal"), ("pred_youden", "youden")]:
        fp, fn = get_fp_fn(predictions, pred_col)
        fp.to_csv(output_dir / f"false_positives_all_repeats_{name}.csv", index=False)
        fn.to_csv(output_dir / f"false_negatives_all_repeats_{name}.csv", index=False)

    case_for_errors = case_level_predictions.rename(columns={"p_mean": "p"})
    for pred_col, name in [
        ("pred_0_5", "default_0_5"),
        ("pred_mcc_majority", "mcc_majority_vote"),
        ("pred_youden_majority", "youden_majority_vote"),
    ]:
        fp, fn = get_fp_fn(case_for_errors, pred_col)
        fp.to_csv(output_dir / f"false_positives_case_level_{name}.csv", index=False)
        fn.to_csv(output_dir / f"false_negatives_case_level_{name}.csv", index=False)

    plot_roc_curve(result.y_true, result.probabilities, output_dir / "roc_curve_pooled_repeats.png")
    plot_pr_curve(result.y_true, result.probabilities, output_dir / "precision_recall_curve_pooled_repeats.png")
    plot_calibration_curve(result.y_true, result.probabilities, output_dir / "calibration_curve_pooled_repeats.png")

    metadata = {
        "config": asdict(config),
        "n_outer_test_predictions": int(len(result.y_true)),
        "n_unique_samples": int(len(np.unique(result.indices))),
        "positive_rate_outer_predictions": float(np.mean(result.y_true)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated CV naive last-observed-value baseline.")
    parser.add_argument("--data-path", required=True, help="Input CSV or Parquet containing feature and target.")
    parser.add_argument("--csv-sep", default=";", help="CSV separator if input is CSV.")
    parser.add_argument("--target-col", default="fever", help="Binary target column.")
    parser.add_argument("--naive-feature", default="last_bt_fever_lag_1", help="Single numeric feature used as naive predictor.")
    parser.add_argument("--id-cols", nargs="*", default=["encounter_id", "subject_reference"], help="Metadata columns to carry into outputs.")
    parser.add_argument("--output-dir", default="artifacts/naive_repeated_cv", help="Output directory.")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--lower-values-mean-higher-risk", action="store_true", help="Invert min-max probabilities after scaling.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    data = read_table(Path(args.data_path), sep=args.csv_sep)
    X, y, metadata = prepare_xy(
        data,
        target_col=args.target_col,
        naive_feature=args.naive_feature,
        id_cols=args.id_cols,
    )

    config = NaiveCVConfig(
        outer_splits=args.outer_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
        naive_feature=args.naive_feature,
        higher_values_mean_higher_risk=not args.lower_values_mean_higher_risk,
    )

    LOGGER.info(
        "Starting repeated CV naive baseline with %d samples, feature=%s, repeats=%d",
        len(X),
        args.naive_feature,
        args.n_repeats,
    )
    result = repeated_cv_naive(X, y, config)

    predictions = make_prediction_table(result, metadata)
    case_level_predictions = make_case_level_pooled_predictions(predictions)
    metrics = evaluate_all_thresholds(result, n_boot=config.bootstrap_iterations, seed=config.seed)
    per_repeat_metrics = evaluate_per_repeat(result)
    repeat_variance = summarize_repeat_variance(per_repeat_metrics)
    calibration = summarize_calibration(result, n_boot=config.bootstrap_iterations, seed=config.seed)

    save_outputs(
        Path(args.output_dir),
        result=result,
        predictions=predictions,
        case_level_predictions=case_level_predictions,
        metrics=metrics,
        per_repeat_metrics=per_repeat_metrics,
        repeat_variance=repeat_variance,
        calibration=calibration,
        config=config,
    )

    LOGGER.info("Done. Outputs saved to %s", Path(args.output_dir).resolve())
    LOGGER.info("ROC AUC: %.3f", safe_auc(result.y_true, result.probabilities))
    LOGGER.info("AUPRC: %.3f", safe_auprc(result.y_true, result.probabilities))


if __name__ == "__main__":
    main()
