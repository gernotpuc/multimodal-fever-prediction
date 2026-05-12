"""
External validation pipeline for a last-observed-value naive baseline.

This script trains/calibrates the naive baseline using internal data only and evaluates
it on one or more holdout/external datasets.

Baseline idea:
- use one numeric predictor, e.g. last observed temperature in a prior window
- fit min-max scaling using the full internal training data only
- derive MCC/Youden thresholds from internal probabilities only
- apply the fixed scaling and thresholds to each external dataset

This makes the naive baseline comparable to external-validation ML pipelines while
avoiding leakage from the external data.

Example:
    python scripts/external_validate_naive_baseline.py \
        --internal-data artifacts/preprocessed/internal.parquet \
        --external-data holdout=artifacts/preprocessed/holdout.parquet external=artifacts/preprocessed/external.parquet \
        --target-col fever \
        --naive-feature last_bt_fever_lag_1 \
        --id-cols encounter_id subject_reference \
        --output-dir artifacts/models/naive_external_validation
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
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

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NaiveExternalConfig:
    target_col: str
    naive_feature: str
    id_cols: List[str]
    seed: int = 42
    bootstrap_iterations: int = 2000
    higher_values_mean_higher_risk: bool = True


# -----------------------------------------------------------------------------
# Logging and I/O
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


def parse_named_paths(values: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in values:
        if "=" in item:
            name, path = item.split("=", 1)
            out[name.strip()] = Path(path)
        else:
            path = Path(item)
            out[path.stem] = path
    return out


def prepare_dataset(
    data: pd.DataFrame,
    *,
    target_col: str,
    naive_feature: str,
    id_cols: Sequence[str],
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    missing = [col for col in [target_col, naive_feature] if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset '{dataset_name}' is missing required columns: {missing}")

    working = data.copy()
    working[target_col] = working[target_col].astype(int)
    working[naive_feature] = pd.to_numeric(working[naive_feature], errors="coerce")

    valid = working[[target_col, naive_feature]].notna().all(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        LOGGER.warning("Dataset '%s': dropped %d rows with missing target/naive feature.", dataset_name, dropped)

    working = working.loc[valid].copy()
    X = working[[naive_feature]].copy()
    y = working[target_col].copy()
    metadata_cols = [col for col in id_cols if col in working.columns]
    metadata = working[metadata_cols].copy()

    if y.nunique() < 2:
        LOGGER.warning("Dataset '%s' has fewer than two classes; AUC/calibration may be undefined.", dataset_name)

    return X, y, metadata


# -----------------------------------------------------------------------------
# Naive scaling and thresholds
# -----------------------------------------------------------------------------

def fit_minmax_scaler_from_internal(
    values: np.ndarray,
    *,
    higher_values_mean_higher_risk: bool,
) -> Dict[str, float]:
    return {
        "train_min": float(np.min(values)),
        "train_max": float(np.max(values)),
        "higher_values_mean_higher_risk": float(higher_values_mean_higher_risk),
    }


def transform_minmax_probabilities(values: np.ndarray, scaler: Dict[str, float]) -> np.ndarray:
    train_min = scaler["train_min"]
    train_max = scaler["train_max"]
    denom = (train_max - train_min) + 1e-12
    prob = (values - train_min) / denom
    prob = np.clip(prob, 0.0, 1.0)
    if not bool(scaler["higher_values_mean_higher_risk"]):
        prob = 1.0 - prob
    return prob


def optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return {"default": 0.5, "f1": 0.5, "mcc": 0.5, "youden": 0.5}

    f1_values = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_threshold = float(thresholds[np.argmax(f1_values[:-1])])

    mcc_values = [matthews_corrcoef(y_true, (y_prob >= threshold).astype(int)) for threshold in thresholds]
    mcc_threshold = float(thresholds[int(np.argmax(mcc_values))])

    youden_values = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        youden_values.append(sensitivity + specificity - 1)
    youden_threshold = float(thresholds[int(np.argmax(youden_values))])

    return {"default": 0.5, "f1": f1_threshold, "mcc": mcc_threshold, "youden": youden_threshold}


# -----------------------------------------------------------------------------
# Metrics and calibration
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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": safe_roc_auc_score(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, *, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        rows.append(compute_binary_metrics(y_true[idx], y_pred[idx], y_prob[idx]))
    boot = pd.DataFrame(rows)
    summary = []
    for metric in boot.columns:
        mean, low, high = ci(boot[metric])
        summary.append({"metric": metric, "mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(summary)


def logits_from_probabilities(y_prob: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    return np.log(y_prob + eps) - np.log(1 - y_prob + eps)


def calibration_in_the_large(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    logits = logits_from_probabilities(y_prob)
    model = LogisticRegression(solver="lbfgs")
    model.fit(logits.reshape(-1, 1), y_true)
    return float(model.intercept_[0])


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
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


def bootstrap_calibration(y_true: np.ndarray, y_prob: np.ndarray, *, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, pr = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        rows.append(compute_calibration_metrics(yt, pr))

    boot = pd.DataFrame(rows)
    summary = []
    for metric in ["brier", "citl", "slope"]:
        if metric not in boot.columns:
            summary.append({"metric": metric, "bootstrap_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue
        mean, low, high = ci(boot[metric])
        summary.append({"metric": metric, "bootstrap_mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(summary)


# -----------------------------------------------------------------------------
# Output tables
# -----------------------------------------------------------------------------

def make_prediction_table(
    *,
    dataset_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    df = pd.DataFrame({
        "dataset": dataset_name,
        "row_index": y_true.index.to_numpy(),
        "y_true": y_true.to_numpy(dtype=int),
        "p": probabilities,
        "pred_0_5": (probabilities >= thresholds["default"]).astype(int),
        "pred_mcc": (probabilities >= thresholds["mcc"]).astype(int),
        "pred_youden": (probabilities >= thresholds["youden"]).astype(int),
    })
    if not metadata.empty:
        meta = metadata.reset_index().rename(columns={"index": "row_index"})
        df = df.merge(meta, on="row_index", how="left")

    for pred_col in ["pred_0_5", "pred_mcc", "pred_youden"]:
        df[f"FP_{pred_col}"] = df["y_true"].eq(0) & df[pred_col].eq(1)
        df[f"FN_{pred_col}"] = df["y_true"].eq(1) & df[pred_col].eq(0)
    return df


def evaluate_thresholds(predictions: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    strategy_cols = {
        "default_0_5": "pred_0_5",
        "mcc_optimal": "pred_mcc",
        "youden": "pred_youden",
    }
    rows = []
    for strategy, pred_col in strategy_cols.items():
        boot = bootstrap_metrics(
            predictions["y_true"].to_numpy(dtype=int),
            predictions[pred_col].to_numpy(dtype=int),
            predictions["p"].to_numpy(dtype=float),
            n_boot=n_boot,
            seed=seed,
        )
        boot.insert(0, "threshold_strategy", strategy)
        boot.insert(0, "dataset", predictions["dataset"].iloc[0])
        rows.append(boot)
    return pd.concat(rows, ignore_index=True)


def summarize_calibration_for_predictions(predictions: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_prob = predictions["p"].to_numpy(dtype=float)
    point = compute_calibration_metrics(y_true, y_prob)
    boot = bootstrap_calibration(y_true, y_prob, n_boot=n_boot, seed=seed)
    boot["point_estimate"] = boot["metric"].map(point)
    boot.insert(0, "dataset", predictions["dataset"].iloc[0])
    return boot[["dataset", "metric", "point_estimate", "bootstrap_mean", "ci_low", "ci_high"]]


def get_fp_fn(df: pd.DataFrame, pred_col: str, prob_col: str = "p") -> Tuple[pd.DataFrame, pd.DataFrame]:
    fp = df[(df["y_true"] == 0) & (df[pred_col] == 1)].copy().sort_values(prob_col, ascending=False)
    fn = df[(df["y_true"] == 1) & (df[pred_col] == 0)].copy().sort_values(prob_col, ascending=True)
    return fp, fn


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = safe_roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc_value = safe_auprc(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AUPRC={auprc_value:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, title: str, *, n_bins: int = 10) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Naive")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed outcome frequency")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run_external_validation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = NaiveExternalConfig(
        target_col=args.target_col,
        naive_feature=args.naive_feature,
        id_cols=args.id_cols,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
        higher_values_mean_higher_risk=not args.lower_values_mean_higher_risk,
    )

    internal = read_table(Path(args.internal_data), sep=args.csv_sep)
    X_internal, y_internal, _ = prepare_dataset(
        internal,
        target_col=args.target_col,
        naive_feature=args.naive_feature,
        id_cols=args.id_cols,
        dataset_name="internal",
    )

    scaler = fit_minmax_scaler_from_internal(
        X_internal[args.naive_feature].to_numpy(dtype=float),
        higher_values_mean_higher_risk=config.higher_values_mean_higher_risk,
    )
    internal_prob = transform_minmax_probabilities(X_internal[args.naive_feature].to_numpy(dtype=float), scaler)
    thresholds = optimal_thresholds(y_internal.to_numpy(dtype=int), internal_prob)

    pd.DataFrame([thresholds]).to_csv(output_dir / "thresholds_from_internal.csv", index=False)
    pd.DataFrame([scaler]).to_csv(output_dir / "internal_minmax_scaler.csv", index=False)
    pd.DataFrame({
        "row_index": X_internal.index,
        "dataset": "internal_threshold_source",
        "y_true": y_internal.to_numpy(dtype=int),
        "p": internal_prob,
    }).to_csv(output_dir / "internal_probabilities_for_thresholds.csv", index=False)

    all_predictions = []
    all_metrics = []
    all_calibration = []

    for dataset_name, dataset_path in parse_named_paths(args.external_data).items():
        LOGGER.info("Evaluating dataset '%s': %s", dataset_name, dataset_path)
        external = read_table(dataset_path, sep=args.csv_sep)
        X_ext, y_ext, metadata = prepare_dataset(
            external,
            target_col=args.target_col,
            naive_feature=args.naive_feature,
            id_cols=args.id_cols,
            dataset_name=dataset_name,
        )
        p_ext = transform_minmax_probabilities(X_ext[args.naive_feature].to_numpy(dtype=float), scaler)
        predictions = make_prediction_table(
            dataset_name=dataset_name,
            y_true=y_ext,
            probabilities=p_ext,
            metadata=metadata,
            thresholds=thresholds,
        )
        all_predictions.append(predictions)

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(dataset_dir / "predictions.csv", index=False)

        metrics = evaluate_thresholds(predictions, n_boot=config.bootstrap_iterations, seed=config.seed)
        calibration = summarize_calibration_for_predictions(predictions, n_boot=config.bootstrap_iterations, seed=config.seed)
        metrics.to_csv(dataset_dir / "metrics.csv", index=False)
        calibration.to_csv(dataset_dir / "calibration.csv", index=False)
        all_metrics.append(metrics)
        all_calibration.append(calibration)

        for pred_col, strategy in [("pred_0_5", "default_0_5"), ("pred_mcc", "mcc_optimal"), ("pred_youden", "youden")]:
            fp, fn = get_fp_fn(predictions, pred_col)
            fp.to_csv(dataset_dir / f"false_positives_{strategy}.csv", index=False)
            fn.to_csv(dataset_dir / f"false_negatives_{strategy}.csv", index=False)

        y_np = y_ext.to_numpy(dtype=int)
        plot_roc_curve(y_np, p_ext, dataset_dir / "roc_curve.png", title=f"ROC Curve – Naive – {dataset_name}")
        plot_pr_curve(y_np, p_ext, dataset_dir / "precision_recall_curve.png", title=f"Precision–Recall Curve – Naive – {dataset_name}")
        plot_calibration_curve(y_np, p_ext, dataset_dir / "calibration_curve.png", title=f"Calibration Curve – Naive – {dataset_name}")

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "external_predictions_all_datasets.csv", index=False)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(output_dir / "external_metrics_all_datasets.csv", index=False)
    if all_calibration:
        pd.concat(all_calibration, ignore_index=True).to_csv(output_dir / "external_calibration_all_datasets.csv", index=False)

    metadata = {
        "config": asdict(config),
        "internal_data": args.internal_data,
        "external_data": {name: str(path) for name, path in parse_named_paths(args.external_data).items()},
        "thresholds": thresholds,
        "scaler": scaler,
        "n_internal": int(len(X_internal)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Done. Outputs saved to %s", output_dir.resolve())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External validation for a naive last-observed-value baseline.")
    parser.add_argument("--internal-data", required=True, help="Internal training CSV/Parquet.")
    parser.add_argument("--external-data", nargs="+", required=True, help="External datasets as name=path or plain paths.")
    parser.add_argument("--csv-sep", default=";", help="CSV separator for CSV inputs.")
    parser.add_argument("--target-col", default="fever")
    parser.add_argument("--naive-feature", default="last_bt_fever_lag_1")
    parser.add_argument("--id-cols", nargs="*", default=["encounter_id", "subject_reference"])
    parser.add_argument("--output-dir", default="artifacts/naive_external_validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--lower-values-mean-higher-risk", action="store_true", help="Invert min-max probabilities after scaling.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    run_external_validation(args)
