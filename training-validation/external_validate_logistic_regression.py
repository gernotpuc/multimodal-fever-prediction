"""
External validation pipeline for logistic regression.

This script tunes and trains logistic regression on an internal dataset, then validates
on one or more holdout/external datasets.

Design goals:
- GitHub-ready CLI script
- no hard-coded paths
- hyperparameter tuning on internal data only
- optional Platt/isotonic calibration using an internal calibration split only
- MCC/Youden thresholds derived from internal OOF probabilities only
- same metric/calibration/error-analysis outputs as the nested-CV scripts

Example:
    python scripts/external_validate_logistic_regression.py \
        --internal-data artifacts/preprocessed/internal.parquet \
        --external-data holdout=artifacts/preprocessed/holdout.parquet external=artifacts/preprocessed/external.parquet \
        --target-col fever \
        --id-cols encounter_id subject_reference \
        --output-dir artifacts/models/lr_external_validation \
        --calibration-mode uncal \
        --threshold-oof-splits 5 \
        --n-trials 50 \
        --save-model

If --feature-cols is omitted, all numeric/bool columns except target/id columns are used.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except Exception:
    FrozenEstimator = None
    HAS_FROZEN = False

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalValidationConfig:
    target_col: str
    id_cols: List[str]
    calibration_mode: str = "uncal"
    calibration_size: float = 0.20
    threshold_oof_splits: int = 5
    tuning_inner_splits: int = 5
    n_trials: int = 50
    seed: int = 42
    bootstrap_iterations: int = 2000
    optimize_metric: str = "auc"
    class_weight: Optional[str] = "balanced"
    max_iter: int = 5000


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
            name = name.strip()
            if not name:
                raise ValueError(f"Invalid named path: {item}")
            out[name] = Path(path)
        else:
            path = Path(item)
            out[path.stem] = path
    return out


def infer_feature_columns(
    data: pd.DataFrame,
    *,
    target_col: str,
    id_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]],
) -> List[str]:
    if feature_cols:
        missing = [col for col in feature_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from internal data: {missing}")
        return list(feature_cols)

    excluded = set(id_cols) | {target_col}
    candidate_cols = [col for col in data.columns if col not in excluded]
    numeric_cols = data[candidate_cols].select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found. Pass --feature-cols explicitly.")
    return numeric_cols


def prepare_dataset(
    data: pd.DataFrame,
    *,
    features: Sequence[str],
    target_col: str,
    id_cols: Sequence[str],
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    missing = [col for col in [target_col, *features] if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset '{dataset_name}' is missing required columns: {missing}")

    working = data.copy()
    working[target_col] = working[target_col].astype(int)
    X = working[list(features)].copy()
    y = working[target_col].copy()

    valid = X.notna().all(axis=1) & y.notna()
    dropped = int((~valid).sum())
    if dropped:
        LOGGER.warning("Dataset '%s': dropped %d rows with missing features/target.", dataset_name, dropped)

    X = X.loc[valid]
    y = y.loc[valid]
    metadata_cols = [col for col in id_cols if col in working.columns]
    metadata = working.loc[valid, metadata_cols].copy()

    if y.nunique() < 2:
        LOGGER.warning("Dataset '%s' has fewer than two target classes; AUC/calibration may be undefined.", dataset_name)

    return X, y, metadata


# -----------------------------------------------------------------------------
# Metrics
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
    rows: List[Dict[str, float]] = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

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
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
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
# Logistic regression model/tuning/calibration
# -----------------------------------------------------------------------------

def build_lr_pipeline(params: Dict[str, Any], *, class_weight: Optional[str], max_iter: int) -> Pipeline:
    penalty = params["penalty"]
    solver = "liblinear" if penalty in {"l1", "l2"} else "saga"
    l1_ratio = params.get("l1_ratio") if penalty == "elasticnet" else None

    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                penalty=penalty,
                C=params["C"],
                l1_ratio=l1_ratio,
                solver=solver,
                class_weight=class_weight,
                max_iter=max_iter,
                n_jobs=-1 if solver == "saga" else None,
                random_state=42 if solver == "saga" else None,
            ),
        ),
    ])


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
                precision, recall, _ = precision_recall_curve(y.iloc[valid_idx], prob)
                score = auc(recall, precision)
            else:
                score = roc_auc_score(y.iloc[valid_idx], prob)
            scores.append(score)
        return float(np.mean(scores))
    return objective


def tune_lr_on_internal(X: pd.DataFrame, y: pd.Series, config: ExternalValidationConfig) -> Tuple[Dict[str, Any], optuna.Study]:
    LOGGER.info("Tuning LR hyperparameters on internal data only")
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective_lr(
            X,
            y,
            inner_splits=config.tuning_inner_splits,
            seed=config.seed,
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            optimize_metric=config.optimize_metric,
        ),
        n_trials=config.n_trials,
        show_progress_bar=False,
    )
    return study.best_params, study


def make_prefit_calibrator(base_model: Pipeline, *, method: str):
    if HAS_FROZEN:
        return CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)
    return CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")


def fit_lr_with_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    params: Dict[str, Any],
    calibration_mode: str,
    calibration_size: float,
    seed: int,
    class_weight: Optional[str],
    max_iter: int,
):
    if calibration_mode not in {"uncal", "platt", "iso"}:
        raise ValueError("calibration_mode must be one of: uncal, platt, iso")

    if calibration_mode == "uncal":
        model = build_lr_pipeline(params, class_weight=class_weight, max_iter=max_iter)
        model.fit(X_train, y_train)
        return model, None, None

    X_model, X_cal, y_model, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=calibration_size,
        random_state=seed,
        stratify=y_train,
    )
    base_model = build_lr_pipeline(params, class_weight=class_weight, max_iter=max_iter)
    base_model.fit(X_model, y_model)

    cal_platt = None
    cal_iso = None
    if calibration_mode == "platt":
        cal_platt = make_prefit_calibrator(base_model, method="sigmoid")
        cal_platt.fit(X_cal, y_cal)
    elif calibration_mode == "iso":
        cal_iso = make_prefit_calibrator(base_model, method="isotonic")
        cal_iso.fit(X_cal, y_cal)

    return base_model, cal_platt, cal_iso


def predict_lr_probabilities(base_model, cal_platt, cal_iso, X: pd.DataFrame, *, calibration_mode: str) -> np.ndarray:
    if calibration_mode == "uncal":
        return base_model.predict_proba(X)[:, 1]
    if calibration_mode == "platt":
        if cal_platt is None:
            raise ValueError("Platt calibrator is missing.")
        return cal_platt.predict_proba(X)[:, 1]
    if calibration_mode == "iso":
        if cal_iso is None:
            raise ValueError("Isotonic calibrator is missing.")
        return cal_iso.predict_proba(X)[:, 1]
    raise ValueError(f"Invalid calibration mode: {calibration_mode}")


def oof_predict_proba_lr(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: Dict[str, Any],
    calibration_mode: str,
    n_splits: int,
    calibration_size: float,
    seed: int,
    class_weight: Optional[str],
    max_iter: int,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        base, cal_platt, cal_iso = fit_lr_with_calibration(
            X_train,
            y_train,
            params=params,
            calibration_mode=calibration_mode,
            calibration_size=calibration_size,
            seed=seed,
            class_weight=class_weight,
            max_iter=max_iter,
        )
        oof[valid_idx] = predict_lr_probabilities(base, cal_platt, cal_iso, X_valid, calibration_mode=calibration_mode)
    return oof


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
    df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "row_index": y_true.index.to_numpy(),
            "y_true": y_true.to_numpy(dtype=int),
            "p": probabilities,
            "pred_0_5": (probabilities >= thresholds["default"]).astype(int),
            "pred_mcc": (probabilities >= thresholds["mcc"]).astype(int),
            "pred_youden": (probabilities >= thresholds["youden"]).astype(int),
        }
    )
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
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
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

    internal = read_table(Path(args.internal_data), sep=args.csv_sep)
    features = infer_feature_columns(
        internal,
        target_col=args.target_col,
        id_cols=args.id_cols,
        feature_cols=args.feature_cols,
    )
    X_internal, y_internal, _ = prepare_dataset(
        internal,
        features=features,
        target_col=args.target_col,
        id_cols=args.id_cols,
        dataset_name="internal",
    )

    config = ExternalValidationConfig(
        target_col=args.target_col,
        id_cols=args.id_cols,
        calibration_mode=args.calibration_mode,
        calibration_size=args.calibration_size,
        threshold_oof_splits=args.threshold_oof_splits,
        tuning_inner_splits=args.tuning_inner_splits,
        n_trials=args.n_trials,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
        optimize_metric=args.optimize_metric,
        class_weight=None if args.no_class_weight_balanced else "balanced",
        max_iter=args.max_iter,
    )

    best_params, study = tune_lr_on_internal(X_internal, y_internal, config)

    LOGGER.info("Fitting LR model on internal data")
    base_model, cal_platt, cal_iso = fit_lr_with_calibration(
        X_internal,
        y_internal,
        params=best_params,
        calibration_mode=config.calibration_mode,
        calibration_size=config.calibration_size,
        seed=config.seed,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
    )

    LOGGER.info("Deriving thresholds from internal OOF probabilities")
    internal_oof_prob = oof_predict_proba_lr(
        X_internal,
        y_internal,
        params=best_params,
        calibration_mode=config.calibration_mode,
        n_splits=config.threshold_oof_splits,
        calibration_size=config.calibration_size,
        seed=config.seed,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
    )
    thresholds = optimal_thresholds(y_internal.to_numpy(), internal_oof_prob)

    pd.DataFrame([thresholds]).to_csv(output_dir / "thresholds_from_internal_oof.csv", index=False)
    pd.DataFrame([best_params]).to_csv(output_dir / "best_params.csv", index=False)
    pd.DataFrame(
        {"row_index": X_internal.index, "dataset": "internal_oof_threshold_source", "y_true": y_internal.to_numpy(dtype=int), "p": internal_oof_prob}
    ).to_csv(output_dir / "internal_oof_probabilities_for_thresholds.csv", index=False)

    all_predictions = []
    all_metrics = []
    all_calibration = []

    for dataset_name, dataset_path in parse_named_paths(args.external_data).items():
        LOGGER.info("Evaluating external dataset '%s': %s", dataset_name, dataset_path)
        external = read_table(dataset_path, sep=args.csv_sep)
        X_ext, y_ext, ext_metadata = prepare_dataset(
            external,
            features=features,
            target_col=args.target_col,
            id_cols=args.id_cols,
            dataset_name=dataset_name,
        )

        X_ext = X_ext[features]
        p_ext = predict_lr_probabilities(base_model, cal_platt, cal_iso, X_ext, calibration_mode=config.calibration_mode)
        predictions = make_prediction_table(
            dataset_name=dataset_name,
            y_true=y_ext,
            probabilities=p_ext,
            metadata=ext_metadata,
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
        plot_roc_curve(y_np, p_ext, dataset_dir / "roc_curve.png", title=f"ROC Curve – Logistic Regression – {dataset_name}")
        plot_pr_curve(y_np, p_ext, dataset_dir / "precision_recall_curve.png", title=f"Precision–Recall Curve – Logistic Regression – {dataset_name}")
        plot_calibration_curve(y_np, p_ext, dataset_dir / "calibration_curve.png", title=f"Calibration Curve – Logistic Regression – {dataset_name}")

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "external_predictions_all_datasets.csv", index=False)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(output_dir / "external_metrics_all_datasets.csv", index=False)
    if all_calibration:
        pd.concat(all_calibration, ignore_index=True).to_csv(output_dir / "external_calibration_all_datasets.csv", index=False)

    metadata = {
        "config": asdict(config),
        "features": list(features),
        "internal_data": args.internal_data,
        "external_data": {name: str(path) for name, path in parse_named_paths(args.external_data).items()},
        "thresholds": thresholds,
        "best_params": best_params,
        "best_tuning_value": float(study.best_value),
        "n_internal": int(len(X_internal)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.save_model:
        joblib.dump(
            {
                "base_model": base_model,
                "cal_platt": cal_platt,
                "cal_iso": cal_iso,
                "features": features,
                "thresholds": thresholds,
                "best_params": best_params,
            },
            output_dir / "logistic_regression_external_validation_model.joblib",
        )

    LOGGER.info("Done. Outputs saved to %s", output_dir.resolve())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/tune logistic regression on internal data and externally validate.")

    parser.add_argument("--internal-data", required=True, help="Internal training CSV/Parquet.")
    parser.add_argument("--external-data", nargs="+", required=True, help="External datasets as name=path or plain paths.")
    parser.add_argument("--csv-sep", default=";", help="CSV separator for CSV inputs.")
    parser.add_argument("--target-col", default="fever")
    parser.add_argument("--id-cols", nargs="*", default=["encounter_id", "subject_reference"])
    parser.add_argument("--feature-cols", nargs="*", default=None)
    parser.add_argument("--output-dir", default="artifacts/lr_external_validation")

    parser.add_argument("--calibration-mode", choices=["uncal", "platt", "iso"], default="uncal")
    parser.add_argument("--calibration-size", type=float, default=0.20)
    parser.add_argument("--threshold-oof-splits", type=int, default=5)
    parser.add_argument("--tuning-inner-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--optimize-metric", choices=["auc", "auprc"], default="auc")
    parser.add_argument("--no-class-weight-balanced", action="store_true")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    run_external_validation(args)
