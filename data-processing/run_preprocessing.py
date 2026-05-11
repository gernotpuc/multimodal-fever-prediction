"""
Run the preprocessing pipeline using the generic preprocessing module.

This script is intended as a reproducible project-level runner. Keep the reusable
logic in `src/generic_preprocessing_pipeline.py`; keep dataset-specific filenames,
feature lists, and output locations here or in a config file.

Example:
    python scripts/run_preprocessing.py \
        --data-dir data/raw \
        --output-dir artifacts/preprocessed \
        --elixhauser-reference data/reference/HCUP_ELIXHAUSER_REFERENCE.xlsx

Expected input files by default:
    body_temperature.csv
    heart_rate.csv
    mean_arterial_pressure.csv
    so2.csv
    crp.csv
    bili.csv
    leua.csv
    krea.csv
    hb.csv
    conditions_stay.csv
    conditions_past.csv
    patient_info.csv
    ab_groups.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from tabular_features import (
    add_fourier_features,
    add_static_covariates,
    add_statistical_lag_features,
    calculate_elixhauser_score,
    create_temperature_features,
    create_time_lags_and_merge_features,
    do_train_test_split,
    ffill_impute_within_entity,
    preprocess_measurement_streams,
    setup_logging,
)

LOGGER = logging.getLogger("run_preprocessing")


DEFAULT_FEATURE_SPACE: list[str] = [
    # Static features
    "cond_hemat",
    "cond_leuk",
    "cond_solid",
    "age",
    "elixhauser_score",
    "sex_male",
    "length_stay",

    # Temporal features
    "time_lag_1_bt_max",
    "time_lag_2_bt_max",
    "time_lag_1_bt_mean",
    "time_lag_2_bt_mean",
    "time_lag_1_krea_max",
    "time_lag_2_krea_max",
    "time_lag_1_bili_max",
    "time_lag_2_bili_max",
    "time_lag_1_leua_max",
    "time_lag_2_leua_max",
    "time_lag_1_crp_max",
    "time_lag_2_crp_max",
    "time_lag_1_hb_max",
    "time_lag_2_hb_max",
    "time_lag_1_heart_rate_max",
    "time_lag_2_heart_rate_max",
    "time_lag_1_mean_arterial_pressure_max",
    "time_lag_2_mean_arterial_pressure_max",
    "time_lag_1_so2_max",
    "time_lag_2_so2_max",
    "fever_variability_lag_1",
    "fever_variability_lag_2",
    "last_bt_fever_lag_1",
    "fever_diff_lag_1",
    "fever_diff_lag_2",
    "fever_percent_lag_2",
    "fever_percent_lag_1",
    "fever_lag_1_evening",
    "fever_lag_1_afternoon",
    "fever_lag_1_morning",
    "fever_lag_2_evening",
    "fever_lag_2_afternoon",
    "fever_lag_2_morning",
    "fever_points_lag_1",
    "fever_points_lag_2",
    "fever_change_lag_1",
    "fever_change_lag_all",
    "fever_range_lag_1",
    "fever_range_lag_2",
    "trend",
    "fever_lag_1_night",
    "fever_lag_2_night",
    "skewness_lag_all",
    "fourier_max_magnitude",
]


INPUT_FILENAMES: Dict[str, str] = {
    # Vital signs
    "body_temperature": "body_temperature.csv",
    "heart_rate": "heart_rate.csv",
    "mean_arterial_pressure": "mean_arterial_pressure.csv",
    "so2": "so2.csv",

    # Labs
    "crp": "crp.csv",
    "bili": "bili.csv",
    "leua": "leua.csv",
    "krea": "krea.csv",
    "hb": "hb.csv",

    # Static covariates
    "conditions_stay": "conditions_stay.csv",
    "conditions_past": "conditions_past.csv",
    "patient_info": "patient_info.csv",

    # Interventions
    "ab_groups": "ab_groups.csv",
}


VALUE_COLUMN_MAP: Dict[str, str] = {
    "heart_rate": "heart_rate",
    "so2": "so2",
    "mean_arterial_pressure": "mean_arterial_pressure",
    "crp": "crp",
    "bili": "bili",
    "leua": "leua",
    "krea": "krea",
    "hb": "hb",
}


COVARIATE_ORDER: Sequence[str] = (
    "crp",
    "heart_rate",
    "mean_arterial_pressure",
    "so2",
    "bili",
    "hb",
    "krea",
    "leua",
)


def read_csv_checked(path: Path, *, sep: str = ",") -> pd.DataFrame:
    """Read a CSV and fail clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, sep=sep)


def load_inputs(data_dir: Path, *, csv_sep: str = ",") -> Dict[str, pd.DataFrame]:
    """Load all expected input CSVs."""
    LOGGER.info("Loading CSV inputs from %s", data_dir)
    return {
        name: read_csv_checked(data_dir / filename, sep=csv_sep)
        for name, filename in INPUT_FILENAMES.items()
    }


def standardize_demographics(patient_info: pd.DataFrame) -> pd.DataFrame:
    """Standardize demographic column names used by the preprocessing module."""
    out = patient_info.copy()
    if "gender" in out.columns and "sex" not in out.columns:
        out = out.rename(columns={"gender": "sex"})
    return out


def save_outputs(
    output_dir: Path,
    *,
    data_ml: pd.DataFrame,
    enriched_full: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save preprocessing artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data_ml.to_parquet(output_dir / "data_ml.parquet", index=False)
    enriched_full.to_parquet(output_dir / "enriched_full.parquet", index=False)
    X_train.to_parquet(output_dir / "X_train.parquet")
    X_test.to_parquet(output_dir / "X_test.parquet")
    y_train.to_frame("fever").to_parquet(output_dir / "y_train.parquet")
    y_test.to_frame("fever").to_parquet(output_dir / "y_test.parquet")

    LOGGER.info("Saved artifacts to %s", output_dir.resolve())


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    elixhauser_reference = Path(args.elixhauser_reference)

    if not elixhauser_reference.exists():
        raise FileNotFoundError(f"Missing Elixhauser reference file: {elixhauser_reference}")

    inputs = load_inputs(data_dir, csv_sep=args.csv_sep)
    inputs["patient_info"] = standardize_demographics(inputs["patient_info"])

    measurements = {
        "temperature": inputs["body_temperature"],
        "heart_rate": inputs["heart_rate"],
        "mean_arterial_pressure": inputs["mean_arterial_pressure"],
        "so2": inputs["so2"],
        "crp": inputs["crp"],
        "bili": inputs["bili"],
        "leua": inputs["leua"],
        "krea": inputs["krea"],
        "hb": inputs["hb"],
    }

    preprocessed = preprocess_measurement_streams(
        measurements=measurements,
        interventions=inputs["ab_groups"],
        value_column_map=VALUE_COLUMN_MAP,
        temperature_stream="temperature",
        strict=True,
        copy=True,
    )
    LOGGER.info("Step 1 done: measurement preprocessing complete")

    covariates = {
        metric: preprocessed.measurements[metric]
        for metric in COVARIATE_ORDER
        if metric in preprocessed.measurements
    }

    data = create_time_lags_and_merge_features(
        temperature=preprocessed.measurements["temperature"],
        covariates=covariates,
        strict=True,
        copy=True,
    )
    LOGGER.info("Step 2 done: time lags and as-of merges complete")

    data = add_fourier_features(
        data,
        entity_col="encounter_id",
        value_col="value",
        lag_cols=("time_lag_1", "time_lag_2"),
        strict=False,
        copy=True,
    )
    LOGGER.info("Step 3 done: Fourier features merged")

    if args.drop_fourier_arrays:
        array_columns = [
            "fourier_magnitude",
            "fourier_phase",
            "fourier_real_part",
            "fourier_imaginary_part",
        ]
        data = data.drop(columns=[c for c in array_columns if c in data.columns])
        LOGGER.info("Dropped vector-valued Fourier columns")

    data = ffill_impute_within_entity(
        data,
        entity_col="encounter_id",
        time_col="recorded_time",
        columns=None,
        exclude=("value",),
        limit=None,
        strict=True,
        copy=True,
    )
    LOGGER.info("Step 4 done: within-encounter forward-fill imputation complete")

    data = add_statistical_lag_features(
        data,
        entity_col="encounter_id",
        lag1_col="time_lag_1",
        lag2_col="time_lag_2",
        target_lag_col="time_lag_target",
        max_columns=(
            "value",
            "krea",
            "bili",
            "crp",
            "leua",
            "hb",
            "heart_rate",
            "so2",
            "mean_arterial_pressure",
        ),
        value_col="value",
        drop_missing_required=True,
        strict=True,
        copy=True,
    )
    LOGGER.info("Step 5 done: statistical lag features added")

    data = create_temperature_features(
        data,
        entity_col="encounter_id",
        time_col="recorded_time",
        value_col="value",
        lag1_col="time_lag_1",
        lag2_col="time_lag_2",
        target_lag_col="time_lag_target",
        fever_threshold=args.fever_threshold,
        min_required_bt_max_col="time_lag_1_bt_max",
        fillna_value=0.0,
        strict=True,
        copy=True,
    )
    LOGGER.info("Step 6 done: temperature-derived features added")

    data = calculate_elixhauser_score(
        df=data,
        conditions=inputs["conditions_past"],
        reference_xlsx_path=elixhauser_reference,
        subject_col="subject_reference",
        diagnosis_col="conditions",
        return_indicators=True,
        strict=True,
    )
    LOGGER.info("Step 7 done: Elixhauser comorbidities merged")

    data_ml, enriched_full = add_static_covariates(
        data,
        feature_space=DEFAULT_FEATURE_SPACE,
        conditions=inputs["conditions_stay"],
        demographics=inputs["patient_info"],
        fever_threshold=args.fever_threshold,
        min_age_years=args.min_age_years,
        strict=True,
        copy=True,
    )
    LOGGER.info("Step 8 done: static covariates merged; final ML dataset created")

    split = do_train_test_split(
        data_ml,
        feature_space=DEFAULT_FEATURE_SPACE,
        target_col="fever",
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=not args.no_stratify,
        scale=args.scale,
        strict=True,
        copy=True,
    )
    LOGGER.info("Step 9 done: train/test split complete")

    save_outputs(
        output_dir,
        data_ml=data_ml,
        enriched_full=enriched_full,
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
    )

    LOGGER.info("Preprocessing complete")
    LOGGER.info("Final ML rows: %d", len(data_ml))
    LOGGER.info("Train rows: %d | Test rows: %d", len(split.X_train), len(split.X_test))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clinical time-series preprocessing pipeline.")

    parser.add_argument("--data-dir", required=True, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", default="artifacts/preprocessed", help="Directory for output artifacts.")
    parser.add_argument(
        "--elixhauser-reference",
        required=True,
        help="Path to HCUP Elixhauser reference Excel file.",
    )
    parser.add_argument("--csv-sep", default=",", help="Input CSV separator.")
    parser.add_argument("--fever-threshold", type=float, default=38.0)
    parser.add_argument("--min-age-years", type=int, default=18)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=23)
    parser.add_argument("--scale", action="store_true", help="Apply MinMax scaling fit only on training data.")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified train/test split.")
    parser.add_argument(
        "--drop-fourier-arrays",
        action="store_true",
        help="Drop vector-valued FFT columns and keep scalar summaries only.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    setup_logging(parsed_args.verbose)
    run(parsed_args)
