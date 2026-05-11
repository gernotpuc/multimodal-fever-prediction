"""
Generic preprocessing utilities for longitudinal clinical time-series data.

This module is a GitHub-ready refactor of a notebook-style preprocessing script.
It provides reusable, testable functions for:
- timestamp normalization
- measurement stream cleanup
- antibiotic/intervention-window creation
- time-lag feature engineering
- as-of merging of covariates
- forward-fill imputation within encounter only
- statistical and Fourier features
- Elixhauser comorbidity scoring from a reference table
- static covariate creation
- train/test splitting without scaling leakage

The functions are intentionally configurable and avoid hard-coded cohort paths.

Example usage is expected from a project-specific driver script, e.g.:

    from generic_preprocessing_pipeline import (
        preprocess_measurement_streams,
        create_time_lags_and_merge_features,
        add_statistical_lag_features,
        create_temperature_features,
        add_static_covariates,
        do_train_test_split,
    )

    pre = preprocess_measurement_streams(
        measurements={
            "temperature": df_temp,
            "heart_rate": df_heart_rates,
            "so2": df_so2,
            "mean_arterial_pressure": df_bp,
            "crp": df_crp,
            "bili": df_bili,
            "leua": df_leua,
            "krea": df_krea,
            "hb": df_hb,
        },
        interventions=df_ab_groups,
        value_column_map={
            "heart_rate": "heart_rate",
            "so2": "so2",
            "mean_arterial_pressure": "mean_arterial_pressure",
            "crp": "crp",
            "bili": "bili",
            "leua": "leua",
            "krea": "krea",
            "hb": "hb",
        },
    )

    merged = create_time_lags_and_merge_features(
        temperature=pre.measurements["temperature"],
        covariates={k: v for k, v in pre.measurements.items() if k != "temperature"},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import logging
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessedStreams:
    """Container for preprocessed measurement streams and intervention data."""

    measurements: Dict[str, pd.DataFrame]
    interventions: pd.DataFrame


@dataclass(frozen=True)
class ElixhauserResult:
    """Elixhauser comorbidity indicators and unweighted score."""

    indicators: pd.DataFrame
    score: pd.DataFrame


@dataclass(frozen=True)
class SplitResult:
    """Container for train/test split outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_full: pd.DataFrame
    y_full: pd.Series
    scaler: Optional[MinMaxScaler]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str = "dataframe") -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def as_utc_datetime(series: pd.Series) -> pd.Series:
    """Convert a Series to UTC datetime, coercing invalid values to NaT."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def add_recorded_date(
    df: pd.DataFrame,
    *,
    time_col: str = "recorded_time",
    date_col: str = "recorded_date",
    copy: bool = True,
) -> pd.DataFrame:
    """Parse timestamps and add a date column derived from a timestamp column."""
    require_columns(df, [time_col], name="df")
    out = df.copy() if copy else df
    out[time_col] = as_utc_datetime(out[time_col])
    out[date_col] = out[time_col].dt.date
    return out


def make_short_id(value: Any) -> str:
    """Use final path/URL segment as short entity ID."""
    return str(value).rstrip("/").rsplit("/", 1)[-1]


def normalize_subject_reference(series: pd.Series, prefix: str = "Patient/") -> pd.Series:
    """Ensure subject references share a common prefix."""
    s = series.astype(str)
    return pd.Series(np.where(s.str.startswith(prefix), s, prefix + s), index=series.index)


def handle_error_or_warn(message: str, exc: Exception, *, strict: bool) -> None:
    if strict:
        raise exc
    LOGGER.warning("%s: %s", message, exc, exc_info=True)


# -----------------------------------------------------------------------------
# Measurement preprocessing
# -----------------------------------------------------------------------------

def preprocess_measurement_streams(
    measurements: Mapping[str, pd.DataFrame],
    interventions: pd.DataFrame,
    *,
    value_column_map: Optional[Mapping[str, str]] = None,
    lab_streams_to_dropna: Sequence[str] = ("bili", "crp", "hb", "krea", "leua"),
    entity_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    temperature_stream: str = "temperature",
    intervention_col: str = "intervention",
    min_temperature: float = 34.0,
    zero_temperature_is_invalid: bool = True,
    zero_heart_rate_is_missing: bool = True,
    copy: bool = True,
    strict: bool = True,
) -> PreprocessedStreams:
    """
    Preprocess longitudinal measurement streams.

    Steps:
    - optionally drop missing values in configured lab streams
    - normalize timestamps and add recorded_date
    - create an intervention flag on the temperature stream using min/max intervention times
    - rename generic value columns to metric-specific names
    - filter implausible temperature values
    - set heart_rate==0 to NaN if requested

    Parameters
    ----------
    measurements:
        Mapping from stream name to dataframe. Each dataframe should contain at least
        entity_col, time_col, and usually value_col.
    interventions:
        Dataframe containing intervention records with entity_col and time_col.
    value_column_map:
        Mapping from stream name to desired value-column name. The temperature stream
        can be omitted to keep its column as `value`.
    """
    value_column_map = dict(value_column_map or {})
    processed: Dict[str, pd.DataFrame] = {}

    for stream_name, stream_df in measurements.items():
        try:
            out = stream_df.copy() if copy else stream_df
            require_columns(out, [entity_col, time_col], name=stream_name)

            if stream_name in lab_streams_to_dropna and value_col in out.columns:
                out = out.dropna(subset=[value_col])

            out = add_recorded_date(out, time_col=time_col, copy=False)

            if stream_name in value_column_map and value_col in out.columns:
                out = out.rename(columns={value_col: value_column_map[stream_name]})

            processed[stream_name] = out
        except Exception as exc:
            handle_error_or_warn(f"Failed preprocessing stream '{stream_name}'", exc, strict=strict)
            processed[stream_name] = stream_df.copy() if copy else stream_df

    interventions_out = interventions.copy() if copy else interventions

    try:
        require_columns(interventions_out, [entity_col, time_col], name="interventions")
        interventions_out = add_recorded_date(interventions_out, time_col=time_col, copy=False)

        if temperature_stream in processed:
            temp = processed[temperature_stream]
            require_columns(temp, [entity_col, time_col], name=temperature_stream)

            grouped = (
                interventions_out.groupby(entity_col)[time_col]
                .agg(min_timestamp="min", max_timestamp="max")
                .reset_index()
            )
            temp = temp.merge(grouped, on=entity_col, how="left")
            in_window = (temp[time_col] >= temp["min_timestamp"]) & (temp[time_col] <= temp["max_timestamp"])
            temp[intervention_col] = in_window.fillna(False).astype(int)
            temp = temp.drop(columns=["min_timestamp", "max_timestamp"])
            processed[temperature_stream] = temp
    except Exception as exc:
        handle_error_or_warn("Failed creating intervention flag", exc, strict=strict)

    try:
        if temperature_stream in processed:
            temp = processed[temperature_stream]
            require_columns(temp, [value_col], name=temperature_stream)
            mask = temp[value_col] >= min_temperature
            if zero_temperature_is_invalid:
                mask &= temp[value_col] != 0
            processed[temperature_stream] = temp.loc[mask].copy()
    except Exception as exc:
        handle_error_or_warn("Failed filtering temperature stream", exc, strict=strict)

    try:
        if zero_heart_rate_is_missing and "heart_rate" in processed:
            hr = processed["heart_rate"]
            if "heart_rate" in hr.columns:
                hr.loc[hr["heart_rate"] == 0, "heart_rate"] = np.nan
            processed["heart_rate"] = hr
    except Exception as exc:
        handle_error_or_warn("Failed cleaning heart_rate stream", exc, strict=strict)

    return PreprocessedStreams(measurements=processed, interventions=interventions_out)


# -----------------------------------------------------------------------------
# Time-lag windows and as-of merging
# -----------------------------------------------------------------------------

def compute_time_lag_flags_for_group(
    group: pd.DataFrame,
    *,
    time_col: str = "recorded_time",
    intervention_col: str = "intervention",
    lag2_hours: Tuple[float, float] = (0, 24),
    lag1_hours: Tuple[float, float] = (24, 48),
    target_hours: Tuple[float, float] = (48, 72),
    prior_hours: float = 24,
) -> pd.DataFrame:
    """Compute lag-window flags relative to earliest intervention timestamp in one group."""
    intervention_time = group.loc[group[intervention_col] == 1, time_col].min()
    out = pd.DataFrame(index=group.index)

    if pd.isna(intervention_time):
        out["time_lag_2"] = False
        out["time_lag_1"] = False
        out["time_lag_target"] = False
        out["time_lag_prior_24h"] = False
        return out

    seconds = (group[time_col] - intervention_time).dt.total_seconds()

    out["time_lag_2"] = (seconds >= lag2_hours[0] * 3600) & (seconds <= lag2_hours[1] * 3600)
    out["time_lag_1"] = (seconds > lag1_hours[0] * 3600) & (seconds <= lag1_hours[1] * 3600)
    out["time_lag_target"] = (seconds > target_hours[0] * 3600) & (seconds <= target_hours[1] * 3600)

    prior_start = intervention_time - pd.Timedelta(hours=prior_hours)
    out["time_lag_prior_24h"] = (group[time_col] >= prior_start) & (group[time_col] < intervention_time)
    return out


def merge_asof_by_entity(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    value_col: str,
    time_col: str = "recorded_time",
    entity_col: str = "encounter_id",
    direction: str = "backward",
    tolerance: Optional[pd.Timedelta] = None,
    allow_exact_matches: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """As-of merge within entity, dropping null merge keys and sorting safely."""
    require_columns(base, [entity_col, time_col], name="base")
    require_columns(other, [entity_col, time_col, value_col], name=f"other({value_col})")

    base2 = base.copy()
    other2 = other[[entity_col, time_col, value_col]].copy()

    base2 = base2.loc[base2[time_col].notna()].copy()
    other2 = other2.loc[other2[time_col].notna()].copy()

    if other2.empty:
        message = f"All rows in other({value_col}) have null '{time_col}'. Merge skipped."
        if strict:
            raise ValueError(message)
        LOGGER.warning(message)
        return base.copy()

    base_sorted = base2.sort_values([time_col, entity_col])
    other_sorted = other2.sort_values([time_col, entity_col])

    merged = pd.merge_asof(
        base_sorted,
        other_sorted,
        on=time_col,
        by=entity_col,
        direction=direction,
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
    )

    return merged.sort_values([entity_col, time_col])


def create_time_lags_and_merge_features(
    temperature: pd.DataFrame,
    covariates: Mapping[str, pd.DataFrame],
    *,
    entity_col: str = "encounter_id",
    subject_col: str = "subject_reference",
    time_col: str = "recorded_time",
    value_col: str = "value",
    intervention_col: str = "intervention",
    forward_fill_cols: Sequence[str] = ("crp", "bili", "krea", "leua", "hb", "heart_rate"),
    asof_direction: str = "backward",
    asof_tolerance: Optional[pd.Timedelta] = None,
    copy: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Create lag flags on the temperature timeline and as-of merge covariates.

    `covariates` should map metric names to dataframes. Each dataframe should contain
    entity_col, time_col, and a value column with the same name as the metric key.
    """
    temp = temperature.copy() if copy else temperature

    try:
        require_columns(temp, [entity_col, subject_col, time_col, value_col, intervention_col], name="temperature")
        temp[time_col] = as_utc_datetime(temp[time_col])
    except Exception as exc:
        handle_error_or_warn("Temperature schema/timestamp validation failed", exc, strict=strict)

    for metric_name, cov_df in covariates.items():
        try:
            require_columns(cov_df, [entity_col, time_col, metric_name], name=metric_name)
            if copy:
                covariates = dict(covariates)
                covariates[metric_name] = cov_df.copy()
            covariates[metric_name][time_col] = as_utc_datetime(covariates[metric_name][time_col])
        except Exception as exc:
            handle_error_or_warn(f"Covariate validation failed for '{metric_name}'", exc, strict=strict)

    lag_flags = (
        temp.groupby(entity_col, group_keys=False)
        .apply(lambda g: compute_time_lag_flags_for_group(g, time_col=time_col, intervention_col=intervention_col))
        .sort_index()
    )
    temp = temp.join(lag_flags)

    base_cols = [
        entity_col,
        subject_col,
        time_col,
        value_col,
        intervention_col,
        "time_lag_1",
        "time_lag_2",
        "time_lag_target",
    ]
    if "time_lag_prior_24h" in temp.columns:
        base_cols.append("time_lag_prior_24h")

    base = temp[base_cols].copy()

    for metric_name, cov_df in covariates.items():
        try:
            base = merge_asof_by_entity(
                base,
                cov_df,
                value_col=metric_name,
                time_col=time_col,
                entity_col=entity_col,
                direction=asof_direction,
                tolerance=asof_tolerance,
                strict=strict,
            )
        except Exception as exc:
            handle_error_or_warn(f"As-of merge failed for '{metric_name}'", exc, strict=strict)

    for column in forward_fill_cols:
        if column in base.columns:
            base[column] = base.groupby(entity_col)[column].ffill()

    return base


# -----------------------------------------------------------------------------
# Imputation
# -----------------------------------------------------------------------------

def ffill_impute_within_entity(
    df: pd.DataFrame,
    *,
    entity_col: str = "encounter_id",
    time_col: Optional[str] = "recorded_time",
    columns: Optional[Sequence[str]] = None,
    exclude: Sequence[str] = ("value",),
    limit: Optional[int] = None,
    copy: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Forward-fill selected columns within each entity only."""
    try:
        require_columns(df, [entity_col], name="df")
    except Exception as exc:
        handle_error_or_warn(f"Missing entity column '{entity_col}'", exc, strict=strict)
        return df.copy() if copy else df

    out = df.copy() if copy else df

    if columns is None:
        numeric_columns = out.select_dtypes(include=["number"]).columns.tolist()
        columns_to_fill = [column for column in numeric_columns if column not in set(exclude)]
    else:
        columns_to_fill = list(columns)

    missing = [column for column in columns_to_fill if column not in out.columns]
    if missing:
        message = f"Skipping missing columns for forward fill: {missing}"
        if strict:
            raise ValueError(message)
        LOGGER.warning(message)

    columns_to_fill = [column for column in columns_to_fill if column in out.columns]
    if not columns_to_fill:
        return out

    if time_col is not None and time_col in out.columns:
        out = out.sort_values([entity_col, time_col])

    out[columns_to_fill] = out.groupby(entity_col)[columns_to_fill].ffill(limit=limit)
    return out


# -----------------------------------------------------------------------------
# Statistical lag features
# -----------------------------------------------------------------------------

def add_statistical_lag_features(
    df: pd.DataFrame,
    *,
    entity_col: str = "encounter_id",
    lag1_col: str = "time_lag_1",
    lag2_col: str = "time_lag_2",
    target_lag_col: str = "time_lag_target",
    max_columns: Sequence[str] = (
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
    value_col: str = "value",
    drop_missing_required: bool = True,
    copy: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Add per-entity max/mean features over configured lag windows."""
    out = df.copy() if copy else df

    try:
        require_columns(out, [entity_col, lag1_col, lag2_col, target_lag_col, value_col], name="df")
        missing_max_cols = [column for column in max_columns if column not in out.columns]
        if missing_max_cols:
            raise ValueError(f"Missing columns listed in max_columns: {missing_max_cols}")
    except Exception as exc:
        handle_error_or_warn("Schema validation failed for lag feature computation", exc, strict=strict)
        return out

    out[lag1_col] = out[lag1_col].astype(bool)
    out[lag2_col] = out[lag2_col].astype(bool)
    out[target_lag_col] = out[target_lag_col].astype(bool)

    lag1_max = (
        out.loc[out[lag1_col], [entity_col, *max_columns]]
        .groupby(entity_col, as_index=False)
        .max()
        .rename(columns={column: f"{lag1_col}_{column}_max" for column in max_columns})
    )
    lag2_max = (
        out.loc[out[lag2_col], [entity_col, *max_columns]]
        .groupby(entity_col, as_index=False)
        .max()
        .rename(columns={column: f"{lag2_col}_{column}_max" for column in max_columns})
    )
    out = out.merge(lag1_max, on=entity_col, how="left").merge(lag2_max, on=entity_col, how="left")

    lag1_mean = (
        out.loc[out[lag1_col], [entity_col, value_col]]
        .groupby(entity_col, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: f"{lag1_col}_bt_mean"})
    )
    lag2_mean = (
        out.loc[out[lag2_col], [entity_col, value_col]]
        .groupby(entity_col, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: f"{lag2_col}_bt_mean"})
    )
    target_max = (
        out.loc[out[target_lag_col], [entity_col, value_col]]
        .groupby(entity_col, as_index=False)[value_col]
        .max()
        .rename(columns={value_col: f"{target_lag_col}_bt_max"})
    )
    out = out.merge(lag1_mean, on=entity_col, how="left")
    out = out.merge(lag2_mean, on=entity_col, how="left")
    out = out.merge(target_max, on=entity_col, how="left")

    rename_map = {
        f"{lag1_col}_{value_col}_max": f"{lag1_col}_bt_max",
        f"{lag2_col}_{value_col}_max": f"{lag2_col}_bt_max",
    }
    out = out.rename(columns=rename_map)

    if drop_missing_required:
        out = out.dropna(subset=[f"{lag1_col}_bt_max", f"{lag2_col}_bt_max"], how="any")

    return out


# -----------------------------------------------------------------------------
# Temperature/fever feature engineering
# -----------------------------------------------------------------------------

def time_of_day_bucket(hours: pd.Series) -> pd.Series:
    """Bucket hour-of-day into night, morning, afternoon, evening."""
    return pd.cut(
        hours,
        bins=[-1, 4, 10, 16, 22, 24],
        labels=["night", "morning", "afternoon", "evening", "night"],
        right=True,
        include_lowest=True,
        ordered=False,
    )


def compute_fever_stats_for_lag(
    df: pd.DataFrame,
    *,
    lag_col: str,
    entity_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    fever_threshold: float = 38.0,
    valid_temperature_threshold: float = 35.0,
) -> pd.DataFrame:
    """Compute fever summary statistics within one lag window."""
    df_lag = df.loc[df[lag_col]].copy()
    columns = [
        "fever_percent",
        "fever_points",
        "fever_change_pct",
        "last_bt",
        "fever_diff",
        "fever_evening",
        "fever_afternoon",
        "fever_morning",
        "fever_night",
        "fever_std",
        "fever_range",
    ]

    if df_lag.empty:
        return pd.DataFrame(columns=columns).set_index(pd.Index([], name=entity_col))

    df_lag = df_lag.sort_values([entity_col, time_col])
    grouped = df_lag.groupby(entity_col)[value_col]

    fever_percent = grouped.apply(lambda x: float((x >= fever_threshold).mean() * 100.0))
    fever_points = grouped.apply(lambda x: int((x >= fever_threshold).sum()))
    first_value = grouped.first()
    last_value = grouped.last()
    fever_change_pct = (last_value - first_value) / first_value.replace(0, np.nan) * 100.0
    fever_diff = last_value - first_value
    fever_std = grouped.std()
    fever_range = grouped.max() - grouped.min()

    df_lag["time_of_day"] = time_of_day_bucket(df_lag[time_col].dt.hour)

    def tod_max(label: str) -> pd.Series:
        sub = df_lag.loc[df_lag["time_of_day"] == label].groupby(entity_col)[value_col]
        return sub.apply(lambda x: x.max() if (x >= valid_temperature_threshold).any() else np.nan)

    out = pd.DataFrame(
        {
            "fever_percent": fever_percent,
            "fever_points": fever_points,
            "fever_change_pct": fever_change_pct,
            "last_bt": last_value,
            "fever_diff": fever_diff,
            "fever_evening": tod_max("evening"),
            "fever_afternoon": tod_max("afternoon"),
            "fever_morning": tod_max("morning"),
            "fever_night": tod_max("night"),
            "fever_std": fever_std,
            "fever_range": fever_range,
        }
    )
    out.index.name = entity_col
    return out


def compute_cross_lag_features(
    df: pd.DataFrame,
    *,
    entity_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    lag1_col: str = "time_lag_1",
    lag2_col: str = "time_lag_2",
) -> pd.DataFrame:
    """Compute skewness and change over union of lag1/lag2 windows."""
    df_union = df.loc[df[lag1_col] | df[lag2_col]].sort_values([entity_col, time_col])

    if df_union.empty:
        return pd.DataFrame(columns=["fever_change_lag_all", "skewness_lag_all"]).set_index(pd.Index([], name=entity_col))

    skewness = df_union.groupby(entity_col)[value_col].skew().rename("skewness_lag_all")

    def change(group: pd.DataFrame) -> float:
        lag1_values = group.loc[group[lag1_col], value_col]
        lag2_values = group.loc[group[lag2_col], value_col]
        if lag1_values.empty or lag2_values.empty:
            return np.nan
        denominator = lag1_values.iloc[0]
        if denominator == 0 or pd.isna(denominator):
            return np.nan
        return float(((lag1_values.iloc[0] - lag2_values.iloc[-1]) / denominator) * 100.0)

    change_series = df_union.groupby(entity_col).apply(change).rename("fever_change_lag_all")
    out = pd.concat([change_series, skewness], axis=1)
    out.index.name = entity_col
    return out


def compute_trend_per_entity(
    df: pd.DataFrame,
    *,
    entity_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    use_mask: Optional[pd.Series] = None,
) -> pd.Series:
    """Compute simple linear trend of value over time per entity."""
    data = df.copy()
    if use_mask is not None:
        data = data.loc[use_mask]

    if data.empty:
        return pd.Series(dtype="float64", name="trend")

    data = data.sort_values([entity_col, time_col])

    def trend(group: pd.DataFrame) -> float:
        values = pd.to_numeric(group[value_col], errors="coerce").to_numpy()
        times = group[time_col].astype("int64") / 1e9
        times = times - times.min()
        mask = np.isfinite(values) & np.isfinite(times)
        values = values[mask]
        times = times[mask]
        if len(values) < 2:
            return np.nan
        slope, _intercept = np.polyfit(times, values, deg=1)
        return float(slope)

    return data.groupby(entity_col, group_keys=False).apply(trend).rename("trend")


def create_temperature_features(
    df: pd.DataFrame,
    *,
    entity_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    lag1_col: str = "time_lag_1",
    lag2_col: str = "time_lag_2",
    target_lag_col: str = "time_lag_target",
    fever_threshold: float = 38.0,
    min_required_bt_max_col: str = "time_lag_1_bt_max",
    fillna_value: float = 0.0,
    copy: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Create temperature-derived statistical features and analytic-cohort filters."""
    out = df.copy() if copy else df

    try:
        require_columns(out, [entity_col, time_col, value_col, lag1_col, lag2_col, target_lag_col], name="df")
        if min_required_bt_max_col not in out.columns:
            raise ValueError(f"Missing required column '{min_required_bt_max_col}'.")
        out[time_col] = as_utc_datetime(out[time_col])
    except Exception as exc:
        handle_error_or_warn("Schema validation/datetime parsing failed", exc, strict=strict)
        return out

    out[lag1_col] = out[lag1_col].astype(bool)
    out[lag2_col] = out[lag2_col].astype(bool)
    out[target_lag_col] = out[target_lag_col].astype(bool)

    lag1_stats = compute_fever_stats_for_lag(
        out,
        lag_col=lag1_col,
        entity_col=entity_col,
        time_col=time_col,
        value_col=value_col,
        fever_threshold=fever_threshold,
    ).add_prefix("lag1_")
    lag2_stats = compute_fever_stats_for_lag(
        out,
        lag_col=lag2_col,
        entity_col=entity_col,
        time_col=time_col,
        value_col=value_col,
        fever_threshold=fever_threshold,
    ).add_prefix("lag2_")
    cross_stats = compute_cross_lag_features(
        out,
        entity_col=entity_col,
        time_col=time_col,
        value_col=value_col,
        lag1_col=lag1_col,
        lag2_col=lag2_col,
    )

    out = out.merge(lag1_stats, left_on=entity_col, right_index=True, how="left")
    out = out.merge(lag2_stats, left_on=entity_col, right_index=True, how="left")
    out = out.merge(cross_stats, left_on=entity_col, right_index=True, how="left")

    rename_map = {
        "lag1_fever_points": "fever_points_lag_1",
        "lag1_fever_percent": "fever_percent_lag_1",
        "lag1_fever_change_pct": "fever_change_lag_1",
        "lag1_last_bt": "last_bt_fever_lag_1",
        "lag1_fever_diff": "fever_diff_lag_1",
        "lag1_fever_evening": "fever_lag_1_evening",
        "lag1_fever_afternoon": "fever_lag_1_afternoon",
        "lag1_fever_morning": "fever_lag_1_morning",
        "lag1_fever_night": "fever_lag_1_night",
        "lag1_fever_std": "fever_variability_lag_1",
        "lag1_fever_range": "fever_range_lag_1",
        "lag2_fever_points": "fever_points_lag_2",
        "lag2_fever_percent": "fever_percent_lag_2",
        "lag2_fever_change_pct": "fever_change_lag_2",
        "lag2_last_bt": "last_bt_fever_lag_2",
        "lag2_fever_diff": "fever_diff_lag_2",
        "lag2_fever_evening": "fever_lag_2_evening",
        "lag2_fever_afternoon": "fever_lag_2_afternoon",
        "lag2_fever_morning": "fever_lag_2_morning",
        "lag2_fever_night": "fever_lag_2_night",
        "lag2_fever_std": "fever_variability_lag_2",
        "lag2_fever_range": "fever_range_lag_2",
    }
    out = out.rename(columns=rename_map)

    engineered_cols = list(rename_map.values()) + ["fever_change_lag_all", "skewness_lag_all"]
    for column in engineered_cols:
        if column in out.columns:
            out[column] = out[column].fillna(fillna_value)

    out = out.loc[out[min_required_bt_max_col] >= fever_threshold]

    has_target = out.groupby(entity_col)[target_lag_col].transform("sum") >= 1
    has_lag1 = out.groupby(entity_col)[lag1_col].transform("sum") >= 1
    out = out.loc[has_target & has_lag1]

    out["time_of_day"] = pd.cut(
        out[time_col].dt.hour,
        bins=[-1, 4, 10, 16, 23],
        labels=["night", "morning", "afternoon", "evening"],
        include_lowest=True,
    ).astype(str)

    mask = out[lag1_col] | out[lag2_col]
    trend = compute_trend_per_entity(
        out,
        entity_col=entity_col,
        time_col=time_col,
        value_col=value_col,
        use_mask=mask,
    )
    out = out.merge(trend, left_on=entity_col, right_index=True, how="left")

    return out


# -----------------------------------------------------------------------------
# Fourier features
# -----------------------------------------------------------------------------

FOURIER_FEATURE_COLUMNS = [
    "fourier_magnitude",
    "fourier_phase",
    "fourier_real_part",
    "fourier_imaginary_part",
    "fourier_mean_magnitude",
    "fourier_std_dev_magnitude",
    "fourier_max_magnitude",
    "fourier_min_magnitude",
]


def create_fourier_features_for_group(
    group: pd.DataFrame,
    *,
    value_col: str = "value",
    lag_cols: Tuple[str, ...] = ("time_lag_1", "time_lag_2"),
    strict: bool = True,
) -> pd.Series:
    """Compute Fourier-based features for one grouped time series."""

    def fail(message: str, exc: Optional[Exception] = None) -> pd.Series:
        if exc is not None and strict:
            raise exc
        if exc is not None:
            LOGGER.warning("%s: %s", message, exc, exc_info=True)
        return pd.Series([np.nan] * len(FOURIER_FEATURE_COLUMNS), index=FOURIER_FEATURE_COLUMNS)

    try:
        require_columns(group, [value_col, *lag_cols], name="group")
    except Exception as exc:
        return fail("Missing required columns for Fourier features", exc)

    mask = pd.Series(False, index=group.index)
    for column in lag_cols:
        mask = mask | group[column].astype(bool)

    values = pd.to_numeric(group.loc[mask, value_col], errors="coerce").dropna()
    if values.empty:
        return fail("No valid values in selected lag window")

    fft = np.fft.fft(values.to_numpy())
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    real_part = np.real(fft)
    imaginary_part = np.imag(fft)

    return pd.Series(
        [
            magnitude,
            phase,
            real_part,
            imaginary_part,
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.max(magnitude)),
            float(np.min(magnitude)),
        ],
        index=FOURIER_FEATURE_COLUMNS,
    )


def add_fourier_features(
    df: pd.DataFrame,
    *,
    entity_col: str = "encounter_id",
    value_col: str = "value",
    lag_cols: Tuple[str, ...] = ("time_lag_1", "time_lag_2"),
    copy: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Compute Fourier features per entity and merge them back."""
    out = df.copy() if copy else df
    require_columns(out, [entity_col, value_col, *lag_cols], name="df")

    features = out.groupby(entity_col).apply(
        lambda group: create_fourier_features_for_group(
            group,
            value_col=value_col,
            lag_cols=lag_cols,
            strict=strict,
        )
    )
    return out.merge(features, left_on=entity_col, right_index=True, how="left")


# -----------------------------------------------------------------------------
# Elixhauser comorbidity scoring
# -----------------------------------------------------------------------------

def normalize_icd10(code: object) -> str:
    """Normalize ICD-10 codes by uppercasing and removing non-alphanumerics."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ""
    return re.sub(r"[^0-9A-Z]", "", str(code).upper().strip())


def calculate_elixhauser_score(
    df: pd.DataFrame,
    conditions: pd.DataFrame,
    *,
    reference_xlsx_path: Union[str, Path],
    subject_col: str = "subject_reference",
    diagnosis_col: str = "conditions",
    reference_code_col: str = "ICD-10-CM Diagnosis",
    reference_meta_columns: Sequence[str] = ("ICD-10-CM Code Description", "# Comorbidities"),
    return_indicators: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Calculate an unweighted Elixhauser score from diagnosis codes and a reference table.

    The reference Excel is expected to contain one ICD-code column plus one or more
    binary comorbidity indicator columns.
    """
    out = df.copy()

    try:
        require_columns(out, [subject_col], name="df")
        require_columns(conditions, [subject_col, diagnosis_col], name="conditions")
        reference = pd.read_excel(reference_xlsx_path)
        require_columns(reference, [reference_code_col], name="reference")
    except Exception as exc:
        handle_error_or_warn("Elixhauser input validation failed", exc, strict=strict)
        return out

    meta_like = set(reference_meta_columns) | {reference_code_col}
    comorbidity_columns = [column for column in reference.columns if column not in meta_like]
    if not comorbidity_columns:
        message = f"No comorbidity indicator columns found in reference: {reference.columns.tolist()}"
        if strict:
            raise ValueError(message)
        LOGGER.warning(message)
        return out

    reference_small = reference[[reference_code_col] + comorbidity_columns].copy()
    reference_small["icd_norm"] = reference_small[reference_code_col].map(normalize_icd10)
    reference_small = reference_small.loc[reference_small["icd_norm"] != ""].drop_duplicates(subset=["icd_norm"])

    conds = conditions[[subject_col, diagnosis_col]].dropna().copy()
    conds["icd_norm"] = conds[diagnosis_col].map(normalize_icd10)
    conds = conds.loc[conds["icd_norm"] != ""].copy()

    joined = conds.merge(reference_small.drop(columns=[reference_code_col]), on="icd_norm", how="left")
    joined[comorbidity_columns] = joined[comorbidity_columns].fillna(0)

    for column in comorbidity_columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0).astype(int)

    indicators = joined.groupby(subject_col, as_index=True)[comorbidity_columns].max()
    score = indicators.sum(axis=1).rename("elixhauser_score").astype(int).to_frame()

    out = out.merge(score, left_on=subject_col, right_index=True, how="left")
    out["elixhauser_score"] = out["elixhauser_score"].fillna(0).astype(int)

    if return_indicators:
        out = out.merge(indicators, left_on=subject_col, right_index=True, how="left")
        out[comorbidity_columns] = out[comorbidity_columns].fillna(0).astype(int)

    return out


# -----------------------------------------------------------------------------
# Static covariates and target creation
# -----------------------------------------------------------------------------

def compile_prefix_regex(prefixes: Sequence[str]) -> re.Pattern:
    escaped = "|".join(re.escape(prefix) for prefix in prefixes)
    return re.compile(rf"^(?:{escaped})(?:\.|$)")


def categorize_condition_codes(
    codes: Sequence[str],
    *,
    leukemia_regex: re.Pattern,
    hematologic_regex: re.Pattern,
    solid_regex: re.Pattern,
) -> str:
    """Categorize ICD codes using priority: leukemia > hematologic > solid > other."""
    hematologic = False
    solid = False

    for code in codes:
        code_str = str(code)
        if leukemia_regex.match(code_str):
            return "leuk"
        if hematologic_regex.match(code_str):
            hematologic = True
        if solid_regex.match(code_str):
            solid = True

    if hematologic:
        return "hemat"
    if solid:
        return "solid"
    return "other"


def add_static_covariates(
    data: pd.DataFrame,
    *,
    feature_space: Sequence[str],
    conditions: pd.DataFrame,
    demographics: pd.DataFrame,
    entity_col: str = "encounter_id",
    subject_col: str = "subject_reference",
    time_col: str = "recorded_time",
    diagnosis_col: str = "conditions",
    birth_date_col: str = "birth_date",
    sex_col: str = "sex",
    target_col: str = "fever",
    target_temperature_col: str = "time_lag_target_bt_max",
    target_lag_col: str = "time_lag_target",
    lag1_col: str = "time_lag_1",
    fever_threshold: float = 38.0,
    min_age_years: int = 18,
    leukemia_prefixes: Sequence[str] = ("C91", "C92", "C93", "C94", "C95"),
    hematologic_prefixes: Sequence[str] = ("C81", "C82", "C83", "C84", "C85", "C86", "C88", "C90", "C96"),
    solid_prefixes: Sequence[str] = (
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14",
        "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C30", "C31", "C32",
        "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C43", "C44", "C45", "C46", "C47", "C48", "C49",
        "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C64", "C65",
        "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C77", "C78", "C79", "C80",
    ),
    copy: bool = True,
    strict: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add condition category, sex, age, length-stay proxy, and target label."""
    out = data.copy() if copy else data
    conds = conditions.copy() if copy else conditions
    demo = demographics.copy() if copy else demographics

    try:
        require_columns(out, [entity_col, subject_col, lag1_col, time_col, target_lag_col, target_temperature_col], name="data")
        require_columns(conds, [subject_col, diagnosis_col], name="conditions")
        require_columns(demo, [subject_col, birth_date_col, sex_col], name="demographics")
    except Exception as exc:
        handle_error_or_warn("Static-covariate schema validation failed", exc, strict=strict)
        return pd.DataFrame(), pd.DataFrame()

    out[time_col] = as_utc_datetime(out[time_col])
    demo[birth_date_col] = as_utc_datetime(demo[birth_date_col])
    out[target_lag_col] = out[target_lag_col].astype(bool)
    out[lag1_col] = out[lag1_col].astype(bool)

    leukemia_regex = compile_prefix_regex(leukemia_prefixes)
    hematologic_regex = compile_prefix_regex(hematologic_prefixes)
    solid_regex = compile_prefix_regex(solid_prefixes)

    conds_small = conds[[subject_col, diagnosis_col]].dropna().copy()
    grouped_codes = conds_small.groupby(subject_col)[diagnosis_col].apply(list)
    cond_category = grouped_codes.apply(
        lambda codes: categorize_condition_codes(
            codes,
            leukemia_regex=leukemia_regex,
            hematologic_regex=hematologic_regex,
            solid_regex=solid_regex,
        )
    ).rename("cond_category")

    out = out.merge(cond_category, left_on=subject_col, right_index=True, how="left")
    out["cond_category"] = out["cond_category"].fillna("other")
    out = pd.get_dummies(out, columns=["cond_category"], prefix="cond")

    out[subject_col] = normalize_subject_reference(out[subject_col])
    demo[subject_col] = normalize_subject_reference(demo[subject_col])
    out = out.merge(demo[[subject_col, birth_date_col, sex_col]], on=subject_col, how="left")
    out = pd.get_dummies(out, columns=[sex_col], prefix=sex_col)

    min_recorded_time = out.groupby(entity_col)[time_col].min().rename("recorded_time_min")
    out = out.merge(min_recorded_time, on=entity_col, how="left")
    out["age_at_start"] = np.floor((out["recorded_time_min"] - out[birth_date_col]).dt.days / 365.25)
    out = out.loc[out["age_at_start"] >= min_age_years].copy()
    out["age"] = np.floor((out[time_col] - out[birth_date_col]).dt.days / 365.25)

    out[target_col] = (pd.to_numeric(out[target_temperature_col], errors="coerce") >= fever_threshold).astype(bool)

    min_non_target = out.loc[~out[target_lag_col]].groupby(entity_col)[time_col].min()
    min_target = out.loc[out[target_lag_col]].groupby(entity_col)[time_col].min()
    length_stay = (min_target - min_non_target).dt.days.rename("length_stay")
    out = out.merge(length_stay, on=entity_col, how="left")

    meta_columns = [entity_col, subject_col, target_col]
    output_columns = meta_columns + list(feature_space)
    missing_output = [column for column in output_columns if column not in out.columns]
    if missing_output:
        if strict:
            raise KeyError(f"Requested output columns missing: {missing_output}")
        LOGGER.warning("Requested output columns missing and will be skipped: %s", missing_output)
        output_columns = [column for column in output_columns if column in out.columns]

    subset = out[output_columns].copy()
    dedup_columns = [column for column in subset.columns if column != "age"]
    subset = subset.drop_duplicates(subset=dedup_columns)

    return subset, out


# -----------------------------------------------------------------------------
# Train/test split
# -----------------------------------------------------------------------------

def do_train_test_split(
    data: pd.DataFrame,
    *,
    feature_space: Sequence[str],
    target_col: str = "fever",
    test_size: float = 0.20,
    random_state: int = 23,
    stratify: bool = True,
    scale: Union[bool, str] = False,
    copy: bool = True,
    strict: bool = True,
) -> SplitResult:
    """Split into train/test sets and optionally fit MinMaxScaler on training data only."""
    try:
        df = data.copy() if copy else data
        require_columns(df, list(feature_space), name="data")
        require_columns(df, [target_col], name="data")

        X_full = df.loc[:, feature_space]
        y_full = df.loc[:, target_col]

        if stratify:
            unique_values = pd.Series(y_full).dropna().unique()
            if len(unique_values) < 2:
                raise ValueError(f"Cannot stratify: target '{target_col}' has <2 unique values: {unique_values}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=test_size,
            stratify=y_full if stratify else None,
            random_state=random_state,
        )

        scale_flag = scale if isinstance(scale, bool) else str(scale).strip().lower() in {"yes", "true", "1"}
        scaler: Optional[MinMaxScaler] = None

        if scale_flag:
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        return SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_full=X_full,
            y_full=y_full,
            scaler=scaler,
        )
    except Exception as exc:
        if strict:
            raise
        LOGGER.warning("Train/test split failed: %s", exc, exc_info=True)
        empty_X = pd.DataFrame(index=data.index)
        empty_y = pd.Series(index=data.index, dtype="float64")
        return SplitResult(empty_X, empty_X, empty_y, empty_y, empty_X, empty_y, None)
