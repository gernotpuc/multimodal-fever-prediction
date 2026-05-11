"""
Generic Streamlit validation app for case-vignette review studies.

This app lets reviewers inspect one case at a time using:
- longitudinal temperature/vital-sign time series
- tabular/static features
- clinical notes or other text documents
- a binary assessment form with confidence/probability and comments

It is designed to be GitHub-ready:
- no hard-coded institution-specific labels
- configurable paths via Streamlit secrets, environment variables, or CLI-style defaults
- generic column mappings
- optional logo
- downloadable and locally persisted reviewer responses

Run:
    streamlit run app.py

Optional `.streamlit/secrets.toml`:
    app_title = "Case Vignette Validation Study"
    study_task = "Assess whether the target outcome will occur."
    data_dir = "data/validation"
    notes_dir = "data/validation/notes"
    responses_csv = "outputs/validation_responses.csv"
    logo_path = "assets/logo.png"

Expected default files inside `data_dir`:
    train_data.csv                  # longitudinal temperature data, sep=';'
    data_tabular.csv                # static/tabular features, sep=';'
    heart_rates.csv                 # optional, sep=';'
    mean_arterial_pressure.csv      # optional, sep=';'
    so2.csv                         # optional, sep=';'

Expected default note filenames:
    Encounter_<id>.txt
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    app_title: str
    page_icon: str
    layout: str
    study_task: str
    study_context: str
    instructions: str
    data_dir: Path
    notes_dir: Path
    responses_csv: Path
    logo_path: Optional[Path]
    csv_sep: str
    note_prefix: str
    note_suffix: str
    id_column: str
    time_column: str
    value_column: str
    temperature_file: str
    tabular_file: str
    heart_rate_file: str
    map_file: str
    so2_file: str
    fever_threshold: float
    temperature_clip_low: float
    temperature_clip_high: float
    target_question: str
    positive_label: str
    negative_label: str


def get_secret_or_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read config from Streamlit secrets, then environment, then default."""
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name.upper(), default)


def load_config() -> AppConfig:
    data_dir = Path(get_secret_or_env("data_dir", "data/validation"))
    notes_dir = Path(get_secret_or_env("notes_dir", str(data_dir / "notes")))
    responses_csv = Path(get_secret_or_env("responses_csv", "outputs/validation_responses.csv"))
    logo_raw = get_secret_or_env("logo_path", None)

    default_instructions = f"""
**Task.** Review the case information and answer the assessment question.

**What to do**
1. Review the time series, tabular values, and notes.
2. Choose the most appropriate binary decision.
3. Enter your confidence/probability from 0 to 1.
4. Optionally add comments.
5. Save your assessment and download the CSV when finished.
""".strip()

    return AppConfig(
        app_title=get_secret_or_env("app_title", "Case Vignette Validation Study") or "Case Vignette Validation Study",
        page_icon=get_secret_or_env("page_icon", "🧪") or "🧪",
        layout=get_secret_or_env("layout", "wide") or "wide",
        study_task=get_secret_or_env("study_task", "Assess the target outcome for each case.") or "Assess the target outcome for each case.",
        study_context=get_secret_or_env("study_context", "") or "",
        instructions=get_secret_or_env("instructions", default_instructions) or default_instructions,
        data_dir=data_dir,
        notes_dir=notes_dir,
        responses_csv=responses_csv,
        logo_path=Path(logo_raw) if logo_raw else None,
        csv_sep=get_secret_or_env("csv_sep", ";") or ";",
        note_prefix=get_secret_or_env("note_prefix", "Encounter_") or "Encounter_",
        note_suffix=get_secret_or_env("note_suffix", ".txt") or ".txt",
        id_column=get_secret_or_env("id_column", "encounter_id") or "encounter_id",
        time_column=get_secret_or_env("time_column", "recorded_time") or "recorded_time",
        value_column=get_secret_or_env("value_column", "value") or "value",
        temperature_file=get_secret_or_env("temperature_file", "train_data.csv") or "train_data.csv",
        tabular_file=get_secret_or_env("tabular_file", "data_tabular.csv") or "data_tabular.csv",
        heart_rate_file=get_secret_or_env("heart_rate_file", "heart_rates.csv") or "heart_rates.csv",
        map_file=get_secret_or_env("map_file", "mean_arterial_pressure.csv") or "mean_arterial_pressure.csv",
        so2_file=get_secret_or_env("so2_file", "so2.csv") or "so2.csv",
        fever_threshold=float(get_secret_or_env("fever_threshold", "38.0") or 38.0),
        temperature_clip_low=float(get_secret_or_env("temperature_clip_low", "30.0") or 30.0),
        temperature_clip_high=float(get_secret_or_env("temperature_clip_high", "45.0") or 45.0),
        target_question=get_secret_or_env(
            "target_question",
            "Will the target outcome occur for this case?",
        ) or "Will the target outcome occur for this case?",
        positive_label=get_secret_or_env("positive_label", "Yes") or "Yes",
        negative_label=get_secret_or_env("negative_label", "No") or "No",
    )


CONFIG = load_config()
st.set_page_config(page_title=CONFIG.app_title, page_icon=CONFIG.page_icon, layout=CONFIG.layout)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class CaseBundle:
    case_id: str
    age: Optional[float]
    sex: str
    comorbidity_score: Optional[float]
    labs_table: pd.DataFrame
    vitals_table: pd.DataFrame
    temperature_df: pd.DataFrame
    vital_curves: Dict[str, pd.DataFrame]
    notes_text: str


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def normalize_entity_id(entity_id: object) -> str:
    """Normalize entity IDs to a bare string ID."""
    if entity_id is None or pd.isna(entity_id):
        return ""
    value = str(entity_id).strip()
    for prefix in ("Encounter/", "Encounter_", "Patient/", "Patient_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    if "/" in value:
        value = value.rstrip("/").rsplit("/", 1)[-1]
    return value


def parse_boolish(series: pd.Series) -> pd.Series:
    """Parse common boolean encodings."""
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})


def safe_numeric(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=True)
def load_measurements(
    csv_path: str,
    *,
    id_column: str,
    time_column: str,
    value_column: str,
    csv_sep: str,
    clip_low: Optional[float] = None,
    clip_high: Optional[float] = None,
    remove_zero: bool = True,
) -> pd.DataFrame:
    """Load a longitudinal measurement CSV."""
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(columns=[id_column, time_column, value_column])

    df = pd.read_csv(path, sep=csv_sep)
    required = [id_column, time_column, value_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df[id_column] = df[id_column].map(normalize_entity_id)
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce", utc=True).dt.tz_localize(None)
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=[id_column, time_column, value_column])

    if remove_zero:
        df = df[df[value_column] != 0]

    if clip_low is not None or clip_high is not None:
        df[value_column] = df[value_column].clip(lower=clip_low, upper=clip_high)

    return df


@st.cache_data(show_spinner=True)
def load_tabular(csv_path: str, *, id_column: str, csv_sep: str) -> pd.DataFrame:
    """Load case-level tabular data and normalize known columns when present."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing tabular file: {path}")

    df = pd.read_csv(path, sep=csv_sep)
    if id_column not in df.columns:
        raise ValueError(f"{path} is missing ID column '{id_column}'.")

    df[id_column] = df[id_column].map(normalize_entity_id)

    if "sex_male" in df.columns:
        parsed = parse_boolish(df["sex_male"])
        df["sex_male"] = parsed.fillna(False)

    return df


@st.cache_data(show_spinner=True)
def list_note_entities(notes_dir: str, note_prefix: str, note_suffix: str) -> List[str]:
    """List entity IDs with note files."""
    path = Path(notes_dir)
    if not path.exists():
        return []

    ids: List[str] = []
    for note_path in path.glob(f"{note_prefix}*{note_suffix}"):
        stem = note_path.stem
        if note_prefix and stem.startswith(note_prefix.rstrip("_")):
            # Handles both prefix with and without final underscore robustly below.
            pass
        if note_prefix and stem.startswith(note_prefix):
            stem = stem[len(note_prefix) :]
        ids.append(normalize_entity_id(stem))
    return sorted(set(ids))


def read_note_text(notes_dir: Path, entity_id: str, note_prefix: str, note_suffix: str) -> str:
    """Read text notes for one entity."""
    candidates = [
        notes_dir / f"{note_prefix}{entity_id}{note_suffix}",
        notes_dir / f"{entity_id}{note_suffix}",
    ]

    for path in candidates:
        if path.exists():
            for encoding in ("utf-8", "ISO-8859-1"):
                try:
                    return path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
                except Exception as exc:
                    return f"(Could not read note file {path}: {exc})"

    return "(No note file found for this case.)"


def pick_candidates(
    temperature_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
    note_ids: Iterable[str],
    *,
    id_column: str,
) -> List[str]:
    """Return cases that have temperature data, tabular data, and notes."""
    return sorted(set(temperature_df[id_column]) & set(tabular_df[id_column]) & set(note_ids))


def build_temperature_curve(
    temperature_df: pd.DataFrame,
    entity_id: str,
    *,
    id_column: str,
    time_column: str,
    value_column: str,
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    sub = temperature_df.loc[temperature_df[id_column] == entity_id, [time_column, value_column]].copy()
    sub = sub.sort_values(time_column)

    if sub.empty:
        return pd.DataFrame(columns=["Hour", "Temperature_C"]), None, None

    t0 = sub[time_column].min()
    tmax = sub[time_column].max()
    out = pd.DataFrame(
        {
            "Hour": (sub[time_column] - t0).dt.total_seconds() / 3600.0,
            "Temperature_C": sub[value_column].astype(float),
        }
    )
    return out, t0, tmax


def build_vital_curve(
    vital_df: pd.DataFrame,
    entity_id: str,
    t0: Optional[pd.Timestamp],
    tmax: Optional[pd.Timestamp],
    *,
    id_column: str,
    time_column: str,
    value_column: str,
) -> pd.DataFrame:
    if t0 is None or tmax is None or vital_df.empty:
        return pd.DataFrame(columns=["Hour", "Value"])

    sub = vital_df.loc[vital_df[id_column] == entity_id, [time_column, value_column]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Hour", "Value"])

    sub = sub[sub[time_column] <= tmax].sort_values(time_column)
    if sub.empty:
        return pd.DataFrame(columns=["Hour", "Value"])

    return pd.DataFrame(
        {
            "Hour": (sub[time_column] - t0).dt.total_seconds() / 3600.0,
            "Value": sub[value_column].astype(float),
        }
    )


# -----------------------------------------------------------------------------
# Tables and charts
# -----------------------------------------------------------------------------

LAB_FEATURES = [
    ("Creatinine", "time_lag_1_krea_max", "mg/dL"),
    ("Bilirubin", "time_lag_1_bili_max", "mg/dL"),
    ("Leucocytes", "time_lag_1_leua_max", "10⁹/L"),
    ("CRP", "time_lag_1_crp_max", "mg/L"),
    ("Hemoglobin", "time_lag_1_hb_max", "g/dL"),
]

VITAL_FEATURES = [
    ("Heart Rate (max, lag 1)", "time_lag_1_heart_rate_max", "bpm"),
    ("Mean Arterial Pressure (max, lag 1)", "time_lag_1_mean_arterial_pressure_max", "mmHg"),
    ("SO₂ (max, lag 1)", "time_lag_1_so2_max", "%"),
]


def make_feature_table(row: pd.Series, specs: List[Tuple[str, str, str]], label_col: str) -> pd.DataFrame:
    records = []
    for label, column, unit in specs:
        records.append({label_col: label, "Value": safe_numeric(row.get(column, np.nan)), "Unit": unit})
    return pd.DataFrame(records)


def infer_sex(row: pd.Series) -> str:
    if "sex" in row.index and pd.notna(row["sex"]):
        return str(row["sex"])
    if "sex_male" in row.index:
        return "Male" if bool(row["sex_male"]) else "Female"
    return "Unknown"


def overlay_timeseries_plotly(
    temperature_df: pd.DataFrame,
    vital_curves: Dict[str, pd.DataFrame],
    *,
    fever_threshold: float,
) -> go.Figure:
    """Shared-x plot with separate y-axes for temperature, HR, MAP, and SO2."""
    fig = go.Figure()

    if not temperature_df.empty:
        fig.add_trace(
            go.Scatter(
                x=temperature_df["Hour"],
                y=temperature_df["Temperature_C"],
                mode="lines+markers",
                name="Temperature (°C)",
                yaxis="y",
                line=dict(width=2.5),
                marker=dict(size=5),
            )
        )

    axis_specs = {
        "Heart Rate": {"axis": "y2", "label": "HR (bpm)", "range": [0, 250], "position": 0.88},
        "MAP": {"axis": "y3", "label": "MAP (mmHg)", "range": [0, 200], "position": 0.94},
        "SO2": {"axis": "y4", "label": "SO₂ (%)", "range": [50, 100], "position": 1.0},
    }

    for name, spec in axis_specs.items():
        df = vital_curves.get(name, pd.DataFrame())
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=df["Hour"],
                    y=df["Value"],
                    mode="lines+markers",
                    name=spec["label"],
                    yaxis=spec["axis"],
                    line=dict(width=1.6),
                    marker=dict(size=4),
                    opacity=0.60,
                )
            )

    x_max = float(temperature_df["Hour"].max()) if not temperature_df.empty else None
    dtick = 2 if x_max is not None and x_max <= 24 else 6 if x_max is not None and x_max <= 72 else 12

    fig.update_layout(
        height=600,
        margin=dict(t=30, r=240, b=50, l=70),
        hovermode="x unified",
        xaxis=dict(
            domain=[0.0, 0.85],
            title="Hours since first temperature measurement",
            rangemode="tozero",
            range=[0, x_max] if x_max is not None else None,
            dtick=dtick,
        ),
        yaxis=dict(
            title="Temperature (°C)",
            range=[34, 43],
            showgrid=True,
            ticks="outside",
            showline=True,
        ),
        yaxis2=dict(
            title=axis_specs["Heart Rate"]["label"],
            overlaying="y",
            side="right",
            anchor="free",
            position=axis_specs["Heart Rate"]["position"],
            range=axis_specs["Heart Rate"]["range"],
            showgrid=False,
            showline=True,
        ),
        yaxis3=dict(
            title=axis_specs["MAP"]["label"],
            overlaying="y",
            side="right",
            anchor="free",
            position=axis_specs["MAP"]["position"],
            range=axis_specs["MAP"]["range"],
            showgrid=False,
            showline=True,
        ),
        yaxis4=dict(
            title=axis_specs["SO2"]["label"],
            overlaying="y",
            side="right",
            anchor="free",
            position=axis_specs["SO2"]["position"],
            range=axis_specs["SO2"]["range"],
            showgrid=False,
            showline=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.add_hline(y=fever_threshold, line_dash="dash", line_width=1, line_color="gray", yref="y")
    return fig


# -----------------------------------------------------------------------------
# Case building and response persistence
# -----------------------------------------------------------------------------

def build_case(
    tabular_df: pd.DataFrame,
    temperature_df: pd.DataFrame,
    heart_rate_df: pd.DataFrame,
    map_df: pd.DataFrame,
    so2_df: pd.DataFrame,
    case_id: str,
    config: AppConfig,
) -> CaseBundle:
    row_df = tabular_df.loc[tabular_df[config.id_column] == case_id]
    if row_df.empty:
        raise ValueError(f"No tabular row for case_id={case_id}")
    row = row_df.iloc[0]

    temperature_curve, t0, tmax = build_temperature_curve(
        temperature_df,
        case_id,
        id_column=config.id_column,
        time_column=config.time_column,
        value_column=config.value_column,
    )

    vital_curves = {
        "Heart Rate": build_vital_curve(
            heart_rate_df,
            case_id,
            t0,
            tmax,
            id_column=config.id_column,
            time_column=config.time_column,
            value_column=config.value_column,
        ),
        "MAP": build_vital_curve(
            map_df,
            case_id,
            t0,
            tmax,
            id_column=config.id_column,
            time_column=config.time_column,
            value_column=config.value_column,
        ),
        "SO2": build_vital_curve(
            so2_df,
            case_id,
            t0,
            tmax,
            id_column=config.id_column,
            time_column=config.time_column,
            value_column=config.value_column,
        ),
    }

    return CaseBundle(
        case_id=case_id,
        age=safe_numeric(row.get("age", np.nan)),
        sex=infer_sex(row),
        comorbidity_score=safe_numeric(row.get("elixhauser_score", np.nan)),
        labs_table=make_feature_table(row, LAB_FEATURES, label_col="Laboratory value"),
        vitals_table=make_feature_table(row, VITAL_FEATURES, label_col="Vital sign"),
        temperature_df=temperature_curve,
        vital_curves=vital_curves,
        notes_text=read_note_text(config.notes_dir, case_id, config.note_prefix, config.note_suffix),
    )


def initialize_state(config: AppConfig) -> None:
    if "case_id" not in st.session_state:
        st.session_state.case_id = None
    if "responses" not in st.session_state:
        if config.responses_csv.exists():
            try:
                st.session_state.responses = pd.read_csv(config.responses_csv).to_dict(orient="records")
            except Exception:
                st.session_state.responses = []
        else:
            st.session_state.responses = []


def deduplicate_responses() -> None:
    if not st.session_state.get("responses"):
        return
    df = pd.DataFrame(st.session_state.responses)
    if "case_id" in df.columns:
        df = df.drop_duplicates(subset=["case_id"], keep="last")
    elif "encounter_id" in df.columns:
        df = df.drop_duplicates(subset=["encounter_id"], keep="last")
    st.session_state.responses = df.to_dict(orient="records")


def persist_responses(config: AppConfig) -> None:
    config.responses_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(st.session_state.responses)
    if not df.empty:
        df.to_csv(config.responses_csv, index=False)


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------

def main() -> None:
    initialize_state(CONFIG)
    deduplicate_responses()

    temperature_df = load_measurements(
        str(CONFIG.data_dir / CONFIG.temperature_file),
        id_column=CONFIG.id_column,
        time_column=CONFIG.time_column,
        value_column=CONFIG.value_column,
        csv_sep=CONFIG.csv_sep,
        clip_low=CONFIG.temperature_clip_low,
        clip_high=CONFIG.temperature_clip_high,
    )
    tabular_df = load_tabular(str(CONFIG.data_dir / CONFIG.tabular_file), id_column=CONFIG.id_column, csv_sep=CONFIG.csv_sep)
    heart_rate_df = load_measurements(
        str(CONFIG.data_dir / CONFIG.heart_rate_file),
        id_column=CONFIG.id_column,
        time_column=CONFIG.time_column,
        value_column=CONFIG.value_column,
        csv_sep=CONFIG.csv_sep,
        clip_low=0,
        clip_high=250,
    )
    map_df = load_measurements(
        str(CONFIG.data_dir / CONFIG.map_file),
        id_column=CONFIG.id_column,
        time_column=CONFIG.time_column,
        value_column=CONFIG.value_column,
        csv_sep=CONFIG.csv_sep,
        clip_low=0,
        clip_high=200,
    )
    so2_df = load_measurements(
        str(CONFIG.data_dir / CONFIG.so2_file),
        id_column=CONFIG.id_column,
        time_column=CONFIG.time_column,
        value_column=CONFIG.value_column,
        csv_sep=CONFIG.csv_sep,
        clip_low=50,
        clip_high=100,
    )

    note_ids = list_note_entities(str(CONFIG.notes_dir), CONFIG.note_prefix, CONFIG.note_suffix)
    candidates = pick_candidates(temperature_df, tabular_df, note_ids, id_column=CONFIG.id_column)

    if not candidates:
        st.error("No overlapping cases found after normalizing IDs.")
        st.stop()

    with st.sidebar:
        st.header("Case Selection")
        picked = st.selectbox("Case", ["(random)"] + candidates, index=0)
        if st.button("🔀 Random case"):
            st.session_state.case_id = random.choice(candidates)
        st.caption(f"Available cases: {len(candidates)}")
        st.caption(f"Saved assessments: {len(st.session_state.responses)}")

    if picked != "(random)":
        st.session_state.case_id = picked
    elif st.session_state.case_id is None:
        st.session_state.case_id = random.choice(candidates)

    title_col, logo_col = st.columns([6, 1])
    with title_col:
        st.title(CONFIG.app_title)
    with logo_col:
        if CONFIG.logo_path and CONFIG.logo_path.exists():
            st.image(str(CONFIG.logo_path), use_container_width=True)

    with st.expander("Instructions", expanded=True):
        st.markdown(CONFIG.instructions)
        if CONFIG.study_context:
            st.markdown("---")
            st.markdown(CONFIG.study_context)

    case = build_case(
        tabular_df=tabular_df,
        temperature_df=temperature_df,
        heart_rate_df=heart_rate_df,
        map_df=map_df,
        so2_df=so2_df,
        case_id=st.session_state.case_id,
        config=CONFIG,
    )

    st.caption(f"Case ID: `{case.case_id}`")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Age", "Unknown" if pd.isna(case.age) else f"{int(round(case.age))}")
    with c2:
        st.metric("Sex", case.sex)
    with c3:
        st.metric("Comorbidity Score", "Unknown" if pd.isna(case.comorbidity_score) else f"{case.comorbidity_score:.0f}")

    st.subheader("Longitudinal Measurements")
    if case.temperature_df.empty:
        st.info("No temperature data available.")
    else:
        fig = overlay_timeseries_plotly(
            case.temperature_df,
            case.vital_curves,
            fever_threshold=CONFIG.fever_threshold,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.subheader("Laboratory Values")
    lab_table = case.labs_table.copy()
    lab_table["Value"] = pd.to_numeric(lab_table["Value"], errors="coerce")
    st.dataframe(lab_table, hide_index=True, use_container_width=True)

    with st.expander("Vital-sign summary table", expanded=False):
        vital_table = case.vitals_table.copy()
        vital_table["Value"] = pd.to_numeric(vital_table["Value"], errors="coerce")
        st.dataframe(vital_table, hide_index=True, use_container_width=True)

    st.subheader("Case Notes")
    st.text_area("Notes", value=case.notes_text, height=350, key="notes_box", disabled=True)

    st.subheader("Assessment")
    col1, col2 = st.columns([1, 1])
    with col1:
        decision = st.radio(
            CONFIG.target_question,
            [CONFIG.positive_label, CONFIG.negative_label],
            horizontal=True,
        )
    with col2:
        probability = st.slider("Confidence/probability", 0.0, 1.0, 0.5, 0.01)

    comment = st.text_area(
        "Optional comment",
        placeholder="Which patterns influenced your decision?",
        height=110,
    )

    if st.button("Save / Update assessment"):
        new_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case.case_id,
            "decision": decision,
            "probability": probability,
            "age": case.age,
            "sex": case.sex,
            "comorbidity_score": case.comorbidity_score,
            "peak_temperature": float(case.temperature_df["Temperature_C"].max()) if not case.temperature_df.empty else np.nan,
            "comment": comment.strip(),
        }

        existing_idx = next(
            (idx for idx, row in enumerate(st.session_state.responses) if row.get("case_id") == case.case_id),
            None,
        )
        if existing_idx is not None:
            st.session_state.responses[existing_idx] = new_entry
            st.info("Updated existing assessment for this case.")
        else:
            st.session_state.responses.append(new_entry)
            st.success("Assessment saved.")

        deduplicate_responses()
        persist_responses(CONFIG)

    if st.session_state.responses:
        st.subheader("Saved Assessments")
        responses_df = pd.DataFrame(st.session_state.responses).drop_duplicates(subset=["case_id"], keep="last")
        st.dataframe(responses_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=responses_df.to_csv(index=False).encode("utf-8"),
            file_name=f"validation_assessments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
