#!/usr/bin/env python3
"""Run a lightweight, reproducible smoke test of the released analysis code.

The demo uses the repository's logistic-regression nested-CV pipeline because it
runs on a standard CPU without downloading restricted clinical data or large
foundation-model weights. By default it evaluates the bundled synthetic
encounter-level dataset. A reader can instead supply a prepared CSV of their own.

This demo is intended to verify installation, I/O, model execution, and output
creation. It is not intended to reproduce the performance estimates reported in
the manuscript.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the multimodal-fever-prediction demo.")
    parser.add_argument(
        "--input",
        default=str(script_dir / "example_input.csv"),
        help="Prepared encounter-level CSV. Defaults to the bundled synthetic example.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "demo_output"),
        help="Directory for demo outputs.",
    )
    parser.add_argument("--target-col", default="fever", help="Binary target column (0/1).")
    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=["encounter_id"],
        help="Identifier columns copied to prediction outputs.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        help="Optional explicit numeric feature columns. If omitted, numeric non-ID/non-target columns are used.",
    )
    parser.add_argument("--csv-sep", default=",", help="CSV separator for the input file.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_input(path: Path, sep: str, target_col: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, sep=sep)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from {path}")
    values = set(pd.Series(df[target_col]).dropna().astype(int).unique().tolist())
    if not values.issubset({0, 1}) or len(values) < 2:
        raise ValueError(
            f"Target column '{target_col}' must contain both binary classes 0 and 1; found {sorted(values)}"
        )
    if len(df) < 30:
        raise ValueError("The demo requires at least 30 rows so that stratified nested CV can be run safely.")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = repo_root / "training-validation" / "lr_cv.py"
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    validate_input(input_path, args.csv_sep, args.target_col)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(pipeline),
        "--data-path",
        str(input_path),
        "--csv-sep",
        args.csv_sep,
        "--target-col",
        args.target_col,
        "--output-dir",
        str(output_dir),
        "--outer-splits",
        "3",
        "--inner-splits",
        "2",
        "--n-repeats",
        "1",
        "--n-trials",
        "2",
        "--bootstrap-iterations",
        "100",
        "--seed",
        str(args.seed),
    ]

    if args.id_cols:
        cmd.extend(["--id-cols", *args.id_cols])
    if args.feature_cols:
        cmd.extend(["--feature-cols", *args.feature_cols])

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    print("Running released logistic-regression nested-CV pipeline on:")
    print(f"  {input_path}")
    print(f"Outputs will be written to:")
    print(f"  {output_dir}")
    print()

    started = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)
    elapsed = time.perf_counter() - started

    metrics_path = output_dir / "nested_cv_metrics_pooled.csv"
    predictions_path = output_dir / "nested_cv_predictions_case_level_pooled.csv"

    if not metrics_path.exists() or not predictions_path.exists():
        raise RuntimeError("Demo pipeline completed but expected output files were not created.")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)

    print("\nDemo completed successfully.")
    print(f"Case-level predictions: {len(predictions)} encounters")
    print(f"Runtime: {elapsed:.1f} seconds")
    print(f"Primary output: {predictions_path}")
    print(f"Metric summary: {metrics_path}")

    # Print a compact, version-tolerant summary without assuming a fixed row order.
    if {"threshold_strategy", "metric", "mean"}.issubset(metrics.columns):
        wanted = metrics[
            (metrics["threshold_strategy"] == "default_0_5")
            & (metrics["metric"].isin(["auc", "auprc", "sensitivity", "specificity", "accuracy"]))
        ]
        if not wanted.empty:
            print("\nSelected metrics at the prespecified 0.5 threshold:")
            for _, row in wanted.iterrows():
                ci_text = ""
                if {"ci_low", "ci_high"}.issubset(metrics.columns):
                    ci_text = f" (95% CI {row['ci_low']:.3f}-{row['ci_high']:.3f})"
                print(f"  {row['metric']}: {row['mean']:.3f}{ci_text}")
    else:
        print("\nMetric file created; see nested_cv_metrics_pooled.csv for the full summary.")


if __name__ == "__main__":
    main()

