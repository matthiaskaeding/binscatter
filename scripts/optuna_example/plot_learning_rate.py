# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "plotly",
#     "pyarrow",
# ]
# ///

"""Plot learning-rate vs loss relationships from Optuna trials."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.binscatter.core import binscatter  # noqa: E402

CONTROL_COLS = [
    "num_leaves",
    "min_child_samples",
    "feature_fraction",
    "lambda_l1",
    "lambda_l2",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build binscatter plots showing learning-rate vs RMSE relationships."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "artifacts" / "optuna_lightgbm_trials.parquet",
        help="Parquet file produced by optimize_lightgbm.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts",
    )
    parser.add_argument("--num-bins", type=int, default=25)
    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(
            f"{args.results_path} missing. Run optimize_lightgbm.py first."
        )
    df = pd.read_parquet(args.results_path)
    df = df.rename(columns={"value": "rmse"}) if "value" in df.columns else df
    if "learning_rate" not in df.columns:
        raise ValueError("Results file must contain a 'learning_rate' column.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plain_fig = binscatter(
        df,
        "learning_rate",
        "rmse",
        num_bins=args.num_bins,
    )
    plain_path = args.output_dir / "learning_rate_rmse_binscatter.html"
    plain_fig.write_html(plain_path)

    available_controls = [col for col in CONTROL_COLS if col in df.columns]
    if available_controls:
        controls_fig = binscatter(
            df,
            "learning_rate",
            "rmse",
            controls=available_controls,
            num_bins=args.num_bins,
        )
        controls_path = args.output_dir / "learning_rate_rmse_binscatter_controls.html"
        controls_fig.write_html(controls_path)
        print(f"Saved binscatter with controls to {controls_path}")
    else:
        print("No control columns found; skipped controlled binscatter plot.")

    print(f"Saved plain binscatter to {plain_path}")


if __name__ == "__main__":
    main()
