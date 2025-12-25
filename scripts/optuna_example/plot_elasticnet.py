# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "plotly",
#     "pyarrow",
#     "kaleido",
# ]
# ///

"""Visualize ElasticNet Optuna trials using binscatter plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from binscatter.core import binscatter  # noqa: E402

OUT_DIR = ROOT / "images" / "elasticnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _binscatter_plot(df: pd.DataFrame, x: str, controls: list[str], num_bins: int) -> None:
    base_name = f"{x}_rmse"
    scatter_fig = px.scatter(
        df,
        x=x,
        y="rmse",
        template="simple_white",
        color_discrete_sequence=["black"],
    )
    scatter_fig.update_layout(showlegend=False)
    scatter_fig.update_xaxes(title_text=x)
    scatter_fig.update_yaxes(title_text="RMSE")
    scatter_fig.update_traces(marker={"size": 7})
    scatter_fig.write_image(OUT_DIR / f"{base_name}_scatter.png", width=960, height=640)

    figure = binscatter(df, x, "rmse", num_bins=num_bins)
    figure.write_image(OUT_DIR / f"{base_name}_binscatter.png", width=960, height=640)

    if controls:
        control_fig = binscatter(
            df,
            x,
            "rmse",
            controls=controls,
            num_bins=num_bins,
        )
        control_fig.write_image(
            OUT_DIR / f"{base_name}_binscatter_controls.png", width=960, height=640
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ElasticNet Optuna trial results.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "artifacts" / "optuna_elasticnet_trials.parquet",
        help="Parquet file produced by optimize_elasticnet.py",
    )
    parser.add_argument("--num-bins", type=int, default=30)
    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(
            f"{args.results_path} missing. Run optimize_elasticnet.py first."
        )

    df = pd.read_parquet(args.results_path)
    if "rmse" not in df.columns:
        raise ValueError("Trials parquet must contain an 'rmse' column.")

    available_controls = [
        col for col in ("alpha", "l1_ratio") if col in df.columns
    ]
    for hyper in ["alpha", "l1_ratio"]:
        if hyper not in df.columns:
            continue
        controls = [c for c in available_controls if c != hyper]
        _binscatter_plot(df, hyper, controls, args.num_bins)
        print(f"Saved plots for {hyper}")


if __name__ == "__main__":
    main()
