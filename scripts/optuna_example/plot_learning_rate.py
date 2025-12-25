# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "kaleido",
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
import plotly.express as px
import plotly.express as px
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from binscatter.core import binscatter  # noqa: E402

HYPERPARAM_COLUMNS = [
    "learning_rate",
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
        default=ROOT / "images" / "lightgbm",
    )
    parser.add_argument("--num-bins", type=int, default=25)
    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(
            f"{args.results_path} missing. Run optimize_lightgbm.py first."
        )
    df = pd.read_parquet(args.results_path)
    df = df.rename(columns={"value": "rmse"}) if "value" in df.columns else df
    missing = [col for col in HYPERPARAM_COLUMNS if col not in df.columns]
    if "learning_rate" in missing:
        raise ValueError("Results file must contain a 'learning_rate' column.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = ROOT / "images" / "lightgbm"
    image_dir.mkdir(parents=True, exist_ok=True)

    for parameter in [col for col in HYPERPARAM_COLUMNS if col in df.columns]:
        prefix = f"{parameter}_rmse"
        scatter_fig = px.scatter(
            df,
            x=parameter,
            y="rmse",
            template="simple_white",
            color_discrete_sequence=["black"],
        )
        scatter_fig.update_layout(showlegend=False)
        scatter_fig.update_xaxes(title_text=parameter)
        scatter_fig.update_yaxes(title_text="RMSE")
        scatter_fig.write_image(
            image_dir / f"{prefix}_scatter.png",
            width=960,
            height=640,
        )

        plain_fig = binscatter(
            df,
            parameter,
            "rmse",
            num_bins=args.num_bins,
        )
        plain_fig.write_image(
            image_dir / f"{prefix}_binscatter.png", width=960, height=640
        )

        controls = [c for c in HYPERPARAM_COLUMNS if c != parameter and c in df.columns]
        if controls:
            controls_fig = binscatter(
                df,
                parameter,
                "rmse",
                controls=controls,
                num_bins=args.num_bins,
            )
            controls_fig.write_image(
                image_dir / f"{prefix}_binscatter_controls.png",
                width=960,
                height=640,
            )
            print(
                f"Saved {parameter} binscatter with controls to "
                f"{image_dir / (prefix + '_binscatter_controls.png')}"
            )
        else:
            print(f"No controls available for {parameter}; skipped controlled plot.")

        print(
            f"Saved {parameter} binscatter to "
            f"{image_dir / (prefix + '_binscatter.png')}"
        )


if __name__ == "__main__":
    main()
