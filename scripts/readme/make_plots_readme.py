# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "binscatter",
#     "polars>=1.22.0",
#     "kaleido>=0.2.1",
#     "plotly>=6.3",
#     "numpy>=2.3",
#     "pandas",
# ]
#
# [tool.uv.sources]
# binscatter = { path = "../..", editable = true }
# ///
"""Generate binscatter demo figures for README and blog posts.

Uses plotly built-in datasets (gapminder, tips, iris) plus optional local
artifacts (state_data, optuna trials) when available.
"""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import polars as pl
from binscatter import binscatter

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
IMAGES = ROOT / "images" / "readme"
IMAGES.mkdir(parents=True, exist_ok=True)


def _require_artifact(path: Path) -> Path:
    if not path.exists():
        msg = f"Missing {path}. Run the prerequisite prep scripts first."
        raise FileNotFoundError(msg)
    return path


def _write_fig(fig, filename: str, **write_kwargs) -> None:
    target = IMAGES / filename
    fig.write_image(str(target), scale=write_kwargs.pop("scale", 2), **write_kwargs)
    print(f"Wrote {target}")  # noqa: T201


def _load_state_df() -> pl.DataFrame:
    return pl.read_parquet(
        _require_artifact(ARTIFACTS / "state_data_processed.parquet")
    ).select(
        "mtr90_lag3",
        "lnpat",
        "top_corp_lag3",
        "real_gdp_pc",
        "population_density",
        "rd_credit_lag3",
        "statenum",
        "year",
    )


def _load_optuna_df(name: str, columns: list[str]) -> pl.DataFrame:
    return pl.read_parquet(
        _require_artifact(ARTIFACTS / f"optuna_{name}_trials.parquet")
    ).select(*columns)


def build_readme_plot() -> None:
    df = _load_state_df()
    controls = [
        "top_corp_lag3",
        "real_gdp_pc",
        "population_density",
        "rd_credit_lag3",
        "statenum",
        "year",
    ]
    fig = binscatter(
        df,
        x="mtr90_lag3",
        y="lnpat",
        controls=controls,
        num_bins="rule-of-thumb",
    )
    _write_fig(fig, "binscatter_controls.png")


def build_lightgbm_plot() -> None:
    df = _load_optuna_df(
        "lightgbm",
        [
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "feature_fraction",
            "lambda_l1",
            "rmse",
        ],
    )
    # Without controls
    fig = binscatter(
        df,
        x="learning_rate",
        y="rmse",
    )
    _write_fig(fig, "lightgbm_learning_rate.png")

    # With controls
    fig_controls = binscatter(
        df,
        x="learning_rate",
        y="rmse",
        controls=[
            "num_leaves",
            "min_child_samples",
            "feature_fraction",
            "lambda_l1",
        ],
    )
    _write_fig(fig_controls, "lightgbm_learning_rate_controls.png")


def build_gapminder_plots() -> None:
    df_pl = pl.from_pandas(px.data.gapminder()).with_columns(
        pl.col("gdpPercap").log().alias("log_gdp"),
        pl.col("lifeExp").log().alias("log_life"),
    )
    fig = binscatter(
        df_pl,
        "gdpPercap",
        "lifeExp",
        num_bins=40,
    )
    _write_fig(fig, "gapminder_gdp_lifeexp.png")

    fig_log = binscatter(
        df_pl,
        "log_gdp",
        "log_life",
    )
    _write_fig(fig_log, "gapminder_log_axes.png")


def main() -> None:
    builders = [
        ("gapminder", build_gapminder_plots),
        ("lightgbm", build_lightgbm_plot),
        ("readme (state data)", build_readme_plot),
    ]

    failed = []
    for name, builder in builders:
        try:
            builder()
        except Exception as e:
            print(f"ERROR [{name}]: {e}")  # noqa: T201
            failed.append(name)

    if failed:
        print(f"\nSkipped {len(failed)} dataset(s): {', '.join(failed)}")  # noqa: T201
    else:
        print("\nAll plots generated successfully.")  # noqa: T201


if __name__ == "__main__":
    main()
