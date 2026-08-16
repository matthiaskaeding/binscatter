# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "binscatter",
#     "polars>=1.22.0",
#     "kaleido>=0.2.1",
#     "plotly>=6.3",
#     "numpy>=2.3",
#     "pandas",
#     "pyarrow>=17",
# ]
#
# [tool.uv.sources]
# binscatter = { path = "../..", editable = true }
# ///
"""Generate binscatter demo figures for README and blog posts.

Uses optional local artifacts (state_data, optuna trials) when available.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from binscatter import binscatter

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
IMAGES = ROOT / "images" / "readme"
IMAGES.mkdir(parents=True, exist_ok=True)

REQUIRED_README_IMAGES = [
    "binscatter_controls.png",
]


def _write_binscatter_variants(
    filename: str,
    /,
    *,
    add_dpi_variant: bool,
    dpi_filename: str | None = None,
    args: tuple,
    kwargs: dict,
) -> None:
    fig = binscatter(*args, **kwargs)
    _write_fig(fig, filename)
    if not add_dpi_variant:
        return
    dpi_kwargs = {**kwargs, "num_bins": "dpi"}
    dpi_name = dpi_filename or filename.replace(".png", "_dpi.png")
    fig_dpi = binscatter(*args, **dpi_kwargs)
    _write_fig(fig_dpi, dpi_name)


def _require_artifact(path: Path) -> Path:
    if not path.exists():
        msg = f"Missing {path}. Run the prerequisite prep scripts first."
        raise FileNotFoundError(msg)
    return path


def _write_fig(fig, filename: str, **write_kwargs) -> None:
    target = IMAGES / filename
    fig.write_image(str(target), scale=write_kwargs.pop("scale", 2), **write_kwargs)
    print(f"Wrote {target}")


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
        poly_line=1,
    )
    fig.update_layout(
        xaxis_title="Log net-of-tax rate (3-year lag)",
        yaxis_title="Log number of patents",
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
    _write_binscatter_variants(
        "lightgbm_learning_rate.png",
        add_dpi_variant=True,
        args=(df,),
        kwargs={
            "x": "learning_rate",
            "y": "rmse",
            "num_bins": "rule-of-thumb",
        },
    )

    # With controls
    _write_binscatter_variants(
        "lightgbm_learning_rate_controls.png",
        add_dpi_variant=True,
        args=(df,),
        kwargs={
            "x": "learning_rate",
            "y": "rmse",
            "controls": [
                "num_leaves",
                "min_child_samples",
                "feature_fraction",
                "lambda_l1",
            ],
            "num_bins": "rule-of-thumb",
        },
    )


def main() -> None:
    builders = [
        ("lightgbm", build_lightgbm_plot),
        ("readme (state data)", build_readme_plot),
    ]

    failed = []
    for name, builder in builders:
        try:
            builder()
        except Exception as e:
            print(f"ERROR [{name}]: {e}")
            failed.append(name)

    if failed:
        print(f"\nSkipped {len(failed)} dataset(s): {', '.join(failed)}")
    else:
        print("\nAll plots generated successfully.")

    missing = [name for name in REQUIRED_README_IMAGES if not (IMAGES / name).exists()]
    if missing:
        files = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing README image(s): {files}. Gapminder plots must render successfully."
        )


if __name__ == "__main__":
    main()
