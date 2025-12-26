from __future__ import annotations

from pathlib import Path

import polars as pl
from binscatter import binscatter

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
IMAGES = ROOT / "images" / "readme"
IMAGES.mkdir(parents=True, exist_ok=True)


def _require_artifact(path: Path) -> Path:
    if not path.exists():
        msg = (
            f"{path} not found. Generate README replication data via "
            "`just make-data-replication` and Optuna artifacts via scripts in "
            "`scripts/optuna_example/`."
        )
        raise FileNotFoundError(msg)
    return path


def _write_fig(fig, filename: str, **write_kwargs) -> None:
    target = IMAGES / filename
    fig.write_image(str(target), **write_kwargs)
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
    name_x = "Log net of tax rate"
    name_y = "Log number of patents"

    df = _load_state_df().with_columns(
        pl.col("mtr90_lag3").alias(name_x),
        pl.col("lnpat").alias(name_y),
    )
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
        x=name_x,
        y=name_y,
        controls=controls,
        num_bins="rule-of-thumb",
    )
    _write_fig(fig, "binscatter_controls.png", scale=2)


def build_elasticnet_plot() -> None:
    df = _load_optuna_df(
        "elasticnet",
        ["alpha", "l1_ratio", "rmse", "duration_seconds"],
    )
    fig = binscatter(
        df,
        x="alpha",
        y="rmse",
        controls=["l1_ratio", "duration_seconds"],
        num_bins=18,
    )
    _write_fig(fig, "elasticnet_alpha.png", scale=2)


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
    fig = binscatter(
        df,
        x="learning_rate",
        y="rmse",
        controls=[
            "num_leaves",
            "min_child_samples",
            "feature_fraction",
            "lambda_l1",
        ],
        num_bins=15,
    )
    _write_fig(fig, "lightgbm_learning_rate.png", scale=2)


def main() -> None:
    build_readme_plot()
    build_elasticnet_plot()
    build_lightgbm_plot()


if __name__ == "__main__":
    main()
