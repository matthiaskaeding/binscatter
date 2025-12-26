# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars>=1.22.0",
#     "kaleido>=0.2.1",
#     "plotly>=6.3",
#     "numpy>=2.3",
# ]
# ///
"""Generate binscatter demo figures for README and blog posts.

The script expects the canonical README and Optuna artifacts to exist already.
Spotify, flight, and Pokémon datasets are synthesized when their parquet files
are absent so we can demo additional narratives without manual downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import plotly.express as px
import polars as pl
from binscatter import binscatter

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
IMAGES = ROOT / "images" / "readme"
IMAGES.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)


def _require_artifact(path: Path) -> Path:
    if not path.exists():
        msg = f"Missing {path}. Run the prerequisite prep scripts first."
        raise FileNotFoundError(msg)
    return path


def _ensure_dataset(path: Path, builder: Callable[[], pl.DataFrame]) -> Path:
    if not path.exists():
        df = builder()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        print(f"Generated synthetic dataset at {path}")  # noqa: T201
    return path


def _write_fig(fig, filename: str, **write_kwargs) -> None:
    target = IMAGES / filename
    fig.write_image(str(target), scale=write_kwargs.pop("scale", 2), **write_kwargs)
    print(f"Wrote {target}")  # noqa: T201


def _load_state_df() -> pl.DataFrame:
    return (
        pl.read_parquet(_require_artifact(ARTIFACTS / "state_data_processed.parquet"))
        .select(
            "mtr90_lag3",
            "lnpat",
            "top_corp_lag3",
            "real_gdp_pc",
            "population_density",
            "rd_credit_lag3",
            "statenum",
            "year",
        )
    )


def _load_optuna_df(name: str, columns: list[str]) -> pl.DataFrame:
    return pl.read_parquet(
        _require_artifact(ARTIFACTS / f"optuna_{name}_trials.parquet")
    ).select(*columns)


def _build_spotify_synthetic(rows: int = 1000) -> pl.DataFrame:
    dance = RNG.beta(2.5, 1.8, size=rows)
    energy = RNG.beta(2.0, 2.0, size=rows)
    tempo = RNG.normal(120, 15, size=rows)
    valence = np.clip(RNG.beta(2.2, 2.0, size=rows), 0, 1)
    # Inverted-U for popularity: medium valence wins
    popularity = np.clip(
        30
        + 45 * dance
        - 80 * (valence - 0.55) ** 2
        + RNG.normal(0, 8, size=rows),
        0,
        100,
    )
    return pl.DataFrame(
        {
            "track_id": [f"track_{i}" for i in range(rows)],
            "danceability": dance,
            "energy": energy,
            "tempo": tempo,
            "popularity": popularity,
            "valence": valence,
        }
    )


def _build_flights_synthetic(rows: int = 1500) -> pl.DataFrame:
    distance = RNG.uniform(150, 2800, size=rows)
    dep_delay = RNG.normal(5, 25, size=rows)
    arr_delay = dep_delay * 0.7 + RNG.normal(0, 15, size=rows) + distance / 1500
    carriers = RNG.choice(["AA", "DL", "UA", "WN", "B6", "AS"], size=rows)
    return pl.DataFrame(
        {
            "flight_id": np.arange(rows),
            "carrier": carriers,
            "distance": distance,
            "dep_delay": dep_delay,
            "arr_delay": arr_delay,
        }
    )


def _build_pokemon_synthetic(rows: int = 600) -> pl.DataFrame:
    tiers = RNG.choice(["Starter", "Rare", "Legendary"], size=rows, p=[0.6, 0.3, 0.1])
    tier_bonus = {"Starter": 0, "Rare": 30, "Legendary": 80}
    attack = RNG.normal(70, 20, size=rows) + np.vectorize(tier_bonus.get)(tiers)
    defense = RNG.normal(65, 18, size=rows) + np.vectorize(tier_bonus.get)(tiers)
    speed = RNG.normal(70, 22, size=rows) + np.vectorize(tier_bonus.get)(tiers)
    total = attack + defense + speed + RNG.normal(50, 10, size=rows)
    return pl.DataFrame(
        {
            "name": [f"Pokemon_{i}" for i in range(rows)],
            "tier": tiers,
            "attack": attack,
            "defense": defense,
            "speed": speed,
            "total": total,
        }
    )


def _load_spotify_df() -> pl.DataFrame:
    path = _ensure_dataset(ARTIFACTS / "spotify_tracks.parquet", _build_spotify_synthetic)
    return pl.read_parquet(path).select(
        "danceability",
        "popularity",
        "energy",
        "valence",
        "tempo",
    )


def _load_flight_df() -> pl.DataFrame:
    path = _ensure_dataset(ARTIFACTS / "flight_delays.parquet", _build_flights_synthetic)
    return pl.read_parquet(path).select(
        "dep_delay",
        "arr_delay",
        "distance",
        "carrier",
    )


def _load_pokemon_df() -> pl.DataFrame:
    path = _ensure_dataset(ARTIFACTS / "pokemon_stats.parquet", _build_pokemon_synthetic)
    return pl.read_parquet(path).select(
        "attack",
        "defense",
        "speed",
        "total",
    )


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
    _write_fig(fig, "elasticnet_alpha.png")


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
    _write_fig(fig, "lightgbm_learning_rate.png")


def build_gapminder_plots() -> None:
    df_pl = pl.from_pandas(px.data.gapminder())
    fig = binscatter(
        df_pl,
        x="gdpPercap",
        y="lifeExp",
    )
    _write_fig(fig, "gapminder_gdp_lifeexp.png")

    df_log = df_pl.select(
        pl.col("continent"),
        pl.col("year"),
        pl.col("lifeExp"),
        pl.col("gdpPercap"),
        pl.col("gdpPercap").log().alias("log_gdp"),
        pl.col("lifeExp").log().alias("log_life"),
    )
    fig_log = binscatter(
        df_log,
        x="log_gdp",
        y="log_life",
    )
    _write_fig(fig_log, "gapminder_log_axes.png")


def build_spotify_plots() -> None:
    df = _load_spotify_df()
    fig = binscatter(
        df,
        x="danceability",
        y="popularity",
        controls=["tempo"],
        num_bins=20,
    )
    _write_fig(fig, "spotify_dance_popularity.png")

    fig_energy = binscatter(
        df,
        x="energy",
        y="valence",
        controls=["tempo"],
        num_bins=20,
    )
    _write_fig(fig_energy, "spotify_energy_valence.png")

    fig_valence = binscatter(
        df,
        x="valence",
        y="popularity",
        controls=["tempo"],
        num_bins=22,
    )
    _write_fig(fig_valence, "spotify_valence_popularity.png")


def build_flight_plots() -> None:
    df = _load_flight_df()
    fig = binscatter(
        df,
        x="dep_delay",
        y="arr_delay",
        controls=["distance"],
        num_bins=20,
    )
    _write_fig(fig, "flight_delay_relationship.png")

    dist_fig = binscatter(
        df,
        x="distance",
        y="arr_delay",
        controls=["carrier"],
        num_bins=20,
    )
    _write_fig(dist_fig, "flight_distance_delay.png")


def build_pokemon_plots() -> None:
    df = _load_pokemon_df()
    fig = binscatter(
        df,
        x="attack",
        y="defense",
        num_bins=18,
    )
    _write_fig(fig, "pokemon_attack_defense.png")

    speed_fig = binscatter(
        df,
        x="speed",
        y="total",
        num_bins=18,
    )
    _write_fig(speed_fig, "pokemon_speed_total.png")


def main() -> None:
    build_readme_plot()
    build_elasticnet_plot()
    build_lightgbm_plot()
    build_gapminder_plots()
    build_spotify_plots()
    build_flight_plots()
    build_pokemon_plots()


if __name__ == "__main__":
    main()
