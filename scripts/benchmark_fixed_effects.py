"""Benchmark the fixed-effect absorption path against one-hot encoding.

Run with ``make benchmark-fe`` or ``uv run scripts/benchmark_fixed_effects.py``.

Absorbing a categorical control replaces G-1 dummy columns with two group-level
aggregations, so cost stops depending on the number of levels. One-hot encoding
does not merely get slower: ``_ensure_feature_moments`` builds ``k(k+1)/2``
sum-product aggregations, which scales roughly cubically in levels.

Several absorbed factors are the case where the driver-side cost stops being flat:
the projection is solved from the distinct fixed-effect intersections, so what
matters is how many of those the data actually contains. ``bench_two_way`` reports
that count alongside the timings.

Correctness of the absorption path lives in tests/test_fixed_effects.py -- this
script only measures.
"""

import logging
import time
import warnings

import narwhals as nw
import numpy as np
import pandas as pd

import binscatter.fixed_effects as fe_mod
from binscatter import binscatter, core

warnings.filterwarnings("ignore")

# One-hot becomes impractical fast, so cap what we ask of it. 400 levels already
# takes ~6 minutes at n=20,000.
ONE_HOT_MAX_LEVELS = 200
CARDINALITIES = (50, 100, 200, 1_000, 10_000, 50_000)


def make_frame(n: int, n_groups: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "y": 0.7 * x + rng.normal(scale=2.0, size=n_groups)[g] + rng.normal(size=n),
            "firm_id": g,
        }
    )


def time_call(df: pd.DataFrame, absorb: bool, columns=("firm_id",)) -> float:
    """Time one call, forcing the requested path.

    Absorption is unconditional in production, so the comparison is made by
    disabling it -- ``absorb=False`` sends every categorical to the dummy builder,
    which is what the absorbed path is being measured against.
    """
    original = core.select_absorbed
    if not absorb:
        core.select_absorbed = lambda *a, **k: ()
    try:
        start = time.perf_counter()
        binscatter(
            df,
            "x",
            "y",
            controls=list(columns),
            categorical=list(columns),
            num_bins=10,
            return_type="native",
        )
        return time.perf_counter() - start
    finally:
        core.select_absorbed = original


def bench_cardinality(n: int = 20_000) -> None:
    print(f"\nAbsorbed vs one-hot by cardinality (pandas, n={n:,}, num_bins=10)")
    print(f"{'levels':>8} {'one-hot':>12} {'absorbed':>11} {'speedup':>10}")
    for n_groups in CARDINALITIES:
        df = make_frame(n, n_groups)
        absorbed = time_call(df, absorb=True)
        if n_groups <= ONE_HOT_MAX_LEVELS:
            one_hot = time_call(df, absorb=False)
            speedup = f"{one_hot / absorbed:>9.0f}x"
            one_hot_s = f"{one_hot:>11.2f}s"
        else:
            one_hot_s, speedup = f"{'not viable':>12}", f"{'-':>10}"
        print(f"{n_groups:>8} {one_hot_s} {absorbed:>10.3f}s {speedup}")


def bench_scale(n_groups: int = 5_000) -> None:
    print(f"\nAbsorbed scaling in rows ({n_groups:,} levels)")
    print(f"{'rows':>12} {'elapsed':>11}")
    for n in (50_000, 200_000, 1_000_000):
        print(f"{n:>12,} {time_call(make_frame(n, n_groups), absorb=True):>10.3f}s")


def make_two_way_frame(
    n: int, n_firms: int, n_years: int, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, n)
    year = rng.integers(0, n_years, n)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "y": (
                0.7 * x
                + rng.normal(scale=2.0, size=n_firms)[firm]
                + rng.normal(scale=1.0, size=n_years)[year]
                + rng.normal(size=n)
            ),
            "firm_id": firm,
            "year": year,
        }
    )


def bench_two_way(n: int = 50_000) -> None:
    """Two crossed factors, where the driver-side cost is the joint cell count.

    ``cells`` is what the aggregated projection actually holds: one row per observed
    firm-year pair, not per observation. That is the quantity to watch -- it is
    bounded by ``min(n, G_1 * G_2)``, so a genuine panel compresses hard while
    near-one-to-one factors (matched employer-employee data) do not.
    """
    print(f"\nTwo-way absorption (pandas, n={n:,}, num_bins=10)")
    print(f"{'firms':>8} {'years':>7} {'cells':>10} {'sweeps':>8} {'elapsed':>11}")
    for n_firms, n_years in ((200, 10), (2_000, 20), (20_000, 50)):
        df = make_two_way_frame(n, n_firms, n_years)
        row_codes = np.column_stack([df["firm_id"].to_numpy(), df["year"].to_numpy()])
        projector = fe_mod.FEProjector.from_row_codes(row_codes, ("firm_id", "year"))
        sweeps = count_sweeps(projector)
        elapsed = time_call(df, absorb=True, columns=("firm_id", "year"))
        print(
            f"{n_firms:>8,} {n_years:>7} {projector.num_cells:>10,} "
            f"{sweeps:>8} {elapsed:>10.3f}s"
        )


def count_sweeps(projector) -> int:
    """Gauss-Seidel sweeps needed for one right-hand side, via the debug log."""
    rhs = projector.matvec(
        np.random.default_rng(0).normal(size=(projector.total_levels, 1))
    )
    records: list[int] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.args[0])  # type: ignore[assignment]
    logger = logging.getLogger("binscatter.fixed_effects")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        projector.solve(rhs)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return records[0] if records else 0


def bench_discovery(n: int = 1_000_000, n_groups: int = 5_000) -> None:
    """Cost of the exact COUNT(DISTINCT) used to route between the two paths.

    Approximate counts were considered and rejected: narwhals exposes none, and
    duckdb's HyperLogLog returned 1249 for 1000 true levels -- enough error to flip
    the absorb decision near the threshold and make bin selection irreproducible.
    """
    print(f"\nCardinality discovery vs the aggregation it routes to (n={n:,})")
    lazy = nw.from_native(make_frame(n, n_groups)).lazy()

    start = time.perf_counter()
    fe_mod.select_absorbed(lazy, ("firm_id",))
    count = time.perf_counter() - start

    start = time.perf_counter()
    lazy.group_by("firm_id").agg(
        nw.len().alias("c"), nw.col("y").sum().alias("s")
    ).collect()
    group = time.perf_counter() - start

    print(f"  select_absorbed : {count:.3f}s")
    print(f"  group_by        : {group:.3f}s")
    print(f"  discovery share : {count / (count + group):.0%}")


if __name__ == "__main__":
    print(f"FE_TOL = {fe_mod.FE_TOL:.0e}  MAX_FE_CELLS = {fe_mod.MAX_FE_CELLS:,}")
    bench_cardinality()
    bench_scale()
    bench_two_way()
    bench_discovery()
