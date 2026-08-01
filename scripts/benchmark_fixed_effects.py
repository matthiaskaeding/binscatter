"""Benchmark the fixed-effect absorption path against one-hot encoding.

Run with ``make benchmark-fe`` or ``uv run scripts/benchmark_fixed_effects.py``.

Absorbing a categorical control replaces G-1 dummy columns with two group-level
aggregations, so cost stops depending on the number of levels. One-hot encoding
does not merely get slower: ``_ensure_feature_moments`` builds ``k(k+1)/2``
sum-product aggregations, which scales roughly cubically in levels.

Correctness of the absorption path lives in tests/test_fixed_effects.py -- this
script only measures.
"""

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


def time_call(df: pd.DataFrame, absorb: bool) -> float:
    """Time one call, forcing the requested path.

    ``absorb=True`` drops ABSORB_MIN_LEVELS so absorption is exercised even below
    the production threshold -- otherwise the low-cardinality rows would silently
    compare the one-hot path against itself, and report a 1x speedup.
    """
    original = core.select_absorbed
    original_threshold = fe_mod.ABSORB_MIN_LEVELS
    if absorb:
        fe_mod.ABSORB_MIN_LEVELS = 2
    else:
        core.select_absorbed = lambda *a, **k: None
    try:
        start = time.perf_counter()
        binscatter(
            df,
            "x",
            "y",
            controls=["firm_id"],
            categorical=["firm_id"],
            num_bins=10,
            return_type="native",
        )
        return time.perf_counter() - start
    finally:
        core.select_absorbed = original
        fe_mod.ABSORB_MIN_LEVELS = original_threshold


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
    print(f"ABSORB_MIN_LEVELS = {fe_mod.ABSORB_MIN_LEVELS}")
    bench_cardinality()
    bench_scale()
    bench_discovery()
