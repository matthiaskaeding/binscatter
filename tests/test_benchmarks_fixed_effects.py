"""Benchmarks for the fixed-effect absorption path.

The claim absorption makes is that cost stops depending on the number of levels.
These check that, and guard against silently falling back to one-hot encoding --
which does not merely get slower, it becomes unusable: the one-hot path builds
``k(k+1)/2`` sum-product aggregations, so it scales roughly cubically in levels.

Measured on pandas, n=20,000, num_bins=10 (see CHANGELOG 0.4.0):

    levels   one-hot    absorbed
        50     1.08s       0.03s
       100     5.92s       0.03s
       200    46.35s       0.03s
       400   >8 min        0.03s

Budgets below are deliberately loose -- they exist to catch order-of-magnitude
regressions on shared CI runners, not to police small fluctuations. Run with
``--skip-benchmarks`` to drop them entirely.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

import binscatter.core as core
import binscatter.fixed_effects as fe_mod
from binscatter import binscatter

pytestmark = pytest.mark.benchmark


@contextmanager
def timer() -> Generator[dict, None, None]:
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def make_fe_frame(n: int, n_groups: int, seed: int = 0) -> pd.DataFrame:
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


def run(df: pd.DataFrame, **kwargs) -> float:
    with timer() as t:
        binscatter(
            df,
            "x",
            "y",
            controls=["firm_id"],
            categorical=["firm_id"],
            num_bins=10,
            return_type="native",
            **kwargs,
        )
    return t["elapsed"]


@pytest.mark.parametrize("n_groups", [100, 1_000, 10_000])
def test_absorbed_cost_is_flat_in_cardinality(n_groups: int) -> None:
    """The whole point: runtime must not grow with the number of levels."""
    elapsed = run(make_fe_frame(50_000, n_groups))
    assert elapsed < 15.0, (
        f"{n_groups} levels took {elapsed:.2f}s; absorption should be roughly "
        "independent of cardinality, so this suggests a fallback to one-hot"
    )
    print(f"\n  absorbed, {n_groups:>6} levels: {elapsed:.3f}s")


def test_absorbed_beats_one_hot_at_moderate_cardinality(monkeypatch) -> None:
    """At 150 levels the one-hot path is already orders of magnitude slower.

    Sized to keep CI quick: at 200 levels / n=20,000 the one-hot leg alone takes
    ~30s, and the ratio is just as clear here.
    """
    df = make_fe_frame(10_000, 150)

    absorbed = run(df)

    monkeypatch.setattr(core, "select_absorbed", lambda *a, **k: None)
    one_hot = run(df)

    print(
        f"\n  150 levels: absorbed={absorbed:.3f}s one-hot={one_hot:.3f}s "
        f"({one_hot / absorbed:.0f}x)"
    )
    assert absorbed < one_hot, (
        f"absorbed ({absorbed:.3f}s) should beat one-hot ({one_hot:.3f}s)"
    )
    assert one_hot / absorbed > 10, (
        f"expected a large speedup at 150 levels, got {one_hot / absorbed:.1f}x"
    )


def test_high_cardinality_is_tractable() -> None:
    """50k levels: ~1.25 billion aggregations on the one-hot path, i.e. never."""
    elapsed = run(make_fe_frame(200_000, 50_000))
    assert elapsed < 60.0, f"50k levels took {elapsed:.2f}s"
    print(f"\n  absorbed, 50,000 levels, n=200,000: {elapsed:.3f}s")


def test_select_absorbed_overhead_is_bounded() -> None:
    """select_absorbed runs on every call with a categorical control.

    It is an exact COUNT(DISTINCT), i.e. a full extra pass. Approximate counts
    were considered and rejected: narwhals does not expose one, and duckdb's HLL
    returned 1249 for 1000 true levels -- enough error to flip the absorb decision
    near the threshold and make bin selection irreproducible.
    """
    df = make_fe_frame(200_000, 5_000)
    lazy = nw.from_native(df).lazy()

    with timer() as t_count:
        fe_mod.select_absorbed(lazy, ("firm_id",))

    with timer() as t_group:
        lazy.group_by("firm_id").agg(
            nw.len().alias("c"), nw.col("y").sum().alias("s")
        ).collect()

    share = t_count["elapsed"] / (t_count["elapsed"] + t_group["elapsed"])
    print(
        f"\n  select_absorbed={t_count['elapsed']:.3f}s "
        f"group_by={t_group['elapsed']:.3f}s share={share:.0%}"
    )
    assert share < 0.8, (
        f"cardinality discovery is {share:.0%} of the aggregation cost; "
        "if this grows, revisit the approximate-count decision above"
    )


def test_threshold_keeps_small_categoricals_cheap(monkeypatch) -> None:
    """Below ABSORB_MIN_LEVELS nothing is absorbed, so cost must match the old path."""
    df = make_fe_frame(50_000, 20)

    default = run(df)

    monkeypatch.setattr(core, "select_absorbed", lambda *a, **k: None)
    legacy = run(df)

    print(f"\n  20 levels: default={default:.3f}s legacy={legacy:.3f}s")
    # Only the extra n_unique pass separates them.
    assert default < legacy * 3 + 1.0, (
        f"default path ({default:.3f}s) is far slower than the legacy one "
        f"({legacy:.3f}s) at a cardinality where nothing should be absorbed"
    )
