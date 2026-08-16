"""Benchmark fixed-effect absorption against explicit one-hot encoding.

Run the short, repeatable suite used for README-sized comparisons with
``make benchmark-fe-fast``. Run the larger scaling exercise with
``make benchmark-fe``. Both commands accept copyable Markdown output through
``uv run scripts/benchmark_fixed_effects.py [--quick] --markdown``.

Correctness belongs in ``tests/test_fixed_effects.py``. This script only measures.
It deliberately reports timings instead of enforcing machine-dependent thresholds.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import median
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd

import binscatter.fixed_effects as fe_mod
from binscatter import binscatter, core

FULL_CARDINALITIES = (50, 100, 200, 1_000, 10_000, 50_000)
QUICK_CARDINALITIES = (10, 25, 50)
FULL_TWO_WAY = ((200, 10), (2_000, 20), (20_000, 50))
QUICK_TWO_WAY = ((200, 10), (1_000, 20))


@dataclass(frozen=True)
class BenchmarkResult:
    """One timing row, independent of how it will be rendered."""

    scenario: str
    rows: int
    levels: str
    cells: int | None
    absorbed_seconds: float
    one_hot_seconds: float | None = None
    sweeps: int | None = None

    @property
    def speedup(self) -> float | None:
        if self.one_hot_seconds is None or self.absorbed_seconds <= 0.0:
            return None
        return self.one_hot_seconds / self.absorbed_seconds


def make_frame(n: int, n_groups: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.integers(0, n_groups, n)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "y": (
                0.7 * x
                + rng.normal(scale=2.0, size=n_groups)[group]
                + rng.normal(size=n)
            ),
            "firm_id": group,
        }
    )


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


@contextmanager
def _forced_path(absorb: bool):
    """Temporarily route categoricals through absorption or explicit dummies."""
    original = core.select_absorbed
    if not absorb:
        core.select_absorbed = lambda *args, **kwargs: ()
    try:
        yield
    finally:
        core.select_absorbed = original


def _measure(call: Callable[[], Any], repeats: int) -> float:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return float(median(timings))


def time_call(
    df: pd.DataFrame,
    *,
    absorb: bool,
    columns: tuple[str, ...] = ("firm_id",),
    repeats: int = 1,
) -> float:
    """Median wall time for one binscatter specification on a forced FE path."""

    def call() -> None:
        with _forced_path(absorb):
            binscatter(
                df,
                "x",
                "y",
                controls=list(columns),
                categorical=list(columns),
                num_bins=10,
                return_type="native",
            )

    return _measure(call, repeats)


def benchmark_cardinality(
    *,
    n: int,
    cardinalities: Sequence[int],
    one_hot_max_levels: int,
    repeats: int,
) -> list[BenchmarkResult]:
    """One absorbed factor as its number of levels grows."""
    results = []
    for n_groups in cardinalities:
        frame = make_frame(n, n_groups)
        absorbed = time_call(frame, absorb=True, repeats=repeats)
        one_hot = (
            time_call(frame, absorb=False, repeats=repeats)
            if n_groups <= one_hot_max_levels
            else None
        )
        results.append(
            BenchmarkResult(
                scenario="one-way cardinality",
                rows=n,
                levels=f"{n_groups:,}",
                cells=int(frame["firm_id"].nunique()),
                absorbed_seconds=absorbed,
                one_hot_seconds=one_hot,
            )
        )
    return results


def benchmark_rows(
    *, rows: Sequence[int], n_groups: int, repeats: int
) -> list[BenchmarkResult]:
    """Absorbed one-way FE scaling in observation count."""
    return [
        BenchmarkResult(
            scenario="row scaling",
            rows=n,
            levels=f"{n_groups:,}",
            cells=int((frame := make_frame(n, n_groups))["firm_id"].nunique()),
            absorbed_seconds=time_call(frame, absorb=True, repeats=repeats),
        )
        for n in rows
    ]


def benchmark_two_way(
    *, n: int, cases: Sequence[tuple[int, int]], repeats: int
) -> list[BenchmarkResult]:
    """Crossed FEs, where cost follows observed intersections rather than rows."""
    results = []
    for n_firms, n_years in cases:
        frame = make_two_way_frame(n, n_firms, n_years)
        row_codes = np.column_stack(
            [frame["firm_id"].to_numpy(), frame["year"].to_numpy()]
        )
        projector = fe_mod.FEProjector.from_row_codes(row_codes, ("firm_id", "year"))
        results.append(
            BenchmarkResult(
                scenario="two-way absorption",
                rows=n,
                levels=f"{n_firms:,} × {n_years:,}",
                cells=projector.num_cells,
                sweeps=count_sweeps(projector),
                absorbed_seconds=time_call(
                    frame,
                    absorb=True,
                    columns=("firm_id", "year"),
                    repeats=repeats,
                ),
            )
        )
    return results


def count_sweeps(projector: fe_mod.FEProjector) -> int:
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


def benchmark_discovery(*, n: int, n_groups: int, repeats: int) -> tuple[float, float]:
    """Time exact cardinality discovery and the group aggregation it precedes."""
    lazy = nw.from_native(make_frame(n, n_groups)).lazy()
    count = _measure(lambda: fe_mod.select_absorbed(lazy, ("firm_id",)), repeats)
    group = _measure(
        lambda: (
            lazy.group_by("firm_id")
            .agg(nw.len().alias("c"), nw.col("y").sum().alias("s"))
            .collect()
        ),
        repeats,
    )
    return count, group


def run_suite(
    *, quick: bool, repeats: int
) -> tuple[list[BenchmarkResult], tuple[float, float]]:
    """Run either the short README suite or the larger exploratory suite."""
    if quick:
        results = benchmark_cardinality(
            n=5_000,
            cardinalities=QUICK_CARDINALITIES,
            one_hot_max_levels=max(QUICK_CARDINALITIES),
            repeats=repeats,
        )
        results += benchmark_rows(
            rows=(5_000, 20_000, 100_000), n_groups=5_000, repeats=repeats
        )
        results += benchmark_two_way(n=20_000, cases=QUICK_TWO_WAY, repeats=repeats)
        discovery = benchmark_discovery(n=100_000, n_groups=5_000, repeats=repeats)
    else:
        results = benchmark_cardinality(
            n=20_000,
            cardinalities=FULL_CARDINALITIES,
            one_hot_max_levels=200,
            repeats=repeats,
        )
        results += benchmark_rows(
            rows=(50_000, 200_000, 1_000_000),
            n_groups=5_000,
            repeats=repeats,
        )
        results += benchmark_two_way(n=50_000, cases=FULL_TWO_WAY, repeats=repeats)
        discovery = benchmark_discovery(n=1_000_000, n_groups=5_000, repeats=repeats)
    return results, discovery


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}s"


def _speedup(result: BenchmarkResult) -> str:
    return "—" if result.speedup is None else f"{result.speedup:.1f}×"


def markdown_report(
    results: Sequence[BenchmarkResult], discovery: tuple[float, float]
) -> str:
    """Render benchmark results as a README-ready Markdown table."""
    lines = [
        "| scenario | rows | levels | observed cells | absorbed | one-hot | speedup | sweeps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.scenario,
                    f"{result.rows:,}",
                    result.levels,
                    "—" if result.cells is None else f"{result.cells:,}",
                    _seconds(result.absorbed_seconds),
                    _seconds(result.one_hot_seconds),
                    _speedup(result),
                    "—" if result.sweeps is None else str(result.sweeps),
                ]
            )
            + " |"
        )
    count, group = discovery
    lines += [
        "",
        (
            f"Cardinality discovery: {count:.3f}s; grouped aggregation: {group:.3f}s; "
            f"discovery share: {count / (count + group):.0%}."
        ),
    ]
    return "\n".join(lines)


def console_report(
    results: Sequence[BenchmarkResult], discovery: tuple[float, float]
) -> str:
    """Compact aligned output for terminal use."""
    lines = [
        (
            f"{'scenario':<22} {'rows':>10} {'levels':>16} {'cells':>10} "
            f"{'absorbed':>10} {'one-hot':>10} {'speedup':>9} {'sweeps':>7}"
        )
    ]
    for result in results:
        lines.append(
            f"{result.scenario:<22} {result.rows:>10,} {result.levels:>16} "
            f"{('—' if result.cells is None else f'{result.cells:,}'):>10} "
            f"{_seconds(result.absorbed_seconds):>10} "
            f"{_seconds(result.one_hot_seconds):>10} {_speedup(result):>9} "
            f"{('—' if result.sweeps is None else result.sweeps):>7}"
        )
    count, group = discovery
    lines += [
        "",
        (
            f"cardinality discovery {count:.3f}s; group_by {group:.3f}s "
            f"({count / (count + group):.0%} discovery share)"
        ),
    ]
    return "\n".join(lines)


def environment_line() -> str:
    """Describe the runtime and machine resources behind a timing report."""
    system = platform.system()
    if system == "Darwin":
        version = platform.mac_ver()[0] or platform.release()
        operating_system = f"macOS {version}"
    else:
        operating_system = f"{system} {platform.release()}"

    cpu_count = os.cpu_count()
    cpu = f"{cpu_count} logical CPUs" if cpu_count is not None else "CPU count unknown"
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        memory = "RAM unknown"
    else:
        memory_gib = memory_bytes / 1024**3
        memory = f"{memory_gib:g} GiB RAM"

    return (
        f"Python {platform.python_version()}, pandas {pd.__version__}; "
        f"{operating_system} {platform.machine()}; {cpu}, {memory}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the short suite intended for routine development and README checks",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="median repetitions per timing (default: 3 quick, 1 full)",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="print a Markdown table"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repeats = args.repeats if args.repeats is not None else (3 if args.quick else 1)
    if repeats < 1:
        raise SystemExit("--repeats must be at least one")
    warnings.filterwarnings("ignore")
    print(environment_line())
    print(f"FE_TOL={fe_mod.FE_TOL:.0e}; MAX_FE_CELLS={fe_mod.MAX_FE_CELLS:,}")
    results, discovery = run_suite(quick=args.quick, repeats=repeats)
    report = (
        markdown_report(results, discovery)
        if args.markdown
        else console_report(results, discovery)
    )
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
