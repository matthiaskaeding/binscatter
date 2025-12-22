from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from binscatter import binscatter


@dataclass
class BenchmarkResult:
    label: str
    seconds: float


def make_dataframe(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    x = rng.normal(size=n_rows)
    w_num = rng.normal(scale=0.5, size=n_rows)
    w_cat = np.where(rng.random(size=n_rows) > 0.5, "A", "B")
    y = 1.5 * x - 0.25 * w_num + rng.normal(scale=0.75, size=n_rows)
    return pd.DataFrame({"x": x, "y": y, "w_num": w_num, "segment": w_cat})


def run_benchmark(df: pd.DataFrame, label_prefix: str) -> Iterable[BenchmarkResult]:
    # Force eager materialization to exclude generation time from measurements.
    df_clean = df.copy()

    start = time.perf_counter()
    controls = ["w_num", "segment"]
    binscatter(
        df_clean,
        "x",
        "y",
        controls=controls,
        num_bins=30,
        return_type="native",
    )
    fixed_time = time.perf_counter() - start

    start = time.perf_counter()
    binscatter(
        df_clean,
        "x",
        "y",
        controls=controls,
        num_bins="rule-of-thumb",
        return_type="native",
    )
    rot_time = time.perf_counter() - start

    yield BenchmarkResult(f"{label_prefix} | fixed=30", fixed_time)
    yield BenchmarkResult(f"{label_prefix} | rule-of-thumb", rot_time)


def main() -> None:
    rng = np.random.default_rng(12345)
    sizes = [
        (100_000, "medium (100k rows)"),
        (1_000_000, "large (1M rows)"),
    ]
    all_results: list[BenchmarkResult] = []
    for n_rows, label in sizes:
        df = make_dataframe(n_rows, rng)
        all_results.extend(run_benchmark(df, label))

    print("Binscatter benchmark (fixed bins vs rule-of-thumb)\n")
    for result in all_results:
        print(f"{result.label:<30s}: {result.seconds:0.3f}s")


if __name__ == "__main__":
    main()
