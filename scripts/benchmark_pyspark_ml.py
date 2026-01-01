# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "binscatter",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "pyspark",
#     "tabulate",
# ]
#
# [tool.uv.sources]
# binscatter = { path = "..", editable = true }
# ///
"""Benchmark PySpark ML-native regression vs current aggregation approach.

This script compares performance of:
1. Current approach: aggregate to bins, build XTX manually, solve on driver
2. PySpark ML approach: VectorAssembler + LinearRegression (once implemented)

Varies:
- Dataset size (rows)
- Number of controls (numeric + categorical)
- Number of bins

Usage:
    uv run scripts/benchmark_pyspark_ml.py
    uv run scripts/benchmark_pyspark_ml.py --quick  # Smaller test matrix
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    num_rows: int
    num_controls: int
    num_categorical: int
    num_bins: int
    name: str

    @property
    def total_controls(self) -> int:
        return self.num_controls + self.num_categorical


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    current_time: float | None = None
    pyspark_ml_time: float | None = None
    error: str | None = None

    @property
    def speedup(self) -> float | None:
        """Speedup factor (current / pyspark_ml). >1 means PySpark ML is faster."""
        if self.current_time and self.pyspark_ml_time:
            return self.current_time / self.pyspark_ml_time
        return None


def make_synthetic_data(
    num_rows: int, num_controls: int, num_categorical: int, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic data with specified number of controls.

    Args:
        num_rows: Number of observations
        num_controls: Number of numeric controls
        num_categorical: Number of categorical controls
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed=seed)
    x = rng.normal(loc=0.0, scale=1.0, size=num_rows)

    # Generate numeric controls
    controls_data = {}
    control_effects = []
    for i in range(num_controls):
        control_name = f"ctrl_num_{i}"
        controls_data[control_name] = rng.normal(loc=0.0, scale=1.5, size=num_rows)
        control_effects.append(0.1 * controls_data[control_name])

    # Generate categorical controls
    categories_pool = ["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"]
    for i in range(num_categorical):
        control_name = f"ctrl_cat_{i}"
        n_levels = min(3 + i, len(categories_pool))
        controls_data[control_name] = rng.choice(
            categories_pool[:n_levels], size=num_rows
        )
        # Add effect for first category
        control_effects.append(
            0.15 * (controls_data[control_name] == categories_pool[0]).astype(float)
        )

    # Generate y with effects from x and controls
    noise = rng.normal(scale=0.5, size=num_rows)
    y = 0.75 * x + sum(control_effects) + noise

    return pd.DataFrame({"x": x, "y": y, **controls_data})


def get_control_columns(num_controls: int, num_categorical: int) -> list[str]:
    """Get list of control column names."""
    controls = [f"ctrl_num_{i}" for i in range(num_controls)]
    controls += [f"ctrl_cat_{i}" for i in range(num_categorical)]
    return controls


def benchmark_current_approach(
    df: pd.DataFrame, controls: list[str], num_bins: int
) -> float:
    """Benchmark the current aggregation-based approach.

    Returns time in seconds.
    """
    from binscatter import binscatter

    # Import PySpark and create session
    try:
        from pyspark.sql import SparkSession
    except ImportError as err:
        raise RuntimeError("PySpark required for benchmarking") from err

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("binscatter-benchmark-current")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        spark_df = spark.createDataFrame(df)

        # Warm up (first run often slower due to JVM initialization)
        _ = binscatter(spark_df, x="x", y="y", controls=controls, num_bins=num_bins)

        # Actual timed run
        start = time.perf_counter()
        _ = binscatter(spark_df, x="x", y="y", controls=controls, num_bins=num_bins)
        elapsed = time.perf_counter() - start

        return elapsed
    finally:
        spark.stop()


def benchmark_pyspark_ml_approach(
    df: pd.DataFrame, controls: list[str], num_bins: int
) -> float:
    """Benchmark the PySpark ML approach (once implemented).

    For now, returns None to indicate not implemented.
    """
    # TODO: Implement once PySpark ML path is available
    # This will use the new _partial_out_controls_pyspark function
    return None


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    logger.info(f"Running benchmark: {config.name}")
    result = BenchmarkResult(config=config)

    try:
        # Generate data
        df = make_synthetic_data(
            config.num_rows, config.num_controls, config.num_categorical
        )
        controls = get_control_columns(config.num_controls, config.num_categorical)

        # Benchmark current approach
        logger.info("  - Current approach...")
        result.current_time = benchmark_current_approach(df, controls, config.num_bins)
        logger.info(f"    Completed in {result.current_time:.3f}s")

        # Benchmark PySpark ML approach
        logger.info("  - PySpark ML approach...")
        result.pyspark_ml_time = benchmark_pyspark_ml_approach(
            df, controls, config.num_bins
        )
        if result.pyspark_ml_time:
            logger.info(f"    Completed in {result.pyspark_ml_time:.3f}s")
        else:
            logger.info("    Not yet implemented")

    except Exception as e:
        logger.error(f"  - Error: {e}")
        result.error = str(e)

    return result


def create_benchmark_matrix(quick: bool = False) -> list[BenchmarkConfig]:
    """Create matrix of benchmark configurations."""
    if quick:
        # Quick test matrix for development
        return [
            BenchmarkConfig(
                num_rows=50_000,
                num_controls=2,
                num_categorical=1,
                num_bins=20,
                name="small_few_controls",
            ),
            BenchmarkConfig(
                num_rows=100_000,
                num_controls=5,
                num_categorical=2,
                num_bins=50,
                name="medium_many_controls",
            ),
        ]

    # Full benchmark matrix
    configs = []

    # Vary dataset size (controls and bins constant)
    for num_rows in [100_000, 500_000, 1_000_000]:
        configs.append(
            BenchmarkConfig(
                num_rows=num_rows,
                num_controls=3,
                num_categorical=2,
                num_bins=50,
                name=f"vary_size_{num_rows//1000}k",
            )
        )

    # Vary number of controls (size and bins constant)
    for num_controls, num_categorical in [(1, 0), (3, 2), (5, 3), (10, 5)]:
        configs.append(
            BenchmarkConfig(
                num_rows=250_000,
                num_controls=num_controls,
                num_categorical=num_categorical,
                num_bins=50,
                name=f"vary_controls_{num_controls + num_categorical}",
            )
        )

    # Vary number of bins (size and controls constant)
    for num_bins in [20, 50, 100]:
        configs.append(
            BenchmarkConfig(
                num_rows=250_000,
                num_controls=3,
                num_categorical=2,
                num_bins=num_bins,
                name=f"vary_bins_{num_bins}",
            )
        )

    return configs


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback if tabulate not available
        print("\nBenchmark Results:")
        print("-" * 100)
        for r in results:
            print(f"\n{r.config.name}:")
            print(f"  Rows: {r.config.num_rows:,}")
            print(f"  Controls: {r.config.total_controls} (numeric: {r.config.num_controls}, categorical: {r.config.num_categorical})")
            print(f"  Bins: {r.config.num_bins}")
            print(f"  Current approach: {r.current_time:.3f}s" if r.current_time else "  Current approach: N/A")
            print(f"  PySpark ML: {r.pyspark_ml_time:.3f}s" if r.pyspark_ml_time else "  PySpark ML: Not implemented")
            if r.speedup:
                print(f"  Speedup: {r.speedup:.2f}x")
            if r.error:
                print(f"  Error: {r.error}")
        return

    # Use tabulate for nice formatting
    headers = [
        "Config",
        "Rows",
        "Controls",
        "Bins",
        "Current (s)",
        "PySpark ML (s)",
        "Speedup",
    ]

    rows = []
    for r in results:
        speedup_str = f"{r.speedup:.2f}x" if r.speedup else "N/A"
        current_str = f"{r.current_time:.3f}" if r.current_time else "ERROR"
        pyspark_str = f"{r.pyspark_ml_time:.3f}" if r.pyspark_ml_time else "N/A"

        rows.append([
            r.config.name,
            f"{r.config.num_rows:,}",
            r.config.total_controls,
            r.config.num_bins,
            current_str,
            pyspark_str,
            speedup_str,
        ])

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS: PySpark ML vs Current Approach")
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("\nSpeedup > 1.0 means PySpark ML is faster")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PySpark ML regression vs current approach"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test matrix (fewer configurations)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    # Suppress verbose logging from dependencies
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)

    configs = create_benchmark_matrix(quick=args.quick)
    logger.info(f"Running {len(configs)} benchmark configurations...")

    results = []
    for config in configs:
        result = run_benchmark(config)
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
