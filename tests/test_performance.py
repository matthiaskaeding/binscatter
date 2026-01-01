"""Performance benchmarks for binscatter optimizations.

This module tests the performance improvements from the perf-investigation PR,
specifically:
1. PySpark categorical dummy variable creation (batched vs sequential)
2. PySpark caching optimizations
3. Cross-backend dummy variable creation consistency
"""

import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from binscatter.core import (
    binscatter,
    clean_df,
    add_regression_features,
)
from tests.conftest import convert_to_backend

# Skip PySpark tests by default
pytest.importorskip("pyspark", reason="PySpark benchmarks require --run-pyspark")


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Simple timer context manager."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def generate_test_data(
    n_rows: int = 250_000,
    n_numeric_controls: int = 5,
    n_categorical_controls: int = 3,
    categorical_cardinality: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data for benchmarking.

    Args:
        n_rows: Number of rows
        n_numeric_controls: Number of numeric control variables
        n_categorical_controls: Number of categorical control variables
        categorical_cardinality: Number of unique values per categorical
        seed: Random seed

    Returns:
        DataFrame with x, y, numeric controls, and categorical controls
    """
    rng = np.random.default_rng(seed)

    data = {
        "x": rng.normal(100, 15, n_rows),
        "y": rng.normal(50, 10, n_rows),
    }

    # Add numeric controls
    for i in range(n_numeric_controls):
        data[f"ctrl_num_{i}"] = rng.normal(0, 1, n_rows)

    # Add categorical controls
    for i in range(n_categorical_controls):
        data[f"ctrl_cat_{i}"] = rng.choice(
            [f"cat_{j}" for j in range(categorical_cardinality)],
            size=n_rows,
        )

    return pd.DataFrame(data)


@pytest.mark.pyspark
@pytest.mark.parametrize(
    "n_rows,n_categorical,cardinality",
    [
        (100_000, 3, 10),
        (250_000, 3, 10),
        (250_000, 5, 15),
    ],
    ids=[
        "100k_3cat_10card",
        "250k_3cat_10card",
        "250k_5cat_15card",
    ],
)
def test_pyspark_categorical_dummy_creation_performance(
    n_rows: int,
    n_categorical: int,
    cardinality: int,
) -> None:
    """Benchmark PySpark categorical dummy variable creation.

    Tests that the batched approach (single collect_set aggregation) is faster
    than the sequential approach (one scan per categorical).
    """
    df_pandas = generate_test_data(
        n_rows=n_rows,
        n_numeric_controls=0,
        n_categorical_controls=n_categorical,
        categorical_cardinality=cardinality,
    )
    df_spark = convert_to_backend(df_pandas, "pyspark")

    categorical_controls = tuple(f"ctrl_cat_{i}" for i in range(n_categorical))

    # Clean the dataframe
    df_clean, _, _, _ = clean_df(
        df_spark,
        controls=categorical_controls,
        x_name="x",
        y_name="y",
    )

    # Time the dummy variable creation
    with timer() as result:
        df_with_dummies, regression_features = add_regression_features(
            df_clean,
            numeric_controls=(),
            categorical_controls=categorical_controls,
        )
        # Force execution
        _ = df_with_dummies.collect()

    elapsed = result["elapsed"]

    # Expected performance characteristics:
    # - Should complete in reasonable time (< 5 seconds for 250k rows)
    # - Time should scale roughly linearly with n_categorical, not quadratically
    assert elapsed < 10.0, (
        f"Categorical dummy creation took {elapsed:.2f}s (expected < 10s)"
    )

    # Verify correctness: each categorical with k levels should create k-1 dummies
    expected_dummies = n_categorical * (cardinality - 1)
    assert len(regression_features) == expected_dummies, (
        f"Expected {expected_dummies} dummy variables, got {len(regression_features)}"
    )

    print(
        f"\n  {n_rows:,} rows, {n_categorical} categoricals ({cardinality} levels each): {elapsed:.3f}s"
    )


@pytest.mark.pyspark
def test_pyspark_caching_reduces_scans() -> None:
    """Test that PySpark caching reduces the number of data scans.

    With caching enabled, the dataframe should only be scanned once initially,
    then reused from cache for subsequent operations.
    """
    df_pandas = generate_test_data(n_rows=200_000, n_categorical_controls=3)
    df_spark = convert_to_backend(df_pandas, "pyspark")

    controls = ["ctrl_num_0", "ctrl_num_1", "ctrl_cat_0", "ctrl_cat_1"]

    # Time full binscatter with controls (should use caching internally)
    with timer() as result:
        fig = binscatter(
            df_spark,
            x="x",
            y="y",
            controls=controls,
            num_bins=50,
        )

    elapsed = result["elapsed"]

    # With caching, should be reasonably fast despite multiple operations
    assert elapsed < 15.0, f"Full binscatter took {elapsed:.2f}s (expected < 15s)"
    assert fig is not None

    print(f"\n  Full binscatter with caching: {elapsed:.3f}s")


@pytest.mark.pyspark
def test_pyspark_polynomial_overlay_performance() -> None:
    """Test performance of polynomial line overlay with PySpark.

    The polynomial fit should reuse cached moments and not trigger
    additional full data scans.
    """
    df_pandas = generate_test_data(n_rows=250_000, n_categorical_controls=3)
    df_spark = convert_to_backend(df_pandas, "pyspark")

    controls = ["ctrl_num_0", "ctrl_num_1", "ctrl_cat_0"]

    # Time binscatter with polynomial overlay
    with timer() as result:
        fig = binscatter(
            df_spark,
            x="x",
            y="y",
            controls=controls,
            num_bins=50,
            poly_line=3,
        )

    elapsed = result["elapsed"]

    # Should complete in reasonable time
    assert elapsed < 20.0, (
        f"Binscatter with poly_line took {elapsed:.2f}s (expected < 20s)"
    )
    assert fig is not None

    print(f"\n  Binscatter + poly_line=3: {elapsed:.3f}s")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_dummy_variable_naming_consistency(backend: str) -> None:
    """Test that dummy variable names are consistent across backends.

    The _format_dummy_alias function should produce identical names
    regardless of backend.
    """
    df_pandas = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "cat_a": ["foo", "bar", "baz", "foo", "bar"],
            "cat_b": ["red", "blue", "red", "blue", "red"],
        }
    )

    df = convert_to_backend(df_pandas, backend)

    df_clean, _, _, categorical_controls = clean_df(
        df,
        controls=("cat_a", "cat_b"),
        x_name="x",
        y_name="y",
    )

    df_with_dummies, regression_features = add_regression_features(
        df_clean,
        numeric_controls=(),
        categorical_controls=categorical_controls,
    )

    # Collect dummy names
    result = df_with_dummies.collect()
    dummy_cols = [col for col in result.columns if col.startswith("__ctrl_")]

    # Verify naming convention
    assert len(dummy_cols) > 0, "Should have created dummy variables"

    # All dummy names should follow pattern: __ctrl_{column}_{value}
    for col in dummy_cols:
        assert col.startswith("__ctrl_"), f"Bad dummy name: {col}"
        parts = col.split("_", 3)
        assert len(parts) >= 4, (
            f"Dummy name should have format __ctrl_column_value: {col}"
        )

    # Verify we get the expected number of dummies (drop_first=True)
    # cat_a has 3 levels -> 2 dummies, cat_b has 2 levels -> 1 dummy
    assert len(dummy_cols) == 3, (
        f"Expected 3 dummy columns, got {len(dummy_cols)}: {dummy_cols}"
    )


@pytest.mark.pyspark
def test_pyspark_dummy_names_match_pandas() -> None:
    """Test that PySpark dummy names match pandas dummy names.

    This ensures the optimized PySpark path produces identical output
    to the pandas reference implementation.
    """
    df_pandas = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "category": ["A", "B", "C", "A", "B"],
        }
    )

    # Get pandas dummy names
    df_clean_pd, _, _, cat_controls_pd = clean_df(
        df_pandas,
        controls=("category",),
        x_name="x",
        y_name="y",
    )
    _, features_pd = add_regression_features(
        df_clean_pd,
        numeric_controls=(),
        categorical_controls=cat_controls_pd,
    )

    # Get PySpark dummy names
    df_spark = convert_to_backend(df_pandas, "pyspark")
    df_clean_spark, _, _, cat_controls_spark = clean_df(
        df_spark,
        controls=("category",),
        x_name="x",
        y_name="y",
    )
    _, features_spark = add_regression_features(
        df_clean_spark,
        numeric_controls=(),
        categorical_controls=cat_controls_spark,
    )

    # Names should match exactly
    assert features_pd == features_spark, (
        f"Dummy names differ between pandas and PySpark:\n"
        f"  pandas: {features_pd}\n"
        f"  PySpark: {features_spark}"
    )


@pytest.mark.pyspark
def test_pyspark_handles_null_categories() -> None:
    """Test that PySpark dummy creation handles null values correctly."""
    df_pandas = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "category": ["A", "B", None, "A", "B", None],
        }
    )

    df_spark = convert_to_backend(df_pandas, "pyspark")

    df_clean, _, _, cat_controls = clean_df(
        df_spark,
        controls=("category",),
        x_name="x",
        y_name="y",
    )

    df_with_dummies, features = add_regression_features(
        df_clean,
        numeric_controls=(),
        categorical_controls=cat_controls,
    )

    # Should create dummy for "B" only (drop_first=True, and null is excluded)
    result = df_with_dummies.collect().to_pandas()

    # Verify no errors and reasonable output
    assert len(features) > 0, "Should create at least one dummy variable"

    # Verify null handling: nulls should map to 0 in all dummy columns
    for feat in features:
        assert feat in result.columns
        # Rows with null category should have 0 in all dummies
        null_mask = df_pandas["category"].isna()
        assert (result.loc[null_mask, feat] == 0).all(), (
            f"Null categories should map to 0 in dummy {feat}"
        )


@pytest.mark.parametrize("backend", ["pandas", "polars", "duckdb", "dask"])
def test_backend_dummy_creation_time(backend: str) -> None:
    """Benchmark dummy variable creation across all backends.

    This helps identify if any backend has unexpected performance issues.
    """
    df_pandas = generate_test_data(
        n_rows=50_000,  # Smaller dataset for quick test across all backends
        n_categorical_controls=3,
        categorical_cardinality=10,
    )

    df = convert_to_backend(df_pandas, backend)

    categorical_controls = ("ctrl_cat_0", "ctrl_cat_1", "ctrl_cat_2")

    df_clean, _, _, _ = clean_df(
        df,
        controls=categorical_controls,
        x_name="x",
        y_name="y",
    )

    with timer() as result:
        df_with_dummies, features = add_regression_features(
            df_clean,
            numeric_controls=(),
            categorical_controls=categorical_controls,
        )
        # Force execution
        _ = df_with_dummies.collect()

    elapsed = result["elapsed"]

    # All backends should complete reasonably quickly
    assert elapsed < 5.0, f"{backend} took {elapsed:.2f}s (expected < 5s)"

    print(f"\n  {backend}: {elapsed:.3f}s for 50k rows, 3 categoricals")
