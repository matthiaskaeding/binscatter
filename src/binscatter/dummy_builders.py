"""Dummy variable builders for categorical controls.

This module provides backend-specific implementations for creating dummy variables
from categorical columns, with optimizations for each supported backend.
"""

import hashlib
from typing import Any, Tuple

import narwhals as nw
from narwhals import Implementation


def configure_build_dummies(implementation: Implementation):  # ty panics on full type
    # Returns: Callable[[nw.LazyFrame, Tuple[str, ...]], Tuple[nw.LazyFrame, Tuple[str, ...]]]
    """Configure and return the appropriate dummy variable builder for the given backend.

    Args:
        implementation: The narwhals Implementation type

    Returns:
        A function that builds dummy variables for the backend
    """
    if implementation in (Implementation.PANDAS, Implementation.POLARS):
        return build_dummies_pandas_polars
    if implementation is Implementation.PYSPARK:
        return build_dummies_pyspark
    return build_dummies_fallback


def format_dummy_alias(column: str, value: Any) -> str:
    """Create a safe, unique dummy variable name.

    Uses a hash suffix to guarantee uniqueness even when different values
    would sanitize to the same string (e.g., "foo/bar" vs "foo_bar").

    Args:
        column: The source categorical column name
        value: The category value

    Returns:
        A safe column name like "__ctrl_category_value_a1b2c3d4"
    """
    import re

    # Convert to string and sanitize: keep only alphanumeric and underscores
    str_value = str(value)
    safe_value = re.sub(r"[^a-zA-Z0-9_]", "_", str_value)

    # Remove consecutive underscores and trim
    safe_value = re.sub(r"_+", "_", safe_value).strip("_")

    # Create a short hash of the original value for uniqueness
    # This prevents collisions like "foo/bar" vs "foo_bar"
    value_hash = hashlib.md5(str_value.encode("utf-8")).hexdigest()[:8]

    # Truncate if too long (leave room for prefix, column, hash, and underscores)
    # Typical database column limit is 64 chars
    # Format: __ctrl_{column}_{safe_value}_{8-char-hash}
    prefix_len = len(f"__ctrl_{column}_")
    hash_len = 8  # MD5 hash truncated to 8 chars
    max_value_len = 64 - prefix_len - hash_len - 1  # -1 for underscore before hash

    if len(safe_value) > max_value_len:
        safe_value = safe_value[:max_value_len]

    # Always include hash to guarantee uniqueness
    return f"__ctrl_{column}_{safe_value}_{value_hash}"


def build_dummies_pandas_polars(df, categorical_controls: Tuple[str, ...]):
    """Build dummy variables using native pandas/polars implementations.

    Uses pd.get_dummies() or pl.to_dummies() for efficient dummy creation.

    Args:
        df: Input dataframe
        categorical_controls: Tuple of categorical column names

    Returns:
        Tuple of (dataframe with dummies, tuple of dummy column names)
    """
    if not categorical_controls:
        return df, ()

    native = df._compliant_frame.native
    dummy_cols: list[str] = []

    if native.__class__.__module__.startswith("pandas"):
        import pandas as pd

        base = native.copy()
        sep = "__binscatter__"
        dummies = pd.get_dummies(
            base[list(categorical_controls)],
            prefix={c: c for c in categorical_controls},
            prefix_sep=sep,
            drop_first=True,
        )
        rename_map = {}
        for name in dummies.columns:
            col, value = name.split(sep, 1)
            alias = format_dummy_alias(col, value)
            rename_map[name] = alias
            dummy_cols.append(alias)
        dummies = dummies.rename(columns=rename_map)
        base = base.join(dummies)
        return nw.from_native(base).lazy(), tuple(dummy_cols)

    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return build_dummies_fallback(df, categorical_controls)

    if isinstance(native, pl.LazyFrame):
        dataset = native.collect()
    else:
        dataset = native
    sep = "__binscatter__"
    dummies_df = dataset.select(list(categorical_controls)).to_dummies(
        drop_first=True, separator=sep
    )
    rename_map: dict[str, str] = {}
    for name in dummies_df.columns:
        col, value = name.split(sep, 1)
        alias = format_dummy_alias(col, value)
        rename_map[name] = alias
        dummy_cols.append(alias)
    dummies_df = dummies_df.rename(rename_map)
    dataset = dataset.hstack(dummies_df)
    return nw.from_native(dataset).lazy(), tuple(dummy_cols)


def build_dummies_pyspark(df, categorical_controls: Tuple[str, ...]):
    """Build dummy variables using PySpark with batched aggregation.

    Batches all categorical discovery into a single agg(*collect_set(...)) call
    to avoid multiple scans. This is the key optimization for PySpark.

    Args:
        df: Input dataframe
        categorical_controls: Tuple of categorical column names

    Returns:
        Tuple of (dataframe with dummies, tuple of dummy column names)
    """
    if not categorical_controls:
        return df, ()

    from pyspark.sql import functions as F  # type: ignore[import-not-found]

    native = df._compliant_frame.native
    agg_exprs = [
        F.sort_array(F.collect_set(F.col(column))).alias(column)
        for column in categorical_controls
    ]
    distinct_row = native.agg(*agg_exprs).collect()[0]
    updated = native
    dummy_cols: list[str] = []
    for column in categorical_controls:
        values = distinct_row[column] or []
        categories = sorted(v for v in values if v is not None)
        if len(categories) <= 1:
            continue
        for value in categories[1:]:
            alias = format_dummy_alias(column, value)
            updated = updated.withColumn(
                alias,
                F.when(F.col(column) == value, F.lit(1.0)).otherwise(0.0),
            )
            dummy_cols.append(alias)

    return nw.from_native(updated).lazy(), tuple(dummy_cols)


def build_dummies_fallback(df, categorical_controls: Tuple[str, ...]):
    """Build dummy variables using generic narwhals implementation.

    Fallback for backends without specialized implementations.

    Args:
        df: Input dataframe
        categorical_controls: Tuple of categorical column names

    Returns:
        Tuple of (dataframe with dummies, tuple of dummy column names)
    """
    if not categorical_controls:
        return df, ()

    from typing import Any, List

    dummy_exprs = []
    dummy_cols: list[str] = []
    for column in categorical_controls:
        distinct_values: List[Any] = (
            df.select(column).unique().collect().get_column(column).sort().to_list()
        )
        if len(distinct_values) <= 1:
            continue
        for value in distinct_values[1:]:
            alias = format_dummy_alias(column, value)
            expr = (nw.col(column) == value).cast(nw.Float64).alias(alias)
            dummy_exprs.append(expr)
            dummy_cols.append(alias)

    if not dummy_exprs:
        return df, ()

    return df.with_columns(*dummy_exprs), tuple(dummy_cols)
