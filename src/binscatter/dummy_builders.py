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
    if implementation is Implementation.PANDAS:
        return build_dummies_pandas
    if implementation is Implementation.POLARS:
        return build_dummies_polars
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


def build_rename_map(columns, separator: str) -> Tuple[dict[str, str], list[str]]:
    """Build rename mapping and list of dummy column names.

    Args:
        columns: Column names from dummies dataframe
        separator: Separator used in column names (e.g., "__binscatter__")

    Returns:
        Tuple of (rename_map dict, list of new dummy column names)
    """
    rename_map: dict[str, str] = {}
    dummy_cols: list[str] = []
    for name in columns:
        col, value = name.split(separator, 1)
        alias = format_dummy_alias(col, value)
        rename_map[name] = alias
        dummy_cols.append(alias)
    return rename_map, dummy_cols


def build_dummies_pandas(df, categorical_controls: Tuple[str, ...]):
    """Build dummy variables using native pandas implementation.

    Uses pd.get_dummies() for efficient dummy creation.

    Args:
        df: Input dataframe
        categorical_controls: Tuple of categorical column names

    Returns:
        Tuple of (dataframe with dummies, tuple of dummy column names)
    """
    if not categorical_controls:
        return df, ()

    import pandas as pd

    native = nw.to_native(df)
    base = native.copy()
    sep = "__binscatter__"
    dummies = pd.get_dummies(
        base[list(categorical_controls)],
        prefix={c: c for c in categorical_controls},
        prefix_sep=sep,
        drop_first=True,
    )
    rename_map, dummy_cols = build_rename_map(dummies.columns, sep)
    dummies = dummies.rename(columns=rename_map)
    base = base.join(dummies)
    return nw.from_native(base).lazy(), tuple(dummy_cols)


def build_dummies_polars(df, categorical_controls: Tuple[str, ...]):
    """Build dummy variables using native polars implementation.

    Uses pl.to_dummies() for efficient dummy creation.
    Only collects the categorical columns, keeping the main dataframe lazy.

    Args:
        df: Input dataframe
        categorical_controls: Tuple of categorical column names

    Returns:
        Tuple of (dataframe with dummies, tuple of dummy column names)
    """
    if not categorical_controls:
        return df, ()

    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return build_dummies_fallback(df, categorical_controls)

    # Only collect the categorical columns to create dummies
    sep = "__binscatter__"
    categoricals_collected = df.select(categorical_controls).collect()

    # Convert to native polars
    native_categoricals = nw.to_native(categoricals_collected)

    # Create all dummies (drop_first=False) then manually drop first sorted category
    # This ensures consistency with pandas which drops first alphabetically
    dummies_df = native_categoricals.to_dummies(drop_first=False, separator=sep)

    # For each categorical column, drop the first dummy (alphabetically sorted)
    cols_to_drop = []
    for col in categorical_controls:
        # Get dummy column names for this categorical
        dummy_prefix = f"{col}{sep}"
        dummy_cols_for_this_cat = [
            c for c in dummies_df.columns if c.startswith(dummy_prefix)
        ]
        # Sort and drop the first (matches pandas behavior)
        if dummy_cols_for_this_cat:
            sorted_dummies = sorted(dummy_cols_for_this_cat)
            cols_to_drop.append(sorted_dummies[0])

    # Drop the first category for each categorical variable
    if cols_to_drop:
        dummies_df = dummies_df.drop(cols_to_drop)

    # Rename columns using our helper
    rename_map, dummy_cols = build_rename_map(dummies_df.columns, sep)
    dummies_df = dummies_df.rename(rename_map)

    # Convert dummies back to polars LazyFrame
    dummies_lazy = dummies_df.lazy()

    # Horizontally concatenate original lazy frame with dummies
    native_df = nw.to_native(df)
    result_native = pl.concat([native_df, dummies_lazy], how="horizontal")

    return nw.from_native(result_native), tuple(dummy_cols)


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

    native = nw.to_native(df)
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
    """Build dummy variables using narwhals expressions.

    Fallback for backends without specialized implementations.
    Uses narwhals expressions to preserve lazy evaluation.

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
