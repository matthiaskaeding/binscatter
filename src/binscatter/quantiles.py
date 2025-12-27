from __future__ import annotations

import logging
from typing import Callable, List, Tuple, TYPE_CHECKING, Any, cast

import narwhals as nw
from narwhals import Implementation

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from .core import Profile

logger = logging.getLogger(__name__)


def _make_probs(num_bins: int) -> List[float]:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    return [i / num_bins for i in range(num_bins + 1)]


QuantileComputer = Callable[[nw.LazyFrame, str], Tuple[float, ...]]


def configure_compute_quantiles(
    num_bins: int, implementation: Implementation
) -> QuantileComputer:
    """Return a function that computes quantile edges for the given backend."""
    probs = _make_probs(num_bins)

    if implementation == Implementation.PANDAS:
        return lambda df, x: _quantiles_from_pandas(df, x, probs)
    if implementation == Implementation.DASK:
        return lambda df, x: _quantiles_from_dask(df, x, probs)
    if implementation == Implementation.POLARS:
        return lambda df, x: _quantiles_from_polars(df, x, probs)
    if implementation == Implementation.PYSPARK:
        return lambda df, x: _quantiles_from_pyspark(df, x, probs)

    return lambda df, x: _quantiles_fallback(df, x, probs)


def configure_add_bins(
    profile: "Profile",
) -> Callable[[nw.LazyFrame, Tuple[float, ...]], nw.LazyFrame]:
    """Return a function that assigns bin labels given quantiles."""
    if profile.implementation == Implementation.PANDAS:
        return lambda df, quantiles: _assign_bins_pandas(df, profile, quantiles)
    if profile.implementation == Implementation.DASK:
        return lambda df, quantiles: _assign_bins_dask(df, profile, quantiles)
    if profile.implementation == Implementation.POLARS:
        return lambda df, quantiles: _assign_bins_polars(df, profile, quantiles)
    if profile.implementation == Implementation.PYSPARK:
        return lambda df, quantiles: _assign_bins_pyspark(df, profile, quantiles)
    if profile.implementation == Implementation.DUCKDB:
        return lambda df, quantiles: _assign_bins_duckdb(df, profile, quantiles)

    return lambda df, quantiles: _assign_bins_fallback(df, profile, quantiles)


def _quantiles_from_pandas(
    df: nw.LazyFrame, x_name: str, probs: List[float]
) -> Tuple[float, ...]:
    try:
        df_native = df.to_native()
        x = df_native[x_name]
        quantiles = x.quantile(probs)
    except Exception as err:  # pragma: no cover - dependency optional
        raise RuntimeError("Pandas quantile computation failed") from err
    return _to_float_tuple(quantiles)


def _quantiles_from_dask(
    df: nw.LazyFrame, x_name: str, probs: List[float]
) -> Tuple[float, ...]:
    try:
        df_native = df.to_native()
        quantiles = df_native[x_name].quantile(probs).compute()
    except Exception as err:  # pragma: no cover - dependency optional
        raise RuntimeError("Dask quantile computation failed") from err
    return _to_float_tuple(quantiles)


def _quantiles_from_polars(
    df: nw.LazyFrame, x_name: str, probs: List[float]
) -> Tuple[float, ...]:
    try:
        import polars as pl
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("Polars support requires polars to be installed") from err

    df_native = df.to_native()
    x_col = pl.col(x_name)
    exprs = [x_col.quantile(p, interpolation="linear").alias(f"q{p}") for p in probs]
    qs = df_native.select(exprs).collect()
    return tuple(float(qs.item(0, i)) for i in range(qs.width))


def _quantiles_from_pyspark(
    df: nw.LazyFrame, x_name: str, probs: List[float]
) -> Tuple[float, ...]:
    try:
        sdf = df.to_native()
        splits = sdf.approxQuantile(
            x_name,
            list(probs),
            relativeError=0.01,
        )
    except Exception as err:  # pragma: no cover - optional dependency
        raise RuntimeError("PySpark quantile computation failed") from err
    return tuple(float(v) for v in splits)


def _quantiles_fallback(
    df: nw.LazyFrame, x_name: str, probs: List[float]
) -> Tuple[float, ...]:
    x_expr = nw.col(x_name)
    try:
        qs = df.select(
            [x_expr.quantile(p, interpolation="linear").alias(f"q{p}") for p in probs]
        ).collect()
    except TypeError:
        expr = cast(Any, x_expr)
        qs = df.select([expr.quantile(p).alias(f"q{p}") for p in probs]).collect()
    except Exception as err:  # pragma: no cover - defensive logging
        logger.error("Fallback quantile computation failed for df type %s", type(df))
        raise err
    qs_nw = qs if hasattr(qs, "to_native") else nw.from_native(qs)
    num_cols = qs_nw.shape[1]
    return tuple(float(qs_nw.item(0, i)) for i in range(num_cols))


def _assign_bins_pandas(
    df: nw.LazyFrame, profile: "Profile", quantiles: Tuple[float, ...]
) -> nw.LazyFrame:
    try:
        from pandas import cut
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("Pandas support requires pandas to be installed.") from err

    df_native = df.to_native()
    # Convert to bin edges: replace first/last with -inf/+inf
    # Filter out interior values equal to min (they create empty first bin)
    q_min, q_max = quantiles[0], quantiles[-1]
    interior = tuple(q for q in quantiles[1:-1] if q != q_min)
    # If interior is empty but min != max, use max as threshold for 2 bins
    if not interior and q_min != q_max:
        interior = (q_max,)
    edges = (float("-inf"), *interior, float("inf"))
    num_labels = len(edges) - 1
    buckets = cut(
        df_native[profile.x_name],
        bins=edges,
        labels=range(num_labels),
        right=False,
    )
    df_native[profile.bin_name] = buckets
    return nw.from_native(df_native).lazy()


def _assign_bins_dask(
    df: nw.LazyFrame, profile: "Profile", quantiles: Tuple[float, ...]
) -> nw.LazyFrame:
    try:
        from pandas import cut
    except ImportError as err:  # pragma: no cover
        raise ImportError("Dask support requires pandas to be installed.") from err

    df_native = df.to_native()
    # Convert to bin edges: replace first/last with -inf/+inf
    # Filter out interior values equal to min (they create empty first bin)
    q_min, q_max = quantiles[0], quantiles[-1]
    interior = tuple(q for q in quantiles[1:-1] if q != q_min)
    # If interior is empty but min != max, use max as threshold for 2 bins
    if not interior and q_min != q_max:
        interior = (q_max,)
    edges = (float("-inf"), *interior, float("inf"))
    labels = range(len(edges) - 1)
    df_native[profile.bin_name] = df_native[profile.x_name].map_partitions(
        cut,
        bins=edges,
        labels=labels,
        right=False,
    )
    return nw.from_native(df_native).lazy()


def _assign_bins_polars(
    df: nw.LazyFrame, profile: "Profile", quantiles: Tuple[float, ...]
) -> nw.LazyFrame:
    try:
        import polars as pl
    except ImportError as err:  # pragma: no cover
        raise ImportError("Polars support requires polars to be installed") from err

    df_native = df.to_native()
    x_col = pl.col(profile.x_name)
    # Interior thresholds are quantiles[1:-1], filter out values equal to min
    q_min, q_max = quantiles[0], quantiles[-1]
    thresholds = tuple(q for q in quantiles[1:-1] if q != q_min)
    # If thresholds is empty but min != max, use max as threshold for 2 bins
    if not thresholds and q_min != q_max:
        thresholds = (q_max,)
    if not thresholds:
        expr = pl.lit(0).alias(profile.bin_name)
    else:
        expr = pl.when(x_col.lt(thresholds[0])).then(pl.lit(0))
        for idx, thr in enumerate(thresholds[1:], start=1):
            expr = expr.when(x_col.lt(thr)).then(pl.lit(idx))
        expr = expr.otherwise(pl.lit(len(thresholds))).alias(profile.bin_name)
    return nw.from_native(df_native.with_columns(expr)).lazy()


def _assign_bins_pyspark(
    df: nw.LazyFrame, profile: "Profile", quantiles: Tuple[float, ...]
) -> nw.LazyFrame:
    try:
        from pyspark.ml.feature import Bucketizer
        from pyspark.sql.functions import col, lit
    except ImportError as err:  # pragma: no cover
        raise ImportError("PySpark support requires pyspark to be installed.") from err

    sdf = df.to_native()
    # Interior quantiles wrapped with -inf/+inf for Bucketizer
    # Filter out values equal to min (they create empty first bin)
    q_min, q_max = quantiles[0], quantiles[-1]
    interior = tuple(q for q in quantiles[1:-1] if q != q_min)
    # If interior is empty but min != max, use max as threshold for 2 bins
    if not interior and q_min != q_max:
        interior = (q_max,)
    if not interior:
        # Single bin case: assign all to bin 0
        sdf_binned = sdf.withColumn(profile.bin_name, lit(0))
    else:
        splits = [float("-inf"), *interior, float("inf")]
        bucketizer = Bucketizer(
            splits=splits,
            inputCol=profile.x_name,
            outputCol=profile.bin_name,
            handleInvalid="keep",
        )
        sdf_binned = bucketizer.transform(sdf).withColumn(
            profile.bin_name, col(profile.bin_name).cast("int")
        )
    return nw.from_native(sdf_binned).lazy()


def _assign_bins_duckdb(
    df: nw.LazyFrame, profile: "Profile", quantiles: Tuple[float, ...]
) -> nw.LazyFrame:
    try:
        import duckdb  # noqa: F401
    except ImportError as err:  # pragma: no cover
        raise ImportError("DuckDB support requires duckdb to be installed.") from err

    rel = df.to_native()
    # Use CASE WHEN to assign bins based on quantiles
    # Filter out values equal to min (they create empty first bin)
    q_min, q_max = quantiles[0], quantiles[-1]
    thresholds = tuple(q for q in quantiles[1:-1] if q != q_min)
    # If thresholds is empty but min != max, use max as threshold for 2 bins
    if not thresholds and q_min != q_max:
        thresholds = (q_max,)
    if not thresholds:
        case_expr = "0"
    else:
        parts = []
        for idx, thr in enumerate(thresholds):
            parts.append(f"WHEN {profile.x_name} < {thr} THEN {idx}")
        parts.append(f"ELSE {len(thresholds)}")
        case_expr = "CASE " + " ".join(parts) + " END"

    rel_with_bins = rel.project(f"*, {case_expr} AS {profile.bin_name}")
    return nw.from_native(rel_with_bins).lazy()


def _assign_bins_fallback(
    df: nw.LazyFrame, profile: "Profile", quantiles: Tuple[float, ...]
) -> nw.LazyFrame:
    # Build quantile bins frame for join_asof
    # Filter out values equal to min (they create empty first bin)
    q_min, q_max = quantiles[0], quantiles[-1]
    interior = tuple(q for q in quantiles[1:-1] if q != q_min)
    # If interior is empty but min != max, use max as threshold for 2 bins
    if not interior and q_min != q_max:
        interior = (q_max,)
    if not interior:
        # Single bin case
        return df.with_columns(nw.lit(0).alias(profile.bin_name))

    quantile_data = {
        "quantile": list(interior),
        profile.bin_name: list(range(len(interior))),
    }
    quantile_bins = nw.from_dict(quantile_data, backend=df.implementation).lazy()

    return (
        df.sort(profile.x_name)
        .join_asof(
            quantile_bins,
            left_on=profile.x_name,
            right_on="quantile",
            strategy="forward",
        )
        .drop("quantile")
    )


def _to_float_tuple(values: Any) -> Tuple[float, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple)):
        seq = values
    else:
        try:
            seq = list(values)
        except TypeError:
            seq = [values]
    return tuple(float(v) for v in seq)
