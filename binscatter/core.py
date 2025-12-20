from collections.abc import Iterable as IterableABC
import logging
import math
import operator
import uuid
from functools import reduce
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Tuple,
    cast,
    overload,
)

import narwhals as nw
import narwhals.selectors as ncs
import numpy as np
import plotly.express as px
from narwhals import Implementation
from narwhals.typing import IntoDataFrame
from plotly import graph_objects as go

logger = logging.getLogger(__name__)


@overload
def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    controls: Iterable[str] | str | None = None,
    num_bins=20,
    return_type: Literal["plotly"] = "plotly",
) -> go.Figure: ...


@overload
def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    controls: Iterable[str] | str | None = None,
    num_bins=20,
    return_type: Literal["native"] = "native",
) -> object: ...


def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    controls: Iterable[str] | str | None = None,
    num_bins: int = 20,
    return_type: Literal["plotly", "native"] = "plotly",
) -> object:
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df (IntoDataFrame): Input dataframe - must be a type supported by narwhals
        x (str): Name of x column
        y (str): Name y column
        controls (Iterable[str]): Names of control variables (numeric). These are partialled out
            following Cattaneo et al. (2024).
        num_bins (int, optional): Number of bins to use. Defaults to 20
        return_type (str): Return type. Default "plotly" gives a plotly plot.
        Otherwise "native" returns a dataframe that is natural match to input dataframe.


    Returns:
        plotly plot (default) if return_type == "plotly". Otherwise native dataframe, depending on input.
    """
    if return_type not in ("plotly", "native"):
        msg = f"Invalid return_type: {return_type}"
        raise ValueError(msg)

    # Prepare dataframe: sort, remove non numerics and add bins
    df_prepped, profile = prep(df, x, y, controls, num_bins)

    # Currently there are 2 cases:
    # (1) no controls: the easy one, just compute the means by bin
    # (2) controls: here we need to compute regression coefficients
    # and partial out the effect of the controls
    # (see section 2.2 in Cattaneo, Crump, Farrell and Feng (2024))
    if not controls:
        df_plotting: nw.LazyFrame = (
            df_prepped.group_by(profile.bin_name)
            .agg(profile.x_col.mean(), profile.y_col.mean())
            .with_columns(nw.col(profile.bin_name).cast(nw.Int32))
        ).lazy()
    else:
        df_plotting = partial_out_controls(df_prepped, profile)

    match return_type:
        case "plotly":
            return make_plot_plotly(df_plotting, profile)
        case "native":
            df_out_nw = df_plotting.rename({profile.bin_name: "bin"}).sort("bin")
            logger.debug(
                "Type of df_out_nw: %s, implementation: %s",
                type(df_out_nw),
                df_out_nw.implementation,
            )

            if profile.implementation in (
                Implementation.PYSPARK,
                Implementation.DUCKDB,
                Implementation.DASK,
            ):
                return df_out_nw.to_native()
            else:
                return df_out_nw.collect().to_native()


class Profile(NamedTuple):
    """Main profile which holds bunch of data derived from dataframe."""

    x_name: str
    y_name: str
    controls: Tuple[str, ...]
    num_bins: int
    bin_name: str
    x_bounds: Tuple[float, float]
    distinct_suffix: str
    is_lazy_input: bool
    implementation: Implementation
    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]

    @property
    def x_col(self) -> nw.Expr:
        return nw.col(self.x_name)

    @property
    def y_col(self) -> nw.Expr:
        return nw.col(self.y_name)


def prep(
    df_in: IntoDataFrame,
    x_name: str,
    y_name: str,
    controls: Iterable[str] | str | None = None,
    num_bins: int = 20,
) -> Tuple[nw.LazyFrame, Profile]:
    """Prepares the input data and derives profile.

    Args:
        df: Input dataframe.
        x_name: name of x col
        y_name: name of y col
        controls: Iterable of control vars
        num_bins: Number of bins to use for binscatter. Must be less than number of rows.

    Returns:
        tuple: (narwhals.LazyFrame, Profile)
            - Sorted input dataframe converted to a narwhals LazyFrame
            - Profile object with metadata about the data

    Raises:
        AssertionError: If input validation fails
    """
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1")
    if not isinstance(x_name, str):
        raise TypeError("x_name must be a string")
    if not isinstance(y_name, str):
        raise TypeError("y_name must be a string")

    if controls is None:
        controls = ()
    elif isinstance(controls, str):
        controls = (controls,)
    else:
        try:
            controls = tuple(controls)
        except TypeError:
            raise TypeError(
                f"controls must be a string, iterable, or None, got {type(controls)}"
            )
    if not all(isinstance(c, str) for c in controls):
        raise TypeError("controls must contain only strings")

    dfn: nw.DataFrame | nw.LazyFrame = nw.from_native(df_in)
    logger.debug("Type after calling to native: %s", type(dfn.to_native()))
    if type(dfn) is nw.DataFrame:
        is_lazy_input = False
    elif type(dfn) is nw.LazyFrame:
        is_lazy_input = True
    else:
        msg = f"Unexpected narwhals type {(type(dfn))}"
        raise ValueError(msg)
    dfl: nw.LazyFrame = dfn.lazy()

    try:
        df = dfl.select(x_name, y_name, *controls)
    except Exception as e:
        cols = dfl.columns
        for c in [x_name, y_name, *controls]:
            if c not in cols:
                msg = f"{c} not in input dataframe"
                raise ValueError(msg)
        raise e

    assert num_bins > 1

    # Find name for bins
    distinct_suffix = str(uuid.uuid4()).replace("-", "_")
    bin_name = f"bins____{distinct_suffix}"

    cols_numeric, cols_cat = get_columns(df)
    union_cols = set(cols_numeric) | set(cols_cat)
    if set(df.columns) - union_cols:
        missing = [c for c in df.columns if c not in union_cols]
        msg = f"Columns with unsupported types: {missing}"
        raise TypeError(msg)
    missing_controls = [c for c in controls if c not in union_cols]
    if missing_controls:
        msg = f"Unknown control columns (neither numeric nor categorical): {missing_controls}"
        raise TypeError(msg)

    df_filtered = _remove_bad_values(df, cols_numeric, cols_cat)

    # We need the range of x for plotting
    bounds_df = df_filtered.select(
        nw.col(x_name).min().alias("x_min"),
        nw.col(x_name).max().alias("x_max"),
    ).collect()
    x_bounds = (bounds_df.item(0, "x_min"), bounds_df.item(0, "x_max"))
    for val, fun in zip(x_bounds, ["min", "max"]):
        if not math.isfinite(val):
            msg = f"{fun}({x_name})={val}"
            raise ValueError(msg)

    profile = Profile(
        num_bins=num_bins,
        x_name=x_name,
        y_name=y_name,
        bin_name=bin_name,
        controls=controls,
        x_bounds=x_bounds,
        distinct_suffix=distinct_suffix,
        is_lazy_input=is_lazy_input,
        implementation=df_filtered.implementation,
        numeric_columns=cols_numeric,
        categorical_columns=cols_cat,
    )
    logger.debug("Profile: %s", profile)

    quantile_handler = configure_quantile_handler(profile)
    try:
        df_with_bins = quantile_handler(df_filtered)
    except ValueError as err:
        err_text = str(err)
        if (
            "Quantiles are not unique" in err_text
            or "Bin edges must be unique" in err_text
        ):
            raise ValueError(
                "Quantiles are not unique. Decrease number of bins."
            ) from err
        raise

    return df_with_bins.lazy(), profile


def partial_out_controls(df_prepped: nw.LazyFrame, profile: Profile) -> nw.LazyFrame:
    """Compute binscatter means after partialling out controls following Cattaneo et al. (2024)."""

    controls = list(profile.controls)
    if not controls:
        raise ValueError("Controls must be provided for partial_out_controls")

    numeric_controls = [c for c in controls if c in profile.numeric_columns]
    categorical_controls = [c for c in controls if c in profile.categorical_columns]
    unknown_controls = [
        c for c in controls if c not in numeric_controls + categorical_controls
    ]
    if unknown_controls:
        msg = f"Controls with unsupported types: {unknown_controls}"
        raise TypeError(msg)

    control_aliases: list[str] = []
    new_columns = []

    for c in numeric_controls:
        alias = f"__ctrl_{len(control_aliases)}"
        new_columns.append(nw.col(c).cast(nw.Float64).alias(alias))
        control_aliases.append(alias)

    dummy_exprs: list[nw.Expr] = []
    dummy_aliases: list[str] = []
    for c in categorical_controls:
        if c not in profile.categorical_columns:
            raise TypeError(f"Control '{c}' is not recognized as categorical")

        unique_values = df_prepped.select(c).unique().collect().get_column(c)
        values = unique_values.to_list()
        if len(values) <= 1:
            continue
        for value in values[1:]:
            alias = f"__ctrl_{len(control_aliases) + len(dummy_aliases)}"
            expr = (nw.col(c) == value).cast(nw.Float64).alias(alias)
            dummy_exprs.append(expr)
            dummy_aliases.append(alias)

    df_augmented = (
        df_prepped.with_columns(*new_columns, *dummy_exprs)
        if (new_columns or dummy_exprs)
        else df_prepped
    )
    control_aliases.extend(dummy_aliases)

    bin_index = profile.bin_name

    agg_exprs = [
        nw.len().alias("__count"),
        profile.x_col.mean().alias(profile.x_name),
        profile.y_col.sum().alias("__sum_y"),
    ]
    agg_exprs.extend(nw.col(alias).sum().alias(alias) for alias in control_aliases)

    per_bin = (
        df_augmented.group_by(bin_index).agg(*agg_exprs).sort(bin_index)
    ).collect()

    counts = per_bin.get_column("__count").to_numpy()
    if counts.size < profile.num_bins:
        msg = "Quantiles are not unique. Decrease number of bins."
        raise ValueError(msg)
    sum_y = per_bin.get_column("__sum_y").to_numpy()
    if control_aliases:
        bin_control_sums = np.column_stack(
            [per_bin.get_column(alias).to_numpy() for alias in control_aliases]
        )
    else:
        bin_control_sums = np.zeros((profile.num_bins, 0))

    total_exprs = [nw.len().alias("__total_count")]
    total_exprs.extend(
        nw.col(alias).sum().alias(f"__total_ctrl_{idx}")
        for idx, alias in enumerate(control_aliases)
    )
    total_exprs.extend(
        (nw.col(alias) * profile.y_col).sum().alias(f"__wy_{idx}")
        for idx, alias in enumerate(control_aliases)
    )
    for i, alias_i in enumerate(control_aliases):
        for j, alias_j in enumerate(control_aliases[i:], start=i):
            total_exprs.append(
                (nw.col(alias_i) * nw.col(alias_j)).sum().alias(f"__ww_{i}_{j}")
            )

    totals = df_augmented.select(*total_exprs).collect()
    total_count = totals.item(0, "__total_count")
    if control_aliases:
        total_ctrl_sums = np.array(
            [
                totals.item(0, f"__total_ctrl_{idx}")
                for idx in range(len(control_aliases))
            ]
        )
        wy = np.array(
            [totals.item(0, f"__wy_{idx}") for idx in range(len(control_aliases))]
        )
        ww = np.zeros((len(control_aliases), len(control_aliases)))
        for i in range(len(control_aliases)):
            for j in range(i, len(control_aliases)):
                alias = f"__ww_{i}_{j}"
                value = totals.item(0, alias)
                ww[i, j] = value
                ww[j, i] = value
    else:
        total_ctrl_sums = np.array([])
        wy = np.array([])
        ww = np.zeros((0, 0))

    # Assemble normal equations
    num_bins = profile.num_bins
    k = len(control_aliases)
    size = num_bins + k
    XTX = np.zeros((size, size))
    XTy = np.zeros(size)

    XTX[:num_bins, :num_bins] = np.diag(counts)
    if k:
        XTX[:num_bins, num_bins:] = bin_control_sums
        XTX[num_bins:, :num_bins] = bin_control_sums.T
        XTX[num_bins:, num_bins:] = ww
        XTy[num_bins:] = wy

    XTy[:num_bins] = sum_y

    try:
        theta = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        theta, *_ = np.linalg.lstsq(XTX, XTy, rcond=None)

    beta = theta[:num_bins]
    gamma = theta[num_bins:]
    mean_controls = total_ctrl_sums / total_count if k else np.array([])
    fitted = beta + (mean_controls @ gamma if k else 0.0)

    y_vals = nw.new_series(
        name=profile.y_name, values=fitted, backend=per_bin.implementation
    )

    df_plotting = per_bin.select(bin_index, profile.x_name).with_columns(y_vals).lazy()

    return df_plotting


def make_plot_plotly(df_prepped: nw.LazyFrame, profile: Profile) -> go.Figure:
    """Make plot from prepared dataframe.

    Args:
      df_prepped (nw.LazyFrame): Prepared dataframe. Has three columns: bin, x, y with names in profile"""
    data = df_prepped.select(profile.x_name, profile.y_name).collect()
    if data.shape[0] < profile.num_bins:
        raise ValueError("Quantiles are not unique. Decrease number of bins.")

    x = data.get_column(profile.x_name).to_list()
    if len(set(x)) < profile.num_bins:
        msg = f"Unique number of bins is {len(set(x))} fewer than {profile.num_bins} as desired. Decrease parameter num_bins."
        raise ValueError(msg)
    y = data.get_column(profile.y_name).to_list()

    fig = px.scatter(x=x, y=y, range_x=profile.x_bounds).update_layout(
        title="Binscatter",
        xaxis_title=profile.x_name,
        yaxis_title=profile.y_name,
    )

    return fig


def _remove_bad_values(
    df: nw.LazyFrame, cols_numeric: Iterable[str], cols_cat: Iterable[str]
) -> nw.LazyFrame:
    """Removes nulls and non-finite values for the provided columns."""

    bad_conditions = []

    for c in cols_numeric:
        col = nw.col(c)
        bad_conditions.append(col.is_null() | ~col.is_finite() | col.is_nan())

    for c in cols_cat:
        bad_conditions.append(nw.col(c).is_null())

    if not bad_conditions:
        return df

    final_bad_condition = reduce(operator.or_, bad_conditions)

    return df.filter(~final_bad_condition)


def get_columns(
    frame: nw.LazyFrame | nw.DataFrame,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return tuples of numeric and categorical column names for a narwhals frame."""

    def _safe_columns(selection: Any) -> Tuple[str, ...]:
        if selection is None:
            return tuple()
        columns: Tuple[str, ...] = tuple()
        if hasattr(selection, "columns"):
            try:
                columns = tuple(selection.columns)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - backend quirk
                columns = tuple()
        if columns:
            return columns
        if hasattr(selection, "collect_schema"):
            try:
                schema = selection.collect_schema()
            except Exception:  # pragma: no cover - backend quirk
                schema = None
            else:
                if schema is not None:
                    if hasattr(schema, "names") and callable(schema.names):
                        return tuple(schema.names())
                    if isinstance(schema, dict):
                        return tuple(schema.keys())
        return tuple()

    numeric_cols = _safe_columns(frame.select(ncs.numeric()))
    frame_columns = tuple(frame.columns)
    categorical_cols = tuple(col for col in frame_columns if col not in numeric_cols)
    return numeric_cols, categorical_cols


# Quantiles


# Defined here for testability
def _add_fallback(
    df: nw.LazyFrame, profile: Profile, probs: List[float]
) -> nw.LazyFrame:
    try:
        qs = df.select(
            [
                profile.x_col.quantile(p, interpolation="linear").alias(f"q{p}")
                for p in probs
            ]
        ).collect()
    except TypeError:
        expr = cast(Any, profile.x_col)
        qs = df.select([expr.quantile(p).alias(f"q{p}") for p in probs]).collect()
    except Exception as e:
        logger.error(
            "Tried making quantiles with and without interpolation method for df of type: %s",
            type(df),
        )
        raise e
    qs_long = (
        qs.unpivot(variable_name="prob", value_name="quantile")
        .sort("quantile")
        .with_row_index(profile.bin_name)
    )

    quantile_bins = qs_long.select("quantile", profile.bin_name).lazy()

    # Sorting is not always necessary - but for safety we sort
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


def _make_probs(num_bins) -> List[float]:
    return [i / num_bins for i in range(1, num_bins + 1)]


def configure_quantile_handler(profile: Profile) -> Callable:
    probs = _make_probs(profile.num_bins)

    def add_fallback(df: nw.LazyFrame):
        return _add_fallback(df, profile, probs)

    def add_to_dask(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            from pandas import cut
        except ImportError:
            raise ImportError("Dask support requires dask and pandas to be installed.")

        df_native = df.to_native()
        logger.debug("Type of df_native (should be dask): %s", type(df_native))
        quantiles = df_native[profile.x_name].quantile(probs[:-1]).compute()
        bins = (float("-inf"), *quantiles, float("inf"))
        df_native[profile.bin_name] = df_native[profile.x_name].map_partitions(
            cut,
            bins=bins,
            labels=range(len(probs)),
            include_lowest=False,
            right=False,
        )

        return nw.from_native(df_native).lazy()

    def add_to_pandas(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            from pandas import cut
        except ImportError:
            raise ImportError("Pandas support requires pandas to be installed.")
        df_native = df.to_native()
        x = df_native[profile.x_name]
        quantiles = x.quantile(probs[:-1])

        bins = (float("-Inf"), *quantiles, float("Inf"))
        buckets = cut(
            df_native[profile.x_name],
            bins=bins,
            labels=range(len(probs)),
            include_lowest=False,
            right=False,
        )
        df_native[profile.bin_name] = buckets

        return nw.from_native(df_native).lazy()

    def add_to_polars(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars support requires Polars to be installed.")
        # Because cut and qcut are not stable we use when-then
        df_native = df.to_native()
        x_col = pl.col(profile.x_name)

        qs = df_native.select(
            [x_col.quantile(p, interpolation="linear").alias(f"q{p}") for p in probs]
        ).collect()
        expr = pl
        n = qs.width
        for i in range(n):
            thr = qs.item(0, i)
            cond = x_col.le(thr) if i == n - 1 else x_col.lt(thr)
            expr = expr.when(cond).then(pl.lit(i))
        expr = expr.alias(profile.bin_name)
        df_native_with_bin = df_native.with_columns(expr)

        return nw.from_native(df_native_with_bin).lazy()

    def add_to_duckdb(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            import duckdb

            rel = df.to_native()
            assert isinstance(rel, duckdb.DuckDBPyRelation), f"{type(rel)=}"

        except Exception as e:
            raise RuntimeError(
                "Failed to use df.to_native(); DuckDB may not be installed."
            ) from e

        order_expr = f"{profile.x_name} ASC"
        rel_with_bins = rel.project(
            f"*, ntile({len(probs)}) OVER (ORDER BY {order_expr}) - 1 AS {profile.bin_name}"
        )
        assert isinstance(rel_with_bins, duckdb.DuckDBPyRelation), (
            f"{type(rel_with_bins)=}"
        )

        return nw.from_native(rel_with_bins).lazy()

    def add_to_pyspark(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            from pyspark.ml.feature import Bucketizer
            from pyspark.sql.functions import col
        except ImportError as e:
            raise ImportError(
                f"PySpark support requires pyspark to be installed. Original error: {e}"
            ) from e
        sdf = df.to_native()
        qs = sdf.approxQuantile(profile.x_name, (0.0, *probs), relativeError=0.01)
        if logger.isEnabledFor(logging.DEBUG):
            sample = sdf.sample(False, 0.02, seed=1).select(profile.x_name).toPandas()
            pd_qs = sample[profile.x_name].quantile((0.0, *probs)).to_list()
            logger.debug(
                "Pyspark vs pandas (sample) quantiles: %s", list(zip(qs, pd_qs))
            )
        if len(set(qs)) < len(qs):
            raise ValueError("Quantiles not unique. Decrease number of bins.")

        bucketizer = Bucketizer(
            splits=qs,
            inputCol=profile.x_name,
            outputCol=profile.bin_name,
            handleInvalid="keep",
        )

        sdf_binned = bucketizer.transform(sdf).withColumn(
            profile.bin_name, col(profile.bin_name).cast("int")
        )

        return nw.from_native(sdf_binned).lazy()

    if profile.implementation == Implementation.PANDAS:
        return add_to_pandas
    elif profile.implementation == Implementation.POLARS:
        return add_to_polars
    elif profile.implementation == Implementation.PYSPARK:
        return add_to_pyspark
    elif profile.implementation == Implementation.DUCKDB:
        return add_to_duckdb
    elif profile.implementation == Implementation.DASK:
        return add_to_dask
    else:
        return add_fallback


def _compute_quantiles(
    df: nw.DataFrame, colname: str, probs: Iterable[float], bin_name: str
) -> nw.LazyFrame:
    """Get multiple quantiles in one operation"""
    col = nw.col(colname)
    if df.implementation != nw.Implementation.PYSPARK:
        qs = df.select(
            [col.quantile(p, interpolation="linear").alias(f"q{p}") for p in probs]
        )
    else:
        # Pyspark - ugly hack
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            raise ImportError("PySpark support requires pyspark to be installed.")
        spark = SparkSession.builder.getOrCreate()

        quantiles: list[float] = (
            df.select(colname).to_native().approxQuantile(colname, probs, 0.03)
        )
        q_data = {}
        for p, q in zip(probs, quantiles):
            k = f"q{p}"
            q_data[k] = [q]
        qs_spark = spark.createDataFrame(q_data)
        qs = nw.from_native(qs_spark)

    return (
        qs.unpivot(variable_name="prob", value_name="quantile")
        .sort("quantile")
        .with_row_index(bin_name, order_by="quantile")
        .lazy()
    )
