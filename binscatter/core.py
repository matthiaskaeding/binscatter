from typing import Iterable, Literal, Tuple, Callable, Any, List
import numpy as np
import narwhals as nw
from narwhals.typing import IntoDataFrame
from narwhals import Implementation
import plotly.express as px
import narwhals.selectors as ncs
from typing import NamedTuple
import uuid
import math
from functools import reduce
import operator
import logging
import plotly

logger = logging.getLogger(__name__)


def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    controls: Iterable[str] = (),
    num_bins=20,
    return_type: Literal["plotly", "native"] = "plotly",
) -> plotly.graph_objects.Figure | Any:
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df (IntoDataFrame): Input dataframe - must be a type supported by narwhals
        x (str): Name of x column
        y (str): Name y column
        controls (Iterable[str]): names of control variables (not used yet)
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
        )
    else:
        x_controls = make_controls_mat(df_prepped, controls)
        x_bins = make_b(df_prepped, profile)
        df_plotting = partial_out_controls(x_bins, x_controls, df_prepped, profile)

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
    controls: Tuple[str]
    num_bins: int
    bin_name: str
    x_bounds: Tuple[float, float]
    distinct_suffix: str
    is_lazy_input: bool
    implementation: Implementation

    @property
    def x_col(self) -> nw.Expr:
        return nw.col(self.x_name)

    @property
    def y_col(self) -> nw.Expr:
        return nw.col(self.y_name)


def prep(
    df_in: IntoDataFrame,
    x_name: str | None,
    y_name: str | None,
    controls: Iterable[str] | str = (),
    num_bins: int = 20,
) -> Tuple[nw.DataFrame, Profile]:
    """Prepares the input data and derives profile.

    Args:
        df: Input dataframe.
        x_name: name of x col
        y_name: name of y col
        controls: Iterable of control vars
        num_bins: Number of bins to use for binscatter. Must be less than number of rows.

    Returns:
        tuple: (polars.DataFrame, Config)
            - Sorted input dataframe converted to polars
            - Config object with metadata about the data

    Raises:
        AssertionError: If input validation fails
    """
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1")
    if not isinstance(x_name, str):
        raise TypeError("x_name must be a string")
    if not isinstance(y_name, str):
        raise TypeError("y_name must be a string")
    if controls:
        raise NotImplementedError("Controls not yet implemented")

    dfn = nw.from_native(df_in)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Type after calling to native: %s", type(dfn.to_native()))
    if type(dfn) is nw.DataFrame:
        is_lazy_input = False
    elif type(dfn) is nw.LazyFrame:
        is_lazy_input = True
    else:
        msg = f"Unexpected narwhals type {(type(dfn))}"
        raise ValueError(msg)
    dfl = dfn.lazy()

    if isinstance(controls, str):
        controls = (controls,)
    else:
        controls = tuple(controls)

    try:
        df = dfl.select(x_name, y_name, *controls)
        assert df
    except Exception as e:
        cols = dfl.columns
        for c in [x_name, y_name, *controls]:
            if c not in cols:
                msg = f"{x_name} not in input dataframe"
                raise ValueError(msg)
        raise e

    assert num_bins > 1

    # Find name for bins
    distinct_suffix = str(uuid.uuid4()).replace("-", "_")
    bin_name = f"bins____{distinct_suffix}"
    df_filtered = _remove_bad_values(df)

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
    )
    logger.debug("Profile: %s", profile)

    quantile_handler = configure_quantile_handler(profile)
    df_with_bins = quantile_handler(df_filtered)

    return df_with_bins, profile


def make_controls_mat(df: nw.DataFrame, controls: Iterable[str]) -> np.ndarray:
    """Creates a matrix of control variables by combining numeric and categorical features.

    Args:
        df (pl.DataFrame): Input dataframe containing control variables
        controls (Iterable[str]): Names of control variables to include

    Returns:
        numpy.ndarray: Matrix of control variables with numeric columns and dummy-encoded categorical columns
    """
    dfc = df.select(controls)
    df_num = dfc.select(ncs.numeric())
    df_cat = dfc.select(~ncs.categorical())
    if df_cat.shape[1] == 0:
        return df_num.to_numpy()

    sub_dfs = []
    for col in df_cat.columns:
        tmp = df_cat.get_column(col).to_dummies(drop_first=True).to_numpy()
        sub_dfs.append(tmp)
    x_cat = np.concat(sub_dfs, axis=1)
    if df_num.shape[1] == 0:
        return x_cat
    x_num = df_num.to_numpy()

    return np.concat((x_num, x_cat), axis=1)


def make_b(df_prepped: nw.DataFrame, profile: Profile):
    """Makes the design matrix corresponding to the bins"""

    B = df_prepped.select(profile.bin_name).to_dummies(drop_first=False, separator="_")
    # assert B.shape[1] == config.num_bins, (
    #     f"B must have {config.num_bins} columns but has {B.width}, this indicates too many bins"
    # )
    # Reorder so that column i corresponds always to bin i
    cols = [f"{profile.bin_name}_{i}" for i in range(profile.num_bins)]

    return B.select(cols).to_numpy()


def partial_out_controls(
    x_bins: np.array, x_controls: np.ndarray, df_prepped: nw.DataFrame, profile: Profile
) -> nw.DataFrame:
    """Concatenate bin dummies and control variables horizontally."""
    x_conc = np.concatenate((x_bins, x_controls), axis=1)

    y = df_prepped.get_column(profile.y_name).to_numpy()
    theta = np.linalg.lstsq(x_conc, y)[0]
    x_controls_means = np.mean(x_controls, axis=0)
    # Extract coefficients
    beta = theta[: profile.num_bins]
    gamma = theta[profile.num_bins :]
    # Evaluate
    y_vals = nw.new_series(
        name=profile.y_name,
        values=beta + np.dot(x_controls_means, gamma),
        backend=df_prepped.implementation,
    )

    # Build output data frame - should be analog to the one build without controls
    df_plotting = (
        df_prepped.group_by(profile.bin_name)
        .agg(
            profile.x_col.mean(),
        )
        .sort(profile.bin_name)
        .with_columns(y_vals)
    )

    assert df_plotting.shape[0] == profile.num_bins
    assert df_plotting.shape[1] == 3

    return df_plotting


def make_plot_plotly(df_prepped: nw.LazyFrame, profile: Profile):
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


def _remove_bad_values(df: nw.LazyFrame) -> nw.LazyFrame:
    """Removes nulls and infinites"""
    # TODO makes this less inefficient

    bad_conditions = []

    df_num = df.select(ncs.numeric())  # HAS to be present
    cols_numeric = df_num.columns
    for c in cols_numeric:
        col = nw.col(c)
        bad_cond = col.is_null() | ~col.is_finite() | col.is_nan()
        bad_conditions.append(bad_cond)

    df_cat = df.select(ncs.categorical())  # Might be present
    if df_cat is not None and hasattr(df_cat, "columns") and df_cat.columns:
        cols_cat = df_cat.columns
        for c in cols_cat:
            bad_conditions.append(nw.col(c).is_null())

    final_bad_condition = reduce(operator.or_, bad_conditions)

    return df.filter(~final_bad_condition)


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
        qs = df.select(
            [profile.x_col.quantile(p).alias(f"q{p}") for p in probs]
        ).collect()
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

    # Sorting is not always necessary - but for safety we sort
    return (
        df.sort(profile.x_name)
        .join_asof(
            qs_long.select("quantile", profile.bin_name),
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

    def add_fallback(df: nw.DataFrame):
        return _add_fallback(df, profile, probs)

    def add_to_dask(df: nw.DataFrame) -> nw.LazyFrame:
        try:
            import dask.dataframe as dd
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

        return nw.from_native(df_native_with_bin)

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

        return nw.from_native(rel_with_bins)

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

        return nw.from_native(sdf_binned)

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
    )
