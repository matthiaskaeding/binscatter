from typing import Iterable, Literal, Tuple
import numpy as np
import narwhals as nw
from narwhals.typing import IntoDataFrame
from narwhals import Implementation
import plotly.express as px
import narwhals.selectors as ncs
from typing import List, NamedTuple
import uuid
import math
from functools import reduce
import operator


def binscatter(
    df: IntoDataFrame,
    x: str,
    y: str,
    controls: Iterable[str] = [],
    num_bins=20,
    return_type: Literal["plotly", "native"] = "plotly",
):
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df (Union[polars.DataFrame, pandas.DataFrame]): Input dataframe
        x (str): Name of x column
        y (str): Name y column
        controls (Iterable[str]): names of control variables (not used yet)
        num_bins (int, optional): Number of bins to use. Defaults to 20
        return_type (str): Return type. Default a ggplot, otherwise "polars" for polars dataframe or "pandas" for pandas dataframe

    Returns:
        plotnine.ggplot: A ggplot object containing the binned scatter plot with x and y axis labels
    """
    if return_type not in ("plotly", "native"):
        msg = f"Invalid return_type: {return_type}"
        raise ValueError(msg)

    # Prepare dataframe: sort, remove non numerics and add bins
    df_prepped, config = prep(df, x, y, controls, num_bins)

    # Currently there are 2 cases:
    # (1) no controls: the easy one, just compute the means by bin
    # (2) controls: here we need to compute regression coefficients
    # and partial out the effect of the controls
    # (see section 2.2 in Cattaneo, Crump, Farrell and Feng (2024))
    if not controls:
        df_plotting: nw.DataFrame = (
            df_prepped.group_by(config.bin_name)
            .agg(config.x_col.mean(), config.y_col.mean())
            .collect()
        )
    else:
        x_controls = make_controls_mat(df_prepped, controls)
        x_bins = make_b(df_prepped, config)
        df_plotting = partial_out_controls(x_bins, x_controls, df_prepped, config)

    if df_plotting.shape[0] < config.num_bins:
        raise ValueError("Quantiles are not unique. Decrease number of bins.")

    match return_type:
        case "plotly":
            return make_plot_plotly(df_plotting, config)
        case "native":
            return df_plotting.to_native().collect()


class Config(NamedTuple):
    """Main config which holds bunch of data derived from dataframe."""

    x_name: str
    y_name: str
    controls: Tuple[str]
    num_bins: int
    bin_name: str
    x_bounds: Tuple[float, float]
    distinct_suffix: str

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
) -> Tuple[nw.DataFrame, Config]:
    """Prepares the input data and builds configuration.

    Args:
        df: Input dataframe, either polars or pandas. Must have at least 2 columns.
            First column is treated as y variable, second as x variable.
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

    dfl = nw.from_native(df_in).lazy()

    if isinstance(controls, str):
        controls = (controls,)
    else:
        controls = tuple(controls)

    try:
        df = dfl.select(x_name, y_name, *controls)
    except Exception as e:
        cols = dfl.columns
        for c in [x_name, y_name, *controls]:
            if c not in cols:
                msg = f"{x_name} not in input dataframe"
                raise ValueError(msg)
        raise e

    assert num_bins > 1

    # Find name for bins
    distinct_suffix = str(uuid.uuid4())
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

    config = Config(
        num_bins=num_bins,
        x_name=x_name,
        y_name=y_name,
        bin_name=bin_name,
        controls=controls,
        x_bounds=x_bounds,
        distinct_suffix=distinct_suffix,
    )
    df_prepped = add_quantile_bins(df_filtered, config)

    return df_prepped, config


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


def make_b(df_prepped: nw.DataFrame, config: Config):
    """Makes the design matrix corresponding to the bins"""

    B = df_prepped.select(config.bin_name).to_dummies(drop_first=False, separator="_")
    # assert B.shape[1] == config.num_bins, (
    #     f"B must have {config.num_bins} columns but has {B.width}, this indicates too many bins"
    # )
    # Reorder so that column i corresponds always to bin i
    cols = [f"{config.bin_name}_{i}" for i in range(config.num_bins)]

    return B.select(cols).to_numpy()


def partial_out_controls(
    x_bins: np.array, x_controls: np.ndarray, df_prepped: nw.DataFrame, config: Config
) -> nw.DataFrame:
    """Concatenate bin dummies and control variables horizontally."""
    x_conc = np.concatenate((x_bins, x_controls), axis=1)

    y = df_prepped.get_column(config.y_name).to_numpy()
    theta = np.linalg.lstsq(x_conc, y)[0]
    x_controls_means = np.mean(x_controls, axis=0)
    # Extract coefficients
    beta = theta[: config.num_bins]
    gamma = theta[config.num_bins :]
    # Evaluate
    y_vals = nw.new_series(
        name=config.y_name,
        values=beta + np.dot(x_controls_means, gamma),
        backend=df_prepped.implementation,
    )

    # Build output data frame - should be analog to the one build without controls
    df_plotting = (
        df_prepped.group_by(config.bin_name)
        .agg(
            config.x_col.mean(),
        )
        .sort(config.bin_name)
        .with_columns(y_vals)
    )

    assert df_plotting.shape[0] == config.num_bins
    assert df_plotting.shape[1] == 3

    return df_plotting


def make_plot_plotly(df_prepped: nw.DataFrame, config: Config):
    """Make plot from prepared dataframe.

    Args:
      df_prepped (nw.LazyFrame): Prepared dataframe. Has three columns: bin, x, y with names in config"""
    data = df_prepped.select(config.x_name, config.y_name)
    x = data.get_column(config.x_name).to_list()
    y = data.get_column(config.y_name).to_list()

    fig = px.scatter(x=x, y=y, range_x=config.x_bounds).update_layout(
        title="Binscatter",
        xaxis_title=config.x_name,
        yaxis_title=config.y_name,
    )

    return fig


def _remove_bad_values(df: nw.LazyFrame) -> nw.LazyFrame:
    """Removes nulls and infinites"""

    bad_conditions = []

    cols_numeric = df.select(ncs.numeric()).columns
    for c in cols_numeric:
        col = nw.col(c)
        bad_cond = col.is_null() | ~col.is_finite() | col.is_nan()
        bad_conditions.append(bad_cond)

    cols_cat = df.select(ncs.categorical()).columns
    for c in cols_cat:
        bad_conditions.append(nw.col(c).is_null())

    final_bad_condition = reduce(operator.or_, bad_conditions)

    return df.filter(~final_bad_condition)


# Quantiles


# Specific handlers
def add_to_pandas(
    df: nw.DataFrame, x_name: str, bin_name: str, probs: List[float]
) -> nw.LazyFrame:
    try:
        from pandas import qcut
    except ImportError:
        raise ImportError("Pandas support requires pandas to be installed.")
    df_native = df.to_native()
    buckets = qcut(df_native[x_name], labels=False, q=probs)
    df_native[bin_name] = buckets

    return nw.from_native(df_native).lazy()


def add_to_polars(
    df: nw.DataFrame, x_name: str, bin_name: str, probs: List[float]
) -> nw.LazyFrame:
    try:
        import polars as pl
    except ImportError:
        raise ImportError("Polars support requires Polars to be installed.")
    # Because cut and qcut are not stable we use when-then
    df_native = df.to_native()
    x_col = pl.col(x_name)

    qs = df_native.select([x_col.quantile(p).alias(f"q{p}") for p in probs]).collect()
    expr = pl
    n = qs.width
    for i in range(n):
        thr = qs.item(0, i)
        cond = x_col.le(thr) if i == n - 1 else x_col.lt(thr)
        expr = expr.when(cond).then(pl.lit(i))
    expr = expr.alias(bin_name)
    df_native_with_bin = df_native.with_columns(expr)

    return nw.from_native(df_native_with_bin)


def add_to_duckdb(
    df: nw.DataFrame, x_name: str, bin_name: str, probs: List[float]
) -> nw.LazyFrame:
    try:
        rel = df.to_native()
    except Exception as e:
        raise RuntimeError(
            "Failed to use df.to_native(); DuckDB may not be installed."
        ) from e

    order_expr = f"{x_name} ASC"
    rel_with_bins = rel.project(
        f"*, ntile({len(probs)}) OVER (ORDER BY {order_expr}) - 1 AS {bin_name}"
    )

    return nw.from_native(rel_with_bins)


def add_to_pyspark(
    df: nw.DataFrame, x_name: str, bin_name: str, probs: List[float], x_max: float
) -> nw.LazyFrame:
    try:
        from pyspark.ml.feature import Bucketizer
    except ImportError:
        raise ImportError("PySpark support requires pyspark to be installed.")

    sdf = df.to_native()
    qs = sdf.approxQuantile(x_name, probs[:-1], relativeError=1e-3)
    qs.append(x_max)

    if len(set(qs)) < len(qs):
        raise ValueError("Quantiles not unique. Decrease number of bins.")
    # Build strictly increasing splits for Bucketizer: (a, b] bins
    splits = [-float("inf"), *qs, float("inf")]

    bucketizer = Bucketizer(
        splits=splits,
        inputCol=x_name,
        outputCol=bin_name,
        handleInvalid="keep",
    )

    sdf_binned = bucketizer.transform(sdf)

    return nw.from_native(sdf_binned)


def add_quantile_bins(df: nw.DataFrame, config: Config):
    # Common arguments
    kwargs = {
        "df": df,
        "x_name": config.x_name,
        "bin_name": config.bin_name,
        "probs": [i / config.num_bins for i in range(1, config.num_bins + 1)],
    }

    if df.implementation == Implementation.PANDAS:
        quantile_handler = add_to_pandas
    elif df.implementation == Implementation.POLARS:
        quantile_handler = add_to_polars
    elif df.implementation == Implementation.PYSPARK:
        quantile_handler = add_to_pyspark
        kwargs["x_max"] = config.x_bounds[1]
    elif df.implementation == Implementation.DUCKDB:
        quantile_handler = add_to_duckdb
    else:
        raise NotImplementedError(
            f"No implementation available for {df.implementation}"
        )

    return quantile_handler(**kwargs)


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
