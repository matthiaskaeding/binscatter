import polars as pl
from polars import selectors as cs
import pandas as pd
from dataclasses import dataclass
from plotnine import ggplot, aes, geom_point, xlim, ylim
from typing import Union, Iterable
import numpy as np


@dataclass
class Config:
    num_bins: int
    x_name: str
    y_name: str
    N: int
    x_col: pl.expr.Expr
    y_col: pl.expr.Expr
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bin_name: str


def prep(
    df: Union[pl.DataFrame, pd.DataFrame],
    x: str | None,
    y: str | None,
    controls: Iterable[str] = [],
    num_bins: int = 20,
):
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
    assert num_bins > 1
    assert isinstance(df, pl.DataFrame) or isinstance(df, pd.DataFrame)
    assert isinstance(x, str)
    assert isinstance(y, str)

    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    N = df.height
    assert df.width >= 2
    assert N > 0
    assert num_bins < N
    assert num_bins > 1
    assert df.shape[1] > 1
    cols = df.columns
    assert x in cols
    assert y in cols
    assert all(x in cols for x in controls), "Not all controls are in df"

    bin_name = "bins____"
    for i in range(100):
        if bin_name in cols:
            continue
        bin_name = bin_name + "_"
    if bin_name in cols:
        raise ValueError(
            f"'{bin_name}' and 99 versions with less underscores are columns in df"
        )

    df = (
        df.lazy()
        .select(x, y, *controls)
        .drop_nans()
        .drop_nulls()
        .filter(~pl.any_horizontal(cs.numeric().is_infinite()))
        .sort(x)
        .collect()
    )

    x_col = pl.col(x)
    y_col = pl.col(y)

    mins = df.select(
        x_col.min().alias("x_min"),
        x_col.max().alias("x_max"),
        y_col.min().alias("y_min"),
        y_col.max().alias("y_max"),
    )

    config = Config(
        num_bins=num_bins,
        x_name=x,
        y_name=y,
        N=df.height,
        x_col=x_col,
        y_col=y_col,
        x_min=mins.item(0, "x_min"),
        x_max=mins.item(0, "x_max"),
        y_min=mins.item(0, "y_min"),
        y_max=mins.item(0, "y_max"),
        bin_name=bin_name,
    )

    return df, config


def make_controls_mat(df: pl.DataFrame, controls: Iterable[str]) -> np.ndarray:
    """Creates a matrix of control variables by combining numeric and categorical features.

    Args:
        df (pl.DataFrame): Input dataframe containing control variables
        controls (Iterable[str]): Names of control variables to include

    Returns:
        numpy.ndarray: Matrix of control variables with numeric columns and dummy-encoded categorical columns
    """
    dfc = df.select(controls)
    df_num = dfc.select(cs.numeric())
    df_cat = dfc.select(~cs.numeric())
    if df_cat.width == 0:
        return df_num.to_numpy()
    x_cat = df_cat.to_dummies(drop_first=True).to_numpy()
    if df_num.width == 0:
        return x_cat
    x_num = df_num.to_numpy()

    return np.concat((x_num, x_cat), axis=1)


def comp_scatter_quants(df: Union[pl.DataFrame, pd.DataFrame], config: Config):
    """Makes data used for scatter plot by computing quantile bins.

    Args:
        df (Union[polars.DataFrame, pandas.DataFrame]): Input dataframe with x and y variables
        config (Config): Configuration object with metadata about the data

    Returns:
        polars.DataFrame: Input dataframe with additional 'bin' column containing quantile bin assignments
    """

    # Make expression for making the quantiles
    # Because pl.qcut is unstable we build a when - then expression
    probs = [i / config.num_bins for i in range(1, config.num_bins + 1)]
    x = df.get_column(config.x_name)
    x_quantiles = [x.quantile(quantile=p) for p in probs]
    x_col = config.x_col

    expr = pl
    for i, q in enumerate(x_quantiles):
        if i < config.num_bins - 1:
            expr = expr.when(x_col.lt(q)).then(pl.lit(i))
        else:
            expr = expr.when(x_col.le(q)).then(pl.lit(i))

    return df.with_columns(expr.alias(config.bin_name))


def make_b(df_prepped, config):
    """Makes the design matrix corresponding to the bins"""

    B = df_prepped.select(config.bin_name).to_dummies(drop_first=False)
    assert B.width == config.num_bins
    # Reorder so that column i corresponds always to bin i
    cols = [f"{config.bin_name}_{i}" for i in range(config.num_bins)]
    B = B.select(cols)

    return B.to_numpy()


def partial_out_controls(x_bins: np.array, x_controls, df_prepped, config):
    """Concatenate bin dummies and control variables horizontally."""
    x_conc = np.concatenate((x_bins, x_controls), axis=1)
    assert x_conc.shape[0] == config.N

    y = df_prepped.get_column(config.y_name).to_numpy()
    theta = np.linalg.lstsq(x_conc, y)[0]
    x_controls_means = np.mean(x_controls, axis=0)
    # Extract coefficients
    beta = theta[: config.num_bins]
    gamma = theta[config.num_bins :]
    # Evaluate
    y_vals = pl.Series(config.y_name, beta + np.dot(x_controls_means, gamma))
    # Build output data frame - should be analog to the one build without controls
    df_plotting = (
        df_prepped.group_by(config.bin_name)
        .agg(
            config.x_col.mean(),
        )
        .sort(config.bin_name)
        .with_columns(y_vals)
    )
    assert df_plotting.height == config.num_bins
    assert df_plotting.width == 3

    return df_plotting


def _print_shape(x, name):
    print(f"{name}-shape = {x.shape}")


# Main function
def binscatter(
    df: Union[pl.DataFrame, pd.DataFrame],
    x: str,
    y: str,
    controls: Iterable[str] = [],
    num_bins=20,
    return_type: str = "ggplot",
) -> Union[ggplot, pl.DataFrame, pd.DataFrame]:
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df (Union[polars.DataFrame, pandas.DataFrame]): Input dataframe
        x (str): Name of x column
        y (str): Name y column
        covariates (Iterable[str]): names of control variables
        num_bins (int, optional): Number of bins to use. Defaults to 20
        return_type (str): Return type. Default a ggplot, otherwise "polars" for polars dataframe or "pandas" for pandas dataframe

    Returns:
        plotnine.ggplot: A ggplot object containing the binned scatter plot with x and y axis labels
    """
    assert return_type in ("ggplot", "polars", "pandas"), ""
    df, config = prep(df, x, y, controls, num_bins)
    df_prepped = comp_scatter_quants(df, config)
    # Currently there are 2 cases:
    # (1) no controls: the easy one, just compute the means by bin
    # (2) controls: here we need to compute regression coefficients
    # and partial out the effect of the controls
    # (see section 2.2 in Cattaneo, Crump, Farrell and Feng (2024))
    if not controls:
        df_plotting = df_prepped.group_by(config.bin_name).agg(
            config.x_col.mean(), config.y_col.mean()
        )
    else:
        x_controls = make_controls_mat(df_prepped, controls)
        x_bins = make_b(df_prepped, config)
        df_plotting = partial_out_controls(x_bins, x_controls, df_prepped, config)

    p = (
        ggplot(df_plotting)
        + aes(config.x_name, config.y_name)
        + geom_point()
        + xlim(config.x_min, config.x_max)
        + ylim(config.y_min, config.y_max)
    )

    match return_type:
        case "ggplot":
            return p
        case "polars":
            return df_plotting
        case "pandas":
            return df_plotting.to_pandas()
