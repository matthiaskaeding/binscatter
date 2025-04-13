import polars as pl
import pandas as pd
from dataclasses import dataclass
from plotnine import ggplot, aes, geom_point, xlim, ylim
from typing import Union


@dataclass
class Config:
    J: int
    x_name: str
    y_name: str
    N: int
    x_col: pl.expr.Expr
    y_col: pl.expr.Expr
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def prep(df: Union[pl.DataFrame, pd.DataFrame], J: int):
    """Prepares the input data and builds configuration.

    Args:
        df: Input dataframe, either polars or pandas. Must have at least 2 columns.
            First column is treated as y variable, second as x variable.
        J: Number of bins to use for binscatter. Must be less than number of rows.

    Returns:
        tuple: (polars.DataFrame, Config)
            - Sorted input dataframe converted to polars
            - Config object with metadata about the data

    Raises:
        AssertionError: If input validation fails
    """
    assert J > 1
    assert isinstance(df, pl.DataFrame) or isinstance(df, pd.DataFrame)
    N = df.shape[0]
    assert J < N
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    assert df.width >= 2
    N = df.height
    assert N > 0
    assert J < N
    cols = df.columns
    x_col = pl.col(cols[1])
    y_col = pl.col(cols[0])
    mins = df.select(
        x_col.min().alias("x_min"),
        x_col.max().alias("x_max"),
        y_col.min().alias("y_min"),
        y_col.max().alias("y_max"),
    )

    config = Config(
        J=J,
        y_name=cols[0],
        x_name=cols[1],
        N=df.height,
        x_col=x_col,
        y_col=y_col,
        x_min=mins.item(0, "x_min"),
        x_max=mins.item(0, "x_max"),
        y_min=mins.item(0, "y_min"),
        y_max=mins.item(0, "y_max"),
    )
    df = df.sort(config.x_name)  # Sort for faster quantiles

    return df, config


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
    probs = [i / config.J for i in range(1, config.J + 1)]
    x = df.get_column(config.x_name)
    x_quantiles = [x.quantile(quantile=p) for p in probs]
    x_col = config.x_col

    expr = pl
    for i, q in enumerate(x_quantiles):
        if i < config.J - 1:
            expr = expr.when(x_col.lt(q)).then(pl.lit(i))
        else:
            expr = expr.when(x_col.le(q)).then(pl.lit(i))

    return df.with_columns(expr.alias("bin"))


def binscatter(df: Union[pl.DataFrame, pd.DataFrame], J=20):
    """Creates a binned scatter plot by grouping x values into quantile bins and plotting mean y values.

    Args:
        df (Union[polars.DataFrame, pandas.DataFrame]): Input dataframe with exactly two columns - y variable and x variable
        J (int, optional): Number of bins to use. Defaults to 20.

    Returns:
        plotnine.ggplot: A ggplot object containing the binned scatter plot with x and y axis labels
    """

    df, config = prep(df, J)
    df_prepped = comp_scatter_quants(df, config)
    df_plotting = (
        df_prepped.group_by("bin")
        .agg(config.x_col.mean(), config.y_col.mean())
        .sort("bin")
    )

    p = (
        ggplot(df_plotting)
        + aes(config.x_name, config.y_name)
        + geom_point()
        + xlim(config.x_min, config.x_max)
        + ylim(config.y_min, config.y_max)
    )

    return p
