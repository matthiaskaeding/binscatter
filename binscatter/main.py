import polars as pl
import pandas as pd
from dataclasses import dataclass
from plotnine import ggplot, aes, geom_point, xlim, ylim, labs


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


def prep(df, J):
    """Prepares the input data and builds configuration"""
    assert isinstance(df, pl.DataFrame) or isinstance(df, pd.DataFrame)
    N = df.shape[0]
    assert J < N
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    assert df.width >= 2
    assert df.height > 0
    cols = df.columns
    assert "__x_col_max__" not in cols, (
        "Variable name __x_col_max__ not allowed in input dataframe"
    )

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


def comp_scatter_quants(df: pl.DataFrame, config):
    """Makes data used for scatter plot"""

    # Make expression for making the quantiles
    # Because pl.qcut is unstable we build a when - then expression
    probs = [i / config.J for i in range(1, config.J + 1)]
    x = df.get_column(config.x_name)
    x_quantiles = [x.quantile(quantile=p) for p in probs]

    x_col = config.x_col
    expr = pl
    for i, q in enumerate(x_quantiles):
        expr = expr.when(x_col.le(q)).then(pl.lit(i))
    expr = expr.alias("bin")
    df = df.with_columns(
        expr,
    )

    return df


def scatter(df, J=20):
    df, config = prep(df, J)
    df = comp_scatter_quants(df, config)

    df_plotting = (
        df.group_by("bin")
        .agg(config.y_col.mean(), config.x_col.max().alias("__x_col_max__"))
        .sort("bin")
    )

    p = (
        ggplot(df_plotting)
        + aes("__x_col_max__", config.y_name)
        + geom_point()
        + xlim(config.x_min, config.x_max)
        + ylim(config.y_min, config.y_max)
        + labs(x=config.x_name, y=config.y_name)
    )

    return p
