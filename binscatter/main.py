import polars as pl
import pandas as pd
from dataclasses import dataclass


@dataclass
class Config:
    J: int
    name_x: str
    name_y: str
    N: int
    x_col: pl.expr.Expr
    y_col: pl.expr.Expr


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

    config = Config(
        J=J,
        name_y=cols[0],
        name_x=cols[1],
        N=df.height,
        x_col=pl.col(cols[1]),
        y_col=pl.col(cols[0]),
    )
    df = df.sort(config.name_x)  # Sort for faster quantiles

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
    df = df.with_columns(expr)

    return df


def simple_scatter(df, J):
    df, config = prep(df, J)
    df = comp_scatter_quants(df, config)

    df.group_by("bin").agg(config.y_col.mean()).sort("bin")
