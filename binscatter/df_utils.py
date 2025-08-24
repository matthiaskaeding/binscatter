import narwhals as nw
from typing import List, Sequence
from math import ceil, floor
import narwhals.selectors as ncs


def _remove_bad_values(df: nw.DataFrame) -> nw.DataFrame:
    """Removes nulls and infinites"""
    cols_numeric = df.select(ncs.numeric()).columns
    mask = None

    for c in cols_numeric:
        col = df[c]
        cond = col.is_null() | ~col.is_finite()
        mask = cond if not mask else mask | cond

    cols_cat = df.select(ncs.categorical()).columns
    for c in cols_cat:
        col = df[c]
        cond = col.is_null()
        mask = cond if not mask else mask | cond

    return df.filter(~mask)


def _compute_quantiles(df: nw.DataFrame, colname: str, quantiles_list) -> nw.Series:
    """Get multiple quantiles in one operation"""
    col = nw.col(colname)
    qs = df.select(
        [col.quantile(q, interpolation="linear").alias(f"q{q}") for q in quantiles_list]
    )

    return qs.unpivot(variable_name="q", value_name="val").get_column("val").sort()


def _get_quantile_bins(
    df: nw.DataFrame, colname: str, quantiles_sorted: Sequence[float]
) -> nw.Series:
    """
    Adds bin var based on quantiles
    Args:
      df (nw.DataFrame)
    """
    assert df[colname].is_sorted()

    df_q = nw.from_dict({"threshold": quantiles_sorted}, backend=df.implementation)
    df_q = df_q.with_row_index("bin")
    try:
        joined = df.select(colname).join_asof(
            df_q, left_on=colname, right_on="threshold", strategy="forward"
        )
    except nw.exceptions.NarwhalsError:
        # Sometimes making the quantiles changes the datatypes and then we need to cast
        joined = df.select(
            nw.col(colname).cast(df_q.get_column("threshold").dtype)
        ).join_asof(df_q, left_on=colname, right_on="threshold", strategy="forward")

    return joined.get_column("bin").sort()


def _quantiles_from_sorted(s: nw.Series, probs: Sequence[float]) -> List[int]:
    """Compute quantiles and positions from a sorted narwhals series.
    Quantiles are compute using the equivalent of the "lower" method"""
    assert s.is_sorted()

    n = s.shape[0]
    qs = [None] * len(probs)
    for i, p in enumerate(probs):
        pos = (n - 1) * p
        lo = floor(pos)
        hi = ceil(pos)
        w = pos - lo
        q = (1 - w) * s.item(lo) + w * s.item(hi)
        qs[i] = q

    return qs


def _print_shape(x, name):
    print(f"{name}-shape = {x.shape}")
