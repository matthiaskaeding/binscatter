import narwhals as nw
from typing import List, Sequence, Iterable
from math import ceil, floor
import narwhals.selectors as ncs


def _remove_bad_values(df: nw.DataFrame) -> nw.DataFrame:
    """Removes nulls and infinites"""
    cols_numeric = df.select(ncs.numeric()).columns
    mask = None

    for c in cols_numeric:
        col = df.get_column(c)
        cond = col.is_null() | ~col.is_finite()
        mask = cond if not mask else mask | cond

    cols_cat = df.select(ncs.categorical()).columns
    for c in cols_cat:
        col = df.get_column(c)
        cond = col.is_null()
        mask = cond if not mask else mask | cond

    return df.filter(~mask)


def _compute_quantiles(
    df: nw.DataFrame, colname: str, probs: Iterable[float], bin_name: str
) -> nw.LazyFrame:
    """Get multiple quantiles in one operation"""
    col = nw.col(colname)
    qs = df.select(
        [col.quantile(q, interpolation="linear").alias(f"q{q}") for q in probs]
    )

    return (
        qs.unpivot(variable_name="prob", value_name="quantile")
        .sort("quantile")
        .with_row_index(bin_name, order_by="quantile")
    )


def _add_quantile_bins(
    df: nw.LazyFrame, colname: str, df_q: nw.LazyFrame
) -> nw.LazyFrame:
    """
    Adds bin var based on quantiles
    Args:
      df (nw.DataFrame)
    """
    try:
        joined = df.join_asof(
            df_q, left_on=colname, right_on="quantile", strategy="forward"
        )
    except nw.exceptions.NarwhalsError:
        # Sometimes making the quantiles changes the datatypes and then we need to cast
        s = df_q.collect_schema()
        des_type = s["quantile"]
        joined = df.with_colummns(nw.col(colname).cast(des_type)).join_asof(
            df_q, left_on=colname, right_on="quantile", strategy="forward"
        )

    return joined


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
