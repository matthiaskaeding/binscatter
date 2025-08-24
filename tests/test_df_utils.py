import narwhals as nw
import numpy as np
import pandas as pd
from binscatter.df_utils import (
    _get_quantile_bins,
    _remove_bad_values,
    _compute_quantiles,
    _quantiles_from_sorted,
)
import polars as pl
# Python


def make_nw_df(data):
    # Helper to create narwhals DataFrame from pandas
    return nw.from_native(data)


def test_filter_all_numeric_basic():
    data = {"a": [1.0, 2.3, np.nan, 4.0, np.inf, None], "b": [5, 6, 7, None, 9, 10]}
    df_pandas = pd.DataFrame(data)
    df = nw.from_native(df_pandas)

    filtered = _remove_bad_values(df)
    assert filtered.shape[0] == 2
    assert filtered["a"].to_numpy().tolist() == [1.0, 2.3]
    assert filtered["b"].to_numpy().tolist() == [5, 6]

    df_polars = pl.DataFrame(data)
    df_from_polars = nw.from_native(df_polars)

    filtered_polars = _remove_bad_values(df_from_polars)
    assert filtered_polars.shape[0] == 2
    assert filtered_polars["a"].to_numpy().tolist() == [1.0, 2.3]
    assert filtered_polars["b"].to_numpy().tolist() == [5, 6]


def test_multiple_quantiles_basic():
    data = {"x": [1, 2, 3, 4, 5]}
    dfp = pd.DataFrame(data)
    df = nw.from_native(dfp)
    quantiles = [0.0, 0.5, 1.0]
    result = _compute_quantiles(df, "x", quantiles)

    assert result.to_list() == [1, 3, 5]


def test_quantiles_from_sorted_basic():
    data = {"x": [1, 2, 3, 4, 5]}
    dfp = pd.DataFrame(data)
    df = nw.from_native(dfp).sort("x")
    s = df["x"]
    probs = [0.0, 0.5, 1.0]
    qs = _quantiles_from_sorted(s, probs)

    assert qs == [1, 3, 5]


def test_add_quantile_bins():
    data = {"x": [1, 2, 3, 4, 5]}
    dfp = pd.DataFrame(data)
    df = nw.from_native(dfp).sort("x")

    thresholds = [2, 4, 5]
    bins = _get_quantile_bins(df, "x", thresholds)
    assert list(bins) == [0, 0, 1, 1, 2]
