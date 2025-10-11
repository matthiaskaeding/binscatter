import narwhals as nw
import numpy as np
import pandas as pd
from binscatter.core import (
    _remove_bad_values,
)
import polars as pl


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
