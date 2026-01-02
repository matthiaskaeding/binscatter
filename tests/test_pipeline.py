import pytest
import pandas as pd

import narwhals as nw

from binscatter.core import _clean_controls, clean_df, add_regression_features
from tests.conftest import DF_BACKENDS, convert_to_backend

DF_TYPE_PARAMS = [
    pytest.param(df_type) for df_type in [b for b in DF_BACKENDS if b != "pyspark"]
]
if "pyspark" in DF_BACKENDS:
    DF_TYPE_PARAMS.append(pytest.param("pyspark", marks=pytest.mark.pyspark))


def test_clean_controls_returns_tuple():
    assert _clean_controls(None) == ()
    assert _clean_controls("a") == ("a",)
    assert _clean_controls(["a", "b"]) == ("a", "b")


@pytest.mark.parametrize(
    "controls,expected",
    [([], ((), ())), (["z"], (("z",), ())), (["cat"], ((), ("cat",)))],
)
@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_clean_df_splits_controls(controls, expected, df_type):
    df_pd = pd.DataFrame({"x": [1, 2], "y": [2, 3], "z": [3, 4], "cat": ["a", "b"]})
    df = convert_to_backend(df_pd, df_type)
    df_lazy, is_lazy, numeric_controls, categorical_controls = clean_df(
        df, tuple(controls), "x", "y"
    )
    assert isinstance(df_lazy, nw.LazyFrame)
    assert numeric_controls == expected[0]
    assert categorical_controls == expected[1]
    # DuckDB, Dask, and PySpark inputs are detected as lazy by narwhals
    if df_type in ("duckdb", "dask", "pyspark"):
        assert is_lazy
    else:
        assert not is_lazy


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_maybe_add_regression_features_creates_dummies(df_type):
    df_pd = pd.DataFrame({"x": [0, 1, 2], "cat": ["a", "b", "a"]})
    df_native = convert_to_backend(df_pd, df_type)
    df = nw.from_native(df_native).lazy()
    df_augmented, features = add_regression_features(
        df, numeric_controls=(), categorical_controls=("cat",)
    )
    collected = df_augmented.collect()
    dummy_cols = [c for c in collected.columns if c.startswith("__ctrl")]
    assert dummy_cols
    assert features == tuple(dummy_cols)
