import polars as pl
import pytest

import narwhals as nw

from binscatter.core import _clean_controls, clean_df, maybe_add_regression_features


def test_clean_controls_returns_tuple():
    assert _clean_controls(None) == ()
    assert _clean_controls("a") == ("a",)
    assert _clean_controls(["a", "b"]) == ("a", "b")


@pytest.mark.parametrize(
    "controls,expected",
    [([], ((), ())), (["z"], (("z",), ())), (["cat"], ((), ("cat",)))],
)
def test_clean_df_splits_controls(controls, expected):
    df = pl.DataFrame({"x": [1, 2], "y": [2, 3], "z": [3, 4], "cat": ["a", "b"]})
    df_lazy, is_lazy, numeric_controls, categorical_controls = clean_df(
        df, tuple(controls), "x", "y"
    )
    assert isinstance(df_lazy, nw.LazyFrame)
    assert numeric_controls == expected[0]
    assert categorical_controls == expected[1]
    assert not is_lazy


def test_maybe_add_regression_features_creates_dummies():
    df = nw.from_native(pl.DataFrame({"x": [0, 1, 2], "cat": ["a", "b", "a"]})).lazy()
    df_augmented, features = maybe_add_regression_features(
        df, numeric_controls=(), categorical_controls=("cat",)
    )
    collected = df_augmented.collect()
    dummy_cols = [c for c in collected.columns if c.startswith("__ctrl")]
    assert dummy_cols
    assert features == tuple(dummy_cols)
