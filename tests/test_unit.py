from binscatter.core import prep
import polars as pl
import narwhals as nw

import pytest


def test_prep():
    """Tests the prep function"""

    df = pl.DataFrame({"y": [1, 2, 3, 4, 5], "x": [2, 1, 5, 3, 4]}).select("x", "y")
    num_bins = 3
    df_out, config = prep(df, "x", "y", [], num_bins)

    # Check config values
    assert config.num_bins == num_bins
    assert config.y_name == "y"
    assert config.x_name == "x"
    assert isinstance(config.x_col, nw.Expr)
    assert isinstance(config.y_col, nw.Expr)

    # Test with pandas DataFrame
    df_pd = pl.DataFrame({"y": [1, 2, 3], "x": [3, 1, 2]})
    num_bins = 2
    df_out, config = prep(df_pd, "x", "y", [], num_bins)

    # Check conversion to polars
    assert isinstance(df_out, nw.LazyFrame)
    assert config.y_name == "y"
    assert config.x_name == "x"

    # Test invalid inputs

    with pytest.raises((TypeError, ValueError)):
        prep(df, None, None, num_bins=60)  # Invalid x/y names and too many bins

    with pytest.raises((TypeError, ValueError)):
        prep(pl.DataFrame({"x": [1]}), None, None, 0)  # Invalid x/y names and num_bins
