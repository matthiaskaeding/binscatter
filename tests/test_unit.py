from binscatter.main import prep
import polars as pl
import numpy as np
from binscatter.main import add_quantile_bins, make_b
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


def test_make_b():
    for N, num_bins in [(100, 5), (201, 10), (323, 15), (400, 20)]:
        # Create df_prepped with bins from 0 to num_bins-1
        bin_size = N / num_bins
        bins = [int(i // bin_size) for i in range(N)]
        df_prepped = pl.DataFrame({"bins": bins}).sort("bins")

        # Create config object
        class Config:
            def __init__(self):
                self.bin_name = "bins"
                self.N = N
                self.num_bins = num_bins

        config = Config()

        B = make_b(df_prepped, config)
        assert B.shape == (N, num_bins)

        # Test that when B[:,j] = 1, df_prepped has bin = j
        for j in range(num_bins):
            rows_with_1 = np.where(B[:, j] == 1)[0]
            bin_vals = df_prepped.slice(rows_with_1[0], len(rows_with_1))["bins"]
            assert (bin_vals == j).all()

        # Test row sums = 1
        assert all(np.sum(B, axis=1) == 1), f"Not all sums are 1 {N=} {num_bins=}"

        # Test column sums match bin counts
        bin_counts = (
            df_prepped.group_by("bins").len().sort("bins").get_column("len").to_numpy()
        )
        col_counts = np.sum(B, axis=0)

        assert np.array_equal(bin_counts, col_counts)
