from binscatter.main import prep
import polars as pl
import numpy as np
from binscatter.main import add_quantile_bins, make_b
import narwhals as nw


def test_prep():
    """Tests the prep function"""

    df = pl.DataFrame({"y": [1, 2, 3, 4, 5], "x": [2, 1, 5, 3, 4]}).select("x", "y")
    num_bins = 3
    df_out, config = prep(df, "x", "y", [], num_bins)

    # Check config values
    assert config.num_bins == num_bins
    assert config.y_name == "y"
    assert config.x_name == "x"
    assert config.N == 5
    assert isinstance(config.x_col, nw.Expr)
    assert isinstance(config.y_col, nw.Expr)

    # Check DataFrame is sorted by x
    assert df_out.get_column("x").is_sorted()

    # Test with pandas DataFrame
    df_pd = pl.DataFrame({"y": [1, 2, 3], "x": [3, 1, 2]})
    num_bins = 2
    df_out, config = prep(df_pd, "x", "y", [], num_bins)

    # Check conversion to polars
    assert isinstance(df_out, nw.DataFrame)
    assert config.N == 3
    assert config.y_name == "y"
    assert config.x_name == "x"

    # Test invalid inputs
    try:
        prep(df, None, None, num_bins=60)  # J >= N
        assert False
    except AssertionError:
        pass

    try:
        prep(pl.DataFrame({"x": [1]}), None, None, 0)  # Single column
        assert False
    except AssertionError:
        pass

    print("All tests passed!")


def test_max_x_assigned_to_max_bin():
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],
            "y": [10.5, 20.2, 30.7, 40.1, 50.9, 60.3, 70.8, 80.4, 90.6, 110.5],
        }
    )

    J = 3
    df_prepped, cfg = prep(df, "x", "y", num_bins=J)
    df_prepped = add_quantile_bins(df_prepped, cfg.x_name, cfg.bin_name, cfg.num_bins)
    bin_name = cfg.bin_name
    bins = df_prepped.get_column(bin_name).unique().sort().to_list()
    desired_bins = list(range(J))
    assert bins == desired_bins

    df_prepped = df_prepped.to_native()

    bin_of_max = (
        df_prepped.filter(pl.col("x") == pl.col("x").max()).select(bin_name).item()
    )
    assert bin_of_max == J - 1


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
