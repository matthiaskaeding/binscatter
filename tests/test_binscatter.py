from binscatter.main import prep
import polars as pl
import numpy as np
from binscatter.main import comp_scatter_quants, binscatter
from plotnine import ggplot


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
    assert isinstance(config.x_col, pl.Expr)
    assert isinstance(config.y_col, pl.Expr)

    # Check DataFrame is sorted by x
    assert df_out.get_column("x").is_sorted()

    # Test with pandas DataFrame
    df_pd = pl.DataFrame({"y": [1, 2, 3], "x": [3, 1, 2]})
    num_bins = 2
    df_out, config = prep(df_pd, "x", "y", [], num_bins)

    # Check conversion to polars
    assert isinstance(df_out, pl.DataFrame)
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
    df_prepped = comp_scatter_quants(df_prepped, cfg)
    bin_name = cfg.bin_name
    bins = df_prepped.get_column(bin_name).unique().sort().to_list()
    desired_bins = list(range(J))
    assert bins == desired_bins

    bin_of_max = (
        df_prepped.filter(pl.col("x") == pl.col("x").max()).select(bin_name).item()
    )
    assert bin_of_max == J - 1


def test_binscatter():
    """Test that scatter() creates a binned scatter plot correctly"""
    # Create test data
    x = pl.Series("x0", range(100))
    y = pl.Series("y0", [i + np.random.normal(0, 5) for i in range(100)])
    df = pl.DataFrame([x, y])

    # Test with default bins
    p = binscatter(df, "x0", "y0")
    assert isinstance(p, ggplot)

    # Test with custom bins
    p = binscatter(df, "x0", "y0", num_bins=10)
    assert isinstance(p, ggplot)

    # Test that plot has correct labels
    print(p.labels)
    assert p.labels.x == "x0"
    assert p.labels.y == "y0"

    # Test with small dataset
    df_small = pl.DataFrame({"y": [1, 2, 3], "x": [4, 5, 6]})
    p = binscatter(df_small, "x", "y", num_bins=2)
    assert isinstance(p, ggplot)

    # Test with negative values
    df_neg = pl.DataFrame({"y": [-1, -2, 0, 1], "x": [-4, -2, 0, 2]})
    p = binscatter(df_neg, "x", "y", num_bins=2)
    assert isinstance(p, ggplot)

    # Test with non-monotonic relationship
    df_nonmono = pl.DataFrame({"y": [1, 3, 2, 4], "x": [1, 2, 3, 4]})
    p = binscatter(df_nonmono, "x", "y", num_bins=2)
    assert isinstance(p, ggplot)

    # Test with floating point values
    df_float = pl.DataFrame({"y": [1.5, 2.5, 3.5], "x": [0.1, 0.2, 0.3]})
    p = binscatter(df_float, "x", "y", num_bins=2)
    assert isinstance(p, ggplot)
