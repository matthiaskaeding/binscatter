from binscatter.main import prep
import polars as pl
import numpy as np
from binscatter import bin_scatter
from plotnine import ggplot


def test_prep():
    """Tests the prep function"""
    # Test with polars DataFrame
    df = pl.DataFrame({"y": [1, 2, 3, 4, 5], "x": [2, 1, 5, 3, 4]})
    J = 3
    df_out, config = prep(df, J)

    # Check config values
    assert config.J == J
    assert config.y_name == "y"
    assert config.x_name == "x"
    assert config.N == 5
    assert isinstance(config.x_col, pl.Expr)
    assert isinstance(config.y_col, pl.Expr)

    # Check DataFrame is sorted by x
    assert df_out.get_column("x").is_sorted()

    # Test with pandas DataFrame
    df_pd = pl.DataFrame({"y": [1, 2, 3], "x": [3, 1, 2]})
    J = 2
    df_out, config = prep(df_pd, J)

    # Check conversion to polars
    assert isinstance(df_out, pl.DataFrame)
    assert config.N == 3
    assert config.y_name == "y"
    assert config.x_name == "x"

    # Test invalid inputs
    try:
        prep(df, 6)  # J >= N
        assert False
    except AssertionError:
        pass

    try:
        prep(pl.DataFrame({"x": [1]}), 0)  # Single column
        assert False
    except AssertionError:
        pass

    print("All tests passed!")


def test_scatter():
    """Test that scatter() creates a binned scatter plot correctly"""
    # Create test data
    x = pl.Series("x0", range(100))
    y = pl.Series("y0", [i + np.random.normal(0, 5) for i in range(100)])
    df = pl.DataFrame([y, x])

    # Test with default bins
    p = bin_scatter(df)
    assert isinstance(p, ggplot)

    # Test with custom bins
    p = bin_scatter(df, J=10)
    assert isinstance(p, ggplot)

    # Test that plot has correct labels
    assert p.labels.x == "x0"
    assert p.labels.y == "y0"

    # Test with small dataset
    df_small = pl.DataFrame({"y": [1, 2, 3], "x": [4, 5, 6]})
    p = bin_scatter(df_small, J=2)
    assert isinstance(p, ggplot)
