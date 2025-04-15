import polars as pl
import numpy as np
from binscatter.main import binscatter
from plotnine import ggplot


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
