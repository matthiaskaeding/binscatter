import polars as pl
import numpy as np
from binscatter.main import binscatter
from plotnine import ggplot
import pandas as pd


def test_binscatter(monkeypatch):
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

    # Test with controls
    N = 1000
    x = np.random.normal(0, 1, N)
    z = np.random.normal(0, 1, N)
    # y depends on both x and z
    y = 2 * x + 3 * z + np.random.normal(0, 0.1, N)

    # Create categorical variable
    categories = ["A", "B", "C"]
    cat = np.random.choice(categories, size=N)

    df_controls = pl.DataFrame({"x": x, "y": y, "z": z, "category": cat})

    # Test binscatter with numeric and categorical controls
    p = binscatter(df_controls, "x", "y", controls=["z", "category"])
    assert isinstance(p, ggplot)

    # Test with multiple controls including categorical
    w = np.random.normal(0, 1, N)
    df_controls = df_controls.with_columns(pl.Series("w", w))
    p = binscatter(df_controls, "x", "y", controls=["z", "w", "category"])
    assert isinstance(p, ggplot)

    r = binscatter(
        df_controls, "x", "y", controls=["z", "w", "category"], return_type="polars"
    )
    assert isinstance(r, pl.DataFrame)
    r = binscatter(
        df_controls, "x", "y", controls=["z", "w", "category"], return_type="pandas"
    )
    assert isinstance(r, pd.DataFrame)

    r = binscatter(df_controls, "x", "y", return_type="polars")
    assert isinstance(r, pl.DataFrame)
    r = binscatter(df_controls, "x", "y", return_type="pandas")
    assert isinstance(r, pd.DataFrame)

    monkeypatch.setattr(ggplot, "show", lambda self: None)
    r = binscatter(df_controls, "x", "y", return_type="none")
    assert r is None
