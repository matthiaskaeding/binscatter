from binscatter.main import prep
import polars as pl


def test_prep():
    """Tests the prep function"""
    # Test with polars DataFrame
    df = pl.DataFrame({"y": [1, 2, 3, 4, 5], "x": [2, 1, 5, 3, 4]})
    J = 3
    df_out, config = prep(df, J)

    # Check config values
    assert config.J == J
    assert config.name_y == "y"
    assert config.name_x == "x"
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
    assert config.name_y == "y"
    assert config.name_x == "x"

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
