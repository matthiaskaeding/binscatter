import polars as pl
import numpy as np
from binscatter.main import binscatter
import plotly.graph_objs as go
import duckdb
import pytest


@pytest.fixture
def df_good():
    x = pl.Series("x0", range(100))
    y = pl.Series("y0", [i + np.random.normal(0, 5) for i in range(100)])
    return pl.DataFrame([x, y])


@pytest.fixture
def df_x_num():
    x = pl.Series("x0", np.random.normal(0, 100, 100))
    y = pl.Series("y0", [i + np.random.normal(0, 5) for i in range(100)])
    return pl.DataFrame([x, y])


@pytest.fixture
def df_missing_column():
    x = pl.Series("x0", range(100))
    return pl.DataFrame([x])  # missing y


@pytest.fixture
def df_nulls():
    x = pl.Series("x0", [None] * 100)
    y = pl.Series("y0", [None] * 100)
    return pl.DataFrame([x, y])


@pytest.fixture
def df_duplicates():
    x = pl.Series("x0", [1] * 100)
    y = pl.Series("y0", [2] * 100)
    return pl.DataFrame([x, y])


fixt_dat = [
    ("df_good", False),
    ("df_x_num", False),
    ("df_missing_column", True),
    ("df_nulls", True),
    ("df_duplicates", True),
]


@pytest.mark.parametrize(
    "df_fixture,expect_error",
    fixt_dat,
)
def test_binscatter(df_fixture, expect_error, request):
    df = request.getfixturevalue(df_fixture)
    if expect_error:
        with pytest.raises(Exception):
            binscatter(df, "x0", "y0")
    else:
        p = binscatter(df, "x0", "y0")
        assert isinstance(p, go.Figure)


@pytest.mark.parametrize(
    "df_fixture,expect_error",
    fixt_dat,
)
def test_binscatter_pandas(df_fixture, expect_error, request):
    df = request.getfixturevalue(df_fixture)
    df = df.to_pandas()
    if expect_error:
        with pytest.raises(Exception):
            binscatter(df, "x0", "y0")
    else:
        p = binscatter(df, "x0", "y0")
        assert isinstance(p, go.Figure)


@pytest.mark.parametrize(
    "df_fixture,expect_error",
    fixt_dat,
)
def test_binscatter_duckdb(df_fixture, expect_error, request):
    df = request.getfixturevalue(df_fixture)

    con = duckdb.connect()
    con.register("df", df.to_pandas())
    df_duckdb = con.execute("SELECT * FROM df").df()
    if expect_error:
        with pytest.raises(Exception):
            binscatter(df_duckdb, "x0", "y0")
    else:
        p = binscatter(df_duckdb, "x0", "y0")
        assert isinstance(p, go.Figure)


# def test_binscatter(df):
#     """Test that scatter() creates a binned scatter plot correctly"""

#     p = binscatter(df, "x0", "y0")
#     assert isinstance(p, go.Figure)

#     # Test with duckdb
#     con = duckdb.connect()
#     con.register("df", df.to_pandas())
#     df_duckdb = con.execute("SELECT * FROM df").df()
#     p_duckdb = binscatter(df_duckdb, "x0", "y0")
#     assert isinstance(p_duckdb, go.Figure)

#     # Test with custom bins
#     p = binscatter(df, "x0", "y0", num_bins=10)
#     assert isinstance(p, go.Figure)

#     # # Test that plot has correct labels
#     # print(p.labels)
#     # assert p.labels.x == "x0"
#     # assert p.labels.y == "y0"

#     # # Test with small dataset
#     # df_small = pl.DataFrame({"y": [1, 2, 3], "x": [4, 5, 6]})
#     # p = binscatter(df_small, "x", "y", num_bins=2)
#     # assert isinstance(p, go.Figure)

#     # # Test with negative values
#     # df_neg = pl.DataFrame({"y": [-1, -2, 0, 1], "x": [-4, -2, 0, 2]})
#     # p = binscatter(df_neg, "x", "y", num_bins=2)
#     # assert isinstance(p, go.Figure)

#     # # Test with non-monotonic relationship
#     # df_nonmono = pl.DataFrame({"y": [1, 3, 2, 4], "x": [1, 2, 3, 4]})
#     # p = binscatter(df_nonmono, "x", "y", num_bins=2)
#     # assert isinstance(p, go.Figure)

#     # # Test with floating point values
#     # df_float = pl.DataFrame({"y": [1.5, 2.5, 3.5], "x": [0.1, 0.2, 0.3]})
#     # p = binscatter(df_float, "x", "y", num_bins=2)
#     # assert isinstance(p, go.Figure)

#     # # Test with controls
#     # N = 1000
#     # x = np.random.normal(0, 1, N)
#     # z = np.random.normal(0, 1, N)
#     # # y depends on both x and z
#     # y = 2 * x + 3 * z + np.random.normal(0, 0.1, N)

#     # # Create categorical variable
#     # categories = ["A", "B", "C"]
#     # cat = np.random.choice(categories, size=N)

#     # df_controls = pl.DataFrame({"x": x, "y": y, "z": z, "category": cat})

#     # # Test binscatter with numeric and categorical controls
#     # # p = binscatter(df_controls, "x", "y", controls=["z", "category"])
#     # # assert isinstance(p, go.Figure)

#     # # Test with multiple controls including categorical
#     # w = np.random.normal(0, 1, N)
#     # df_controls = df_controls.with_columns(pl.Series("w", w))
#     # # p = binscatter(df_controls, "x", "y", controls=["z", "w", "category"])
#     # # assert isinstance(p, go.Figure)

#     # r = binscatter(df_controls, "x", "y", return_type="native")
#     # assert isinstance(r, pl.DataFrame)
#     # # r = binscatter(
#     # #     df_controls, "x", "y", controls=["z", "w", "category"], return_type="pandas"
#     # # )
#     # # assert isinstance(r, pd.DataFrame)

#     # # r = binscatter(df_controls, "x", "y", return_type="polars")
#     # # assert isinstance(r, pl.DataFrame)
#     # # r = binscatter(df_controls, "x", "y", return_type="pandas")
#     # # assert isinstance(r, pd.DataFrame)

#     # # monkeypatch.setattr(go.Figure, "show", lambda self: None)
#     # # r = binscatter(df_controls, "x", "y", return_type="none")
#     # # assert r is None


# # def test_binscatter_libs():
# #     # cuDF, Modin, PyArrow, Dask, DuckDB, Ibis, PySpark, SQLFrame
