import polars as pl
import numpy as np
from binscatter.core import binscatter
import plotly.graph_objs as go
import duckdb
import pytest
import pandas as pd
from pyspark.sql import SparkSession
import pyspark
import dask.dataframe as dd

RNG = np.random.default_rng(42)


def quantile_bins(x, n_bins=10):
    """
    Returns:
      idx   : int array of bin indices in [0, n_bins-1]
      edges : float array of bin edges of length n_bins+1
              Intervals are [edges[i], edges[i+1]) for i<n_bins-1
              and [edges[-2], edges[-1]] for the last bin.
    """
    x = np.asarray(x)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    # NumPy changed 'interpolation' -> 'method' in newer versions; support both:
    try:
        edges = np.quantile(x, qs, method="linear")
    except TypeError:
        edges = np.quantile(x, qs, interpolation="linear")

    # Left-closed bins via searchsorted(..., side='right'), then clip to keep the last bin closed.
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)  # ensures max(x) falls in the last bin

    return idx, edges


@pytest.fixture
def df_good(N=10000):
    x = np.arange(N)
    y = x + RNG.normal(0, 5, size=N)
    return pd.DataFrame({"x0": x, "y0": y})


@pytest.fixture
def df_x_num(N=1000_000):
    x = pd.Series(RNG.normal(0, 100, N), name="x0")
    y = pd.Series(np.arange(N) + RNG.normal(0, 5, N), name="y0")
    return pd.concat([x, y], axis=1)


@pytest.fixture
def df_missing_column():
    # Only x0 present; y0 is missing
    x = pd.Series(np.arange(100), name="x0")
    return pd.DataFrame({"x0": x})


@pytest.fixture
def df_nulls():
    # Both columns are entirely null
    x = pd.Series([np.nan] * 100, name="x0")
    y = pd.Series([np.nan] * 100, name="y0")
    return pd.concat([x, y], axis=1)


@pytest.fixture
def df_duplicates():
    # Duplicate rows: constant values in both columns
    x = pd.Series([1] * 100, name="x0")
    y = pd.Series([2] * 100, name="y0")
    return pd.concat([x, y], axis=1)


fixt_dat = [
    ("df_good", False),
    ("df_x_num", False),
    ("df_missing_column", True),
    ("df_nulls", True),
    ("df_duplicates", True),
]
fix_data_types = []
DF_TYPES = [
    "pandas",
    "polars",
    "duckdb",
    "dask",
    "pyspark",
]
for df_type in DF_TYPES:
    for pair in fixt_dat:
        triple = (*pair, df_type)
        fix_data_types.append(triple)


def conv(df: pd.DataFrame, df_type):
    match df_type:
        case "pandas":
            return df
        case "polars":
            return pl.from_pandas(df)
        case "duckdb":
            return duckdb.from_df(df)
        case "pyspark":
            spark = SparkSession.builder.getOrCreate()
            return spark.createDataFrame(df)
        case "dask":
            return dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize(
    "df_fixture,expect_error,df_type",
    fix_data_types,
)
def test_binscatter(df_fixture, expect_error, df_type, request):
    df = request.getfixturevalue(df_fixture)
    df_to_type = conv(df, df_type)
    num_bins = 20
    if expect_error:
        try:
            binscatter(df_to_type, "x0", "y0", num_bins=num_bins)
        except Exception:
            pass
        else:
            assert False, """Expected an error but binscatter ran successfully"""
    else:
        p = binscatter(df_to_type, "x0", "y0", num_bins=num_bins)
        assert isinstance(p, go.Figure)

        quant_df = binscatter(
            df_to_type, "x0", "y0", num_bins=num_bins, return_type="native"
        )
        # Reference: bin using pandas qcut, groupby bin, take mean, sort by bin
        # Use qcut to assign bins
        bins, _ = quantile_bins(df["x0"], num_bins)
        ref = (
            df.assign(bin=bins)
            .groupby("bin")[["x0", "y0"]]
            .mean()
            .reset_index()
            .sort_values("bin")
        )
        # Convert quant_df to pandas if needed
        match df_type:
            case "pandas":
                assert isinstance(quant_df, pd.DataFrame)
                quant_df_pd = quant_df
            case "polars":
                assert isinstance(quant_df, pl.DataFrame)
                quant_df_pd = quant_df.to_pandas()
            case "duckdb":
                want = duckdb.duckdb.DuckDBPyRelation
                assert isinstance(quant_df, want), f"{want=}\ngot={type(quant_df)}"
                quant_df_pd = quant_df.df()
            case "pyspark":
                assert isinstance(quant_df, pyspark.sql.DataFrame)
                quant_df_pd = quant_df.toPandas()
            case "dask":
                assert isinstance(quant_df, dd.DataFrame), (
                    f"Must be dd.DataFrame is {type(quant_df)}"
                )
                quant_df_pd = quant_df.compute()
        assert isinstance(quant_df_pd, pd.DataFrame)
        assert "x0" in quant_df_pd.columns
        assert "y0" in quant_df_pd.columns
        assert quant_df_pd.shape[1] == 3
        assert quant_df_pd.shape[0] == num_bins

        quant_df_pd.sort_values(quant_df_pd.columns[0], inplace=True)

        assert quant_df_pd.shape == ref.shape
        if df_type in ("pyspark", "dask"):
            rtol = 0.05
            atol = 12
        else:
            rtol = 0.01
            atol = 1e-8

        lhs = quant_df_pd.reset_index()[["x0", "y0"]]
        rhs = ref.reset_index()[["x0", "y0"]]
        np.testing.assert_allclose(
            lhs["x0"].to_numpy(), rhs["x0"].to_numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            lhs["y0"].to_numpy(), rhs["y0"].to_numpy(), rtol=rtol, atol=atol
        )
