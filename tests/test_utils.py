import narwhals as nw
import narwhals.selectors as ncs
import numpy as np
import pandas as pd
from binscatter.core import (
    _remove_bad_values,
    split_columns,
    partial_out_controls,
    Profile,
)
import polars as pl
import duckdb
import dask.dataframe as dd
import pytest

try:  # pragma: no cover - optional dependency
    from pyspark.sql import SparkSession
except ImportError:  # pragma: no cover - optional dependency
    SparkSession = None


def test_filter_all_numeric_basic():
    data = {"a": [1.0, 2.3, np.nan, 4.0, np.inf, None], "b": [5, 6, 7, None, 9, 10]}
    df_pandas = pd.DataFrame(data)
    df = nw.from_native(df_pandas)

    cols_numeric = tuple(df.select(ncs.numeric()).columns)
    cols_cat = tuple(df.select(ncs.categorical()).columns)
    filtered = _remove_bad_values(df, cols_numeric, cols_cat)
    assert filtered.shape[0] == 2
    assert filtered["a"].to_numpy().tolist() == [1.0, 2.3]
    assert filtered["b"].to_numpy().tolist() == [5, 6]

    df_polars = pl.DataFrame(data)
    df_from_polars = nw.from_native(df_polars)

    cols_numeric_polars = tuple(df_from_polars.select(ncs.numeric()).columns)
    cols_cat_polars = tuple(df_from_polars.select(ncs.categorical()).columns)
    filtered_polars = _remove_bad_values(
        df_from_polars, cols_numeric_polars, cols_cat_polars
    )
    assert filtered_polars.shape[0] == 2
    assert filtered_polars["a"].to_numpy().tolist() == [1.0, 2.3]
    assert filtered_polars["b"].to_numpy().tolist() == [5, 6]


@pytest.mark.parametrize(
    "frame_factory",
    [
        lambda df: nw.from_native(df),
        lambda df: nw.from_native(pl.from_pandas(df)).lazy(),
        lambda df: nw.from_native(duckdb.from_df(df)).lazy(),
        lambda df: nw.from_native(dd.from_pandas(df, npartitions=2)),
    ],
)
def test_get_columns_numeric_categorical(frame_factory):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "cat": ["a", "b", "c"]})
    frame = frame_factory(df)
    numeric, categorical = split_columns(frame)
    assert set(numeric) == {"x", "y"}
    assert set(categorical) == {"cat"}


@pytest.mark.pyspark
def test_get_columns_pyspark():
    if SparkSession is None:
        pytest.skip("PySpark not installed")
    spark = (
        SparkSession.builder.master("local[1]").appName("binscatter-test").getOrCreate()
    )
    try:
        df = spark.createDataFrame(
            pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "cat": ["a", "b", "c"]})
        )
        frame = nw.from_native(df)
        numeric, categorical = split_columns(frame)
        assert set(numeric) == {"x", "y"}
        assert set(categorical) == {"cat"}
    finally:
        spark.stop()


def _sample_profile(bin_count: int, controls: list[str]) -> Profile:
    return Profile(
        x_name="x0",
        y_name="y0",
        num_bins=bin_count,
        bin_name="bin",
        distinct_suffix="test",
        is_lazy_input=True,
        implementation=nw.from_native(
            pd.DataFrame({"x0": [0], "y0": [0]})
        ).implementation,
        regression_features=tuple(controls),
    )


def test_partial_out_controls_matches_closed_form():
    rng = np.random.default_rng(123)
    n = 500
    x = rng.normal(loc=1.0, scale=1.0, size=n)
    control_matrix = rng.normal(loc=np.arange(1, 6), scale=1.0, size=(n, 5))
    intercept = np.ones((n, 1))
    X_design = np.hstack((intercept, control_matrix))
    beta_true = rng.normal(size=6)
    y = X_design @ beta_true + rng.normal(scale=0.3, size=n)
    bins = np.repeat(np.arange(10), repeats=n // 10)
    bins = np.pad(bins, (0, n - bins.size), mode="edge")

    df_native = pd.DataFrame(
        {
            "x0": x,
            "y0": y,
            "c1": control_matrix[:, 0],
            "c2": control_matrix[:, 1],
            "c3": control_matrix[:, 2],
            "c4": control_matrix[:, 3],
            "c5": control_matrix[:, 4],
            "bin": bins,
        }
    )
    frame = nw.from_native(df_native).lazy()
    profile = _sample_profile(10, ["c1", "c2", "c3", "c4", "c5"])

    result, _ = partial_out_controls(frame, profile)
    result = result.collect()
    x_means = result.get_column("x0").to_numpy()
    y_estimated = result.get_column("y0").to_numpy()

    B = pd.get_dummies(df_native["bin"], drop_first=False).to_numpy()
    design = np.column_stack([B, control_matrix])
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta = theta[: profile.num_bins]
    gamma = theta[profile.num_bins :]
    mean_controls = control_matrix.mean(axis=0)
    y_reference = beta + mean_controls @ gamma

    np.testing.assert_allclose(
        x_means, df_native.groupby("bin")["x0"].mean().to_numpy()
    )
    np.testing.assert_allclose(y_estimated, y_reference, rtol=1e-6, atol=1e-6)
