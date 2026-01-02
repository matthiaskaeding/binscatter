import narwhals as nw
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

from tests.conftest import (
    DF_BACKENDS,
    convert_to_backend,
    to_pandas_native,
    SparkSession,
)

DF_TYPE_PARAMS = [
    pytest.param(df_type) for df_type in [b for b in DF_BACKENDS if b != "pyspark"]
]
if "pyspark" in DF_BACKENDS:
    DF_TYPE_PARAMS.append(pytest.param("pyspark", marks=pytest.mark.pyspark))


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_filter_all_numeric_basic(df_type):
    data = {"a": [1.0, 2.3, np.nan, 4.0, np.inf, None], "b": [5, 6, 7, None, 9, 10]}
    df_pandas = pd.DataFrame(data)
    df_native = convert_to_backend(df_pandas, df_type)
    df = nw.from_native(df_native)

    # Get column info using split_columns which handles all backends correctly
    df_lazy = df.lazy()
    cols_numeric, cols_cat = split_columns(df_lazy)
    filtered = _remove_bad_values(df_lazy, cols_numeric, cols_cat)
    # Collect the result for assertions
    filtered_df = filtered.collect()
    assert filtered_df.shape[0] == 2
    assert filtered_df["a"].to_numpy().tolist() == [1.0, 2.3]
    assert filtered_df["b"].to_numpy().tolist() == [5, 6]


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
        polynomial_features=tuple(),
        x_bounds=(0.0, 1.0),
    )


def _assert_remove_bad_values_backend(backend: str):
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, np.inf, 4.0],
            "y": [2.0, 3.0, 4.0, None],
            "cat": ["a", "b", "c", None],
        }
    )
    native = convert_to_backend(df, backend)
    frame = nw.from_native(native)
    lazy = frame.lazy() if isinstance(frame, nw.DataFrame) else frame
    cols_numeric, cols_cat = split_columns(lazy)
    cleaned = _remove_bad_values(lazy, cols_numeric, cols_cat)
    result_native = cleaned.collect().to_native()
    pdf = to_pandas_native(result_native)
    assert len(pdf) == 1
    assert pdf["x"].iloc[0] == 1.0


@pytest.mark.parametrize("backend", [b for b in DF_BACKENDS if b != "pyspark"])
def test_remove_bad_values_all_backends(backend):
    _assert_remove_bad_values_backend(backend)


@pytest.mark.pyspark
def test_remove_bad_values_pyspark():
    if SparkSession is None:
        pytest.skip("PySpark not installed")
    _assert_remove_bad_values_backend("pyspark")


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_partial_out_controls_matches_closed_form(df_type):
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

    df_pandas = pd.DataFrame(
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
    df_native = convert_to_backend(df_pandas, df_type)
    frame = nw.from_native(df_native).lazy()
    profile = _sample_profile(10, ["c1", "c2", "c3", "c4", "c5"])

    result, _ = partial_out_controls(frame, profile)
    result = result.collect()
    x_means = result.get_column("x0").to_numpy()
    y_estimated = result.get_column("y0").to_numpy()

    B = pd.get_dummies(df_pandas["bin"], drop_first=False).to_numpy()
    design = np.column_stack([B, control_matrix])
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta = theta[: profile.num_bins]
    gamma = theta[profile.num_bins :]
    mean_controls = control_matrix.mean(axis=0)
    y_reference = beta + mean_controls @ gamma

    # Use looser tolerance for distributed backends
    if df_type in ("dask", "pyspark"):
        rtol, atol = 1e-3, 1e-2
    else:
        rtol, atol = 1e-6, 1e-6

    np.testing.assert_allclose(
        x_means, df_pandas.groupby("bin")["x0"].mean().to_numpy(), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(y_estimated, y_reference, rtol=rtol, atol=atol)
