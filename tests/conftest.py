from __future__ import annotations

from typing import Callable

import pandas as pd
import polars as pl
import duckdb
import dask.dataframe as dd
import pytest

try:  # pragma: no cover - optional dependency
    from pyspark.sql import SparkSession
except ImportError:  # pragma: no cover - optional dependency
    SparkSession = None

DF_BACKENDS = ["pandas", "polars", "duckdb", "dask"]
if SparkSession is not None:
    DF_BACKENDS.append("pyspark")


def convert_to_backend(df: pd.DataFrame, backend: str):
    match backend:
        case "pandas":
            return df
        case "polars":
            return pl.from_pandas(df)
        case "duckdb":
            return duckdb.from_df(df)
        case "dask":
            return dd.from_pandas(df, npartitions=2)
        case "pyspark":
            if SparkSession is None:
                raise RuntimeError("PySpark not available")
            spark = (
                SparkSession.builder.master("local[1]")
                .appName("binscatter-tests")
                .getOrCreate()
            )
            return spark.createDataFrame(df)
        case _:
            raise ValueError(f"Unknown backend '{backend}'")


def to_pandas_native(df_native):
    if isinstance(df_native, pd.DataFrame):
        return df_native
    if hasattr(df_native, "to_pandas"):
        return df_native.to_pandas()
    if hasattr(df_native, "df"):
        return df_native.df()
    if isinstance(df_native, dd.DataFrame):
        return df_native.compute()
    if SparkSession is not None and hasattr(df_native, "toPandas"):
        return df_native.toPandas()
    raise TypeError(f"Unsupported dataframe type: {type(df_native)}")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pyspark",
        action="store_true",
        help="Run tests that require PySpark (skipped by default)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "pyspark: mark test as requiring PySpark and --run-pyspark"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-pyspark"):
        return

    skip_marker = pytest.mark.skip(reason="use --run-pyspark to include PySpark tests")
    for item in items:
        if "pyspark" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def backend_names() -> list[str]:
    return DF_BACKENDS


@pytest.fixture(scope="session")
def backend_converter() -> Callable:
    return convert_to_backend
