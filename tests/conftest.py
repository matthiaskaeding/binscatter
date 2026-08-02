from __future__ import annotations

from collections.abc import Callable

import dask.dataframe as dd
import duckdb
import pandas as pd
import polars as pl
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
            # pandas>=3 stores strings in the "str" dtype, whose missing value
            # createDataFrame turns into the literal string "NaN". Round-trip
            # through object dtype so nulls stay null on the Spark side.
            cleaned = df.astype(object).where(df.notna(), None)
            if not df.isna().all().any():
                return spark.createDataFrame(cleaned)

            # Spark cannot infer a type for a column containing only nulls. Build
            # just enough schema from the pandas dtypes so tests of all-null inputs
            # reach binscatter itself instead of failing in the test converter.
            from pandas.api.types import (
                is_bool_dtype,
                is_datetime64_any_dtype,
                is_integer_dtype,
                is_numeric_dtype,
            )
            from pyspark.sql.types import (
                BooleanType,
                DoubleType,
                LongType,
                StringType,
                StructField,
                StructType,
                TimestampType,
            )

            fields = []
            for name, dtype in df.dtypes.items():
                if is_bool_dtype(dtype):
                    spark_type = BooleanType()
                elif is_integer_dtype(dtype):
                    spark_type = LongType()
                elif is_numeric_dtype(dtype):
                    spark_type = DoubleType()
                elif is_datetime64_any_dtype(dtype):
                    spark_type = TimestampType()
                else:
                    spark_type = StringType()
                fields.append(StructField(str(name), spark_type, nullable=True))
            return spark.createDataFrame(cleaned, schema=StructType(fields))
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


#: Backends kept under ``--quick``. These cover eager row-oriented, eager columnar,
#: and lazy SQL inputs while retaining exact quantiles for comparison with external
#: references. Dask and PySpark are distributed, use approximate quantiles, and stay
#: in the full suite.
QUICK_BACKENDS = frozenset({"pandas", "polars", "duckdb"})


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-pyspark",
        action="store_true",
        help="Run tests that require PySpark (skipped by default)",
    )
    parser.addoption(
        "--quick",
        action="store_true",
        help=(
            "Run only tests marked 'quick', and only on the in-process backends: a "
            "representative sample across every module, for a fast confidence check"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "pyspark: mark test as requiring PySpark and --run-pyspark"
    )
    config.addinivalue_line(
        "markers",
        "quick: representative of its module; included in a --quick smoke run",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--run-pyspark"):
        skip_marker = pytest.mark.skip(
            reason="use --run-pyspark to include PySpark tests"
        )
        for item in items:
            if "pyspark" in item.keywords:
                item.add_marker(skip_marker)

    if not config.getoption("--quick"):
        return

    # ``--quick`` trims two axes at once: which tests run, and which backends they
    # run on. Marking whole functions and letting the backend axis be cut here keeps
    # the marks off individual ``pytest.param`` entries, so a test stays selected
    # when someone adds a backend.
    selected, deselected = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        backend = None
        if callspec:
            backend = callspec.params.get("df_type", callspec.params.get("backend"))
        keep = "quick" in item.keywords and (
            backend is None or backend in QUICK_BACKENDS
        )
        (selected if keep else deselected).append(item)

    if not selected:
        raise pytest.UsageError(
            "--quick selected no tests. Every module should carry at least one "
            "@pytest.mark.quick; see the testing section of AGENTS.md."
        )

    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.fixture(scope="session")
def backend_names() -> list[str]:
    return DF_BACKENDS


@pytest.fixture(scope="session")
def backend_converter() -> Callable:
    return convert_to_backend
