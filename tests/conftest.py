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


#: Behaviour the tests specify and binscatter does not have yet, tracked in #95.
#:
#: Listed here rather than as decorators beside each test: this is a work queue, and
#: one deletable block is easier to draw down than 62 marks scattered over three
#: files -- several of which would have to hang off individual ``pytest.param``
#: entries, since only some parametrizations fail.
#:
#: Every entry is ``strict``, so a test that starts passing turns red and forces its
#: line to be removed. A bare test id covers every parametrization; an id with
#: brackets covers exactly one. Both were confirmed to fail deterministically.
KNOWN_FAILURES: dict[str, tuple[str, ...]] = {
    "#95 group 1: normal equations are built from raw uncentered moments": (
        "tests/test_binscatter.py::test_automatic_selectors_are_location_invariant",
        "tests/test_binscatter.py::test_control_translation_is_invariant_across_fe_routes",
        "tests/test_binscatter.py::test_dpi_selector_caps_bins_for_the_sample_size",
        "tests/test_binscatter.py::test_dpi_selector_ignores_redundant_controls",
        "tests/test_binscatter.py::test_dpi_selector_is_invariant_to_control_units",
        "tests/test_binscatter.py::test_extension_dtypes_match_explicit_dense_encoding[duckdb]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[huge-dask]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[huge-duckdb]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[huge-pandas]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[huge-polars]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[tiny-dask]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[tiny-duckdb]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[tiny-pandas]",
        "tests/test_binscatter.py::test_extreme_control_scales_match_dense_ols[tiny-polars]",
        "tests/test_binscatter.py::test_intervals_are_invariant_to_control_units",
        "tests/test_binscatter.py::test_intervals_are_location_invariant[control-pointwise]",
        "tests/test_binscatter.py::test_intervals_are_location_invariant[control-rbc]",
        "tests/test_binscatter.py::test_intervals_are_location_invariant[x_and_y-rbc]",
        "tests/test_binscatter.py::test_intervals_ignore_redundant_controls",
        "tests/test_binscatter.py::test_large_int64_control_products_match_dense_ols[dask]",
        "tests/test_binscatter.py::test_large_int64_control_products_match_dense_ols[duckdb]",
        "tests/test_binscatter.py::test_large_int64_control_products_match_dense_ols[pandas]",
        "tests/test_binscatter.py::test_large_int64_control_products_match_dense_ols[polars]",
        "tests/test_binscatter.py::test_large_int64_response_sums_match_dense_ols[dask]",
        "tests/test_binscatter.py::test_large_int64_response_sums_match_dense_ols[pandas]",
        "tests/test_binscatter.py::test_large_int64_response_sums_match_dense_ols[polars]",
        "tests/test_binscatter.py::test_multiway_dots_are_stable_at_large_x_y_locations[y_location]",
        "tests/test_binscatter.py::test_near_collinear_controls_match_dense_ols[dask]",
        "tests/test_binscatter.py::test_near_collinear_controls_match_dense_ols[duckdb]",
        "tests/test_binscatter.py::test_near_collinear_controls_match_dense_ols[pandas]",
        "tests/test_binscatter.py::test_near_collinear_controls_match_dense_ols[polars]",
        "tests/test_binscatter.py::test_polynomial_grid_does_not_collapse_at_large_x_location",
        "tests/test_binscatter.py::test_polynomial_line_handles_extreme_but_finite_x_values",
        "tests/test_binscatter.py::test_polynomial_line_is_location_invariant",
        "tests/test_sparse_fe.py::test_fixed_effect_dgp_handles_large_int64_control_products",
        "tests/test_sparse_fe.py::test_fixed_effect_dgp_handles_large_int64_response_sums",
        "tests/test_sparse_fe.py::test_fixed_effect_dgp_handles_near_collinear_numeric_controls",
        "tests/test_sparse_fe.py::test_fixed_effect_dgp_is_invariant_to_numeric_control_units",
    ),
    "#95 group 2: the design is accepted where a refusal is specified": (
        "tests/test_binscatter.py::test_binscatter_rejects_non_integer_explicit_bin_counts",
        "tests/test_binscatter.py::test_control_collinear_with_bins_is_rejected_at_any_scale",
        "tests/test_binscatter.py::test_reserved_output_bin_name_has_a_clear_input_error",
        "tests/test_fixed_effects.py::test_disconnected_bin_group_structure_is_rejected",
        "tests/test_fixed_effects.py::test_one_way_rejects_numeric_control_collinear_with_bins",
        "tests/test_sparse_fe.py::test_multiway_rejects_connected_but_unidentified_design",
        "tests/test_sparse_fe.py::test_multiway_rejects_disconnected_bin_factor_components",
        "tests/test_sparse_fe.py::test_multiway_rejects_numeric_control_collinear_with_bins",
    ),
    "#95 group 3: internal aggregation aliases collide with user column names": (
        "tests/test_fixed_effects.py::test_temporary_aliases_do_not_shadow_control_columns",
        "tests/test_sparse_fe.py::test_multiway_temporary_aliases_do_not_shadow_control_columns",
    ),
    "#95: native output does not preserve the input backend": (
        "tests/test_binscatter.py::test_controlled_native_output_preserves_input_backend[dask]",
        "tests/test_binscatter.py::test_controlled_native_output_preserves_input_backend[duckdb]",
        "tests/test_fixed_effects.py::test_one_way_native_output_preserves_input_backend[dask]",
        "tests/test_fixed_effects.py::test_one_way_native_output_preserves_input_backend[duckdb]",
        "tests/test_sparse_fe.py::test_multiway_native_output_preserves_input_backend[dask]",
        "tests/test_sparse_fe.py::test_multiway_native_output_preserves_input_backend[duckdb]",
    ),
    "#95 group 4: an all-null or empty column reaches quantiles.py as float(None)": (
        "tests/test_binscatter.py::test_binscatter[df_all_invalid-dask]",
        "tests/test_binscatter.py::test_binscatter[df_all_invalid-duckdb]",
        "tests/test_binscatter.py::test_binscatter[df_all_invalid-pandas]",
        "tests/test_binscatter.py::test_binscatter[df_all_invalid-polars]",
        "tests/test_binscatter.py::test_binscatter[df_nulls-dask]",
        "tests/test_binscatter.py::test_binscatter[df_nulls-duckdb]",
        "tests/test_binscatter.py::test_binscatter[df_nulls-pandas]",
        "tests/test_binscatter.py::test_binscatter[df_nulls-polars]",
    ),
}


def _known_failure_reason(nodeid: str) -> str | None:
    """The reason ``nodeid`` is expected to fail, or ``None`` if it is not."""
    for reason, ids in KNOWN_FAILURES.items():
        if nodeid in ids or nodeid.split("[")[0] in ids:
            return reason
    return None


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

    for item in items:
        reason = _known_failure_reason(item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))

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
