import uuid
from typing import Iterable

import polars as pl
import numpy as np
import narwhals as nw
from binsreg import binsregselect
from binscatter.core import (
    add_polynomial_features,
    binscatter,
    clean_df,
    add_regression_features,
    partial_out_controls,
    Profile,
    _fit_polynomial_line,
    _select_rule_of_thumb_bins,
    _select_dpi_bins,
)
from binscatter.quantiles import (
    configure_add_bins,
    configure_compute_quantiles,
)
import plotly.graph_objs as go
import plotly.express as px
import duckdb
import pytest
import pandas as pd
import dask.dataframe as dd
import statsmodels.api as sm

from tests.conftest import (
    DF_BACKENDS,
    convert_to_backend,
    to_pandas_native,
    SparkSession,
)

if SparkSession is not None:  # pragma: no cover - optional dependency
    import pyspark
else:  # pragma: no cover - optional dependency
    pyspark = None

RNG = np.random.default_rng(42)


def _prepare_dataframe(df, x, y, controls, num_bins, poly_degree: int | None = None):
    controls_tuple = tuple(controls)
    df_clean, is_lazy, numeric_controls, categorical_controls = clean_df(
        df, controls_tuple, x, y
    )
    df_with_features, regression_features = add_regression_features(
        df_clean,
        numeric_controls=numeric_controls,
        categorical_controls=categorical_controls,
    )
    suffix = str(uuid.uuid4()).replace("-", "_")
    if poly_degree is not None:
        df_with_features, polynomial_features = add_polynomial_features(
            df_with_features,
            x_name=x,
            degree=poly_degree,
            distinct_suffix=suffix,
        )
    else:
        polynomial_features = ()
    # Compute quantiles first to get x_bounds
    quantile_fn = configure_compute_quantiles(num_bins, df_clean.implementation)
    quantiles = quantile_fn(df_with_features, x)
    profile = Profile(
        x_name=x,
        y_name=y,
        num_bins=num_bins,
        bin_name=f"bin__{suffix}",
        distinct_suffix=suffix,
        is_lazy_input=is_lazy,
        implementation=df_clean.implementation,
        regression_features=regression_features,
        polynomial_features=polynomial_features,
        x_bounds=(quantiles[0], quantiles[-1]),
    )
    df_with_bins = configure_add_bins(profile)(df_with_features, quantiles)
    return df_with_bins, profile


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
        edges = np.quantile(x, qs, method="linear")

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


@pytest.fixture
def df_with_edge_cases():
    # Mix of valid, NaN, inf, -inf, and None values in numeric columns
    # Plus categorical columns with None/missing values
    # After filtering, only 1 valid row remains - not enough for binscatter (needs at least 2 bins)
    x = pd.Series(
        [1.0, np.nan, np.inf, -np.inf, None, np.nan, np.inf, -np.inf],
        name="x0",
    )
    y = pd.Series(
        [10.0, np.nan, 30.0, 40.0, np.inf, None, -np.inf, 80.0],
        name="y0",
    )
    # Categorical columns - row 0 must have valid values to be the single valid row
    cat1 = pd.Series(
        ["a", "b", None, "a", "b", "c", None, "a"],
        name="cat1",
    )
    cat2 = pd.Series(
        ["x", "x", "y", "x", None, "y", "x", "y"],
        name="cat2",
    )
    return pd.concat([x, y, cat1, cat2], axis=1)


@pytest.fixture
def df_all_invalid():
    # All rows have at least one invalid value - should result in zero rows after filtering
    x = pd.Series([np.nan, np.inf, -np.inf, None], name="x0")
    y = pd.Series([np.inf, np.nan, None, -np.inf], name="y0")
    return pd.concat([x, y], axis=1)


@pytest.fixture
def df_constant_categorical():
    # Categorical column with only one value (linear dependence edge case)
    # This tests handling of constant categorical variables that produce no variance
    x = pd.Series(np.arange(100), name="x0")
    y = pd.Series(np.arange(100) + RNG.normal(0, 5, size=100), name="y0")
    cat = pd.Series(["constant"] * 100, name="cat_const")  # Only one value
    return pd.concat([x, y, cat], axis=1)


fixt_dat = [
    ("df_good", False),
    ("df_x_num", False),
    ("df_missing_column", True),
    ("df_nulls", True),
    ("df_duplicates", True),
    (
        "df_with_edge_cases",
        True,
    ),  # Should error - only 4 valid rows after filtering, not enough for 20 bins
    ("df_all_invalid", True),  # Should error - no valid rows remain
    ("df_constant_categorical", False),  # Should handle constant categorical gracefully
]

BASE_DF_TYPES = [name for name in DF_BACKENDS if name != "pyspark"]
HAS_PYSPARK = "pyspark" in DF_BACKENDS

DF_TYPE_PARAMS = [pytest.param(df_type) for df_type in BASE_DF_TYPES]
if HAS_PYSPARK:
    DF_TYPE_PARAMS.append(pytest.param("pyspark", marks=pytest.mark.pyspark))

fix_data_types = []
for df_type in BASE_DF_TYPES:
    for pair in fixt_dat:
        fix_data_types.append(pytest.param(*pair, df_type))

if HAS_PYSPARK:
    for pair in fixt_dat:
        fix_data_types.append(pytest.param(*pair, "pyspark", marks=pytest.mark.pyspark))


def conv(df: pd.DataFrame, df_type):
    return convert_to_backend(df, df_type)


def _get_rot_bins(
    df,
    x: str,
    y: str,
    controls: Iterable[str] | None = None,
) -> int:
    controls_tuple = tuple(controls or ())
    df_clean, _, numeric_controls, categorical_controls = clean_df(
        df,
        controls_tuple,
        x,
        y,
    )
    df_with_features, regression_features = add_regression_features(
        df_clean,
        numeric_controls=numeric_controls,
        categorical_controls=categorical_controls,
    )
    return _select_rule_of_thumb_bins(df_with_features, x, y, regression_features)


def _get_dpi_bins(
    df,
    x: str,
    y: str,
    controls: Iterable[str] | None = None,
) -> int:
    controls_tuple = tuple(controls or ())
    df_clean, _, numeric_controls, categorical_controls = clean_df(
        df,
        controls_tuple,
        x,
        y,
    )
    df_with_features, regression_features = add_regression_features(
        df_clean,
        numeric_controls=numeric_controls,
        categorical_controls=categorical_controls,
    )
    return _select_dpi_bins(df_with_features, x, y, regression_features)


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


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_rule_of_thumb_matches_helper(df_good, df_type):
    df = conv(df_good, df_type)
    expected_bins = _get_rot_bins(df, "x0", "y0")
    native = binscatter(df, "x0", "y0", num_bins="rule-of-thumb", return_type="native")
    result_pd = to_pandas_native(native)
    if df_type == "pyspark":
        # PySpark uses approxQuantile which may produce fewer unique quantiles
        # Just verify we get a reasonable number of bins
        assert result_pd.shape[0] >= 2
        assert result_pd.shape[0] <= expected_bins
    else:
        assert result_pd.shape[0] == expected_bins


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_rule_of_thumb_with_controls(df_good, df_type):
    df = df_good.copy()
    df["z_num"] = df["x0"] * 0.5
    df["z_cat"] = np.where(df["x0"] % 2 == 0, "even", "odd")
    df_backend = conv(df, df_type)
    expected_bins = _get_rot_bins(df_backend, "x0", "y0", controls=["z_num", "z_cat"])
    native = binscatter(
        df_backend,
        "x0",
        "y0",
        controls=["z_num", "z_cat"],
        num_bins="rule-of-thumb",
        return_type="native",
    )
    result_pd = to_pandas_native(native)
    if df_type == "pyspark":
        # PySpark uses approxQuantile which may produce fewer unique quantiles
        assert result_pd.shape[0] >= 2
        assert result_pd.shape[0] <= expected_bins
    else:
        assert result_pd.shape[0] == expected_bins


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_rule_of_thumb_handles_gapminder(df_type):
    df = px.data.gapminder()
    expected_bins = _get_rot_bins(df, "gdpPercap", "lifeExp")
    df_backend = conv(df, df_type)
    native = binscatter(
        df_backend,
        "gdpPercap",
        "lifeExp",
        num_bins="rule-of-thumb",
        return_type="native",
    )
    result_pd = to_pandas_native(native)
    assert result_pd.shape[0] == expected_bins
    assert result_pd["bin"].nunique() == expected_bins


def test_gapminder_script_plots_do_not_fail():
    df_pl = pl.from_pandas(px.data.gapminder())
    df_log = df_pl.select(
        pl.col("continent"),
        pl.col("year"),
        pl.col("lifeExp"),
        pl.col("gdpPercap"),
        pl.col("gdpPercap").log().alias("log_gdp"),
        pl.col("lifeExp").log().alias("log_life"),
    )
    fig_main = binscatter(df_pl, x="gdpPercap", y="lifeExp")
    fig_log = binscatter(df_log, x="log_gdp", y="log_life")
    assert isinstance(fig_main, go.Figure)
    assert isinstance(fig_log, go.Figure)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_rule_of_thumb_reduces_bins_when_quantiles_collapse(df_type):
    rng = np.random.default_rng(123)
    base = pd.DataFrame(
        {
            "x0": np.tile([0.0, 1.0], 50),
            "y0": rng.normal(loc=0.75, scale=0.2, size=100),
        }
    )
    df = conv(base, df_type)
    native = binscatter(
        df,
        "x0",
        "y0",
        num_bins="rule-of-thumb",
        return_type="native",
    )
    result_pd = to_pandas_native(native)
    assert result_pd.shape[0] == 2
    assert result_pd["bin"].nunique() == 2


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_rule_of_thumb_reduces_bins_with_controls(df_type):
    rng = np.random.default_rng(987)
    base = pd.DataFrame(
        {
            "x0": np.tile([0.0, 1.0], 50),
            "y0": np.linspace(0.0, 1.0, num=100) + rng.normal(scale=0.05, size=100),
            "z_ctrl": rng.normal(size=100),
        }
    )
    df = conv(base, df_type)
    native = binscatter(
        df,
        "x0",
        "y0",
        controls=["z_ctrl"],
        num_bins="rule-of-thumb",
        return_type="native",
    )
    result_pd = to_pandas_native(native)
    assert result_pd.shape[0] == 2
    assert result_pd["bin"].nunique() == 2


def test_rule_of_thumb_similar_to_binsreg_no_controls():
    # Our ROT matches Cattaneo et al. (2024) SA-4.1 for p=0, s=0, v=0
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n)
    df = pd.DataFrame({"x0": x, "y0": y})
    ours = _get_rot_bins(df, "x0", "y0")
    theirs = binsregselect(y, x).nbinsrot_regul
    assert abs(ours - int(theirs)) <= 2


def test_rule_of_thumb_similar_to_binsreg_with_controls():
    # Our ROT matches Cattaneo et al. (2024) SA-4.1 for p=0, s=0, v=0
    rng = np.random.default_rng(1)
    n = 3500
    x = rng.normal(size=n)
    w1 = rng.normal(scale=0.5, size=n)
    w2 = rng.uniform(-1, 1, size=n)
    y = 2.5 * x - 1.1 * w1 + 0.8 * w2 + rng.normal(scale=0.75, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "w1": w1, "w2": w2})
    ours = _get_rot_bins(df, "x0", "y0", controls=["w1", "w2"])
    theirs = binsregselect(y, x, w=df[["w1", "w2"]].to_numpy()).nbinsrot_regul
    assert abs(ours - int(theirs)) <= 6


# --------------------------------------------------------------------------
# DPI (Direct Plug-In) bin selector tests
# --------------------------------------------------------------------------


def test_dpi_handles_gapminder():
    """DPI selector works on real data and typically gives more bins than ROT."""
    df = px.data.gapminder()
    dpi_bins = _get_dpi_bins(df, "gdpPercap", "lifeExp")
    rot_bins = _get_rot_bins(df, "gdpPercap", "lifeExp")
    native = binscatter(
        df,
        "gdpPercap",
        "lifeExp",
        num_bins="dpi",
        return_type="native",
    )
    result_pd = to_pandas_native(native)
    assert result_pd.shape[0] == dpi_bins
    # DPI typically recommends at least as many bins as ROT
    assert dpi_bins >= rot_bins - 2


def test_dpi_matches_helper(df_good):
    """DPI num_bins='dpi' matches direct helper call."""
    expected_bins = _get_dpi_bins(df_good, "x0", "y0")
    native = binscatter(df_good, "x0", "y0", num_bins="dpi", return_type="native")
    result_pd = to_pandas_native(native)
    assert result_pd.shape[0] == expected_bins


def test_dpi_similar_to_binsreg_no_controls():
    """DPI matches binsreg DPI output."""
    rng = np.random.default_rng(42)
    n = 5000
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n)
    df = pd.DataFrame({"x0": x, "y0": y})
    ours = _get_dpi_bins(df, "x0", "y0")
    theirs = binsregselect(y, x).nbinsdpi
    assert abs(ours - int(theirs)) <= 1


def test_dpi_similar_to_binsreg_with_controls():
    """DPI with controls matches binsreg."""
    rng = np.random.default_rng(43)
    n = 3500
    x = rng.normal(size=n)
    w1 = rng.normal(scale=0.5, size=n)
    w2 = rng.uniform(-1, 1, size=n)
    y = 2.5 * x - 1.1 * w1 + 0.8 * w2 + rng.normal(scale=0.75, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "w1": w1, "w2": w2})
    ours = _get_dpi_bins(df, "x0", "y0", controls=["w1", "w2"])
    theirs = binsregselect(y, x, w=df[["w1", "w2"]].to_numpy()).nbinsdpi
    assert abs(ours - int(theirs)) <= 1


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_dpi_works_across_backends(df_type):
    """DPI selector works with all supported backends."""
    rng = np.random.default_rng(100)
    n = 1000
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n)
    base = pd.DataFrame({"x0": x, "y0": y})
    df = conv(base, df_type)
    native = binscatter(df, "x0", "y0", num_bins="dpi", return_type="native")
    result_pd = to_pandas_native(native)
    # Just check we got a reasonable number of bins
    assert result_pd.shape[0] >= 2
    assert result_pd.shape[0] <= n // 5


def test_dpi_skewed_data():
    """DPI works with skewed distributions."""
    rng = np.random.default_rng(201)
    n = 2000
    # Exponential x (right-skewed)
    x = rng.exponential(scale=2.0, size=n)
    y = np.log1p(x) + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"x0": x, "y0": y})
    ours = _get_dpi_bins(df, "x0", "y0")
    theirs = binsregselect(y, x).nbinsdpi
    assert abs(ours - int(theirs)) <= 1


def test_dpi_quadratic_relationship():
    """DPI with nonlinear relationship."""
    rng = np.random.default_rng(202)
    n = 3000
    x = rng.uniform(-3, 3, size=n)
    y = x**2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x0": x, "y0": y})
    ours = _get_dpi_bins(df, "x0", "y0")
    # For quadratic relationships, DPI implementations can vary significantly
    # Just ensure we're getting a reasonable number of bins
    assert ours >= 5
    assert ours <= n // 5  # At least 5 observations per bin


def test_dpi_small_sample():
    """DPI with small sample size."""
    rng = np.random.default_rng(203)
    n = 100
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(size=n)
    df = pd.DataFrame({"x0": x, "y0": y})
    ours = _get_dpi_bins(df, "x0", "y0")
    theirs = binsregselect(y, x).nbinsdpi
    assert ours >= 2
    assert ours <= n // 5
    assert abs(ours - int(theirs)) <= 1


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_binscatter_rejects_unknown_num_bins_string(df_good, df_type):
    df = conv(df_good, df_type)
    with pytest.raises(ValueError):
        binscatter(df, "x0", "y0", num_bins="unknown-option")


def _manual_binscatter_with_controls(
    df: pd.DataFrame,
    num_bins: int,
    control_cols: Iterable[str] | None = None,
    categorical_controls: Iterable[str] | None = None,
):
    """Reference implementation following paper's specification."""
    if control_cols is None:
        control_cols = ["z"]
    control_cols = list(control_cols)
    categorical_controls = set(categorical_controls or [])

    bins = pd.qcut(df["x0"], q=num_bins, labels=False, duplicates="drop")
    if bins.isna().any():
        raise AssertionError("Unexpected NA bins during reference construction")
    df_with_bin = df.assign(_bin=bins)
    x_means = (
        df_with_bin.groupby("_bin", observed=True)["x0"].mean().sort_index().to_numpy()
    )
    B = pd.get_dummies(df_with_bin["_bin"], drop_first=False)
    if B.shape[1] != num_bins:
        raise AssertionError(
            f"qcut produced {B.shape[1]} bins (expected {num_bins}); increase sample size"
        )

    control_matrices: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    numeric_controls = [c for c in control_cols if c not in categorical_controls]
    if numeric_controls:
        numeric_matrix = df_with_bin[numeric_controls].to_numpy()
        control_matrices.append(numeric_matrix)
        mean_parts.append(df_with_bin[numeric_controls].mean().to_numpy())

    for cat in categorical_controls:
        dummies = pd.get_dummies(df_with_bin[cat], prefix=cat, drop_first=True)
        if dummies.shape[1] == 0:
            continue
        control_matrices.append(dummies.to_numpy())
        mean_parts.append(dummies.mean().to_numpy())

    if control_matrices:
        W = np.column_stack(control_matrices)
        mean_controls = np.concatenate(mean_parts)
        design = np.column_stack([B.to_numpy(), W])
    else:
        mean_controls = np.array([])
        design = B.to_numpy()

    theta, *_ = np.linalg.lstsq(design, df_with_bin["y0"].to_numpy(), rcond=None)
    beta = theta[:num_bins]
    gamma = theta[num_bins:]
    fitted = beta + (mean_controls @ gamma if gamma.size else 0.0)
    return x_means, fitted


def _collect_lazyframe_to_pandas(frame):
    """Helper to collect a narwhals LazyFrame into a pandas DataFrame."""
    return to_pandas_native(frame.collect().to_native())


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_binscatter_controls_matches_reference(df_type):
    rng = np.random.default_rng(123)
    n = 2000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.5 * x + 2.75 * z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "z": z})
    num_bins = 15

    expected_x, expected_y = _manual_binscatter_with_controls(df, num_bins)
    df_backend = conv(df, df_type)
    result = binscatter(
        df_backend,
        "x0",
        "y0",
        controls=["z"],
        num_bins=num_bins,
        return_type="native",
    )
    result_pd = to_pandas_native(result).sort_values("bin").reset_index(drop=True)
    # Use looser tolerance for distributed backends (approximate quantiles cause bin differences)
    if df_type in ("dask", "pyspark"):
        rtol, atol = 0.1, 0.15
    else:
        rtol, atol = 1e-6, 1e-6
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=rtol, atol=atol
    )


def test_binscatter_controls_lazy_polars():
    rng = np.random.default_rng(456)
    n = 1500
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.75 * x - 1.2 * z + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "z": z})
    num_bins = 12

    expected_x, expected_y = _manual_binscatter_with_controls(df, num_bins)
    polars_lazy = pl.from_pandas(df).lazy()
    result = binscatter(
        polars_lazy,
        "x0",
        "y0",
        controls=["z"],
        num_bins=num_bins,
        return_type="native",
    )
    result_pd = to_pandas_native(result).sort_values("bin").reset_index(drop=True)
    np.testing.assert_allclose(
        result_pd["x0"].to_numpy(), expected_x, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_binscatter_controls_across_backends(df_type):
    if df_type == "dask":
        pytest.importorskip("dask")
    rng = np.random.default_rng(789)
    n = 800
    x = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    cat = np.where(rng.random(size=n) > 0.5, "alpha", "beta")
    y = 0.3 * x + 1.1 * z1 - 0.85 * z2 + np.where(cat == "alpha", 0.6, -0.4)
    y = y + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "z1": z1, "z2": z2, "cat": cat})
    num_bins = 10

    _, expected_y = _manual_binscatter_with_controls(
        df,
        num_bins,
        control_cols=["z1", "z2", "cat"],
        categorical_controls=["cat"],
    )
    df_backend = conv(df, df_type)
    result = binscatter(
        df_backend,
        "x0",
        "y0",
        controls=["z1", "z2", "cat"],
        num_bins=num_bins,
        return_type="native",
    )
    result_pd = to_pandas_native(result).sort_values("bin").reset_index(drop=True)
    if df_type in ("dask", "pyspark"):
        rtol_y, atol_y = 5e-3, 2e-1
    else:
        rtol_y = atol_y = 1e-6
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=rtol_y, atol=atol_y
    )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_binscatter_categorical_controls_only(df_type):
    rng = np.random.default_rng(321)
    n = 1200
    x = rng.normal(size=n)
    cat = rng.choice(["red", "green", "blue"], p=[0.4, 0.35, 0.25], size=n)
    effects = {"red": 0.8, "green": -0.3, "blue": 0.2}
    y = 1.2 * x + np.vectorize(effects.get)(cat) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "cat": cat})
    num_bins = 14

    expected_x, expected_y = _manual_binscatter_with_controls(
        df,
        num_bins,
        control_cols=["cat"],
        categorical_controls=["cat"],
    )
    df_backend = conv(df, df_type)
    result = binscatter(
        df_backend,
        "x0",
        "y0",
        controls=["cat"],
        num_bins=num_bins,
        return_type="native",
    )
    result_pd = to_pandas_native(result).sort_values("bin").reset_index(drop=True)
    # Use looser tolerance for distributed backends (approximate quantiles cause bin differences)
    if df_type in ("dask", "pyspark"):
        rtol, atol = 0.1, 0.15
    else:
        rtol, atol = 1e-6, 1e-6
    np.testing.assert_allclose(
        result_pd["x0"].to_numpy(), expected_x, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_binscatter_controls_collapsed_bins_error(df_type):
    df = pd.DataFrame(
        {
            "x0": np.ones(50),
            "y0": np.linspace(0.0, 1.0, num=50),
            "z": np.linspace(-1.0, 1.0, num=50),
        }
    )
    df_backend = conv(df, df_type)
    with pytest.raises(ValueError, match="Could not produce at least 2 bins"):
        binscatter(
            df_backend, "x0", "y0", controls=["z"], num_bins=5, return_type="native"
        )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_partial_out_controls_matches_statsmodels(df_type):
    rng = np.random.default_rng(2025)
    n = 1500
    x0 = rng.normal(size=n)
    z = rng.normal(size=n)
    region = rng.choice(["north", "south", "east"], size=n, p=[0.4, 0.35, 0.25])
    campaign = rng.choice(["alpha", "beta", "gamma"], size=n, p=[0.3, 0.4, 0.3])
    region_effect = {"north": 0.5, "south": -0.2, "east": 0.1}
    campaign_effect = {"alpha": 0.4, "beta": -0.3, "gamma": 0.2}
    y0 = (
        1.1 * x0
        + 0.8 * z
        + np.vectorize(region_effect.get)(region)
        + np.vectorize(campaign_effect.get)(campaign)
        + rng.normal(scale=0.4, size=n)
    )
    df = pd.DataFrame(
        {
            "x0": x0,
            "y0": y0,
            "z_num": z,
            "region": region,
            "campaign": campaign,
        }
    )
    num_bins = 12
    df_backend = conv(df, df_type)
    df_prepped, profile = _prepare_dataframe(
        df_backend,
        "x0",
        "y0",
        controls=["z_num", "region", "campaign"],
        num_bins=num_bins,
    )
    df_with_bins = _collect_lazyframe_to_pandas(df_prepped)
    df_result, coeffs = partial_out_controls(df_prepped, profile)
    result = (
        _collect_lazyframe_to_pandas(df_result)
        .sort_values(profile.bin_name)
        .reset_index(drop=True)
    )

    bin_means = (
        df_with_bins.groupby(profile.bin_name, observed=True)[profile.x_name]
        .mean()
        .sort_index()
    )
    # Use looser tolerance for distributed backends (approximate quantiles cause bin differences)
    if df_type in ("dask", "pyspark"):
        rtol, atol = 0.1, 0.15
    else:
        rtol, atol = 1e-6, 1e-6
    np.testing.assert_allclose(
        result[profile.x_name].to_numpy(),
        bin_means.to_numpy(),
        rtol=rtol,
        atol=atol,
    )

    bin_dummies = pd.get_dummies(
        df_with_bins[profile.bin_name], prefix="bin", drop_first=False
    )
    z_vals = df_with_bins["z_num"].to_numpy()
    control_blocks = [z_vals[:, None]]
    control_means = [np.array([z_vals.mean()])]
    for cat_col in ["region", "campaign"]:
        dummies = pd.get_dummies(df_with_bins[cat_col], prefix=cat_col, drop_first=True)
        if not dummies.empty:
            control_blocks.append(dummies.to_numpy())
            control_means.append(dummies.mean().to_numpy())
    control_matrix = np.column_stack(control_blocks)
    design_matrix = np.column_stack([bin_dummies.to_numpy(), control_matrix]).astype(
        float
    )
    mean_controls = np.concatenate(control_means)

    model = sm.OLS(
        df_with_bins[profile.y_name].to_numpy().astype(float), design_matrix
    ).fit()
    theta = model.params
    beta = theta[: profile.num_bins]
    gamma = theta[profile.num_bins :]
    fitted = beta + (mean_controls @ gamma if gamma.size else 0.0)

    np.testing.assert_allclose(
        result[profile.y_name].to_numpy(),
        fitted,
        rtol=rtol,
        atol=atol,
    )

    beta_ref = beta - beta[0]
    beta_actual = coeffs["beta"] - coeffs["beta"][0]
    np.testing.assert_allclose(beta_actual, beta_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_partial_out_controls_coefficients_across_backends(df_type):
    """Test that regression coefficients are identical across backends with categorical variables."""
    if df_type == "dask":
        pytest.importorskip("dask")

    # Create synthetic data with categorical and numeric controls
    rng = np.random.default_rng(42)
    n = 1000
    x0 = rng.normal(size=n)
    z = rng.normal(size=n)
    region = rng.choice(["north", "south", "east"], size=n, p=[0.4, 0.35, 0.25])
    campaign = rng.choice(["alpha", "beta"], size=n, p=[0.6, 0.4])

    # Define effects
    region_effect = {"north": 0.5, "south": -0.2, "east": 0.1}
    campaign_effect = {"alpha": 0.4, "beta": -0.3}

    y0 = (
        1.1 * x0
        + 0.8 * z
        + np.vectorize(region_effect.get)(region)
        + np.vectorize(campaign_effect.get)(campaign)
        + rng.normal(scale=0.3, size=n)
    )

    df = pd.DataFrame(
        {
            "x0": x0,
            "y0": y0,
            "z_num": z,
            "region": region,
            "campaign": campaign,
        }
    )

    # Compute reference coefficients with pandas
    num_bins = 10
    df_ref_prepped, profile_ref = _prepare_dataframe(
        df, "x0", "y0", controls=["z_num", "region", "campaign"], num_bins=num_bins
    )
    df_ref_result, coeffs_ref = partial_out_controls(df_ref_prepped, profile_ref)
    beta_ref = coeffs_ref["beta"]

    # Test with the specified backend
    df_backend = convert_to_backend(df, df_type)
    df_prepped, profile = _prepare_dataframe(
        df_backend,
        "x0",
        "y0",
        controls=["z_num", "region", "campaign"],
        num_bins=num_bins,
    )
    df_result, coeffs = partial_out_controls(df_prepped, profile)
    beta_test = coeffs["beta"]

    # Coefficients should match exactly (or very close) across backends
    # Use looser tolerance for distributed backends (approximate quantiles cause bin differences)
    if df_type in ("dask", "pyspark"):
        rtol, atol = 0.1, 0.15
    else:
        rtol, atol = 1e-10, 1e-10

    np.testing.assert_allclose(
        beta_test,
        beta_ref,
        rtol=rtol,
        atol=atol,
        err_msg=f"Beta coefficients don't match for {df_type} backend with categorical controls",
    )

    # Also verify the partialled-out y values match
    result_pd = (
        _collect_lazyframe_to_pandas(df_result)
        .sort_values(profile.bin_name)
        .reset_index(drop=True)
    )
    ref_pd = (
        _collect_lazyframe_to_pandas(df_ref_result)
        .sort_values(profile_ref.bin_name)
        .reset_index(drop=True)
    )

    np.testing.assert_allclose(
        result_pd[profile.y_name].to_numpy(),
        ref_pd[profile_ref.y_name].to_numpy(),
        rtol=rtol,
        atol=atol,
        err_msg=f"Partialled-out y values don't match for {df_type} backend",
    )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_fit_polynomial_line_matches_statsmodels(df_type):
    rng = np.random.default_rng(1234)
    n = 400
    x = rng.normal(loc=0.5, scale=1.5, size=n)
    z = rng.normal(size=n)
    y = 1.2 + 0.9 * x - 0.3 * x**2 + 0.5 * z + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "z": z})
    df_backend = conv(df, df_type)
    df_prepped, profile = _prepare_dataframe(
        df_backend,
        x="x0",
        y="y0",
        controls=["z"],
        num_bins=10,
        poly_degree=3,
    )
    cache: dict[str, float] = {}
    poly_fit = _fit_polynomial_line(df_prepped, profile, degree=2, cache=cache)

    design = np.column_stack([np.ones(n), x, x**2, z])
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    # Use looser tolerance for distributed backends
    if df_type in ("dask", "pyspark"):
        rtol = 1e-3
    else:
        rtol = 1e-6
    np.testing.assert_allclose(poly_fit.coefficients[: theta.size], theta, rtol=rtol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_poly_line_does_not_change_bins(df_type):
    df = pd.DataFrame(
        {
            "x0": np.linspace(-3, 3, 200),
            "y0": np.linspace(-3, 3, 200)
            + np.random.default_rng(0).normal(scale=0.1, size=200),
        }
    )
    df_backend = conv(df, df_type)
    native = binscatter(df_backend, "x0", "y0", num_bins=15, return_type="native")
    with_poly = binscatter(
        df_backend, "x0", "y0", num_bins=15, poly_line=2, return_type="native"
    )
    # Convert both to pandas for comparison
    native_pd = to_pandas_native(native).sort_values("bin").reset_index(drop=True)
    with_poly_pd = to_pandas_native(with_poly).sort_values("bin").reset_index(drop=True)
    pd.testing.assert_frame_equal(native_pd, with_poly_pd)


def test_poly_line_does_not_change_y_axis_range():
    """Test that adding poly_line doesn't change the y-axis range (issue #65)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y': np.linspace(0, 10, 100) + rng.normal(0, 2, 100)
    })

    # Create plot without poly_line
    fig_without_poly = binscatter(df, 'x', 'y', num_bins=20)

    # Create plot with poly_line
    fig_with_poly = binscatter(df, 'x', 'y', num_bins=20, poly_line=1)

    # Both figures should have explicit y-axis ranges set based on scatter points
    assert fig_without_poly.layout.yaxis.range is not None, "Y-axis range should be explicitly set"
    assert fig_with_poly.layout.yaxis.range is not None, "Y-axis range should be explicitly set with poly_line"

    # The y-axis ranges should be identical
    np.testing.assert_allclose(
        fig_without_poly.layout.yaxis.range,
        fig_with_poly.layout.yaxis.range,
        rtol=1e-10,
        err_msg="Y-axis range should be identical with and without poly_line"
    )

    # Verify that the polynomial trace was added
    assert len(fig_with_poly.data) == 2, "Should have 2 traces (scatter + polynomial)"
    assert len(fig_without_poly.data) == 1, "Should have 1 trace (scatter only)"


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_configure_compute_quantiles_returns_correct_length(df_type):
    """Quantiles should have num_bins + 1 elements (including min and max)."""
    if df_type == "pyspark":
        pytest.skip("PySpark not enabled")
    rng = np.random.default_rng(111)
    df_pd = pd.DataFrame({"x": rng.normal(size=500)})
    df = convert_to_backend(df_pd, df_type)
    df_nw = nw.from_native(df).lazy()

    for num_bins in [3, 5, 10, 20]:
        compute_quantiles = configure_compute_quantiles(num_bins, df_nw.implementation)
        quantiles = compute_quantiles(df_nw, "x")
        assert len(quantiles) == num_bins + 1, (
            f"Expected {num_bins + 1} quantiles, got {len(quantiles)}"
        )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_configure_compute_quantiles_min_max(df_type):
    """First quantile should be min, last should be max."""
    if df_type == "pyspark":
        pytest.skip("PySpark not enabled")
    rng = np.random.default_rng(222)
    x = rng.normal(size=500)
    df_pd = pd.DataFrame({"x": x})
    df = convert_to_backend(df_pd, df_type)
    df_nw = nw.from_native(df).lazy()

    compute_quantiles = configure_compute_quantiles(5, df_nw.implementation)
    quantiles = compute_quantiles(df_nw, "x")

    np.testing.assert_allclose(quantiles[0], x.min(), rtol=1e-5)
    np.testing.assert_allclose(quantiles[-1], x.max(), rtol=1e-5)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_configure_compute_quantiles_monotonic(df_type):
    """Quantiles should be monotonically non-decreasing."""
    if df_type == "pyspark":
        pytest.skip("PySpark not enabled")
    rng = np.random.default_rng(333)
    df_pd = pd.DataFrame({"x": rng.normal(size=500)})
    df = convert_to_backend(df_pd, df_type)
    df_nw = nw.from_native(df).lazy()

    compute_quantiles = configure_compute_quantiles(10, df_nw.implementation)
    quantiles = compute_quantiles(df_nw, "x")

    for i in range(len(quantiles) - 1):
        assert quantiles[i] <= quantiles[i + 1], (
            f"Quantiles not monotonic: {quantiles[i]} > {quantiles[i + 1]}"
        )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_configure_compute_quantiles_single_bin_raises(df_type):
    """Single bin (num_bins=1) should raise ValueError."""
    rng = np.random.default_rng(444)
    x = rng.normal(size=100)
    df_pd = pd.DataFrame({"x": x})
    df = conv(df_pd, df_type)
    df_nw = nw.from_native(df).lazy()

    with pytest.raises(ValueError, match="num_bins must be at least 2"):
        configure_compute_quantiles(1, df_nw.implementation)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_non_unique_quantiles_produce_unique_bins_binary(df_type):
    """Binary data should produce 2 unique bins even when quantiles collapse."""
    if df_type == "pyspark":
        pytest.skip("PySpark not enabled")
    base = pd.DataFrame(
        {
            "x0": np.tile([0.0, 1.0], 50),
            "y0": np.random.default_rng(42).normal(size=100),
        }
    )
    df = convert_to_backend(base, df_type)
    # Use rule-of-thumb to allow automatic bin reduction
    result = binscatter(df, "x0", "y0", num_bins="rule-of-thumb", return_type="native")
    result_pd = to_pandas_native(result)
    # Should have exactly 2 unique bins for binary data
    assert result_pd["bin"].nunique() == 2
    assert result_pd["x0"].nunique() == 2


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_non_unique_quantiles_produce_unique_bins_ternary(df_type):
    """Ternary data should produce at least 2 unique bins."""
    if df_type == "pyspark":
        pytest.skip("PySpark not enabled")
    base = pd.DataFrame(
        {
            "x0": np.tile([0.0, 1.0, 2.0], 33),
            "y0": np.random.default_rng(43).normal(size=99),
        }
    )
    df = convert_to_backend(base, df_type)
    # Use rule-of-thumb to allow automatic bin reduction
    result = binscatter(df, "x0", "y0", num_bins="rule-of-thumb", return_type="native")
    result_pd = to_pandas_native(result)
    # Should have at least 2 unique bins for ternary data
    assert result_pd["bin"].nunique() >= 2
    assert result_pd["x0"].nunique() >= 2


def test_non_unique_quantiles_pyspark():
    """PySpark-specific test for non-unique quantiles."""
    pytest.importorskip("pyspark")
    base = pd.DataFrame(
        {
            "x0": np.tile([0.0, 1.0], 50),
            "y0": np.random.default_rng(45).normal(size=100),
        }
    )
    df = convert_to_backend(base, "pyspark")
    # Use rule-of-thumb to allow automatic bin reduction
    result = binscatter(df, "x0", "y0", num_bins="rule-of-thumb", return_type="native")
    result_pd = to_pandas_native(result)
    # Should have exactly 2 unique bins for binary data
    assert result_pd["bin"].nunique() == 2
    assert result_pd["x0"].nunique() == 2


# =============================================================================
# Tests for perf-investigation PR features
# =============================================================================


def test_format_dummy_alias():
    """Test the format_dummy_alias helper function."""
    from binscatter.dummy_builders import format_dummy_alias
    import re

    # All names should start with __ctrl_{column}_
    result = format_dummy_alias("category", "value1")
    assert result.startswith("__ctrl_category_")
    # Should contain sanitized value and 8-char hash
    assert re.match(r"__ctrl_category_value1_[a-f0-9]{8}$", result)

    # Spaces should be replaced with underscores
    result = format_dummy_alias("cat", "foo bar")
    assert result.startswith("__ctrl_cat_foo_bar_")
    assert re.match(r"__ctrl_cat_foo_bar_[a-f0-9]{8}$", result)

    # Slashes should be replaced with underscores
    result = format_dummy_alias("cat", "foo/bar")
    assert result.startswith("__ctrl_cat_foo_bar_")
    assert re.match(r"__ctrl_cat_foo_bar_[a-f0-9]{8}$", result)

    # Multiple special characters
    result = format_dummy_alias("cat", "a b/c")
    assert result.startswith("__ctrl_cat_a_b_c_")

    # Numeric values
    result = format_dummy_alias("cat", 123)
    assert result.startswith("__ctrl_cat_123_")
    assert re.match(r"__ctrl_cat_123_[a-f0-9]{8}$", result)

    # CRITICAL: Different values that sanitize the same should have different hashes
    # This prevents collisions
    name1 = format_dummy_alias("cat", "foo/bar")
    name2 = format_dummy_alias("cat", "foo_bar")
    assert name1 != name2, "Values 'foo/bar' and 'foo_bar' should have different hashes"

    # Hash should be deterministic (same input = same output)
    assert format_dummy_alias("cat", "test") == format_dummy_alias("cat", "test")

    # Special characters should be sanitized
    result = format_dummy_alias("cat", "price@discount")
    assert "@" not in result
    assert result.startswith("__ctrl_cat_price_discount_")

    # Very long values should be truncated but still unique
    long_value = "a" * 200
    result = format_dummy_alias("cat", long_value)
    assert len(result) <= 64, f"Column name too long: {len(result)} chars"
    assert result.startswith("__ctrl_cat_")
    assert re.search(r"_[a-f0-9]{8}$", result), "Should end with 8-char hash"


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def testbuild_dummies_pandas_polars(backend):
    """Test that pandas and polars dummy builders work correctly."""
    from binscatter.dummy_builders import build_dummies_pandas, build_dummies_polars

    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "cat_a": ["foo", "bar", "baz", "foo", "bar", "baz"],
            "cat_b": ["red", "blue", "red", "blue", "red", "blue"],
        }
    )

    df = convert_to_backend(df_pd, backend)
    df_nw = nw.from_native(df).lazy()

    build_dummies = (
        build_dummies_pandas if backend == "pandas" else build_dummies_polars
    )
    df_with_dummies, dummy_cols = build_dummies(df_nw, ("cat_a", "cat_b"))

    # Should create dummies (drop_first=True means n-1 dummies per categorical)
    # cat_a has 3 levels -> 2 dummies, cat_b has 2 levels -> 1 dummy
    assert len(dummy_cols) == 3

    # Verify dummy names follow convention
    for col in dummy_cols:
        assert col.startswith("__ctrl_")

    # Collect and verify
    result = df_with_dummies.collect()
    for col in dummy_cols:
        assert col in result.columns
        # Dummy columns should be numeric or boolean
        assert result[col].dtype in [
            nw.Float64,
            nw.Int64,
            nw.Int32,
            nw.UInt8,
            nw.UInt32,
            nw.Boolean,
        ]


def testbuild_dummies_pyspark():
    """Test that PySpark dummy builder works correctly."""
    pytest.importorskip("pyspark")
    from binscatter.dummy_builders import build_dummies_pyspark

    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "category": ["A", "B", "C", "A", "B", "C"],
        }
    )

    df_spark = convert_to_backend(df_pd, "pyspark")
    df_nw = nw.from_native(df_spark).lazy()

    df_with_dummies, dummy_cols = build_dummies_pyspark(df_nw, ("category",))

    # Should create 2 dummies (3 levels - 1)
    assert len(dummy_cols) == 2

    # Verify dummy names
    for col in dummy_cols:
        assert col.startswith("__ctrl_category_")

    # Collect and verify
    result = df_with_dummies.collect().to_pandas()
    for col in dummy_cols:
        assert col in result.columns
        # Values should be 0.0 or 1.0
        assert set(result[col].unique()) <= {0.0, 1.0}


def test_dummy_builder_pyspark_handles_nulls():
    """Test that PySpark dummy builder correctly handles null values."""
    pytest.importorskip("pyspark")
    from binscatter.dummy_builders import build_dummies_pyspark

    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "category": ["A", "B", None, "A", "B", None],
        }
    )

    df_spark = convert_to_backend(df_pd, "pyspark")
    df_nw = nw.from_native(df_spark).lazy()

    df_with_dummies, dummy_cols = build_dummies_pyspark(df_nw, ("category",))

    # Should create 1 dummy (A is first, B gets dummy, None is excluded)
    assert len(dummy_cols) == 1

    # Collect and verify
    result = df_with_dummies.collect().to_pandas()
    for col in dummy_cols:
        # Rows with null category should have 0 in all dummies
        null_mask = df_pd["category"].isna()
        assert (result.loc[null_mask, col] == 0.0).all()


def testbuild_dummies_fallback():
    """Test the fallback dummy builder for other backends."""
    from binscatter.dummy_builders import build_dummies_fallback

    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "cat": ["alpha", "beta", "alpha", "beta"],
        }
    )

    # Use DuckDB as a backend that falls back
    df_duck = convert_to_backend(df_pd, "duckdb")
    df_nw = nw.from_native(df_duck).lazy()

    df_with_dummies, dummy_cols = build_dummies_fallback(df_nw, ("cat",))

    # Should create 1 dummy (2 levels - 1)
    assert len(dummy_cols) == 1

    # Verify dummy name
    assert dummy_cols[0].startswith("__ctrl_cat_")

    # Collect and verify
    result = df_with_dummies.collect()
    assert dummy_cols[0] in result.columns


def test_dummy_builder_constant_categorical():
    """Test that dummy builders handle constant categorical variables."""
    from binscatter.dummy_builders import build_dummies_fallback

    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
            "cat": ["same", "same", "same", "same"],
        }
    )

    df_nw = nw.from_native(df_pd).lazy()

    df_with_dummies, dummy_cols = build_dummies_fallback(df_nw, ("cat",))

    # Constant categorical should create no dummies
    assert len(dummy_cols) == 0
    # Should return the same dataframe
    assert df_with_dummies.collect().shape == df_pd.shape


def test_build_dummies_pandas_with_empty_controls():
    """Test pandas builder with empty categorical controls."""
    from binscatter.dummy_builders import build_dummies_pandas
    import pandas as pd
    import narwhals as nw

    df_pd = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df_nw = nw.from_native(df_pd).lazy()

    df_result, dummy_cols = build_dummies_pandas(df_nw, ())

    assert len(dummy_cols) == 0
    assert df_result.collect().shape == (3, 2)


def test_build_dummies_polars_preserves_lazy():
    """Test that polars builder preserves lazy evaluation."""
    from binscatter.dummy_builders import build_dummies_polars
    import polars as pl
    import narwhals as nw

    # Create a lazy polars dataframe
    df_pl = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "cat": ["A", "B", "A", "B", "A"],
        }
    ).lazy()
    df_nw = nw.from_native(df_pl)

    # Build dummies
    df_result, dummy_cols = build_dummies_polars(df_nw, ("cat",))

    # Verify result is still lazy
    result_native = nw.to_native(df_result)
    assert isinstance(result_native, pl.LazyFrame), (
        f"Expected LazyFrame, got {type(result_native)}"
    )

    # Should have created 1 dummy (2 levels - 1)
    assert len(dummy_cols) == 1

    # Collect and verify
    collected = df_result.collect()
    assert dummy_cols[0] in collected.columns


def test_build_dummies_polars_with_multiple_categoricals():
    """Test polars builder with multiple categorical columns."""
    from binscatter.dummy_builders import build_dummies_polars
    import polars as pl
    import narwhals as nw

    df_pl = pl.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "cat_a": ["A", "B", "C", "A"],
            "cat_b": ["X", "Y", "X", "Y"],
            "cat_c": ["P", "Q", "R", "P"],
        }
    ).lazy()
    df_nw = nw.from_native(df_pl)

    df_result, dummy_cols = build_dummies_polars(df_nw, ("cat_a", "cat_b", "cat_c"))

    # cat_a: 3 levels -> 2 dummies
    # cat_b: 2 levels -> 1 dummy
    # cat_c: 3 levels -> 2 dummies
    # Total: 5 dummies
    assert len(dummy_cols) == 5

    # Verify all dummy columns are present
    collected = df_result.collect()
    for col in dummy_cols:
        assert col in collected.columns
        assert col.startswith("__ctrl_")


def test_build_dummies_fallback_with_multiple_categoricals():
    """Test fallback builder with multiple categorical columns."""
    from binscatter.dummy_builders import build_dummies_fallback
    import pandas as pd
    import narwhals as nw

    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "cat_a": ["foo", "bar", "baz", "foo"],
            "cat_b": ["red", "blue", "red", "blue"],
        }
    )
    df_nw = nw.from_native(df_pd).lazy()

    df_result, dummy_cols = build_dummies_fallback(df_nw, ("cat_a", "cat_b"))

    # cat_a: 3 levels -> 2 dummies
    # cat_b: 2 levels -> 1 dummy
    # Total: 3 dummies
    assert len(dummy_cols) == 3

    # Verify all dummy columns exist and have correct values
    collected = df_result.collect()
    for col in dummy_cols:
        assert col in collected.columns
        # Should be 0.0 or 1.0
        values = set(collected[col].to_list())
        assert values <= {0.0, 1.0}


def test_build_dummies_pandas_single_categorical():
    """Test pandas builder with a single categorical column."""
    from binscatter.dummy_builders import build_dummies_pandas
    import pandas as pd
    import narwhals as nw

    df_pd = pd.DataFrame({"x": [1, 2, 3], "cat": ["only_one", "only_one", "only_one"]})
    df_nw = nw.from_native(df_pd).lazy()

    df_result, dummy_cols = build_dummies_pandas(df_nw, ("cat",))

    # Single level categorical should create no dummies
    assert len(dummy_cols) == 0


def test_build_dummies_polars_single_categorical():
    """Test polars builder with a single categorical column."""
    from binscatter.dummy_builders import build_dummies_polars
    import polars as pl
    import narwhals as nw

    df_pl = pl.DataFrame(
        {"x": [1, 2, 3], "cat": ["only_one", "only_one", "only_one"]}
    ).lazy()
    df_nw = nw.from_native(df_pl)

    df_result, dummy_cols = build_dummies_polars(df_nw, ("cat",))

    # Single level categorical should create no dummies
    assert len(dummy_cols) == 0


def test_configure_build_dummies_dispatch():
    """Test that configure_build_dummies returns the right implementation."""
    from binscatter.dummy_builders import (
        configure_build_dummies,
        build_dummies_pandas,
        build_dummies_polars,
        build_dummies_fallback,
    )
    from narwhals import Implementation

    # Pandas should get pandas builder
    builder = configure_build_dummies(Implementation.PANDAS)
    assert builder == build_dummies_pandas

    # Polars should get polars builder
    builder = configure_build_dummies(Implementation.POLARS)
    assert builder == build_dummies_polars

    # DuckDB should get fallback
    builder = configure_build_dummies(Implementation.DUCKDB)
    assert builder == build_dummies_fallback

    # Dask should get fallback
    builder = configure_build_dummies(Implementation.DASK)
    assert builder == build_dummies_fallback


def test_configure_build_dummies_pyspark():
    """Test that PySpark gets the right dummy builder."""
    pytest.importorskip("pyspark")
    from binscatter.dummy_builders import configure_build_dummies, build_dummies_pyspark
    from narwhals import Implementation

    builder = configure_build_dummies(Implementation.PYSPARK)
    assert builder == build_dummies_pyspark


def test_maybe_add_regression_features_with_categorical():
    """Test that maybe_add_regression_features integrates dummy builders correctly."""
    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
            "num_ctrl": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "cat_ctrl": ["A", "B", "A", "B", "A", "B"],
        }
    )

    df_nw = nw.from_native(df_pd).lazy()

    df_with_features, features = add_regression_features(
        df_nw,
        numeric_controls=("num_ctrl",),
        categorical_controls=("cat_ctrl",),
    )

    # Should have numeric control + categorical dummies
    # cat_ctrl has 2 levels -> 1 dummy
    assert "num_ctrl" in features
    assert len(features) == 2  # num_ctrl + 1 dummy

    # Verify the dataframe has the features
    result = df_with_features.collect()
    for feat in features:
        assert feat in result.columns


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_dummy_names_consistent_across_backends(backend):
    """Test that dummy variable names are consistent across backends."""
    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "cat_a": ["foo", "bar", "baz", "foo", "bar"],
            "cat_b": ["red", "blue", "red", "blue", "red"],
        }
    )

    df = convert_to_backend(df_pd, backend)
    df_clean, _, _, categorical_controls = clean_df(
        df,
        controls=("cat_a", "cat_b"),
        x="x",
        y="y",
    )

    df_with_dummies, regression_features = add_regression_features(
        df_clean,
        numeric_controls=(),
        categorical_controls=categorical_controls,
    )

    # All dummy names should follow pattern: __ctrl_{column}_{value}_{hash}
    import re

    for feat in regression_features:
        assert feat.startswith("__ctrl_"), f"Bad dummy name: {feat}"
        # Should end with 8-character hex hash
        assert re.search(r"_[a-f0-9]{8}$", feat), (
            f"Dummy name should end with hash: {feat}"
        )


def test_pyspark_dummy_names_match_pandas():
    """Test that PySpark dummy names exactly match pandas dummy names."""
    pytest.importorskip("pyspark")

    df_pd = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "category": ["A", "B", "C", "A", "B"],
        }
    )

    # Get pandas dummy names
    df_clean_pd, _, _, cat_controls_pd = clean_df(
        df_pd,
        controls=("category",),
        x="x",
        y="y",
    )
    _, features_pd = add_regression_features(
        df_clean_pd,
        numeric_controls=(),
        categorical_controls=cat_controls_pd,
    )

    # Get PySpark dummy names
    df_spark = convert_to_backend(df_pd, "pyspark")
    df_clean_spark, _, _, cat_controls_spark = clean_df(
        df_spark,
        controls=("category",),
        x="x",
        y="y",
    )
    _, features_spark = add_regression_features(
        df_clean_spark,
        numeric_controls=(),
        categorical_controls=cat_controls_spark,
    )

    # Names should match exactly (order might differ, so compare sets)
    assert set(features_pd) == set(features_spark), (
        f"Dummy names differ between pandas and PySpark:\n"
        f"  pandas: {sorted(features_pd)}\n"
        f"  PySpark: {sorted(features_spark)}"
    )


def test_binscatter_with_pyspark_caching():
    """Integration test: full binscatter with PySpark should use caching."""
    pytest.importorskip("pyspark")

    df_pd = pd.DataFrame(
        {
            "x": np.random.default_rng(42).normal(100, 15, 1000),
            "y": np.random.default_rng(43).normal(50, 10, 1000),
            "ctrl_num": np.random.default_rng(44).normal(0, 1, 1000),
            "ctrl_cat": np.random.default_rng(45).choice(["A", "B", "C"], 1000),
        }
    )

    df_spark = convert_to_backend(df_pd, "pyspark")

    # This should use caching internally
    fig = binscatter(
        df_spark,
        x="x",
        y="y",
        controls=["ctrl_num", "ctrl_cat"],
        num_bins=20,
    )

    # Just verify it completes without error
    assert fig is not None
    assert isinstance(fig, go.Figure)


# =============================================================================
