from typing import Iterable

import polars as pl
import numpy as np
from binscatter.core import binscatter, prep, partial_out_controls
import plotly.graph_objs as go
import duckdb
import pytest
import pandas as pd
import dask.dataframe as dd
import statsmodels.api as sm

try:  # pragma: no cover - optional dependency
    from pyspark.sql import SparkSession
    import pyspark
except ImportError:  # pragma: no cover - optional dependency
    SparkSession = None
    pyspark = None

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


fixt_dat = [
    ("df_good", False),
    ("df_x_num", False),
    ("df_missing_column", True),
    ("df_nulls", True),
    ("df_duplicates", True),
]

BASE_DF_TYPES = ["pandas", "polars", "duckdb", "dask"]
HAS_PYSPARK = SparkSession is not None

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
    match df_type:
        case "pandas":
            return df
        case "polars":
            return pl.from_pandas(df)
        case "duckdb":
            return duckdb.from_df(df)
        case "pyspark":
            if SparkSession is None:
                pytest.skip("PySpark not available")
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


def _to_pandas(df_native):
    if isinstance(df_native, pd.DataFrame):
        return df_native
    if hasattr(df_native, "to_pandas"):
        return df_native.to_pandas()
    if hasattr(df_native, "df"):
        return df_native.df()
    if isinstance(df_native, dd.DataFrame):
        return df_native.compute()
    if pyspark is not None and isinstance(df_native, pyspark.sql.DataFrame):
        return df_native.toPandas()
    raise TypeError(f"Unsupported dataframe type: {type(df_native)}")


def _collect_lazyframe_to_pandas(frame):
    """Helper to collect a narwhals LazyFrame into a pandas DataFrame."""
    return _to_pandas(frame.collect().to_native())


def test_binscatter_controls_matches_reference():
    rng = np.random.default_rng(123)
    n = 2000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.5 * x + 2.75 * z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x0": x, "y0": y, "z": z})
    num_bins = 15

    expected_x, expected_y = _manual_binscatter_with_controls(df, num_bins)
    result = binscatter(
        df,
        "x0",
        "y0",
        controls=["z"],
        num_bins=num_bins,
        return_type="native",
    )
    result_pd = _to_pandas(result).sort_values("bin").reset_index(drop=True)
    if df_type in ("dask", "pyspark"):
        rtol_y, atol_y = 5e-3, 2e-1
    else:
        rtol_y = atol_y = 1e-6
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=rtol_y, atol=atol_y
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
    result_pd = _to_pandas(result).sort_values("bin").reset_index(drop=True)
    if df_type in ("dask", "pyspark"):
        rtol_x, atol_x = 1e-3, 1e-1
        rtol_y, atol_y = 5e-3, 2e-1
    else:
        rtol_x = atol_x = 1e-6
        rtol_y = atol_y = 1e-6
    np.testing.assert_allclose(
        result_pd["x0"].to_numpy(), expected_x, rtol=rtol_x, atol=atol_x
    )
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=rtol_y, atol=atol_y
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
    result_pd = _to_pandas(result).sort_values("bin").reset_index(drop=True)
    if df_type in ("dask", "pyspark"):
        rtol_y, atol_y = 5e-3, 2e-1
    else:
        rtol_y = atol_y = 1e-6
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=rtol_y, atol=atol_y
    )


def test_binscatter_categorical_controls_only():
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
    result = binscatter(
        df,
        "x0",
        "y0",
        controls=["cat"],
        num_bins=num_bins,
        return_type="native",
    )
    result_pd = _to_pandas(result).sort_values("bin").reset_index(drop=True)
    np.testing.assert_allclose(
        result_pd["x0"].to_numpy(), expected_x, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        result_pd["y0"].to_numpy(), expected_y, rtol=1e-6, atol=1e-6
    )


def test_binscatter_controls_collapsed_bins_error():
    df = pd.DataFrame(
        {
            "x0": np.ones(50),
            "y0": np.linspace(0.0, 1.0, num=50),
            "z": np.linspace(-1.0, 1.0, num=50),
        }
    )
    with pytest.raises(ValueError, match="Quantiles are not unique"):
        binscatter(df, "x0", "y0", controls=["z"], num_bins=5, return_type="native")


def test_partial_out_controls_matches_statsmodels():
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
    df_prepped, profile = prep(
        df,
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
    np.testing.assert_allclose(
        result[profile.x_name].to_numpy(),
        bin_means.to_numpy(),
        rtol=1e-6,
        atol=1e-6,
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
    design_matrix = np.column_stack([bin_dummies.to_numpy(), control_matrix])
    mean_controls = np.concatenate(control_means)

    model = sm.OLS(df_with_bins[profile.y_name].to_numpy(), design_matrix).fit()
    theta = model.params
    beta = theta[: profile.num_bins]
    gamma = theta[profile.num_bins :]
    fitted = beta + (mean_controls @ gamma if gamma.size else 0.0)

    np.testing.assert_allclose(
        result[profile.y_name].to_numpy(),
        fitted,
        rtol=1e-6,
        atol=1e-6,
    )

    beta_ref = beta - beta[0]
    beta_actual = coeffs["beta"] - coeffs["beta"][0]
    np.testing.assert_allclose(beta_actual, beta_ref, rtol=1e-6, atol=1e-6)
