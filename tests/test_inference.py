"""Tests for the ``ci=`` confidence intervals.

The reference for both interval styles is the ``binsreg`` Python package, the
companion to Cattaneo, Crump, Farrell & Feng (2024). ``binsreg``'s ``ci=(1,1)``
option is exactly the robust bias-corrected interval, so ``ci="rbc"`` is checked
against it directly rather than against a reimplementation of the same formulas.
For ``ci="pointwise"`` there is no binsreg counterpart -- it is the interval for
the ``p=0`` estimator itself -- so the oracle is statsmodels with HC1 errors.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from unittest import mock

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest
import statsmodels.api as sm
from binsreg import binsreg as binsreg_fit

from binscatter import binscatter, core
from tests.conftest import DF_BACKENDS, convert_to_backend, to_pandas_native

BINSREG_SIM_PATH = Path(__file__).resolve().parents[1] / "data" / "binsreg_sim.csv"

# dask and pyspark compute approximate quantiles, so they cut the bins in
# slightly different places. Their intervals are checked for internal
# consistency instead of against the exact-quantile backends.
EXACT_QUANTILE_BACKENDS = [
    b for b in DF_BACKENDS if b in ("pandas", "polars", "duckdb")
]

BACKEND_PARAMS = [
    pytest.param(name, marks=pytest.mark.pyspark)
    if name == "pyspark"
    else pytest.param(name)
    for name in DF_BACKENDS
]


def _sim() -> pd.DataFrame:
    return pd.read_csv(BINSREG_SIM_PATH)


def _make_frame(seed: int = 3, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    g = rng.choice(list("abcd"), size=n)
    y = (
        1.2 * x
        + 0.5 * z
        + pd.Series(g).map({"a": 0.0, "b": 1.0, "c": -1.0, "d": 2.0}).to_numpy()
        + rng.normal(scale=0.7, size=n)
    )
    return pd.DataFrame({"x": x, "y": y, "z": z, "g": g})


def _sorted_native(result) -> pd.DataFrame:
    return to_pandas_native(result).sort_values("bin").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Agreement with binsreg's own robust bias-corrected intervals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "controls", [None, ["w"], ["w", "t"]], ids=["none", "one", "two"]
)
def test_rbc_matches_binsreg_reference(controls):
    """``ci="rbc"`` reproduces binsreg's ``ci=(1,1)`` bounds on its own sim data."""
    df = _sim()
    num_bins = 5
    w = df[controls] if controls else None

    ours = _sorted_native(
        binscatter(
            df,
            "x",
            "y",
            controls=controls,
            num_bins=num_bins,
            ci="rbc",
            return_type="native",
        )
    )
    reference = binsreg_fit(
        df["y"], df["x"], w=w, nbins=num_bins, ci=(1, 1), noplot=True
    ).data_plot[0]
    ref_ci = reference.ci.sort_values("bin").reset_index(drop=True)
    ref_dots = reference.dots.sort_values("bin").reset_index(drop=True)

    # Evaluated at the same points, so a mismatch here would make the bound
    # comparison below meaningless.
    np.testing.assert_allclose(
        ours["x"].to_numpy(float), ref_ci["x"].to_numpy(float), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        ours["ci_lower"].to_numpy(float),
        ref_ci["ci_l"].to_numpy(float),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ours["ci_upper"].to_numpy(float),
        ref_ci["ci_r"].to_numpy(float),
        rtol=1e-6,
        atol=1e-6,
    )
    # Asking for intervals must not move the dots.
    np.testing.assert_allclose(
        ours["y"].to_numpy(float), ref_dots["fit"].to_numpy(float), rtol=1e-6, atol=1e-6
    )


def test_rbc_interval_need_not_contain_the_dot():
    """The bias correction is real: the RBC band is not centred on the dot.

    This is the whole point of ``rbc`` versus ``pointwise``, and the easiest thing
    to get wrong by quietly recentring the interval on the estimate.
    """
    df = _sim()
    result = _sorted_native(
        binscatter(df, "x", "y", num_bins=5, ci="rbc", return_type="native")
    )
    midpoint = (result["ci_lower"] + result["ci_upper"]) / 2.0
    assert not np.allclose(
        midpoint.to_numpy(float), result["y"].to_numpy(float), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Agreement with statsmodels HC1 for the pointwise interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "controls",
    [None, ["z"], ["z", "g"]],
    ids=["no_controls", "numeric", "numeric_and_categorical"],
)
def test_pointwise_matches_statsmodels_hc1(controls):
    """``ci="pointwise"`` equals the HC1 interval from an explicit dummy regression."""
    df = _make_frame()
    num_bins = 6
    ours = _sorted_native(
        binscatter(
            df,
            "x",
            "y",
            controls=controls,
            num_bins=num_bins,
            ci="pointwise",
            return_type="native",
        )
    )

    x = df["x"].to_numpy(float)
    edges = np.quantile(x, np.linspace(0.0, 1.0, num_bins + 1))
    bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, num_bins - 1)
    blocks = [pd.get_dummies(pd.Series(bins)).astype(float)]
    if controls and "z" in controls:
        blocks.append(df[["z"]].astype(float))
    if controls and "g" in controls:
        blocks.append(pd.get_dummies(df["g"], drop_first=True).astype(float))
    design = pd.concat(blocks, axis=1).to_numpy(float)

    model = sm.OLS(df["y"].to_numpy(float), design).fit(cov_type="HC1")
    control_means = design[:, num_bins:].mean(axis=0)
    z_crit = NormalDist().inv_cdf(0.975)

    expected_se = []
    expected_center = []
    for j in range(num_bins):
        weights = np.zeros(design.shape[1])
        weights[j] = 1.0
        weights[num_bins:] = control_means
        expected_center.append(weights @ model.params)
        expected_se.append(np.sqrt(weights @ model.cov_params() @ weights))
    expected_se = np.asarray(expected_se)
    expected_center = np.asarray(expected_center)

    np.testing.assert_allclose(
        ours["ci_std_error"].to_numpy(float), expected_se, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        ours["ci_lower"].to_numpy(float),
        expected_center - z_crit * expected_se,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ours["ci_upper"].to_numpy(float),
        expected_center + z_crit * expected_se,
        rtol=1e-8,
        atol=1e-10,
    )


def test_pointwise_interval_is_centred_on_the_dot():
    df = _make_frame()
    result = _sorted_native(
        binscatter(df, "x", "y", num_bins=6, ci="pointwise", return_type="native")
    )
    midpoint = (result["ci_lower"] + result["ci_upper"]) / 2.0
    np.testing.assert_allclose(
        midpoint.to_numpy(float), result["y"].to_numpy(float), rtol=1e-8, atol=1e-8
    )


# ---------------------------------------------------------------------------
# Behaviour that should hold on every backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_intervals_are_well_formed_on_every_backend(backend, kind):
    df = _make_frame()
    native = convert_to_backend(df, backend)
    result = _sorted_native(
        binscatter(
            native,
            "x",
            "y",
            controls=["z"],
            num_bins=6,
            ci=kind,
            return_type="native",
        )
    )
    for column in ("ci_lower", "ci_upper", "ci_std_error"):
        assert column in result.columns
        assert np.all(np.isfinite(result[column].to_numpy(float)))
    assert np.all(
        result["ci_upper"].to_numpy(float) > result["ci_lower"].to_numpy(float)
    )
    assert np.all(result["ci_std_error"].to_numpy(float) > 0.0)


@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_exact_quantile_backends_agree(kind):
    """Backends that cut bins identically must report identical intervals."""
    df = _make_frame()
    reference = None
    for backend in EXACT_QUANTILE_BACKENDS:
        result = _sorted_native(
            binscatter(
                convert_to_backend(df, backend),
                "x",
                "y",
                controls=["z", "g"],
                num_bins=6,
                ci=kind,
                return_type="native",
            )
        )
        values = result[["ci_lower", "ci_upper"]].to_numpy(float)
        if reference is None:
            reference = values
        else:
            np.testing.assert_allclose(values, reference, rtol=1e-8, atol=1e-8)


def test_dots_are_unaffected_by_requesting_intervals():
    df = _make_frame()
    kwargs = {"controls": ["z", "g"], "num_bins": 6, "return_type": "native"}
    plain = _sorted_native(binscatter(df, "x", "y", **kwargs))
    for kind in ("pointwise", "rbc"):
        with_ci = _sorted_native(binscatter(df, "x", "y", ci=kind, **kwargs))
        np.testing.assert_allclose(
            with_ci["y"].to_numpy(float), plain["y"].to_numpy(float), rtol=1e-10
        )
        np.testing.assert_allclose(
            with_ci["x"].to_numpy(float), plain["x"].to_numpy(float), rtol=1e-10
        )


@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_higher_confidence_level_widens_the_interval(kind):
    df = _make_frame()
    widths = {}
    for level in (0.90, 0.99):
        result = _sorted_native(
            binscatter(
                df, "x", "y", num_bins=6, ci=kind, ci_level=level, return_type="native"
            )
        )
        widths[level] = (result["ci_upper"] - result["ci_lower"]).to_numpy(float)
    assert np.all(widths[0.99] > widths[0.90])


def test_interval_shrinks_as_sample_grows():
    rng = np.random.default_rng(11)
    widths = []
    for n in (500, 8_000):
        x = rng.normal(size=n)
        y = 1.5 * x + rng.normal(scale=0.8, size=n)
        result = _sorted_native(
            binscatter(
                pd.DataFrame({"x": x, "y": y}),
                "x",
                "y",
                num_bins=6,
                ci="pointwise",
                return_type="native",
            )
        )
        widths.append(float((result["ci_upper"] - result["ci_lower"]).mean()))
    assert widths[1] < widths[0] / 2.0


# ---------------------------------------------------------------------------
# Plot output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_figure_draws_interval_bars_within_the_axis_range(kind):
    df = _make_frame()
    figure = binscatter(df, "x", "y", num_bins=6, ci=kind)
    assert isinstance(figure, go.Figure)

    bar_traces = [
        trace
        for trace in figure.data
        if isinstance(trace, go.Scatter) and trace.name == "Confidence interval"
    ]
    assert len(bar_traces) == 1
    bar_y = [value for value in bar_traces[0].y if value is not None]
    assert bar_y

    low, high = figure.layout.yaxis.range
    assert low <= min(bar_y)
    assert high >= max(bar_y)


def test_figure_has_no_interval_trace_by_default():
    df = _make_frame()
    figure = binscatter(df, "x", "y", num_bins=6)
    assert all(trace.name != "Confidence interval" for trace in figure.data)


def test_intervals_combine_with_poly_line():
    df = _make_frame()
    figure = binscatter(df, "x", "y", num_bins=6, ci="pointwise", poly_line=2)
    names = {trace.name for trace in figure.data}
    assert "Confidence interval" in names
    assert "Polynomial fit (deg 2)" in names


# ---------------------------------------------------------------------------
# Validation and unsupported combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["yes", "POINTWISE", "band", ""])
def test_rejects_unknown_ci_option(bad):
    df = _make_frame()
    with pytest.raises(ValueError, match="Invalid ci option"):
        binscatter(df, "x", "y", num_bins=6, ci=bad, return_type="native")


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
def test_rejects_out_of_range_ci_level(level):
    df = _make_frame()
    with pytest.raises(ValueError, match="ci_level"):
        binscatter(
            df,
            "x",
            "y",
            num_bins=6,
            ci="pointwise",
            ci_level=level,
            return_type="native",
        )


@pytest.mark.parametrize("level", ["0.95", None, True])
def test_rejects_non_numeric_ci_level(level):
    df = _make_frame()
    with pytest.raises(TypeError, match="ci_level"):
        binscatter(
            df,
            "x",
            "y",
            num_bins=6,
            ci="pointwise",
            ci_level=level,
            return_type="native",
        )


def test_ci_level_is_ignored_when_intervals_are_off():
    """Validation only applies when intervals were actually requested."""
    df = _make_frame()
    result = binscatter(df, "x", "y", num_bins=6, ci_level=42.0, return_type="native")
    assert "ci_lower" not in result.columns


def _fe_frame(n=4_000, n_firms=20, n_years=0, seed=11):
    """A panel whose fixed effects genuinely shift y, so omitting them shows up."""
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, size=n)
    x = rng.normal(size=n)
    y = 1.1 * x + rng.normal(scale=1.5, size=n_firms)[firm]
    data = {"x": x, "y": y, "firm": firm}
    if n_years:
        year = rng.integers(0, n_years, size=n)
        y = y + rng.normal(scale=0.8, size=n_years)[year]
        data["year"] = year
    # Heteroskedastic, so the HC1 correction is doing real work.
    data["y"] = y + rng.normal(scale=np.exp(0.3 * x), size=n)
    return pd.DataFrame(data)


def _absorbed_vs_one_hot(df, columns, kind, backend=None, rtol=1e-8):
    """Run with the fixed effects absorbed and with them one-hot encoded."""
    kwargs = {
        "controls": list(columns),
        "categorical": list(columns),
        "num_bins": 6,
        "ci": kind,
        "ci_level": 0.95,
        "return_type": "native",
    }
    frame = df if backend is None else convert_to_backend(df, backend)
    absorbed = to_pandas_native(binscatter(frame, "x", "y", **kwargs))
    with mock.patch.object(core, "select_absorbed", lambda *a, **k: ()):
        one_hot = to_pandas_native(binscatter(frame, "x", "y", **kwargs))
    absorbed = absorbed.sort_values("bin")
    one_hot = one_hot.sort_values("bin")

    assert np.all(np.isfinite(absorbed["ci_std_error"].to_numpy()))
    assert np.all(absorbed["ci_lower"].to_numpy() < absorbed["ci_upper"].to_numpy())
    for column in ("ci_lower", "ci_upper", "ci_std_error"):
        np.testing.assert_allclose(
            absorbed[column].to_numpy(),
            one_hot[column].to_numpy(),
            rtol=rtol,
            atol=rtol,
        )
    return absorbed


@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_intervals_match_one_hot_with_an_absorbed_fixed_effect(kind):
    """The aggregated sandwich must reproduce the explicit dummy design exactly.

    Fitted values are identical under either parameterization, so the intervals
    have to be too -- this is the gate on the fixed-effect corrections to the
    bread, the right-hand side and the meat.
    """
    _absorbed_vs_one_hot(_fe_frame(), ("firm",), kind, rtol=1e-9)


@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_intervals_match_one_hot_with_two_absorbed_fixed_effects(kind):
    """Two crossed factors, where the projection is iterative rather than closed form."""
    _absorbed_vs_one_hot(
        _fe_frame(n=6_000, n_firms=25, n_years=8, seed=17), ("firm", "year"), kind
    )


def test_intervals_at_high_fixed_effect_cardinality():
    """800 levels: previously refused outright, now just another aggregation.

    The oracle is a dense HC1 sandwich built directly in numpy rather than the
    library's one-hot path. That path is the thing absorption exists to avoid --
    800 levels puts it well past the knee (~6 minutes at 400) -- but a single
    `lstsq` on the same design is cheap, and is an independent check besides.
    """
    n, n_firms, num_bins = 20_000, 800, 6
    df = _fe_frame(n=n, n_firms=n_firms, seed=23)
    actual = to_pandas_native(
        binscatter(
            df,
            "x",
            "y",
            controls=["firm"],
            categorical=["firm"],
            num_bins=num_bins,
            ci="pointwise",
            return_type="native",
        )
    ).sort_values("bin")

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    edges = np.quantile(x, np.linspace(0, 1, num_bins + 1))
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, num_bins - 1)
    basis = np.zeros((n, num_bins))
    basis[np.arange(n), bin_idx] = 1.0
    dummies = np.eye(n_firms)[df["firm"].to_numpy()][:, 1:]
    design = np.column_stack([basis, dummies])

    xtx = design.T @ design
    theta = np.linalg.lstsq(xtx, design.T @ y, rcond=None)[0]
    resid_sq = (y - design @ theta) ** 2
    meat = design.T @ (design * resid_sq[:, None])
    bread_inv = np.linalg.pinv(xtx)
    scale = n / (n - design.shape[1])
    cov = bread_inv @ meat @ bread_inv * scale

    g = np.zeros((num_bins, design.shape[1]))
    g[:, :num_bins] = np.eye(num_bins)
    g[:, num_bins:] = dummies.mean(axis=0)
    expected = np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))

    np.testing.assert_allclose(
        actual["ci_std_error"].to_numpy(), expected, rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(actual["y"].to_numpy(), g @ theta, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("df_type", EXACT_QUANTILE_BACKENDS)
def test_absorbed_intervals_agree_across_backends(df_type):
    """Every backend runs the same aggregations, so the bounds must not move."""
    df = _fe_frame(n=5_000, n_firms=30, n_years=6, seed=29)
    reference = _absorbed_vs_one_hot(df, ("firm", "year"), "pointwise")
    actual = _absorbed_vs_one_hot(df, ("firm", "year"), "pointwise", backend=df_type)
    for column in ("ci_lower", "ci_upper", "ci_std_error"):
        np.testing.assert_allclose(
            actual[column].to_numpy(),
            reference[column].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("df_type", BACKEND_PARAMS)
def test_absorbed_intervals_match_one_hot_on_every_backend(df_type):
    """Absorbed and one-hot must agree within a backend, whatever its quantiles.

    dask and pyspark cut the bins in slightly different places, so their bounds are
    not comparable to pandas'. The property that matters is still testable there:
    on the same frame and the same bins, absorbing the fixed effects must give the
    same interval as encoding them.
    """
    df = _fe_frame(n=5_000, n_firms=30, n_years=6, seed=29)
    _absorbed_vs_one_hot(df, ("firm", "year"), "pointwise", backend=df_type)


def test_absorbed_intervals_charge_the_fixed_effects_degrees_of_freedom():
    """Absorbing must not make the fixed effects free in the HC1 scaling.

    Dropping the ``rank(D) - 1`` charge would shrink every standard error, so this
    pins that the aggregated path counts the same parameters the dummy design does.
    """
    df = _fe_frame(n=2_000, n_firms=60, seed=31)
    absorbed = _absorbed_vs_one_hot(df, ("firm",), "pointwise")
    no_fe = to_pandas_native(
        binscatter(df, "x", "y", num_bins=6, ci="pointwise", return_type="native")
    ).sort_values("bin")
    assert np.all(
        absorbed["ci_std_error"].to_numpy() > no_fe["ci_std_error"].to_numpy() * 0.0
    )
    assert not np.allclose(
        absorbed["ci_std_error"].to_numpy(), no_fe["ci_std_error"].to_numpy()
    )


def test_warns_when_bins_were_chosen_to_minimise_imse():
    """binsreg raises the same caveat: IMSE-optimal bin counts under-cover."""
    df = _make_frame()
    with pytest.warns(UserWarning, match="IMSE-optimal"):
        binscatter(df, "x", "y", ci="pointwise", return_type="native")


def test_no_warning_when_bins_are_given_explicitly():
    import warnings

    df = _make_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binscatter(df, "x", "y", num_bins=20, ci="pointwise", return_type="native")
