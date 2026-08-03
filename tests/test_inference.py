"""Tests for the ``ci=`` confidence intervals.

The reference for both interval styles is the ``binsreg`` Python package, the
companion to Cattaneo, Crump, Farrell & Feng (2024). Its ``ci=(p, s)`` option
names the fit the interval is built from, and both of our styles have an exact
counterpart there: ``ci="pointwise"`` is ``ci=(0, 0)``, the interval for the
piecewise-constant estimator the dots show, and ``ci="rbc"`` is ``ci=(1, 1)``,
the robust bias-corrected interval built from the next-order fit. Both are
checked against binsreg directly rather than against a reimplementation of the
same formulas.

``ci="pointwise"`` is additionally checked against statsmodels with HC1 errors,
which is an oracle of a different kind: it spells the estimator out as an
explicit dummy regression, so it pins down the standard errors themselves rather
than only the bounds, and it would not move with us if binsreg and this package
ever drifted together.
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest
import statsmodels.api as sm
from binsreg import binsreg as binsreg_fit

from binscatter import binscatter
from binscatter.fixed_effects import ABSORB_MIN_LEVELS
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


def _make_multi_control_frame(seed: int = 11, n: int = 900) -> pd.DataFrame:
    """Synthetic data with a numeric control and two categorical controls.

    ``g`` has four levels, ``h`` has three, and both are well under the
    fixed-effect absorption threshold, so ``binscatter`` reaches them as
    ordinary dummy-encoded controls rather than absorbing them.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    g = rng.choice(list("abcd"), size=n)
    h = rng.choice(["lo", "mid", "hi"], size=n)
    g_effect = pd.Series(g).map({"a": 0.0, "b": 1.0, "c": -1.0, "d": 2.0}).to_numpy()
    h_effect = pd.Series(h).map({"lo": -0.5, "mid": 0.3, "hi": 1.1}).to_numpy()
    y = 1.2 * x + 0.5 * z + g_effect + h_effect + rng.normal(scale=0.7, size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z, "g": g, "h": h})


#: binsreg names an interval by the fit it is built from: ``(p, s)`` is a degree-p
#: spline with s continuous derivatives on the same bins. ``(0, 0)`` is the
#: piecewise-constant fit the dots show, ``(1, 1)`` the next-order fit the bias
#: correction uses.
BINSREG_CI_SPEC = {"pointwise": (0, 0), "rbc": (1, 1)}

CI_KINDS = list(BINSREG_CI_SPEC)


def _binsreg_ci(
    df: pd.DataFrame,
    kind: str,
    num_bins: int,
    *,
    w=None,
    level: float | None = None,
    x: str = "x",
    y: str = "y",
    **binsreg_kwargs,
) -> pd.DataFrame:
    """binsreg's interval frame for ``kind``, in bin order.

    ``level`` is this package's convention -- a probability -- and is converted to
    binsreg's percentage on the way in.
    """
    if level is not None:
        binsreg_kwargs["level"] = level * 100.0
    reference = binsreg_fit(
        df[y],
        df[x],
        w=w,
        nbins=num_bins,
        ci=BINSREG_CI_SPEC[kind],
        noplot=True,
        **binsreg_kwargs,
    ).data_plot[0]
    return reference.ci.sort_values("bin").reset_index(drop=True)


def _assert_matches_binsreg(ours: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Same evaluation points, same bounds.

    The x check comes first on purpose: intervals evaluated at different points
    would make the bound comparison meaningless rather than merely wrong.
    """
    assert len(ours) == len(reference)
    np.testing.assert_allclose(
        ours["x"].to_numpy(float), reference["x"].to_numpy(float), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        ours["ci_lower"].to_numpy(float),
        reference["ci_l"].to_numpy(float),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ours["ci_upper"].to_numpy(float),
        reference["ci_r"].to_numpy(float),
        rtol=1e-6,
        atol=1e-6,
    )


def _binsreg_w(df: pd.DataFrame, controls: list[str]) -> np.ndarray:
    """Build binsreg's ``w`` matrix, dummy-encoding categoricals as binscatter does.

    binsreg has no notion of a categorical control -- it only accepts numeric
    columns in ``w`` -- so this reproduces binscatter's own encoding
    (``drop_first=True``, alphabetically sorted levels) by hand.
    """
    blocks = []
    for name in controls:
        column = df[name]
        if pd.api.types.is_numeric_dtype(column):
            blocks.append(column.to_numpy(dtype=float).reshape(-1, 1))
        else:
            blocks.append(pd.get_dummies(column, drop_first=True).to_numpy(dtype=float))
    return np.column_stack(blocks)


# ---------------------------------------------------------------------------
# Agreement with binsreg's own robust bias-corrected intervals
# ---------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("kind", CI_KINDS)
@pytest.mark.parametrize(
    "controls", [None, ["w"], ["w", "t"]], ids=["none", "one", "two"]
)
def test_matches_binsreg_reference(controls, kind):
    """Both styles reproduce binsreg's own bounds on its own sim data."""
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
            ci=kind,
            return_type="native",
        )
    )
    reference = binsreg_fit(
        df["y"],
        df["x"],
        w=w,
        nbins=num_bins,
        ci=BINSREG_CI_SPEC[kind],
        noplot=True,
    ).data_plot[0]
    _assert_matches_binsreg(
        ours, reference.ci.sort_values("bin").reset_index(drop=True)
    )
    # Asking for intervals must not move the dots.
    ref_dots = reference.dots.sort_values("bin").reset_index(drop=True)
    np.testing.assert_allclose(
        ours["y"].to_numpy(float), ref_dots["fit"].to_numpy(float), rtol=1e-6, atol=1e-6
    )


@pytest.mark.quick
@pytest.mark.parametrize("kind", CI_KINDS)
@pytest.mark.parametrize(
    "controls",
    [["g"], ["z", "g"], ["g", "h"], ["z", "g", "h"]],
    ids=[
        "one_categorical",
        "numeric_and_categorical",
        "two_categoricals",
        "numeric_and_two_categoricals",
    ],
)
def test_matches_binsreg_reference_with_categorical_controls(controls, kind):
    """Both styles agree with binsreg when controls include several categoricals.

    ``test_matches_binsreg_reference`` above only exercises numeric controls, up
    to two of them, from the binsreg simulation data. Categorical controls take a
    different path internally (dummy-encoded rather than passed through as-is), so
    they get their own comparison here, including combinations with more than one
    categorical and a mix of numeric and categorical.
    """
    df = _make_multi_control_frame()
    num_bins = 6
    w = _binsreg_w(df, controls)

    ours = _sorted_native(
        binscatter(
            df,
            "x",
            "y",
            controls=controls,
            num_bins=num_bins,
            ci=kind,
            return_type="native",
        )
    )
    reference = binsreg_fit(
        df["y"],
        df["x"],
        w=w,
        nbins=num_bins,
        ci=BINSREG_CI_SPEC[kind],
        noplot=True,
    ).data_plot[0]
    _assert_matches_binsreg(
        ours, reference.ci.sort_values("bin").reset_index(drop=True)
    )
    ref_dots = reference.dots.sort_values("bin").reset_index(drop=True)
    np.testing.assert_allclose(
        ours["y"].to_numpy(float), ref_dots["fit"].to_numpy(float), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("kind", CI_KINDS)
def test_std_error_matches_the_one_implied_by_binsreg(kind):
    """The reported ``ci_std_error`` is the one binsreg's own bounds are built from.

    The bound comparisons would still pass if the standard error column were
    scaled or stale, since nothing else reads it. binsreg reports no standard
    error directly, so it is recovered from the symmetric bounds.
    """
    df = _sim()
    num_bins = 5
    ours = _sorted_native(
        binscatter(
            df,
            "x",
            "y",
            controls=["w"],
            num_bins=num_bins,
            ci=kind,
            return_type="native",
        )
    )
    reference = _binsreg_ci(df, kind, num_bins, w=df[["w"]])
    z_crit = NormalDist().inv_cdf(0.975)
    implied = (
        reference["ci_r"].to_numpy(float) - reference["ci_l"].to_numpy(float)
    ) / (2.0 * z_crit)
    np.testing.assert_allclose(
        ours["ci_std_error"].to_numpy(float), implied, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize("level", [0.80, 0.90, 0.99])
@pytest.mark.parametrize("kind", CI_KINDS)
def test_non_default_ci_level_matches_binsreg(kind, level):
    """``ci_level`` is checked against binsreg, not merely for monotonicity.

    ``test_higher_confidence_level_widens_the_interval`` only pins the ordering,
    which any monotone function of the level would satisfy. This pins the actual
    critical value at levels either side of the default.
    """
    df = _sim()
    num_bins = 5
    ours = _sorted_native(
        binscatter(
            df,
            "x",
            "y",
            controls=["w"],
            num_bins=num_bins,
            ci=kind,
            ci_level=level,
            return_type="native",
        )
    )
    _assert_matches_binsreg(
        ours, _binsreg_ci(df, kind, num_bins, w=df[["w"]], level=level)
    )


@pytest.mark.parametrize("num_bins", [2, 3, 20, 40])
@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_across_bin_counts(kind, num_bins):
    """Agreement holds at the extremes of the bin count, not just around five.

    Two bins is the smallest legal count and leaves the spline basis with a single
    interior knot; forty spreads the same sample thin enough that per-bin counts
    get small. Both are places where an off-by-one in the basis assembly would
    show up and a mid-sized bin count would not.
    """
    df = _sim()
    ours = _sorted_native(
        binscatter(df, "x", "y", num_bins=num_bins, ci=kind, return_type="native")
    )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, num_bins))


@pytest.mark.parametrize("method", ["dpi", "rot", "rule-of-thumb"])
@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_when_the_bin_count_was_chosen_automatically(kind, method):
    """Intervals are right on top of an automatically selected bin count too.

    The bin selectors have their own binsreg comparisons in
    ``test_binscatter.py``, but nothing checked that the interval machinery is
    handed the selected count intact. binsreg is given the count we chose, so this
    is a statement about the intervals and not a second test of the selector.
    """
    df = _sim()
    # Only ``pointwise`` warns at an IMSE-optimal bin count. ``rbc`` corrects the
    # bias that makes such a count a problem, so it is valid there and stays quiet.
    expect_warning = (
        pytest.warns(UserWarning, match="IMSE-optimal")
        if kind == "pointwise"
        else contextlib.nullcontext()
    )
    with expect_warning:
        ours = _sorted_native(
            binscatter(df, "x", "y", num_bins=method, ci=kind, return_type="native")
        )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, len(ours)))


@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_with_many_numeric_controls(kind):
    """Six controls, where the sim data tops out at two.

    The control block of the sandwich is assembled index by index, so a mistake
    there can stay hidden while ``k`` is small enough for the block to be almost
    square with the basis.
    """
    rng = np.random.default_rng(4)
    n, k = 1_500, 6
    controls = rng.normal(size=(n, k))
    x = rng.normal(size=n)
    y = 1.3 * x + controls @ (np.arange(1, k + 1) * 0.2) + rng.normal(size=n)
    names = [f"c{i}" for i in range(k)]
    df = pd.DataFrame(
        {"x": x, "y": y, **{n_: controls[:, i] for i, n_ in enumerate(names)}}
    )

    ours = _sorted_native(
        binscatter(
            df, "x", "y", controls=names, num_bins=6, ci=kind, return_type="native"
        )
    )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, 6, w=df[names]))


@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_under_strong_heteroskedasticity(kind):
    """The errors are HC1, and this is the data that tells HC1 apart from OLS.

    Every other comparison here runs on roughly homoskedastic data, where a
    classical variance estimate would land close enough to pass. Here the noise
    scale grows with ``|x|``, so the two disagree substantially.
    """
    rng = np.random.default_rng(17)
    n = 1_500
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.2 + np.abs(x), size=n)
    df = pd.DataFrame({"x": x, "y": y})

    ours = _sorted_native(
        binscatter(df, "x", "y", num_bins=8, ci=kind, return_type="native")
    )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, 8))


@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_when_x_has_mass_points(kind):
    """Heavy ties in ``x`` put many rows on a bin edge.

    ``binsreg`` is run with ``masspoints="off"`` because its mass-point handling
    changes the binning itself, which this package does not do; with that off both
    cut the same bins and the intervals are comparable.
    """
    rng = np.random.default_rng(0)
    n = 2_000
    x = rng.integers(0, 15, size=n).astype(float)
    y = 0.5 * x + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y})

    ours = _sorted_native(
        binscatter(df, "x", "y", num_bins=5, ci=kind, return_type="native")
    )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, 5, masspoints="off"))


@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_after_missing_rows_are_dropped(kind):
    """Rows with a null in ``x``, ``y`` or a control drop out of the variance too.

    A null that survived into the second pass would poison a whole bin's residual
    sum, so the reference is binsreg run on the frame with those rows already
    removed.
    """
    rng = np.random.default_rng(2)
    n = 1_000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = x + 0.5 * z + rng.normal(size=n)
    x[:20] = np.nan
    y[30:40] = np.nan
    z[50:55] = np.nan
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    complete = df.dropna().reset_index(drop=True)

    ours = _sorted_native(
        binscatter(
            df, "x", "y", controls=["z"], num_bins=5, ci=kind, return_type="native"
        )
    )
    _assert_matches_binsreg(ours, _binsreg_ci(complete, kind, 5, w=complete[["z"]]))


@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_with_an_integer_coded_categorical_control(kind):
    """``categorical=`` must reach the interval, not just the dots.

    An integer-coded identifier left alone enters as one linear term; declared
    categorical it becomes a block of dummies, which changes the design the
    sandwich is built from. binsreg is given that dummy block explicitly.
    """
    rng = np.random.default_rng(7)
    n = 1_200
    group = rng.integers(0, 5, size=n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.2 * x + 0.4 * z + group * 0.7 + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z, "grp": group})
    w = np.column_stack(
        [z.reshape(-1, 1), pd.get_dummies(df["grp"], drop_first=True).to_numpy(float)]
    )

    ours = _sorted_native(
        binscatter(
            df,
            "x",
            "y",
            controls=["z", "grp"],
            categorical=["grp"],
            num_bins=6,
            ci=kind,
            return_type="native",
        )
    )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, 6, w=w))


@pytest.mark.parametrize("kind", CI_KINDS)
def test_matches_binsreg_just_below_the_absorption_threshold(kind):
    """The widest dummy block that still reaches the interval machinery.

    One level more and the control would be absorbed as a fixed effect, which
    ``test_absorbed_fixed_effect_raises_a_pointed_error`` covers. This is the
    other side of that boundary, and the only test where the dummy block
    dominates the design.
    """
    levels = ABSORB_MIN_LEVELS - 1
    rng = np.random.default_rng(23)
    n = 6_000
    group = rng.integers(0, levels, size=n)
    x = rng.normal(size=n)
    y = 1.1 * x + group * 0.05 + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "g": pd.Series(group).astype(str)})
    w = pd.get_dummies(df["g"], drop_first=True).to_numpy(float)

    ours = _sorted_native(
        binscatter(
            df, "x", "y", controls=["g"], num_bins=6, ci=kind, return_type="native"
        )
    )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, 6, w=w))


@pytest.mark.parametrize(("kind", "num_basis_columns"), [("pointwise", 5), ("rbc", 6)])
def test_rank_deficient_controls_differ_from_binsreg_only_in_the_dof_factor(
    kind, num_basis_columns
):
    """The one input case where the two packages disagree, pinned to its cause.

    Two perfectly collinear controls leave the design singular. Both packages
    still produce the interval -- the fitted value and its variance are identified
    even where the coefficients are not -- but they count the redundant column
    differently in the HC1 finite-sample correction ``n / (n - k)``: binsreg drops
    it, this package keeps it. So the standard errors sit at exactly the ratio
    below, which is 1.0005 here and shrinks with the sample.

    Asserting that exact ratio rather than a loose tolerance is what makes this a
    test: any *other* divergence appearing in the rank-deficient path would break
    it, and a change to the correction itself is supposed to break it.
    """
    rng = np.random.default_rng(1)
    n = 1_000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.2 * x + 0.5 * z + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z, "z2": 2 * z})

    ours = _sorted_native(
        binscatter(
            df,
            "x",
            "y",
            controls=["z", "z2"],
            num_bins=5,
            ci=kind,
            return_type="native",
        )
    )
    reference = _binsreg_ci(df, kind, 5, w=df[["z", "z2"]])

    assert np.all(np.isfinite(ours["ci_std_error"].to_numpy(float)))
    assert np.all(ours["ci_upper"].to_numpy(float) > ours["ci_lower"].to_numpy(float))

    half_width_ours = (
        ours["ci_upper"].to_numpy(float) - ours["ci_lower"].to_numpy(float)
    ) / 2.0
    half_width_ref = (
        reference["ci_r"].to_numpy(float) - reference["ci_l"].to_numpy(float)
    ) / 2.0
    expected_ratio = np.sqrt(
        (n - (num_basis_columns + 1)) / (n - (num_basis_columns + 2))
    )
    np.testing.assert_allclose(
        half_width_ours / half_width_ref, expected_ratio, rtol=1e-9
    )


@pytest.mark.parametrize("backend", EXACT_QUANTILE_BACKENDS)
@pytest.mark.parametrize("kind", CI_KINDS)
def test_exact_quantile_backends_match_binsreg(backend, kind):
    """Every exact-quantile backend is compared to binsreg, not only to pandas.

    ``test_exact_quantile_backends_agree`` pins them to each other, which a shared
    mistake would survive; the other binsreg comparisons all feed in a pandas
    frame. This closes the triangle.
    """
    df = _sim()
    num_bins = 5
    ours = _sorted_native(
        binscatter(
            convert_to_backend(df, backend),
            "x",
            "y",
            controls=["w"],
            num_bins=num_bins,
            ci=kind,
            return_type="native",
        )
    )
    _assert_matches_binsreg(ours, _binsreg_ci(df, kind, num_bins, w=df[["w"]]))


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


@pytest.mark.quick
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


@pytest.mark.quick
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


@pytest.mark.parametrize("kind", CI_KINDS)
def test_poly_line_does_not_move_the_intervals(kind):
    """The overlay is fitted from the raw rows and must not feed back into the bounds.

    ``test_intervals_combine_with_poly_line`` only checks that both traces are
    drawn; this checks the numbers underneath are the ones computed without it.
    """
    df = _make_frame()
    kwargs = {"controls": ["z"], "num_bins": 6, "ci": kind, "return_type": "native"}
    plain = _sorted_native(binscatter(df, "x", "y", **kwargs))
    with_line = _sorted_native(binscatter(df, "x", "y", poly_line=2, **kwargs))
    for column in ("ci_lower", "ci_upper", "ci_std_error"):
        np.testing.assert_allclose(
            with_line[column].to_numpy(float),
            plain[column].to_numpy(float),
            rtol=1e-10,
            atol=1e-12,
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


def test_absorbed_fixed_effect_raises_a_pointed_error():
    """Absorption never forms the group block the sandwich variance needs."""
    rng = np.random.default_rng(5)
    n = 4_000
    firm = rng.integers(0, 120, size=n)  # above ABSORB_MIN_LEVELS
    x = rng.normal(size=n)
    y = 1.1 * x + firm * 0.01 + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "firm": firm})
    with pytest.raises(NotImplementedError, match="absorbed as a fixed effect"):
        binscatter(
            df,
            "x",
            "y",
            controls=["firm"],
            categorical=["firm"],
            num_bins=6,
            ci="pointwise",
            return_type="native",
        )


def test_warns_when_bins_were_chosen_to_minimise_imse():
    """A pointwise interval at the IMSE-optimal bin count under-covers.

    The bias and the standard error are the same order there by construction, and
    ``pointwise`` accounts only for the latter.
    """
    df = _make_frame()
    with pytest.warns(UserWarning, match="IMSE-optimal"):
        binscatter(df, "x", "y", ci="pointwise", return_type="native")


def test_rbc_does_not_warn_under_automatic_bin_selection():
    """Robust bias correction is valid *at* the IMSE-optimal bin count.

    That is the whole point of building the interval from the next-order fit, so
    warning here would steer users away from the regime ``rbc`` exists to serve.
    binsreg likewise forces the bias-corrected degree when it selects the bin count
    and stays silent.
    """
    df = _make_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        binscatter(df, "x", "y", ci="rbc", return_type="native")
    assert not [w for w in caught if "IMSE" in str(w.message)]


def test_no_warning_when_bins_are_given_explicitly():
    import warnings

    df = _make_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binscatter(df, "x", "y", num_bins=20, ci="pointwise", return_type="native")
