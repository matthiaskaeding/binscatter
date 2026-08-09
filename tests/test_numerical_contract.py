"""Invariances the estimator has to satisfy however it is computed.

The dots are the conditional mean of ``y`` given the bin, at the sample-average
control values. That estimand does not know what units a control was measured in,
where its origin sits, which basis was chosen for the space the controls span, or
how many redundant copies of a column were passed in. Neither may the answer.

Every test here is written the way ``test_fe_contract.py`` is written, and for the
same reason -- so it stays valid when the machinery underneath is replaced:

* Nothing is imported from ``binscatter`` except :func:`binscatter`. No private
  helpers, no module constants, no monkeypatching.
* Nothing asserts *how* an answer was reached: no route is inspected, no fixed
  effect is forced down one path or another. Multi-factor cases are here because
  they are the numerically hard ones, not to test a particular way of handling them.
* Correctness is judged against the same call on transformed data. No oracle is
  needed: a shift, a rescaling, a change of basis inside the control span, and a
  duplicated column all leave the estimand exactly where it was, so binscatter's
  own answer before the change is the reference.

The tests marked ``xfail`` are the ones binscatter does not satisfy today, tracked
in #95. Three causes, and the reason on each mark says which:

* The normal equations are assembled from moments about zero rather than about the
  mean, so the answer carries an error that grows with the square of the ratio
  between a column's location and its spread. Most of the marks are this.
* Degrees of freedom are counted as control columns rather than as the rank of the
  design, so a column carrying no new direction still costs one.
* An unidentified design gets whichever least-squares solution the control's units
  happen to imply, rather than a refusal or a pinned convention.
* The per-bin aggregates are built under fixed column names, which collide with a
  user column of the same name.

Every mark is ``strict``, so a test that starts passing turns red and forces the
mark off rather than quietly going green.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from binscatter import binscatter

#: A location large enough to dominate the spread of the shifted column without
#: coming anywhere near the range of a float. Roughly what a POSIX timestamp, a
#: population count or a currency amount in minor units looks like.
BIG = 1_000_000_000.0

#: Agreement seen between two mathematically identical calls is at machine
#: precision; this leaves room for a different summation order without admitting a
#: genuine disagreement.
RTOL = 1e-7
ATOL = 1e-6

UNCENTERED_MOMENTS = "#95: normal equations are built from moments about zero"
NOMINAL_DEGREES_OF_FREEDOM = "#95: degrees of freedom count control columns, not rank"
REDUNDANT_CONSTANT = "#95: an explicit constant collapses the fitted dots to zero"


def dots(df, **kwargs) -> np.ndarray:
    """The plotted y values, in bin order."""
    out = binscatter(df, "x", "y", return_type="native", **kwargs)
    return out.sort_values("bin")["y"].to_numpy(dtype=float)


def selected_bins(df, **kwargs) -> int:
    """How many bins the call ended up with."""
    out = binscatter(df, "x", "y", return_type="native", **kwargs)
    return len(out)


def interval_values(df, **kwargs) -> np.ndarray:
    """Centres, bounds and standard errors, in bin order."""
    out = binscatter(df, "x", "y", return_type="native", **kwargs)
    columns = ["y", "ci_lower", "ci_upper", "ci_std_error"]
    return out.sort_values("bin")[columns].to_numpy(dtype=float)


def polynomial_line(df, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """The overlay trace's x and y, which only the figure carries."""
    figure = binscatter(df, "x", "y", **kwargs)
    trace = figure.data[1]
    return (
        np.asarray(trace.x, dtype=float),
        np.asarray(trace.y, dtype=float),
    )


def panel(n=2000, firms=30, years=25, seed=91) -> pd.DataFrame:
    """A panel with one numeric control and two factors.

    Deliberately awkward: the factor effects are large relative to the residual, so
    a solve that loses precision on the cross-moments between the dummies and the
    numeric control has somewhere visible to put the error.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    firm = rng.integers(0, firms, size=n)
    year = rng.integers(0, years, size=n)
    y = (
        1.2 * x
        + 0.8 * z
        + rng.normal(size=firms)[firm]
        + rng.normal(size=years)[year]
        + rng.normal(scale=0.2, size=n)
    )
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
            "firm": [f"firm_{value:03d}" for value in firm],
            "year": [f"year_{value:02d}" for value in year],
        }
    )


def simple_frame(n=1200, seed=2026) -> pd.DataFrame:
    """One numeric control, no factors: the smallest case the invariances apply to."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.2 * x + 0.8 * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


CONTROL_SETS = [
    pytest.param(["z"], id="numeric_only"),
    pytest.param(["z", "firm"], id="one_factor"),
    pytest.param(["z", "firm", "year"], id="two_factors"),
]


# --------------------------------------------------------------------------
# 1. Location: where a column sits is not information
# --------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
@pytest.mark.parametrize("controls", CONTROL_SETS)
def test_shifting_a_control_does_not_move_the_dots(controls):
    """Adding a constant to a control adds a constant to the average it is held at.

    The dots are evaluated at the sample mean of the controls, so the shift cancels
    exactly. A user whose control is a year, a timestamp or a price level gets the
    same picture as one who centred first, or the answer depends on the origin of a
    variable that has none.
    """
    frame = panel()
    baseline = dots(frame, controls=controls, num_bins=10)
    shifted = dots(frame.assign(z=frame["z"] + BIG), controls=controls, num_bins=10)

    np.testing.assert_allclose(shifted, baseline, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("controls", CONTROL_SETS)
def test_shifting_x_and_y_shifts_the_dots_by_the_same_amount(controls):
    """Moving the response moves the curve; moving the predictor moves neither."""
    frame = panel()
    baseline = dots(frame, controls=controls, num_bins=10)
    moved = dots(
        frame.assign(x=frame["x"] + BIG, y=frame["y"] + BIG),
        controls=controls,
        num_bins=10,
    )

    np.testing.assert_allclose(moved - BIG, baseline, rtol=RTOL, atol=ATOL)


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
@pytest.mark.parametrize("selector", ["rot", "dpi"])
@pytest.mark.parametrize("shift", ["control", "x_and_y"])
def test_automatic_selectors_are_location_invariant(selector, shift):
    """A selector that reads the origin of a column selects on an artefact.

    Both selectors estimate variance and curvature constants from the same
    cross-products the dots come from, so a location that swamps the spread is
    enough to change the bin count -- or to make the variance constant come out
    non-positive and the selector give up.
    """
    frame = panel(n=1500, firms=20, years=15)
    controls = ["z", "firm"]
    baseline = selected_bins(frame, controls=controls, num_bins=selector)
    if shift == "control":
        changed = frame.assign(z=frame["z"] + BIG)
    else:
        changed = frame.assign(x=frame["x"] + BIG, y=frame["y"] + BIG)

    assert selected_bins(changed, controls=controls, num_bins=selector) == baseline


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_intervals_are_location_invariant(kind):
    """Standard errors describe spread, which a shift of the control cannot touch."""
    frame = simple_frame()
    baseline = interval_values(frame, controls=["z"], num_bins=10, ci=kind)
    shifted = interval_values(
        frame.assign(z=frame["z"] + BIG), controls=["z"], num_bins=10, ci=kind
    )

    np.testing.assert_allclose(shifted, baseline, rtol=RTOL, atol=ATOL)


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
@pytest.mark.parametrize("controls", CONTROL_SETS)
def test_polynomial_line_is_location_invariant(controls):
    """The overlay is fitted on the same data as the dots and shares their invariances."""
    frame = panel()
    _, baseline = polynomial_line(frame, controls=controls, num_bins=10, poly_line=2)
    _, shifted = polynomial_line(
        frame.assign(z=frame["z"] + BIG), controls=controls, num_bins=10, poly_line=2
    )

    np.testing.assert_allclose(shifted, baseline, rtol=RTOL, atol=ATOL)


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
def test_polynomial_grid_does_not_collapse_at_a_large_x_location():
    """A visible spread in x stays visible when its location is much larger.

    The overlay is drawn on a grid spanning the x range. Built from raw powers of a
    large x, the grid loses the spread to rounding and the line is drawn on a
    handful of distinct points, or on one.
    """
    centred_x = np.linspace(-1.0, 1.0, 1200)
    y = 0.5 + 2.0 * centred_x + 0.3 * centred_x**2
    baseline_x, baseline_y = polynomial_line(
        pd.DataFrame({"x": centred_x, "y": y}), num_bins=10, poly_line=2
    )
    shifted_x, shifted_y = polynomial_line(
        pd.DataFrame({"x": 1e10 + centred_x, "y": y}), num_bins=10, poly_line=2
    )

    assert baseline_x.size > 1
    assert shifted_x.size == baseline_x.size
    np.testing.assert_allclose(shifted_x - 1e10, baseline_x, rtol=0.0, atol=3e-6)
    np.testing.assert_allclose(shifted_y, baseline_y, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# 2. Units: a control's scale is a choice of measurement
# --------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("controls", CONTROL_SETS)
def test_rescaling_a_control_does_not_move_the_dots(controls):
    """Metres or millimetres, the coefficient absorbs the factor and the dots do not."""
    frame = panel()
    baseline = dots(frame, controls=controls, num_bins=10)
    rescaled = dots(
        frame.assign(z=frame["z"] * 10_000_000.0), controls=controls, num_bins=10
    )

    np.testing.assert_allclose(rescaled, baseline, rtol=RTOL, atol=ATOL)


def test_the_rule_of_thumb_selector_is_invariant_to_control_units():
    """The IMSE-optimal bin count cannot depend on how a control was measured.

    The DPI selector is held to the same requirement in
    :func:`test_the_dpi_selector_stays_answerable_when_a_control_is_rescaled`,
    which has to run out of process because it currently does not terminate.
    """
    frame = simple_frame(n=800, seed=123)
    baseline = selected_bins(frame, controls=["z"], num_bins="rot")
    rescaled = selected_bins(
        frame.assign(z=frame["z"] * 10_000_000.0), controls=["z"], num_bins="rot"
    )

    assert rescaled == baseline


#: Run out of process, under a wall-clock and address-space budget: the failure
#: being tested for is a call that never comes back, which cannot be asserted on
#: from inside the process it would hang. ``sys.argv[1]`` scales the control.
DPI_UNIT_SCRIPT = """
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3,) * 2)
import numpy as np, pandas as pd
from binscatter import binscatter

rng = np.random.default_rng(123)
n = 300
x = rng.normal(size=n)
z = rng.normal(size=n)
y = 1.2 * x + 0.8 * z + rng.normal(scale=0.5, size=n)
frame = pd.DataFrame({"x": x, "y": y, "z": z * float(sys.argv[1])})
try:
    out = binscatter(
        frame, "x", "y", controls=["z"], num_bins="dpi", return_type="native"
    )
except ValueError:
    print("refused")
else:
    print(len(out))
"""


def run_selector_script(scale: str, timeout: float) -> str:
    """The script's verdict for this scale: a bin count, ``refused``, or a failure."""
    try:
        finished = subprocess.run(
            [sys.executable, "-c", DPI_UNIT_SCRIPT, scale],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"did not finish within {timeout:.0f}s"
    if finished.returncode != 0:
        return f"exited {finished.returncode}: {finished.stderr.strip()[-200:]}"
    return finished.stdout.strip()


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
def test_the_dpi_selector_stays_answerable_when_a_control_is_rescaled():
    """Rescaling a control must not turn the call into one that cannot return.

    The bin count is chosen from an estimated variance-to-bias ratio. Estimated
    from moments about zero, that ratio is not scale-free, and a control measured
    in smaller units drives the count far past the number of rows -- at which point
    the call is not slow, it is unanswerable: every bin edge it then asks for has
    to be computed before anything else can happen.

    The budget is generous relative to a working selector, which answers this in
    well under a second.
    """
    pytest.importorskip("resource", reason="POSIX resource limits required")

    baseline = run_selector_script("1.0", timeout=120.0)
    assert baseline.isdigit(), baseline

    rescaled = run_selector_script("1e7", timeout=60.0)
    assert rescaled in {baseline, "refused"}, rescaled


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_intervals_are_invariant_to_control_units(kind):
    """Rescaling a control changes its coefficient, not its covariance contribution."""
    frame = simple_frame()
    baseline = interval_values(frame, controls=["z"], num_bins=10, ci=kind)
    rescaled = interval_values(
        frame.assign(z=frame["z"] * 10_000_000.0), controls=["z"], num_bins=10, ci=kind
    )

    np.testing.assert_allclose(rescaled, baseline, rtol=RTOL, atol=ATOL)


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
@pytest.mark.parametrize("scale", [1e-170, 1e200], ids=["tiny", "huge"])
def test_extreme_control_scales_agree_with_the_latent_control(scale):
    """A finite control may not disappear through underflow or overflow.

    ``z`` is the same variable in both calls, differing only by a power of ten that
    is exact in binary. Forming ``z**2`` overflows for the huge case and underflows
    to zero for the tiny one, so a solve that squares the column before scaling it
    silently drops the control.
    """
    rng = np.random.default_rng(103)
    x = rng.normal(size=1200)
    latent = rng.normal(size=x.size)
    y = 2.0 * x + 3.0 * latent + rng.normal(scale=0.1, size=x.size)
    frame = pd.DataFrame({"x": x, "y": y, "latent": latent})

    baseline = dots(frame, controls=["latent"], num_bins=10)
    scaled = dots(frame.assign(z=scale * frame["latent"]), controls=["z"], num_bins=10)

    np.testing.assert_allclose(scaled, baseline, rtol=RTOL, atol=ATOL)


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
def test_large_integer_controls_agree_with_their_scaled_copy():
    """Integer identifiers big enough that their squares leave the int64 range.

    Nine-digit integers are ordinary data -- account numbers, populations, cents.
    Their cross-products are not representable, so an aggregation that stays in
    integer arithmetic wraps, and one that casts late loses the low bits.
    """
    rng = np.random.default_rng(101)
    x = rng.normal(size=1200)
    z = rng.integers(-5_000_000_000, 5_000_000_000, size=x.size, dtype=np.int64)
    y = 2.0 * x + 1e-9 * z + rng.normal(scale=0.1, size=x.size)
    frame = pd.DataFrame({"x": x, "y": y, "z": z})

    baseline = dots(
        frame.assign(scaled=frame["z"] / 1e9), controls=["scaled"], num_bins=10
    )
    actual = dots(frame, controls=["z"], num_bins=10)

    np.testing.assert_allclose(actual, baseline, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# 3. Basis: only the space the controls span can matter
# --------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=UNCENTERED_MOMENTS)
def test_near_collinear_controls_agree_with_a_conditioned_basis():
    """``(z1, z2)`` and ``(z1, (z2 - z1) * 1e8)`` span the same plane.

    Partialling out a space is a projection, so the two calls have identical
    residuals and identical dots. The difference is conditioning: the first basis
    is nearly collinear and the second is not. Squaring the condition number by
    forming the cross-products about zero is what turns a solvable problem into a
    visibly different curve.
    """
    rng = np.random.default_rng(3)
    n = 1200
    x = rng.normal(size=n)
    z1 = rng.normal(size=n)
    noise = rng.normal(size=n)
    z2 = z1 + 1e-8 * noise
    y = 2.0 * x + 3.0 * z1 + 0.2 * noise + rng.normal(scale=0.1, size=n)
    frame = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})

    actual = dots(frame, controls=["z1", "z2"], num_bins=10)
    expected = dots(
        frame.assign(spread=(frame["z2"] - frame["z1"]) * 1e8),
        controls=["z1", "spread"],
        num_bins=10,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_rotating_the_controls_does_not_move_the_dots():
    """A change of basis inside the control block is invisible to the projection."""
    rng = np.random.default_rng(77)
    n = 1500
    x = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    y = 0.9 * x + 1.4 * z1 - 0.6 * z2 + rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})

    baseline = dots(frame, controls=["z1", "z2"], num_bins=10)
    rotated = dots(
        frame.assign(
            a=0.6 * frame["z1"] + 0.8 * frame["z2"],
            b=-0.8 * frame["z1"] + 0.6 * frame["z2"],
        ),
        controls=["a", "b"],
        num_bins=10,
    )

    np.testing.assert_allclose(rotated, baseline, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# 4. Redundancy: a column carrying no new direction changes nothing
# --------------------------------------------------------------------------


REDUNDANCIES = [
    pytest.param("duplicate", id="duplicate"),
    pytest.param("constant", id="constant"),
]


def add_redundant(frame: pd.DataFrame, redundancy: str) -> pd.DataFrame:
    """A copy of ``z``, or a constant column: neither adds an estimable direction."""
    return frame.assign(
        redundant=frame["z"] if redundancy == "duplicate" else 1.0,
    )


@pytest.mark.parametrize(
    "redundancy",
    [
        pytest.param("duplicate", id="duplicate"),
        pytest.param(
            "constant",
            id="constant",
            marks=pytest.mark.xfail(
                sys.platform == "darwin", strict=True, reason=REDUNDANT_CONSTANT
            ),
        ),
    ],
)
def test_redundant_controls_do_not_move_the_dots(redundancy):
    """The fitted curve stays estimable when redundancy is inside the controls.

    The constant case currently exposes a platform-dependent least-squares
    solution: it passes with the Linux CI stack and collapses to zero on macOS.
    """
    frame = simple_frame()
    baseline = dots(frame, controls=["z"], num_bins=10)
    actual = dots(
        add_redundant(frame, redundancy), controls=["z", "redundant"], num_bins=10
    )

    np.testing.assert_allclose(actual, baseline, rtol=1e-9, atol=1e-9)


@pytest.mark.xfail(strict=True, reason=NOMINAL_DEGREES_OF_FREEDOM)
@pytest.mark.parametrize("redundancy", REDUNDANCIES)
@pytest.mark.parametrize("kind", ["pointwise", "rbc"])
def test_redundant_controls_do_not_change_the_intervals(kind, redundancy):
    """The HC1 correction divides by the rank of the design, not by its width.

    The centres are unaffected; the bounds and standard errors come out scaled by
    ``sqrt((n - k) / (n - k - 1))``, the redundant column having been charged a
    degree of freedom it cannot spend.

    This mark and ``test_rank_deficient_controls_differ_from_binsreg_only_in_the_dof
    _factor`` in ``tests/test_inference.py`` disagree on purpose: that test pins the
    current accounting against `binsreg`'s, which drops the redundant column, and
    the two cannot both hold. Whichever way the question is settled, one of them has
    to move -- which is the point of recording the specification here rather than
    leaving it in an issue.
    """
    frame = simple_frame()
    baseline = interval_values(frame, controls=["z"], num_bins=10, ci=kind)
    actual = interval_values(
        add_redundant(frame, redundancy),
        controls=["z", "redundant"],
        num_bins=10,
        ci=kind,
    )

    np.testing.assert_allclose(actual, baseline, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("redundancy", REDUNDANCIES)
@pytest.mark.parametrize(
    "selector",
    [
        "rot",
        pytest.param(
            "dpi",
            marks=pytest.mark.xfail(strict=True, reason=NOMINAL_DEGREES_OF_FREEDOM),
        ),
    ],
)
def test_redundant_controls_do_not_change_the_selected_bin_count(selector, redundancy):
    """Rank-zero additions leave the selector's degrees of freedom where they were.

    The rule-of-thumb count survives the extra column; the plug-in count does not,
    the redundant column entering its variance estimate as a spent degree of
    freedom. The two selectors sharing a data path makes the difference a fact
    about the degrees-of-freedom accounting rather than about either formula.
    """
    rng = np.random.default_rng(48)
    n = 800
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.sin(x) + 0.5 * z + rng.normal(scale=0.8, size=n)
    frame = pd.DataFrame({"x": x, "y": y, "z": z})

    baseline = selected_bins(frame, controls=["z"], num_bins=selector)
    actual = selected_bins(
        add_redundant(frame, redundancy), controls=["z", "redundant"], num_bins=selector
    )

    assert actual == baseline


# --------------------------------------------------------------------------
# 5. Designs where the answer is not determined
# --------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.xfail(
    strict=True,
    reason="#95: an unidentified design gets whichever solution the units imply",
)
def test_a_control_collinear_with_the_bins_is_not_answered_arbitrarily():
    """The curve at average controls is not identified when W reproduces the bins.

    Every split of the fit between the bin dummies and the control is equally
    consistent with the data, and which one a solver lands on depends on the
    control's units. Refusing is one defensible answer and pinning a convention is
    another; returning a number whose value moves when the units do is not.
    """
    rng = np.random.default_rng(106)
    x = rng.normal(size=1200)
    y = 1.2 * x + rng.normal(scale=0.2, size=x.size)
    edges = pd.Series(x).quantile([index / 10 for index in range(11)]).to_numpy()
    bin_index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    frame = pd.DataFrame({"x": x, "y": y})

    answers = []
    for scale in (1.0, 1e-9):
        try:
            answers.append(
                dots(
                    frame.assign(collinear=scale * bin_index.astype(float)),
                    controls=["collinear"],
                    num_bins=10,
                )
            )
        except ValueError:
            return  # refused, which is a defensible answer

    np.testing.assert_allclose(answers[0], answers[1], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# 6. Column names are data, not instructions
# --------------------------------------------------------------------------


COLLIDES_WITH_A_TEMPORARY = "#95: bin-level aggregates are built under fixed names"


@pytest.mark.parametrize(
    "name",
    [
        "__count",
        "__sum_y",
        "__fe_count",
        "__fe_resp_0",
        "__moment_sum__z",
        "bin",
    ],
)
def test_a_control_named_like_an_internal_column_is_still_a_control(name):
    """Whatever a column is called, it is one of the user's columns.

    An implementation that builds temporaries under fixed names collides with a
    user column that happens to share one -- and the names it picks are not part of
    any documented contract, so a user cannot avoid them. The fixed-effect collector
    already steps aside from names the frame carries; the per-bin aggregates do not,
    and a control named ``__count`` or ``__sum_y`` still fails with
    ``Expected unique column names``, which names neither the column nor the cause.
    """
    frame = panel(n=1500, firms=20, years=15)
    baseline = dots(frame, controls=["z", "firm"], num_bins=8)
    renamed = frame.rename(columns={"z": name})

    np.testing.assert_allclose(
        dots(renamed, controls=[name, "firm"], num_bins=8),
        baseline,
        rtol=RTOL,
        atol=ATOL,
    )


# --------------------------------------------------------------------------
# 7. Degenerate input is refused with a message that names the problem
# --------------------------------------------------------------------------


def all_null(column: str) -> pd.DataFrame:
    """The usual frame, with one column emptied out."""
    frame = simple_frame(n=200)
    frame[column] = np.nan
    return frame


@pytest.mark.parametrize("column", ["x", "y", "z"])
def test_an_all_null_column_is_refused_as_a_data_problem(column):
    """No usable value in a column is a data problem, and has to read as one.

    A ``TypeError`` from coercing a missing quantile is a report about the
    implementation; the caller needs one about the data.
    """
    controls = ["z"] if column == "z" else None

    with pytest.raises(ValueError) as excinfo:
        binscatter(
            all_null(column),
            "x",
            "y",
            controls=controls,
            num_bins=5,
            return_type="native",
        )

    assert "float()" not in str(excinfo.value)


@pytest.mark.xfail(
    strict=True, reason="#95: the refusal names x whichever column is unusable"
)
@pytest.mark.parametrize("column", ["y", "z"])
def test_the_refusal_names_the_column_at_fault(column):
    """Every empty column is reported as a problem with the distribution of ``x``.

    The message is right about what happened -- there were not two distinct bin
    edges -- and wrong about where to look, which is the part a user acts on.
    """
    controls = ["z"] if column == "z" else None

    with pytest.raises(ValueError, match=rf"\b{column}\b"):
        binscatter(
            all_null(column),
            "x",
            "y",
            controls=controls,
            num_bins=5,
            return_type="native",
        )


def test_an_empty_frame_is_refused_as_a_data_problem():
    """Zero rows is the limiting case of the same problem."""
    frame = pd.DataFrame({"x": [], "y": []}, dtype=float)

    with pytest.raises(ValueError) as excinfo:
        binscatter(frame, "x", "y", num_bins=5, return_type="native")

    assert "float()" not in str(excinfo.value)


@pytest.mark.parametrize(
    "num_bins",
    [pytest.param(10.5, id="float"), pytest.param("10", id="numeric_string")],
)
def test_a_non_integer_bin_count_is_refused(num_bins):
    """Bins are counted, so anything that is not a count is a mistake, not a hint."""
    frame = simple_frame(n=200)

    with pytest.raises((TypeError, ValueError)):
        binscatter(frame, "x", "y", num_bins=num_bins, return_type="native")
