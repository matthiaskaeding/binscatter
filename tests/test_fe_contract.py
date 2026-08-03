"""Agreement with `binsreg` across a wide range of fixed-effect specifications.

`binsreg` is the authors' own implementation of the estimator this package
reimplements, so it -- not any oracle written here -- decides what the right answer
is. Fixed effects reach it as dummy columns in ``w``: it has no notion of absorbing
anything, which is exactly what makes it a good judge of an implementation that
does. Whatever route binscatter takes, the dots have to land on binsreg's.

``test_fixed_effects.py`` tests *this* implementation: it lowers
``ABSORB_MIN_LEVELS`` to force the absorbed route, stubs ``select_absorbed`` to
force the one-hot route, and compares the two against each other. That catches
regressions in the current design and specifies nothing -- an implementation that
always absorbs, or that has no one-hot fallback, fails all of it without being wrong.

This file is the portable half, and holds to three rules so it survives replacing the
fixed-effect machinery entirely:

* Nothing is imported from ``binscatter`` except :func:`binscatter`. No private
  functions, no module constants, no monkeypatching of either.
* Correctness is judged against `binsreg`. Never against another binscatter run.
* Nothing asserts *how* the answer was reached. Only
  :func:`test_high_cardinality_is_not_quadratic` cares about cost, and it measures
  wall time rather than inspecting the route.

Cardinalities are realistic rather than forced. The sweep deliberately spans both
sides of any sane absorption threshold -- five levels through several thousand -- so
whichever route an implementation picks for a given specification, binsreg is
computing the same estimand and the comparison holds.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from binscatter import binscatter
from tests.conftest import DF_BACKENDS, convert_to_backend, to_pandas_native

binsreg_module = pytest.importorskip("binsreg")
binsreg_fit = binsreg_module.binsreg

#: Agreement seen in practice is ~1e-15; this leaves room for the accumulation
#: differences between summing per bin and summing per row without admitting a
#: genuine disagreement.
RTOL = 1e-6
ATOL = 1e-6


def make_panel(
    n=3000,
    levels=(60,),
    numeric=("age",),
    seed=0,
    weights=None,
    nested=False,
):
    """A panel with the requested factor cardinalities and numeric controls.

    ``weights`` draws levels from a skewed distribution rather than uniformly, which
    is what produces singleton and near-singleton levels. ``nested`` makes the second
    factor a coarsening of the first, so the design is rank-deficient by more than
    the usual amount.
    """
    rng = np.random.default_rng(seed)
    codes = []
    for size in levels:
        if weights == "skewed":
            probability = np.r_[0.5, np.repeat(0.5 / (size - 1), size - 1)]
            codes.append(rng.choice(size, n, p=probability))
        else:
            codes.append(rng.integers(0, size, n))
    # Guarantee every level appears, so the dummy coding is the same on both sides.
    for code, size in zip(codes, levels):
        code[:size] = np.arange(size)
    if nested and len(levels) >= 2:
        codes[1] = codes[0] // max(1, levels[0] // levels[1])

    frame: dict[str, object] = {}
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.5, size=n)
    for index, name in enumerate(numeric):
        column = rng.normal(size=n) * (1.0 + index)
        frame[name] = column
        x = x + 0.3 * column
        y = y + 1.5 * column
    for scale, size, code in zip((2.0, 1.2, 0.9, 0.6), levels, codes):
        y = y + rng.normal(scale=scale, size=size)[code]

    frame["x"] = x
    frame["y"] = y
    for index, code in enumerate(codes):
        frame[f"fac{index}"] = [f"f{index}_{value:04d}" for value in code]
    return pd.DataFrame(frame)


def binsreg_dots(df, num_bins, numeric=(), factors=()):
    """binsreg's dot values for the same specification, fixed effects as dummies."""
    blocks = []
    if numeric:
        blocks.append(df[list(numeric)].to_numpy(dtype=float))
    for column in factors:
        blocks.append(pd.get_dummies(df[column], drop_first=True).to_numpy(dtype=float))
    w = np.column_stack(blocks) if blocks else None

    reference = binsreg_fit(df["y"], df["x"], w=w, nbins=num_bins, noplot=True)
    dots_frame = reference.data_plot[0].dots.sort_values("x")
    return dots_frame["fit"].to_numpy(dtype=float)


def dots(df, **kwargs):
    """The plotted y values, sorted by bin, from whatever backend came in."""
    out = binscatter(df, "x", "y", return_type="native", **kwargs)
    return to_pandas_native(out).sort_values("bin")["y"].to_numpy()


# --------------------------------------------------------------------------
# 1. The sweep: agreement with binsreg across fixed-effect specifications
# --------------------------------------------------------------------------
# Each case names the factor cardinalities, the numeric controls and the bin count.
# The cardinalities are the point: an implementation that absorbs will absorb some of
# these and one-hot encode others, and binsreg encodes all of them, so every row is a
# check that the route did not change the estimand.

SPECIFICATIONS = [
    pytest.param((5,), ("age",), 6, {}, id="one_factor_5_levels"),
    pytest.param((25,), ("age",), 8, {}, id="one_factor_25_levels"),
    pytest.param((60,), ("age",), 8, {}, id="one_factor_60_levels"),
    pytest.param((200,), ("age",), 6, {}, id="one_factor_200_levels"),
    pytest.param((5,), (), 6, {}, id="one_factor_no_numeric"),
    pytest.param((60,), (), 8, {}, id="one_factor_60_no_numeric"),
    pytest.param((60,), ("age", "score"), 8, {}, id="one_factor_two_numeric"),
    pytest.param((8, 5), ("age",), 6, {}, id="two_factors_both_small"),
    pytest.param((60, 8), ("age",), 8, {}, id="two_factors_mixed_sizes"),
    pytest.param((60, 55), ("age",), 6, {}, id="two_factors_both_large"),
    pytest.param((60, 55), (), 6, {}, id="two_factors_large_no_numeric"),
    pytest.param((40, 12, 5), ("age",), 8, {}, id="three_factors"),
    pytest.param((60, 25, 12), ("age",), 6, {}, id="three_factors_larger"),
    pytest.param((60, 25, 12), ("age", "score"), 6, {}, id="three_factors_two_numeric"),
    pytest.param((60,), ("age",), 2, {}, id="two_bins"),
    pytest.param((60,), ("age",), 20, {}, id="twenty_bins"),
    pytest.param((60, 8), ("age",), 4, {"weights": "skewed"}, id="skewed_levels"),
    pytest.param((200,), ("age",), 5, {"weights": "skewed"}, id="singleton_heavy"),
    pytest.param((60, 6), ("age",), 6, {"nested": True}, id="nested_factors"),
]


@pytest.mark.quick
@pytest.mark.parametrize(("levels", "numeric", "num_bins", "options"), SPECIFICATIONS)
def test_dots_match_binsreg(levels, numeric, num_bins, options):
    """The dots must be binsreg's, whatever route was taken to reach them."""
    df = make_panel(levels=levels, numeric=numeric, **options)
    factors = [f"fac{index}" for index in range(len(levels))]

    np.testing.assert_allclose(
        dots(df, controls=[*numeric, *factors], num_bins=num_bins),
        binsreg_dots(df, num_bins, numeric, factors),
        rtol=RTOL,
        atol=ATOL,
    )


def test_integer_identifiers_match_binsreg():
    """An integer identifier named in ``categorical=`` is a factor, not a slope."""
    df = make_panel(levels=(60,), numeric=("age",))
    codes = pd.factorize(df["fac0"], sort=True)[0]
    as_integers = df.assign(fac0=codes)

    np.testing.assert_allclose(
        dots(
            as_integers,
            controls=["age", "fac0"],
            categorical=["fac0"],
            num_bins=8,
        ),
        binsreg_dots(df, 8, ("age",), ("fac0",)),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("df_type", DF_BACKENDS)
def test_every_backend_matches_binsreg(df_type):
    """A backend is a way of computing the answer, not a different answer.

    Dask and PySpark cut bins on approximate quantiles, so they are held to a loose
    tolerance: the point is that no engine is quietly wrong, not that approximate
    knots reproduce exact ones.
    """
    if df_type == "pyspark":
        pytest.importorskip("pyspark")
    df = make_panel(n=2000, levels=(60, 8), numeric=("age",))
    controls = ["age", "fac0", "fac1"]

    actual = dots(convert_to_backend(df, df_type), controls=controls, num_bins=6)
    expected = binsreg_dots(df, 6, ("age",), ("fac0", "fac1"))

    loose = df_type in {"dask", "pyspark"}
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.1 if loose else RTOL,
        atol=0.15 if loose else ATOL,
    )


# --------------------------------------------------------------------------
# 2. Things that must not move the dots
# --------------------------------------------------------------------------
# No oracle needed: these compare binscatter against itself under a change that
# cannot carry information, so they hold for any implementation without saying
# anything about how it works.


def test_row_order_does_not_change_the_dots():
    """Rows are a set, whatever order a backend's aggregates come back in."""
    df = make_panel(n=2000, levels=(60, 8))
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    controls = ["age", "fac0", "fac1"]

    np.testing.assert_allclose(
        dots(shuffled, controls=controls, num_bins=6),
        dots(df, controls=controls, num_bins=6),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "relabel",
    [
        pytest.param(lambda v: f"z{v}", id="string"),
        pytest.param(lambda v: f"grüße-{v}/ü", id="unicode"),
        pytest.param(lambda v: "L" * 100 + str(v), id="very_long"),
    ],
)
def test_level_labels_do_not_change_the_dots(relabel):
    """A level is an equivalence class; its name is not data."""
    df = make_panel(n=2000, levels=(60,))
    renamed = df.assign(fac0=[relabel(value) for value in df["fac0"]])

    np.testing.assert_allclose(
        dots(renamed, controls=["age", "fac0"], num_bins=6),
        dots(df, controls=["age", "fac0"], num_bins=6),
        rtol=RTOL,
        atol=ATOL,
    )


def test_factors_sharing_label_values_stay_separate():
    """Two factors both labelled ``a, b, c`` are still two factors.

    Level names are only unique within a factor. An implementation that keys levels
    globally merges ``fac0 == "a"`` with ``fac1 == "a"`` into one effect, which
    raises nothing and quietly moves the dots.
    """
    df = make_panel(n=2000, levels=(8, 8), seed=23)
    shared = list("abcdefgh")
    relabelled = df.assign(
        fac0=[shared[int(value.split("_")[1])] for value in df["fac0"]],
        fac1=[shared[int(value.split("_")[1])] for value in df["fac1"]],
    )

    np.testing.assert_allclose(
        dots(relabelled, controls=["age", "fac0", "fac1"], num_bins=5),
        binsreg_dots(relabelled, 5, ("age",), ("fac0", "fac1")),
        rtol=RTOL,
        atol=ATOL,
    )


def test_a_constant_factor_is_a_no_op():
    """A one-level factor carries no information and must not be able to matter."""
    df = make_panel(n=2000, levels=(60,)).assign(only_one="same")

    np.testing.assert_allclose(
        dots(df, controls=["age", "fac0", "only_one"], num_bins=6),
        dots(df, controls=["age", "fac0"], num_bins=6),
        rtol=RTOL,
        atol=ATOL,
    )


def test_nulls_in_a_factor_drop_their_rows():
    """A null level is a missing row, not a level named null."""
    df = make_panel(n=2000, levels=(25,), seed=41)
    df.loc[:49, "fac0"] = None
    complete = df.dropna(subset=["fac0"]).reset_index(drop=True)

    np.testing.assert_allclose(
        dots(df, controls=["age", "fac0"], num_bins=5),
        binsreg_dots(complete, 5, ("age",), ("fac0",)),
        rtol=RTOL,
        atol=ATOL,
    )


# --------------------------------------------------------------------------
# 3. The reason absorption exists
# --------------------------------------------------------------------------


@pytest.mark.quick
def test_high_cardinality_is_not_quadratic():
    """Cost is the whole point, so it is part of the contract.

    Not compared against binsreg: at this width the reference is itself slow, and
    what is being checked is not the estimand but that some route exists which does
    not build a design thousands of columns wide. This is the one test here that an
    implementation without a fast path is expected to fail.
    """
    df = make_panel(n=20_000, levels=(5_000,), seed=5)

    start = time.perf_counter()
    out = dots(df, controls=["age", "fac0"], num_bins=6)
    elapsed = time.perf_counter() - start

    assert np.all(np.isfinite(out))
    assert elapsed < 30.0, (
        f"{elapsed:.1f}s for 5,000 levels; a quadratic route is being taken"
    )
