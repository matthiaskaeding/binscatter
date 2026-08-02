"""What any fixed-effect implementation has to satisfy, whichever one is installed.

The other two fixed-effect files test *this* implementation: they lower
``ABSORB_MIN_LEVELS`` to force the absorbed route, stub ``select_absorbed_factors``
to force the one-hot route, and compare the two. That is a differential test between
two paths of one design -- excellent at catching regressions in it, and worthless as
a specification, because it presumes both paths exist and that a module constant
switches between them. An implementation that always absorbs, or that has no one-hot
fallback, fails all of it without being wrong.

This file is the portable half. Three rules, so that it survives a rewrite of the
fixed-effect machinery:

* Nothing is imported from ``binscatter`` except :func:`binscatter`. No private
  functions, no module constants, no monkeypatching of either.
* Correctness is judged against a dense one-hot OLS built here with numpy, which is
  the definition absorption claims to compute. Never against another binscatter run.
* Nothing asserts *how* the answer was reached -- not which factors were absorbed,
  not how many aggregations ran. Only :func:`test_high_cardinality_is_not_quadratic`
  cares about cost, and it measures wall time rather than inspecting the route.

The cardinalities are realistic on purpose. Nothing here forces absorption, so a
factor is given enough levels that any implementation worth having will absorb it;
the oracle comparison is then valid either way, and stays valid if the threshold
moves.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from binscatter import binscatter
from tests.conftest import DF_BACKENDS, convert_to_backend, to_pandas_native

#: High enough that an implementation which absorbs will absorb ``firm``, without
#: making the dense oracle expensive to build.
BIG_FACTOR_LEVELS = 60


def make_panel(n=4000, levels=(BIG_FACTOR_LEVELS, 8, 5), seed=0, noise=0.5):
    """A panel with one high-cardinality factor and two small ones."""
    rng = np.random.default_rng(seed)
    codes = [rng.integers(0, g, n) for g in levels]
    age = rng.normal(size=n)
    x = rng.normal(size=n) + 0.3 * age
    y = 0.8 * x + 1.5 * age + rng.normal(scale=noise, size=n)
    for scale, g, code in zip((2.0, 1.2, 0.9), levels, codes):
        y = y + rng.normal(scale=scale, size=g)[code]
    frame = {"x": x, "y": y, "age": age}
    for idx, code in enumerate(codes):
        frame[f"fac{idx}"] = [f"f{idx}_{value:03d}" for value in code]
    return pd.DataFrame(frame)


def dots(df, **kwargs):
    """The plotted y values, sorted by bin, from whatever backend came in."""
    out = binscatter(df, "x", "y", return_type="native", **kwargs)
    return to_pandas_native(out).sort_values("bin")["y"].to_numpy()


def dense_ols_dots(df, num_bins, numeric=(), categorical=()):
    """The same estimand, from a textbook one-hot design solved with numpy.

    Bin dummies (no intercept), the numeric controls as they are, and each
    categorical as dummies less a reference level. The reported curve is the bin
    coefficient evaluated at the sample mean of every control, which is what a
    binscatter dot is. Built here rather than imported so that this file does not
    depend on the machinery it is judging.
    """
    edges = np.unique(np.quantile(df["x"], np.linspace(0.0, 1.0, num_bins + 1)))
    width = edges.size - 1
    idx = np.clip(np.searchsorted(edges, df["x"], side="right") - 1, 0, width - 1)

    blocks = [np.eye(width)[idx]]
    if numeric:
        blocks.append(df[list(numeric)].to_numpy(dtype=float))
    for column in categorical:
        codes, uniques = pd.factorize(df[column], sort=True)
        blocks.append(np.eye(len(uniques))[codes][:, 1:])
    design = np.column_stack(blocks)

    beta, *_ = np.linalg.lstsq(design, df["y"].to_numpy(dtype=float), rcond=None)
    return beta[:width] + design[:, width:].mean(axis=0) @ beta[width:]


# --------------------------------------------------------------------------
# 1. The estimand
# --------------------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize(
    ("controls", "numeric", "categorical"),
    [
        pytest.param(["fac0"], (), ("fac0",), id="one_factor"),
        pytest.param(["age", "fac0"], ("age",), ("fac0",), id="one_factor_and_numeric"),
        pytest.param(
            ["age", "fac0", "fac1"],
            ("age",),
            ("fac0", "fac1"),
            id="two_factors_and_numeric",
        ),
        pytest.param(
            ["age", "fac0", "fac1", "fac2"],
            ("age",),
            ("fac0", "fac1", "fac2"),
            id="three_factors_and_numeric",
        ),
        pytest.param(["fac0", "fac1"], (), ("fac0", "fac1"), id="factors_only"),
    ],
)
def test_dots_match_dense_ols(controls, numeric, categorical):
    """The dots are the one-hot OLS curve at average controls. That is the contract."""
    df = make_panel()
    np.testing.assert_allclose(
        dots(df, controls=controls, num_bins=8),
        dense_ols_dots(df, 8, numeric, categorical),
        rtol=1e-8,
        atol=1e-8,
    )


# --------------------------------------------------------------------------
# 2. Things that must not change the answer
# --------------------------------------------------------------------------


def test_row_order_does_not_change_the_dots():
    """Rows are a set, not a sequence, whatever order the aggregates come back in."""
    df = make_panel(n=2000)
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    controls = ["age", "fac0", "fac1"]
    np.testing.assert_allclose(
        dots(shuffled, controls=controls, num_bins=6),
        dots(df, controls=controls, num_bins=6),
        rtol=1e-8,
        atol=1e-8,
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
    df = make_panel(n=2000)
    renamed = df.assign(fac0=[relabel(v) for v in df["fac0"]])
    np.testing.assert_allclose(
        dots(renamed, controls=["age", "fac0"], num_bins=6),
        dots(df, controls=["age", "fac0"], num_bins=6),
        rtol=1e-8,
        atol=1e-8,
    )


def test_integer_identifiers_match_their_string_spelling():
    """``categorical=`` on integer codes must mean what string labels mean."""
    df = make_panel(n=2000)
    codes = pd.factorize(df["fac0"], sort=True)[0]
    as_int = df.assign(fac0=codes)
    as_str = df.assign(fac0=[f"L{value}" for value in codes])
    np.testing.assert_allclose(
        dots(as_int, controls=["age", "fac0"], categorical=["fac0"], num_bins=6),
        dots(as_str, controls=["age", "fac0"], num_bins=6),
        rtol=1e-8,
        atol=1e-8,
    )


def test_a_constant_factor_is_a_no_op():
    """A one-level factor carries no information and must not be able to matter."""
    df = make_panel(n=2000).assign(only_one="same")
    np.testing.assert_allclose(
        dots(df, controls=["age", "fac0", "only_one"], num_bins=6),
        dots(df, controls=["age", "fac0"], num_bins=6),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("df_type", DF_BACKENDS)
def test_every_backend_gives_the_same_dots(df_type):
    """A backend is a way of computing the answer, not a different answer.

    Dask and PySpark cut bins on approximate quantiles, so they are compared against
    the exact answer loosely -- the point is that no engine is quietly wrong, not
    that they agree to machine precision on bin edges.
    """
    if df_type == "pyspark":
        pytest.importorskip("pyspark")
    df = make_panel(n=2000)
    controls = ["age", "fac0", "fac1"]
    actual = dots(convert_to_backend(df, df_type), controls=controls, num_bins=6)
    expected = dense_ols_dots(df, 6, ("age",), ("fac0", "fac1"))
    loose = df_type in {"dask", "pyspark"}
    np.testing.assert_allclose(
        actual, expected, rtol=0.1 if loose else 1e-8, atol=0.15 if loose else 1e-8
    )


# --------------------------------------------------------------------------
# 3. Structures a group-mean method has to survive
# --------------------------------------------------------------------------


def test_nested_factors_match_dense_ols():
    """County inside state: the design is rank-deficient beyond the usual amount."""
    rng = np.random.default_rng(11)
    n = 3000
    county = rng.integers(0, BIG_FACTOR_LEVELS, n)
    state = county // 10
    x = rng.normal(size=n)
    y = (
        0.8 * x
        + rng.normal(scale=2.0, size=BIG_FACTOR_LEVELS)[county]
        + rng.normal(scale=0.5, size=n)
    )
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "county": [f"c{v:02d}" for v in county],
            "state": [f"s{v}" for v in state],
        }
    )
    np.testing.assert_allclose(
        dots(df, controls=["county", "state"], num_bins=6),
        dense_ols_dots(df, 6, (), ("county", "state")),
        rtol=1e-7,
        atol=1e-7,
    )


def test_singleton_levels_match_dense_ols():
    """Levels seen once are where a group-mean method has least to work with."""
    df = make_panel(n=1200, levels=(200, 8, 5), seed=7)
    np.testing.assert_allclose(
        dots(df, controls=["age", "fac0"], num_bins=5),
        dense_ols_dots(df, 5, ("age",), ("fac0",)),
        rtol=1e-7,
        atol=1e-7,
    )


def test_factors_sharing_label_values_stay_separate():
    """Two factors both labelled ``a, b, c`` are still two factors.

    Level names are only unique within a factor. An implementation that keys levels
    globally merges ``fac0 == "a"`` with ``fac1 == "a"`` into one effect, which
    raises nothing and quietly moves the dots.
    """
    df = make_panel(n=2000, levels=(8, 8, 5), seed=23)
    shared = list("abcdefgh")
    df = df.assign(
        fac0=[shared[int(v.split("_")[1])] for v in df["fac0"]],
        fac1=[shared[int(v.split("_")[1])] for v in df["fac1"]],
    )
    np.testing.assert_allclose(
        dots(df, controls=["age", "fac0", "fac1"], num_bins=5),
        dense_ols_dots(df, 5, ("age",), ("fac0", "fac1")),
        rtol=1e-7,
        atol=1e-7,
    )


def test_nulls_in_a_factor_drop_their_rows():
    """A null level is a missing row, not a level named null."""
    df = make_panel(n=2000, seed=41)
    df.loc[:49, "fac0"] = None
    complete = df.dropna(subset=["fac0"])
    np.testing.assert_allclose(
        dots(df, controls=["age", "fac0"], num_bins=5),
        dense_ols_dots(complete, 5, ("age",), ("fac0",)),
        rtol=1e-7,
        atol=1e-7,
    )


# --------------------------------------------------------------------------
# 4. The reason absorption exists
# --------------------------------------------------------------------------


def test_high_cardinality_is_not_quadratic():
    """Cost is the whole point, so it is part of the contract.

    One-hot encoding 5,000 levels means a design of that width and an aggregation
    count quadratic in it, which does not finish in any reasonable time. This does
    not check *how* an implementation avoids that -- only that it does.
    """
    df = make_panel(n=20_000, levels=(5_000, 4, 3), seed=5)
    start = time.perf_counter()
    out = dots(df, controls=["age", "fac0"], num_bins=6)
    elapsed = time.perf_counter() - start

    assert np.all(np.isfinite(out))
    assert elapsed < 30.0, (
        f"{elapsed:.1f}s for 5,000 levels; a quadratic route is being taken"
    )
