"""Tests for absorbing two or more categorical controls via a sparse solve.

Multi-way absorption is, like the one-way case, a silent rewrite of an existing code
path: nothing raises if it is subtly wrong, the plotted values just quietly move. So
the gate is the same -- equivalence against the one-hot path on every in-process
backend, backed by an independent dense-OLS oracle -- plus the two things one-way
absorption never had to handle: a rank deficiency of ``F`` rather than one, and a
crosstab that can be too big to collect.

Panel sizes here are chosen against the *one-hot* side of each comparison, not the
absorbed side. Encoding ``G`` levels costs ``k(k+1)/2`` aggregations in ``k = G + J``,
so the levels drive the runtime quadratically while contributing nothing extra to what
the comparison proves -- two factors are two factors at 8 levels each. Tests that only
read cardinalities keep realistic ones; tests that pay for the encoding do not.
"""

from __future__ import annotations

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

import binscatter.fixed_effects as fe_mod
import binscatter.sparse_fe as sparse_mod
from binscatter import binscatter, core
from tests.conftest import DF_BACKENDS, convert_to_backend
from tests.test_fixed_effects import run_native, tolerances, without_absorption

pytest.importorskip("scipy")

#: Backends the equivalence tests run on. PySpark is deliberately absent, unlike in
#: ``test_fixed_effects.py``. Multi-way absorption reaches the backend only through
#: ``group_by`` -- per-factor aggregates and two-key crosstabs -- and both of those are
#: already exercised on PySpark by the one-way absorption tests, which collect the same
#: two-key ``(bin, fe)`` crosstab through the same narwhals calls. Everything past the
#: aggregates is numpy and scipy on the driver and cannot see the backend at all. So a
#: PySpark run here re-pays JVM startup to cover nothing this file is testing.
DF_TYPE_PARAMS = [df_type for df_type in DF_BACKENDS if df_type != "pyspark"]


def make_two_way(n=4000, g1=60, g2=25, seed=0):
    """A panel with two crossed factors, both entering the outcome."""
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, g1, n)
    f2 = rng.integers(0, g2, n)
    age = rng.normal(size=n)
    x = rng.normal(size=n) + 0.3 * age
    y = (
        0.8 * x
        + 1.5 * age
        + rng.normal(scale=2.0, size=g1)[f1]
        + rng.normal(scale=1.2, size=g2)[f2]
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "age": age,
            "firm": [f"f{i:03d}" for i in f1],
            "year": [f"y{i:02d}" for i in f2],
        }
    )


def make_three_way(n=5000, g1=40, g2=25, g3=12, seed=3):
    """Three crossed factors plus a numeric control.

    Three factors mean three pairwise crosstabs rather than one, which is the part
    of the assembly that grows quadratically in the number of absorbed columns.
    """
    rng = np.random.default_rng(seed)
    codes = [rng.integers(0, g, n) for g in (g1, g2, g3)]
    age = rng.normal(size=n)
    x = rng.normal(size=n) + 0.3 * age
    y = 0.8 * x + 1.5 * age + rng.normal(scale=0.5, size=n)
    for scale, g, code in zip((2.0, 1.2, 0.9), (g1, g2, g3), codes):
        y = y + rng.normal(scale=scale, size=g)[code]
    frame = {"x": x, "y": y, "age": age}
    for idx, code in enumerate(codes):
        frame[f"fac{idx}"] = [f"{idx}_{v:03d}" for v in code]
    return pd.DataFrame(frame)


#: Controls for the three-way panel: one numeric, three categorical.
THREE_WAY_CONTROLS = ["age", "fac0", "fac1", "fac2"]
THREE_WAY_CATS = ("fac0", "fac1", "fac2")


@pytest.fixture
def force_multiway(monkeypatch):
    """Absorb even tiny categoricals, so equivalence runs on small data.

    Production absorbs at ``ABSORB_MIN_LEVELS`` or more; the one-hot path this is
    compared against costs O(G^2) aggregations, so a realistic threshold would make
    the comparison unusably slow. Same code, smaller inputs.
    """
    monkeypatch.setattr(fe_mod, "ABSORB_MIN_LEVELS", 2)


def spy_on(monkeypatch, name):
    """Record the ``fe_names`` each call to ``core.<name>`` is handed.

    Returns the list the spy appends to. Routing is asserted from this rather than
    from nothing having crashed, so the recorded value is the whole point.
    """
    seen: list[tuple[str, ...]] = []
    original = getattr(core, name)

    def spy(frame, x, y, regression_features, fe_names=()):
        seen.append(fe_names)
        return original(frame, x, y, regression_features, fe_names)

    monkeypatch.setattr(core, name, spy)
    return seen


def dense_reference(df, num_bins, numeric, categorical):
    """Bin means from an explicit one-hot OLS, evaluated at the control means.

    Deliberately not the library's own machinery: this is the textbook design
    matrix, built and solved with :func:`numpy.linalg.lstsq`, which is what
    absorption claims to reproduce.
    """
    edges = np.unique(np.quantile(df["x"], np.linspace(0.0, 1.0, num_bins + 1)))
    num_bins = edges.size - 1
    idx = np.clip(np.searchsorted(edges, df["x"], side="right") - 1, 0, num_bins - 1)

    blocks = [np.eye(num_bins)[idx]]
    if numeric:
        blocks.append(df[list(numeric)].to_numpy(dtype=float))
    for column in categorical:
        codes, _ = pd.factorize(df[column])
        blocks.append(np.eye(codes.max() + 1)[codes][:, 1:])
    design = np.column_stack(blocks)

    beta, *_ = np.linalg.lstsq(design, df["y"].to_numpy(dtype=float), rcond=None)
    # Controls are evaluated at their sample means, matching the plotted curve.
    return beta[:num_bins] + design[:, num_bins:].mean(axis=0) @ beta[num_bins:]


# --------------------------------------------------------------------------
# 1. Equivalence: the gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_two_way_matches_one_hot(df_type, monkeypatch, force_multiway):
    """Absorbing both factors must reproduce encoding both.

    Sized for the one-hot side: 28 levels rather than the 85 the oracle tests use,
    which is the same comparison an order of magnitude cheaper.
    """
    df = make_two_way(n=1500, g1=20, g2=8)
    native = convert_to_backend(df, df_type)
    controls = ["age", "firm", "year"]

    absorbed = run_native(native, controls=controls, num_bins=8)

    without_absorption(monkeypatch)
    one_hot = run_native(native, controls=controls, num_bins=8)

    rtol, atol = tolerances(df_type)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_two_way_matches_dense_ols(df_type, force_multiway):
    """Independent oracle: a one-hot design solved with numpy."""
    df = make_two_way()
    native = convert_to_backend(df, df_type)

    absorbed = run_native(native, controls=["age", "firm", "year"], num_bins=8)
    expected = dense_reference(df, 8, ("age",), ("firm", "year"))

    # The oracle cuts bins on exact quantiles. dask cuts on approximate ones, so the
    # two fits are over genuinely different bins -- the same situation
    # ``reshaped_input`` covers elsewhere. The equivalence test above is the tight
    # check on dask, since both of its runs share the same edges.
    rtol, atol = tolerances(df_type, reshaped_input=True)
    np.testing.assert_allclose(absorbed, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_three_way_matches_dense_ols(df_type, force_multiway):
    """Three factors: the pairwise crosstabs grow, the answer must not move."""
    df = make_three_way()
    native = convert_to_backend(df, df_type)

    absorbed = run_native(native, controls=THREE_WAY_CONTROLS, num_bins=8)
    expected = dense_reference(df, 8, ("age",), THREE_WAY_CATS)

    # See the two-way case: the oracle cuts on exact quantiles, dask on approximate
    # ones, so only a loose comparison is meaningful there.
    rtol, atol = tolerances(df_type, reshaped_input=True)
    np.testing.assert_allclose(absorbed, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_three_way_matches_one_hot(df_type, monkeypatch, force_multiway):
    """Absorbing all three factors must reproduce encoding all three.

    Sized for the one-hot side, like the two-way case. Three factors still mean
    three pairwise crosstabs at 25 levels; only the encoding gets cheaper.
    """
    df = make_three_way(n=1500, g1=12, g2=8, g3=5)
    native = convert_to_backend(df, df_type)

    absorbed = run_native(native, controls=THREE_WAY_CONTROLS, num_bins=8)

    without_absorption(monkeypatch)
    one_hot = run_native(native, controls=THREE_WAY_CONTROLS, num_bins=8)

    rtol, atol = tolerances(df_type)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)


def test_two_way_without_numeric_controls(force_multiway):
    """The control block is empty here, so the bin and level blocks stand alone."""
    df = make_two_way()

    absorbed = run_native(df, controls=["firm", "year"], num_bins=8)
    expected = dense_reference(df, 8, (), ("firm", "year"))

    np.testing.assert_allclose(absorbed, expected, rtol=1e-9, atol=1e-9)


def test_nested_factors_match_dense_ols(force_multiway):
    """County nested in state: the crosstab is one-to-one in one direction.

    Nesting makes the joint design rank-deficient by more than the usual ``F``,
    which a least-squares solve has to absorb without the fitted values moving.
    """
    rng = np.random.default_rng(11)
    n = 3000
    county = rng.integers(0, 60, n)
    state = county // 10  # each county sits in exactly one state
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=2.0, size=60)[county] + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "county": [f"c{v:02d}" for v in county],
            "state": [f"s{v}" for v in state],
        }
    )

    absorbed = run_native(df, controls=["county", "state"], num_bins=6)
    expected = dense_reference(df, 6, (), ("county", "state"))

    np.testing.assert_allclose(absorbed, expected, rtol=1e-8, atol=1e-8)


def test_singleton_levels_match_dense_ols(force_multiway):
    """Levels seen exactly once are the degenerate case for a group-mean method."""
    df = make_two_way(n=1200, g1=200, g2=15, seed=7)

    absorbed = run_native(df, controls=["age", "firm", "year"], num_bins=5)
    expected = dense_reference(df, 5, ("age",), ("firm", "year"))

    np.testing.assert_allclose(absorbed, expected, rtol=1e-8, atol=1e-8)


def test_integer_coded_factors_via_categorical(force_multiway):
    """Integer identifiers only become factors when named in ``categorical=``."""
    df = make_two_way()
    df["firm_id"] = df["firm"].str[1:].astype(int)
    df["year_id"] = df["year"].str[1:].astype(int)

    absorbed = run_native(
        df,
        controls=["firm_id", "year_id"],
        categorical=["firm_id", "year_id"],
        num_bins=8,
    )
    expected = dense_reference(df, 8, (), ("firm_id", "year_id"))

    np.testing.assert_allclose(absorbed, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# 2. Routing
# --------------------------------------------------------------------------


def test_selects_every_factor_above_the_threshold():
    df = make_two_way(g1=60, g2=55)
    lazy, _, _, categorical = core.clean_df(
        df, ("firm", "year"), "x", "y", ("firm", "year")
    )
    assert fe_mod.select_absorbed_factors(lazy, categorical) == ("firm", "year")


def test_ranks_by_cardinality_and_drops_small_factors():
    """``year`` is below the threshold, so it stays on the one-hot path."""
    df = make_two_way(g1=60, g2=25)
    lazy, _, _, categorical = core.clean_df(
        df, ("firm", "year"), "x", "y", ("firm", "year")
    )
    assert fe_mod.select_absorbed_factors(lazy, categorical) == ("firm",)


def test_max_absorbed_caps_the_selection():
    df = make_two_way(g1=60, g2=55)
    lazy, _, _, categorical = core.clean_df(
        df, ("firm", "year"), "x", "y", ("firm", "year")
    )
    assert fe_mod.select_absorbed_factors(lazy, categorical, max_absorbed=1) == (
        "firm",
    )


def test_dpi_caps_absorption_at_one_factor(monkeypatch, force_multiway):
    """The DPI sandwich is one-way only, so DPI must never see two factors.

    Capping is safe because absorbing and encoding are the same estimator; this
    asserts the routing directly rather than by observing that nothing crashed.

    The cap keys off ``auto_bins == "dpi"`` alone, so the cardinalities are free to
    be small: whether both factors clear the production threshold is what
    :func:`test_selects_every_factor_above_the_threshold` is for. Lowering them here
    keeps the one-hot encoding of the *un*absorbed factor -- which is what this test
    would otherwise spend all its time on -- down to four dummy columns.
    """
    df = make_two_way(n=1500, g1=8, g2=5)
    seen = spy_on(monkeypatch, "_select_dpi_bins")

    binscatter(
        df,
        "x",
        "y",
        controls=["firm", "year"],
        num_bins="dpi",
        return_type="native",
    )

    assert seen == [("firm",)], "DPI must be handed at most one absorbed factor"


def test_rot_absorbs_every_factor(monkeypatch, force_multiway):
    """ROT reduces to the residual sum of squares, which the joint solve gives."""
    df = make_two_way(g1=60, g2=55)
    seen = spy_on(monkeypatch, "_select_rule_of_thumb_bins")

    out = binscatter(
        df,
        "x",
        "y",
        controls=["firm", "year"],
        num_bins="rot",
        return_type="native",
    )

    assert seen == [("firm", "year")]
    assert len(out) >= 2


def test_rot_bin_count_matches_the_one_hot_path(monkeypatch, force_multiway):
    """Absorbed and encoded are the same model, so ROT must pick the same count."""
    df = make_two_way(g1=20, g2=12)

    absorbed = core._select_rule_of_thumb_bins(
        *_rot_inputs(df, max_absorbed=None),
    )
    without_absorption(monkeypatch)
    one_hot = core._select_rule_of_thumb_bins(*_rot_inputs(df, max_absorbed=None))

    assert absorbed == one_hot


def _rot_inputs(df, max_absorbed):
    lazy, _, numeric, categorical = core.clean_df(df, ("age", "firm", "year"), "x", "y")
    frame, features, absorbed = core.add_regression_features(
        lazy,
        numeric_controls=numeric,
        categorical_controls=categorical,
        max_absorbed=max_absorbed,
    )
    return frame, "x", "y", features, absorbed


# --------------------------------------------------------------------------
# 3. Guards
# --------------------------------------------------------------------------


def test_crosstab_budget_names_the_offending_columns():
    """The worker-x-firm regime must fail with an explanation, not an allocation."""
    with pytest.raises(ValueError, match="crosstab entries"):
        sparse_mod._check_crosstab_budget(
            ("worker", "firm"),
            (1_000_000, 100_000),
            num_bins=10,
            total_count=50_000_000.0,
        )

    with pytest.raises(ValueError, match=r"worker \(1,000,000 levels\)"):
        sparse_mod._check_crosstab_budget(
            ("worker", "firm"),
            (1_000_000, 100_000),
            num_bins=10,
            total_count=50_000_000.0,
        )


def test_crosstab_budget_allows_the_firm_by_year_regime():
    """100k firms x 30 years is the case this method exists to serve."""
    sparse_mod._check_crosstab_budget(
        ("firm", "year"), (100_000, 30), num_bins=20, total_count=3_000_000.0
    )


def test_absorption_caps_at_one_factor_without_scipy(monkeypatch, force_multiway):
    """A missing scipy must cost speed, not results.

    Two high-cardinality categoricals absorbed one and one-hot encoded the other
    long before this module existed, and that route needs no sparse solver. So the
    fallback has to be that same route, not an ImportError.
    """
    df = make_two_way(n=1500, g1=12, g2=6, seed=4)
    controls = ["age", "firm", "year"]

    with_scipy = run_native(df, controls=controls, num_bins=6)

    monkeypatch.setattr(core, "sparse_available", lambda: False)
    without_scipy = run_native(df, controls=controls, num_bins=6)

    np.testing.assert_allclose(without_scipy, with_scipy, rtol=1e-9, atol=1e-9)


def test_scipy_fallback_leaves_one_factor_absorbed(monkeypatch, force_multiway):
    """The fallback is a cap, not a retreat to encoding everything.

    Small cardinalities for the same reason as the DPI cap: the fallback keys off
    ``sparse_available()`` alone, and the levels would only be paid for in the
    one-hot encoding of the factor that stays behind.
    """
    df = make_two_way(n=1500, g1=8, g2=5)
    monkeypatch.setattr(core, "sparse_available", lambda: False)
    seen: list[tuple[str, ...]] = []
    original = core.add_regression_features

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result[2])
        return result

    monkeypatch.setattr(core, "add_regression_features", spy)
    binscatter(
        df, "x", "y", controls=["firm", "year"], num_bins=6, return_type="native"
    )

    assert seen == [("firm",)], "one factor stays absorbed, the rest are encoded"


def test_require_sparse_message_names_the_extra():
    """If a future caller forgets to check availability, it must fail legibly."""
    assert "binscatter[multiway]" in sparse_mod._SCIPY_HINT


def test_confidence_intervals_name_every_absorbed_factor(force_multiway):
    df = make_two_way()
    with pytest.raises(NotImplementedError, match="'firm', 'year'"):
        binscatter(
            df,
            "x",
            "y",
            controls=["age", "firm", "year"],
            num_bins=8,
            ci="pointwise",
            return_type="native",
        )


def test_multi_fe_moments_rejects_a_single_factor():
    df = make_two_way(n=200, g1=5, g2=3)
    lazy, *_ = core.clean_df(df, ("firm",), "x", "y", ("firm",))
    with pytest.raises(ValueError, match="at least two factors"):
        sparse_mod.collect_multi_fe_moments(
            lazy,
            fe_names=("firm",),
            feature_names=(),
            response_exprs={"y": nw.col("y")},
        )


# --------------------------------------------------------------------------
# 4. Structure of the collected moments
# --------------------------------------------------------------------------


def test_collected_moments_reproduce_the_dense_blocks():
    """D'D, D'W and D'y must equal what a dense dummy matrix would give."""
    df = make_two_way(n=600, g1=8, g2=5, seed=2)
    lazy, _, _, _ = core.clean_df(df, ("age", "firm", "year"), "x", "y")

    moments = sparse_mod.collect_multi_fe_moments(
        lazy,
        fe_names=("firm", "year"),
        feature_names=("age",),
        response_exprs={"y": nw.col("y")},
    )

    firm_codes, firm_levels = pd.factorize(df["firm"], sort=True)
    year_codes, year_levels = pd.factorize(df["year"], sort=True)
    dense = np.column_stack(
        [
            np.eye(firm_levels.size)[firm_codes],
            np.eye(year_levels.size)[year_codes],
        ]
    )

    # The collected level order follows the backend's group_by, so compare after
    # sorting each factor's block back onto the factorize order.
    order = _level_order(moments, ("firm", "year"), lazy, (firm_levels, year_levels))

    np.testing.assert_allclose(moments.counts[order], dense.sum(axis=0))
    np.testing.assert_allclose(
        moments.dtd.toarray()[np.ix_(order, order)], dense.T @ dense
    )
    np.testing.assert_allclose(
        moments.dtw[order, 0], dense.T @ df["age"].to_numpy(dtype=float)
    )
    np.testing.assert_allclose(
        moments.response_sums["y"][order], dense.T @ df["y"].to_numpy(dtype=float)
    )
    assert moments.total_count == len(df)
    assert moments.cardinalities == (firm_levels.size, year_levels.size)


def _level_order(moments, fe_names, lazy, level_arrays):
    """Positions that reorder the stacked level axis onto sorted label order.

    ``group_by`` ordering is not guaranteed, so the collector's level order is
    whatever the backend produced. Rebuilding it from the same ``group_by`` lets the
    blocks be compared against a dense design built in ``factorize`` order.
    """
    order: list[int] = []
    for idx, (name, levels) in enumerate(zip(fe_names, level_arrays)):
        grouped = lazy.group_by(name).agg(nw.len().alias("__n")).collect()
        position = {
            label: moments.offsets[idx] + i
            for i, label in enumerate(grouped.get_column(name).to_list())
        }
        order.extend(position[label] for label in levels)
    return np.asarray(order, dtype=int)
