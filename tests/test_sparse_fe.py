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

#: Every backend, with PySpark opt-in behind ``--run-pyspark`` as elsewhere in the
#: suite. Only the three tests in section 1 are parametrized over this: what a backend
#: can get wrong is how the aggregates come back -- their order, their nulls, their
#: quantile edges -- and one equivalence check exercises the same ``group_by`` calls
#: that every other test would re-run at the same price. Argument shapes and factor
#: pathologies cannot see the backend at all, so they stay on pandas.
DF_TYPE_PARAMS = [
    pytest.param(df_type) for df_type in DF_BACKENDS if df_type != "pyspark"
]
if "pyspark" in DF_BACKENDS:
    DF_TYPE_PARAMS.append(pytest.param("pyspark", marks=pytest.mark.pyspark))

#: Backends that cut bins on exact quantiles, so a comparison against a reference
#: fitted on ``numpy.quantile`` edges is meaningful at full precision. dask and
#: PySpark approximate, which would put the two fits over genuinely different bins --
#: the oracle is simply not a useful check there, and asserting it at the 0.1 relative
#: tolerance that would be needed asserts almost nothing. Equivalence covers them.
EXACT_QUANTILE_BACKENDS = [
    df_type for df_type in DF_BACKENDS if df_type not in ("dask", "pyspark")
]


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


def make_call_panel(n=1200, seed=0):
    """One frame carrying every column the argument sweep needs.

    Building it once rather than per variant keeps the sweep to what it is actually
    varying -- the call -- and keeps the levels small, since each variant pays for
    the one-hot side of its own comparison.
    """
    df = make_two_way(n=n, g1=15, g2=8, seed=seed)
    rng = np.random.default_rng(seed + 1)
    df["region"] = [f"r{v}" for v in rng.integers(0, 5, n)]
    df["const"] = "same"
    df["firm_id"] = df["firm"].str[1:].astype(int)
    df["year_id"] = df["year"].str[1:].astype(int)
    return df


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


def assert_matches_one_hot(df, monkeypatch, rtol=1e-9, atol=1e-9, **kwargs):
    """Run ``binscatter`` absorbed and one-hot encoded, and compare the dots.

    This is the gate in one line: the two routes are the same estimator, so every
    test that varies an argument or the data is this call with something changed.
    """
    absorbed = run_native(df, **kwargs)
    without_absorption(monkeypatch)
    one_hot = run_native(df, **kwargs)

    assert len(absorbed) == len(one_hot)
    np.testing.assert_allclose(absorbed, one_hot, rtol=rtol, atol=atol)
    return absorbed


# --------------------------------------------------------------------------
# 1. The gate, on every backend
# --------------------------------------------------------------------------
# What a backend can get wrong is the aggregates: the order they come back in, how
# they treat nulls, and where they cut bins. That is three tests. Everything below
# this section runs on pandas, because nothing else in the module can see a backend.


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_matches_one_hot_on_every_backend(df_type, monkeypatch, force_multiway):
    """Absorbing both factors must reproduce encoding both, on each engine.

    Sized for the one-hot side: encoding ``G`` levels costs ``k(k+1)/2`` aggregations,
    so the levels drive the runtime quadratically while two factors are two factors at
    any cardinality. The oracle tests keep realistic ones -- they never encode.
    """
    native = convert_to_backend(make_two_way(n=1200, g1=10, g2=5), df_type)

    rtol, atol = tolerances(df_type)
    assert_matches_one_hot(
        native,
        monkeypatch,
        rtol=rtol,
        atol=atol,
        controls=["age", "firm", "year"],
        num_bins=8,
    )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_nulls_in_factor_columns_are_dropped(df_type, force_multiway):
    """A null in either factor drops the row from both crosstabs, not just its own."""
    df = make_two_way(n=1200, g1=15, g2=6, seed=41)
    df.loc[:39, "firm"] = None
    df.loc[60:99, "year"] = None
    clean = df.dropna(subset=["firm", "year"])

    with_nulls = run_native(
        convert_to_backend(df, df_type), controls=["firm", "year"], num_bins=5
    )
    dropped = run_native(
        convert_to_backend(clean, df_type), controls=["firm", "year"], num_bins=5
    )

    rtol, atol = tolerances(df_type, reshaped_input=True)
    np.testing.assert_allclose(with_nulls, dropped, rtol=rtol, atol=atol)


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_row_order_does_not_change_result(df_type, force_multiway):
    """Levels are stacked onto one axis by label, from F+1 separate ``group_by`` calls.

    Nothing guarantees those calls agree on an order, or keep one between runs, so a
    shuffle is the cheapest way to catch an axis that was mapped by position.
    """
    df = make_two_way(n=1200, g1=15, g2=6, seed=13)
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    controls = ["age", "firm", "year"]

    a = run_native(convert_to_backend(df, df_type), controls=controls, num_bins=5)
    b = run_native(convert_to_backend(shuffled, df_type), controls=controls, num_bins=5)

    rtol, atol = tolerances(df_type, reshaped_input=True)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------
# 2. The independent oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("df_type", EXACT_QUANTILE_BACKENDS)
@pytest.mark.parametrize("factors", [2, 3], ids=["two_way", "three_way"])
def test_matches_dense_ols(df_type, factors, force_multiway):
    """A one-hot design built and solved with numpy, not the library's machinery.

    Three factors as well as two because the pairwise crosstabs grow quadratically in
    the number of absorbed columns -- three factors mean three crosstabs, not one.
    """
    if factors == 2:
        df = make_two_way()
        controls, numeric, cats = ["age", "firm", "year"], ("age",), ("firm", "year")
    else:
        df = make_three_way()
        controls, numeric, cats = THREE_WAY_CONTROLS, ("age",), THREE_WAY_CATS

    absorbed = run_native(
        convert_to_backend(df, df_type), controls=controls, num_bins=8
    )
    expected = dense_reference(df, 8, numeric, cats)

    np.testing.assert_allclose(absorbed, expected, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# 3. The arguments ``binscatter()`` takes
# --------------------------------------------------------------------------
# Absorption sits behind a handful of the public arguments, and each of them reaches
# it differently: ``controls`` decides how many factors there are, ``categorical``
# decides what counts as one, and ``num_bins`` decides which selector runs and
# therefore how many get absorbed at all. Every one of these is the same estimator as
# the one-hot route, so every one is the same assertion.


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"controls": ["age", "firm", "year"]}, id="two_factors"),
        pytest.param(
            {"controls": ["age", "firm", "year", "region"]}, id="three_factors"
        ),
        pytest.param({"controls": ["firm", "year"]}, id="no_numeric_control"),
        pytest.param({"controls": ["age", "firm"]}, id="one_factor_no_sparse_path"),
        pytest.param(
            {"controls": ["age", "firm", "year", "const"]}, id="constant_factor"
        ),
        pytest.param(
            {
                "controls": ["firm_id", "year_id"],
                "categorical": ["firm_id", "year_id"],
            },
            id="integer_coded_via_categorical",
        ),
        pytest.param(
            {"controls": ["age", "firm", "year"], "num_bins": "rot"}, id="rot"
        ),
        pytest.param(
            {"controls": ["age", "firm", "year"], "num_bins": "dpi"}, id="dpi"
        ),
    ],
)
def test_call_variants_match_one_hot(kwargs, monkeypatch, force_multiway):
    """Vary one argument at a time; the dots must not care which route was taken."""
    kwargs = {"num_bins": 6, **kwargs}
    # The selectors reach the absorbed moments through a different path than a fixed
    # bin count does, and both round to an integer, so they get a looser tolerance.
    tol = 1e-8 if isinstance(kwargs["num_bins"], str) else 1e-9

    assert_matches_one_hot(make_call_panel(), monkeypatch, rtol=tol, atol=tol, **kwargs)


def test_poly_line_with_absorbed_factors(monkeypatch, force_multiway):
    """The overlay is fitted with no bin dummies at all, which is its own solve.

    ``_fit_polynomial_line`` has a separate multi-way branch that hands
    ``solve_absorbed_system`` empty bin blocks, so the plotted line goes through code
    the dot tests never reach. It is also the one absorbed quantity a user sees that
    is not a dot, which is why it is asserted on the figure rather than the frame.
    """
    df = make_call_panel()
    controls = ["age", "firm", "year"]

    absorbed = binscatter(df, "x", "y", controls=controls, num_bins=6, poly_line=2)

    without_absorption(monkeypatch)
    one_hot = binscatter(df, "x", "y", controls=controls, num_bins=6, poly_line=2)

    np.testing.assert_allclose(
        absorbed.data[1].y, one_hot.data[1].y, rtol=1e-8, atol=1e-8
    )


# --------------------------------------------------------------------------
# 4. What the factors themselves can look like
# --------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(lambda i: f"g{i}", id="string"),
        pytest.param(lambda i: i, id="int"),
        pytest.param(lambda i: f"grüße-{i}/ü", id="unicode"),
    ],
)
def test_factor_label_types(labels, force_multiway):
    """Labels index the stacked level axis, so exotic ones must round-trip."""
    df = make_two_way(n=900, g1=8, g2=5, seed=5)
    df["firm"] = [labels(int(v[1:])) for v in df["firm"]]
    df["year"] = [labels(int(v[1:])) for v in df["year"]]

    kwargs = {"controls": ["age", "firm", "year"], "num_bins": 4}
    if not isinstance(labels(0), str):
        kwargs["categorical"] = ["firm", "year"]

    out = run_native(df, **kwargs)
    expected = dense_reference(df, 4, ("age",), ("firm", "year"))

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-8)


def test_factors_sharing_label_values_stay_separate(force_multiway):
    """``firm`` and ``year`` both labelled ``a, b, c`` must not collide.

    Every level lands in one parameter vector, addressed by ``offsets[i] + index``,
    and each factor carries its own label index. A shared index would silently merge
    ``firm == "a"`` with ``year == "a"`` into one effect -- no error, just wrong dots.
    This is the failure mode that only exists once there are two factors.
    """
    df = make_two_way(n=1200, g1=8, g2=5, seed=23)
    shared = list("abcdefgh")
    df["firm"] = [shared[int(v[1:])] for v in df["firm"]]
    df["year"] = [shared[int(v[1:])] for v in df["year"]]

    absorbed = run_native(df, controls=["age", "firm", "year"], num_bins=5)
    expected = dense_reference(df, 5, ("age",), ("firm", "year"))

    np.testing.assert_allclose(absorbed, expected, rtol=1e-8, atol=1e-8)


def test_level_confined_to_a_single_bin(force_multiway):
    """A level appearing in one bin only makes its row of ``D'B`` a single entry.

    That is the sparsest the incidence blocks get, and the point where a solve can
    quietly attribute the bin effect to the level instead.
    """
    df = make_two_way(n=1200, g1=10, g2=5, seed=19)
    # Give one firm the largest x values outright, so it lives in the top bin alone.
    tail = np.argsort(df["x"].to_numpy())[-60:]
    df.loc[df.index[tail], "firm"] = "solo"

    absorbed = run_native(df, controls=["age", "firm", "year"], num_bins=5)
    expected = dense_reference(df, 5, ("age",), ("firm", "year"))

    np.testing.assert_allclose(absorbed, expected, rtol=1e-7, atol=1e-7)


# --------------------------------------------------------------------------
# 5. Routing
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


def test_rot_absorbs_every_factor(monkeypatch, force_multiway):
    """ROT must take the joint solve, not quietly settle for one factor.

    The output cannot show this: absorbing one factor and encoding the other is the
    same estimator, so :func:`test_rot_selector_matches_one_hot` would pass either
    way. What would change is the cost, silently, which is the whole point of the
    multi-way branch -- so this is asserted on the call rather than on the dots.
    """
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


# --------------------------------------------------------------------------
# 6. Guards
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


def test_crosstab_budget_reaches_the_caller(monkeypatch, force_multiway):
    """The budget has to stop a real call, not just answer when asked directly.

    :func:`test_crosstab_budget_names_the_offending_columns` pins the arithmetic and
    the message; this pins that the check is still wired into the path a user takes.
    """
    monkeypatch.setattr(sparse_mod, "MAX_CROSSTAB_ENTRIES", 10)
    df = make_two_way(n=500, g1=20, g2=10)

    with pytest.raises(ValueError, match="crosstab entries"):
        binscatter(
            df,
            "x",
            "y",
            controls=["firm", "year"],
            num_bins=5,
            return_type="native",
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
# 7. Structure of the collected moments
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
