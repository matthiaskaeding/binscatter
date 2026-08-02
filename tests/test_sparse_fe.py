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
from tests.test_fixed_effects import (
    assert_complete_case_rows,
    run_native,
    tolerances,
    without_absorption,
)

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


def make_bridge_panel(*, bridge_count: int):
    """Two dense panel components joined by zero, one, or two factor-axis bridges.

    One cross-component row makes the ordinary incidence graph connected but leaves
    a nonconstant bin contrast aliased with the two factor blocks. A second bridge
    through the other factor removes that last harmful null direction.
    """
    if bridge_count not in (0, 1, 2):
        raise ValueError("bridge_count must be zero, one, or two")
    rng = np.random.default_rng(20260803)
    n, num_bins = 1000, 4
    x = np.linspace(-2.0, 2.0, n)
    edges = pd.Series(x).quantile([idx / num_bins for idx in range(num_bins + 1)])
    bin_idx = np.clip(
        np.searchsorted(edges.to_numpy(), x, side="right") - 1, 0, num_bins - 1
    )
    low = np.flatnonzero(bin_idx < 2)
    high = np.flatnonzero(bin_idx >= 2)

    firm = np.empty(n, dtype=int)
    year = np.empty(n, dtype=int)
    firm[low] = rng.integers(0, 30, low.size)
    year[low] = rng.integers(0, 27, low.size)
    firm[high] = rng.integers(30, 60, high.size)
    year[high] = rng.integers(27, 55, high.size)
    # Guarantee every level is represented before inserting the bridges.
    firm[low[:30]] = np.arange(30)
    year[low[30:57]] = np.arange(27)
    firm[high[:30]] = 30 + np.arange(30)
    year[high[30:58]] = 27 + np.arange(28)

    if bridge_count >= 1:
        # This joins the incidence graph through ``firm`` but does not identify the
        # remaining low-versus-high bin contrast in the full additive design.
        firm[high[100]] = 0
    if bridge_count == 2:
        year[low[100]] = 27

    bin_effect = np.array([-2.0, -0.5, 1.0, 3.0])
    firm_effect = rng.normal(scale=2.0, size=60)
    year_effect = rng.normal(scale=1.0, size=55)
    y = bin_effect[bin_idx] + firm_effect[firm] + year_effect[year]
    expected = bin_effect + firm_effect[firm].mean() + year_effect[year].mean()
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "firm": [f"f{value:02d}" for value in firm],
            "year": [f"y{value:02d}" for value in year],
            "bin_control": bin_idx.astype(float),
        }
    )
    return frame, expected


def make_known_dgp(route: str):
    """Noiseless, imbalanced DGP with a known sample-average adjusted curve."""
    rng = np.random.default_rng(20260802)
    n, num_bins = 6000, 8
    firm_prob = np.r_[0.82, np.repeat(0.18 / 59.0, 59)]
    year_prob = np.r_[0.55, np.repeat(0.45 / 54.0, 54)]
    firm = rng.choice(60, n, p=firm_prob)
    year = rng.choice(55, n, p=year_prob)
    firm[:60] = np.arange(60)
    year[60:115] = np.arange(55)

    firm_effect = rng.normal(scale=3.0, size=60)
    year_effect = rng.normal(scale=2.0, size=55)
    z = 0.7 * firm_effect[firm] - 0.4 * year_effect[year] + rng.normal(size=n)
    x = 0.8 * firm_effect[firm] + 0.5 * year_effect[year] + 0.6 * z + rng.normal(size=n)
    edges = np.quantile(x, np.linspace(0.0, 1.0, num_bins + 1))
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, num_bins - 1)
    bin_effect = np.array([-2.0, -1.3, -0.5, 0.1, 0.8, 1.7, 2.9, 4.2])

    y = bin_effect[bin_idx] + 1.75 * z
    expected = bin_effect + 1.75 * z.mean()
    if route in ("one_way", "two_way"):
        y = y + firm_effect[firm]
        expected = expected + firm_effect[firm].mean()
    if route == "two_way":
        y = y + year_effect[year]
        expected = expected + year_effect[year].mean()

    options = {
        "no_fe": (["z"], []),
        "one_way": (["z", "firm"], ["firm"]),
        "two_way": (["z", "firm", "year"], ["firm", "year"]),
    }
    if route not in options:
        raise ValueError(f"unknown DGP route: {route}")
    controls, categorical = options[route]
    frame = pd.DataFrame({"x": x, "y": y, "z": z, "firm": firm, "year": year})
    return frame, expected, controls, categorical, num_bins


def production_oracle_case(factors: int):
    """A production-threshold panel and its numeric/categorical control metadata."""
    if factors == 2:
        return (
            make_two_way(g1=60, g2=55),
            ["age", "firm", "year"],
            ("age",),
            ("firm", "year"),
        )
    if factors == 3:
        return (
            make_three_way(g1=60, g2=55, g3=50),
            THREE_WAY_CONTROLS,
            ("age",),
            THREE_WAY_CATS,
        )
    raise ValueError(f"unsupported factor count: {factors}")


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


@pytest.mark.quick
@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_matches_one_hot_on_every_backend(df_type, monkeypatch, force_multiway):
    """Absorbing both factors must reproduce encoding both, on each engine.

    Sized for the one-hot side: encoding ``G`` levels costs ``k(k+1)/2`` aggregations,
    so the levels drive the runtime quadratically while two factors are two factors at
    any cardinality. The oracle tests keep realistic ones -- they never encode.
    """
    native = convert_to_backend(make_two_way(n=1200, g1=10, g2=5), df_type)
    if df_type == "pyspark":
        native = native.repartition(3)

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
def test_integer_coded_factors_match_one_hot_on_every_backend(
    df_type, monkeypatch, force_multiway
):
    """Explicit integer factor declarations are backend-independent."""
    frame = make_two_way(n=1200, g1=10, g2=5, seed=12)
    frame["firm_id"] = pd.factorize(frame["firm"], sort=True)[0]
    frame["year_id"] = pd.factorize(frame["year"], sort=True)[0]
    native = convert_to_backend(frame, df_type)
    if df_type == "pyspark":
        native = native.repartition(3)
    rtol, atol = tolerances(df_type)
    assert_matches_one_hot(
        native,
        monkeypatch,
        rtol=rtol,
        atol=atol,
        controls=["age", "firm_id", "year_id"],
        categorical=["firm_id", "year_id"],
        num_bins=8,
    )


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
def test_multiway_native_output_preserves_input_backend(df_type, force_multiway):
    """Multiway controlled output stays native to the input dataframe backend."""
    native = convert_to_backend(make_two_way(n=1200, g1=10, g2=5, seed=14), df_type)
    if df_type == "pyspark":
        native = native.repartition(3)
    result = binscatter(
        native,
        "x",
        "y",
        controls=["age", "firm", "year"],
        num_bins=8,
        return_type="native",
    )
    assert isinstance(result, type(native))


@pytest.mark.parametrize("df_type", DF_TYPE_PARAMS)
@pytest.mark.parametrize(
    "invalid_column",
    ["x", "y", "age", "firm", "year"],
    ids=[
        "x_nan",
        "y_infinity",
        "control_infinity",
        "first_factor_null",
        "second_factor_null",
    ],
)
def test_invalid_rows_are_dropped_before_multiway_absorption(
    df_type, invalid_column, monkeypatch, force_multiway
):
    """Exactly the complete cases, not merely similar dots, reach absorption."""
    df = make_two_way(n=1200, g1=15, g2=6, seed=41)
    invalid_value = None if invalid_column in ("firm", "year") else np.inf
    if invalid_column == "x":
        invalid_value = np.nan
    df.loc[:39, invalid_column] = invalid_value
    clean = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["x", "y", "age", "firm", "year"])
        .reset_index(drop=True)
    )

    seen: list[int] = []
    original = core.partial_out_controls

    def checked_partial_out(frame, profile, *args, **kwargs):
        assert_complete_case_rows(frame, clean, ("x", "y", "age", "firm", "year"))
        seen.append(len(clean))
        return original(frame, profile, *args, **kwargs)

    monkeypatch.setattr(core, "partial_out_controls", checked_partial_out)
    run_native(
        convert_to_backend(df, df_type),
        controls=["age", "firm", "year"],
        num_bins=5,
    )
    assert seen == [len(clean)]


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


@pytest.mark.quick
@pytest.mark.parametrize(
    "route", ["no_fe", "one_way", "two_way"], ids=["no_fe", "one_fe", "two_fe"]
)
def test_known_dgp_recovers_sample_average_curve(route):
    """Exact ground truth under severe imbalance and control/x confounding.

    Parametrizing the routes makes the non-FE, one-FE, and two-FE specifications
    report independently instead of the first failure hiding the others.
    """
    df, expected, controls, categorical, num_bins = make_known_dgp(route)
    actual = run_native(
        df, controls=controls, categorical=categorical, num_bins=num_bins
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("route", ["one_way", "two_way"], ids=["one_fe", "two_fe"])
def test_known_dgp_is_invariant_to_control_order(route):
    """Control and factor order is not part of the statistical specification."""
    df, _, controls, categorical, num_bins = make_known_dgp(route)
    baseline = run_native(
        df, controls=controls, categorical=categorical, num_bins=num_bins
    )
    permuted = run_native(
        df,
        controls=list(reversed(controls)),
        categorical=list(reversed(categorical)),
        num_bins=num_bins,
    )
    np.testing.assert_allclose(permuted, baseline, rtol=1e-10, atol=1e-10)


def test_known_two_way_dgp_is_invariant_to_row_replication():
    """Duplicating every observation changes weights by a common factor only."""
    df, _, controls, categorical, num_bins = make_known_dgp("two_way")
    baseline = run_native(
        df, controls=controls, categorical=categorical, num_bins=num_bins
    )
    replicated = run_native(
        pd.concat([df, df], ignore_index=True),
        controls=controls,
        categorical=categorical,
        num_bins=num_bins,
    )
    np.testing.assert_allclose(replicated, baseline, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("route", ["one_way", "two_way"], ids=["one_fe", "two_fe"])
@pytest.mark.parametrize("scale", [1e-170, 1e200], ids=["tiny", "huge"])
def test_fixed_effect_dgp_is_invariant_to_numeric_control_units(route, scale):
    """Absorbed estimators remain the same under extreme finite control units."""
    frame, expected, controls, categorical, num_bins = make_known_dgp(route)
    frame["z"] = scale * frame["z"]
    actual = run_native(
        frame, controls=controls, categorical=categorical, num_bins=num_bins
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("route", ["one_way", "two_way"], ids=["one_fe", "two_fe"])
def test_fixed_effect_dgp_handles_near_collinear_numeric_controls(route):
    """A small identified control direction survives one- and multiway absorption."""
    frame, expected, controls, categorical, num_bins = make_known_dgp(route)
    noise = np.random.default_rng(20260804).normal(size=len(frame))
    frame["z2"] = frame["z"] + 1e-8 * noise
    frame["y"] = frame["y"] + 0.2 * noise
    controls = ["z", "z2", *controls[1:]]
    actual = run_native(
        frame, controls=controls, categorical=categorical, num_bins=num_bins
    )
    np.testing.assert_allclose(
        actual, expected + 0.2 * noise.mean(), rtol=1e-7, atol=1e-7
    )


@pytest.mark.parametrize("route", ["one_way", "two_way"], ids=["one_fe", "two_fe"])
def test_fixed_effect_dgp_handles_large_int64_control_products(route):
    """FE moment aggregation casts integer cross-products before they overflow."""
    frame, expected, controls, categorical, num_bins = make_known_dgp(route)
    large_int = np.random.default_rng(20260805).integers(
        -5_000_000_000,
        5_000_000_000,
        size=len(frame),
        dtype=np.int64,
    )
    frame["large_int"] = large_int
    frame["y"] = frame["y"] + 1e-9 * large_int
    controls = ["z", "large_int", *controls[1:]]
    actual = run_native(
        frame, controls=controls, categorical=categorical, num_bins=num_bins
    )
    np.testing.assert_allclose(
        actual, expected + 1e-9 * large_int.mean(), rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("route", ["one_way", "two_way"], ids=["one_fe", "two_fe"])
def test_fixed_effect_dgp_handles_large_int64_response_sums(route):
    """FE response aggregates may exceed int64 even when every row is valid."""
    frame, _, controls, categorical, num_bins = make_known_dgp(route)
    rng = np.random.default_rng(20260806)
    z = rng.integers(-1000, 1000, size=len(frame), dtype=np.int64)
    edges = (
        frame["x"].quantile([idx / num_bins for idx in range(num_bins + 1)]).to_numpy()
    )
    bin_idx = np.clip(
        np.searchsorted(edges, frame["x"], side="right") - 1,
        0,
        num_bins - 1,
    )
    bin_effect = 1000 * np.array([-2, -1, 0, 1, 2, 3, 5, 8], dtype=np.int64)
    firm_effect = rng.integers(-1000, 1000, size=60, dtype=np.int64)
    year_effect = rng.integers(-1000, 1000, size=55, dtype=np.int64)
    firm = frame["firm"].to_numpy(dtype=int)
    year = frame["year"].to_numpy(dtype=int)

    base = 8_000_000_000_000_000
    y = base + bin_effect[bin_idx] + 3 * z
    expected = base + bin_effect.astype(float) + 3.0 * z.mean()
    if route in ("one_way", "two_way"):
        y = y + firm_effect[firm]
        expected = expected + firm_effect[firm].mean()
    if route == "two_way":
        y = y + year_effect[year]
        expected = expected + year_effect[year].mean()

    frame["z"] = z
    frame["y"] = y.astype(np.int64)
    actual = run_native(
        frame, controls=controls, categorical=categorical, num_bins=num_bins
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=4.0)


@pytest.mark.parametrize("route", ["one_way", "two_way"], ids=["one_fe", "two_fe"])
@pytest.mark.parametrize("redundancy", ["duplicate", "constant", "fe_span"])
def test_fixed_effect_dgp_ignores_harmless_control_rank_deficiency(route, redundancy):
    """Only rank loss that aliases the bin curve is an identification failure."""
    frame, expected, controls, categorical, num_bins = make_known_dgp(route)
    if redundancy == "duplicate":
        frame["redundant"] = frame["z"]
    elif redundancy == "constant":
        frame["redundant"] = 1.0
    else:
        # This numeric column is already exactly spanned by the firm effects.
        frame["redundant"] = pd.factorize(frame["firm"], sort=True)[0]
    controls = ["z", "redundant", *controls[1:]]
    actual = run_native(
        frame, controls=controls, categorical=categorical, num_bins=num_bins
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.quick
@pytest.mark.parametrize("df_type", EXACT_QUANTILE_BACKENDS)
@pytest.mark.parametrize("factors", [2, 3], ids=["two_way", "three_way"])
def test_matches_dense_ols(df_type, factors):
    """A one-hot design built and solved with numpy, not the library's machinery.

    Three factors as well as two because the pairwise crosstabs grow quadratically in
    the number of absorbed columns -- three factors mean three crosstabs, not one.
    """
    df, controls, numeric, cats = production_oracle_case(factors)
    absorbed = run_native(
        convert_to_backend(df, df_type), controls=controls, num_bins=8
    )
    expected = dense_reference(df, 8, numeric, cats)

    np.testing.assert_allclose(absorbed, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("original_name", "control_name"),
    [("firm", "__mfe_resp_0"), ("year", "__mfe_count")],
    ids=["response_alias", "count_alias"],
)
def test_multiway_temporary_aliases_do_not_shadow_control_columns(
    original_name, control_name
):
    """Multiway aggregation temporaries cannot reserve valid user column names."""
    df, controls, _, _ = production_oracle_case(2)
    expected = run_native(df, controls=controls, num_bins=8)
    renamed = df.rename(columns={original_name: control_name})
    renamed_controls = [
        control_name if name == original_name else name for name in controls
    ]
    actual = run_native(
        renamed,
        controls=renamed_controls,
        num_bins=8,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("df_type", EXACT_QUANTILE_BACKENDS)
@pytest.mark.parametrize("factors", [2, 3], ids=["two_way", "three_way"])
def test_additive_factor_shifts_recover_their_sample_average(df_type, factors):
    """Changing FE levels moves every dot by the count-weighted average shift."""
    df, controls, _, cats = production_oracle_case(factors)
    baseline = run_native(
        convert_to_backend(df, df_type), controls=controls, num_bins=8
    )
    # Arbitrary additive effects in every absorbed factor must change only the
    # count-weighted overall level.  This catches a solve that gets within-bin
    # contrasts right but forgets (or misweights) fixed-effect level recovery.
    shifted = df.copy()
    total_shift = np.zeros(len(df), dtype=float)
    for factor_idx, name in enumerate(cats, start=1):
        codes, _ = pd.factorize(df[name], sort=True)
        total_shift += np.sin((codes + 1) * factor_idx) * (3.0 + factor_idx)
    shifted["y"] = shifted["y"] + total_shift
    shifted_dots = run_native(
        convert_to_backend(shifted, df_type), controls=controls, num_bins=8
    )
    np.testing.assert_allclose(
        shifted_dots,
        baseline + total_shift.mean(),
        rtol=1e-8,
        atol=1e-8,
    )


# --------------------------------------------------------------------------
# 3. The arguments ``binscatter()`` takes
# --------------------------------------------------------------------------
# Absorption sits behind a handful of the public arguments, and each of them reaches
# it differently: ``controls`` decides how many factors there are, ``categorical``
# decides what counts as one, and ``num_bins`` decides which selector runs and
# therefore how many get absorbed at all. Every one of these is the same estimator as
# the one-hot route, so every one is the same assertion.


@pytest.mark.quick
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


@pytest.mark.quick
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


def test_multiway_rejects_disconnected_bin_factor_components():
    """No normalization can identify relative dots across disconnected components."""
    disconnected, _ = make_bridge_panel(bridge_count=0)
    with pytest.raises(ValueError, match="not identif|collinear with .*bin indicators"):
        run_native(disconnected, controls=["firm", "year"], num_bins=4)


def test_multiway_rejects_connected_but_unidentified_design():
    """Graph connectivity alone is insufficient with bins plus two FE blocks."""
    one_bridge, _ = make_bridge_panel(bridge_count=1)
    with pytest.raises(ValueError, match="not identif|collinear with .*bin indicators"):
        run_native(one_bridge, controls=["firm", "year"], num_bins=4)


def test_independently_bridged_multiway_dgp_recovers_known_curve():
    """Two factor-axis bridges identify the noiseless sample-average curve."""
    identified, expected = make_bridge_panel(bridge_count=2)
    actual = run_native(identified, controls=["firm", "year"], num_bins=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_independently_bridged_multiway_design_matches_dense_ols():
    """The boundary identified case also agrees with an explicit dummy design."""
    identified, _ = make_bridge_panel(bridge_count=2)
    actual = run_native(identified, controls=["firm", "year"], num_bins=4)
    expected = dense_reference(identified, 4, (), ("firm", "year"))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_multiway_rejects_numeric_control_collinear_with_bins():
    """Dots at average controls are not estimable when a control reproduces bins."""
    identified, _ = make_bridge_panel(bridge_count=2)
    with pytest.raises(ValueError, match="not identif|collinear with .*bin indicators"):
        run_native(
            identified,
            controls=["bin_control", "firm", "year"],
            num_bins=4,
        )


def test_crosstab_budget_error_explains_the_limit():
    """The worker-by-firm regime fails before attempting an oversized allocation."""
    with pytest.raises(ValueError, match="crosstab entries"):
        sparse_mod._check_crosstab_budget(
            ("worker", "firm"),
            (1_000_000, 100_000),
            num_bins=10,
            total_count=50_000_000.0,
        )


def test_crosstab_budget_error_names_the_offending_columns():
    """The remediation message identifies the high-cardinality factor pair."""
    with pytest.raises(ValueError, match=r"worker \(1,000,000 levels\)"):
        sparse_mod._check_crosstab_budget(
            ("worker", "firm"),
            (1_000_000, 100_000),
            num_bins=10,
            total_count=50_000_000.0,
        )


def test_crosstab_budget_reaches_the_caller(monkeypatch, force_multiway):
    """The budget has to stop a real call, not just answer when asked directly.

    The direct guard tests pin the arithmetic and message; this pins that the check
    is still wired into the path a user takes.
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


@pytest.mark.parametrize(
    "block",
    ["counts", "dtd", "dtw", "dty"],
    ids=["level_counts", "factor_cross_product", "control_sums", "response_sums"],
)
def test_collected_moment_block_matches_dense_design(block):
    """Every collected normal-equation block is an independent dense oracle."""
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

    if block == "counts":
        np.testing.assert_allclose(moments.counts[order], dense.sum(axis=0))
    elif block == "dtd":
        np.testing.assert_allclose(
            moments.dtd.toarray()[np.ix_(order, order)], dense.T @ dense
        )
    elif block == "dtw":
        np.testing.assert_allclose(
            moments.dtw[order, 0], dense.T @ df["age"].to_numpy(dtype=float)
        )
    else:
        np.testing.assert_allclose(
            moments.response_sums["y"][order],
            dense.T @ df["y"].to_numpy(dtype=float),
        )


def test_collected_moment_metadata_matches_dense_design():
    """Moment metadata reports the observation count and both factor cardinalities."""
    df = make_two_way(n=600, g1=8, g2=5, seed=2)
    lazy, _, _, _ = core.clean_df(df, ("age", "firm", "year"), "x", "y")
    moments = sparse_mod.collect_multi_fe_moments(
        lazy,
        fe_names=("firm", "year"),
        feature_names=("age",),
        response_exprs={"y": nw.col("y")},
    )
    firm_levels = pd.unique(df["firm"])
    year_levels = pd.unique(df["year"])
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
