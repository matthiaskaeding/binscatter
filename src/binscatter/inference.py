"""Confidence intervals for binscatter dots, following Cattaneo et al. (2024).

Two interval styles are supported, matching what ``binsreg`` reports:

``pointwise``
    The heteroskedasticity-robust interval for the binscatter estimator itself,
    i.e. for the piecewise-constant (``p=0``) fit that the dots show. It answers
    "how precisely is this bin's mean estimated", and is centred on the dot.

``rbc``
    The robust bias-corrected interval. The ``p=0`` estimator is biased for the
    underlying regression function because it is flat within a bin, and that bias
    does not vanish fast enough to be ignored. The correction is to build the
    interval from the next-order fit (``p+1=1``, ``s+1=1``: a continuous linear
    spline on the same bins) evaluated at the same point. The result is a valid
    interval for the true conditional mean, but it is *not* centred on the dot --
    the gap between them is the estimated bias.

Both use HC1 standard errors, and both evaluate controls at their sample means so
the interval refers to the same curve the dots trace.

Everything here is computed from bin-level aggregates: two passes over the data,
neither of which materializes a per-row design matrix or residual vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal

import narwhals as nw
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from .core import Profile

logger = logging.getLogger(__name__)

CIKind = Literal["pointwise", "rbc"]

#: Number of basis columns each style spends per bin boundary. ``pointwise`` uses
#: one dummy per bin; ``rbc`` uses a hat function per knot, so it has one more.
_BASIS_SIZE = {"pointwise": 0, "rbc": 1}


@dataclass(frozen=True)
class IntervalResult:
    """Per-bin interval bounds, in bin order."""

    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    std_error: np.ndarray
    kind: str


def _z_value(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("ci_level must lie strictly between 0 and 1")
    return NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)


def _basis_terms(
    kind: CIKind, edges: np.ndarray, j: int
) -> tuple[tuple[int, float, float], ...]:
    """Basis columns active in bin ``j``, as ``(column, const, x_coefficient)``.

    Within a bin every basis function is affine in ``x``, so a column is fully
    described by the pair ``(A, B)`` with value ``A + B*x``. That is what lets the
    same assembly code build both designs, and both the bread and the meat of the
    sandwich.
    """
    if kind == "pointwise":
        return ((j, 1.0, 0.0),)
    a = float(edges[j])
    b = float(edges[j + 1])
    h = b - a
    if not np.isfinite(h) or h <= 0.0:
        msg = (
            "Bin edges must be finite and strictly increasing to build robust "
            "bias-corrected intervals."
        )
        raise ValueError(msg)
    # (b - x)/h on the left knot, (x - a)/h on the right knot.
    return ((j, b / h, -1.0 / h), (j + 1, -a / h, 1.0 / h))


def _assemble(
    kind: CIKind,
    edges: np.ndarray,
    num_bins: int,
    num_controls: int,
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    """Build a symmetric ``X' diag(weight) X`` block matrix from bin aggregates.

    ``m0/m1/m2`` are per-bin sums of ``weight * x**0/1/2``, ``p0/p1`` per-bin sums
    of ``weight * control * x**0/1``, and ``cross`` the global control-by-control
    sums of ``weight * control * control``. With ``weight = 1`` this is the bread;
    with ``weight = residual**2`` it is the meat.
    """
    n_basis = num_bins + _BASIS_SIZE[kind]
    size = n_basis + num_controls
    out = np.zeros((size, size), dtype=float)

    for j in range(num_bins):
        terms = _basis_terms(kind, edges, j)
        for col_p, a_p, b_p in terms:
            for col_q, a_q, b_q in terms:
                out[col_p, col_q] += (
                    a_p * a_q * m0[j]
                    + (a_p * b_q + b_p * a_q) * m1[j]
                    + b_p * b_q * m2[j]
                )
            if num_controls:
                block = a_p * p0[j] + b_p * p1[j]
                out[col_p, n_basis:] += block
                out[n_basis:, col_p] += block

    if num_controls:
        out[n_basis:, n_basis:] = cross
    return out


def _assemble_rhs(
    kind: CIKind,
    edges: np.ndarray,
    num_bins: int,
    num_controls: int,
    r0: np.ndarray,
    r1: np.ndarray,
    wy: np.ndarray,
) -> np.ndarray:
    """Build ``X'y`` from per-bin sums of ``y`` and ``x*y`` plus the control block."""
    n_basis = num_bins + _BASIS_SIZE[kind]
    out = np.zeros(n_basis + num_controls, dtype=float)
    for j in range(num_bins):
        for col_p, a_p, b_p in _basis_terms(kind, edges, j):
            out[col_p] += a_p * r0[j] + b_p * r1[j]
    if num_controls:
        out[n_basis:] = wy
    return out


def _evaluation_vectors(
    kind: CIKind,
    edges: np.ndarray,
    num_bins: int,
    eval_x: np.ndarray,
    control_means: np.ndarray,
) -> np.ndarray:
    """Rows of the ``(J, size)`` matrix whose ``j``-th row evaluates the fit in bin ``j``.

    Controls enter at their sample means, so the evaluated curve is the same one
    the dots trace, and the interval inherits the controls' estimation noise.
    """
    num_controls = control_means.size
    n_basis = num_bins + _BASIS_SIZE[kind]
    g = np.zeros((num_bins, n_basis + num_controls), dtype=float)
    for j in range(num_bins):
        for col_p, a_p, b_p in _basis_terms(kind, edges, j):
            g[j, col_p] += a_p + b_p * float(eval_x[j])
        if num_controls:
            g[j, n_basis:] = control_means
    return g


def _fitted_curve_expr(
    kind: CIKind, x_name: str, edges: np.ndarray, coefficients: np.ndarray
) -> nw.Expr:
    """Express the fitted bin curve as a flat expression in ``x``.

    Rewriting the piecewise fit in a knot basis -- indicators for the step
    function, hinges for the spline -- keeps the residual pass to a single
    expression over ``x``. Joining the coefficients back on the bin column, or
    nesting ``when/then`` per bin, would work on some backends and fall over on
    others.
    """
    x_col = nw.col(x_name)
    num_bins = len(edges) - 1
    if kind == "pointwise":
        # beta_0 + sum_k (beta_k - beta_{k-1}) * 1{x >= edge_k}
        expr = nw.lit(float(coefficients[0]))
        for k in range(1, num_bins):
            step = float(coefficients[k] - coefficients[k - 1])
            expr = expr + nw.lit(step) * (x_col >= float(edges[k])).cast(nw.Float64)
        return expr

    # Continuous linear spline: intercept + slope*x + sum_k dslope_k * max(x - e_k, 0),
    # with the leading slope and each slope change read off the knot values.
    slopes = np.diff(coefficients) / np.diff(edges)
    expr = (
        nw.lit(float(coefficients[0] - slopes[0] * edges[0]))
        + nw.lit(float(slopes[0])) * x_col
    )
    for k in range(1, num_bins):
        change = float(slopes[k] - slopes[k - 1])
        if change == 0.0:
            continue
        expr = expr + nw.lit(change) * (x_col - float(edges[k])).clip(lower_bound=0.0)
    return expr


def _control_expr(control_names: tuple[str, ...], gamma: np.ndarray) -> nw.Expr | None:
    if not control_names:
        return None
    expr = nw.lit(float(gamma[0])) * nw.col(control_names[0])
    for name, coef in zip(control_names[1:], gamma[1:]):
        expr = expr + nw.lit(float(coef)) * nw.col(name)
    return expr


def _solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def compute_intervals(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    kind: CIKind,
    level: float,
) -> IntervalResult:
    """Compute per-bin confidence intervals for the binscatter dots.

    Returns bounds in bin order, matching the row order of the plotting frame.
    """
    if profile.fe_name is not None:
        msg = (
            f"Confidence intervals are not available when a categorical control is "
            f"absorbed as a fixed effect (here: '{profile.fe_name}'). The interval "
            "needs the fixed effect's contribution to the sandwich variance, which "
            "absorption deliberately never forms. Reduce the cardinality of that "
            "control, or drop it, to use ci=."
        )
        raise NotImplementedError(msg)

    edges = np.asarray(profile.bin_edges, dtype=float)
    num_bins = profile.num_bins
    if edges.size != num_bins + 1:
        msg = f"Expected {num_bins + 1} bin edges, got {edges.size}."
        raise ValueError(msg)
    controls = profile.regression_features
    k = len(controls)
    z = _z_value(level)
    needs_x_moments = kind == "rbc"

    bread, rhs, bin_stats = _first_pass(df_prepped, profile, kind, needs_x_moments)
    coefficients = _solve(bread, rhs)
    n_basis = num_bins + _BASIS_SIZE[kind]
    gamma = coefficients[n_basis:]

    meat = _second_pass(
        df_prepped,
        profile,
        kind,
        edges,
        coefficients[:n_basis],
        gamma,
        needs_x_moments,
    )

    total_count = float(bin_stats["counts"].sum())
    dof = max(total_count - (n_basis + k), 1.0)
    bread_inv = np.linalg.pinv(bread)
    cov = bread_inv @ meat @ bread_inv * (total_count / dof)

    control_means = (
        bin_stats["control_sums"].sum(axis=0) / total_count
        if k
        else np.zeros(0, dtype=float)
    )
    g = _evaluation_vectors(kind, edges, num_bins, bin_stats["mean_x"], control_means)

    center = g @ coefficients
    variance = np.einsum("ij,jk,ik->i", g, cov, g)
    std_error = np.sqrt(np.clip(variance, 0.0, None))
    return IntervalResult(
        center=center,
        lower=center - z * std_error,
        upper=center + z * std_error,
        std_error=std_error,
        kind=kind,
    )


def _first_pass(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    kind: CIKind,
    needs_x_moments: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Aggregate the design cross-moments and solve-side totals."""
    x_col = profile.x_col
    y_col = profile.y_col
    controls = profile.regression_features
    num_bins = profile.num_bins
    k = len(controls)

    # Products are materialized as columns before grouping: the dask backend only
    # accepts plain column aggregations inside group_by, so composing the product
    # inside .sum() works everywhere except there.
    products: dict[str, nw.Expr] = {}
    if needs_x_moments:
        products["__ci_x2"] = x_col * x_col
        products["__ci_xy"] = x_col * y_col
        for idx, name in enumerate(controls):
            products[f"__ci_wx_src_{idx}"] = nw.col(name) * x_col

    agg = [
        nw.len().alias("__ci_n"),
        x_col.mean().alias("__ci_mean_x"),
        y_col.sum().alias("__ci_sum_y"),
    ]
    if needs_x_moments:
        agg += [
            x_col.sum().alias("__ci_sum_x"),
            nw.col("__ci_x2").sum().alias("__ci_sum_x2"),
            nw.col("__ci_xy").sum().alias("__ci_sum_xy"),
        ]
    for idx, name in enumerate(controls):
        agg.append(nw.col(name).sum().alias(f"__ci_w_{idx}"))
        if needs_x_moments:
            agg.append(nw.col(f"__ci_wx_src_{idx}").sum().alias(f"__ci_wx_{idx}"))

    source = df_prepped.with_columns(**products) if products else df_prepped
    per_bin = (
        source.group_by(profile.bin_name).agg(*agg).sort(profile.bin_name).collect()
    )
    if per_bin.shape[0] != num_bins:
        msg = (
            f"Confidence intervals need every bin populated, but {per_bin.shape[0]} of "
            f"{num_bins} bins contain rows. Decrease num_bins."
        )
        raise ValueError(msg)

    counts = per_bin.get_column("__ci_n").to_numpy().astype(float)
    m0 = counts
    m1 = (
        per_bin.get_column("__ci_sum_x").to_numpy().astype(float)
        if needs_x_moments
        else np.zeros(num_bins)
    )
    m2 = (
        per_bin.get_column("__ci_sum_x2").to_numpy().astype(float)
        if needs_x_moments
        else np.zeros(num_bins)
    )
    r0 = per_bin.get_column("__ci_sum_y").to_numpy().astype(float)
    r1 = (
        per_bin.get_column("__ci_sum_xy").to_numpy().astype(float)
        if needs_x_moments
        else np.zeros(num_bins)
    )
    p0 = _stack(per_bin, "__ci_w_{}", k, num_bins)
    p1 = (
        _stack(per_bin, "__ci_wx_{}", k, num_bins)
        if needs_x_moments
        else np.zeros((num_bins, k))
    )

    cross, wy = _control_totals(df_prepped, controls, y_col)
    edges = np.asarray(profile.bin_edges, dtype=float)
    bread = _assemble(kind, edges, num_bins, k, m0, m1, m2, p0, p1, cross)
    rhs = _assemble_rhs(kind, edges, num_bins, k, r0, r1, wy)
    stats = {
        "counts": counts,
        "mean_x": per_bin.get_column("__ci_mean_x").to_numpy().astype(float),
        "control_sums": p0,
    }
    return bread, rhs, stats


def _second_pass(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    kind: CIKind,
    edges: np.ndarray,
    basis_coefficients: np.ndarray,
    gamma: np.ndarray,
    needs_x_moments: bool,
) -> np.ndarray:
    """Aggregate the residual-weighted cross-moments that form the sandwich meat."""
    x_col = profile.x_col
    controls = profile.regression_features
    num_bins = profile.num_bins
    k = len(controls)

    fitted = _fitted_curve_expr(kind, profile.x_name, edges, basis_coefficients)
    control_part = _control_expr(controls, gamma)
    if control_part is not None:
        fitted = fitted + control_part
    resid_sq = (profile.y_col - fitted) ** 2

    # As in the first pass, every product becomes a column before grouping so the
    # aggregations stay plain enough for dask.
    with_resid = df_prepped.with_columns(__ci_usq=resid_sq)
    usq = nw.col("__ci_usq")
    products: dict[str, nw.Expr] = {}
    if needs_x_moments:
        products["__ci_u1_src"] = usq * x_col
        products["__ci_u2_src"] = usq * x_col * x_col
    for idx, name in enumerate(controls):
        products[f"__ci_uw_src_{idx}"] = usq * nw.col(name)
        if needs_x_moments:
            products[f"__ci_uwx_src_{idx}"] = usq * nw.col(name) * x_col
    source = with_resid.with_columns(**products) if products else with_resid

    agg = [usq.sum().alias("__ci_u0")]
    if needs_x_moments:
        agg += [
            nw.col("__ci_u1_src").sum().alias("__ci_u1"),
            nw.col("__ci_u2_src").sum().alias("__ci_u2"),
        ]
    for idx in range(k):
        agg.append(nw.col(f"__ci_uw_src_{idx}").sum().alias(f"__ci_uw_{idx}"))
        if needs_x_moments:
            agg.append(nw.col(f"__ci_uwx_src_{idx}").sum().alias(f"__ci_uwx_{idx}"))

    per_bin = (
        source.group_by(profile.bin_name).agg(*agg).sort(profile.bin_name).collect()
    )

    m0 = per_bin.get_column("__ci_u0").to_numpy().astype(float)
    m1 = (
        per_bin.get_column("__ci_u1").to_numpy().astype(float)
        if needs_x_moments
        else np.zeros(num_bins)
    )
    m2 = (
        per_bin.get_column("__ci_u2").to_numpy().astype(float)
        if needs_x_moments
        else np.zeros(num_bins)
    )
    p0 = _stack(per_bin, "__ci_uw_{}", k, num_bins)
    p1 = (
        _stack(per_bin, "__ci_uwx_{}", k, num_bins)
        if needs_x_moments
        else np.zeros((num_bins, k))
    )

    cross = np.zeros((k, k), dtype=float)
    if k:
        # A plain select, unlike group_by, takes composed expressions on every backend.
        exprs = [
            (usq * nw.col(controls[i]) * nw.col(controls[j]))
            .sum()
            .alias(f"__ci_uww_{i}_{j}")
            for i in range(k)
            for j in range(i, k)
        ]
        totals = with_resid.select(*exprs).collect()
        for i in range(k):
            for j in range(i, k):
                value = float(totals.item(0, f"__ci_uww_{i}_{j}") or 0.0)
                cross[i, j] = value
                cross[j, i] = value

    return _assemble(kind, edges, num_bins, k, m0, m1, m2, p0, p1, cross)


def _control_totals(
    df_prepped: nw.LazyFrame, controls: tuple[str, ...], y_col: nw.Expr
) -> tuple[np.ndarray, np.ndarray]:
    """Global control-by-control and control-by-y cross-moments."""
    k = len(controls)
    if not k:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    exprs = [
        (nw.col(c) * y_col).sum().alias(f"__ci_wy_{i}") for i, c in enumerate(controls)
    ]
    for i in range(k):
        for j in range(i, k):
            exprs.append(
                (nw.col(controls[i]) * nw.col(controls[j]))
                .sum()
                .alias(f"__ci_ww_{i}_{j}")
            )
    totals = df_prepped.select(*exprs).collect()
    wy = np.array(
        [float(totals.item(0, f"__ci_wy_{i}") or 0.0) for i in range(k)], dtype=float
    )
    cross = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i, k):
            value = float(totals.item(0, f"__ci_ww_{i}_{j}") or 0.0)
            cross[i, j] = value
            cross[j, i] = value
    return cross, wy


def _stack(per_bin: nw.DataFrame, template: str, k: int, num_bins: int) -> np.ndarray:
    if not k:
        return np.zeros((num_bins, 0), dtype=float)
    return np.column_stack(
        [
            per_bin.get_column(template.format(i)).to_numpy().astype(float)
            for i in range(k)
        ]
    )
