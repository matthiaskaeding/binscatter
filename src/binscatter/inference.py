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
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
import numpy as np

from .fixed_effects import FEProjector, stack_group_sums, symmetrize

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
    """Build ``X' w`` from per-bin sums of ``w`` and ``x*w`` plus the control block.

    With vector inputs this is the ``X'y`` right-hand side. The same combination
    also builds any other ``X'``-weighted quantity the sandwich needs, so ``r0``,
    ``r1`` and ``wy`` may instead carry a leading axis of groups -- ``(g, J)`` and
    ``(g, k)`` -- giving ``(g, size)`` out. That is what turns per-level bin
    crosstabs into ``D'X`` and per-cell moments into their ``X`` contributions,
    without a second copy of the basis bookkeeping.
    """
    n_basis = num_bins + _BASIS_SIZE[kind]
    flat = r0.ndim == 1
    rows0 = np.atleast_2d(r0)
    rows1 = np.atleast_2d(r1)
    out = np.zeros((rows0.shape[0], n_basis + num_controls), dtype=float)
    for j in range(num_bins):
        for col_p, a_p, b_p in _basis_terms(kind, edges, j):
            out[:, col_p] += a_p * rows0[:, j] + b_p * rows1[:, j]
    if num_controls:
        out[:, n_basis:] = np.atleast_2d(wy)
    return out[0] if flat else out


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


@dataclass(frozen=True)
class _FEDesign:
    """Everything the sandwich needs about the absorbed fixed effects.

    Attributes:
        projector: solves ``(D'D) alpha = rhs`` from the observed intersections.
        cell_codes: ``(C, F)`` the distinct fixed-effect tuples.
        sums: ``(L, size)`` the stacked group sums ``D'X`` of the design.
        alpha: ``(L, size)`` its projection, ``G^+ D'X``.
        alpha_y: ``(L,)`` the projection of ``D'y``.
        cell_effects: ``(C, size)`` ``D alpha`` per cell -- the value the design's
            fixed-effect component takes on every row of that cell.
        level_index: per factor, the label-to-code mapping. The second pass reuses
            it so both passes number the levels identically.
    """

    projector: FEProjector
    cell_codes: np.ndarray
    sums: np.ndarray
    alpha: np.ndarray
    alpha_y: np.ndarray
    cell_effects: np.ndarray
    level_index: list[dict[Any, int]]


def _fe_codes(
    frame: nw.DataFrame,
    fe_names: tuple[str, ...],
    indices: list[dict[Any, int]] | None = None,
) -> tuple[np.ndarray, list[dict[Any, int]]]:
    """Map each factor's labels to dense integer codes, ``(rows, F)``.

    Labels are mapped through per-factor dictionaries rather than by position,
    because ``group_by`` output order is not stable across backends. Pass the
    dictionaries from an earlier call to reuse the same numbering.
    """
    codes = np.empty((frame.shape[0], len(fe_names)), dtype=np.int64)
    reuse = indices is not None
    built: list[dict[Any, int]] = indices if reuse else []
    for f, name in enumerate(fe_names):
        labels = frame.get_column(name).to_list()
        if reuse:
            index = built[f]
        else:
            index = {}
            for label in labels:
                if label not in index:
                    index[label] = len(index)
            built.append(index)
        codes[:, f] = [index[label] for label in labels]
    return codes, built


def _by_level_and_bin(
    values: np.ndarray,
    projector: FEProjector,
    codes: np.ndarray,
    bin_idx: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """Spread per-row values into an ``(L, J)`` level-by-bin crosstab.

    Flattening ``(level, bin)`` into one index keeps this to a single bincount per
    factor, rather than materializing a dense indicator per input row.
    """
    return np.concatenate(
        [
            np.bincount(
                codes[:, f] * num_bins + bin_idx,
                weights=values,
                minlength=int(projector.counts[f].size) * num_bins,
            ).reshape(int(projector.counts[f].size), num_bins)
            for f in range(projector.num_factors)
        ]
    )


def _fe_design_sums(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    kind: CIKind,
    needs_x_moments: bool,
    bin_labels: Sequence[Any],
) -> _FEDesign:
    """Aggregate ``D'X`` and ``D'y`` over the observed fixed-effect intersections.

    One ``group_by(*fe_names, bin_name)``. The bin key is needed because each basis
    column is affine in ``x`` with bin-dependent coefficients, so the group sums of
    the basis have to be built bin by bin through :func:`_basis_terms` -- exactly as
    the bread is.
    """
    fe_names = profile.fe_names
    controls = profile.regression_features
    k = len(controls)
    num_bins = profile.num_bins

    products: dict[str, nw.Expr] = {}
    if needs_x_moments:
        products["__ci_fe_x"] = profile.x_col

    agg = [nw.len().alias("__ci_fe_n"), profile.y_col.sum().alias("__ci_fe_y")]
    if needs_x_moments:
        agg.append(nw.col("__ci_fe_x").sum().alias("__ci_fe_sx"))
    for idx, name in enumerate(controls):
        agg.append(nw.col(name).sum().alias(f"__ci_fe_w_{idx}"))

    source = df_prepped.with_columns(**products) if products else df_prepped
    cells = source.group_by(*fe_names, profile.bin_name).agg(*agg).collect()

    codes, level_index = _fe_codes(cells, fe_names)
    counts = cells.get_column("__ci_fe_n").to_numpy().astype(float)
    projector = FEProjector.from_row_codes(codes, fe_names, row_weights=counts)

    bin_index = {label: i for i, label in enumerate(bin_labels)}
    bin_idx = np.fromiter(
        (bin_index[label] for label in cells.get_column(profile.bin_name).to_list()),
        dtype=np.int64,
        count=cells.shape[0],
    )

    r0 = _by_level_and_bin(counts, projector, codes, bin_idx, num_bins)
    if needs_x_moments:
        r1 = _by_level_and_bin(
            cells.get_column("__ci_fe_sx").to_numpy().astype(float),
            projector,
            codes,
            bin_idx,
            num_bins,
        )
    else:
        r1 = np.zeros_like(r0)

    control_sums = (
        stack_group_sums(
            np.column_stack(
                [
                    cells.get_column(f"__ci_fe_w_{idx}").to_numpy().astype(float)
                    for idx in range(k)
                ]
            ),
            projector,
            codes,
        )
        if k
        else np.zeros((projector.total_levels, 0), dtype=float)
    )
    y_sums = stack_group_sums(
        cells.get_column("__ci_fe_y").to_numpy().astype(float), projector, codes
    )

    edges = np.asarray(profile.bin_edges, dtype=float)
    sums = _assemble_rhs(kind, edges, num_bins, k, r0, r1, control_sums)
    alpha = projector.solve(np.column_stack([sums, y_sums]))

    logger.debug(
        "[inference] fixed effects %s: levels=%s cells=%d",
        fe_names,
        projector.level_counts,
        projector.num_cells,
    )
    return _FEDesign(
        projector=projector,
        cell_codes=projector.cell_codes,
        sums=sums,
        alpha=alpha[:, :-1],
        alpha_y=alpha[:, -1],
        cell_effects=projector.row_effects(alpha[:, :-1], projector.cell_codes),
        level_index=level_index,
    )


def compute_intervals(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    kind: CIKind,
    level: float,
) -> IntervalResult:
    """Compute per-bin confidence intervals for the binscatter dots.

    Returns bounds in bin order, matching the row order of the plotting frame.
    """
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
    n_basis = num_bins + _BASIS_SIZE[kind]
    total_count = float(bin_stats["counts"].sum())
    fe_params = 0

    fe: _FEDesign | None = None
    column_means = np.zeros(0, dtype=float)
    if profile.fe_names:
        # Absorbing the fixed effects means the design the sandwich refers to is
        # M_D X, not X. Both the bread and the right-hand side pick up a correction
        # built purely from the group sums; neither needs a per-row quantity.
        fe = _fe_design_sums(
            df_prepped, profile, kind, needs_x_moments, bin_stats["bin_labels"]
        )
        column_means = bin_stats["column_sums"] / total_count
        y_mean = bin_stats["sum_y"] / total_count
        bread = symmetrize(
            bread
            - fe.sums.T @ fe.alpha
            + total_count * np.outer(column_means, column_means)
        )
        rhs = rhs - fe.sums.T @ fe.alpha_y + total_count * column_means * y_mean
        # The fixed effects still cost parameters: rank(D), less the intercept the
        # basis already spans. The rank counts connected components, so separable
        # blocks are not charged for the shifts between them.
        fe_params = fe.projector.rank - 1

    coefficients = _solve(bread, rhs)
    gamma = coefficients[n_basis:]

    if fe is None:
        meat = _second_pass(
            df_prepped,
            profile,
            kind,
            edges,
            coefficients[:n_basis],
            gamma,
            needs_x_moments,
        )
    else:
        meat = _fe_second_pass(
            df_prepped,
            profile,
            kind,
            edges,
            coefficients[:n_basis],
            gamma,
            needs_x_moments,
            fe,
            column_means,
            bin_stats["bin_labels"],
        )

    dof = max(total_count - (n_basis + k + fe_params), 1.0)
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
        # X'1 and 1'y. Only the absorbed path uses them, to re-centre the design
        # around its column means so the fixed-effect projection keeps the intercept.
        "column_sums": _assemble_rhs(kind, edges, num_bins, k, m0, m1, p0.sum(axis=0)),
        "sum_y": r0.sum(),
        "bin_labels": per_bin.get_column(profile.bin_name).to_list(),
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


def _fe_second_pass(
    df_prepped: nw.LazyFrame,
    profile: Profile,
    kind: CIKind,
    edges: np.ndarray,
    basis_coefficients: np.ndarray,
    gamma: np.ndarray,
    needs_x_moments: bool,
    fe: _FEDesign,
    column_means: np.ndarray,
    bin_labels: Sequence[Any],
) -> np.ndarray:
    """The sandwich meat when fixed effects are absorbed, from aggregates alone.

    The residual the meat needs is not the one :func:`_second_pass` reconstructs.
    Writing ``e = y - X theta`` for the residual before the fixed effects come out,
    the absorbed residual is

        u_i = e_i + mean(e) - (D alpha_e)_i,      alpha_e = G^+ (D'e)

    so it differs from ``e`` by a term constant within each fixed-effect
    intersection. That term is not something the backend has to compute per row:
    ``D'e`` is itself a group sum, so grouping by ``(*fe_names, bin_name)`` and
    collecting ``sum(e)`` is enough to recover the offset in the driver.

    Every moment is therefore aggregated under three weights -- ``1``, ``e`` and
    ``e**2`` -- and recombined here, since

        sum_i u_i**2 m_i = sum_cells [ M2 - 2 d_c M1 + d_c**2 M0 ].

    One expansion covers every block of the meat, rather than one derivation per
    block.
    """
    x_col = profile.x_col
    controls = profile.regression_features
    num_bins = profile.num_bins
    k = len(controls)

    fitted = _fitted_curve_expr(kind, profile.x_name, edges, basis_coefficients)
    control_part = _control_expr(controls, gamma)
    if control_part is not None:
        fitted = fitted + control_part

    # Moments of the design, named once and reused under each weight. These are the
    # same quantities _second_pass collects per bin; only the grouping and the
    # weight differ.
    moments: dict[str, nw.Expr] = {"m0": nw.lit(1.0)}
    if needs_x_moments:
        moments["m1"] = x_col
        moments["m2"] = x_col * x_col
    for idx, name in enumerate(controls):
        moments[f"p0_{idx}"] = nw.col(name)
        if needs_x_moments:
            moments[f"p1_{idx}"] = nw.col(name) * x_col
    for i in range(k):
        for j in range(i, k):
            moments[f"c_{i}_{j}"] = nw.col(controls[i]) * nw.col(controls[j])

    resid = nw.col("__ci_fe_e")
    weights = {"0": nw.lit(1.0), "1": resid, "2": resid * resid}

    # Products become columns before grouping: dask only accepts plain column
    # aggregations inside group_by.
    with_resid = df_prepped.with_columns(__ci_fe_e=profile.y_col - fitted)
    products = {
        f"__ci_fm_{t}_{key}": weight * moment
        for t, weight in weights.items()
        for key, moment in moments.items()
    }
    source = with_resid.with_columns(**products)
    grouped = (
        source.group_by(*profile.fe_names, profile.bin_name)
        .agg(*(nw.col(alias).sum().alias(alias) for alias in products))
        .collect()
    )

    codes, _ = _fe_codes(grouped, profile.fe_names, fe.level_index)
    cells, cell_of_row = np.unique(codes, axis=0, return_inverse=True)
    cell_of_row = np.ravel(cell_of_row)
    if cells.shape != fe.cell_codes.shape:
        msg = (
            "The interval's residual pass saw a different set of fixed-effect "
            "combinations than its design pass. This should not happen on a stable "
            "frame; the two passes must aggregate the same rows."
        )
        raise ValueError(msg)
    bin_index = {label: i for i, label in enumerate(bin_labels)}
    bin_idx = np.fromiter(
        (bin_index[label] for label in grouped.get_column(profile.bin_name).to_list()),
        dtype=np.int64,
        count=grouped.shape[0],
    )

    num_cells = fe.cell_codes.shape[0]

    def spread(alias: str) -> np.ndarray:
        """Scatter one aggregated column into a ``(C, J)`` cell-by-bin table."""
        out = np.zeros((num_cells, num_bins), dtype=float)
        out[cell_of_row, bin_idx] = grouped.get_column(alias).to_numpy().astype(float)
        return out

    tables = {
        t: {key: spread(f"__ci_fm_{t}_{key}") for key in moments} for t in weights
    }

    # The per-cell offset, recovered from sum(e) alone.
    total_count = tables["0"]["m0"].sum()
    resid_by_cell = tables["1"]["m0"].sum(axis=1)
    alpha_e = fe.projector.solve(
        stack_group_sums(resid_by_cell, fe.projector, fe.cell_codes)
    )
    offsets = (
        fe.projector.row_effects(alpha_e, fe.cell_codes)
        - resid_by_cell.sum() / total_count
    )

    squared = {
        key: (
            tables["2"][key]
            - 2.0 * offsets[:, None] * tables["1"][key]
            + (offsets**2)[:, None] * tables["0"][key]
        )
        for key in moments
    }

    zeros_bins = np.zeros(num_bins, dtype=float)
    m0 = squared["m0"].sum(axis=0)
    m1 = squared["m1"].sum(axis=0) if needs_x_moments else zeros_bins
    m2 = squared["m2"].sum(axis=0) if needs_x_moments else zeros_bins
    p0 = _columns(squared, "p0_{}", k, num_bins)
    p1 = (
        _columns(squared, "p1_{}", k, num_bins)
        if needs_x_moments
        else np.zeros((num_bins, k))
    )
    cross = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i, k):
            value = float(squared[f"c_{i}_{j}"].sum())
            cross[i, j] = value
            cross[j, i] = value

    # X'UX, then the terms that turn it into (M_D X)' U (M_D X). Every one of them
    # is a sum over cells, because D alpha is constant within a cell.
    raw = _assemble(kind, edges, num_bins, k, m0, m1, m2, p0, p1, cross)

    per_cell = _assemble_rhs(
        kind,
        edges,
        num_bins,
        k,
        squared["m0"],
        squared["m1"] if needs_x_moments else np.zeros_like(squared["m0"]),
        np.column_stack([squared[f"p0_{idx}"].sum(axis=1) for idx in range(k)])
        if k
        else np.zeros((num_cells, 0)),
    )
    effects = fe.cell_effects
    weighted_cells = squared["m0"].sum(axis=1)

    cross_term = per_cell.T @ effects
    projected = (effects * weighted_cells[:, None]).T @ effects
    projected_total = (effects * weighted_cells[:, None]).sum(axis=0)
    design_total = per_cell.sum(axis=0)
    weight_total = float(weighted_cells.sum())

    meat = (
        raw
        - cross_term
        - cross_term.T
        + projected
        + np.outer(design_total, column_means)
        + np.outer(column_means, design_total)
        - np.outer(projected_total, column_means)
        - np.outer(column_means, projected_total)
        + weight_total * np.outer(column_means, column_means)
    )
    return symmetrize(meat)


def _columns(
    tables: dict[str, np.ndarray], template: str, k: int, num_bins: int
) -> np.ndarray:
    """Stack ``k`` cell-by-bin tables into the ``(J, k)`` per-bin block."""
    if not k:
        return np.zeros((num_bins, 0), dtype=float)
    return np.column_stack([tables[template.format(i)].sum(axis=0) for i in range(k)])


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
