"""Absorb categorical controls as fixed effects, without materializing dummies.

One-hot encoding a categorical control produces G-1 dummy columns, which makes
``_ensure_feature_moments`` build ``k(k+1)/2`` sum-product aggregations and
``partial_out_controls`` solve a dense ``(J+k)x(J+k)`` system. At a few thousand
levels that dies while building the query plan.

Instead, absorb the controls. Stack the group-dummy matrices of every absorbed
column into ``D = [D_1 ... D_F]``. For any two matrices ``A``, ``B`` the residual
cross product after projecting out ``D`` is

    A' M_D B = A'B - (D'A)' (D'D)^+ (D'B)

so the only things needed from the data are ``A'B`` (already collected as scalar
moments) and the group sums ``D'A``, ``D'B``. With one factor ``G = D'D`` is
diagonal and this is the Mundlak transform, ``A'B - S_A' N^-1 S_B``, in closed
form. With several factors ``G`` picks up the pairwise incidence blocks

    G = [[N_1,   C_12,  ...],
         [C_21,  N_2,   ...],
         [...,   ...,   ...]]        N_f = diag(counts),  C_fh = D_f'D_h

and ``alpha = G^+ (D'B)`` has to be solved for. :class:`FEProjector` does that with
block Gauss-Seidel -- the method of alternating projections, but running on the
aggregated crosstabs rather than on rows. The ``C_fh @ alpha_h`` products are
evaluated by ``np.bincount`` over COO-style arrays of the observed fixed-effect
intersections, so ``C_fh`` itself is never built. Cost per sweep is one bincount
per factor over the distinct FE tuples, and memory never depends on ``G_f * G_h``.

Why the answer is well defined even though ``alpha`` is not
-----------------------------------------------------------
``G`` is singular: each ``D_f`` spans the intercept, so it is rank-deficient by at
least ``F-1``, and by more when the design splits into disconnected components. But
``D'B`` always lies in ``range(D') = range(G)``, so the system is *consistent*, and
any two solutions differ by an element of ``null(D)``. Everything consumed
downstream -- the cross product ``S_A' alpha = A'D alpha`` and the level
``1'D alpha / n`` -- is a function of ``D alpha = P_D B`` alone, which is unique.
So a disconnected design needs no special handling here; it only makes individual
group effects unidentified, which nothing in this module reports.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import narwhals as nw
import numpy as np

logger = logging.getLogger(__name__)

_FE_COUNT = "__fe_count"

#: Relative residual ``||G alpha - S|| / ||S||`` the projection is solved to. Well
#: inside the 1e-9 tolerance the cross-backend equivalence tests assert on, so the
#: iteration is not what limits accuracy.
FE_TOL = 1e-11

#: Gauss-Seidel sweeps before falling back to conjugate gradients, and CG iterations
#: before giving up. A three-way design with 275 levels converges in ~11 sweeps; a
#: budget this large is only reached by genuinely pathological connectivity.
FE_MAXITER = 10_000

#: Distinct fixed-effect intersections the driver will hold. This is the one
#: quantity in the absorbed path that grows with the data rather than with the
#: number of levels, so it gets an explicit guard with a readable error instead of
#: an allocator failure.
MAX_FE_CELLS = 5_000_000


def _bincount_columns(
    codes: np.ndarray, values: np.ndarray, minlength: int
) -> np.ndarray:
    """Sum each column of ``values`` by ``codes``, returning ``(minlength, m)``.

    ``np.bincount`` only takes 1-D weights, and ``np.add.at`` on a 2-D target is an
    order of magnitude slower, so loop the columns. ``m`` is the number of response
    columns being projected (bins + controls + responses), typically well under 100.
    """
    if values.ndim == 1:
        return np.bincount(codes, weights=values, minlength=minlength)
    return np.column_stack(
        [
            np.bincount(codes, weights=values[:, j], minlength=minlength)
            for j in range(values.shape[1])
        ]
    )


@dataclass
class FEProjector:
    """Solves ``(D'D) alpha = rhs`` from the observed fixed-effect intersections.

    Attributes:
        cell_codes: ``(C, F)`` the distinct combinations of factor levels present.
        cell_weights: ``(C,)`` rows falling in each combination.
        counts: per factor, ``(G_f,)`` observations per level -- the diagonal ``N_f``.
        names: the absorbed column names, used in error messages.
    """

    cell_codes: np.ndarray
    cell_weights: np.ndarray
    counts: tuple[np.ndarray, ...]
    names: tuple[str, ...]

    @property
    def num_factors(self) -> int:
        return len(self.counts)

    @property
    def level_counts(self) -> tuple[int, ...]:
        return tuple(int(c.size) for c in self.counts)

    @property
    def total_levels(self) -> int:
        """``L = sum_f G_f``, the height of the stacked ``alpha`` and group sums."""
        return int(sum(c.size for c in self.counts))

    @property
    def offsets(self) -> tuple[int, ...]:
        """Start row of each factor's block inside a stacked ``(L, m)`` array."""
        starts: list[int] = [0]
        for c in self.counts[:-1]:
            starts.append(starts[-1] + int(c.size))
        return tuple(starts)

    @property
    def stacked_counts(self) -> np.ndarray:
        """``(L,)`` the per-level counts of every factor, stacked."""
        return np.concatenate(self.counts)

    @property
    def num_cells(self) -> int:
        return int(self.cell_codes.shape[0])

    @cached_property
    def num_components(self) -> int:
        """Connected components of the fixed-effect incidence graph.

        Nodes are levels, and every observed combination joins the levels it
        contains. For two factors these are the mobility groups of Abowd, Creecy
        and Kramarz: each one costs the design a further degree of freedom, because
        its levels can be shifted against each other without changing any fitted
        value.

        Computed by label propagation rather than a union-find loop: the pull step
        is a ``minimum.reduceat`` over a precomputed sort, so a design with millions
        of observed combinations stays vectorized.
        """
        if self.num_factors == 1:
            # A single factor joins nothing, so every level stands alone -- which is
            # right: D_1 is full rank and costs no components.
            return int(self.counts[0].size)

        offsets = self.offsets
        nodes = np.column_stack(
            [offsets[f] + self.cell_codes[:, f] for f in range(self.num_factors)]
        )
        # Per factor, the sort that groups cells by level, so the "smallest label
        # among the cells touching this level" step is one reduceat.
        orders, starts, owners = [], [], []
        for f in range(self.num_factors):
            column = nodes[:, f]
            order = np.argsort(column, kind="stable")
            ordered = column[order]
            first = np.flatnonzero(
                np.concatenate(([True], ordered[1:] != ordered[:-1]))
            )
            orders.append(order)
            starts.append(first)
            owners.append(ordered[first])

        labels = np.arange(self.total_levels)
        for _ in range(self.total_levels + 1):
            cell_label = labels[nodes].min(axis=1)
            updated = labels.copy()
            for f in range(self.num_factors):
                reduced = np.minimum.reduceat(cell_label[orders[f]], starts[f])
                updated[owners[f]] = np.minimum(updated[owners[f]], reduced)
            # Labels only ever decrease and never exceed their own index, so
            # following them repeatedly collapses each chain to its root and keeps
            # the number of sweeps logarithmic rather than proportional to diameter.
            while True:
                jumped = updated[updated]
                if np.array_equal(jumped, updated):
                    break
                updated = jumped
            if np.array_equal(updated, labels):
                break
            labels = updated

        return int(np.unique(labels).size)

    @property
    def rank(self) -> int:
        """``rank(D)`` for the stacked dummy matrix.

        Each factor beyond the first duplicates the intercept once per connected
        component, giving ``sum_f G_f - c (F - 1)``. Exact for one or two factors.
        For three or more it is an upper bound: a component in which two factors are
        nested carries dependencies the incidence graph cannot see, so the true rank
        can be lower. ``fixest`` and ``pyfixest`` make the same approximation.
        """
        return self.total_levels - self.num_components * (self.num_factors - 1)

    @classmethod
    def from_row_codes(
        cls,
        row_codes: np.ndarray,
        names: Sequence[str],
        row_weights: np.ndarray | None = None,
    ) -> FEProjector:
        """Build from integer codes, ``(n, F)``, deduplicating to distinct tuples.

        For callers that already hold materialized rows (the DPI selector), or rows
        of an aggregate that is finer than the fixed-effect tuple -- pass
        ``row_weights`` for the latter, so a ``(fixed effect, bin)`` crosstab
        collapses to cells carrying the right counts. The lazy estimation paths get
        the same object out of :func:`compute_fe_moments` without collecting ``n``
        rows at all.
        """
        row_codes = np.ascontiguousarray(row_codes)
        if row_codes.ndim == 1:
            row_codes = row_codes[:, None]
        cells, cell_of_row = np.unique(row_codes, axis=0, return_inverse=True)
        cell_of_row = np.ravel(cell_of_row)
        weights = np.bincount(
            cell_of_row, weights=row_weights, minlength=cells.shape[0]
        ).astype(float)
        counts = tuple(
            np.bincount(
                cells[:, f], weights=weights, minlength=int(cells[:, f].max()) + 1
            )
            for f in range(cells.shape[1])
        )
        return cls(
            cell_codes=cells,
            cell_weights=weights,
            counts=counts,
            names=tuple(names),
        )

    def _describe(self) -> str:
        pairs = ", ".join(
            f"{name} ({size} levels)"
            for name, size in zip(self.names, self.level_counts)
        )
        return f"{pairs}; {self.num_cells} observed combinations"

    def row_effects(self, alpha: np.ndarray, row_codes: np.ndarray) -> np.ndarray:
        """Return ``D alpha`` evaluated at ``row_codes``, shape ``(n, m)``.

        ``row_codes`` is ``(n, F)``; ``alpha`` is stacked ``(L, m)`` or ``(L,)``.
        """
        offsets = self.offsets
        total = alpha[offsets[0] + row_codes[:, 0]]
        for f in range(1, self.num_factors):
            total = total + alpha[offsets[f] + row_codes[:, f]]
        return total

    def matvec(self, alpha: np.ndarray) -> np.ndarray:
        """Return ``G alpha`` for stacked ``alpha``, shape ``(L, m)`` or ``(L,)``.

        ``(G alpha)_f[g] = sum over cells with c_f == g of w * sum_h alpha_h[c_h]``.
        The diagonal block ``C_ff = N_f`` falls out of the same expression, so one
        bincount per factor covers a whole block row.
        """
        fit = self.row_effects(alpha, self.cell_codes)
        weighted = fit * (
            self.cell_weights if fit.ndim == 1 else self.cell_weights[:, None]
        )
        return np.concatenate(
            [
                np.atleast_1d(
                    _bincount_columns(
                        self.cell_codes[:, f], weighted, int(self.counts[f].size)
                    )
                )
                for f in range(self.num_factors)
            ]
        )

    def _relative_residual(self, alpha: np.ndarray, rhs: np.ndarray) -> float:
        scale = float(np.linalg.norm(rhs))
        if scale == 0.0:
            return 0.0
        return float(np.linalg.norm(self.matvec(alpha) - rhs)) / scale

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return ``alpha`` with ``G alpha = rhs``, for stacked ``rhs`` ``(L, m)``.

        With a single factor ``G`` is diagonal and this is the exact closed form.
        With several it is block Gauss-Seidel, accelerated by conjugate gradients if
        the sweeps stall.
        """
        squeeze = rhs.ndim == 1
        work = rhs[:, None] if squeeze else rhs
        if self.num_factors == 1:
            alpha = work / self.counts[0][:, None]
        else:
            alpha = self._solve_iterative(work)
        return alpha[:, 0] if squeeze else alpha

    def _solve_iterative(self, rhs: np.ndarray) -> np.ndarray:
        offsets = self.offsets
        codes = [self.cell_codes[:, f] for f in range(self.num_factors)]
        alpha = np.zeros((self.total_levels, rhs.shape[1]), dtype=float)
        # Running sum_h alpha_h[c_h] per cell, updated in place so a sweep costs one
        # bincount per factor rather than F.
        fit = np.zeros((self.num_cells, rhs.shape[1]), dtype=float)
        weights = self.cell_weights[:, None]

        sweeps = 0
        for sweeps in range(1, FE_MAXITER + 1):
            for f in range(self.num_factors):
                block = slice(offsets[f], offsets[f] + self.counts[f].size)
                current = alpha[block]
                others = fit - current[codes[f]]
                acc = _bincount_columns(
                    codes[f], weights * others, int(self.counts[f].size)
                )
                updated = (rhs[block] - acc) / self.counts[f][:, None]
                fit += (updated - current)[codes[f]]
                alpha[block] = updated
            if self._relative_residual(alpha, rhs) < FE_TOL:
                logger.debug(
                    "[fixed_effects] projection converged in %d sweeps", sweeps
                )
                return alpha

        residual = self._relative_residual(alpha, rhs)
        logger.debug(
            "[fixed_effects] Gauss-Seidel stalled at %.3e after %d sweeps, trying CG",
            residual,
            sweeps,
        )
        return self._solve_cg(rhs, alpha)

    def _solve_cg(self, rhs: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Jacobi-preconditioned conjugate gradients, warm-started from ``alpha``.

        ``G`` is positive semi-definite and the system is consistent, so CG is valid
        despite the rank deficiency: it stays in the Krylov space of the residual,
        which lies in ``range(G)``.
        """
        inv_diag = (1.0 / self.stacked_counts)[:, None]
        residual = rhs - self.matvec(alpha)
        precond = inv_diag * residual
        direction = precond.copy()
        rz = float(np.sum(residual * precond))

        for _ in range(FE_MAXITER):
            if self._relative_residual(alpha, rhs) < FE_TOL:
                return alpha
            gd = self.matvec(direction)
            denominator = float(np.sum(direction * gd))
            if denominator <= 0.0:
                break
            step = rz / denominator
            alpha = alpha + step * direction
            residual = residual - step * gd
            precond = inv_diag * residual
            rz_next = float(np.sum(residual * precond))
            if rz == 0.0:
                break
            direction = precond + (rz_next / rz) * direction
            rz = rz_next

        achieved = self._relative_residual(alpha, rhs)
        if achieved < FE_TOL:
            return alpha
        msg = (
            "Could not project out the absorbed fixed effects: the alternating "
            f"projection reached a relative residual of {achieved:.3e}, short of the "
            f"{FE_TOL:.0e} tolerance, after {FE_MAXITER} sweeps and as many "
            f"conjugate-gradient steps. Absorbed columns: {self._describe()}. This "
            "means the fixed effects are very poorly connected -- drop one of them, "
            "or reduce the cardinality of one, to get a solvable design."
        )
        raise ValueError(msg)


@dataclass
class FEMoments:
    """Group-level sufficient statistics for the absorbed categoricals.

    Every array is stacked over factors, so its leading dimension is
    ``L = sum_f G_f`` and ``projector.offsets`` locates each factor's block.

    Attributes:
        projector: solves ``(D'D) alpha = rhs`` from the observed intersections.
        bin_sums: ``(L, J)`` bin-by-level crosstab, ``D'B``.
        feature_sums: ``(L, q)`` level sums of the regression features, ``D'W``.
        response_sums: level sums keyed by response name, e.g. ``D'y``.
        counts: ``(L,)`` observations per level.
        total_count: total number of observations.
    """

    projector: FEProjector
    bin_sums: np.ndarray
    feature_sums: np.ndarray
    response_sums: dict[str, np.ndarray]
    counts: np.ndarray
    total_count: float

    def solve_all(self) -> dict[str, np.ndarray]:
        """Project every right-hand side in one solve.

        Returns ``{"bins": (L, J), "features": (L, q), <response>: (L,)}``. Solving
        the stacked system once rather than per block matters for several factors,
        where each solve is an iteration; with one factor it is a single divide
        either way.
        """
        response_names = list(self.response_sums)
        blocks = [self.bin_sums, self.feature_sums]
        blocks += [self.response_sums[name][:, None] for name in response_names]
        alpha = self.projector.solve(np.column_stack(blocks))

        num_bins = self.bin_sums.shape[1]
        num_features = self.feature_sums.shape[1]
        out: dict[str, np.ndarray] = {
            "bins": alpha[:, :num_bins],
            "features": alpha[:, num_bins : num_bins + num_features],
        }
        for i, name in enumerate(response_names):
            out[name] = alpha[:, num_bins + num_features + i]
        return out


def within_correct(
    raw: np.ndarray, sums_a: np.ndarray, alpha_b: np.ndarray
) -> np.ndarray:
    """Return ``raw - (D'A)' alpha_B``, the cross product after absorbing ``D``.

    ``alpha_b`` is the already-solved projection of ``D'B``, so callers pay for one
    stacked solve rather than one per block. ``sums_a`` is ``(L, a)``; ``alpha_b``
    may be ``(L, b)`` for a matrix block or ``(L,)`` for a vector, matching ``raw``.
    """
    return raw - sums_a.T @ alpha_b


def symmetrize(block: np.ndarray) -> np.ndarray:
    """Average a block with its transpose.

    ``A' M_D A`` is symmetric in exact arithmetic, but with several factors the
    projection is iterative, so the computed block is only symmetric to the solver
    tolerance. Downstream solves are better behaved if that is cleaned up.
    """
    return (block + block.T) / 2.0


def group_codes(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(codes, counts)`` for ``groups``, where codes index into counts."""
    _, codes = np.unique(groups, return_inverse=True)
    codes = np.ravel(codes)
    return codes, np.bincount(codes).astype(float)


def factor_codes(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Return ``(n, F)`` integer codes, one column per factor."""
    return np.column_stack([group_codes(col)[0] for col in columns])


def demean_within(
    values: np.ndarray, codes: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Subtract group means from ``values``.

    Single-factor helper, used only where the data has already been materialized;
    the lazy paths use :func:`within_correct` on the moments instead.
    """
    sums = np.bincount(codes, weights=values, minlength=counts.size)
    return values - (sums / counts)[codes]


def demean_centered(
    values: np.ndarray, projector: FEProjector, row_codes: np.ndarray
) -> np.ndarray:
    """Remove centered fixed effects from ``values`` while preserving its mean.

    This projects onto the orthogonal complement of the group indicators constrained
    to have a count-weighted mean of zero. Unlike :func:`demean_within` the intercept
    survives, which keeps the DPI bin basis full rank and makes its coefficients the
    curve evaluated at the sample-average fixed-effect level.

    Adding the mean back is exactly right for any number of factors: ``1`` lies in
    ``range(D)``, so ``1'P_D v = 1'v`` and the projection already preserves the mean.
    """
    sums = stack_group_sums(values, projector, row_codes)
    alpha = projector.solve(sums)
    return values - projector.row_effects(alpha, row_codes) + float(np.mean(values))


def stack_group_sums(
    values: np.ndarray, projector: FEProjector, row_codes: np.ndarray
) -> np.ndarray:
    """Return the stacked group sums ``D'values``, ``(L, m)``.

    ``row_codes`` is ``(rows, F)`` and ``values`` ``(rows,)`` or ``(rows, m)``; the
    rows may be observations or any finer aggregate keyed by the factor levels.
    """
    return np.concatenate(
        [
            np.atleast_1d(
                _bincount_columns(
                    row_codes[:, f], values, int(projector.counts[f].size)
                )
            )
            for f in range(projector.num_factors)
        ]
    )


def select_absorbed(
    df: nw.LazyFrame, categorical_controls: tuple[str, ...]
) -> tuple[str, ...]:
    """Pick the categorical controls to absorb: all of them, highest cardinality first.

    Costs a single pass computing ``n_unique`` for every categorical control at once.

    A constant categorical is never absorbed: it carries no information, its dummy is
    collinear with the intercept, and absorbing it would demean against a single
    degenerate group.

    The order is by descending cardinality so the absorbed tuple, and therefore the
    layout of every stacked array derived from it, is deterministic rather than
    dependent on the order the controls were passed in.
    """
    if not categorical_controls:
        return ()

    counts = df.select(
        *(nw.col(c).n_unique().alias(c) for c in categorical_controls)
    ).collect()
    cardinalities = {c: int(counts.item(0, c)) for c in categorical_controls}
    absorbed = tuple(
        sorted(
            (c for c in categorical_controls if cardinalities[c] > 1),
            key=lambda c: (-cardinalities[c], c),
        )
    )
    logger.debug(
        "[fixed_effects] cardinalities=%s absorbing=%s", cardinalities, absorbed
    )
    return absorbed


def compute_fe_moments(
    df: nw.LazyFrame,
    *,
    fe_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    response_exprs: Mapping[str, nw.Expr],
    bin_name: str | None = None,
    bin_labels: Sequence[Any] | None = None,
) -> FEMoments:
    """Collect the level sufficient statistics needed to absorb ``fe_names``.

    Two aggregations regardless of how many factors are absorbed:

    1. ``group_by(*fe_names)`` gives the distinct fixed-effect intersections, their
       counts, and -- summed down each factor's codes -- every ``N_f`` and ``D_f'A``.
    2. ``group_by(*fe_names, bin_name)`` gives every ``D_f'B``, when bins exist. The
       bin selectors run before bins do, so they omit it and get ``(L, 0)``.

    Group and bin labels are mapped explicitly through dictionaries rather than by
    position, because ``group_by`` output order is not stable across backends.
    """
    response_names = list(response_exprs)
    aliases = {name: f"__fe_resp_{i}" for i, name in enumerate(response_names)}

    # Materialize derived responses before grouping. Dask rejects non-elementary
    # aggregations such as (col * col).sum() inside group_by, so the expression has
    # to become a column first.
    df = df.with_columns(
        *(response_exprs[name].alias(aliases[name]) for name in response_names)
    )

    group_aggs = [nw.len().alias(_FE_COUNT)]
    group_aggs += [
        nw.col(aliases[name]).sum().alias(aliases[name]) for name in response_names
    ]
    group_aggs += [nw.col(c).sum().alias(c) for c in feature_names]

    cells = df.group_by(*fe_names).agg(*group_aggs).collect()
    num_cells = cells.shape[0]
    if num_cells == 0:
        joined = "', '".join(fe_names)
        raise ValueError(f"No groups found for fixed effect column(s) '{joined}'.")
    if num_cells > MAX_FE_CELLS:
        msg = (
            f"Absorbing {list(fe_names)} would require holding {num_cells} distinct "
            f"fixed-effect combinations in memory, above the {MAX_FE_CELLS} limit. "
            "Drop one of these controls, or reduce its cardinality."
        )
        raise ValueError(msg)

    # Map each factor's labels to codes independently, so the two aggregations do not
    # have to agree on tuple ordering.
    level_index: list[dict[Any, int]] = []
    cell_codes = np.empty((num_cells, len(fe_names)), dtype=np.int64)
    for f, name in enumerate(fe_names):
        labels = cells.get_column(name).to_list()
        index: dict[Any, int] = {}
        for label in labels:
            if label not in index:
                index[label] = len(index)
        level_index.append(index)
        cell_codes[:, f] = [index[label] for label in labels]

    cell_weights = cells.get_column(_FE_COUNT).to_numpy().astype(float)
    counts = tuple(
        np.bincount(
            cell_codes[:, f], weights=cell_weights, minlength=len(level_index[f])
        )
        for f in range(len(fe_names))
    )
    projector = FEProjector(
        cell_codes=cell_codes,
        cell_weights=cell_weights,
        counts=counts,
        names=fe_names,
    )

    def stack(values: np.ndarray) -> np.ndarray:
        return stack_group_sums(values, projector, cell_codes)

    response_sums = {
        name: stack(cells.get_column(aliases[name]).to_numpy().astype(float))
        for name in response_names
    }
    if feature_names:
        feature_sums = stack(
            np.column_stack(
                [cells.get_column(c).to_numpy().astype(float) for c in feature_names]
            )
        )
    else:
        feature_sums = np.zeros((projector.total_levels, 0), dtype=float)

    if bin_name is None or bin_labels is None:
        bin_sums = np.zeros((projector.total_levels, 0), dtype=float)
        logger.debug(
            "[fixed_effects] fe=%s levels=%s cells=%d (no crosstab)",
            fe_names,
            projector.level_counts,
            num_cells,
        )
    else:
        bin_index = {label: i for i, label in enumerate(bin_labels)}
        crosstab = (
            df.group_by(*fe_names, bin_name).agg(nw.len().alias(_FE_COUNT)).collect()
        )
        num_rows = crosstab.shape[0]
        crosstab_codes = np.empty((num_rows, len(fe_names)), dtype=np.int64)
        for f, name in enumerate(fe_names):
            index = level_index[f]
            crosstab_codes[:, f] = [
                index[label] for label in crosstab.get_column(name).to_list()
            ]
        crosstab_bins = np.fromiter(
            (bin_index[label] for label in crosstab.get_column(bin_name).to_list()),
            dtype=np.int64,
            count=num_rows,
        )
        crosstab_counts = crosstab.get_column(_FE_COUNT).to_numpy().astype(float)

        # Flatten (level, bin) into a single index so each factor's block is one
        # bincount, rather than materializing a dense (crosstab rows x J) indicator.
        num_bin_labels = len(bin_labels)
        bin_sums = np.concatenate(
            [
                np.bincount(
                    crosstab_codes[:, f] * num_bin_labels + crosstab_bins,
                    weights=crosstab_counts,
                    minlength=int(projector.counts[f].size) * num_bin_labels,
                ).reshape(int(projector.counts[f].size), num_bin_labels)
                for f in range(len(fe_names))
            ]
        )

        logger.debug(
            "[fixed_effects] fe=%s levels=%s cells=%d crosstab_cells=%d",
            fe_names,
            projector.level_counts,
            num_cells,
            num_rows,
        )

    return FEMoments(
        projector=projector,
        bin_sums=bin_sums,
        feature_sums=feature_sums,
        response_sums=response_sums,
        counts=projector.stacked_counts,
        total_count=float(cell_weights.sum()),
    )


def recover_levels(
    moments: FEMoments,
    alphas: Mapping[str, np.ndarray],
    beta: np.ndarray,
    gamma: np.ndarray,
    response: str,
) -> float:
    """Return the count-weighted mean fixed effect.

    Absorbing removes the level from ``beta``: the within system is rank-deficient,
    so ``beta`` is identified only up to a shift. Adding this back makes
    ``beta_j + mean_alpha`` invariant to that shift and puts fitted values back on
    the original ``y`` scale.

    The residual projection is assembled from the already-solved ``alphas`` by
    linearity -- ``G^+ (D'y - D'B beta - D'W gamma)`` equals
    ``alpha_y - alpha_bins beta - alpha_features gamma`` -- so this costs no extra
    iteration.
    """
    alpha = alphas[response].copy()
    if beta.size:
        alpha -= alphas["bins"] @ beta
    if gamma.size:
        alpha -= alphas["features"] @ gamma
    return float(moments.counts @ alpha / moments.total_count)
