"""Absorb two or more categorical controls via a sparse solve on crosstabs.

:mod:`fixed_effects` absorbs a *single* categorical exactly, because ``D'D`` is
diagonal for one factor and the Mundlak correction ``A'B - S_A' N^-1 S_B`` therefore
has a closed form. With two or more factors ``D'D`` picks up off-diagonal incidence
blocks and that closed form is gone -- this is the two-way fixed effects problem.

What survives is the property the library is built on: every block of the normal
equations is still a ``group_by``. ``D_i'D_j`` is the crosstab of factors ``i`` and
``j``, ``D_i'B`` the crosstab of factor ``i`` against the bins, and ``D_i'W`` /
``D_i'y`` are plain group sums. So the full system

    [B W D]' [B W D] theta = [B W D]' y

is assembled from aggregates alone -- no per-row residuals, nothing of size ``n`` on
the driver -- and solved sparsely. Absorption and one-hot encoding are the same
estimator; this is only a cheaper route to it.

Two honest costs relative to the one-factor path:

* The solve is sized by the number of levels rather than being independent of it.
  One factor with 50,000 levels costs the same as five; two factors with 50,000
  levels each mean a system with 100,000 unknowns.
* The incidence crosstabs have to fit on the driver. ``nnz(D_i'D_j) <= min(n, G_i *
  G_j)``, which is comfortable for firm x year and hopeless for worker x firm.
  :data:`MAX_CROSSTAB_ENTRIES` turns the latter into an error naming the columns
  rather than an allocation failure.

scipy is optional, and imported lazily like the optional backends. Without it
:func:`sparse_available` reports ``False`` and the caller caps absorption at one
factor, one-hot encoding the rest -- the same estimator by the slower route that
predates this module, so a missing scipy costs speed rather than results.

The system is rank-deficient by exactly ``F``: the bin dummies span the intercept
and so does every factor. Any least-squares solution will do, because the quantity
the caller wants -- ``beta_j`` plus the count-weighted mean effect -- is invariant to
the shift between them, exactly as in :func:`fixed_effects.recover_levels`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np

logger = logging.getLogger(__name__)

_FE_COUNT = "__mfe_count"

#: Ceiling on stored crosstab entries before the multi-way path refuses to run.
#: Entries are held as numpy index/value triples, so roughly 20 bytes each: 25M is
#: about half a gigabyte, which is a bad day but not a dead process. Beyond it the
#: honest answer is that this method is wrong for the data.
MAX_CROSSTAB_ENTRIES = 25_000_000

_SCIPY_HINT = (
    "Absorbing two or more categorical controls needs a sparse solver. Install it "
    "with `pip install 'binscatter[multiway]'` or `pip install scipy`."
)


def require_sparse():
    """Return ``(scipy.sparse, scipy.sparse.linalg)``, or explain how to get them.

    Reaching this without scipy is a bug rather than a user error: callers check
    :func:`sparse_available` first and fall back to one-hot encoding. The message is
    here so that a caller added later fails legibly instead of with a bare
    ``ModuleNotFoundError``.
    """
    try:
        from scipy import sparse
        from scipy.sparse import linalg as sparse_linalg
    except ImportError as err:  # pragma: no cover - needs scipy uninstalled
        raise ImportError(_SCIPY_HINT) from err
    return sparse, sparse_linalg


def sparse_available() -> bool:
    """Whether scipy is importable, and so whether multi-way absorption can run."""
    try:
        require_sparse()
    except ImportError:  # pragma: no cover - needs scipy uninstalled
        logger.debug("[sparse_fe] scipy unavailable, capping absorption at one factor")
        return False
    return True


@dataclass
class MultiFEMoments:
    """Sufficient statistics for a set of jointly absorbed factors.

    Levels of every factor are stacked into one parameter vector of length ``L``;
    ``offsets[i]:offsets[i + 1]`` is the slice belonging to factor ``i``.

    Attributes:
        names: the absorbed column names, in stacking order.
        offsets: ``F + 1`` boundaries into the stacked level axis.
        counts: ``(L,)`` observations per level.
        dtd: ``(L, L)`` sparse ``D'D`` -- diagonal counts plus incidence blocks.
        dtb: ``(L, J)`` sparse ``D'B``, the level-by-bin crosstab.
        dtw: ``(L, q)`` dense ``D'W``, group sums of the regression features.
        response_sums: ``D'y`` per response name.
        total_count: total number of observations.
    """

    names: tuple[str, ...]
    offsets: tuple[int, ...]
    counts: np.ndarray
    dtd: Any
    dtb: Any
    dtw: np.ndarray
    response_sums: dict[str, np.ndarray]
    total_count: float

    @property
    def num_levels(self) -> int:
        return int(self.counts.size)

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(
            self.offsets[i + 1] - self.offsets[i] for i in range(len(self.names))
        )


@dataclass(frozen=True)
class AbsorbedSolution:
    """The pieces of a joint solve the callers actually consume.

    ``beta`` and ``alpha`` are each identified only up to a shift; ``level`` is the
    count-weighted mean fixed effect, and ``beta + level`` is invariant to it.
    """

    beta: np.ndarray
    gamma: np.ndarray
    alpha: np.ndarray
    level: float
    #: ``theta @ X'y``, the explained sum of squares. ``sum(y**2) - explained_ss`` is
    #: the residual sum of squares, which the rule-of-thumb selector needs.
    explained_ss: float


def _check_crosstab_budget(
    names: tuple[str, ...],
    cardinalities: Sequence[int],
    num_bins: int,
    total_count: float,
) -> None:
    """Raise before collecting incidence blocks that cannot fit on the driver.

    ``nnz(D_i'D_j) <= min(n, G_i * G_j)`` is exact as a bound and usually loose, so
    this over-estimates. Over-estimating is the right direction: the alternative is
    discovering the true size during the allocation that kills the process.
    """
    num_factors = len(cardinalities)
    estimate = float(sum(cardinalities))
    for i in range(num_factors):
        for j in range(i + 1, num_factors):
            estimate += min(total_count, float(cardinalities[i]) * cardinalities[j])
    if num_bins:
        for card in cardinalities:
            estimate += min(total_count, float(card) * num_bins)

    if estimate <= MAX_CROSSTAB_ENTRIES:
        logger.debug(
            "[sparse_fe] crosstab budget ok: ~%.0f entries for %s",
            estimate,
            dict(zip(names, cardinalities)),
        )
        return

    detail = ", ".join(f"{n} ({c:,} levels)" for n, c in zip(names, cardinalities))
    msg = (
        f"Absorbing {detail} jointly would need on the order of {estimate:,.0f} "
        f"crosstab entries on the driver, above the {MAX_CROSSTAB_ENTRIES:,} limit. "
        "Two high-cardinality factors that are close to one-to-one (worker x firm, "
        "patient x hospital) are the case this method cannot serve -- the crosstab "
        "is then roughly one entry per row. Drop one of the columns from controls, "
        "or coarsen it."
    )
    raise ValueError(msg)


def _map_to_positions(column: nw.Series, index: Mapping[Any, int]) -> np.ndarray:
    """Map a crosstab key column onto positions in the stacked level axis.

    Goes through ``.to_list()`` because a crosstab key can be any label type the
    backend allows, and ``.to_numpy()`` on a string column lands on dtypes whose
    scalars do not compare equal to the Python objects the index was built from.
    """
    values = column.to_list()
    return np.fromiter(
        (index[value] for value in values), dtype=np.int64, count=len(values)
    )


def collect_multi_fe_moments(
    df: nw.LazyFrame,
    *,
    fe_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    response_exprs: Mapping[str, nw.Expr],
    bin_name: str | None = None,
    bin_labels: Sequence[Any] | None = None,
) -> MultiFEMoments:
    """Collect every block of the multi-way normal equations that touches ``D``.

    ``F`` per-factor aggregations for the counts and group sums, ``F(F-1)/2``
    pairwise crosstabs for the incidence blocks, and ``F`` more against the bin
    column when bins exist. All of them are plain ``group_by`` calls, so this stays
    on whatever backend the frame came from.

    ``bin_labels`` fixes the crosstab's column order to match the caller's per-bin
    frame; group and bin ordering out of ``group_by`` is not stable across backends,
    so both axes are mapped by label rather than by position.
    """
    sparse, _ = require_sparse()

    if len(fe_names) < 2:
        msg = f"collect_multi_fe_moments needs at least two factors, got {fe_names}."
        raise ValueError(msg)

    response_names = list(response_exprs)
    aliases = {name: f"__mfe_resp_{i}" for i, name in enumerate(response_names)}

    # Derived responses become columns before grouping: dask rejects composed
    # aggregations such as (col * col).sum() inside group_by.
    df = df.with_columns(
        *(response_exprs[name].alias(aliases[name]) for name in response_names)
    )

    group_aggs = [nw.len().alias(_FE_COUNT)]
    group_aggs += [
        nw.col(aliases[name]).sum().alias(aliases[name]) for name in response_names
    ]
    group_aggs += [nw.col(c).sum().alias(c) for c in feature_names]

    indices: list[dict[Any, int]] = []
    offsets: list[int] = [0]
    count_blocks: list[np.ndarray] = []
    feature_blocks: list[np.ndarray] = []
    response_blocks: dict[str, list[np.ndarray]] = {n: [] for n in response_names}

    for fe_name in fe_names:
        grouped = df.group_by(fe_name).agg(*group_aggs).collect()
        labels = grouped.get_column(fe_name).to_list()
        if not labels:
            raise ValueError(f"No groups found for fixed effect column '{fe_name}'.")
        indices.append({label: i for i, label in enumerate(labels)})
        offsets.append(offsets[-1] + len(labels))
        count_blocks.append(
            np.asarray(grouped.get_column(_FE_COUNT).to_list(), dtype=float)
        )
        for name in response_names:
            response_blocks[name].append(
                np.asarray(grouped.get_column(aliases[name]).to_list(), dtype=float)
            )
        if feature_names:
            feature_blocks.append(
                np.column_stack(
                    [
                        np.asarray(grouped.get_column(c).to_list(), dtype=float)
                        for c in feature_names
                    ]
                )
            )
        else:
            feature_blocks.append(np.zeros((len(labels), 0), dtype=float))

    counts = np.concatenate(count_blocks)
    total_count = float(counts[offsets[0] : offsets[1]].sum())
    cardinalities = [offsets[i + 1] - offsets[i] for i in range(len(fe_names))]
    num_bins = 0 if bin_labels is None else len(bin_labels)
    _check_crosstab_budget(fe_names, cardinalities, num_bins, total_count)

    num_levels = offsets[-1]
    dtw = (
        np.vstack(feature_blocks)
        if feature_names
        else np.zeros((num_levels, 0), dtype=float)
    )
    response_sums = {
        name: np.concatenate(blocks) for name, blocks in response_blocks.items()
    }

    # D'D: level counts on the diagonal, pairwise incidence off it. Each pair is
    # collected once and written into both the (i, j) and (j, i) blocks.
    rows = [np.arange(num_levels, dtype=np.int64)]
    cols = [np.arange(num_levels, dtype=np.int64)]
    vals = [counts]
    for i in range(len(fe_names)):
        for j in range(i + 1, len(fe_names)):
            pair = (
                df.group_by(fe_names[i], fe_names[j])
                .agg(nw.len().alias(_FE_COUNT))
                .collect()
            )
            left = _map_to_positions(pair.get_column(fe_names[i]), indices[i])
            right = _map_to_positions(pair.get_column(fe_names[j]), indices[j])
            cell = np.asarray(pair.get_column(_FE_COUNT).to_list(), dtype=float)
            left = left + offsets[i]
            right = right + offsets[j]
            rows.extend((left, right))
            cols.extend((right, left))
            vals.extend((cell, cell))
            logger.debug(
                "[sparse_fe] crosstab %s x %s: %d cells",
                fe_names[i],
                fe_names[j],
                cell.size,
            )

    dtd = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(num_levels, num_levels),
    ).tocsr()

    # D'B: one crosstab per factor against the bin column.
    if bin_name is None or bin_labels is None:
        dtb = sparse.csr_matrix((num_levels, 0))
    else:
        bin_index = {label: i for i, label in enumerate(bin_labels)}
        b_rows: list[np.ndarray] = []
        b_cols: list[np.ndarray] = []
        b_vals: list[np.ndarray] = []
        for i, fe_name in enumerate(fe_names):
            crosstab = (
                df.group_by(bin_name, fe_name).agg(nw.len().alias(_FE_COUNT)).collect()
            )
            level_pos = _map_to_positions(crosstab.get_column(fe_name), indices[i])
            bin_pos = _map_to_positions(crosstab.get_column(bin_name), bin_index)
            b_rows.append(level_pos + offsets[i])
            b_cols.append(bin_pos)
            b_vals.append(
                np.asarray(crosstab.get_column(_FE_COUNT).to_list(), dtype=float)
            )
        dtb = sparse.coo_matrix(
            (
                np.concatenate(b_vals),
                (np.concatenate(b_rows), np.concatenate(b_cols)),
            ),
            shape=(num_levels, len(bin_labels)),
        ).tocsr()

    logger.debug(
        "[sparse_fe] factors=%s levels=%s dtd_nnz=%d dtb_nnz=%d",
        fe_names,
        cardinalities,
        dtd.nnz,
        dtb.nnz,
    )

    return MultiFEMoments(
        names=fe_names,
        offsets=tuple(offsets),
        counts=counts,
        dtd=dtd,
        dtb=dtb,
        dtw=dtw,
        response_sums=response_sums,
        total_count=total_count,
    )


def solve_absorbed_system(
    moments: MultiFEMoments,
    *,
    bin_block: np.ndarray,
    bin_controls: np.ndarray,
    control_block: np.ndarray,
    bin_y: np.ndarray,
    control_y: np.ndarray,
    response: str = "y",
) -> AbsorbedSolution:
    """Solve ``[B W D]'[B W D] theta = [B W D]'y`` sparsely.

    The caller supplies the blocks that do not involve ``D`` -- it has already paid
    for them on the unabsorbed path -- and this adds the level blocks from
    ``moments``. Pass empty bin blocks to fit a design with no bin dummies, which is
    what the polynomial overlay and the rule-of-thumb selector want.
    """
    sparse, sparse_linalg = require_sparse()

    num_bins = int(bin_block.shape[0])
    num_controls = int(control_block.shape[0])
    dty = moments.response_sums[response]

    top = [
        sparse.csr_matrix(np.asarray(bin_block, dtype=float)),
        sparse.csr_matrix(np.asarray(bin_controls, dtype=float)),
        moments.dtb.T,
    ]
    middle = [
        sparse.csr_matrix(np.asarray(bin_controls, dtype=float).T),
        sparse.csr_matrix(np.asarray(control_block, dtype=float)),
        sparse.csr_matrix(moments.dtw.T),
    ]
    bottom = [moments.dtb, sparse.csr_matrix(moments.dtw), moments.dtd]
    xtx = sparse.bmat([top, middle, bottom], format="csr")
    xty = np.concatenate(
        [
            np.asarray(bin_y, dtype=float).reshape(num_bins),
            np.asarray(control_y, dtype=float).reshape(num_controls),
            dty,
        ]
    )

    # Jacobi scaling. The blocks live on wildly different scales -- bin counts,
    # control cross-products and level counts -- and solving the normal equations
    # already squares the condition number, so symmetric diagonal preconditioning is
    # worth the three lines.
    diagonal = np.asarray(xtx.diagonal(), dtype=float)
    scale = np.where(
        diagonal > 0.0, 1.0 / np.sqrt(np.where(diagonal > 0.0, diagonal, 1.0)), 1.0
    )
    scaler = sparse.diags(scale)
    scaled = scaler @ xtx @ scaler

    result = sparse_linalg.lsmr(
        scaled,
        scale * xty,
        atol=1e-14,
        btol=1e-14,
        conlim=0.0,
        maxiter=max(1000, 20 * xtx.shape[0]),
    )
    theta = scale * result[0]

    # The system is rank-deficient by F, so any least-squares solution is fine, but a
    # solve that stopped on the iteration limit is not: it means the answer is
    # whatever the iteration happened to reach.
    istop = int(result[1])
    if istop == 7:
        msg = (
            f"The sparse solve for the absorbed factors {moments.names} hit its "
            "iteration limit without converging. This usually means the factors are "
            "very poorly connected. Reduce the number of absorbed columns, or pass "
            "the controls one-hot encoded instead."
        )
        raise RuntimeError(msg)

    beta = theta[:num_bins]
    gamma = theta[num_bins : num_bins + num_controls]
    alpha = theta[num_bins + num_controls :]
    level = float(moments.counts @ alpha / moments.total_count)
    logger.debug(
        "[sparse_fe] lsmr istop=%d itn=%d unknowns=%d level=%.6g",
        istop,
        int(result[2]),
        xtx.shape[0],
        level,
    )
    return AbsorbedSolution(
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        level=level,
        explained_ss=float(theta @ xty),
    )
