#!/usr/bin/env python3
"""PS-IPM with a stable active-column Schur preconditioner.

The IPM iterates are those of the full canonical LP.  Columns are removed only
from the sparse Cholesky preconditioner, never from the Newton operator.  For

    H = A D A.T + delta I,                 D = diag(1 / theta),

the preconditioner is

    P = A_C D_C A_C.T
        + diag(A_N D_N A_N.T) + delta I,

where C is a stable, fill-aware active set and N contains the omitted coupling
columns.  One-entry columns are already represented exactly by the diagonal.
Very dense columns can be kept out of CHOLMOD and restored exactly with the
base solver's Woodbury correction.  Preconditioned CG applies H matrix-free,
so a converged linear solve is a solve of the full regularized KKT system.

The active set is refreshed only when it is old or Krylov convergence becomes
poor.  A refresh retains a configurable fraction of the previous columns and
fills the remaining budget using Jacobi-scaled column leverage proxies divided
by a symbolic clique-cost penalty.  This directly addresses the changing-fill
bottleneck seen in ``psipm_active_pool`` on very large models.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

import psipm_active_pool as PS


RawLP = PS.RawLP
StandardLP = PS.StandardLP
Solution = PS.Solution
INF = PS.INF
read_mps = PS.read_mps
presolve_raw = PS.presolve_raw
to_standard_form = PS.to_standard_form
recover = PS.recover


def scale_problem(lp: StandardLP, verbose: bool = True) -> StandardLP:
    """Apply the IPX/HiPO-compatible two-sided max-norm equilibration policy.

    HiGHS' interior-point paths equilibrate rows and columns before solving.
    Mode 5 in the shared PS-IPM engine performs the corresponding ten-sweep
    Ruiz max-norm equilibration and records both transformations, including
    the row scaling needed to recover dual multipliers correctly.
    """
    return PS.scale_problem(lp, mode=5, verbose=verbose)


def scaled_tolerance(
    lp: StandardLP, requested: float, max_tightening: float = 10.0
) -> float:
    """Guard original-space accuracy against two-sided scaling amplification.

    The norm bound can be extremely pessimistic because it assumes the largest
    residual occurs on the most strongly scaled row or column.  Tightening is
    therefore capped; the mandatory original-space postsolve check remains the
    final acceptance test.
    """
    requested = float(requested)
    if requested <= 0.0:
        raise ValueError("tolerance must be positive")
    column_scale = np.asarray(lp.scale, dtype=float)
    row_scale = np.asarray(lp.row_scale, dtype=float)
    c_unscaled = lp.c / column_scale
    b_unscaled = lp.b / row_scale
    dual_amplification = (
        (1.0 + float(np.abs(lp.c).max(initial=0.0)))
        * float((1.0 / column_scale).max(initial=1.0))
        / (1.0 + float(np.abs(c_unscaled).max(initial=0.0)))
    )
    primal_amplification = (
        (1.0 + float(np.abs(lp.b).max(initial=0.0)))
        * float((1.0 / row_scale).max(initial=1.0))
        / (1.0 + float(np.abs(b_unscaled).max(initial=0.0)))
    )
    amplification = min(
        max(float(max_tightening), 1.0),
        max(1.0, dual_amplification, primal_amplification),
    )
    return requested / amplification


def _top_k(score: np.ndarray, count: int) -> np.ndarray:
    """Indices of the largest finite entries, deterministically ordered."""
    count = min(max(0, int(count)), score.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    finite = np.flatnonzero(np.isfinite(score))
    if finite.size <= count:
        chosen = finite
    else:
        local = np.argpartition(score[finite], -count)[-count:]
        chosen = finite[local]
    # Score first, then index, makes ties and diagnostics reproducible.
    order = np.lexsort((chosen, -score[chosen]))
    return chosen[order].astype(np.int64, copy=False)


class StableActiveColumnPreconditioner(PS.NormalEquations):
    r"""Exact matrix-free Schur solves with a stable sparse column factor.

    ``core_factor * m`` is a budget, not a sample size: singleton and extracted
    low-rank columns do not consume it.  The omitted diagonal is exact, which
    keeps ``P`` positive definite under the same proximal ``delta`` as the full
    system.  There is intentionally no full-normal-equations fallback; a bad
    preconditioner is refreshed and enlarged once instead of constructing the
    fill-heavy matrix this method is meant to avoid.
    """

    def __init__(
        self,
        A: sp.csc_matrix,
        protected: Optional[np.ndarray] = None,
        core_factor: float = 1.0,
        max_core_factor: float = 3.0,
        retention: float = 0.9,
        factor_max_age: int = 2,
        core_max_age: int = 12,
        refresh_cg: int = 30,
        fill_weight: float = 0.2,
        cg_tol: float = 1e-7,
        cg_tol_max: float = 1e-2,
        forcing_factor: float = 0.1,
        cg_maxit: int = 200,
        retry_growth: float = 1.35,
        matching_backbone: bool = True,
        full_preconditioner_row_threshold: int = 1500,
        full_preconditioner_clique_limit: float = 2e8,
        full_preconditioner_nnz_limit: int = 10_000_000,
        adaptive_refresh_row_threshold: int = 20_000,
        **kwargs,
    ):
        if core_factor <= 0.0:
            raise ValueError("core_factor must be positive")
        if max_core_factor < core_factor:
            raise ValueError("max_core_factor must be at least core_factor")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("retention must lie in [0, 1]")
        if fill_weight < 0.0:
            raise ValueError("fill_weight must be nonnegative")
        if retry_growth <= 1.0:
            raise ValueError("retry_growth must exceed one")
        if not 0.0 < cg_tol <= cg_tol_max < 1.0:
            raise ValueError("require 0 < cg_tol <= cg_tol_max < 1")
        if forcing_factor <= 0.0:
            raise ValueError("forcing_factor must be positive")

        super().__init__(A, **kwargs)
        # _KKTBase counters are class defaults; reset every experimental object.
        self.n_fact = 0
        self.n_solve = 0
        self.t_fact = 0.0
        self.t_solve = 0.0

        factor_counts = np.diff(self.A_factor.indptr).astype(np.int64)
        coupling_local = np.flatnonzero(factor_counts >= 2)
        self.candidates = self.factor_columns[coupling_local]
        self.A_coupling = self.A_factor[:, coupling_local].tocsc()
        self.A2_coupling = sp.csc_matrix(
            (
                np.square(self.A_coupling.data),
                self.A_coupling.indices,
                self.A_coupling.indptr,
            ),
            shape=self.A_coupling.shape,
            copy=False,
        )
        self.A2_factor = sp.csc_matrix(
            (
                np.square(self.A_factor.data),
                self.A_factor.indices,
                self.A_factor.indptr,
            ),
            shape=self.A_factor.shape,
            copy=False,
        )
        self.A2_low_rank = sp.csc_matrix(
            (
                np.square(self.A_low_rank.data),
                self.A_low_rank.indices,
                self.A_low_rank.indptr,
            ),
            shape=self.A_low_rank.shape,
            copy=False,
        )
        counts = factor_counts[coupling_local].astype(float)
        self.clique_burden = 0.5 * counts * np.maximum(counts - 1.0, 0.0)
        self.use_full_preconditioner = bool(
            self.m <= max(0, int(full_preconditioner_row_threshold))
            and self.A_factor.nnz <= max(0, int(full_preconditioner_nnz_limit))
            and float(self.clique_burden.sum())
            <= float(full_preconditioner_clique_limit)
        )

        protected_mask = np.zeros(self.n, dtype=bool)
        if protected is not None:
            supplied = np.asarray(protected, dtype=bool)
            if supplied.shape != (self.n,):
                raise ValueError("protected mask has the wrong dimension")
            protected_mask[:] = supplied
        protected_local = protected_mask[self.candidates]
        mandatory = self.candidates[protected_local]
        self.matching_columns = np.empty(0, dtype=np.int64)
        if (
            matching_backbone
            and not self.use_full_preconditioner
            and self.m
            and self.factor_columns.size
        ):
            structure = self.A_factor.tocsr(copy=True)
            structure.data.fill(1.0)
            matched_local = csgraph.maximum_bipartite_matching(
                structure, perm_type="column"
            )
            matched_local = matched_local[matched_local >= 0]
            matched_global = self.factor_columns[matched_local]
            # Singleton columns are already exact in the diagonal correction.
            positions = np.searchsorted(self.candidates, matched_global)
            valid = positions < self.candidates.size
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices] &= (
                self.candidates[positions[valid_indices]]
                == matched_global[valid_indices]
            )
            self.matching_columns = np.unique(matched_global[valid])
            mandatory = np.union1d(mandatory, self.matching_columns)
        self.mandatory = mandatory

        requested = max(1, int(np.ceil(float(core_factor) * self.m)))
        maximum = max(requested, int(np.ceil(float(max_core_factor) * self.m)))
        if self.use_full_preconditioner:
            requested = maximum = self.candidates.size
        self.core_target = min(self.candidates.size, max(requested, self.mandatory.size))
        self.max_core_target = min(
            self.candidates.size, max(maximum, self.mandatory.size)
        )
        self.retention = float(retention)
        self.factor_max_age = (
            0 if self.use_full_preconditioner else max(0, int(factor_max_age))
        )
        self.core_max_age = max(1, int(core_max_age))
        self.refresh_cg = max(1, int(refresh_cg))
        self.fill_weight = float(fill_weight)
        self.cg_tol = float(cg_tol)
        self.cg_tol_max = float(cg_tol_max)
        self.forcing_factor = float(forcing_factor)
        self.current_cg_tol = self.cg_tol
        self.cg_maxit = max(1, int(cg_maxit))
        self.retry_growth = float(retry_growth)
        self.adaptive_refresh_row_threshold = max(
            0, int(adaptive_refresh_row_threshold)
        )

        self.core = np.empty(0, dtype=np.int64)
        self._weights = None
        self._diag_drop = np.zeros(self.m)
        self.factor_age = self.factor_max_age
        self.core_age = self.core_max_age
        self._initialized = False
        self._last_cg = self.refresh_cg
        self.core_sizes = []
        self.cg_iterations = []
        self.cg_failures = 0
        self.pcg_retries = 0
        self.full_fallbacks = 0
        self.refreshes = 0
        self.retained_fractions = []
        self.preconditioner_nnz = []
        self.cg_tolerances = []

    def observe(self, x, z, w, s, mu) -> None:
        """Set an inexact-Newton forcing tolerance from complementarity."""
        normalized = max(float(mu), 0.0) / (1.0 + max(float(mu), 0.0))
        forcing = self.forcing_factor * normalized
        self.current_cg_tol = min(
            self.cg_tol_max, max(self.cg_tol, float(forcing))
        )

    def _scores(self, weights: np.ndarray, delta: float) -> np.ndarray:
        if not self.candidates.size:
            return np.empty(0)
        candidate_weights = weights[self.candidates]
        diagonal = np.asarray(
            self.A2_factor @ weights[self.factor_columns]
        ).ravel()
        if self.low_rank_columns.size:
            diagonal += np.asarray(
                self.A2_low_rank @ weights[self.low_rank_columns]
            ).ravel()
        diagonal += max(float(delta), np.finfo(float).tiny)
        inverse_diagonal = 1.0 / np.maximum(diagonal, np.finfo(float).tiny)
        leverage = candidate_weights * np.asarray(
            self.A2_coupling.T @ inverse_diagonal
        ).ravel()
        return leverage / (1.0 + self.fill_weight * self.clique_burden)

    def _refresh_core(self, weights: np.ndarray, delta: float, stable: bool) -> None:
        if self.use_full_preconditioner:
            old = self.core
            self.core = self.candidates.copy()
            self.retained_fractions.append(1.0 if old.size else 1.0)
            self.refreshes += 1
            self.core_age = 0
            self._initialized = True
            return
        scores = self._scores(weights, delta)
        target = min(self.core_target, self.candidates.size)
        old = self.core

        selected_local = []
        if self.mandatory.size:
            mandatory_local = np.searchsorted(self.candidates, self.mandatory)
            selected_local.append(mandatory_local)

        if stable and old.size and target:
            old_local = np.searchsorted(self.candidates, old)
            keep_count = min(
                old.size,
                max(0, int(np.floor(self.retention * target))),
            )
            if keep_count:
                keep_rank = _top_k(scores[old_local], keep_count)
                selected_local.append(old_local[keep_rank])

        if selected_local:
            fixed = np.unique(np.concatenate(selected_local))
        else:
            fixed = np.empty(0, dtype=np.int64)
        if fixed.size > target:
            mandatory_count = self.mandatory.size
            if mandatory_count >= target:
                fixed = np.searchsorted(self.candidates, self.mandatory)[:target]
            else:
                mandatory_local = np.searchsorted(self.candidates, self.mandatory)
                is_mandatory = np.isin(fixed, mandatory_local, assume_unique=True)
                optional = fixed[~is_mandatory]
                rank = _top_k(scores[optional], target - mandatory_count)
                fixed = np.concatenate([mandatory_local, optional[rank]])

        remaining = target - fixed.size
        if remaining > 0:
            available_scores = scores.copy()
            available_scores[fixed] = -np.inf
            fixed = np.concatenate([fixed, _top_k(available_scores, remaining)])

        new_core = np.sort(self.candidates[np.unique(fixed)])
        retained = (
            float(np.intersect1d(old, new_core, assume_unique=True).size)
            / max(1, old.size)
            if old.size
            else 1.0
        )
        self.core = new_core
        self.retained_fractions.append(retained)
        self.refreshes += 1
        self.core_age = 0
        self._initialized = True

    def _assemble(self, theta: np.ndarray, delta: float) -> sp.csc_matrix:
        weights = self._weights
        if self.core.size:
            Ac = self.A[:, self.core].tocsc()
            matrix = PS._column_weighted_normal(Ac, weights[self.core])
            core_local = np.searchsorted(self.candidates, self.core)
            diag_core = np.asarray(
                self.A2_coupling[:, core_local] @ weights[self.core]
            ).ravel()
        else:
            matrix = sp.csc_matrix((self.m, self.m))
            diag_core = np.zeros(self.m)

        # A_factor excludes exact Woodbury columns.  Singleton columns need no
        # explicit factor column because their normal contribution is diagonal.
        factor_weights = weights[self.factor_columns]
        diag_full = np.asarray(self.A2_factor @ factor_weights).ravel()
        self._diag_drop = np.maximum(diag_full - diag_core, 0.0)
        matrix = (
            matrix + sp.diags(self._diag_drop + float(delta), format="csc")
        ).tocsc()
        matrix.sort_indices()
        return matrix

    def factorize(self, theta: np.ndarray, delta: float) -> None:
        theta = np.asarray(theta, dtype=float)
        self._weights = 1.0 / theta
        poor_krylov = self._last_cg >= self.refresh_cg
        can_grow = self.core_target < self.max_core_target
        refresh_core = (
            not self._initialized
            or self.core_age >= self.core_max_age
            or (
                poor_krylov
                and (
                    can_grow
                    or self.m <= self.adaptive_refresh_row_threshold
                )
            )
        )
        refactor = (
            self._factor is None
            or self.factor_age >= self.factor_max_age
            or refresh_core
            or poor_krylov
        )
        if refresh_core:
            self._refresh_core(self._weights, delta, stable=bool(self.core.size))
        if refactor:
            super().factorize(theta, delta)
            self.factor_age = 0
            self.preconditioner_nnz.append(self.nnz_factor)
        else:
            # Keep the old SPD preconditioner, but solve the current full
            # operator.  This is an exact lagged-preconditioning step.
            self.theta = theta
            self.delta = float(delta)
            self.factor_age += 1
        self.core_age += 1
        self.core_sizes.append(int(self.core.size))

    def _full_matvec(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.A @ (self._weights * (self.At @ vector))
            + self.delta * vector
        ).ravel()

    def _pcg(self, rhs: np.ndarray):
        iterations = 0

        def count(_):
            nonlocal iterations
            iterations += 1

        operator = spla.LinearOperator((self.m, self.m), matvec=self._full_matvec)
        preconditioner = spla.LinearOperator((self.m, self.m), matvec=self._apply)
        initial = self._apply(rhs)
        solution, info = spla.cg(
            operator,
            rhs,
            x0=initial,
            M=preconditioner,
            rtol=self.current_cg_tol,
            atol=0.0,
            maxiter=self.cg_maxit,
            callback=count,
        )
        scale = max(float(np.linalg.norm(rhs)), 1.0)
        relative_residual = float(
            np.linalg.norm(rhs - self._full_matvec(solution)) / scale
        )
        return solution, info, iterations, relative_residual

    def solve(self, r1: np.ndarray, r2: np.ndarray):
        started = time.perf_counter()
        factor_before = self.t_fact
        w = r1 / self.theta
        rhs = np.asarray(r2 - self.A @ w).ravel()
        dy, info, iterations, relative_residual = self._pcg(rhs)

        self.cg_tolerances.append(self.current_cg_tol)
        acceptable = max(self.fail_tol, 2.0 * self.current_cg_tol)
        failed = (
            info != 0
            or not np.isfinite(relative_residual)
            or relative_residual > acceptable
        )
        if failed and self.core_target < self.max_core_target:
            self.pcg_retries += 1
            grown = max(self.core_target + 1, int(np.ceil(
                self.retry_growth * self.core_target
            )))
            self.core_target = min(self.max_core_target, grown)
            self._refresh_core(self._weights, self.delta, stable=True)
            super().factorize(self.theta, self.delta)
            self.factor_age = 0
            self.core_sizes.append(int(self.core.size))
            self.preconditioner_nnz.append(self.nnz_factor)
            dy, info, retry_iterations, relative_residual = self._pcg(rhs)
            iterations += retry_iterations
            failed = (
                info != 0
                or not np.isfinite(relative_residual)
                or relative_residual > acceptable
            )

        self.cg_iterations.append(iterations)
        self._last_cg = iterations
        if failed:
            self.cg_failures += 1
            self.t_solve += max(
                0.0, time.perf_counter() - started - (self.t_fact - factor_before)
            )
            raise PS.KKTSolveError(
                "stable active-column PCG failed: "
                f"info={info}, relative residual={relative_residual:.2e}, "
                f"iterations={iterations}"
            )

        dx = w + (self.At @ dy) / self.theta
        self.n_solve += 1
        self.t_solve += max(
            0.0, time.perf_counter() - started - (self.t_fact - factor_before)
        )
        return dx, dy


def solve_standard(
    lp: StandardLP,
    core_factor: float = 1.0,
    max_core_factor: float = 3.0,
    retention: float = 0.9,
    factor_max_age: int = 2,
    core_max_age: int = 12,
    refresh_cg: int = 30,
    fill_weight: float = 0.2,
    cg_tol: float = 1e-7,
    cg_tol_max: float = 1e-2,
    forcing_factor: float = 0.1,
    cg_maxit: int = 200,
    retry_growth: float = 1.35,
    matching_backbone: bool = True,
    full_preconditioner_row_threshold: int = 1500,
    full_preconditioner_clique_limit: float = 2e8,
    full_preconditioner_nnz_limit: int = 10_000_000,
    adaptive_refresh_row_threshold: int = 20_000,
    order: str = "best",
    refine: int = 1,
    low_rank_max_columns: int = 8,
    low_rank_min_nnz: int = 1024,
    low_rank_clique_ratio: float = 2.0,
    **kwargs,
) -> Solution:
    """Solve one already canonicalized and scaled LP."""
    printlevel = int(kwargs.get("printlevel", 1))
    solver = StableActiveColumnPreconditioner(
        lp.A,
        protected=~lp.pos,
        core_factor=core_factor,
        max_core_factor=max_core_factor,
        retention=retention,
        factor_max_age=factor_max_age,
        core_max_age=core_max_age,
        refresh_cg=refresh_cg,
        fill_weight=fill_weight,
        cg_tol=cg_tol,
        cg_tol_max=cg_tol_max,
        forcing_factor=forcing_factor,
        cg_maxit=cg_maxit,
        retry_growth=retry_growth,
        matching_backbone=matching_backbone,
        full_preconditioner_row_threshold=full_preconditioner_row_threshold,
        full_preconditioner_clique_limit=full_preconditioner_clique_limit,
        full_preconditioner_nnz_limit=full_preconditioner_nnz_limit,
        adaptive_refresh_row_threshold=adaptive_refresh_row_threshold,
        order=order,
        refine=refine,
        low_rank_max_columns=low_rank_max_columns,
        low_rank_min_nnz=low_rank_min_nnz,
        low_rank_clique_ratio=low_rank_clique_ratio,
    )
    if printlevel:
        if solver.use_full_preconditioner:
            print(
                "  preconditioner : exact Schur (small-row structural gate)"
            )
        else:
            print(
                f"  preconditioner : stable active core {solver.core_target}/"
                f"{solver.candidates.size}, matching backbone "
                f"{solver.matching_columns.size}, max core "
                f"{solver.max_core_target}"
            )
    return PS.psipm_solve(
        lp,
        kkt="stable-active-pcg",
        kkt_solver=solver,
        order=order,
        refine=refine,
        low_rank_max_columns=low_rank_max_columns,
        low_rank_min_nnz=low_rank_min_nnz,
        low_rank_clique_ratio=low_rank_clique_ratio,
        **kwargs,
    )


def solve_mps(
    path: str,
    tol: float = 1e-6,
    printlevel: int = 1,
    presolve: bool = False,
    **kwargs,
):
    """Read, equilibrate, solve, recover, and optionally HiGHS-postsolve."""
    started = time.perf_counter()
    presolver = None
    pstatus = "off"
    raw_orig = read_mps(path)
    raw = raw_orig
    if presolve:
        raw, presolver, pstatus = presolve_raw(
            raw_orig, time_limit=kwargs.get("time_limit", np.inf)
        )
    if printlevel:
        print(f"PS-IPM-SAP  |  {path}")
        print(
            f"  original LP    : m = {raw_orig.A.shape[0]}, "
            f"n = {raw_orig.A.shape[1]}, nnz = {raw_orig.A.nnz} "
            f"(read/presolve {time.perf_counter() - started:.2f}s)"
        )
        if presolve:
            print(f"  HiGHS presolve : {pstatus}")
    lp = to_standard_form(raw, verbose=bool(printlevel))
    scale_problem(lp, verbose=bool(printlevel))
    internal_tol = scaled_tolerance(lp, tol)
    if printlevel and internal_tol < tol:
        print(
            f"  scaled tol     : {internal_tol:.3e} "
            f"(original target {tol:.3e})"
        )
    sol = solve_standard(lp, tol=internal_tol, printlevel=printlevel, **kwargs)
    info_pre = recover(lp, raw, sol)
    info = (
        PS._postsolve_info(presolver, raw, raw_orig, info_pre)
        if presolver is not None and raw is not raw_orig
        else info_pre
    )
    if sol.status == "optimal" and max(info["pinf"], info["dinf"]) > tol:
        sol.status = "postsolve-inaccurate"
    sol.obj = info["obj"]
    return raw_orig, lp, sol, info


def _parse_args():
    parser = argparse.ArgumentParser(
        description="PS-IPM with stable active-column Schur preconditioning"
    )
    parser.add_argument("model")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=2000.0)
    parser.add_argument("--maxit", type=int, default=200)
    parser.add_argument("--inner-maxit", type=int, default=100)
    parser.add_argument("--pc", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--presolve", action="store_true")
    parser.add_argument("--core-factor", type=float, default=1.0)
    parser.add_argument("--max-core-factor", type=float, default=3.0)
    parser.add_argument("--retention", type=float, default=0.9)
    parser.add_argument("--factor-max-age", type=int, default=2)
    parser.add_argument("--core-max-age", type=int, default=12)
    parser.add_argument("--refresh-cg", type=int, default=30)
    parser.add_argument("--fill-weight", type=float, default=0.2)
    parser.add_argument("--cg-tol", type=float, default=1e-7)
    parser.add_argument("--cg-tol-max", type=float, default=1e-2)
    parser.add_argument("--forcing-factor", type=float, default=0.1)
    parser.add_argument("--cg-maxit", type=int, default=200)
    parser.add_argument("--retry-growth", type=float, default=1.35)
    parser.add_argument(
        "--no-matching-backbone", action="store_true",
        help="disable the structural matching backbone (diagnostic only)",
    )
    parser.add_argument("--full-preconditioner-row-threshold", type=int, default=1500)
    parser.add_argument("--full-preconditioner-clique-limit", type=float, default=2e8)
    parser.add_argument("--full-preconditioner-nnz-limit", type=int, default=10_000_000)
    parser.add_argument("--adaptive-refresh-row-threshold", type=int, default=20_000)
    parser.add_argument("--order", default="best")
    parser.add_argument("--printlevel", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _, _, solution, check = solve_mps(
        str(Path(args.model).resolve()),
        tol=args.tol,
        time_limit=args.time_limit,
        maxit=args.maxit,
        inner_maxit=args.inner_maxit,
        pc=args.pc,
        presolve=args.presolve,
        core_factor=args.core_factor,
        max_core_factor=args.max_core_factor,
        retention=args.retention,
        factor_max_age=args.factor_max_age,
        core_max_age=args.core_max_age,
        refresh_cg=args.refresh_cg,
        fill_weight=args.fill_weight,
        cg_tol=args.cg_tol,
        cg_tol_max=args.cg_tol_max,
        forcing_factor=args.forcing_factor,
        cg_maxit=args.cg_maxit,
        retry_growth=args.retry_growth,
        matching_backbone=not args.no_matching_backbone,
        full_preconditioner_row_threshold=args.full_preconditioner_row_threshold,
        full_preconditioner_clique_limit=args.full_preconditioner_clique_limit,
        full_preconditioner_nnz_limit=args.full_preconditioner_nnz_limit,
        adaptive_refresh_row_threshold=args.adaptive_refresh_row_threshold,
        order=args.order,
        printlevel=args.printlevel,
        compl="gap",
    )
    print(
        f"objective {check['obj']:.10e}  pinf {check['pinf']:.2e}  "
        f"dinf {check['dinf']:.2e}  time {solution.time:.2f}s"
    )
