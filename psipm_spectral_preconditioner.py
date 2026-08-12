#!/usr/bin/env python3
"""PS-IPM with a spectrally monitored stable-column preconditioner.

This is the second-generation stable active-column solver.  It keeps the full
regularized Schur operator matrix-free, but strengthens the sparse Cholesky
preconditioner with:

* a numerically weighted row/column matching backbone;
* residual-driven low-churn core replacement;
* Ritz estimates of preconditioned condition and spectral error;
* componentwise spectral certificates for lagged-factor reuse;
* PCG residual replacement, curvature checks and stagnation detection;
* an Eisenstat-Walker-style forcing term informed by the PS-IPM residual; and
* a regularized augmented-system MINRES recovery path.

The preconditioner remains

    P = A_C D_C A_C.T + diag(A_N D_N A_N.T) + delta I,

while every accepted direction solves the full Newton equations to a checked
true-residual tolerance.  No LP column is removed from the Newton operator.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

import psipm_active_pool as PS
import psipm_stable_preconditioner as SAP


RawLP = PS.RawLP
StandardLP = PS.StandardLP
Solution = PS.Solution
read_mps = PS.read_mps
presolve_raw = PS.presolve_raw
to_standard_form = PS.to_standard_form
recover = PS.recover
scaled_tolerance = SAP.scaled_tolerance


def scale_problem(lp: StandardLP, verbose: bool = True) -> StandardLP:
    """Apply bounded, converged two-sided infinity-norm equilibration.

    Unlike the shared legacy scaler, this path is not bypassed merely because
    every individual coefficient lies in [0.1, 10].  Cumulative factors are
    bounded to protect recovery from overflow.  This is the same scaling class
    as HiPO's infinity-norm equilibration, not a claim of bitwise identity.
    """
    Acur = lp.A.tocsc(copy=True)
    rows, columns = Acur.shape
    left = np.ones(rows)
    right = np.ones(columns)
    tiny = np.finfo(float).tiny
    lower, upper = 1e-12, 1e12

    for _ in range(12):
        absolute = abs(Acur)
        row_max = np.asarray(absolute.max(axis=1).todense()).ravel()
        col_max = np.asarray(absolute.max(axis=0).todense()).ravel()
        row_max[row_max == 0.0] = 1.0
        col_max[col_max == 0.0] = 1.0
        row_error = float(
            np.max(np.abs(np.log(np.maximum(row_max, tiny))), initial=0.0)
        )
        column_error = float(
            np.max(np.abs(np.log(np.maximum(col_max, tiny))), initial=0.0)
        )
        error = max(row_error, column_error)
        if error <= 0.05:
            break
        row_step = np.clip(1.0 / np.sqrt(row_max), 1e-4, 1e4)
        col_step = np.clip(1.0 / np.sqrt(col_max), 1e-4, 1e4)
        new_left = np.clip(left * row_step, lower, upper)
        new_right = np.clip(right * col_step, lower, upper)
        row_step = new_left / left
        col_step = new_right / right
        Acur = sp.diags(row_step) @ Acur @ sp.diags(col_step)
        left, right = new_left, new_right

    lp.A = Acur.tocsc()
    lp.A.sort_indices()
    lp.b = lp.b * left
    lp.c = lp.c * right
    lp.u = lp.u / right
    if lp.cert_u is not None:
        lp.cert_u = lp.cert_u / right
    lp.scale = lp.scale * right
    lp.row_scale = left if lp.row_scale is None else lp.row_scale * left
    if verbose:
        print("  scaling        : bounded infinity-norm equilibration")
    return lp


@dataclass
class KrylovResult:
    solution: np.ndarray
    info: int
    iterations: int
    relative_residual: float
    condition_estimate: float
    spectral_error_estimate: float
    residual: np.ndarray
    replacements: int
    residual_checks: int
    stagnated: bool


def _ritz_estimate(alphas, betas):
    """Return extremal Ritz estimates for the symmetrically preconditioned H."""
    count = len(alphas)
    if count == 0:
        return 1.0, 1.0, 1.0
    diagonal = np.empty(count)
    diagonal[0] = 1.0 / alphas[0]
    if count > 1:
        beta = np.asarray(betas[: count - 1])
        alpha = np.asarray(alphas)
        diagonal[1:] = 1.0 / alpha[1:] + beta / alpha[:-1]
        off_diagonal = np.sqrt(np.maximum(beta, 0.0)) / alpha[:-1]
        eigenvalues = la.eigvalsh_tridiagonal(
            diagonal, off_diagonal, check_finite=False
        )
    else:
        eigenvalues = diagonal
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    if not np.isfinite(smallest) or smallest <= 0.0:
        return smallest, largest, np.inf
    return smallest, largest, largest / smallest


class SpectralStableColumnPreconditioner(SAP.StableActiveColumnPreconditioner):
    """Stable active-column Cholesky preconditioner with spectral feedback."""

    def __init__(
        self,
        A: sp.csc_matrix,
        protected: Optional[np.ndarray] = None,
        weighted_backbone: bool = False,
        structural_backbone: bool = True,
        matching_max_rows: int = 50_000,
        matching_max_nnz: int = 25_000_000,
        matching_refresh_max_rows: int = 20_000,
        backbone_max_age: int = 20,
        residual_score_weight: float = 0.0,
        swap_patience: int = 2,
        condition_limit: float = 1e12,
        lag_spectral_ratio: float = 1e8,
        residual_replacement: int = 100,
        residual_gap_tolerance: float = 0.1,
        stagnation_window: int = 3,
        stagnation_reduction: float = 0.8,
        corrector_tightening: float = 1.0,
        minres_fallback: bool = True,
        minres_maxit: int = 400,
        minres_max_dimension: int = 200_000,
        **kwargs,
    ):
        super().__init__(
            A,
            protected=protected,
            matching_backbone=False,
            **kwargs,
        )
        self.weighted_backbone = bool(weighted_backbone)
        self.structural_backbone = bool(structural_backbone)
        self.matching_max_rows = max(0, int(matching_max_rows))
        self.matching_max_nnz = max(0, int(matching_max_nnz))
        self.matching_refresh_max_rows = max(0, int(matching_refresh_max_rows))
        self.backbone_max_age = max(1, int(backbone_max_age))
        self.residual_score_weight = max(0.0, float(residual_score_weight))
        self.swap_patience = max(1, int(swap_patience))
        self.condition_limit = max(1.0, float(condition_limit))
        self.lag_spectral_ratio = max(1.0, float(lag_spectral_ratio))
        self.residual_replacement = max(0, int(residual_replacement))
        self.residual_gap_tolerance = max(
            np.sqrt(np.finfo(float).eps), float(residual_gap_tolerance)
        )
        self.stagnation_window = max(2, int(stagnation_window))
        self.stagnation_reduction = float(stagnation_reduction)
        self.corrector_tightening = min(
            1.0, max(0.01, float(corrector_tightening))
        )
        self.minres_fallback = bool(minres_fallback)
        self.minres_maxit = max(1, int(minres_maxit))
        self.minres_max_dimension = max(0, int(minres_max_dimension))

        self.protected_mandatory = self.mandatory.copy()
        self.matching_columns = np.empty(0, dtype=np.int64)
        self.mandatory = self.protected_mandatory.copy()
        self.backbone_age = self.backbone_max_age
        self.weighted_backbone_refreshes = 0
        self.weighted_backbone_failures = 0
        self._enrichment_vector = None
        self._reference_weights = None
        self._reference_delta = None
        self._factor_generation = 0
        self._poor_streak = 0
        self._last_condition = 1.0
        self._last_spectral_error = 0.0
        self._last_stagnated = False
        self._rhs_index = 0
        self._nonlinear_residual = None
        self._previous_nonlinear_residual = None
        self._ew_tolerance = self.cg_tol_max

        self.condition_estimates = []
        self.spectral_error_estimates = []
        self.lag_spectral_bounds = []
        self.residual_replacements = 0
        self.residual_checks = 0
        self.stagnation_events = 0
        self.core_swaps = 0
        self.minres_fallbacks = 0
        self.minres_iterations = 0
        self.krylov_breakdowns = 0
        self.krylov_iteration_limits = 0

    def set_ipm_context(self, res_d, res_p, res_n, tol) -> None:
        """Update an inexact-Newton forcing term from nonlinear progress."""
        current = max(
            float(res_n),
            float(np.linalg.norm(res_d)),
            float(np.linalg.norm(res_p)),
            np.finfo(float).tiny,
        )
        previous = self._nonlinear_residual
        self._previous_nonlinear_residual = previous
        self._nonlinear_residual = current
        if previous is None or not np.isfinite(previous):
            estimate = self.cg_tol_max
        else:
            ratio = current / max(previous, np.finfo(float).tiny)
            estimate = 0.9 * ratio**1.5
        self._ew_tolerance = min(
            self.cg_tol_max,
            max(self.cg_tol, float(tol) * 0.1, estimate),
        )
        self._rhs_index = 0

    def observe(self, x, z, w, s, mu) -> None:
        normalized = max(float(mu), 0.0) / (1.0 + max(float(mu), 0.0))
        mu_tolerance = self.forcing_factor * normalized
        self.current_cg_tol = min(
            self.cg_tol_max,
            max(self.cg_tol, min(mu_tolerance, self._ew_tolerance)),
        )

    def _linear_tolerance(self) -> float:
        tolerance = self.current_cg_tol
        if self._rhs_index:
            tolerance = max(self.cg_tol, self.corrector_tightening * tolerance)
        self._rhs_index += 1
        return float(tolerance)

    def _weighted_matching(self, weights: np.ndarray) -> np.ndarray:
        if (
            not self.weighted_backbone
            or self.use_full_preconditioner
            or not self.m
            or not self.factor_columns.size
            or self.m > self.matching_max_rows
            or self.A_factor.nnz > self.matching_max_nnz
        ):
            return np.empty(0, dtype=np.int64)

        graph = self.A_factor.tocsr(copy=True)
        local_column = graph.indices
        global_column = self.factor_columns[local_column]
        strength = np.abs(graph.data) * np.sqrt(
            np.maximum(weights[global_column], np.finfo(float).tiny)
        )
        finite = np.isfinite(strength) & (strength > 0.0)
        if not np.any(finite):
            return np.empty(0, dtype=np.int64)
        maximum = float(np.max(strength[finite]))
        strength[~finite] = np.finfo(float).tiny
        graph.data = 1.0 + np.log(maximum / strength)

        try:
            _, matched_local = csgraph.min_weight_full_bipartite_matching(graph)
        except (ValueError, RuntimeError):
            self.weighted_backbone_failures += 1
            matched_local = csgraph.maximum_bipartite_matching(
                graph, perm_type="column"
            )
            matched_local = matched_local[matched_local >= 0]

        matched_global = self.factor_columns[np.asarray(matched_local, dtype=np.int64)]
        positions = np.searchsorted(self.candidates, matched_global)
        valid = positions < self.candidates.size
        indices = np.flatnonzero(valid)
        valid[indices] &= self.candidates[positions[indices]] == matched_global[indices]
        return np.unique(matched_global[valid])

    def _structural_matching(self) -> np.ndarray:
        if (
            not self.structural_backbone
            or self.use_full_preconditioner
            or not self.m
            or not self.factor_columns.size
        ):
            return np.empty(0, dtype=np.int64)
        graph = self.A_factor.tocsr(copy=True)
        graph.data.fill(1.0)
        matched_local = csgraph.maximum_bipartite_matching(
            graph, perm_type="column"
        )
        matched_local = matched_local[matched_local >= 0]
        matched_global = self.factor_columns[matched_local]
        positions = np.searchsorted(self.candidates, matched_global)
        valid = positions < self.candidates.size
        indices = np.flatnonzero(valid)
        valid[indices] &= self.candidates[positions[indices]] == matched_global[indices]
        return np.unique(matched_global[valid])

    def _update_backbone(self, weights: np.ndarray, force: bool = False) -> None:
        if not self.weighted_backbone and self.matching_columns.size:
            return
        if not self.weighted_backbone:
            matching = self._structural_matching()
            if matching.size:
                self.matching_columns = matching
                self.mandatory = np.union1d(self.protected_mandatory, matching)
                self.core_target = min(
                    self.candidates.size,
                    max(self.core_target, self.mandatory.size),
                )
                self.max_core_target = min(
                    self.candidates.size,
                    max(self.max_core_target, self.mandatory.size),
                )
                self.weighted_backbone_refreshes += 1
            return
        if self.matching_columns.size and self.m > self.matching_refresh_max_rows:
            return
        if not force and self.backbone_age < self.backbone_max_age:
            return
        matching = self._weighted_matching(weights)
        if not matching.size and self.structural_backbone:
            matching = self._structural_matching()
        if matching.size or not self.matching_columns.size:
            self.matching_columns = matching
            self.mandatory = np.union1d(self.protected_mandatory, matching)
            self.core_target = min(
                self.candidates.size,
                max(self.core_target, self.mandatory.size),
            )
            self.max_core_target = min(
                self.candidates.size,
                max(self.max_core_target, self.mandatory.size),
            )
            self.weighted_backbone_refreshes += 1
        self.backbone_age = 0

    def _scores(self, weights: np.ndarray, delta: float) -> np.ndarray:
        scores = super()._scores(weights, delta)
        vector = self._enrichment_vector
        if not scores.size or vector is None or vector.size != self.m:
            return scores
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            return scores
        vector = vector / norm
        products = np.asarray(self.A_coupling.T @ vector).ravel()
        diagonal = np.asarray(self.A2_coupling.T @ np.square(vector)).ravel()
        impact = weights[self.candidates] * np.abs(np.square(products) - diagonal)
        impact /= 1.0 + self.fill_weight * self.clique_burden
        score_scale = float(np.max(scores, initial=0.0))
        impact_scale = float(np.max(impact, initial=0.0))
        if impact_scale > 0.0 and np.isfinite(impact_scale):
            scores = scores + self.residual_score_weight * max(score_scale, 1.0) * (
                impact / impact_scale
            )
        return scores

    def _refresh_core(self, weights, delta, stable, force_backbone=False) -> None:
        old = self.core.copy()
        self._update_backbone(weights, force=force_backbone or not self._initialized)
        super()._refresh_core(weights, delta, stable)
        if old.size and self.core.size:
            changed = old.size - np.intersect1d(old, self.core, assume_unique=True).size
            self.core_swaps += max(0, int(changed))

    def _lag_bound(self, weights: np.ndarray, delta: float) -> float:
        if self._reference_weights is None:
            return np.inf
        tiny = np.finfo(float).tiny
        ratio = weights / np.maximum(self._reference_weights, tiny)
        finite = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        if finite.size != ratio.size:
            return np.inf
        bound = max(float(finite.max()), 1.0 / float(finite.min()))
        if self._reference_delta > 0.0 and delta > 0.0:
            delta_ratio = delta / self._reference_delta
            bound = max(bound, delta_ratio, 1.0 / delta_ratio)
        elif delta != self._reference_delta:
            return np.inf
        return float(bound)

    def _factor_current(self, theta: np.ndarray, delta: float) -> None:
        PS.NormalEquations.factorize(self, theta, delta)
        self.factor_age = 0
        self._reference_weights = self._weights.copy()
        self._reference_delta = float(delta)
        self._factor_generation += 1
        self.preconditioner_nnz.append(self.nnz_factor)

    def factorize(self, theta: np.ndarray, delta: float) -> None:
        theta = np.asarray(theta, dtype=float)
        self._weights = 1.0 / theta
        spectral_poor = (
            self._last_condition >= self.condition_limit
            and self._last_cg >= max(10, int(0.8 * self.refresh_cg))
        )
        poor = (
            self._last_cg >= self.refresh_cg
            or spectral_poor
            or self._last_stagnated
        )
        self._poor_streak = self._poor_streak + 1 if poor else 0
        can_grow = self.core_target < self.max_core_target
        swap_due = poor and not can_grow and self._poor_streak >= self.swap_patience
        refresh_core = (
            not self._initialized
            or self.core_age >= self.core_max_age
            or (poor and can_grow)
            or swap_due
        )
        if refresh_core:
            self._refresh_core(
                self._weights,
                delta,
                stable=bool(self.core.size),
                force_backbone=swap_due,
            )
            self._poor_streak = 0

        lag_bound = self._lag_bound(self._weights, float(delta))
        self.lag_spectral_bounds.append(lag_bound)
        refactor = (
            self._factor is None
            or self.factor_age >= self.factor_max_age
            or refresh_core
            or poor
            or lag_bound > self.lag_spectral_ratio
        )
        if refactor:
            self._factor_current(theta, float(delta))
        else:
            self.theta = theta
            self.delta = float(delta)
            self.factor_age += 1
        self.core_age += 1
        self.core_sizes.append(int(self.core.size))
        self.backbone_age += 1

    def _pcg(self, rhs: np.ndarray, tolerance: float) -> KrylovResult:
        scale = max(1.0, float(np.linalg.norm(rhs)))
        solution = self._apply(rhs)
        residual = np.asarray(rhs - self._full_matvec(solution)).ravel()
        preconditioned = np.asarray(self._apply(residual)).ravel()
        rz = float(residual @ preconditioned)
        direction = preconditioned.copy()
        alphas = []
        betas = []
        replacements = 0
        residual_checks = 0
        explicit_history = [float(np.linalg.norm(residual)) / scale]
        info = self.cg_maxit
        iterations = 0

        if explicit_history[-1] <= tolerance:
            info = 0
        elif not np.isfinite(rz) or rz <= 0.0:
            info = -1

        while info > 0 and iterations < self.cg_maxit:
            product = np.asarray(self._full_matvec(direction)).ravel()
            curvature = float(direction @ product)
            curvature_floor = np.finfo(float).tiny * max(
                1.0, float(np.linalg.norm(direction)) * float(np.linalg.norm(product))
            )
            if not np.isfinite(curvature) or curvature <= curvature_floor:
                info = -2
                break
            alpha = rz / curvature
            if not np.isfinite(alpha) or alpha <= 0.0:
                info = -3
                break
            solution += alpha * direction
            residual -= alpha * product
            iterations += 1
            alphas.append(alpha)

            recurrence_relative = float(np.linalg.norm(residual)) / scale
            replace = (
                self.residual_replacement > 0
                and iterations % self.residual_replacement == 0
            )
            verify = replace or recurrence_relative <= tolerance
            if verify:
                true_residual = np.asarray(
                    rhs - self._full_matvec(solution)
                ).ravel()
                residual_checks += 1
                true_norm = float(np.linalg.norm(true_residual))
                relative = true_norm / scale
                explicit_history.append(relative)
                if not np.isfinite(relative):
                    info = -4
                    break
                if relative <= tolerance:
                    residual = true_residual
                    info = 0
                    break
                residual_gap = float(np.linalg.norm(true_residual - residual))
                gap_limit = self.residual_gap_tolerance * max(
                    true_norm, tolerance * scale
                )
                if residual_gap > gap_limit:
                    residual = true_residual
                    replacements += 1
                    preconditioned = np.asarray(self._apply(residual)).ravel()
                    rz = float(residual @ preconditioned)
                    if not np.isfinite(rz) or rz <= 0.0:
                        info = -5
                        break
                    direction = preconditioned.copy()
                    alphas.clear()
                    betas.clear()
                    continue

            preconditioned = np.asarray(self._apply(residual)).ravel()
            next_rz = float(residual @ preconditioned)
            if not np.isfinite(next_rz) or next_rz <= 0.0:
                info = -6
                break
            beta = next_rz / rz
            if not np.isfinite(beta) or beta < 0.0:
                info = -7
                break
            betas.append(beta)
            direction = preconditioned + beta * direction
            rz = next_rz

        residual = np.asarray(rhs - self._full_matvec(solution)).ravel()
        relative = float(np.linalg.norm(residual)) / scale
        if info > 0 and relative <= tolerance:
            info = 0
        smallest, largest, condition = _ritz_estimate(alphas, betas)
        spectral_error = max(abs(smallest - 1.0), abs(largest - 1.0))
        window = self.stagnation_window
        stagnated = (
            len(explicit_history) >= window
            and explicit_history[-1]
            > self.stagnation_reduction * explicit_history[-window]
        )
        return KrylovResult(
            solution=solution,
            info=info,
            iterations=iterations,
            relative_residual=relative,
            condition_estimate=float(condition),
            spectral_error_estimate=float(spectral_error),
            residual=residual,
            replacements=replacements,
            residual_checks=residual_checks,
            stagnated=bool(stagnated),
        )

    def _minres(self, r1, r2, tolerance):
        size = self.n + self.m
        iterations = 0

        def matvec(vector):
            dx = vector[: self.n]
            q = vector[self.n :]
            return np.concatenate(
                [self.theta * dx + self.At @ q, self.A @ dx - self.delta * q]
            )

        def precondition(vector):
            return np.concatenate(
                [vector[: self.n] / self.theta, self._apply(vector[self.n :])]
            )

        def count(_):
            nonlocal iterations
            iterations += 1

        operator = spla.LinearOperator((size, size), matvec=matvec, dtype=float)
        preconditioner = spla.LinearOperator(
            (size, size), matvec=precondition, dtype=float
        )
        rhs = np.concatenate([r1, r2])
        scale = max(1.0, float(np.linalg.norm(rhs)))
        target = max(2.0 * float(tolerance), self.fail_tol)
        solution = np.zeros_like(rhs)
        info = self.minres_maxit
        relative = float(np.linalg.norm(rhs)) / scale

        # MINRES stops on a preconditioned residual.  Explicit refinement is
        # required before accepting the direction in the original KKT norm.
        for _ in range(3):
            correction_rhs = rhs - matvec(solution)
            relative = float(np.linalg.norm(correction_rhs)) / scale
            if np.isfinite(relative) and relative <= target:
                info = 0
                break
            correction, info = spla.minres(
                operator,
                correction_rhs,
                M=preconditioner,
                rtol=max(np.finfo(float).eps, min(float(tolerance), 1e-8)),
                maxiter=self.minres_maxit,
                callback=count,
                check=False,
            )
            solution += correction
            if info < 0:
                break

        relative = float(np.linalg.norm(rhs - matvec(solution))) / scale
        accepted = np.isfinite(relative) and relative <= target
        return solution[: self.n], -solution[self.n :], accepted, iterations, relative

    def _record_krylov(self, result: KrylovResult) -> None:
        self._last_cg = int(result.iterations)
        self._last_condition = float(result.condition_estimate)
        self._last_spectral_error = float(result.spectral_error_estimate)
        self._last_stagnated = bool(result.stagnated)
        self._enrichment_vector = result.residual.copy()
        self.condition_estimates.append(self._last_condition)
        self.spectral_error_estimates.append(self._last_spectral_error)
        self.residual_replacements += int(result.replacements)
        self.residual_checks += int(result.residual_checks)
        self.stagnation_events += int(result.stagnated)

    def solve(self, r1: np.ndarray, r2: np.ndarray):
        started = time.perf_counter()
        factor_before = self.t_fact
        weights_r1 = r1 / self.theta
        rhs = np.asarray(r2 - self.A @ weights_r1).ravel()
        tolerance = self._linear_tolerance()
        result = self._pcg(rhs, tolerance)
        total_iterations = result.iterations

        failed = (
            result.info != 0
            or not np.isfinite(result.relative_residual)
            or result.relative_residual > tolerance
        )
        can_grow = self.core_target < self.max_core_target
        retry_warranted = can_grow or result.info < 0 or result.stagnated
        if (
            failed
            and retry_warranted
            and self.candidates.size
            and not self.use_full_preconditioner
        ):
            self.pcg_retries += 1
            self._enrichment_vector = result.residual.copy()
            if self.core_target < self.max_core_target:
                grown = max(
                    self.core_target + 1,
                    int(np.ceil(self.retry_growth * self.core_target)),
                )
                self.core_target = min(self.max_core_target, grown)
            self._refresh_core(
                self._weights,
                self.delta,
                stable=True,
                force_backbone=result.info < 0 or result.stagnated,
            )
            self._factor_current(self.theta, self.delta)
            self.core_sizes.append(int(self.core.size))
            retry = self._pcg(rhs, tolerance)
            total_iterations += retry.iterations
            result = retry
            failed = (
                result.info != 0
                or not np.isfinite(result.relative_residual)
                or result.relative_residual > tolerance
            )

        # A finite explicitly measured residual below the base factorization
        # guard is usable even if CG encountered roundoff before its tighter
        # inexact-Newton target.  The event still drives a refresh next time.
        if failed and result.relative_residual <= max(
            2.0 * tolerance, self.fail_tol
        ):
            failed = False

        self.cg_tolerances.append(tolerance)
        self.cg_iterations.append(total_iterations)
        self._record_krylov(result)
        if result.info < 0:
            self.krylov_breakdowns += 1
        elif result.info > 0:
            self.krylov_iteration_limits += 1

        if not failed:
            dy = result.solution
            dx = weights_r1 + (self.At @ dy) / self.theta
        elif (
            self.minres_fallback
            and self.n + self.m <= self.minres_max_dimension
            and (result.info < 0 or result.stagnated)
        ):
            self.cg_failures += 1
            self.minres_fallbacks += 1
            dx, dy, accepted, iterations, relative = self._minres(
                r1, r2, tolerance
            )
            self.minres_iterations += iterations
            self.full_fallbacks += 1
            if not accepted:
                self.t_solve += max(
                    0.0, time.perf_counter() - started - (self.t_fact - factor_before)
                )
                raise PS.KKTSolveError(
                    "spectral PCG and augmented MINRES failed: "
                    f"pcg info={result.info}, pcg residual="
                    f"{result.relative_residual:.2e}, minres residual={relative:.2e}"
                )
        else:
            self.cg_failures += 1
            self.t_solve += max(
                0.0, time.perf_counter() - started - (self.t_fact - factor_before)
            )
            raise PS.KKTSolveError(
                "spectral stable-column PCG failed: "
                f"info={result.info}, residual={result.relative_residual:.2e}"
            )

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
    order: str = "best",
    refine: int = 1,
    low_rank_max_columns: int = 8,
    low_rank_min_nnz: int = 1024,
    low_rank_clique_ratio: float = 2.0,
    **kwargs,
) -> Solution:
    printlevel = int(kwargs.get("printlevel", 1))
    solver_option_names = {
        "weighted_backbone",
        "structural_backbone",
        "matching_max_rows",
        "matching_max_nnz",
        "matching_refresh_max_rows",
        "backbone_max_age",
        "residual_score_weight",
        "swap_patience",
        "condition_limit",
        "lag_spectral_ratio",
        "residual_replacement",
        "residual_gap_tolerance",
        "stagnation_window",
        "stagnation_reduction",
        "corrector_tightening",
        "minres_fallback",
        "minres_maxit",
        "minres_max_dimension",
        "full_preconditioner_row_threshold",
        "full_preconditioner_clique_limit",
        "full_preconditioner_nnz_limit",
    }
    solver_options = {
        key: kwargs.pop(key) for key in list(kwargs) if key in solver_option_names
    }
    solver = SpectralStableColumnPreconditioner(
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
        order=order,
        refine=refine,
        low_rank_max_columns=low_rank_max_columns,
        low_rank_min_nnz=low_rank_min_nnz,
        low_rank_clique_ratio=low_rank_clique_ratio,
        **solver_options,
    )
    if printlevel:
        print(
            f"  preconditioner : spectral stable core {solver.core_target}/"
            f"{solver.candidates.size}, max {solver.max_core_target}"
        )
    return PS.psipm_solve(
        lp,
        kkt="spectral-stable-pcg",
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
        print(f"PS-IPM-SSP  |  {path}")
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
        description="PS-IPM with spectrally monitored stable-column PCG"
    )
    parser.add_argument("model")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=2000.0)
    parser.add_argument("--maxit", type=int, default=200)
    parser.add_argument("--inner-maxit", type=int, default=100)
    parser.add_argument("--pc", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--rf", type=float, default=1.0)
    parser.add_argument(
        "--start-projection", choices=("adaptive", "legacy"), default="legacy"
    )
    parser.add_argument("--start-bound-fraction", type=float, default=1e-3)
    parser.add_argument("--start-linear-tolerance", type=float)
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
    parser.add_argument("--condition-limit", type=float, default=1e12)
    parser.add_argument("--lag-spectral-ratio", type=float, default=1e8)
    parser.add_argument("--residual-replacement", type=int, default=100)
    parser.add_argument("--residual-gap-tolerance", type=float, default=0.1)
    matching = parser.add_mutually_exclusive_group()
    matching.add_argument(
        "--weighted-backbone", dest="weighted_backbone", action="store_true"
    )
    matching.add_argument(
        "--no-weighted-backbone", dest="weighted_backbone", action="store_false"
    )
    parser.set_defaults(weighted_backbone=False)
    parser.add_argument("--no-matching-backbone", action="store_true")
    parser.add_argument("--no-minres-fallback", action="store_true")
    parser.add_argument("--minres-maxit", type=int, default=400)
    parser.add_argument("--minres-max-dimension", type=int, default=200_000)
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
        rho=args.rho,
        delta=args.delta,
        rf=args.rf,
        start_projection=args.start_projection,
        start_bound_fraction=args.start_bound_fraction,
        start_linear_tolerance=args.start_linear_tolerance,
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
        condition_limit=args.condition_limit,
        lag_spectral_ratio=args.lag_spectral_ratio,
        residual_replacement=args.residual_replacement,
        residual_gap_tolerance=args.residual_gap_tolerance,
        weighted_backbone=args.weighted_backbone,
        structural_backbone=not args.no_matching_backbone,
        minres_fallback=not args.no_minres_fallback,
        minres_maxit=args.minres_maxit,
        minres_max_dimension=args.minres_max_dimension,
        order=args.order,
        printlevel=args.printlevel,
        compl="gap",
    )
    print(
        f"objective {check['obj']:.10e}  pinf {check['pinf']:.2e}  "
        f"dinf {check['dinf']:.2e}  time {solution.time:.2f}s"
    )
