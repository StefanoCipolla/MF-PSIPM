#!/usr/bin/env python3
"""Fair folder benchmark for generic PSIPM-V3.1, HiGHS IPX, and HiPO.

This driver reuses the mature discovery, process isolation, timing, MPS/RAIL
input, resume, and CSV machinery from ``batch_test_stable_preconditioner``.
Each method/model/repeat runs in a fresh process.  Total time includes input,
optional presolve, conversion/scaling, setup, solve, and original-model checks.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import batch_test_stable_preconditioner as base
import psipm_v3_1_hsd as v3


METHODS = ("psipm-v3", "highs-ipx", "highs-hipo")
base.METHODS = METHODS
# The inherited command builder must launch this V3 entry point.
base.__file__ = __file__
base.__doc__ = __doc__
base.DETAIL_FIELDS = base.DETAIL_FIELDS + (
    "relative_gap",
    "complementarity",
    "hsd_residual",
    "tau",
    "kappa",
    "theta",
    "pricing_rounds",
    "columns_added",
    "columns_dropped",
    "proximal_subproblems",
    "proximal_checks",
    "proximal_last_ratio",
    "pcg_solves",
    "preconditioner_growth_events",
    "preconditioner_max_factor_nonzeros",
    "solver_path",
    "structure",
    "preconditioner",
)


def _use_hybrid_active_pool(raw, args):
    """Select the proven active-pool path only for wide covering models."""
    if args.hybrid_active_pool == "off":
        return False
    if args.hybrid_active_pool == "on":
        return True
    m, n = raw.A.shape
    if m == 0 or m > args.hybrid_max_rows:
        return False
    if n / m < args.hybrid_min_column_row_ratio:
        return False
    tolerance = 64.0 * np.finfo(float).eps
    nonnegative_matrix = (
        raw.A.nnz == 0 or float(raw.A.data.min()) >= -tolerance
    )
    nonnegative_cost = raw.c.size == 0 or float(raw.c.min()) >= -tolerance
    nonnegative_columns = bool(np.all(raw.cl >= -tolerance))
    lower_rows = bool(np.all(np.isfinite(raw.rl)))
    upper_rows = bool(np.all(raw.ru >= 0.5 * base.PS.INF))
    return bool(
        nonnegative_matrix
        and nonnegative_cost
        and nonnegative_columns
        and lower_rows
        and upper_rows
    )


def _run_hybrid_active_pool(standard, args, remaining):
    return base.PS.warm_sparse_solve(
        standard,
        tol=args.tol,
        pc=args.pc,
        kkt="normal",
        compl="gap",
        time_limit=remaining,
        maxit=args.maxit,
        printlevel=args.printlevel,
        sparse_steps=0,
        sparse_inner_maxit=1,
        sparse_pricing_batch=max(args.hybrid_pricing_batch, standard.A.shape[0]),
        sparse_fill_weight=args.fill_weight,
        sparse_max_rounds=args.hybrid_max_rounds,
        sparse_polish_steps=args.hybrid_polish_steps,
        sparse_rank_protection=True,
        sparse_cover_crash=True,
        sparse_use_crash_point=False,
        low_rank_max_columns=args.low_rank_max_columns,
        low_rank_min_nnz=args.low_rank_min_nnz,
        low_rank_clique_ratio=args.low_rank_clique_ratio,
    )


def run_v3(args):
    started = time.perf_counter()
    deadline = started + args.time_limit
    record = base._base_record(args)

    read_started = time.perf_counter()
    raw_original = base.read_instance(args.model)
    record["read_seconds"] = time.perf_counter() - read_started
    record.update(
        rows=int(raw_original.A.shape[0]),
        columns=int(raw_original.A.shape[1]),
        nonzeros=int(raw_original.A.nnz),
        solver_version="psipm-v3.1-hsd-adaptive-core-1",
    )

    setup_started = time.perf_counter()
    raw = raw_original
    presolver = None
    presolve_status = "off"
    if args.presolve == "on":
        raw, presolver, presolve_status = base.PS.presolve_raw(
            raw_original, time_limit=base._remaining(deadline)
        )
    standard = base.PS.to_standard_form(raw, verbose=False)
    base.PS.scale_problem(standard, mode=5, verbose=False)
    internal_tolerance = v3._scaled_tolerance(standard, args.tol)
    use_hybrid = _use_hybrid_active_pool(raw, args)
    box = None
    canonical = None
    if not use_hybrid:
        if bool(np.all(standard.pos)):
            canonical = v3.CanonicalLP(
                standard.c,
                standard.A,
                standard.b,
                name=Path(args.model).name,
                u=standard.u,
            )
        else:
            lower = np.where(standard.pos, 0.0, -np.inf)
            upper = np.where(standard.pos, standard.u, np.inf)
            box = v3.BoxLP(
                standard.c,
                standard.A,
                standard.b,
                lower=lower,
                upper=upper,
                name=Path(args.model).name,
                objective_offset=standard.obj_const,
            )
    record["setup_seconds"] = time.perf_counter() - setup_started
    record.update(
        presolve_status=str(presolve_status),
        internal_tolerance=float(internal_tolerance),
        canonical_rows=int(standard.A.shape[0]),
        canonical_columns=int(standard.A.shape[1]),
        canonical_nonzeros=int(standard.A.nnz),
    )
    remaining = base._remaining(deadline)
    if remaining <= 0.0:
        record.update(
            status="setup-time-limit",
            solve_seconds=0.0,
            postsolve_seconds=0.0,
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=base._peak_memory_mb(),
        )
        return record

    if use_hybrid:
        solve_started = time.perf_counter()
        solution = _run_hybrid_active_pool(standard, args, remaining)
        record["solve_seconds"] = time.perf_counter() - solve_started
        post_started = time.perf_counter()
        check_pre = base.PS.recover(standard, raw, solution)
        check = (
            base.PS._postsolve_info(presolver, raw, raw_original, check_pre)
            if presolver is not None and raw is not raw_original
            else check_pre
        )
        status = solution.status
        if status == "optimal" and max(check["pinf"], check["dinf"]) > args.tol:
            status = "postsolve-inaccurate"
        record["postsolve_seconds"] = time.perf_counter() - post_started
        solve_seconds = float(record["solve_seconds"])
        timing_slack = max(0.05, 0.01 * solve_seconds)
        record.update(
            solver_version="psipm-v3.1-hybrid-active-pool-1",
            solver_path="hybrid-active-pool",
            status=status,
            objective=float(check["obj"]),
            primal_infeasibility=float(check["pinf"]),
            dual_infeasibility=float(check["dinf"]),
            complementarity=float(solution.mu),
            iterations=int(solution.outer_iter),
            outer_iterations=int(solution.outer_iter),
            inner_iterations=int(solution.inner_iter),
            factor_seconds=float(solution.t_fact),
            linear_solve_seconds=float(solution.t_solve),
            active_columns=int(solution.cg_working_columns),
            pricing_rounds=int(solution.cg_iterations),
            columns_added=int(solution.cg_columns_added),
            columns_dropped=int(solution.cg_columns_eliminated),
            proximal_subproblems=int(solution.inexact_subproblems),
            low_rank_columns=int(solution.low_rank_columns),
            pcg_iterations=int(solution.low_rank_pcg_iterations),
            pcg_failures=int(solution.low_rank_pcg_failures),
            structure="wide-covering",
            preconditioner="direct-active-normal",
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=base._peak_memory_mb(),
            timing_consistent=bool(
                solution.t_fact <= solve_seconds + timing_slack
                and solution.t_solve <= solve_seconds + timing_slack
            ),
            error=solution.termination_detail,
        )
        return record

    solve_started = time.perf_counter()
    configured = v3.V3Options(
            tolerance=internal_tolerance,
            time_limit=remaining,
            max_iterations=args.maxit,
            rmp_steps=max(1, args.inner_maxit),
            generic_preconditioner="stable",
            generic_core_factor=args.core_factor,
            generic_max_core_factor=args.max_core_factor,
            generic_retention=args.retention,
            generic_factor_max_age=args.factor_max_age,
            generic_core_max_age=args.core_max_age,
            generic_refresh_cg=args.refresh_cg,
            generic_fill_weight=args.fill_weight,
            generic_retry_growth=args.retry_growth,
            generic_growth_amortization=args.growth_amortization,
            generic_cholmod_order=args.cholmod_order,
            generic_low_rank_max_columns=args.low_rank_max_columns,
            generic_low_rank_min_nnz=args.low_rank_min_nnz,
            generic_low_rank_clique_ratio=args.low_rank_clique_ratio,
            pcg_tolerance_early=args.cg_tol_max,
            pcg_tolerance_late=args.cg_tol,
            pcg_max_iterations=args.cg_maxit,
            print_level=args.printlevel,
    )
    result = (
        v3.solve(canonical, configured)
        if canonical is not None
        else v3.solve_box_lp(box, configured)
    )
    record["solve_seconds"] = time.perf_counter() - solve_started

    post_started = time.perf_counter()
    solution = base.PS.Solution(
        x=result.x,
        y=result.y,
        z=result.z,
        w=result.w,
        status=result.status,
    )
    check_pre = base.PS.recover(standard, raw, solution)
    check = (
        base.PS._postsolve_info(presolver, raw, raw_original, check_pre)
        if presolver is not None and raw is not raw_original
        else check_pre
    )
    status = result.status
    if status == "optimal" and max(
        float(check["pinf"]),
        float(check["dinf"]),
        float(result.relative_gap),
    ) > args.tol:
        status = "postsolve-inaccurate"
    record["postsolve_seconds"] = time.perf_counter() - post_started

    solve_seconds = float(record["solve_seconds"])
    timing_slack = max(0.05, 0.01 * solve_seconds)
    record.update(
        status=status,
        objective=float(check["obj"]),
        primal_infeasibility=float(check["pinf"]),
        dual_infeasibility=float(check["dinf"]),
        relative_gap=float(result.relative_gap),
        complementarity=float(result.complementarity),
        hsd_residual=float(result.hsd_residual),
        tau=float(result.tau),
        kappa=float(result.kappa),
        theta=float(result.theta),
        iterations=int(result.iterations),
        outer_iterations=int(result.proximal.accepted),
        inner_iterations=int(result.iterations),
        factor_seconds=float(result.linear.preconditioner_seconds),
        linear_solve_seconds=float(result.linear.seconds),
        active_columns=int(result.active_columns),
        pcg_iterations=int(result.linear.iterations),
        pcg_max_iterations=int(result.linear.max_iterations),
        pcg_failures=int(result.linear.failures),
        pcg_solves=int(result.linear.solves),
        pricing_rounds=int(result.pricing_rounds),
        columns_added=int(result.columns_added),
        columns_dropped=int(result.columns_dropped),
        proximal_subproblems=int(result.proximal.accepted),
        proximal_checks=int(result.proximal.checks),
        proximal_last_ratio=float(result.proximal.last_ratio),
        preconditioner_core_min=int(result.preconditioner_stats.core_min),
        preconditioner_core_mean=float(result.preconditioner_stats.core_mean),
        preconditioner_core_max=int(result.preconditioner_stats.core_max),
        preconditioner_refreshes=int(result.preconditioner_stats.refreshes),
        preconditioner_factorizations=int(
            result.preconditioner_stats.factorizations
        ),
        preconditioner_retained_mean=float(
            result.preconditioner_stats.retained_mean
        ),
        preconditioner_symbolic_analyses=int(
            result.preconditioner_stats.symbolic_analyses
        ),
        preconditioner_full=bool(
            result.preconditioner_stats.full_preconditioner
        ),
        preconditioner_growth_events=int(
            result.preconditioner_stats.growth_events
        ),
        preconditioner_max_factor_nonzeros=int(
            result.preconditioner_stats.max_factor_nonzeros
        ),
        low_rank_columns=int(result.preconditioner_stats.low_rank_columns),
        solver_path="hsd-adaptive-core",
        structure=result.structure,
        preconditioner=result.preconditioner,
        total_seconds=time.perf_counter() - started,
        peak_memory_mb=base._peak_memory_mb(),
        timing_consistent=bool(
            result.linear.preconditioner_seconds <= solve_seconds + timing_slack
            and result.linear.seconds <= solve_seconds + timing_slack
        ),
        error=result.message,
    )
    return record


def worker(args):
    started = time.perf_counter()
    try:
        if args.method == "psipm-v3":
            record = run_v3(args)
        elif args.method == "highs-ipx":
            record = base.run_highs(args, "ipx")
        elif args.method == "highs-hipo":
            record = base.run_highs(args, "hipo")
        else:
            raise ValueError(f"unknown method {args.method}")
    except Exception as exc:
        record = base._base_record(args)
        record.update(
            status="exception",
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=base._peak_memory_mb(),
            error=f"{type(exc).__name__}: {exc}",
        )
        if args.printlevel:
            record["error"] += "\n" + traceback.format_exc()
    print(json.dumps(record, sort_keys=True, allow_nan=True))


def _objective_difference(candidate, reference):
    if (
        candidate.get("status") != "optimal"
        or reference.get("status") != "optimal"
    ):
        return ""
    try:
        candidate_objective = float(candidate.get("objective"))
        reference_objective = float(reference.get("objective"))
    except (TypeError, ValueError):
        return ""
    if math.isfinite(candidate_objective) and math.isfinite(reference_objective):
        return abs(candidate_objective - reference_objective) / max(
            1.0, abs(reference_objective)
        )
    return ""


def write_summary(detail_path, summary_path):
    rows = base._read_detail_rows(detail_path)
    grouped = {}
    for row in rows:
        key = (row.get("path", ""), int(row.get("repeat", -1)))
        grouped.setdefault(key, {})[row.get("method", "")] = row

    common = [
        "problem", "path", "repeat", "rows", "columns", "nonzeros",
        "presolve", "threads", "tolerance", "time_limit",
    ]
    method_names = {
        "psipm-v3": (
            "status", "total_seconds", "solve_seconds", "objective",
            "primal_infeasibility", "dual_infeasibility", "relative_gap",
            "outer_iterations", "inner_iterations", "pcg_solves",
            "pcg_iterations", "pcg_max_iterations", "pcg_failures",
            "factor_seconds", "linear_solve_seconds", "active_columns",
            "pricing_rounds", "columns_added", "columns_dropped",
            "proximal_subproblems", "proximal_checks",
            "proximal_last_ratio", "preconditioner_core_min",
            "preconditioner_core_mean", "preconditioner_core_max",
            "preconditioner_refreshes", "preconditioner_factorizations",
            "preconditioner_retained_mean",
            "preconditioner_symbolic_analyses",
            "preconditioner_growth_events",
            "preconditioner_max_factor_nonzeros", "preconditioner_full",
            "solver_path", "structure", "preconditioner",
            "peak_memory_mb", "timing_consistent", "error",
        ),
        "highs-ipx": (
            "status", "total_seconds", "solve_seconds", "objective",
            "primal_infeasibility", "dual_infeasibility", "iterations",
            "peak_memory_mb",
        ),
        "highs-hipo": (
            "status", "total_seconds", "solve_seconds", "objective",
            "primal_infeasibility", "dual_infeasibility", "iterations",
            "peak_memory_mb",
        ),
    }
    prefixes = {
        "psipm-v3": "psipm",
        "highs-ipx": "ipx",
        "highs-hipo": "hipo",
    }
    fields = list(common)
    for method, names in method_names.items():
        fields.extend(f"{prefixes[method]}_{name}" for name in names)
    fields.extend(
        (
            "v3_speedup_vs_ipx",
            "v3_speedup_vs_hipo",
            "v3_ipx_objective_reldiff",
            "v3_hipo_objective_reldiff",
        )
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            methods = grouped[key]
            representative = max(methods.values(), key=lambda row: bool(row.get("rows")))
            output = {name: representative.get(name, "") for name in common}
            for method, names in method_names.items():
                source = methods.get(method, {})
                prefix = prefixes[method]
                for name in names:
                    output[f"{prefix}_{name}"] = source.get(name, "")
            v3_record = methods.get("psipm-v3", {})
            output["v3_speedup_vs_ipx"] = base._speedup(
                methods.get("highs-ipx", {}), v3_record
            )
            output["v3_speedup_vs_hipo"] = base._speedup(
                methods.get("highs-hipo", {}), v3_record
            )
            output["v3_ipx_objective_reldiff"] = _objective_difference(
                v3_record, methods.get("highs-ipx", {})
            )
            output["v3_hipo_objective_reldiff"] = _objective_difference(
                v3_record, methods.get("highs-hipo", {})
            )
            writer.writerow(output)


def main():
    base.worker = worker
    base.write_summary = write_summary
    defaults = {
        "--maxit": "500",
        "--inner-maxit": "3",
        "--cg-maxit": "1000",
    }
    for option, value in defaults.items():
        if option not in sys.argv:
            sys.argv.extend((option, value))
    args = base.parse_args()
    return worker(args) if args.worker else base.orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
