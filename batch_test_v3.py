#!/usr/bin/env python3
"""Fair folder benchmark for generic PSIPM-V3, HiGHS IPX, and HiPO.

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
import psipm_v3_hsd as v3


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
    "structure",
    "preconditioner",
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
        solver_version="psipm-v3-hsd-generic-2",
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

    solve_started = time.perf_counter()
    result = v3.solve_box_lp(
        box,
        v3.V3Options(
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
            pcg_tolerance_early=args.cg_tol_max,
            pcg_tolerance_late=args.cg_tol,
            pcg_max_iterations=args.cg_maxit,
            print_level=args.printlevel,
        ),
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
        outer_iterations=int(result.iterations),
        inner_iterations=int(result.linear.iterations),
        factor_seconds=float(result.linear.preconditioner_seconds),
        linear_solve_seconds=float(result.linear.seconds),
        active_columns=int(result.active_columns),
        pcg_iterations=int(result.linear.iterations),
        pcg_failures=int(result.linear.failures),
        pricing_rounds=int(result.pricing_rounds),
        columns_added=int(result.columns_added),
        columns_dropped=int(result.columns_dropped),
        proximal_subproblems=int(result.proximal.accepted),
        proximal_checks=int(result.proximal.checks),
        proximal_last_ratio=float(result.proximal.last_ratio),
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
            "outer_iterations", "inner_iterations", "pcg_iterations",
            "pcg_failures", "factor_seconds", "linear_solve_seconds",
            "structure", "preconditioner", "peak_memory_mb",
            "timing_consistent",
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
    fields.extend(("v3_speedup_vs_ipx", "v3_speedup_vs_hipo"))

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
