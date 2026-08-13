#!/usr/bin/env python3
"""Fair folder benchmark for PS-IPM-SAP, HiGHS IPX, and HiPO.

Examples
--------
Run all three methods on every supported model in a folder::

    python batch_test_stable_preconditioner.py MODELS --csv results.csv

Resume an interrupted benchmark::

    python batch_test_stable_preconditioner.py MODELS --csv results.csv --resume

The detailed CSV has one durable row per method/model/repeat.  A second, wide
``*_summary.csv`` file is generated for direct method comparison.  Each run is
performed in a fresh process.  ``total_seconds`` includes reading, common model
conversion, optional presolve, solver setup, solve, and solution validation;
``solve_seconds`` is also reported separately.

MIP integrality is deliberately dropped: all methods solve the same continuous
LP relaxation.  PS-IPM-SAP uses the same two-sided max-norm equilibration policy
as the HiGHS interior-point paths.  Native OR-Library ``rail*.gz`` files are
supported in addition to MPS, LP, and QPS files accepted by HiGHS.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    import resource
except ImportError:  # Windows
    resource = None


HERE = Path(__file__).resolve().parent
METHODS = ("psipm-sap", "highs-ipx", "highs-hipo")
SUPPORTED_SUFFIXES = (
    ".mps",
    ".mps.gz",
    ".mps.bz2",
    ".lp",
    ".lp.gz",
    ".qps",
    ".qps.gz",
)

DETAIL_FIELDS = (
    "problem",
    "path",
    "repeat",
    "order_position",
    "method",
    "status",
    "rows",
    "columns",
    "nonzeros",
    "presolve",
    "presolve_status",
    "threads",
    "tolerance",
    "internal_tolerance",
    "time_limit",
    "read_seconds",
    "setup_seconds",
    "solve_seconds",
    "postsolve_seconds",
    "total_seconds",
    "process_seconds",
    "iterations",
    "outer_iterations",
    "inner_iterations",
    "factor_seconds",
    "linear_solve_seconds",
    "objective",
    "primal_infeasibility",
    "dual_infeasibility",
    "canonical_rows",
    "canonical_columns",
    "canonical_nonzeros",
    "active_columns",
    "preconditioner_core_min",
    "preconditioner_core_mean",
    "preconditioner_core_max",
    "preconditioner_refreshes",
    "preconditioner_factorizations",
    "preconditioner_retained_mean",
    "preconditioner_symbolic_analyses",
    "preconditioner_retries",
    "preconditioner_full",
    "pcg_iterations",
    "pcg_max_iterations",
    "pcg_failures",
    "low_rank_columns",
    "low_rank_candidate_edges",
    "peak_memory_mb",
    "timing_consistent",
    "solver_version",
    "model_status",
    "run_status",
    "error",
)


def _load_solver():
    path = HERE / "psipm_stable_preconditioner.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required next to {Path(__file__).name}"
        )
    spec = importlib.util.spec_from_file_location(
        "psipm_stable_preconditioner", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAP = _load_solver()
PS = SAP.PS


def _peak_memory_mb():
    if resource is not None:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak / (1024.0**2 if sys.platform == "darwin" else 1024.0)
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / 1024.0**2
    except Exception:
        return float("nan")


def choose_folder(initial_directory=None):
    """Open a native directory chooser and return the selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        selected = filedialog.askdirectory(
            parent=root,
            title="Select the folder containing LP models",
            initialdir=str(initial_directory or Path.home()),
            mustexist=True,
        )
        root.destroy()
    except Exception as exc:
        raise RuntimeError(
            "the graphical folder picker is unavailable; pass the folder "
            "path as the first command-line argument"
        ) from exc
    return str(Path(selected).resolve()) if selected else ""


def _normal_status(value):
    value = str(value)
    mapping = {
        "kOptimal": "optimal",
        "kInfeasible": "infeasible",
        "kUnbounded": "unbounded",
        "kUnboundedOrInfeasible": "unbounded-or-infeasible",
        "kTimeLimit": "time-limit",
        "kIterationLimit": "iteration-limit",
        "kUnknown": "unknown",
    }
    return mapping.get(value.split(".")[-1], value)


def _logical_name(path):
    name = Path(path).name
    lower = name.lower()
    for suffix in (".mps.gz", ".mps.bz2", ".lp.gz", ".qps.gz"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    for suffix in (".mps", ".lp", ".qps", ".gz"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _is_native_rail(path):
    name = Path(path).name.lower()
    return name.startswith("rail") and name.endswith(".gz") and not any(
        name.endswith(suffix) for suffix in (".mps.gz", ".lp.gz", ".qps.gz")
    )


def _is_supported(path):
    name = Path(path).name.lower()
    return _is_native_rail(path) or any(
        name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES
    )


def _model_preference(path):
    """Prefer uncompressed MPS when duplicate encodings are present."""
    name = Path(path).name.lower()
    compressed = int(name.endswith((".gz", ".bz2")))
    kind = 0 if ".mps" in name else (1 if ".lp" in name else 2)
    return compressed, kind, len(name), name


def find_models(folder, pattern="*", recursive=False, allow_duplicates=False):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    files = [
        path.absolute()
        for path in candidates
        if path.is_file()
        and _is_supported(path)
        and fnmatch.fnmatch(path.name, pattern)
    ]
    files.sort(key=lambda path: str(path).lower())
    if allow_duplicates:
        return files, []

    selected = {}
    skipped = []
    for path in files:
        relative = path.relative_to(folder)
        key = (str(relative.parent).lower(), _logical_name(path).lower())
        previous = selected.get(key)
        if previous is None or _model_preference(path) < _model_preference(previous):
            if previous is not None:
                skipped.append(previous)
            selected[key] = path
        else:
            skipped.append(path)
    return sorted(selected.values(), key=lambda path: str(path).lower()), skipped


def read_rail(path):
    """Read the OR-Library column-oriented railway set-covering format."""
    with gzip.open(path, "rb") as handle:
        tokens = np.fromstring(handle.read(), dtype=np.int64, sep=" ")
    if tokens.size < 2:
        raise ValueError(f"empty or invalid RAIL file: {path}")
    m, n = map(int, tokens[:2])
    expected_nnz = int(tokens.size - 2 - 2 * n)
    if m <= 0 or n <= 0 or expected_nnz < 0:
        raise ValueError(f"invalid RAIL dimensions in {path}")

    costs = np.empty(n, dtype=np.float64)
    indptr = np.empty(n + 1, dtype=np.int64)
    indices = np.empty(expected_nnz, dtype=np.int32)
    pos = 2
    nzpos = 0
    indptr[0] = 0
    for column in range(n):
        if pos + 2 > tokens.size:
            raise ValueError(f"truncated RAIL column {column} in {path}")
        costs[column] = tokens[pos]
        count = int(tokens[pos + 1])
        pos += 2
        end = pos + count
        if count < 0 or end > tokens.size:
            raise ValueError(f"invalid RAIL column {column} in {path}")
        indices[nzpos : nzpos + count] = tokens[pos:end] - 1
        nzpos += count
        indptr[column + 1] = nzpos
        pos = end
    if pos != tokens.size or nzpos != expected_nnz:
        raise ValueError(f"unexpected trailing RAIL data in {path}")
    if indices.size and (indices.min() < 0 or indices.max() >= m):
        raise ValueError(f"RAIL row index outside 1..{m} in {path}")

    matrix = sp.csc_matrix(
        (np.ones(expected_nnz), indices, indptr), shape=(m, n)
    )
    return PS.RawLP(
        c=costs,
        A=matrix,
        rl=np.ones(m),
        ru=np.full(m, PS.INF),
        cl=np.zeros(n),
        cu=np.ones(n),
        name=_logical_name(path),
    )


def read_instance(path):
    path = Path(path)
    if _is_native_rail(path):
        return read_rail(path)
    return PS.read_mps(str(path))


def _base_record(args):
    return {
        "problem": _logical_name(args.model),
        "path": str(Path(args.model).resolve()),
        "repeat": int(args.repeat_index),
        "order_position": int(args.order_position),
        "method": args.method,
        "presolve": args.presolve,
        "threads": int(args.threads),
        "tolerance": float(args.tol),
        "time_limit": float(args.time_limit),
    }


def _remaining(deadline):
    return max(0.0, deadline - time.perf_counter())


def _set_highs_options(highs, args, solver, remaining):
    options = {
        "output_flag": bool(args.printlevel),
        "solver": solver,
        "run_crossover": "on" if args.crossover else "off",
        "presolve": args.presolve,
        "threads": int(args.threads),
        "time_limit": float(max(0.001, remaining)),
        "primal_feasibility_tolerance": float(args.tol),
        "dual_feasibility_tolerance": float(args.tol),
        "ipm_optimality_tolerance": float(args.tol),
        "optimality_tolerance": float(args.tol),
        "kkt_tolerance": float(args.tol),
        "random_seed": int(args.seed),
    }
    for name, value in options.items():
        status = highs.setOptionValue(name, value)
        if name == "solver" and str(status).endswith("kError"):
            return status
        if str(status).endswith("kError"):
            raise RuntimeError(f"HiGHS rejected option {name}={value!r}")
    return None


def run_highs(args, solver):
    import highspy

    started = time.perf_counter()
    deadline = started + args.time_limit
    record = _base_record(args)

    if solver == "hipo":
        probe = highspy.Highs()
        probe.setOptionValue("output_flag", False)
        if str(probe.setOptionValue("solver", "hipo")).endswith("kError"):
            record.update(
                status="unavailable",
                read_seconds=0.0,
                setup_seconds=time.perf_counter() - started,
                solve_seconds=0.0,
                postsolve_seconds=0.0,
                total_seconds=time.perf_counter() - started,
                peak_memory_mb=_peak_memory_mb(),
                solver_version=probe.version(),
                error=(
                    "HiPO is unavailable. Install highspy-extras matching "
                    "highspy and restart Python."
                ),
            )
            return record

    read_started = time.perf_counter()
    raw = read_instance(args.model)
    record["read_seconds"] = time.perf_counter() - read_started
    record.update(
        rows=int(raw.A.shape[0]),
        columns=int(raw.A.shape[1]),
        nonzeros=int(raw.A.nnz),
        solver_version=highspy.Highs().version(),
    )

    setup_started = time.perf_counter()
    highs = highspy.Highs()
    option_error = _set_highs_options(
        highs, args, solver, _remaining(deadline)
    )
    if option_error is not None:
        record.update(
            status="unavailable",
            setup_seconds=time.perf_counter() - setup_started,
            solve_seconds=0.0,
            postsolve_seconds=0.0,
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=_peak_memory_mb(),
            error=(
                "HiPO is unavailable. Install highspy-extras matching highspy "
                "and restart Python."
            ),
        )
        return record
    highs.passModel(PS._highs_lp_from_raw(raw))
    record["setup_seconds"] = time.perf_counter() - setup_started
    remaining = _remaining(deadline)
    if remaining <= 0.0:
        record.update(
            status="setup-time-limit",
            solve_seconds=0.0,
            postsolve_seconds=0.0,
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=_peak_memory_mb(),
        )
        return record
    highs.setOptionValue("time_limit", float(remaining))

    solve_started = time.perf_counter()
    run_status = highs.run()
    record["solve_seconds"] = time.perf_counter() - solve_started

    post_started = time.perf_counter()
    model_status = highs.getModelStatus()
    status = _normal_status(model_status)
    info = highs.getInfo()
    solution = highs.getSolution()
    objective = float(highs.getObjectiveValue())
    pinf = float("nan")
    dinf = float("nan")
    if getattr(solution, "value_valid", False):
        check = PS._info_from_highs_solution(raw, solution)
        objective = float(check["obj"])
        pinf = float(check["pinf"])
        dinf = float(check["dinf"])
        if status == "optimal" and max(pinf, dinf) > args.tol:
            status = "postsolve-inaccurate"
    record["postsolve_seconds"] = time.perf_counter() - post_started
    record.update(
        status=status,
        model_status=str(model_status),
        run_status=str(run_status),
        objective=objective,
        primal_infeasibility=pinf,
        dual_infeasibility=dinf,
        iterations=int(getattr(info, "ipm_iteration_count", -1)),
        total_seconds=time.perf_counter() - started,
        peak_memory_mb=_peak_memory_mb(),
        timing_consistent=True,
    )
    return record


def run_psipm(args):
    started = time.perf_counter()
    deadline = started + args.time_limit
    record = _base_record(args)

    read_started = time.perf_counter()
    raw_orig = read_instance(args.model)
    record["read_seconds"] = time.perf_counter() - read_started
    record.update(
        rows=int(raw_orig.A.shape[0]),
        columns=int(raw_orig.A.shape[1]),
        nonzeros=int(raw_orig.A.nnz),
        solver_version="psipm-stable-active-preconditioner-1",
    )

    setup_started = time.perf_counter()
    presolver = None
    pstatus = "off"
    raw = raw_orig
    if args.presolve == "on":
        raw, presolver, pstatus = PS.presolve_raw(
            raw_orig, time_limit=_remaining(deadline)
        )
    lp = PS.to_standard_form(raw, verbose=False)
    lp = SAP.scale_problem(lp, verbose=False)
    record["setup_seconds"] = time.perf_counter() - setup_started
    record.update(
        presolve_status=str(pstatus),
        canonical_rows=int(lp.A.shape[0]),
        canonical_columns=int(lp.A.shape[1]),
        canonical_nonzeros=int(lp.A.nnz),
    )
    remaining = _remaining(deadline)
    if remaining <= 0.0:
        record.update(
            status="setup-time-limit",
            solve_seconds=0.0,
            postsolve_seconds=0.0,
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=_peak_memory_mb(),
        )
        return record

    solve_started = time.perf_counter()
    internal_tol = SAP.scaled_tolerance(lp, args.tol)
    sol = SAP.solve_standard(
        lp,
        tol=internal_tol,
        pc=args.pc,
        compl="gap",
        time_limit=remaining,
        maxit=args.maxit,
        inner_maxit=args.inner_maxit,
        printlevel=args.printlevel,
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
        order=args.cholmod_order,
        low_rank_max_columns=args.low_rank_max_columns,
        low_rank_min_nnz=args.low_rank_min_nnz,
        low_rank_clique_ratio=args.low_rank_clique_ratio,
    )
    record["solve_seconds"] = time.perf_counter() - solve_started

    post_started = time.perf_counter()
    precheck = PS.recover(lp, raw, sol)
    if presolver is not None and raw is not raw_orig:
        check = PS._postsolve_info(presolver, raw, raw_orig, precheck)
    else:
        check = precheck
    status = sol.status
    if status == "optimal" and max(check["pinf"], check["dinf"]) > args.tol:
        status = "postsolve-inaccurate"
    record["postsolve_seconds"] = time.perf_counter() - post_started

    factor_seconds = float(sol.t_fact)
    solve_seconds = float(record["solve_seconds"])
    timing_slack = max(0.05, 0.01 * solve_seconds)
    record.update(
        status=status,
        internal_tolerance=float(internal_tol),
        objective=float(check["obj"]),
        primal_infeasibility=float(check["pinf"]),
        dual_infeasibility=float(check["dinf"]),
        outer_iterations=int(sol.outer_iter),
        inner_iterations=int(sol.inner_iter),
        factor_seconds=factor_seconds,
        linear_solve_seconds=float(sol.t_solve),
        active_columns=int(sol.core_max),
        preconditioner_core_min=int(sol.core_min),
        preconditioner_core_mean=float(sol.core_mean),
        preconditioner_core_max=int(sol.core_max),
        preconditioner_refreshes=int(sol.preconditioner_refreshes),
        preconditioner_factorizations=int(sol.preconditioner_factorizations),
        preconditioner_retained_mean=float(sol.preconditioner_retained_mean),
        preconditioner_symbolic_analyses=int(
            sol.preconditioner_symbolic_analyses
        ),
        preconditioner_retries=int(sol.preconditioner_retries),
        preconditioner_full=bool(sol.preconditioner_full),
        pcg_iterations=int(sol.pcg_iterations),
        pcg_max_iterations=int(sol.pcg_max_iterations),
        pcg_failures=int(sol.pcg_failures),
        low_rank_columns=int(sol.low_rank_columns),
        low_rank_candidate_edges=int(sol.low_rank_candidate_edges),
        total_seconds=time.perf_counter() - started,
        peak_memory_mb=_peak_memory_mb(),
        timing_consistent=bool(factor_seconds <= solve_seconds + timing_slack),
    )
    return record


def worker(args):
    started = time.perf_counter()
    try:
        if args.method == "psipm-sap":
            record = run_psipm(args)
        elif args.method == "highs-ipx":
            record = run_highs(args, "ipx")
        elif args.method == "highs-hipo":
            record = run_highs(args, "hipo")
        else:
            raise ValueError(f"unknown method {args.method}")
    except Exception as exc:
        record = _base_record(args)
        record.update(
            status="exception",
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=_peak_memory_mb(),
            error=f"{type(exc).__name__}: {exc}",
        )
        if args.printlevel:
            record["error"] += "\n" + traceback.format_exc()
    print(json.dumps(record, sort_keys=True, allow_nan=True))


def _read_detail_rows(path):
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _completed_keys(path):
    return {
        (row.get("path"), row.get("method"), int(row.get("repeat", -1)))
        for row in _read_detail_rows(path)
    }


def _append_detail(path, record):
    exists = path.is_file() and path.stat().st_size > 0
    row = {field: record.get(field, "") for field in DETAIL_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(value):
    return math.isfinite(_as_float(value))


def _speedup(reference, candidate):
    reference_time = _as_float(reference.get("total_seconds"))
    candidate_time = _as_float(candidate.get("total_seconds"))
    if (
        reference.get("status") == "optimal"
        and candidate.get("status") == "optimal"
        and reference_time > 0.0
        and candidate_time > 0.0
    ):
        return reference_time / candidate_time
    return ""


def _objective_difference(candidate, reference):
    candidate_obj = _as_float(candidate.get("objective"))
    reference_obj = _as_float(reference.get("objective"))
    if math.isfinite(candidate_obj) and math.isfinite(reference_obj):
        return abs(candidate_obj - reference_obj) / max(1.0, abs(reference_obj))
    return ""


def write_summary(detail_path, summary_path):
    rows = _read_detail_rows(detail_path)
    grouped = {}
    for row in rows:
        key = (row.get("path", ""), int(row.get("repeat", -1)))
        grouped.setdefault(key, {})[row.get("method", "")] = row

    fields = [
        "problem", "path", "repeat", "rows", "columns", "nonzeros",
        "presolve", "threads", "tolerance", "time_limit",
    ]
    method_fields = {
        "psipm-sap": (
            "status", "total_seconds", "solve_seconds", "objective",
            "primal_infeasibility", "dual_infeasibility", "outer_iterations",
            "inner_iterations", "active_columns", "preconditioner_core_mean",
            "preconditioner_refreshes", "preconditioner_retained_mean",
            "preconditioner_factorizations",
            "preconditioner_symbolic_analyses", "preconditioner_retries",
            "preconditioner_full",
            "pcg_iterations", "pcg_max_iterations", "pcg_failures",
            "low_rank_columns", "low_rank_candidate_edges",
            "factor_seconds", "linear_solve_seconds", "peak_memory_mb",
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
        "psipm-sap": "psipm",
        "highs-ipx": "ipx",
        "highs-hipo": "hipo",
    }
    for method, names in method_fields.items():
        fields.extend(f"{prefixes[method]}_{name}" for name in names)
    fields.extend(
        (
            "psipm_speedup_vs_ipx",
            "psipm_speedup_vs_hipo",
            "psipm_ipx_objective_reldiff",
            "psipm_hipo_objective_reldiff",
        )
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            methods = grouped[key]
            representative = max(
                methods.values(), key=lambda row: bool(row.get("rows"))
            )
            output = {
                name: representative.get(name, "")
                for name in fields[:10]
            }
            for method, names in method_fields.items():
                source = methods.get(method, {})
                prefix = prefixes[method]
                for name in names:
                    output[f"{prefix}_{name}"] = source.get(name, "")
            psipm = methods.get("psipm-sap", {})
            ipx = methods.get("highs-ipx", {})
            hipo = methods.get("highs-hipo", {})
            output["psipm_speedup_vs_ipx"] = _speedup(ipx, psipm)
            output["psipm_speedup_vs_hipo"] = _speedup(hipo, psipm)
            output["psipm_ipx_objective_reldiff"] = _objective_difference(
                psipm, ipx
            )
            output["psipm_hipo_objective_reldiff"] = _objective_difference(
                psipm, hipo
            )
            writer.writerow(output)


def _summary_path(args, detail_path):
    if args.summary_csv:
        return Path(args.summary_csv).resolve()
    suffix = detail_path.suffix or ".csv"
    return detail_path.with_name(f"{detail_path.stem}_summary{suffix}")


def _hipo_available():
    try:
        import highspy

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        return not str(highs.setOptionValue("solver", "hipo")).endswith("kError")
    except Exception:
        return False


def _method_order(methods, model_index, repeat, args):
    methods = list(methods)
    if args.order == "fixed" or len(methods) < 2:
        return methods
    if args.order == "rotate":
        rng = random.Random(args.seed)
        rng.shuffle(methods)
        shift = (model_index + repeat) % len(methods)
        return methods[shift:] + methods[:shift]
    rng = random.Random(args.seed + 104729 * repeat + model_index)
    rng.shuffle(methods)
    return methods


def _worker_command(args, model, method, repeat, order_position):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        str(model),
        "--method",
        method,
        "--repeat-index",
        str(repeat),
        "--order-position",
        str(order_position),
        "--tol",
        str(args.tol),
        "--time-limit",
        str(args.time_limit),
        "--threads",
        str(args.threads),
        "--presolve",
        args.presolve,
        "--pc",
        str(args.pc),
        "--maxit",
        str(args.maxit),
        "--inner-maxit",
        str(args.inner_maxit),
        "--core-factor",
        str(args.core_factor),
        "--max-core-factor",
        str(args.max_core_factor),
        "--retention",
        str(args.retention),
        "--factor-max-age",
        str(args.factor_max_age),
        "--core-max-age",
        str(args.core_max_age),
        "--refresh-cg",
        str(args.refresh_cg),
        "--fill-weight",
        str(args.fill_weight),
        "--cg-tol",
        str(args.cg_tol),
        "--cg-tol-max",
        str(args.cg_tol_max),
        "--forcing-factor",
        str(args.forcing_factor),
        "--cg-maxit",
        str(args.cg_maxit),
        "--retry-growth",
        str(args.retry_growth),
        "--cholmod-order",
        args.cholmod_order,
        "--low-rank-max-columns",
        str(args.low_rank_max_columns),
        "--low-rank-min-nnz",
        str(args.low_rank_min_nnz),
        "--low-rank-clique-ratio",
        str(args.low_rank_clique_ratio),
        "--printlevel",
        str(args.printlevel),
        "--seed",
        str(args.seed),
    ]
    if args.crossover:
        command.append("--crossover")
    return command


def _parse_worker_output(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    raise ValueError("worker produced no JSON result")


def _print_summary_table(summary_path):
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print("\nproblem                       PSIPM total/status       IPX total/status         HiPO total/status")
    print("----------------------------  -----------------------  -----------------------  -----------------------")
    for row in rows:
        def cell(prefix):
            seconds = row.get(f"{prefix}_total_seconds", "")
            status = row.get(f"{prefix}_status", "-") or "-"
            if seconds:
                try:
                    return f"{float(seconds):8.2f}s {status:<12}"
                except ValueError:
                    pass
            return f"{'-':>9} {status:<12}"

        print(
            f"{row.get('problem', '')[:28]:<28}  {cell('psipm')}  "
            f"{cell('ipx')}  {cell('hipo')}"
        )


def orchestrate(args):
    models, skipped = find_models(
        args.folder,
        pattern=args.pattern,
        recursive=args.recursive,
        allow_duplicates=args.allow_duplicates,
    )
    if not models:
        raise FileNotFoundError(
            f"no supported models in {Path(args.folder).resolve()} matching "
            f"{args.pattern!r}"
        )
    print(f"found {len(models)} logical model(s) in {Path(args.folder).resolve()}")
    if skipped:
        print(f"deduplicated {len(skipped)} alternate encoding(s)")
    for model in models:
        print(f"  {_logical_name(model):<28} {model.name}")
    if args.list_only:
        return 0

    detail_path = Path(args.csv).resolve()
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and detail_path.exists():
        detail_path.unlink()
    if detail_path.exists() and not args.resume:
        raise FileExistsError(
            f"{detail_path} already exists; use --resume or --overwrite"
        )
    completed = _completed_keys(detail_path) if args.resume else set()

    hipo_ok = _hipo_available()
    if "highs-hipo" in args.methods and not hipo_ok:
        message = (
            "HiPO is unavailable: install highspy-extras matching highspy. "
            "Unavailable rows will still be written to the CSV."
        )
        if args.require_hipo:
            raise RuntimeError(message)
        print(f"warning: {message}")

    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = str(args.threads)

    total_runs = len(models) * args.repeats * len(args.methods)
    completed_now = 0
    for repeat in range(args.repeats):
        for model_index, model in enumerate(models):
            order = _method_order(args.methods, model_index, repeat, args)
            for order_position, method in enumerate(order, 1):
                key = (str(Path(model).resolve()), method, repeat)
                if key in completed:
                    print(
                        f"skip {_logical_name(model)} {method} repeat {repeat}",
                        flush=True,
                    )
                    continue
                completed_now += 1
                print(
                    f"[{completed_now}/{total_runs}] {_logical_name(model)} "
                    f"{method} repeat {repeat} ...",
                    end=" ",
                    flush=True,
                )
                command = _worker_command(
                    args, model, method, repeat, order_position
                )
                process_started = time.perf_counter()
                try:
                    process = subprocess.run(
                        command,
                        text=True,
                        capture_output=True,
                        timeout=args.time_limit + args.timeout_margin,
                        env=env,
                    )
                    process_seconds = time.perf_counter() - process_started
                    record = _parse_worker_output(process.stdout)
                    record["process_seconds"] = process_seconds
                    if process.returncode:
                        record["error"] = (
                            record.get("error", "")
                            + f" worker return code {process.returncode}; "
                            + process.stderr[-4000:]
                        ).strip()
                except subprocess.TimeoutExpired as exc:
                    process_seconds = time.perf_counter() - process_started
                    record = {
                        "problem": _logical_name(model),
                        "path": str(model),
                        "repeat": repeat,
                        "order_position": order_position,
                        "method": method,
                        "status": "external-timeout",
                        "presolve": args.presolve,
                        "threads": args.threads,
                        "tolerance": args.tol,
                        "time_limit": args.time_limit,
                        "total_seconds": args.time_limit,
                        "process_seconds": process_seconds,
                        "error": str(exc),
                    }
                except Exception as exc:
                    process_seconds = time.perf_counter() - process_started
                    record = {
                        "problem": _logical_name(model),
                        "path": str(model),
                        "repeat": repeat,
                        "order_position": order_position,
                        "method": method,
                        "status": "worker-failed",
                        "presolve": args.presolve,
                        "threads": args.threads,
                        "tolerance": args.tol,
                        "time_limit": args.time_limit,
                        "process_seconds": process_seconds,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _append_detail(detail_path, record)
                seconds = _as_float(record.get("total_seconds"))
                seconds_text = f"{seconds:.2f}s" if math.isfinite(seconds) else "-"
                status = str(record.get("status", "unknown"))
                error = str(record.get("error", "")).strip()
                diagnostic = ""
                if error and status in {
                    "exception", "worker-failed", "unavailable", "external-timeout"
                }:
                    diagnostic = " -- " + error.splitlines()[0][:500]
                print(f"{status} ({seconds_text}){diagnostic}", flush=True)
                if args.cooldown > 0.0:
                    time.sleep(args.cooldown)

    summary_path = _summary_path(args, detail_path)
    write_summary(detail_path, summary_path)
    _print_summary_table(summary_path)
    print(f"\ndetailed CSV: {detail_path}")
    print(f"summary CSV : {summary_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", nargs="?", help="folder containing LP models")
    parser.add_argument(
        "--choose-folder",
        action="store_true",
        help="open a graphical dialog to select the model folder",
    )
    parser.add_argument("--pattern", default="*", help="filename glob filter")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--list", dest="list_only", action="store_true")
    parser.add_argument("--csv", default="batch_results.csv")
    parser.add_argument("--summary-csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--order", choices=("rotate", "random", "fixed"), default="rotate")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=2000.0)
    parser.add_argument("--timeout-margin", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--presolve", choices=("on", "off"), default="off")
    parser.add_argument("--crossover", action="store_true")
    parser.add_argument("--require-hipo", action="store_true")
    parser.add_argument("--pc", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--maxit", type=int, default=160)
    parser.add_argument("--inner-maxit", type=int, default=100)
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
    parser.add_argument("--cholmod-order", default="best")
    parser.add_argument("--low-rank-max-columns", type=int, default=8)
    parser.add_argument("--low-rank-min-nnz", type=int, default=1024)
    parser.add_argument("--low-rank-clique-ratio", type=float, default=2.0)
    parser.add_argument("--printlevel", type=int, choices=(0, 1, 2), default=0)

    # Worker-only arguments are intentionally accepted by the same parser.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--order-position", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.model or not args.method:
            parser.error("--worker requires --model and --method")
    else:
        if args.choose_folder and args.folder:
            parser.error("pass a folder path or --choose-folder, not both")
        if args.choose_folder:
            try:
                args.folder = choose_folder()
            except RuntimeError as exc:
                parser.error(str(exc))
            if not args.folder:
                parser.error("folder selection was cancelled")
        elif not args.folder:
            parser.error("folder is required, or use --choose-folder")
    if args.resume and args.overwrite:
        parser.error("choose either --resume or --overwrite")
    if args.repeats <= 0 or args.threads <= 0 or args.time_limit <= 0.0:
        parser.error("repeats, threads, and time limit must be positive")
    if args.timeout_margin < 0.0 or args.cooldown < 0.0:
        parser.error("timeout margin and cooldown must be nonnegative")
    if args.core_factor <= 0.0 or args.max_core_factor < args.core_factor:
        parser.error("require 0 < core-factor <= max-core-factor")
    if not 0.0 <= args.retention <= 1.0:
        parser.error("retention must lie in [0, 1]")
    if not 0.0 < args.cg_tol <= args.cg_tol_max < 1.0 or args.cg_maxit <= 0:
        parser.error("require 0 < cg-tol <= cg-tol-max < 1 and cg-maxit > 0")
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    raise SystemExit(worker(ARGS) if ARGS.worker else orchestrate(ARGS))
