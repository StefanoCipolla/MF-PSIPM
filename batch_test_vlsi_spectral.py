#!/usr/bin/env python3
"""Benchmark spectral PSIPM, HiGHS IPX, and HiPO on Josef and Erhard.

The Bonn ``*.net`` files use DIMACS minimum-cost-flow format.  This driver
parses them directly into an incidence matrix, runs every method in a fresh
process, and writes durable detailed and summary CSV files.

Examples
--------
Run both models from the Iridis scratch directory::

    python batch_test_vlsi_spectral.py /scratch/sc9c23/vlsi --csv vlsi_results.csv

Resume an interrupted run::

    python batch_test_vlsi_spectral.py /scratch/sc9c23/vlsi \
        --csv vlsi_results.csv --resume

Run Josef only, including two repetitions::

    python batch_test_vlsi_spectral.py /scratch/sc9c23/vlsi \
        --models Josef --repeats 2 --csv josef_results.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None


HERE = Path(__file__).resolve().parent
DEFAULT_FOLDER = Path("/scratch/sc9c23/vlsi")
MODEL_FILES = {
    "Josef": "Josef_FlowGraph_1.net",
    "Erhard": "Erhard_FlowGraph_1.net",
}
METHODS = ("psipm-ssp", "highs-ipx", "highs-hipo")

DETAIL_FIELDS = (
    "problem",
    "path",
    "repeat",
    "order_position",
    "method",
    "status",
    "nodes",
    "arcs",
    "nonzeros",
    "supply_nodes",
    "threads",
    "tolerance",
    "time_limit",
    "objective_scale_factor",
    "rho",
    "delta",
    "core_factor",
    "max_core_factor",
    "network_start_epsilon",
    "grounded_rows",
    "grounded_nonzeros",
    "read_seconds",
    "setup_seconds",
    "solve_seconds",
    "validation_seconds",
    "total_seconds",
    "process_seconds",
    "objective",
    "primal_infeasibility",
    "dual_infeasibility",
    "relative_gap",
    "iterations",
    "outer_iterations",
    "inner_iterations",
    "factor_seconds",
    "linear_solve_seconds",
    "active_columns",
    "phase1_rounds",
    "pricing_rounds",
    "columns_added",
    "columns_dropped",
    "low_rank_columns",
    "low_rank_candidate_edges",
    "peak_memory_mb",
    "timing_consistent",
    "solver_version",
    "model_status",
    "error",
)


def _load_solver():
    path = HERE / "psipm_spectral_preconditioner.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} must be next to this tester")
    spec = importlib.util.spec_from_file_location(
        "psipm_spectral_preconditioner_vlsi", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SSP = _load_solver()
PS = SSP.PS


@dataclass
class NetworkData:
    lp: object
    supply_nodes: int


def _peak_memory_mb():
    if resource is not None:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak / (1024.0**2 if sys.platform == "darwin" else 1024.0)
    return float("nan")


def read_dimacs_header(path):
    """Read ``(nodes, arcs)`` without scanning the complete network."""
    with Path(path).open("rb") as handle:
        for raw_line in handle:
            if raw_line.startswith(b"p "):
                fields = raw_line.split()
                if len(fields) != 4 or fields[1] != b"min":
                    raise ValueError(f"unsupported DIMACS problem line: {raw_line!r}")
                return int(fields[2]), int(fields[3])
    raise ValueError(f"no DIMACS problem line found in {path}")


def read_dimacs_network(path) -> NetworkData:
    """Read one Bonn uncapacitated min-cost-flow model with bounded memory.

    The legalization files have zero lower bounds and ``-1`` upper bounds,
    meaning infinity.  Incidence values are stored as int8; numeric products
    promote them to floating point only when required by a solver.
    """
    path = Path(path)
    nodes, arcs = read_dimacs_header(path)
    if nodes >= np.iinfo(np.int32).max:
        raise ValueError("node identifiers do not fit in int32")

    balance = np.zeros(nodes, dtype=np.float64)
    costs = np.empty(arcs, dtype=np.float64)
    indices = np.empty(2 * arcs, dtype=np.int32)
    values = np.empty(2 * arcs, dtype=np.int8)
    arc_index = 0
    supply_nodes = 0

    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for raw_line in handle:
            if not raw_line:
                continue
            record = raw_line[0]
            if record == ord("n"):
                fields = raw_line.split()
                if len(fields) != 3:
                    raise ValueError(f"invalid node record in {path}: {raw_line!r}")
                node = int(fields[1]) - 1
                amount = float(fields[2])
                if node < 0 or node >= nodes:
                    raise ValueError(f"node id {node + 1} is outside 1..{nodes}")
                if balance[node] == 0.0 and amount != 0.0:
                    supply_nodes += 1
                balance[node] = amount
            elif record == ord("a"):
                if arc_index >= arcs:
                    raise ValueError(f"more than {arcs} arc records in {path}")
                fields = raw_line.split()
                if len(fields) != 6:
                    raise ValueError(f"invalid arc record in {path}: {raw_line!r}")
                tail = int(fields[1]) - 1
                head = int(fields[2]) - 1
                lower = float(fields[3])
                upper = float(fields[4])
                if lower != 0.0 or upper != -1.0:
                    raise ValueError(
                        "this VLSI tester expects zero-lower, uncapacitated arcs; "
                        f"found [{lower}, {upper}] in {path}"
                    )
                if not (0 <= tail < nodes and 0 <= head < nodes):
                    raise ValueError("arc endpoint is outside the declared node range")

                position = 2 * arc_index
                # DIMACS uses outflow - inflow = supply.  Store the two rows in
                # sorted order so CSC construction requires no sorting pass.
                if tail < head:
                    indices[position : position + 2] = (tail, head)
                    values[position : position + 2] = (1, -1)
                else:
                    indices[position : position + 2] = (head, tail)
                    values[position : position + 2] = (-1, 1)
                costs[arc_index] = float(fields[5])
                arc_index += 1

    if arc_index != arcs:
        raise ValueError(f"declared {arcs} arcs but read {arc_index} from {path}")
    balance_error = abs(float(balance.sum()))
    if balance_error > 1e-9 * (1.0 + float(np.abs(balance).sum())):
        raise ValueError(f"unbalanced network supplies: sum(b)={balance.sum():.6e}")

    indptr = np.arange(0, 2 * arcs + 1, 2, dtype=np.int64)
    incidence = sp.csc_matrix(
        (values, indices, indptr), shape=(nodes, arcs), copy=False
    )
    lp = PS.StandardLP(
        c=costs,
        A=incidence,
        b=balance,
        u=np.full(arcs, np.inf),
        pos=np.ones(arcs, dtype=bool),
        box=np.zeros(arcs, dtype=bool),
        cert_u=None,
        obj_const=0.0,
        scale=np.ones(arcs),
    )
    return NetworkData(lp=lp, supply_nodes=supply_nodes)


def ground_network(lp):
    """Remove one redundant balance row from every connected component."""
    A = lp.A
    counts = np.diff(A.indptr)
    if counts.size and not np.all(counts == 2):
        raise ValueError("network grounding requires two endpoints per arc")
    first = A.indices[A.indptr[:-1]]
    second = A.indices[A.indptr[:-1] + 1]
    rows = np.concatenate((first, second))
    columns = np.concatenate((second, first))
    graph = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, columns)),
        shape=(A.shape[0], A.shape[0]),
    )
    components, labels = csgraph.connected_components(
        graph, directed=False, return_labels=True
    )
    degrees = np.bincount(A.indices, minlength=A.shape[0])
    maximum = np.zeros(components, dtype=degrees.dtype)
    np.maximum.at(maximum, labels, degrees)
    candidates = degrees == maximum[labels]
    roots = np.full(components, A.shape[0], dtype=np.int64)
    indices = np.flatnonzero(candidates)
    np.minimum.at(roots, labels[indices], indices)
    if np.any(roots >= A.shape[0]):
        raise RuntimeError("failed to choose a grounding row")
    if np.any(np.abs(lp.b[roots]) > 0.0):
        # A nonzero root balance is still redundant when the component sum is
        # zero, but zero-balance roots give better finite-precision recovery.
        component_balance = np.bincount(labels, weights=lp.b)
        if np.max(np.abs(component_balance), initial=0.0) > 1e-12:
            raise ValueError("cannot ground an unbalanced network component")
    keep = np.ones(A.shape[0], dtype=bool)
    keep[roots] = False
    grounded = PS.StandardLP(
        c=lp.c,
        A=A[keep, :].tocsc(),
        b=lp.b[keep],
        u=lp.u,
        pos=lp.pos,
        box=lp.box,
        cert_u=lp.cert_u,
        obj_const=lp.obj_const,
        scale=lp.scale,
    )
    return grounded, roots


def _empty_record(args):
    return {field: "" for field in DETAIL_FIELDS} | {
        "problem": args.problem,
        "path": str(Path(args.model).resolve()),
        "repeat": args.repeat_index,
        "order_position": args.order_position,
        "method": args.method,
        "status": "unknown",
        "threads": args.threads,
        "tolerance": args.tol,
        "time_limit": args.time_limit,
        "rho": "" if args.rho is None else args.rho,
        "delta": "" if args.delta is None else args.delta,
        "core_factor": args.core_factor,
        "max_core_factor": args.max_core_factor,
        "network_start_epsilon": (
            "" if args.network_start_epsilon is None else args.network_start_epsilon
        ),
        "grounded_rows": "",
        "grounded_nonzeros": "",
    }


def _remaining(deadline):
    return max(0.0, deadline - time.perf_counter())


def run_psipm(args):
    started = time.perf_counter()
    deadline = started + args.time_limit
    record = _empty_record(args)

    read_started = time.perf_counter()
    network = read_dimacs_network(args.model)
    full_lp = network.lp
    record["read_seconds"] = time.perf_counter() - read_started
    record.update(
        nodes=full_lp.A.shape[0],
        arcs=full_lp.A.shape[1],
        nonzeros=full_lp.A.nnz,
        supply_nodes=network.supply_nodes,
    )

    setup_started = time.perf_counter()
    if args.ground_network == "auto":
        lp, grounded_roots = ground_network(full_lp)
    else:
        lp = full_lp
        grounded_roots = np.empty(0, dtype=np.int64)
    original_cost = lp.c.copy()
    cost_max = float(np.abs(original_cost).max(initial=0.0))
    objective_scale = (
        float(2.0 ** np.ceil(np.log2(cost_max)))
        if args.objective_scaling == "auto" and cost_max > 1.0
        else 1.0
    )
    lp.c = original_cost / objective_scale
    record.update(
        objective_scale_factor=objective_scale,
        solver_version="psipm-spectral-stable-preconditioner-2-vlsi",
        setup_seconds=time.perf_counter() - setup_started,
        grounded_rows=int(grounded_roots.size),
        grounded_nonzeros=int(full_lp.A.nnz - lp.A.nnz),
    )
    remaining = _remaining(deadline)
    if remaining <= 0.0:
        record.update(
            status="read-time-limit",
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=_peak_memory_mb(),
        )
        return record

    initial = None
    if args.network_start_epsilon is not None:
        epsilon = float(args.network_start_epsilon)
        x0 = np.full(lp.A.shape[1], epsilon)
        y0 = np.zeros(lp.A.shape[0])
        # A uniform positive shift makes z strictly interior while preserving
        # the scaled objective's relative reduced costs.
        shift = 1.0 + max(0.0, -float(lp.c.min(initial=0.0)))
        z0 = lp.c + shift
        w0 = np.zeros(lp.A.shape[1])
        initial = (x0, y0, z0, w0)

    solve_started = time.perf_counter()
    # The incidence matrix is already max-norm equilibrated: every nonzero is
    # +/-1 and every nonempty row and column has max norm one.  Avoid making a
    # redundant copy of this multi-million-row matrix.
    solution = SSP.solve_standard(
        lp,
        tol=args.tol,
        pc=args.pc,
        compl="gap",
        time_limit=remaining,
        maxit=args.maxit,
        inner_maxit=args.inner_maxit,
        rho=args.rho,
        delta=args.delta,
        start_projection=args.start_projection,
        start_linear_tolerance=args.start_linear_tolerance,
        initial=initial,
        printlevel=args.printlevel,
        core_factor=args.core_factor,
        max_core_factor=args.max_core_factor,
        cg_maxit=args.cg_maxit,
        structural_backbone=False,
        weighted_backbone=False,
        low_rank_max_columns=args.low_rank_max_columns,
        low_rank_min_nnz=args.low_rank_min_nnz,
        low_rank_clique_ratio=args.low_rank_clique_ratio,
    )
    solve_seconds = time.perf_counter() - solve_started

    validation_started = time.perf_counter()
    # Undo the positive objective scaling exactly in the dual variables before
    # validating against the original network LP.
    solution.y *= objective_scale
    solution.z *= objective_scale
    solution.w *= objective_scale
    lp.c = original_cost
    pinf, dinf, gap, objective = PS._standard_kkt_metrics(
        lp, solution.x, solution.y, solution.z, solution.w
    )
    if grounded_roots.size:
        # The removed equations are redundant algebraically, but evaluate
        # them explicitly to catch accumulated finite-precision imbalance.
        full_rp = full_lp.b - full_lp.A @ solution.x
        full_pden = 1.0 + float(np.abs(full_lp.b).max(initial=0.0))
        pinf = float(np.abs(full_rp).max(initial=0.0)) / full_pden
    validation_seconds = time.perf_counter() - validation_started
    status = solution.status
    if status == "optimal" and max(pinf, dinf, gap) > args.tol:
        status = "validation-inaccurate"
    timing_slack = max(0.05, 0.01 * solve_seconds)
    record.update(
        status=status,
        solve_seconds=solve_seconds,
        validation_seconds=validation_seconds,
        total_seconds=time.perf_counter() - started,
        objective=float(objective),
        primal_infeasibility=float(pinf),
        dual_infeasibility=float(dinf),
        relative_gap=float(gap),
        outer_iterations=int(solution.outer_iter),
        inner_iterations=int(solution.inner_iter),
        factor_seconds=float(solution.t_fact),
        linear_solve_seconds=float(solution.t_solve),
        active_columns=int(solution.core_max),
        phase1_rounds="",
        pricing_rounds="",
        columns_added="",
        columns_dropped="",
        low_rank_columns=int(solution.low_rank_columns),
        low_rank_candidate_edges=int(solution.low_rank_candidate_edges),
        peak_memory_mb=_peak_memory_mb(),
        timing_consistent=bool(solution.t_fact <= solve_seconds + timing_slack),
    )
    return record


def _highs_status(model_status):
    name = str(model_status)
    mapping = {
        "kOptimal": "optimal",
        "kTimeLimit": "time-limit",
        "kInfeasible": "infeasible",
        "kUnbounded": "unbounded",
        "kUnboundedOrInfeasible": "unbounded-or-infeasible",
        "kIterationLimit": "iteration-limit",
        "kMemoryLimit": "memory-limit",
    }
    for suffix, status in mapping.items():
        if name.endswith(suffix):
            return status
    return name.rsplit(".", 1)[-1]


def run_highs(args, solver_name):
    import highspy

    started = time.perf_counter()
    deadline = started + args.time_limit
    record = _empty_record(args)

    read_started = time.perf_counter()
    network = read_dimacs_network(args.model)
    lp = network.lp
    record["read_seconds"] = time.perf_counter() - read_started
    record.update(
        nodes=lp.A.shape[0],
        arcs=lp.A.shape[1],
        nonzeros=lp.A.nnz,
        supply_nodes=network.supply_nodes,
    )

    setup_started = time.perf_counter()
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", bool(args.printlevel))
    highs.setOptionValue("threads", args.threads)
    highs.setOptionValue("presolve", "off")
    highs.setOptionValue("run_crossover", "off")
    option_status = highs.setOptionValue("solver", solver_name)
    if str(option_status).endswith("kError"):
        record.update(
            status="unavailable",
            solver_version=f"HiGHS {highs.version()} / {solver_name}",
            setup_seconds=time.perf_counter() - setup_started,
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=_peak_memory_mb(),
            error=f"HiGHS does not provide solver={solver_name!r}",
        )
        return record

    model = highspy.HighsLp()
    model.num_col_ = lp.A.shape[1]
    model.num_row_ = lp.A.shape[0]
    model.col_cost_ = lp.c
    model.col_lower_ = np.zeros(lp.A.shape[1])
    model.col_upper_ = lp.u
    model.row_lower_ = lp.b
    model.row_upper_ = lp.b
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = lp.A.indptr
    model.a_matrix_.index_ = lp.A.indices
    model.a_matrix_.value_ = lp.A.data
    pass_status = highs.passModel(model)
    if str(pass_status).endswith("kError"):
        raise RuntimeError(f"HiGHS passModel failed: {pass_status}")
    record["setup_seconds"] = time.perf_counter() - setup_started

    remaining = _remaining(deadline)
    if remaining <= 0.0:
        record.update(
            status="setup-time-limit",
            total_seconds=time.perf_counter() - started,
            peak_memory_mb=_peak_memory_mb(),
        )
        return record
    highs.setOptionValue("time_limit", remaining)

    solve_started = time.perf_counter()
    run_status = highs.run()
    solve_seconds = time.perf_counter() - solve_started
    validation_started = time.perf_counter()
    model_status = highs.getModelStatus()
    info = highs.getInfo()
    validation_seconds = time.perf_counter() - validation_started
    record.update(
        status=_highs_status(model_status),
        solve_seconds=solve_seconds,
        validation_seconds=validation_seconds,
        total_seconds=time.perf_counter() - started,
        objective=float(highs.getObjectiveValue()),
        primal_infeasibility=float(
            getattr(info, "max_primal_infeasibility", np.nan)
        ),
        dual_infeasibility=float(
            getattr(info, "max_dual_infeasibility", np.nan)
        ),
        relative_gap=float(
            getattr(info, "primal_dual_objective_error", np.nan)
        ),
        iterations=int(getattr(info, "ipm_iteration_count", -1)),
        peak_memory_mb=_peak_memory_mb(),
        timing_consistent=True,
        solver_version=f"HiGHS {highs.version()} / {solver_name}",
        model_status=str(model_status),
        error="" if not str(run_status).endswith("kError") else str(run_status),
    )
    return record


def worker(args):
    try:
        if args.method == "psipm-ssp":
            return run_psipm(args)
        if args.method == "highs-ipx":
            return run_highs(args, "ipm")
        return run_highs(args, "hipo")
    except Exception as exc:
        record = _empty_record(args)
        record.update(
            status="exception",
            total_seconds=float("nan"),
            peak_memory_mb=_peak_memory_mb(),
            error=(
                f"{type(exc).__name__}: {exc}\n"
                + traceback.format_exc(limit=12)
            ),
        )
        return record


def discover_models(folder, requested):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"VLSI folder does not exist: {folder}")
    wanted = list(MODEL_FILES) if requested == ["all"] else requested
    found = {}
    for path in folder.rglob("*.net"):
        for problem in wanted:
            if path.name.lower() == MODEL_FILES[problem].lower():
                if problem in found:
                    raise ValueError(f"multiple files found for {problem}")
                found[problem] = path
    missing = [problem for problem in wanted if problem not in found]
    if missing:
        names = ", ".join(MODEL_FILES[name] for name in missing)
        raise FileNotFoundError(f"missing from {folder}: {names}")
    return [(name, found[name]) for name in wanted]


def _read_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DETAIL_FIELDS:
            raise ValueError("existing CSV has a different schema; use --overwrite")
        for row in reader:
            completed.add(
                (row.get("path", ""), row.get("method", ""), row.get("repeat", ""))
            )
    return completed


def _append_record(path, record):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in DETAIL_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_summary(detail_path, summary_path):
    with detail_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["path"], row["repeat"]), {})[row["method"]] = row

    base_fields = ("problem", "path", "repeat", "nodes", "arcs", "nonzeros")
    method_fields = (
        "status",
        "total_seconds",
        "solve_seconds",
        "objective",
        "primal_infeasibility",
        "dual_infeasibility",
        "peak_memory_mb",
    )
    fields = list(base_fields)
    prefixes = {
        "psipm-ssp": "psipm",
        "highs-ipx": "ipx",
        "highs-hipo": "hipo",
    }
    for method in METHODS:
        fields.extend(f"{prefixes[method]}_{name}" for name in method_fields)
    fields.extend(
        (
            "psipm_active_columns",
            "psipm_low_rank_columns",
            "psipm_low_rank_candidate_edges",
            "psipm_speedup_vs_ipx",
            "psipm_speedup_vs_hipo",
        )
    )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for _, methods in sorted(grouped.items()):
            representative = next(iter(methods.values()))
            output = {name: representative.get(name, "") for name in base_fields}
            for method in METHODS:
                row = methods.get(method, {})
                prefix = prefixes[method]
                for name in method_fields:
                    output[f"{prefix}_{name}"] = row.get(name, "")
            psipm = methods.get("psipm-ssp", {})
            output["psipm_active_columns"] = psipm.get("active_columns", "")
            output["psipm_low_rank_columns"] = psipm.get("low_rank_columns", "")
            output["psipm_low_rank_candidate_edges"] = psipm.get(
                "low_rank_candidate_edges", ""
            )
            for reference, field in (
                ("highs-ipx", "psipm_speedup_vs_ipx"),
                ("highs-hipo", "psipm_speedup_vs_hipo"),
            ):
                reference_row = methods.get(reference, {})
                ptime = _as_float(psipm.get("total_seconds"))
                rtime = _as_float(reference_row.get("total_seconds"))
                output[field] = (
                    rtime / ptime
                    if psipm.get("status") == "optimal"
                    and reference_row.get("status") == "optimal"
                    and ptime > 0.0
                    and rtime > 0.0
                    else ""
                )
            writer.writerow(output)


def _worker_command(args, problem, model, method, repeat, position):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--problem",
        problem,
        "--model",
        str(model),
        "--method",
        method,
        "--repeat-index",
        str(repeat),
        "--order-position",
        str(position),
    ]
    forwarded = (
        "tol",
        "time_limit",
        "threads",
        "pc",
        "maxit",
        "inner_maxit",
        "core_factor",
        "max_core_factor",
        "cg_maxit",
        "rho",
        "delta",
        "objective_scaling",
        "start_projection",
        "start_linear_tolerance",
        "network_start_epsilon",
        "ground_network",
        "low_rank_max_columns",
        "low_rank_min_nnz",
        "low_rank_clique_ratio",
        "printlevel",
    )
    for name in forwarded:
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    return command


def _parse_worker_output(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "status" in result:
            return result
    raise ValueError("worker produced no JSON record")


def orchestrate(args):
    models = discover_models(args.folder, args.models)
    print(f"VLSI folder: {Path(args.folder).expanduser().resolve()}")
    for problem, path in models:
        nodes, arcs = read_dimacs_header(path)
        print(f"  {problem:<7} {nodes:,} nodes, {arcs:,} arcs  {path.name}")
    if args.list_only:
        return 0

    detail_path = Path(args.csv).expanduser().resolve()
    summary_path = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else detail_path.with_name(detail_path.stem + "_summary.csv")
    )
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and detail_path.exists():
        detail_path.unlink()
    elif detail_path.exists() and not args.resume:
        raise FileExistsError(
            f"{detail_path} already exists; select --resume or --overwrite"
        )
    completed = _read_completed(detail_path) if args.resume else set()

    jobs = []
    for model_index, (problem, model) in enumerate(models):
        for repeat in range(args.repeats):
            methods = list(args.methods)
            if args.order == "rotate":
                shift = (model_index + repeat) % len(methods)
                methods = methods[shift:] + methods[:shift]
            elif args.order == "random":
                random.Random(args.seed + model_index + 104729 * repeat).shuffle(methods)
            for position, method in enumerate(methods):
                key = (str(model.resolve()), method, str(repeat))
                if key not in completed:
                    jobs.append((problem, model, method, repeat, position))

    print(f"scheduled {len(jobs)} fresh-process run(s)")
    for index, (problem, model, method, repeat, position) in enumerate(jobs, 1):
        print(
            f"[{index}/{len(jobs)}] {problem} {method} repeat {repeat} ... ",
            end="",
            flush=True,
        )
        command = _worker_command(args, problem, model, method, repeat, position)
        environment = os.environ.copy()
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[variable] = str(args.threads)
        process_started = time.perf_counter()
        try:
            completed_process = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.time_limit + args.timeout_margin,
                env=environment,
                check=False,
            )
            process_seconds = time.perf_counter() - process_started
            record = _parse_worker_output(completed_process.stdout)
            record["process_seconds"] = process_seconds
            if completed_process.returncode != 0 and not record.get("error"):
                record["error"] = completed_process.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            process_seconds = time.perf_counter() - process_started
            record = {
                "problem": problem,
                "path": str(model.resolve()),
                "repeat": repeat,
                "order_position": position,
                "method": method,
                "status": "external-timeout",
                "threads": args.threads,
                "tolerance": args.tol,
                "time_limit": args.time_limit,
                "process_seconds": process_seconds,
                "total_seconds": process_seconds,
                "error": str(exc),
            }
        _append_record(detail_path, record)
        seconds = _as_float(record.get("total_seconds"))
        elapsed = f"{seconds:.2f}s" if math.isfinite(seconds) else "-"
        print(f"{record.get('status', 'unknown')} ({elapsed})", flush=True)

    if not detail_path.exists():
        raise RuntimeError("no result rows exist; nothing to summarize")
    write_summary(detail_path, summary_path)
    print(f"detailed CSV: {detail_path}")
    print(f"summary CSV : {summary_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_FOLDER))
    parser.add_argument(
        "--models", nargs="+", choices=("all", "Josef", "Erhard"), default=["all"]
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--list", dest="list_only", action="store_true")
    parser.add_argument("--csv", default="vlsi_spectral_results.csv")
    parser.add_argument("--summary-csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--order", choices=("rotate", "random", "fixed"), default="rotate")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=4000.0)
    parser.add_argument("--timeout-margin", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pc", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--maxit", type=int, default=160)
    parser.add_argument("--inner-maxit", type=int, default=100)
    parser.add_argument("--core-factor", type=float, default=1.0)
    parser.add_argument("--max-core-factor", type=float, default=3.0)
    parser.add_argument("--cg-maxit", type=int, default=200)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument(
        "--objective-scaling", choices=("auto", "off"), default="auto"
    )
    parser.add_argument(
        "--start-projection", choices=("legacy", "adaptive"), default="legacy"
    )
    parser.add_argument("--start-linear-tolerance", type=float)
    parser.add_argument("--network-start-epsilon", type=float)
    parser.add_argument(
        "--ground-network",
        choices=("auto", "off"),
        default="auto",
        help="remove one redundant balance row per connected component",
    )
    parser.add_argument("--low-rank-max-columns", type=int, default=8)
    parser.add_argument("--low-rank-min-nnz", type=int, default=1024)
    parser.add_argument("--low-rank-clique-ratio", type=float, default=2.0)
    parser.add_argument("--printlevel", type=int, choices=(0, 1, 2), default=0)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--problem", default="", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--order-position", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats <= 0 or args.threads <= 0:
        parser.error("repeats and threads must be positive")
    if args.time_limit <= 0.0 or args.timeout_margin < 0.0:
        parser.error("time limit must be positive and timeout margin nonnegative")
    if args.rho is not None and args.rho <= 0.0:
        parser.error("rho must be positive")
    if args.delta is not None and args.delta <= 0.0:
        parser.error("delta must be positive")
    if (
        args.start_linear_tolerance is not None
        and not 0.0 < args.start_linear_tolerance < 1.0
    ):
        parser.error("start-linear-tolerance must lie in (0, 1)")
    if (
        args.network_start_epsilon is not None
        and args.network_start_epsilon <= 0.0
    ):
        parser.error("network-start-epsilon must be positive")
    if args.models != ["all"] and "all" in args.models:
        parser.error("use --models all by itself")
    if args.worker and (not args.model or not args.method):
        parser.error("worker mode requires --model and --method")
    return args


def main():
    args = parse_args()
    if args.worker:
        print(json.dumps(worker(args), allow_nan=True))
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
