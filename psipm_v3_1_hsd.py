#!/usr/bin/env python3
"""PSIPM-V3.1: generic homogeneous self-dual proximal IPM.

The implementation solves sparse equality-and-box LPs

    minimize    c.T x
    subject to  A x = b, lower <= x <= upper.

It combines four ideas:

* the infeasible homogeneous self-dual (HSD) equations, so Phase I artificial
  variables are unnecessary and infeasibility rays can be recovered;
* a proximal-stabilized IPM (PS-IPM) whose primal, dual, and homogeneous
  regularizations stay strictly positive and are not tied to the barrier;
* matrix-free normal-equation products with stable active-column elimination
  inside the preconditioner, never inside the Newton operator;
* an incidence-matrix specialization whose restricted master uses a protected
  spanning forest, full pricing, and reversible column elimination.

This is an experimental research solver, not a replacement for HiGHS.  It
declares optimality only after checking the full, original LP.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph


TINY = float(np.finfo(np.float64).tiny)
EPS = float(np.finfo(np.float64).eps)


@dataclass
class CanonicalLP:
    """A nonnegative equality-form LP."""

    c: np.ndarray
    A: sp.csc_matrix
    b: np.ndarray
    name: str = ""
    objective_scale: float = 1.0
    u: Optional[np.ndarray] = None

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        self.A = self.A.tocsc(copy=False)
        if self.u is None:
            self.u = np.full(self.c.size, np.inf, dtype=np.float64)
        else:
            self.u = np.asarray(self.u, dtype=np.float64)
        if self.A.shape != (self.b.size, self.c.size):
            raise ValueError("A, b, and c have incompatible dimensions")
        if self.u.shape != self.c.shape or np.any(self.u <= 0.0):
            raise ValueError("upper bounds must be positive or +infinity")
        if not np.all(np.isfinite(self.c)) or not np.all(np.isfinite(self.b)):
            raise ValueError("V3 requires finite b and c")
        if not np.all(np.isfinite(self.A.data)):
            raise ValueError("V3 requires finite matrix entries")


@dataclass
class BoxLP:
    """A generic equality-constrained LP with arbitrary box bounds."""

    c: np.ndarray
    A: sp.csc_matrix
    b: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    name: str = ""
    objective_offset: float = 0.0

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        self.A = self.A.tocsc(copy=False)
        n = self.c.size
        self.lower = (
            np.zeros(n, dtype=np.float64)
            if self.lower is None
            else np.asarray(self.lower, dtype=np.float64)
        )
        self.upper = (
            np.full(n, np.inf, dtype=np.float64)
            if self.upper is None
            else np.asarray(self.upper, dtype=np.float64)
        )
        if self.A.shape != (self.b.size, n):
            raise ValueError("A, b, and c have incompatible dimensions")
        if self.lower.shape != (n,) or self.upper.shape != (n,):
            raise ValueError("box bounds have incompatible dimensions")
        if np.any(np.isnan(self.lower)) or np.any(np.isnan(self.upper)):
            raise ValueError("box bounds cannot contain NaN")
        if np.any(self.lower > self.upper):
            raise ValueError("a lower bound exceeds its upper bound")
        if not np.all(np.isfinite(self.c)) or not np.all(np.isfinite(self.b)):
            raise ValueError("V3 requires finite b and c")
        if not np.all(np.isfinite(self.A.data)):
            raise ValueError("V3 requires finite matrix entries")


@dataclass
class BoxTransform:
    """Affine map ``x = shift + T q`` from canonical to box variables."""

    shift: np.ndarray
    source: np.ndarray
    sign: np.ndarray
    original_size: int
    objective_constant: float

    def recover_primal(self, canonical_x: np.ndarray) -> np.ndarray:
        value = self.shift.copy()
        np.add.at(value, self.source, self.sign * canonical_x)
        return value

    def recover_direction(self, canonical_x: np.ndarray) -> np.ndarray:
        value = np.zeros(self.original_size, dtype=np.float64)
        np.add.at(value, self.source, self.sign * canonical_x)
        return value


@dataclass
class V3Options:
    tolerance: float = 1.0e-6
    time_limit: float = 2000.0
    max_iterations: int = 500
    rmp_steps: int = 3
    proximal_steps: int = 1
    proximal_reduction: float = 0.7
    proximal_residual_factor: float = 1.0e4
    proximal_relative_tolerance: float = 0.9
    rho: float = 1.0e-8
    delta: float = 1.0e-8
    eta: float = 1.0e-8
    zeta: float = 1.0e-8
    regularization_floor: float = 1.0e-12
    fraction_to_boundary: float = 0.995
    pool_factor: float = 1.0
    pricing_batch: int = 250_000
    pricing_tolerance: float = 1.0e-7
    max_pricing_rounds: int = 80
    hsd_restart_tau: float = 1.0e-4
    drop_tolerance: float = 1.0e-9
    drop_start: float = 1.0e-3
    drop_hysteresis: float = 10.0
    drop_patience: int = 2
    drop_batch: int = 250_000
    master_column_elimination: bool = False
    pcg_tolerance_early: float = 1.0e-8
    pcg_tolerance_late: float = 1.0e-10
    pcg_max_iterations: int = 1000
    generic_preconditioner: str = "auto"
    generic_core_factor: float = 1.0
    generic_max_core_factor: float = 3.0
    generic_retention: float = 0.9
    generic_factor_max_age: int = 2
    generic_core_max_age: int = 12
    generic_refresh_cg: int = 60
    generic_fill_weight: float = 0.2
    generic_retry_growth: float = 1.6
    generic_growth_amortization: float = 1.0
    generic_cholmod_order: str = "best"
    generic_low_rank_max_columns: int = 8
    generic_low_rank_min_nnz: int = 1024
    generic_low_rank_clique_ratio: float = 2.0
    objective_scaling: bool = True
    print_level: int = 1

    def __post_init__(self):
        floor = max(float(self.regularization_floor), 32.0 * EPS)
        self.rho = max(float(self.rho), floor)
        self.delta = max(float(self.delta), floor)
        self.eta = max(float(self.eta), floor)
        self.zeta = max(float(self.zeta), floor)
        if not (0.0 < self.fraction_to_boundary < 1.0):
            raise ValueError("fraction_to_boundary must lie in (0, 1)")
        if self.rmp_steps < 1 or self.proximal_steps < 1:
            raise ValueError("rmp_steps and proximal_steps must be positive")
        if not (0.0 < self.proximal_reduction < 1.0):
            raise ValueError("proximal_reduction must lie in (0, 1)")
        if self.proximal_residual_factor <= 0.0:
            raise ValueError("proximal_residual_factor must be positive")
        if not (0.0 < self.proximal_relative_tolerance < 1.0):
            raise ValueError("proximal_relative_tolerance must lie in (0, 1)")
        if self.generic_preconditioner not in {"auto", "stable", "diagonal"}:
            raise ValueError(
                "generic_preconditioner must be auto, stable, or diagonal"
            )
        if self.generic_core_factor <= 0.0:
            raise ValueError("generic_core_factor must be positive")
        if self.generic_max_core_factor < self.generic_core_factor:
            raise ValueError(
                "generic_max_core_factor must be at least generic_core_factor"
            )
        if self.generic_retry_growth <= 1.0:
            raise ValueError("generic_retry_growth must exceed one")
        if self.generic_growth_amortization < 0.0:
            raise ValueError("generic_growth_amortization must be nonnegative")
        if self.generic_low_rank_max_columns < 0:
            raise ValueError("generic_low_rank_max_columns must be nonnegative")


@dataclass
class HSDState:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    w: np.ndarray
    tau: float
    kappa: float
    theta: float
    x_center: np.ndarray
    y_center: np.ndarray
    tau_center: float
    theta_center: float
    center_age: int = 0
    center_residual: float = math.inf
    embedding: Optional[HSDEmbedding] = None


@dataclass
class HSDEmbedding:
    """Fixed residual data and normalization of the extended HSD model."""

    bar_b: np.ndarray
    bar_c: np.ndarray
    bar_g: float
    alpha: float


@dataclass
class LinearStats:
    iterations: int = 0
    max_iterations: int = 0
    solves: int = 0
    matvecs: int = 0
    failures: int = 0
    seconds: float = 0.0
    preconditioner_seconds: float = 0.0


@dataclass
class PreconditionerStats:
    """Auditable statistics for the eliminated-column factor core."""

    core_min: int = 0
    core_max: int = 0
    core_sum: int = 0
    core_observations: int = 0
    refreshes: int = 0
    factorizations: int = 0
    symbolic_analyses: int = 0
    retained_sum: float = 0.0
    retained_observations: int = 0
    growth_events: int = 0
    max_factor_nonzeros: int = 0
    low_rank_columns: int = 0
    full_preconditioner: bool = False

    def observe_core(self, size: int) -> None:
        value = int(size)
        if self.core_observations == 0:
            self.core_min = value
        else:
            self.core_min = min(self.core_min, value)
        self.core_max = max(self.core_max, value)
        self.core_sum += value
        self.core_observations += 1

    @property
    def core_mean(self) -> float:
        return self.core_sum / max(1, self.core_observations)

    @property
    def retained_mean(self) -> float:
        return self.retained_sum / max(1, self.retained_observations)


@dataclass
class ProximalStats:
    """Diagnostics for certified inexact proximal subproblems."""

    accepted: int = 0
    checks: int = 0
    last_ratio: float = math.inf
    max_accepted_ratio: float = 0.0
    last_residual_norm: float = math.inf
    last_enlargement_error: float = math.inf
    last_displacement_norm: float = 0.0
    last_target: float = 0.0
    last_hpe_ratio: float = math.inf


@dataclass
class V3Result:
    status: str
    objective: float = math.nan
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    tau: float = math.nan
    kappa: float = math.nan
    theta: float = math.nan
    primal_infeasibility: float = math.inf
    dual_infeasibility: float = math.inf
    relative_gap: float = math.inf
    hsd_residual: float = math.inf
    complementarity: float = math.inf
    iterations: int = 0
    pricing_rounds: int = 0
    active_columns: int = 0
    columns_added: int = 0
    columns_dropped: int = 0
    hsd_restarts: int = 0
    feasible_start_support: int = 0
    structure: str = ""
    preconditioner: str = ""
    setup_seconds: float = 0.0
    solve_seconds: float = 0.0
    pricing_seconds: float = 0.0
    total_seconds: float = 0.0
    linear: LinearStats = field(default_factory=LinearStats)
    preconditioner_stats: PreconditionerStats = field(
        default_factory=PreconditionerStats
    )
    proximal: ProximalStats = field(default_factory=ProximalStats)
    certificate: Optional[np.ndarray] = None
    message: str = ""

    def as_dict(self):
        record = dict(self.__dict__)
        record.pop("x", None)
        record.pop("y", None)
        record.pop("z", None)
        record.pop("w", None)
        record.pop("certificate", None)
        record["linear"] = dict(self.linear.__dict__)
        record["preconditioner_stats"] = dict(
            self.preconditioner_stats.__dict__
        )
        record["preconditioner_stats"]["core_mean"] = (
            self.preconditioner_stats.core_mean
        )
        record["preconditioner_stats"]["retained_mean"] = (
            self.preconditioner_stats.retained_mean
        )
        record["proximal"] = dict(self.proximal.__dict__)
        return record


def _power_of_two_scale(values: np.ndarray) -> float:
    maximum = float(np.max(np.abs(values), initial=0.0))
    if maximum == 0.0 or not np.isfinite(maximum):
        return 1.0
    return float(2.0 ** math.ceil(math.log2(maximum)))


def scale_objective(lp: CanonicalLP, enabled: bool = True) -> CanonicalLP:
    """Power-of-two cost scaling; exact for binary floating-point numbers."""
    factor = _power_of_two_scale(lp.c) if enabled else 1.0
    return CanonicalLP(
        c=lp.c / factor,
        A=lp.A,
        b=lp.b,
        name=lp.name,
        objective_scale=lp.objective_scale * factor,
        u=lp.u,
    )


def canonicalize_box_lp(lp: BoxLP) -> tuple[CanonicalLP, BoxTransform]:
    """Convert arbitrary box bounds to nonnegative variables without rows.

    A finite lower bound uses ``x_j = lower_j + q_j``.  An upper-only
    variable uses ``x_j = upper_j - q_j``.  A free variable is split as
    ``x_j = q_j^+ - q_j^-`` and a fixed variable is substituted exactly.
    """
    n = lp.c.size
    finite_lower = np.isfinite(lp.lower)
    finite_upper = np.isfinite(lp.upper)
    fixed = finite_lower & finite_upper & (lp.lower == lp.upper)

    shift = np.zeros(n, dtype=np.float64)
    shift[finite_lower] = lp.lower[finite_lower]
    upper_only = ~finite_lower & finite_upper
    shift[upper_only] = lp.upper[upper_only]

    lower_columns = np.flatnonzero(finite_lower & ~fixed)
    upper_columns = np.flatnonzero(upper_only)
    free_columns = np.flatnonzero(~finite_lower & ~finite_upper)
    source = np.concatenate((lower_columns, upper_columns, free_columns, free_columns))
    sign = np.concatenate(
        (
            np.ones(lower_columns.size),
            -np.ones(upper_columns.size),
            np.ones(free_columns.size),
            -np.ones(free_columns.size),
        )
    )
    A = lp.A[:, source].tocsc()
    if source.size:
        A = (A @ sp.diags(sign, format="csc")).tocsc()
    A.sum_duplicates()
    A.sort_indices()
    c = lp.c[source] * sign
    upper = np.full(source.size, np.inf, dtype=np.float64)
    if lower_columns.size:
        bounded = finite_upper[lower_columns]
        bounded_positions = np.flatnonzero(bounded)
        upper[bounded_positions] = (
            lp.upper[lower_columns[bounded]] - lp.lower[lower_columns[bounded]]
        )
    b = lp.b - lp.A @ shift
    constant = float(lp.objective_offset + lp.c @ shift)
    canonical = CanonicalLP(c, A, b, name=lp.name, u=upper)
    transform = BoxTransform(
        shift=shift,
        source=source.astype(np.int64, copy=False),
        sign=sign,
        original_size=n,
        objective_constant=constant,
    )
    return canonical, transform


class GenericStructure:
    """Full-column structure used for arbitrary sparse equality matrices."""

    is_network = False

    def __init__(self, A: sp.csc_matrix):
        self.rows, self.columns = A.shape
        # Generic column elimination belongs inside the preconditioner.  The
        # exact Newton operator always contains every model column.
        self.protected = np.ones(self.columns, dtype=bool)

    def initial_mask(self, costs: np.ndarray, factor: float) -> np.ndarray:
        del costs, factor
        return self.protected.copy()

    def hub_feasible_flow(self, b: np.ndarray) -> None:
        del b
        return None


class NetworkTopology:
    """Static graph data and a protected spanning-tree column core."""

    is_network = True

    def __init__(
        self,
        rows: int,
        first: np.ndarray,
        second: np.ndarray,
        tail: np.ndarray,
        head: np.ndarray,
        degree: np.ndarray,
        parent: np.ndarray,
        depth: np.ndarray,
        tree_child: np.ndarray,
        protected: np.ndarray,
        root: int,
        components: int = 1,
    ):
        self.rows = int(rows)
        self.first = first
        self.second = second
        self.tail = tail
        self.head = head
        self.degree = degree
        self.parent = parent
        self.depth = depth
        self.tree_child = tree_child
        self.protected = protected
        self.root = int(root)
        self.components = int(components)
        self.max_depth = int(depth.max(initial=0))
        self.depth_order = np.argsort(depth, kind="stable").astype(np.int32)
        counts = np.bincount(depth, minlength=self.max_depth + 1)
        self.depth_start = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    @classmethod
    def from_incidence(cls, A: sp.csc_matrix):
        A = A.tocsc(copy=False)
        m, n = A.shape
        counts = np.diff(A.indptr)
        if counts.size and not np.all(counts == 2):
            raise ValueError("the network backend requires exactly two entries per column")
        first = A.indices[A.indptr[:-1]].astype(np.int32, copy=True)
        second = A.indices[A.indptr[:-1] + 1].astype(np.int32, copy=True)
        first_value = A.data[A.indptr[:-1]]
        second_value = A.data[A.indptr[:-1] + 1]
        if not (
            np.allclose(np.abs(first_value), 1.0)
            and np.allclose(first_value + second_value, 0.0)
        ):
            raise ValueError("network columns must contain one +1 and one -1")
        first_is_tail = first_value > 0
        tail = np.where(first_is_tail, first, second).astype(np.int32, copy=False)
        head = np.where(first_is_tail, second, first).astype(np.int32, copy=False)

        degree = np.bincount(first, minlength=m) + np.bincount(second, minlength=m)
        root = int(np.argmax(degree))
        graph = sp.csr_matrix(
            (
                np.ones(2 * n, dtype=np.int8),
                (np.concatenate((first, second)), np.concatenate((second, first))),
            ),
            shape=(m, m),
        )
        components, labels = csgraph.connected_components(
            graph, directed=False, return_labels=True
        )
        if components == 1:
            distance, parent = csgraph.shortest_path(
                graph,
                directed=False,
                indices=root,
                unweighted=True,
                return_predecessors=True,
            )
        else:
            # One high-degree representative per component is connected to a
            # temporary virtual root.  A single BFS on the augmented graph
            # produces a shallow spanning forest; virtual edges are discarded.
            maximum_degree = np.zeros(components, dtype=degree.dtype)
            np.maximum.at(maximum_degree, labels, degree)
            candidate = degree == maximum_degree[labels]
            roots = np.full(components, m, dtype=np.int64)
            candidate_index = np.flatnonzero(candidate)
            np.minimum.at(roots, labels[candidate_index], candidate_index)
            root_row = sp.csr_matrix(
                (
                    np.ones(components, dtype=np.int8),
                    (np.zeros(components, dtype=np.int32), roots),
                ),
                shape=(1, m),
            )
            augmented = sp.vstack(
                (
                    sp.hstack((graph, root_row.T), format="csr"),
                    sp.hstack((root_row, sp.csr_matrix((1, 1))), format="csr"),
                ),
                format="csr",
            )
            distance_augmented, parent_augmented = csgraph.shortest_path(
                augmented,
                directed=False,
                indices=m,
                unweighted=True,
                return_predecessors=True,
            )
            distance = distance_augmented[:m] - 1.0
            parent = parent_augmented[:m]
            parent[parent == m] = -1
            root = int(roots[0])
            del augmented, root_row, distance_augmented, parent_augmented, roots
        if not np.all(np.isfinite(distance)):
            raise ValueError("the incidence graph is disconnected")
        parent = parent.astype(np.int32, copy=False)
        parent[root] = -1
        depth = distance.astype(np.int32, copy=False)
        del graph, distance

        first_child = parent[first] == second
        second_child = parent[second] == first
        tree_child = np.full(n, -1, dtype=np.int32)
        tree_child[first_child] = first[first_child]
        tree_child[second_child] = second[second_child]

        # One protected arc per tree relation is enough to keep the RMP graph
        # connected; parallel arcs remain available to pricing and elimination.
        protected_index = np.full(m, n, dtype=np.int64)
        candidates = np.flatnonzero(tree_child >= 0)
        np.minimum.at(protected_index, tree_child[candidates], candidates)
        protected = np.zeros(n, dtype=bool)
        chosen = protected_index[protected_index < n]
        protected[chosen] = True
        if chosen.size != m - components:
            raise RuntimeError("failed to construct a spanning-forest column core")
        return cls(
            m,
            first,
            second,
            tail,
            head,
            degree,
            parent,
            depth,
            tree_child,
            protected,
            root,
            components,
        )

    def initial_mask(self, costs: np.ndarray, factor: float) -> np.ndarray:
        mask = self.protected.copy()
        target = min(mask.size, max(int(math.ceil(factor * self.rows)), int(mask.sum())))
        missing = target - int(mask.sum())
        if missing > 0:
            available = np.flatnonzero(~mask)
            if missing < available.size:
                local = np.argpartition(costs[available], missing - 1)[:missing]
                available = available[local]
            mask[available] = True
        return mask

    def hub_feasible_flow(self, b: np.ndarray) -> Optional[np.ndarray]:
        """Build a feasible flow on directed trees rooted at a network hub.

        A reverse BFS gives one route from every supply into the hub; a forward
        BFS gives one route from the hub to every demand.  Subtree accumulation
        assigns the required flow.  This recognizes the Bonn VLSI construction
        and similar transshipment networks without a Phase-I optimization.
        """
        if self.rows < 3 or not np.any(b > 0.0) or not np.any(b < 0.0):
            return None
        candidate_hubs = np.argpartition(self.degree, -min(2, self.rows))[
            -min(2, self.rows) :
        ]
        candidate_hubs = candidate_hubs[np.argsort(self.degree[candidate_hubs])[::-1]]
        supply_nodes = np.flatnonzero(b > 0.0)
        demand_nodes = np.flatnonzero(b < 0.0)
        directed = sp.csr_matrix(
            (
                np.ones(self.tail.size, dtype=np.int8),
                (self.tail, self.head),
            ),
            shape=(self.rows, self.rows),
        )
        reverse = directed.transpose().tocsr()
        edge_key = self.tail.astype(np.int64) * self.rows + self.head
        edge_order = np.argsort(edge_key, kind="stable")
        sorted_key = edge_key[edge_order]

        def edge_lookup(tail, head):
            key = tail.astype(np.int64) * self.rows + head
            position = np.searchsorted(sorted_key, key)
            valid = (position < sorted_key.size) & (
                sorted_key[np.minimum(position, sorted_key.size - 1)] == key
            )
            if not np.all(valid):
                raise RuntimeError("BFS predecessor edge is absent from the arc list")
            return edge_order[position]

        def route_tree(distance, predecessor, load, toward_root):
            finite = np.isfinite(distance)
            depth = np.zeros(self.rows, dtype=np.int32)
            depth[finite] = distance[finite].astype(np.int32)
            maximum_depth = int(depth.max(initial=0))
            order = np.argsort(depth, kind="stable")
            count = np.bincount(depth, minlength=maximum_depth + 1)
            offset = np.concatenate(([0], np.cumsum(count))).astype(np.int64)
            routed = load.copy()
            flow = np.zeros(self.tail.size, dtype=np.float64)
            for level in range(maximum_depth, 0, -1):
                nodes = order[offset[level] : offset[level + 1]]
                nodes = nodes[finite[nodes] & (routed[nodes] != 0.0)]
                if nodes.size == 0:
                    continue
                parent = predecessor[nodes]
                if toward_root:
                    arcs = edge_lookup(nodes, parent)
                else:
                    arcs = edge_lookup(parent, nodes)
                np.add.at(flow, arcs, routed[nodes])
                np.add.at(routed, parent, routed[nodes])
            return flow

        for hub in candidate_hubs:
            distance_in, predecessor_in = csgraph.shortest_path(
                reverse,
                directed=True,
                indices=int(hub),
                unweighted=True,
                return_predecessors=True,
            )
            if not np.all(np.isfinite(distance_in[supply_nodes])):
                continue
            distance_out, predecessor_out = csgraph.shortest_path(
                directed,
                directed=True,
                indices=int(hub),
                unweighted=True,
                return_predecessors=True,
            )
            if not np.all(np.isfinite(distance_out[demand_nodes])):
                continue
            supply_load = np.maximum(b, 0.0)
            demand_load = np.maximum(-b, 0.0)
            flow = route_tree(distance_in, predecessor_in, supply_load, True)
            flow += route_tree(distance_out, predecessor_out, demand_load, False)
            residual = np.bincount(
                self.tail, weights=flow, minlength=self.rows
            ) - np.bincount(self.head, weights=flow, minlength=self.rows)
            if float(np.linalg.norm(residual - b, ord=np.inf)) <= 1.0e-10:
                return flow
        return None


class TreePreconditioner:
    """Exact solve with an SPD tree-plus-diagonal approximation."""

    def __init__(
        self,
        topology: NetworkTopology,
        active_columns: np.ndarray,
        stats: Optional[PreconditionerStats] = None,
    ):
        self.topology = topology
        self.columns = active_columns
        self.stats = stats
        self.first = topology.first[active_columns]
        self.second = topology.second[active_columns]
        self.child = topology.tree_child[active_columns]
        self.pivot = None
        self.multiplier = None

    def factor(self, weights: np.ndarray, delta: float):
        started = time.perf_counter()
        m = self.topology.rows
        diagonal = delta + np.bincount(
            self.first, weights=weights, minlength=m
        ) + np.bincount(self.second, weights=weights, minlength=m)
        tree = self.child >= 0
        tree_weight = np.bincount(
            self.child[tree], weights=weights[tree], minlength=m
        )

        pivot = diagonal
        multiplier = np.zeros(m, dtype=np.float64)
        order = self.topology.depth_order
        offsets = self.topology.depth_start
        parent = self.topology.parent
        for level in range(self.topology.max_depth, 0, -1):
            nodes = order[offsets[level] : offsets[level + 1]]
            piv = np.maximum(pivot[nodes], TINY)
            edge = tree_weight[nodes]
            multiplier[nodes] = -edge / piv
            np.add.at(pivot, parent[nodes], -(edge * edge) / piv)
        np.maximum(pivot, max(delta * 1.0e-8, TINY), out=pivot)
        self.pivot = pivot
        self.multiplier = multiplier
        if self.stats is not None:
            self.stats.factorizations += 1
            self.stats.observe_core(self.columns.size)
        return time.perf_counter() - started

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        value = np.asarray(rhs, dtype=np.float64).copy()
        order = self.topology.depth_order
        offsets = self.topology.depth_start
        parent = self.topology.parent
        mult = self.multiplier
        for level in range(self.topology.max_depth, 0, -1):
            nodes = order[offsets[level] : offsets[level + 1]]
            np.add.at(value, parent[nodes], -mult[nodes] * value[nodes])
        value /= self.pivot
        for level in range(1, self.topology.max_depth + 1):
            nodes = order[offsets[level] : offsets[level + 1]]
            value[nodes] -= mult[nodes] * value[parent[nodes]]
        return value

    @property
    def backend(self) -> str:
        return "tree-laplacian"

    def record_iterations(
        self, iterations: int, elapsed: float = 0.0, converged: bool = True
    ) -> None:
        del iterations, elapsed, converged


class DiagonalPreconditioner:
    """Jacobi Schur preconditioner available without optional libraries."""

    def __init__(
        self,
        A: sp.csc_matrix,
        stats: Optional[PreconditionerStats] = None,
    ):
        self.A2 = A.copy().tocsc()
        self.A2.data = np.square(self.A2.data)
        self.diagonal = np.ones(A.shape[0], dtype=np.float64)
        self.stats = stats

    def factor(self, weights: np.ndarray, delta: float) -> float:
        started = time.perf_counter()
        self.diagonal = np.asarray(self.A2 @ weights).ravel() + float(delta)
        np.maximum(self.diagonal, max(float(delta), TINY), out=self.diagonal)
        if self.stats is not None:
            self.stats.factorizations += 1
            self.stats.observe_core(0)
        return time.perf_counter() - started

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(rhs, dtype=np.float64) / self.diagonal

    @property
    def backend(self) -> str:
        return "diagonal"

    def record_iterations(
        self, iterations: int, elapsed: float = 0.0, converged: bool = True
    ) -> None:
        del iterations, elapsed, converged


class StableColumnPreconditioner:
    r"""Stable active-column approximation of the exact generic Schur matrix.

    For ``H=A D A.T + delta I``, only a stable, fill-aware column core ``C``
    enters the sparse factorization.  Omitted columns retain their exact
    diagonal contribution.  PCG still multiplies by the complete ``H``, so
    column elimination changes convergence speed but not the Newton step.
    """

    def __init__(
        self,
        A: sp.csc_matrix,
        options: V3Options,
        stats: Optional[PreconditionerStats] = None,
    ):
        self.stats = stats or PreconditionerStats()
        self.diagonal = DiagonalPreconditioner(A, self.stats)
        self.impl = None
        self._backend = "diagonal"
        self._required = options.generic_preconditioner == "stable"
        self._retry_growth = options.generic_retry_growth
        self._growth_trigger = options.generic_refresh_cg
        self._growth_amortization = options.generic_growth_amortization
        self._last_numeric_factor_seconds = 0.0
        self._factor_epoch = 0
        self._last_growth_epoch = -1
        self._seen_core_sizes = 0
        self._seen_retained = 0
        self._seen_refreshes = 0
        self._seen_factorizations = 0
        self._seen_symbolic = 0
        if options.generic_preconditioner == "diagonal":
            return
        try:
            try:
                from .psipm_stable_preconditioner import (
                    StableActiveColumnPreconditioner,
                )
            except ImportError:
                from psipm_stable_preconditioner import (
                    StableActiveColumnPreconditioner,
                )

            self.impl = StableActiveColumnPreconditioner(
                A,
                core_factor=options.generic_core_factor,
                max_core_factor=options.generic_max_core_factor,
                retention=options.generic_retention,
                factor_max_age=options.generic_factor_max_age,
                core_max_age=options.generic_core_max_age,
                # V3.1 grows the core from measured factor/PCG time below.
                refresh_cg=options.pcg_max_iterations + 1,
                fill_weight=options.generic_fill_weight,
                cg_tol=options.pcg_tolerance_late,
                cg_tol_max=options.pcg_tolerance_early,
                cg_maxit=options.pcg_max_iterations,
                retry_growth=options.generic_retry_growth,
                order=options.generic_cholmod_order,
                refine=0,
                low_rank_max_columns=options.generic_low_rank_max_columns,
                low_rank_min_nnz=options.generic_low_rank_min_nnz,
                low_rank_clique_ratio=options.generic_low_rank_clique_ratio,
            )
            self._backend = "stable-active-column"
            self.stats.full_preconditioner = bool(
                self.impl.use_full_preconditioner
            )
            self.stats.low_rank_columns = int(
                self.impl.low_rank_columns.size
            )
        except Exception as exc:
            if self._required:
                raise RuntimeError(
                    "the stable generic preconditioner requires "
                    "scikit-sparse/CHOLMOD and its PSIPM modules"
                ) from exc
            self.impl = None
            self._backend = "diagonal-fallback"

    def factor(self, weights: np.ndarray, delta: float) -> float:
        if self.impl is None:
            return self.diagonal.factor(weights, delta)
        started = time.perf_counter()
        factorizations_before = int(self.impl.n_fact)
        try:
            theta = 1.0 / np.maximum(weights, TINY)
            self.impl.factorize(theta, float(delta))
        except Exception:
            if self._required:
                raise
            self.impl = None
            self._backend = "diagonal-fallback"
            return self.diagonal.factor(weights, delta)
        elapsed = time.perf_counter() - started
        self._factor_epoch += 1
        if self.impl.n_fact > factorizations_before:
            self._last_numeric_factor_seconds = elapsed
        self._update_stats()
        return elapsed

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if self.impl is None:
            return self.diagonal.solve(rhs)
        return np.asarray(self.impl._apply(rhs)).ravel()

    @property
    def backend(self) -> str:
        return self._backend

    def _update_stats(self) -> None:
        if self.impl is None:
            return
        for size in self.impl.core_sizes[self._seen_core_sizes :]:
            self.stats.observe_core(size)
        for retained in self.impl.retained_fractions[self._seen_retained :]:
            self.stats.retained_sum += float(retained)
            self.stats.retained_observations += 1
        self._seen_core_sizes = len(self.impl.core_sizes)
        self._seen_retained = len(self.impl.retained_fractions)
        self.stats.refreshes += self.impl.refreshes - self._seen_refreshes
        self.stats.factorizations += self.impl.n_fact - self._seen_factorizations
        self.stats.symbolic_analyses += self.impl.n_symbolic - self._seen_symbolic
        self._seen_refreshes = self.impl.refreshes
        self._seen_factorizations = self.impl.n_fact
        self._seen_symbolic = self.impl.n_symbolic
        if self.impl.preconditioner_nnz:
            self.stats.max_factor_nonzeros = max(
                self.stats.max_factor_nonzeros,
                max(self.impl.preconditioner_nnz),
            )

    def record_iterations(
        self, iterations: int, elapsed: float = 0.0, converged: bool = True
    ) -> None:
        del converged
        if self.impl is None:
            return
        self.impl._last_cg = int(iterations)
        can_grow = self.impl.core_target < self.impl.max_core_target
        projected_krylov = 4.0 * max(float(elapsed), 0.0)
        growth_affordable = (
            self._last_numeric_factor_seconds <= 0.0
            or projected_krylov
            >= self._growth_amortization * self._last_numeric_factor_seconds
        )
        if (
            iterations >= self._growth_trigger
            and growth_affordable
            and self._last_growth_epoch != self._factor_epoch
        ):
            if can_grow:
                grown = max(
                    self.impl.core_target + 1,
                    int(math.ceil(self._retry_growth * self.impl.core_target)),
                )
                self.impl.core_target = min(self.impl.max_core_target, grown)
                self.stats.growth_events += 1
            # At the core ceiling, poor Krylov convergence still justifies a
            # leverage refresh when one factorization is cheaper than the
            # projected predictor/corrector solves.
            self.impl.core_age = self.impl.core_max_age
            self._last_growth_epoch = self._factor_epoch

    def record_solve(
        self, iterations: int, elapsed: float, converged: bool
    ) -> None:
        self.record_iterations(iterations, elapsed, converged)


def _restricted_matrix(A: sp.csc_matrix, columns: np.ndarray) -> sp.csc_matrix:
    """Return the full CSC matrix without copying when every column is active."""
    if columns.size == A.shape[1] and (
        columns.size == 0 or (columns[0] == 0 and columns[-1] == A.shape[1] - 1)
    ):
        return A
    return A[:, columns].tocsc()


def _make_preconditioner(
    structure: Any,
    A: sp.csc_matrix,
    columns: np.ndarray,
    options: V3Options,
    stats: Optional[PreconditionerStats] = None,
):
    if structure.is_network:
        return TreePreconditioner(structure, columns, stats)
    return StableColumnPreconditioner(
        _restricted_matrix(A, columns), options, stats
    )


def _pcg(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    preconditioner: Any,
    rtol: float,
    max_iterations: int,
    deadline: float,
):
    """Preconditioned CG with true-residual stopping and deadline checks."""
    started = time.perf_counter()
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros_like(rhs), 0, 0, True, time.perf_counter() - started
    x = preconditioner.solve(rhs)
    residual = rhs - matvec(x)
    matvecs = 1
    target = max(rtol * rhs_norm, 64.0 * EPS * rhs_norm)
    if float(np.linalg.norm(residual)) <= target:
        return x, 0, matvecs, True, time.perf_counter() - started
    z = preconditioner.solve(residual)
    direction = z.copy()
    rz = float(residual @ z)
    if not np.isfinite(rz) or rz <= 0.0:
        return x, 0, matvecs, False, time.perf_counter() - started

    for iteration in range(1, max_iterations + 1):
        product = matvec(direction)
        matvecs += 1
        curvature = float(direction @ product)
        if not np.isfinite(curvature) or curvature <= 0.0:
            return x, iteration, matvecs, False, time.perf_counter() - started
        step = rz / curvature
        x += step * direction
        residual -= step * product
        if iteration % 10 == 0:
            residual = rhs - matvec(x)
            matvecs += 1
        if float(np.linalg.norm(residual)) <= target:
            return x, iteration, matvecs, True, time.perf_counter() - started
        if time.perf_counter() >= deadline:
            return x, iteration, matvecs, False, time.perf_counter() - started
        z = preconditioner.solve(residual)
        rz_new = float(residual @ z)
        if not np.isfinite(rz_new) or rz_new <= 0.0:
            return x, iteration, matvecs, False, time.perf_counter() - started
        direction *= rz_new / rz
        direction += z
        rz = rz_new
    return x, max_iterations, matvecs, False, time.perf_counter() - started


class HSDNewtonSystem:
    """Implicit reduced HSD Newton system for one active-column set."""

    def __init__(
        self,
        A: sp.csc_matrix,
        c: np.ndarray,
        upper: np.ndarray,
        b: np.ndarray,
        D: np.ndarray,
        delta: float,
        eta: float,
        state: HSDState,
        embedding: HSDEmbedding,
        preconditioner: Any,
        pcg_tolerance: float,
        options: V3Options,
        deadline: float,
        stats: LinearStats,
    ):
        self.A = A
        self.c = c
        self.upper = upper
        self.b = b
        self.D = D
        self.delta = delta
        self.eta = eta
        self.state = state
        self.embedding = embedding
        self.preconditioner = preconditioner
        self.pcg_tolerance = pcg_tolerance
        self.options = options
        self.deadline = deadline
        self.stats = stats
        stats.preconditioner_seconds += preconditioner.factor(D, delta)
        boxed = np.isfinite(upper)
        upper_ratio = np.zeros_like(c)
        slack = np.ones_like(c)
        slack[boxed] = upper[boxed] * state.tau - state.x[boxed]
        upper_ratio[boxed] = state.w[boxed] / slack[boxed]
        self.slack = slack
        self.upper_ratio = upper_ratio
        self.c_direction = c - upper_ratio * np.where(boxed, upper, 0.0)
        self.c_gap = c + upper_ratio * np.where(boxed, upper, 0.0)
        self.AD_c = A @ (D * self.c_direction)
        self.h = b + self.AD_c
        self.g = b - A @ (D * self.c_gap)
        self.theta_row = embedding.bar_b + A @ (D * embedding.bar_c)
        self.normal_row = A @ (D * embedding.bar_c) - embedding.bar_b
        upper_curvature = float(
            np.sum(upper_ratio[boxed] * upper[boxed] * upper[boxed])
        )
        self.alpha = (
            float(self.c_gap @ (D * self.c_direction))
            + upper_curvature
            + state.kappa / state.tau
            + eta
        )
        self.beta = embedding.bar_g - float(
            self.c_gap @ (D * embedding.bar_c)
        )
        self.gamma = -float(
            embedding.bar_c @ (D * self.c_direction)
        ) - embedding.bar_g
        self.lambda_theta = float(
            embedding.bar_c @ (D * embedding.bar_c)
        ) + options.zeta
        self._h_solution = None
        self._theta_solution = None

    def matvec(self, vector: np.ndarray):
        return self.A @ (self.D * (self.A.T @ vector)) + self.delta * vector

    def solve_rows(self, rhs: np.ndarray):
        solution, iterations, matvecs, okay, elapsed = _pcg(
            self.matvec,
            rhs,
            self.preconditioner,
            self.pcg_tolerance,
            self.options.pcg_max_iterations,
            self.deadline,
        )
        self.stats.solves += 1
        self.stats.iterations += iterations
        self.stats.max_iterations = max(self.stats.max_iterations, iterations)
        self.stats.matvecs += matvecs
        self.stats.seconds += elapsed
        if hasattr(self.preconditioner, "record_solve"):
            self.preconditioner.record_solve(iterations, elapsed, okay)
        else:
            self.preconditioner.record_iterations(iterations)
        if not okay:
            self.stats.failures += 1
        return solution, okay

    def direction(
        self,
        rhs_d,
        rhs_p,
        rhs_g,
        rhs_n,
        rhs_c,
        rhs_upper,
        rhs_t,
    ):
        state = self.state
        boxed = np.isfinite(self.upper)
        upper_rhs = np.zeros_like(rhs_d)
        upper_rhs[boxed] = rhs_upper[boxed] / self.slack[boxed]
        q = rhs_d + rhs_c / state.x - upper_rhs
        f = -rhs_p - self.A @ (self.D * q)
        u, okay_u = self.solve_rows(f)
        if self._h_solution is None:
            self._h_solution, okay_h = self.solve_rows(self.h)
        else:
            okay_h = True
        if self._theta_solution is None:
            self._theta_solution, okay_theta = self.solve_rows(self.theta_row)
        else:
            okay_theta = True
        gap_rhs = (
            rhs_g
            + float(self.c_gap @ (self.D * q))
            + float(
                self.upper[boxed]
                @ (rhs_upper[boxed] / self.slack[boxed])
            )
            + rhs_t / state.tau
            - float(self.g @ u)
        )
        normal_rhs = (
            rhs_n
            - float(self.embedding.bar_c @ (self.D * q))
            - float(self.normal_row @ u)
        )
        border = np.array(
            [
                [
                    self.alpha + float(self.g @ self._h_solution),
                    self.beta - float(self.g @ self._theta_solution),
                ],
                [
                    self.gamma + float(self.normal_row @ self._h_solution),
                    self.lambda_theta
                    - float(self.normal_row @ self._theta_solution),
                ],
            ],
            dtype=np.float64,
        )
        border_rhs = np.array([gap_rhs, normal_rhs], dtype=np.float64)
        determinant = float(np.linalg.det(border))
        border_scale = max(float(np.linalg.norm(border, ord=np.inf)), 1.0)
        if (
            not np.all(np.isfinite(border))
            or not np.all(np.isfinite(border_rhs))
            or abs(determinant) <= 64.0 * EPS * border_scale * border_scale
        ):
            okay_u = False
            border = border + 64.0 * EPS * border_scale * np.eye(2)
        try:
            dtau, dtheta = np.linalg.solve(border, border_rhs)
        except np.linalg.LinAlgError:
            dtau, dtheta = np.linalg.lstsq(border, border_rhs, rcond=None)[0]
            okay_u = False
        dy = u + self._h_solution * dtau - self._theta_solution * dtheta
        dx = self.D * (
            q
            + self.A.T @ dy
            - self.c_direction * dtau
            + self.embedding.bar_c * dtheta
        )
        dz = (rhs_c - state.z * dx) / state.x
        dw = np.zeros_like(dx)
        ds = np.zeros_like(dx)
        ds[boxed] = self.upper[boxed] * dtau - dx[boxed]
        dw[boxed] = (
            rhs_upper[boxed] - state.w[boxed] * ds[boxed]
        ) / self.slack[boxed]
        dkappa = (rhs_t - state.kappa * dtau) / state.tau
        finite = all(
            np.all(np.isfinite(value))
            for value in (
                dx,
                dy,
                dz,
                dw,
                np.asarray([dtau, dtheta, dkappa]),
            )
        )
        return (
            dx,
            dy,
            dz,
            dw,
            float(dtau),
            float(dtheta),
            float(dkappa),
            bool(okay_u and okay_h and okay_theta and finite),
        )


def _maximum_step(values: np.ndarray, direction: np.ndarray) -> float:
    negative = direction < 0.0
    if not np.any(negative):
        return 1.0
    return min(1.0, float(np.min(-values[negative] / direction[negative])))


def _scalar_step(value: float, direction: float) -> float:
    if direction >= 0.0:
        return 1.0
    return min(1.0, -value / direction)


def _hsd_residuals(
    A,
    c,
    b,
    upper,
    state: HSDState,
    embedding: HSDEmbedding,
    rho,
    delta,
    eta,
    zeta,
):
    boxed = np.isfinite(upper)
    rd_original = (
        c * state.tau
        - A.T @ state.y
        - embedding.bar_c * state.theta
        - state.z
        + state.w
    )
    rp_original = (
        b * state.tau - A @ state.x - embedding.bar_b * state.theta
    )
    upper_dual = float(upper[boxed] @ state.w[boxed])
    rg_original = (
        -float(c @ state.x)
        + float(b @ state.y)
        - upper_dual
        + embedding.bar_g * state.theta
        - state.kappa
    )
    rn_original = (
        -float(embedding.bar_b @ state.y)
        + float(embedding.bar_c @ state.x)
        - embedding.bar_g * state.tau
        + embedding.alpha
    )
    rd = rd_original + rho * (state.x - state.x_center)
    rp = rp_original - delta * (state.y - state.y_center)
    rg = rg_original + eta * (state.tau - state.tau_center)
    rn = rn_original + zeta * (state.theta - state.theta_center)
    return (
        rd,
        rp,
        rg,
        rn,
        rd_original,
        rp_original,
        rg_original,
        rn_original,
    )


def _residual_measure(rd, rp, rg, rn, c, b, state):
    dual_scale = 1.0 + float(np.linalg.norm(c, ord=np.inf)) * max(state.tau, 1.0)
    primal_scale = 1.0 + float(np.linalg.norm(b, ord=np.inf)) * max(state.tau, 1.0)
    gap_scale = 1.0 + abs(float(c @ state.x)) + abs(float(b @ state.y)) + state.kappa
    return max(
        float(np.linalg.norm(rd, ord=np.inf)) / dual_scale,
        float(np.linalg.norm(rp, ord=np.inf)) / primal_scale,
        abs(float(rg)) / gap_scale,
        abs(float(rn)) / (1.0 + abs(float(state.embedding.alpha))),
    )


def _project_hsd_primal_cone(values, tau_value, upper):
    """Project ``(values, tau_value)`` onto the HSD primal cone.

    The cone is ``tau >= 0``, ``x >= 0``, and ``x_j <= u_j tau`` for
    finite upper bounds.  For a fixed tau the x projection is componentwise;
    the remaining scalar convex equation is solved by safeguarded semismooth
    Newton iterations.  Uncapacitated networks, including Josef and Erhard,
    take the inexpensive orthant-only branch.
    """
    values = np.asarray(values, dtype=np.float64)
    projected = np.maximum(values, 0.0)
    boxed = np.isfinite(upper)
    if not np.any(boxed):
        return projected, max(float(tau_value), 0.0)

    boxed_values = values[boxed]
    boxed_upper = upper[boxed]
    positive = boxed_values > 0.0
    if not np.any(positive):
        tau = max(float(tau_value), 0.0)
        projected[boxed] = 0.0
        return projected, tau

    a = boxed_values[positive]
    u = boxed_upper[positive]
    derivative_at_zero = -float(tau_value) - float(u @ a)
    if derivative_at_zero >= 0.0:
        tau = 0.0
    else:
        maximum_breakpoint = float(np.max(a / u, initial=0.0))
        lower = 0.0
        upper_tau = max(maximum_breakpoint, float(tau_value), 0.0)
        tau = min(max(float(tau_value), lower), upper_tau)
        tolerance = 64.0 * EPS * max(1.0, upper_tau)
        for _ in range(60):
            active = a > u * tau
            active_u = u[active]
            numerator = float(tau_value) + float(active_u @ a[active])
            denominator = 1.0 + float(active_u @ active_u)
            derivative = denominator * tau - numerator
            if abs(derivative) <= tolerance:
                break
            if derivative < 0.0:
                lower = tau
            else:
                upper_tau = tau
            candidate = max(0.0, numerator / denominator)
            if not (lower < candidate < upper_tau):
                candidate = 0.5 * (lower + upper_tau)
            if abs(candidate - tau) <= tolerance:
                tau = candidate
                break
            tau = candidate

    projected[boxed] = np.clip(boxed_values, 0.0, boxed_upper * tau)
    return projected, tau


def _natural_proximal_criterion(
    A,
    c,
    b,
    upper,
    state: HSDState,
    options: V3Options,
    outer_iteration: int,
):
    """Apply the geometrically tightened natural-residual PS-IPM test.

    This is the HSD counterpart of ``natural_prox_res.m`` and ``prox_eval.m``
    in the original PS-IPM implementation.  If ``q=(x,y,tau)`` and
    ``C`` is the HSD primal cone, it evaluates

        R_k(q) = q - Pi_C(q - (Qq + M(q-q_k))).

    The inner solve is accepted when

        ||R_x|| + ||R_y|| + |R_tau|
          < factor * reduction^k * min(1, ||dx|| + ||dy|| + |dtau|).
    """
    embedding = state.embedding
    gx = c * state.tau - A.T @ state.y - embedding.bar_c * state.theta
    gx += options.rho * (state.x - state.x_center)
    gy = A @ state.x - b * state.tau + embedding.bar_b * state.theta
    gy += options.delta * (state.y - state.y_center)
    gtau = (
        -float(c @ state.x)
        + float(b @ state.y)
        + embedding.bar_g * state.theta
    )
    gtau += options.eta * (state.tau - state.tau_center)
    gtheta = (
        -float(embedding.bar_b @ state.y)
        + float(embedding.bar_c @ state.x)
        - embedding.bar_g * state.tau
        + embedding.alpha
        + options.zeta * (state.theta - state.theta_center)
    )

    projected_x, projected_tau = _project_hsd_primal_cone(
        state.x - gx,
        state.tau - gtau,
        upper,
    )
    rx = state.x - projected_x
    rtau = state.tau - projected_tau
    natural_residual = float(np.linalg.norm(rx))
    natural_residual += (
        float(np.linalg.norm(gy)) + abs(float(rtau)) + abs(float(gtheta))
    )

    displacement = float(np.linalg.norm(state.x - state.x_center))
    displacement += float(np.linalg.norm(state.y - state.y_center))
    displacement += abs(float(state.tau - state.tau_center))
    displacement += abs(float(state.theta - state.theta_center))
    forcing = options.proximal_residual_factor * (
        options.proximal_reduction ** max(int(outer_iteration), 1)
    )
    target = forcing * min(1.0, displacement)
    if target > 0.0:
        ratio = natural_residual / target
    else:
        ratio = 0.0 if natural_residual == 0.0 else math.inf
    accepted = bool(
        np.isfinite(ratio)
        and state.center_age >= options.proximal_steps
        and natural_residual < target
    )
    return accepted, ratio, natural_residual, target, displacement


def _inexact_proximal_criterion(
    rd,
    rp,
    rg,
    rn,
    upper,
    state: HSDState,
    options: V3Options,
):
    """Evaluate the metric HPE ratio as an optional diagnostic.

    Let ``q=(x,y,tau,theta)``, ``qk`` be the current proximal centre, and
    ``M=diag(rho I, delta I, eta, zeta)``. The positive HSD multipliers define
    ``v in T^[epsilon](q)`` with

        epsilon = x'z + (u tau-x)'w + tau kappa.

    The corresponding HPE condition would be

        ||M^-1/2 (v + M(q-qk))||^2 + 2 epsilon
            <= sigma^2 ||M^1/2(q-qk)||^2.

    ``rp`` has the opposite sign to the y block of ``v+M(q-qk)``; this does
    not affect its squared metric norm.
    """
    boxed = np.isfinite(upper)
    slack = upper[boxed] * state.tau - state.x[boxed]
    enlargement_error = float(state.x @ state.z)
    enlargement_error += float(slack @ state.w[boxed])
    enlargement_error += state.tau * state.kappa
    enlargement_error = max(enlargement_error, 0.0)

    residual_squared = float(rd @ rd) / options.rho
    residual_squared += float(rp @ rp) / options.delta
    residual_squared += float(rg * rg) / options.eta
    residual_squared += float(rn * rn) / options.zeta

    dx = state.x - state.x_center
    dy = state.y - state.y_center
    dtau = state.tau - state.tau_center
    dtheta = state.theta - state.theta_center
    displacement_squared = options.rho * float(dx @ dx)
    displacement_squared += options.delta * float(dy @ dy)
    displacement_squared += options.eta * float(dtau * dtau)
    displacement_squared += options.zeta * float(dtheta * dtheta)

    lhs_squared = max(residual_squared + 2.0 * enlargement_error, 0.0)
    lhs = math.sqrt(lhs_squared)
    displacement = math.sqrt(max(displacement_squared, 0.0))
    if displacement > 0.0:
        ratio = lhs / displacement
    else:
        ratio = 0.0 if lhs == 0.0 else math.inf
    finite = all(
        np.isfinite(value)
        for value in (lhs, displacement, ratio, enlargement_error)
    )
    accepted = bool(
        finite
        and state.center_age >= options.proximal_steps
        and ratio <= options.proximal_relative_tolerance
    )
    return (
        accepted,
        ratio,
        math.sqrt(max(residual_squared, 0.0)),
        enlargement_error,
        displacement,
    )


def _initialize_state(A, c, b, upper, primal_start=None):
    n = c.size
    m = b.size
    boxed = np.isfinite(upper)
    # Use a scale-balanced point in every finite box at tau=1.  The expression
    # u/(1+sqrt(u)) is the midpoint for u=1 and grows like sqrt(u) for a large
    # box, avoiding both an enormous lower multiplier and an enormous initial
    # artificial flow.  For one-sided variables use the conventional unit HSD
    # start, enlarged by a supplied feasible-flow magnitude when available.
    # Inverse-row-degree scaling makes z=1/x enormous on hub matrices and can
    # force a needlessly tiny HSD recovery scale even for a well-posed LP.
    x = np.ones(n, dtype=np.float64)
    root_upper = np.sqrt(upper[boxed])
    x[boxed] = upper[boxed] / (1.0 + root_upper)
    if primal_start is not None:
        primal_start = np.asarray(primal_start, dtype=np.float64)
        if primal_start.shape == x.shape:
            unboxed = ~boxed
            x[unboxed] = np.maximum(x[unboxed], primal_start[unboxed])
    y = np.zeros(m, dtype=np.float64)
    # The extended HSD residual vectors make this point exactly feasible even
    # when Ax != b.  Unit complementarity gives the standard normalization
    # alpha = n + n_box + 1 and avoids a numerically tiny recovery scale.
    mu_start = 1.0
    z = mu_start / x
    slack = np.ones(n, dtype=np.float64)
    slack[boxed] = upper[boxed] - x[boxed]
    if np.any(slack[boxed] <= 0.0):
        raise ValueError("strict HSD start violates a finite upper bound")
    w = np.zeros(n, dtype=np.float64)
    w[boxed] = mu_start / slack[boxed]
    kappa = mu_start
    tau = 1.0
    theta = 1.0
    bar_b = np.asarray(b * tau - A @ x).ravel()
    bar_c = np.asarray(c * tau - A.T @ y - z + w).ravel()
    bar_g = (
        float(c @ x)
        - float(b @ y)
        + float(upper[boxed] @ w[boxed])
        + kappa
    )
    alpha = float(x @ z) + float(slack[boxed] @ w[boxed]) + tau * kappa
    embedding = HSDEmbedding(bar_b, bar_c, bar_g, alpha)
    return HSDState(
        x=x,
        y=y,
        z=z,
        w=w,
        tau=tau,
        kappa=kappa,
        theta=theta,
        x_center=x.copy(),
        y_center=y.copy(),
        tau_center=tau,
        theta_center=theta,
        embedding=embedding,
    )


def _resize_state(state: HSDState, old_columns, new_columns, upper, mu):
    size = new_columns.size
    new_x = max(float(mu), 1.0e-16)
    new_z = max(float(mu) / new_x, TINY)
    x = np.full(size, new_x, dtype=np.float64)
    z = np.full(size, new_z, dtype=np.float64)
    w = np.zeros(size, dtype=np.float64)
    boxed = np.isfinite(upper[new_columns])
    local_upper = upper[new_columns]
    x[boxed] = np.minimum(x[boxed], 0.5 * local_upper[boxed] * state.tau)
    z[boxed] = mu / np.maximum(x[boxed], TINY)
    slack = local_upper[boxed] * state.tau - x[boxed]
    w[boxed] = mu / np.maximum(slack, TINY)
    # Both index vectors come from flatnonzero and are sorted.  searchsorted
    # keeps pool changes O(n) in compact NumPy storage even at 10M columns.
    old_position = np.searchsorted(old_columns, new_columns)
    clipped = np.minimum(old_position, max(old_columns.size - 1, 0))
    common = (old_position < old_columns.size) & (old_columns[clipped] == new_columns)
    new_position = np.flatnonzero(common)
    source = old_position[common]
    x[new_position] = state.x[source]
    z[new_position] = state.z[source]
    w[new_position] = state.w[source]
    # A pool change defines a new restricted operator.  Start that proximal
    # sequence at the transferred strictly feasible point; a centre from the
    # previous restricted operator must not be reused for a different one.
    x_center = x.copy()
    return HSDState(
        x=x,
        y=state.y,
        z=z,
        w=w,
        tau=state.tau,
        kappa=state.kappa,
        theta=state.theta,
        x_center=x_center,
        y_center=state.y.copy(),
        tau_center=state.tau,
        theta_center=state.theta,
        center_age=0,
        center_residual=math.inf,
        embedding=None,
    )


def _run_hsd_steps(
    lp: CanonicalLP,
    structure: Any,
    columns: np.ndarray,
    state: Optional[HSDState],
    preconditioner: Any,
    options: V3Options,
    deadline: float,
    iteration_start: int,
    iteration_limit: int,
    linear_stats: LinearStats,
    proximal_stats: ProximalStats,
    primal_start=None,
):
    A = _restricted_matrix(lp.A, columns)
    c = lp.c[columns]
    upper = lp.u[columns]
    if state is None:
        state = _initialize_state(A, c, lp.b, upper, primal_start)
    message = ""

    for local_iteration in range(iteration_limit):
        iteration = iteration_start + local_iteration
        if time.perf_counter() >= deadline:
            return state, local_iteration, "time-limit", math.inf, False
        rd, rp, rg, rn, rd0, rp0, rg0, rn0 = _hsd_residuals(
            A,
            c,
            lp.b,
            upper,
            state,
            state.embedding,
            options.rho,
            options.delta,
            options.eta,
            options.zeta,
        )
        proximal_residual = _residual_measure(rd, rp, rg, rn, c, lp.b, state)
        residual = _residual_measure(rd0, rp0, rg0, rn0, c, lp.b, state)
        if not np.isfinite(state.center_residual):
            state.center_residual = proximal_residual
        boxed = np.isfinite(upper)
        slack = np.ones_like(state.x)
        slack[boxed] = upper[boxed] * state.tau - state.x[boxed]
        complementarity_sum = float(state.x @ state.z)
        complementarity_sum += float(slack[boxed] @ state.w[boxed])
        complementarity_sum += state.tau * state.kappa
        complementarity_count = c.size + int(np.count_nonzero(boxed)) + 1.0
        mu = complementarity_sum / complementarity_count
        recovered_mu = mu / max(state.tau * state.tau, TINY)
        if options.print_level >= 2:
            print(
                f"  hsd {iteration:3d} |W| {columns.size:9d} "
                f"tau {state.tau:.2e} kappa {state.kappa:.2e} "
                f"mu/tau2 {recovered_mu:.2e} residual {residual:.2e}",
                flush=True,
            )

        ratio = max(recovered_mu, residual)
        if ratio > 1.0e-2:
            pcg_tolerance = options.pcg_tolerance_early
        else:
            interpolation = min(1.0, max(0.0, -math.log10(max(ratio, 1.0e-12)) / 8.0))
            pcg_tolerance = options.pcg_tolerance_early ** (1.0 - interpolation)
            pcg_tolerance *= options.pcg_tolerance_late**interpolation

        upper_ratio = np.zeros_like(state.x)
        upper_ratio[boxed] = state.w[boxed] / slack[boxed]
        theta = options.rho + state.z / state.x + upper_ratio
        D = 1.0 / np.maximum(theta, options.regularization_floor)
        system = HSDNewtonSystem(
            A,
            c,
            upper,
            lp.b,
            D,
            options.delta,
            options.eta,
            state,
            state.embedding,
            preconditioner,
            pcg_tolerance,
            options,
            deadline,
            linear_stats,
        )

        affine = system.direction(
            -rd,
            -rp,
            -rg,
            -rn,
            -state.x * state.z,
            np.where(boxed, -slack * state.w, 0.0),
            -state.tau * state.kappa,
        )
        (
            dx_aff,
            dy_aff,
            dz_aff,
            dw_aff,
            dtau_aff,
            dtheta_aff,
            dkappa_aff,
            affine_ok,
        ) = affine
        ds_aff = np.zeros_like(dx_aff)
        ds_aff[boxed] = upper[boxed] * dtau_aff - dx_aff[boxed]
        alpha_aff = min(
            _maximum_step(state.x, dx_aff),
            _scalar_step(state.tau, dtau_aff),
            _maximum_step(state.z, dz_aff),
            _maximum_step(slack[boxed], ds_aff[boxed]),
            _maximum_step(state.w[boxed], dw_aff[boxed]),
            _scalar_step(state.kappa, dkappa_aff),
        )
        mu_aff = (
            float((state.x + alpha_aff * dx_aff) @ (state.z + alpha_aff * dz_aff))
            + float(
                (slack[boxed] + alpha_aff * ds_aff[boxed])
                @ (state.w[boxed] + alpha_aff * dw_aff[boxed])
            )
            + (state.tau + alpha_aff * dtau_aff)
            * (state.kappa + alpha_aff * dkappa_aff)
        ) / complementarity_count
        sigma = float(np.clip((max(mu_aff, 0.0) / max(mu, TINY)) ** 3, 1.0e-6, 0.95))

        corrector = system.direction(
            np.zeros_like(rd),
            np.zeros_like(rp),
            0.0,
            0.0,
            sigma * mu - dx_aff * dz_aff,
            np.where(boxed, sigma * mu - ds_aff * dw_aff, 0.0),
            sigma * mu - dtau_aff * dkappa_aff,
        )
        (
            dx_cor,
            dy_cor,
            dz_cor,
            dw_cor,
            dtau_cor,
            dtheta_cor,
            dkappa_cor,
            corrector_ok,
        ) = corrector
        best = None
        for beta in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0):
            dx_trial = dx_aff + beta * dx_cor
            dy_trial = dy_aff + beta * dy_cor
            dz_trial = dz_aff + beta * dz_cor
            dw_trial = dw_aff + beta * dw_cor
            dtau_trial = dtau_aff + beta * dtau_cor
            dtheta_trial = dtheta_aff + beta * dtheta_cor
            dkappa_trial = dkappa_aff + beta * dkappa_cor
            ds_trial = np.zeros_like(dx_trial)
            ds_trial[boxed] = upper[boxed] * dtau_trial - dx_trial[boxed]
            # A single HSD step preserves the joint contraction of primal,
            # dual, gap, and complementarity residuals.  The beta safeguard
            # prevents a corrector outlier from imposing a microscopic step.
            alpha_trial = options.fraction_to_boundary * min(
                _maximum_step(state.x, dx_trial),
                _scalar_step(state.tau, dtau_trial),
                _maximum_step(state.z, dz_trial),
                _maximum_step(slack[boxed], ds_trial[boxed]),
                _maximum_step(state.w[boxed], dw_trial[boxed]),
                _scalar_step(state.kappa, dkappa_trial),
            )
            if alpha_trial <= 0.0:
                continue
            trial_mu = (
                float(
                    (state.x + alpha_trial * dx_trial)
                    @ (state.z + alpha_trial * dz_trial)
                )
                + float(
                    (slack[boxed] + alpha_trial * ds_trial[boxed])
                    @ (state.w[boxed] + alpha_trial * dw_trial[boxed])
                )
                + (state.tau + alpha_trial * dtau_trial)
                * (state.kappa + alpha_trial * dkappa_trial)
            ) / complementarity_count
            if not np.isfinite(trial_mu) or trial_mu < 0.0:
                continue
            merit = trial_mu / max(alpha_trial, 1.0e-8)
            if best is None or merit < best[0]:
                best = (
                    merit,
                    dx_trial,
                    dy_trial,
                    dz_trial,
                    dw_trial,
                    dtau_trial,
                    dtheta_trial,
                    dkappa_trial,
                    alpha_trial,
                )
        if best is None:
            return state, local_iteration, "stalled", residual, False
        _, dx, dy, dz, dw, dtau, dtheta, dkappa, alpha = best
        if not affine_ok or not corrector_ok:
            alpha *= 0.5
            message = "one or more PCG solves stopped before their requested tolerance"
        if alpha < 1.0e-12:
            return state, local_iteration, "stalled", residual, False

        state.x += alpha * dx
        state.y += alpha * dy
        state.z += alpha * dz
        state.w += alpha * dw
        state.tau += alpha * dtau
        state.theta += alpha * dtheta
        state.kappa += alpha * dkappa
        np.maximum(state.x, TINY, out=state.x)
        np.maximum(state.z, TINY, out=state.z)
        state.w[boxed] = np.maximum(state.w[boxed], TINY)
        state.tau = max(state.tau, TINY)
        state.kappa = max(state.kappa, TINY)
        if np.any(boxed):
            upper_position = upper[boxed] * state.tau
            slack_floor = 64.0 * EPS * np.maximum(1.0, np.abs(upper_position))
            state.x[boxed] = np.minimum(
                state.x[boxed], upper_position - slack_floor
            )
            state.x[boxed] = np.maximum(state.x[boxed], TINY)
        state.center_age += 1

        rd, rp, rg, rn, rd0, rp0, rg0, rn0 = _hsd_residuals(
            A,
            c,
            lp.b,
            upper,
            state,
            state.embedding,
            options.rho,
            options.delta,
            options.eta,
            options.zeta,
        )
        _, hpe_ratio, _, epsilon, _ = (
            _inexact_proximal_criterion(rd, rp, rg, rn, upper, state, options)
        )
        accepted, ratio, natural_residual, target, displacement = (
            _natural_proximal_criterion(
                A,
                c,
                lp.b,
                upper,
                state,
                options,
                proximal_stats.accepted + 1,
            )
        )
        proximal_stats.checks += 1
        proximal_stats.last_ratio = ratio
        proximal_stats.last_residual_norm = natural_residual
        proximal_stats.last_enlargement_error = epsilon
        proximal_stats.last_displacement_norm = displacement
        proximal_stats.last_target = target
        proximal_stats.last_hpe_ratio = hpe_ratio
        if accepted:
            proximal_stats.accepted += 1
            proximal_stats.max_accepted_ratio = max(
                proximal_stats.max_accepted_ratio, ratio
            )
            if options.print_level >= 2:
                print(
                    f"    proximal accept {proximal_stats.accepted}: "
                    f"natural ratio {ratio:.2e} < 1, "
                    f"residual {natural_residual:.2e}, target {target:.2e}",
                    flush=True,
                )
            state.x_center[:] = state.x
            state.y_center[:] = state.y
            state.tau_center = state.tau
            state.theta_center = state.theta
            state.center_age = 0
            state.center_residual = _residual_measure(
                rd0, rp0, rg0, rn0, c, lp.b, state
            )
            return state, local_iteration + 1, message, state.center_residual, True

    return state, iteration_limit, message, residual, False


def _full_metrics(lp: CanonicalLP, columns, state: HSDState, reduced=None):
    tau = max(state.tau, TINY)
    x_active = state.x / tau
    y = state.y / tau
    z_active = state.z / tau
    w_active = state.w / tau
    primal_residual = _restricted_matrix(lp.A, columns) @ x_active - lp.b
    if reduced is None:
        reduced = lp.c - lp.A.T @ y
    active_stationarity = reduced[columns] - z_active + w_active
    upper_active = lp.u[columns]
    boxed_active = np.isfinite(upper_active)
    upper_violation = float(
        np.max(
            np.maximum(x_active[boxed_active] - upper_active[boxed_active], 0.0),
            initial=0.0,
        )
    )
    pinf = max(float(np.linalg.norm(primal_residual, ord=np.inf)), upper_violation) / (
        1.0 + float(np.linalg.norm(lp.b, ord=np.inf))
    )
    inactive = np.ones(lp.c.size, dtype=bool)
    inactive[columns] = False
    inactive_violation = float(
        np.max(np.maximum(-reduced[inactive], 0.0), initial=0.0)
    )
    dinf = max(
        float(np.linalg.norm(active_stationarity, ord=np.inf)),
        inactive_violation,
    ) / (1.0 + float(np.linalg.norm(lp.c, ord=np.inf)))
    primal_objective = float(lp.c[columns] @ x_active)
    dual_objective = float(lp.b @ y) - float(
        upper_active[boxed_active] @ w_active[boxed_active]
    )
    gap = abs(primal_objective - dual_objective) / (
        1.0 + abs(primal_objective) + abs(dual_objective)
    )
    return pinf, dinf, gap, primal_objective, reduced


def _classify_certificate(lp: CanonicalLP, columns, state: HSDState, tolerance):
    if np.any(np.isfinite(lp.u)):
        return None, None, "bounded-variable HSD certificate classification is disabled"
    if state.kappa <= 1.0e-8 or state.tau > 1.0e-7 * state.kappa:
        return None, None, ""
    y = state.y
    dual_ray_slack = -(lp.A.T @ y)
    bty = float(lp.b @ y)
    primal_ray_error = float(np.max(np.maximum(-dual_ray_slack, 0.0), initial=0.0))
    if bty > tolerance and primal_ray_error <= tolerance * (1.0 + abs(bty)):
        return "primal-infeasible", y / bty, "validated HSD/Farkas dual ray"
    x = np.zeros(lp.c.size, dtype=np.float64)
    x[columns] = state.x
    ctx = float(lp.c @ x)
    dual_ray_error = float(np.linalg.norm(lp.A @ x, ord=np.inf))
    if ctx < -tolerance and dual_ray_error <= tolerance * (1.0 + abs(ctx)):
        return "dual-infeasible", x / (-ctx), "validated HSD primal recession ray"
    return None, None, "HSD approached tau=0 but no full-model certificate passed"


def solve(lp: CanonicalLP, options: Optional[V3Options] = None) -> V3Result:
    """Solve a nonnegative equality-and-upper-bound LP with PSIPM-V3."""
    options = options or V3Options()
    total_started = time.perf_counter()
    deadline = total_started + options.time_limit
    lp = scale_objective(lp, options.objective_scaling)
    setup_started = time.perf_counter()
    feasible_flow = None
    try:
        network = NetworkTopology.from_incidence(lp.A)
        feasible_flow = network.hub_feasible_flow(lp.b)
        zero_balance = (
            float(np.abs(lp.b).max(initial=0.0)) == 0.0
            and bool(np.all(np.isfinite(lp.u)))
        )
        if feasible_flow is None and zero_balance:
            feasible_flow = np.zeros(lp.c.size, dtype=np.float64)
        if feasible_flow is None or zero_balance:
            structure = GenericStructure(lp.A)
            structure_name = "generic-sparse"
        else:
            structure = network
            structure_name = "network-incidence"
    except ValueError:
        structure = GenericStructure(lp.A)
        structure_name = "generic-sparse"
    active = structure.initial_mask(lp.c, options.pool_factor)
    if feasible_flow is not None and np.any(feasible_flow > lp.u):
        structure = GenericStructure(lp.A)
        structure_name = "generic-sparse"
        active = structure.initial_mask(lp.c, options.pool_factor)
        feasible_flow = None
    if feasible_flow is not None:
        feasible_support = feasible_flow > 0.0
        active |= feasible_support
        structure.protected |= feasible_support
    columns = np.flatnonzero(active)
    preconditioner_stats = PreconditionerStats()
    preconditioner = _make_preconditioner(
        structure, lp.A, columns, options, preconditioner_stats
    )
    drop_age = np.zeros(lp.c.size, dtype=np.uint8)
    setup_seconds = time.perf_counter() - setup_started
    if options.print_level:
        print(
            f"PSIPM-V3.1 HSD | {lp.name or '<memory>'} | m {lp.A.shape[0]} "
            f"n {lp.A.shape[1]} nnz {lp.A.nnz}",
            flush=True,
        )
        print(
            f"  fixed metrics rho {options.rho:.1e} delta {options.delta:.1e} "
            f"eta {options.eta:.1e} zeta {options.zeta:.1e}; natural forcing "
            f"{options.proximal_residual_factor:.1e}*"
            f"{options.proximal_reduction:.2f}^k; "
            f"{structure_name}; {preconditioner.backend}; "
            f"active core {columns.size}"
            f"{' (hub-feasible)' if feasible_flow is not None else ''}",
            flush=True,
        )

    state = None
    iterations = 0
    pricing_rounds = 0
    columns_added = 0
    columns_dropped = 0
    hsd_restarts = 0
    pricing_seconds = 0.0
    linear_stats = LinearStats()
    proximal_stats = ProximalStats()
    message = ""
    status = "iteration-limit"
    final_metrics = (math.inf, math.inf, math.inf, math.nan, None)
    certificate = None

    while iterations < options.max_iterations:
        if time.perf_counter() >= deadline:
            status = "time-limit"
            break
        steps = min(options.rmp_steps, options.max_iterations - iterations)
        state, completed, step_message, _, proximal_accepted = _run_hsd_steps(
            lp,
            structure,
            columns,
            state,
            preconditioner,
            options,
            deadline,
            iterations,
            steps,
            linear_stats,
            proximal_stats,
            None if feasible_flow is None else feasible_flow[columns],
        )
        iterations += completed
        if step_message:
            message = step_message
        if not proximal_accepted and completed < steps:
            status = "time-limit" if time.perf_counter() >= deadline else "stalled"
            break
        if not proximal_accepted:
            # Keep solving this fixed restricted proximal subproblem.  Pricing
            # and pool changes are allowed only after the natural-residual test.
            continue

        pricing_started = time.perf_counter()
        tau_scale = max(state.tau, 1.0e-12)
        hsd_violation = lp.A.T @ state.y - lp.c * state.tau
        reduced = -hsd_violation / tau_scale
        final_metrics = _full_metrics(lp, columns, state, reduced)
        pinf, dinf, gap, objective, _ = final_metrics
        active_upper = lp.u[columns]
        active_boxed = np.isfinite(active_upper)
        active_slack = active_upper[active_boxed] * state.tau - state.x[active_boxed]
        active_complementarity = float(state.x @ state.z)
        active_complementarity += float(
            active_slack @ state.w[active_boxed]
        )
        active_complementarity += state.tau * state.kappa
        active_complementarity_count = (
            columns.size + int(np.count_nonzero(active_boxed)) + 1.0
        )
        recovered_mu = (
            active_complementarity
            / active_complementarity_count
            / max(state.tau * state.tau, TINY)
        )
        pricing_rounds += 1

        if options.print_level:
            print(
                f"  pool {pricing_rounds:3d} |W| {columns.size:9d} "
                f"pinf {pinf:.2e} dinf {dinf:.2e} gap {gap:.2e} "
                f"mu {recovered_mu:.2e} tau {state.tau:.2e}",
                flush=True,
            )
        if max(pinf, dinf, gap, recovered_mu) <= options.tolerance:
            status = "optimal"
            message = ""
            pricing_seconds += time.perf_counter() - pricing_started
            break

        certificate_status, certificate, certificate_message = _classify_certificate(
            lp, columns, state, options.tolerance
        )
        if certificate_status:
            status = certificate_status
            message = certificate_message
            pricing_seconds += time.perf_counter() - pricing_started
            break
        if certificate_message:
            message = certificate_message

        inactive = ~active
        price_threshold = options.pricing_tolerance * tau_scale
        candidate = np.flatnonzero(inactive & (hsd_violation > price_threshold))
        if candidate.size > options.pricing_batch:
            keep = np.argpartition(hsd_violation[candidate], -options.pricing_batch)[
                -options.pricing_batch :
            ]
            candidate = candidate[keep]

        drop = np.empty(0, dtype=np.int64)
        if options.master_column_elimination and max(pinf, dinf, gap) <= options.drop_start:
            active_reduced = reduced[columns]
            eligible = (
                (state.x / tau_scale <= options.drop_tolerance)
                & (active_reduced >= options.drop_hysteresis * options.pricing_tolerance)
                & ~structure.protected[columns]
            )
            active_eligible = columns[eligible]
            drop_age[columns[~eligible]] = 0
            if active_eligible.size:
                drop_age[active_eligible] = np.minimum(
                    drop_age[active_eligible].astype(np.uint16) + 1, 255
                ).astype(np.uint8)
                drop = active_eligible[drop_age[active_eligible] >= options.drop_patience]
                if drop.size > options.drop_batch:
                    order = np.argpartition(reduced[drop], -options.drop_batch)[
                        -options.drop_batch :
                    ]
                    drop = drop[order]
        else:
            drop_age[columns] = 0

        if pricing_rounds >= options.max_pricing_rounds:
            status = "pricing-round-limit"
            pricing_seconds += time.perf_counter() - pricing_started
            break
        if candidate.size == 0 and drop.size == 0:
            # The restricted master can need several Newton batches after the
            # last pricing event.  Absence of a pool change is not stagnation.
            pricing_seconds += time.perf_counter() - pricing_started
            continue

        old_columns = columns
        active[drop] = False
        active[candidate] = True
        active[structure.protected] = True
        columns = np.flatnonzero(active)
        preconditioner = _make_preconditioner(
            structure, lp.A, columns, options, preconditioner_stats
        )
        old_upper = lp.u[old_columns]
        old_boxed = np.isfinite(old_upper)
        old_slack = old_upper[old_boxed] * state.tau - state.x[old_boxed]
        mu_sum = float(state.x @ state.z) + float(old_slack @ state.w[old_boxed])
        mu_sum += state.tau * state.kappa
        mu = mu_sum / (old_columns.size + int(np.count_nonzero(old_boxed)) + 1.0)
        # The extended HSD residual vectors and normalization are tied to the
        # current restricted master.  Rebuild the strictly feasible embedding
        # whenever that operator changes; theta=0 still represents the same
        # restricted LP, so this is an embedding restart rather than Phase I.
        state = None
        hsd_restarts += 1
        columns_added += int(candidate.size)
        columns_dropped += int(drop.size)
        pricing_seconds += time.perf_counter() - pricing_started

    if state is None:
        state = _initialize_state(
            _restricted_matrix(lp.A, columns),
            lp.c[columns],
            lp.b,
            lp.u[columns],
            None if feasible_flow is None else feasible_flow[columns],
        )
    final_metrics = _full_metrics(lp, columns, state)
    pinf, dinf, gap, objective_scaled, reduced = final_metrics
    rd, rp, rg, rn, rd0, rp0, rg0, rn0 = _hsd_residuals(
        _restricted_matrix(lp.A, columns),
        lp.c[columns],
        lp.b,
        lp.u[columns],
        state,
        state.embedding,
        options.rho,
        options.delta,
        options.eta,
        options.zeta,
    )
    hsd_residual = _residual_measure(
        rd0, rp0, rg0, rn0, lp.c[columns], lp.b, state
    )
    final_upper = lp.u[columns]
    final_boxed = np.isfinite(final_upper)
    final_slack = final_upper[final_boxed] * state.tau - state.x[final_boxed]
    final_mu_sum = float(state.x @ state.z)
    final_mu_sum += float(final_slack @ state.w[final_boxed])
    final_mu_sum += state.tau * state.kappa
    complementarity = (
        final_mu_sum
        / (columns.size + int(np.count_nonzero(final_boxed)) + 1.0)
        / max(state.tau * state.tau, TINY)
    )
    x = np.zeros(lp.c.size, dtype=np.float64)
    x[columns] = state.x / max(state.tau, TINY)
    y = state.y / max(state.tau, TINY) * lp.objective_scale
    y_internal = state.y / max(state.tau, TINY)
    z = np.maximum(lp.c - lp.A.T @ y_internal, 0.0)
    w = np.zeros(lp.c.size, dtype=np.float64)
    z[columns] = state.z / max(state.tau, TINY)
    w[columns] = state.w / max(state.tau, TINY)
    z *= lp.objective_scale
    w *= lp.objective_scale
    solve_seconds = time.perf_counter() - total_started - setup_seconds
    result = V3Result(
        status=status,
        objective=objective_scaled * lp.objective_scale,
        x=x,
        y=y,
        z=z,
        w=w,
        tau=state.tau,
        kappa=state.kappa * lp.objective_scale,
        theta=state.theta,
        primal_infeasibility=pinf,
        dual_infeasibility=dinf,
        relative_gap=gap,
        hsd_residual=hsd_residual,
        complementarity=complementarity * lp.objective_scale,
        iterations=iterations,
        pricing_rounds=pricing_rounds,
        active_columns=columns.size,
        columns_added=columns_added,
        columns_dropped=columns_dropped,
        hsd_restarts=hsd_restarts,
        feasible_start_support=(
            int(np.count_nonzero(feasible_flow)) if feasible_flow is not None else 0
        ),
        structure=structure_name,
        preconditioner=preconditioner.backend,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        pricing_seconds=pricing_seconds,
        total_seconds=time.perf_counter() - total_started,
        linear=linear_stats,
        preconditioner_stats=preconditioner_stats,
        proximal=proximal_stats,
        certificate=certificate,
        message=message,
    )
    if result.status in {"time-limit", "iteration-limit", "stalled"}:
        if result.proximal.last_ratio >= 1.0:
            detail = (
                "current proximal subproblem fails the natural-residual test: "
                f"ratio {result.proximal.last_ratio:.3e} >= 1"
            )
            result.message = f"{result.message}; {detail}" if result.message else detail
    if options.print_level:
        print(
            f"-> {result.status}: obj {result.objective:.10e}, "
            f"pinf {pinf:.2e}, dinf {dinf:.2e}, gap {gap:.2e}, "
            f"{result.total_seconds:.2f}s",
            flush=True,
        )
    return result


def solve_box_lp(
    lp: BoxLP, options: Optional[V3Options] = None
) -> V3Result:
    """Solve an arbitrary equality-and-box LP and recover its variables."""
    started = time.perf_counter()
    configured = options or V3Options()
    canonical, transform = canonicalize_box_lp(lp)
    transform_seconds = time.perf_counter() - started
    if canonical.c.size == 0:
        x = transform.shift.copy()
        residual = np.asarray(lp.A @ x - lp.b).ravel()
        primal_scale = 1.0 + float(np.abs(lp.b).max(initial=0.0))
        pinf = float(np.abs(residual).max(initial=0.0)) / primal_scale
        y = np.zeros(lp.b.size, dtype=np.float64)
        z = np.maximum(lp.c, 0.0)
        w = np.maximum(-lp.c, 0.0)
        status = "optimal" if pinf <= configured.tolerance else "primal-infeasible"
        return V3Result(
            status=status,
            objective=float(lp.c @ x + lp.objective_offset),
            x=x,
            y=y,
            z=z,
            w=w,
            primal_infeasibility=pinf,
            dual_infeasibility=0.0,
            relative_gap=0.0 if status == "optimal" else math.inf,
            complementarity=0.0,
            structure="fixed-box",
            preconditioner="none",
            setup_seconds=transform_seconds,
            total_seconds=time.perf_counter() - started,
            message=(
                "all variables were fixed and violate the equalities"
                if status != "optimal"
                else "all variables were substituted exactly"
            ),
        )
    remaining = max(1.0e-3, configured.time_limit - transform_seconds)
    result = solve(canonical, replace(configured, time_limit=remaining))

    if result.x is None or result.y is None:
        result.total_seconds = time.perf_counter() - started
        result.setup_seconds += transform_seconds
        return result

    x = transform.recover_primal(result.x)
    y = result.y
    reduced = np.asarray(lp.c - lp.A.T @ y).ravel()
    finite_lower = np.isfinite(lp.lower)
    finite_upper = np.isfinite(lp.upper)
    z = np.zeros(lp.c.size, dtype=np.float64)
    w = np.zeros(lp.c.size, dtype=np.float64)
    for position, (source, sign) in enumerate(zip(transform.source, transform.sign)):
        if sign > 0.0 and finite_lower[source]:
            z[source] += result.z[position]
            if finite_upper[source] and np.isfinite(canonical.u[position]):
                w[source] += result.w[position]
        elif sign < 0.0 and finite_upper[source] and not finite_lower[source]:
            # x = upper - q: the lower multiplier of q is the upper
            # multiplier of x.  A free variable's negative split contributes
            # no bound multiplier and therefore enters neither branch.
            w[source] += result.z[position]
    fixed = finite_lower & finite_upper & (lp.lower == lp.upper)
    z[fixed] = np.maximum(reduced[fixed], 0.0)
    w[fixed] = np.maximum(-reduced[fixed], 0.0)

    primal_residual = np.asarray(lp.A @ x - lp.b).ravel()
    lower_violation = np.maximum(lp.lower[finite_lower] - x[finite_lower], 0.0)
    upper_violation = np.maximum(x[finite_upper] - lp.upper[finite_upper], 0.0)
    finite_data = np.concatenate(
        (
            np.abs(lp.b),
            np.abs(lp.lower[finite_lower]),
            np.abs(lp.upper[finite_upper]),
        )
    )
    primal_scale = 1.0 + float(finite_data.max(initial=0.0))
    result.primal_infeasibility = max(
        float(np.abs(primal_residual).max(initial=0.0)),
        float(lower_violation.max(initial=0.0)),
        float(upper_violation.max(initial=0.0)),
    ) / primal_scale
    stationarity = reduced - z + w
    result.dual_infeasibility = float(
        np.abs(stationarity).max(initial=0.0)
    ) / (1.0 + float(np.abs(lp.c).max(initial=0.0)))

    primal_objective = float(lp.c @ x + lp.objective_offset)
    dual_objective = float(lp.b @ y + lp.objective_offset)
    dual_objective += float(lp.lower[finite_lower] @ z[finite_lower])
    dual_objective -= float(lp.upper[finite_upper] @ w[finite_upper])
    result.relative_gap = abs(primal_objective - dual_objective) / (
        1.0 + abs(primal_objective) + abs(dual_objective)
    )
    complementarity_sum = float(
        (x[finite_lower] - lp.lower[finite_lower]) @ z[finite_lower]
    )
    complementarity_sum += float(
        (lp.upper[finite_upper] - x[finite_upper]) @ w[finite_upper]
    )
    result.complementarity = complementarity_sum / max(
        1, int(finite_lower.sum() + finite_upper.sum())
    )
    result.objective = primal_objective
    result.x = x
    result.z = z
    result.w = w
    if result.status == "dual-infeasible" and result.certificate is not None:
        result.certificate = transform.recover_direction(result.certificate)
    if result.status == "optimal" and max(
        result.primal_infeasibility,
        result.dual_infeasibility,
        result.relative_gap,
    ) > configured.tolerance:
        result.status = "postsolve-inaccurate"
        detail = (
            "box-model KKT check failed: "
            f"pinf {result.primal_infeasibility:.3e}, "
            f"dinf {result.dual_infeasibility:.3e}, "
            f"gap {result.relative_gap:.3e}"
        )
        result.message = f"{result.message}; {detail}" if result.message else detail
    result.setup_seconds += transform_seconds
    result.total_seconds = time.perf_counter() - started
    return result


def _load_shared_lp_module():
    try:
        from . import psipm_active_pool as module
    except ImportError:
        import psipm_active_pool as module
    return module


def _scaled_tolerance(standard, requested: float, max_tightening: float = 10.0):
    """Tighten a scaled solve enough to protect original-space KKT accuracy."""
    column_scale = np.asarray(standard.scale, dtype=np.float64)
    row_scale = np.asarray(standard.row_scale, dtype=np.float64)
    c_unscaled = standard.c / column_scale
    b_unscaled = standard.b / row_scale
    dual_amplification = (
        (1.0 + float(np.abs(standard.c).max(initial=0.0)))
        * float((1.0 / column_scale).max(initial=1.0))
        / (1.0 + float(np.abs(c_unscaled).max(initial=0.0)))
    )
    primal_amplification = (
        (1.0 + float(np.abs(standard.b).max(initial=0.0)))
        * float((1.0 / row_scale).max(initial=1.0))
        / (1.0 + float(np.abs(b_unscaled).max(initial=0.0)))
    )
    amplification = min(
        max(float(max_tightening), 1.0),
        max(1.0, dual_amplification, primal_amplification),
    )
    return float(requested) / amplification


def solve_mps(
    path,
    options: Optional[V3Options] = None,
    presolve: bool = False,
    scale: bool = True,
):
    """Read, canonicalize, scale, solve, and check an MPS/LP model.

    The returned tuple is ``(raw_model, standard_model, result, check)``.  All
    reported total time includes file input, optional HiGHS presolve,
    canonicalization, scaling, V3 setup, and the solve itself.
    """
    started = time.perf_counter()
    configured = options or V3Options()
    requested_tolerance = configured.tolerance
    PS = _load_shared_lp_module()
    raw_original = PS.read_mps(str(path))
    raw = raw_original
    presolver = None
    if presolve:
        remaining = max(1.0e-3, configured.time_limit - (time.perf_counter() - started))
        raw, presolver, _ = PS.presolve_raw(raw_original, time_limit=remaining)
    standard = PS.to_standard_form(raw, verbose=bool(configured.print_level))
    if scale:
        PS.scale_problem(standard, mode=5, verbose=bool(configured.print_level))
    internal_tolerance = _scaled_tolerance(standard, configured.tolerance)
    if configured.print_level and internal_tolerance < configured.tolerance:
        print(
            f"  scaled tol     : {internal_tolerance:.3e} "
            f"(original target {configured.tolerance:.3e})",
            flush=True,
        )

    lower = np.where(standard.pos, 0.0, -np.inf)
    upper = np.where(standard.pos, standard.u, np.inf)
    box = BoxLP(
        standard.c,
        standard.A,
        standard.b,
        lower=lower,
        upper=upper,
        name=Path(path).name,
        objective_offset=standard.obj_const,
    )
    preprocessing_seconds = time.perf_counter() - started
    remaining = max(1.0e-3, configured.time_limit - preprocessing_seconds)
    result = solve_box_lp(
        box,
        replace(
            configured,
            tolerance=internal_tolerance,
            time_limit=remaining,
        ),
    )

    solution = PS.Solution(
        x=result.x,
        y=result.y,
        z=result.z,
        w=result.w,
        status=result.status,
        obj=result.objective,
        outer_iter=result.iterations,
        inner_iter=result.linear.iterations,
        time=result.solve_seconds,
        t_fact=result.linear.preconditioner_seconds,
        t_solve=result.linear.seconds,
        res_p=result.primal_infeasibility,
        res_d=result.dual_infeasibility,
        mu=result.complementarity,
    )
    check_pre = PS.recover(standard, raw, solution)
    check = (
        PS._postsolve_info(presolver, raw, raw_original, check_pre)
        if presolver is not None and raw is not raw_original
        else check_pre
    )
    check["gap"] = float(result.relative_gap)
    check["complementarity"] = float(result.complementarity)
    if result.status == "optimal" and max(
        check["pinf"], check["dinf"], result.relative_gap
    ) > requested_tolerance:
        result.status = "postsolve-inaccurate"
        detail = (
            "original-model KKT check failed: "
            f"pinf {check['pinf']:.3e}, dinf {check['dinf']:.3e}, "
            f"gap {result.relative_gap:.3e}"
        )
        result.message = f"{result.message}; {detail}" if result.message else detail
    result.objective = float(check["obj"])
    result.primal_infeasibility = float(check["pinf"])
    result.dual_infeasibility = float(check["dinf"])
    result.setup_seconds += preprocessing_seconds
    result.total_seconds = time.perf_counter() - started
    solution.status = result.status
    solution.obj = result.objective
    solution.time = result.total_seconds
    return raw_original, standard, result, check


def read_dimacs_network(path) -> CanonicalLP:
    """Read a zero-lower DIMACS minimum-cost-flow network."""
    path = Path(path)
    nodes = arcs = None
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"p "):
                fields = line.split()
                if len(fields) != 4 or fields[1] != b"min":
                    raise ValueError(f"unsupported DIMACS problem line: {line!r}")
                nodes, arcs = int(fields[2]), int(fields[3])
                break
    if nodes is None:
        raise ValueError(f"no DIMACS problem line found in {path}")
    balance = np.zeros(nodes, dtype=np.float64)
    costs = np.empty(arcs, dtype=np.float64)
    upper_bounds = np.empty(arcs, dtype=np.float64)
    indices = np.empty(2 * arcs, dtype=np.int32)
    values = np.empty(2 * arcs, dtype=np.int8)
    arc = 0
    with path.open("rb", buffering=16 * 1024 * 1024) as handle:
        for line in handle:
            if not line:
                continue
            if line[0] == ord("n"):
                fields = line.split()
                balance[int(fields[1]) - 1] = float(fields[2])
            elif line[0] == ord("a"):
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError(f"invalid arc record: {line!r}")
                tail, head = int(fields[1]) - 1, int(fields[2]) - 1
                lower, upper = float(fields[3]), float(fields[4])
                if lower != 0.0 or (upper != -1.0 and upper <= 0.0):
                    raise ValueError(
                        "V3 requires zero lower bounds and positive or -1 upper bounds"
                    )
                position = 2 * arc
                if tail < head:
                    indices[position : position + 2] = (tail, head)
                    values[position : position + 2] = (1, -1)
                else:
                    indices[position : position + 2] = (head, tail)
                    values[position : position + 2] = (-1, 1)
                costs[arc] = float(fields[5])
                upper_bounds[arc] = np.inf if upper == -1.0 else upper
                arc += 1
    if arc != arcs:
        raise ValueError(f"declared {arcs} arcs but read {arc}")
    if abs(float(balance.sum())) > 1.0e-9 * (1.0 + float(np.abs(balance).sum())):
        raise ValueError("network supplies are not balanced")
    indptr = np.arange(0, 2 * arcs + 1, 2, dtype=np.int64)
    A = sp.csc_matrix((values, indices, indptr), shape=(nodes, arcs), copy=False)
    return CanonicalLP(costs, A, balance, name=path.name, u=upper_bounds)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="MPS/LP model or DIMACS *.net/*.dimacs model")
    parser.add_argument("--time-limit", type=float, default=2000.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--rmp-steps", type=int, default=3)
    parser.add_argument("--proximal-steps", type=int, default=1)
    parser.add_argument("--proximal-reduction", type=float, default=0.7)
    parser.add_argument("--proximal-residual-factor", type=float, default=1.0e4)
    parser.add_argument("--proximal-relative-tolerance", type=float, default=0.9)
    parser.add_argument("--rho", type=float, default=1.0e-8)
    parser.add_argument("--delta", type=float, default=1.0e-8)
    parser.add_argument("--eta", type=float, default=1.0e-8)
    parser.add_argument("--zeta", type=float, default=1.0e-8)
    parser.add_argument("--pool-factor", type=float, default=1.0)
    parser.add_argument("--pricing-batch", type=int, default=250_000)
    parser.add_argument("--pcg-max-iterations", type=int, default=1000)
    parser.add_argument(
        "--generic-preconditioner",
        choices=("auto", "stable", "diagonal"),
        default="auto",
    )
    parser.add_argument("--generic-core-factor", type=float, default=1.0)
    parser.add_argument("--generic-max-core-factor", type=float, default=3.0)
    parser.add_argument("--generic-retention", type=float, default=0.9)
    parser.add_argument("--generic-factor-max-age", type=int, default=2)
    parser.add_argument("--generic-core-max-age", type=int, default=12)
    parser.add_argument("--generic-refresh-cg", type=int, default=60)
    parser.add_argument("--generic-fill-weight", type=float, default=0.2)
    parser.add_argument("--generic-retry-growth", type=float, default=1.6)
    parser.add_argument(
        "--generic-growth-amortization", type=float, default=1.0
    )
    parser.add_argument("--generic-cholmod-order", default="best")
    parser.add_argument("--generic-low-rank-max-columns", type=int, default=8)
    parser.add_argument("--generic-low-rank-min-nnz", type=int, default=1024)
    parser.add_argument(
        "--generic-low-rank-clique-ratio", type=float, default=2.0
    )
    parser.add_argument("--presolve", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--print-level", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-output", help="write result diagnostics to this JSON file")
    return parser


def main():
    args = _parser().parse_args()
    options = V3Options(
        tolerance=args.tolerance,
        time_limit=args.time_limit,
        max_iterations=args.max_iterations,
        rmp_steps=args.rmp_steps,
        proximal_steps=args.proximal_steps,
        proximal_reduction=args.proximal_reduction,
        proximal_residual_factor=args.proximal_residual_factor,
        proximal_relative_tolerance=args.proximal_relative_tolerance,
        rho=args.rho,
        delta=args.delta,
        eta=args.eta,
        zeta=args.zeta,
        pool_factor=args.pool_factor,
        pricing_batch=args.pricing_batch,
        pcg_max_iterations=args.pcg_max_iterations,
        generic_preconditioner=args.generic_preconditioner,
        generic_core_factor=args.generic_core_factor,
        generic_max_core_factor=args.generic_max_core_factor,
        generic_retention=args.generic_retention,
        generic_factor_max_age=args.generic_factor_max_age,
        generic_core_max_age=args.generic_core_max_age,
        generic_refresh_cg=args.generic_refresh_cg,
        generic_fill_weight=args.generic_fill_weight,
        generic_retry_growth=args.generic_retry_growth,
        generic_growth_amortization=args.generic_growth_amortization,
        generic_cholmod_order=args.generic_cholmod_order,
        generic_low_rank_max_columns=args.generic_low_rank_max_columns,
        generic_low_rank_min_nnz=args.generic_low_rank_min_nnz,
        generic_low_rank_clique_ratio=args.generic_low_rank_clique_ratio,
        print_level=args.print_level,
    )
    suffixes = [suffix.lower() for suffix in Path(args.model).suffixes]
    if suffixes and suffixes[-1] in {".net", ".dimacs"}:
        result = solve(read_dimacs_network(args.model), options)
    else:
        _, _, result, _ = solve_mps(
            args.model,
            options,
            presolve=args.presolve,
            scale=not args.no_scale,
        )
    json_text = json.dumps(result.as_dict(), indent=2, sort_keys=True)
    if args.json_output:
        Path(args.json_output).write_text(json_text + "\n", encoding="utf-8")
    if args.json:
        print(json_text)


if __name__ == "__main__":
    main()
