# PSIPM-V3 HSD

`psipm_v3_hsd.py` is an experimental generic LP solver built around an
infeasible homogeneous self-dual proximal-stabilized interior-point method.
It accepts arbitrary sparse equality-and-box LPs and MPS/LP files.  Incidence
matrices automatically use an additional network specialization.  It is a new
file: the V2 solvers are unchanged.

## Mathematical system

For `min c'x`, `Ax=b`, `0<=x<=u`, choose a strictly interior start and define
fixed residual data `bar_b`, `bar_c`, `bar_g`, and normalization `alpha`. V3
solves the extended bounded HSD equations

```
c tau - A' y - bar_c theta - z + w + rho (x-xc) = 0
b tau - A x - bar_b theta - delta (y-yc) = 0
-c' x + b' y - u' w + bar_g theta - kappa + eta (tau-tauc) = 0
-bar_b' y + bar_c' x - bar_g tau + alpha + zeta (theta-thetac) = 0
X z = mu e, (u tau-X)w = mu e, tau kappa = mu.
```

At initialization `theta=tau=1`, all four embedding equations are exactly
satisfied, and unit complementarity gives `alpha=n+n_box+1`. The identity
`alpha theta = x'z + (u tau-x)'w + tau kappa` excludes the zero embedding.
At a solution `theta -> 0`; `tau>0` recovers a primal-dual solution by division
by `tau`, while a validated `tau -> 0` ray can certify infeasibility.

`rho`, `delta`, `eta`, and `zeta` default to `1e-8`, are clipped to a strictly
positive floor, and never decrease with `mu`. V3 uses the geometrically tightened
natural-residual inexactness rule from the original MATLAB PS-IPM.  For
`q=(x,y,tau,theta)`, HSD cone `C`, skew HSD operator `Q`, and proximal metric `M`,

```
Rk(q) = q - projection_C(q - (Qq + M(q-qk))),
||Rx|| + ||Ry|| + |Rtau| + |Rtheta|
    < 1e4 * 0.7^k * min(1, ||q-qk||_block).
```

The forcing sequence is summable.  On acceptance V3 uses the PS-IPM centre
update `qk <- q`; the previous HPE extragradient update is no longer used.  The
HPE ratio remains available only as a diagnostic.  Pricing and active-pool
changes wait until the current restricted subproblem passes the natural-
residual test.  At a fixed point, centre displacements vanish and the
unregularized HSD equations remain.

Eliminating `dx` gives the SPD matrix

```
H = A diag(1 / (rho + z/x + w/(u tau-x))) A' + delta I.
```

V3 applies `H` matrix-free.  On a generic sparse matrix it factorizes the
stable active-column preconditioner

```
P = A_C D_C A_C' + diag(A_N D_N A_N') + delta I,
```

where `C` is a retained, leverage-ranked, fill-aware column core and `N` is
the eliminated complement.  PCG always multiplies by the full `H`, so this
elimination cannot change the Newton equation.  The core grows after difficult
Krylov solves.  A Jacobi fallback is available when CHOLMOD is unavailable.

For incidence matrices, a protected spanning-forest core can instead define a
tree-plus-diagonal preconditioner whose numeric LDL updates and solves are
linear in the number of rows. The two-scalar `(tau,theta)` HSD border is
recovered from Schur solves with `H`, so no doubled embedding matrix is stored.

## Generic box transformation

The public `BoxLP` interface is the primary solver interface and accepts any
sparse equality matrix with finite or infinite lower and upper bounds.
V3 substitutes fixed variables, shifts finite lower bounds, flips upper-only
variables, and splits free variables.  MPS inequalities become bounded row
slacks through `psipm_active_pool.py`.  The MPS route applies the same ten-sweep
two-sided Ruiz scaling used by the IPX/HiPO comparison code and maps primal and
dual variables back before the final KKT check.

## Column technology

For generic matrices, every model column remains in the iterate and full
matrix-free Newton operator.  Column elimination is a core technology inside
the sparse preconditioner only.

For incidence matrices, the restricted master always contains a spanning
forest.  For Bonn-style
transshipment networks, directed BFS trees construct an exact nonnegative
feasible flow and its support is also protected.  Full-model reduced costs
price inactive columns.  New columns start near their lower bound while
preserving complementarity.  Non-tree columns can be eliminated only after
small full-model KKT residuals, a reduced-cost margin, and repeated inactivity;
all elimination is reversible.  Optimality and Farkas rays are validated on
the full original model.

## Commands

Run a generic MPS model with stable active-column preconditioning:

```bash
python psipm_v3_hsd.py model.mps --time-limit 600 \
  --generic-preconditioner stable --json-output model_v3.json
```

Optional HiGHS presolve and explicit fallback selection are available as
`--presolve`, `--generic-preconditioner auto`, and
`--generic-preconditioner diagonal`.

Compare generic V3, IPX, and HiPO on every supported model in a folder:

```bash
python batch_test_v3.py /scratch/sc9c23/mps --time-limit 600 \
  --threads 1 --csv v3_comparison.csv --overwrite
```

Use `--choose-folder` instead of the folder argument for a graphical chooser,
or `--resume` to continue an interrupted CSV.  Every run uses a fresh process;
the detailed CSV and `*_summary.csv` include full wall time and separate setup,
solve, factorization, and validation timings.

Run V3 on Josef:

```bash
python psipm_v3_hsd.py /scratch/sc9c23/vlsi/Josef_FlowGraph_1.net \
  --time-limit 2000 --proximal-reduction 0.7 \
  --proximal-residual-factor 1e4 \
  --print-level 1 --json
```

Compare V3, IPX, and HiPO fairly in separate processes:

```bash
python test_vlsi_v3.py /scratch/sc9c23/vlsi/Josef_FlowGraph_1.net \
  --time-limit 2000 --threads 1 --csv josef_v3_comparison.csv
```

Run correctness tests:

```bash
python -m unittest -v test_psipm_v3_hsd.py
```

## Generic HSD validation

The normalized four-equation HSD implementation passes regression checks for
arbitrary non-incidence matrices, every bound type, exact initialization,
the complete bordered Newton linearization, original-space multiplier
recovery, infeasibility certificates, and finite-box optimality. With a
600-second total limit on the local machine:

* `datt256_lp`: optimal in 414.87 seconds, objective `255.9999992304`, original
  `pinf=6.48e-8`, `dinf=5.02e-13`, gap `1.50e-9`, `tau=0.801`.
* `s250r10`: optimal in 266.07 seconds, objective `-0.1726770385`, original
  `pinf=1.32e-15`, `dinf=9.90e-11`, gap `4.83e-10`, `tau=1.046`.
* `s82`: time limit after 602.08 measured seconds including preprocessing;
  it did not approach original-model KKT feasibility. This remains a known
  performance limitation, not a solved result.

## Josef validation status

With the natural-residual rule, a 300-second one-thread run accepted 19
proximal subproblems, completed 19 pricing rounds, added 4,330,753 columns, and
had no failed PCG solves.  It stopped at the time limit with natural-residual
ratio `1.237` for subproblem 20, so that centre update was correctly rejected.
The final active pool had 8,729,036 columns; primal infeasibility was `9.36e-1`,
dual infeasibility `4.27e-1`, and relative gap `4.39e-3`.  This validates the
intended inexact PS-IPM/column-generation interaction, but is not a solved
Josef result.  See `josef_natural_inexactness_300s.json`.

The measurements below predate this natural-residual rule and are retained as
historical baselines only.

The local Josef model has 3,783,012 nodes, 9,278,971 arcs, and 18,557,942
nonzeros.  V3 detected an exact feasible flow supported on 3,935,959 arcs in
2.3 seconds; its protected tree/support union contains 4,398,283 columns.

The latest bounded 12-iteration run took 42.34 solver-seconds and ended at the
iteration limit, with 7,394,722 active columns, primal infeasibility 1.01,
relative gap `3.09e-5`, and no failed PCG solves.  This is a successful
large-scale execution and a substantial improvement over the earlier V2 path,
but it is **not yet a solved Josef result**.  HiPO remains the robust reference
for this instance until a long V3 run reaches the requested full-model KKT
tolerance.  In particular, the experiment indicates that Josef needs a broad
column support during feasibility/early optimization; column elimination is
expected to become useful late rather than making the initial master tiny.

## Erhard validation

Erhard has 7,678,083 nodes, 21,881,099 arcs, and 43,762,198 nonzeros.  V3
constructed an exactly feasible directed-tree flow on 8,520,799 arcs.  In a
one-thread, 300-second comparison, V3, IPX, and HiPO all reached the time limit.
IPX had the strongest KKT residuals.  See `ERHARD_V3_VALIDATION.md` and
`erhard_v3_validation.csv` for original-unit metrics and the detailed result.
