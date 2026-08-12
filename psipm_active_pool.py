"""
psipm.py -- Proximal-Stabilized Interior Point Method (PS-IPM) for Linear Programming.

Python port of the MATLAB code of

    S. Cipolla, J. Gondzio, "Proximal Stabilized Interior Point Methods and
    Low-Frequency-Update Preconditioning Techniques", JOTA (2023).
    https://github.com/StefanoCipolla/PS-IPM

Differences w.r.t. the MATLAB reference implementation
------------------------------------------------------
1.  LP only: the Hessian Q is identically zero and is removed from all formulas.
2.  Native two-sided bounds.  The MATLAB `LP_Convert_to_Standard_Form` handles a
    box  l <= x <= u  by shifting to  w >= 0  and appending an *extra row*
    w + w2 = u - l  together with an extra slack.  For an instance such as
    datt256 (262,144 boxed columns) this would add 262,144 rows and columns.
    Here the upper bounds are handled inside the IPM, in the usual way: a slack
    s = u - x >= 0 with dual w >= 0 is carried, so that the Newton diagonal
    becomes  Theta^{-1} = Z/X + W/S  and the problem size is unchanged.
3.  Linear algebra.  Because Q = 0 the (1,1) block of the augmented system is
    diagonal, so the normal equations
        (A Theta^{-1} A^T + delta I) dy = ...
    can be formed and are *symmetric positive definite* for any delta > 0
    (the proximal/dual regularization makes them PD even if A is rank
    deficient).  They are factorized with CHOLMOD; the symbolic analysis is
    computed once and reused for every numerical factorization.
    The MATLAB augmented-system LDL^T path is also available
    (`kkt="augmented"`, via QDLDL) and is the better choice when
    m >> n or when A Theta A^T fills in.
4.  Mehrotra's starting point, the natural proximal residual, the stopping
    tests and Gondzio's multiple centrality correctors are all extended to the
    boxed case.

Author of the port: generated for S. Cipolla.
"""

from __future__ import annotations

import bz2
import contextlib
import gzip
import os
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

INF = 1.0e20  # HiGHS convention for "infinite" bound

_EPS = float(np.finfo(np.float64).eps)
# Smallest distance from a bound that still has meaning in double precision.
# A slack closer than this to zero cannot be told apart from zero, and dividing
# by it would blow up the Newton diagonal Theta.
_POS_FLOOR = 8.0 * _EPS

# --------------------------------------------------------------------------- #
# Optional sparse factorization back-ends
# --------------------------------------------------------------------------- #
try:  # CHOLMOD (SuiteSparse) -- supernodal Cholesky, symbolic analysis reuse
    import sksparse.cholmod as _cholmod

    HAVE_CHOLMOD = True
    # CHOLMOD warns about near singularity at every late IPM iteration; the
    # proximal regularization keeps the system solvable, so this is expected.
    warnings.filterwarnings("ignore", category=_cholmod.CholmodWarning)
except Exception:  # pragma: no cover
    HAVE_CHOLMOD = False

try:  # QDLDL -- LDL^T for quasi-definite matrices, symbolic analysis reuse
    import qdldl as _qdldl

    HAVE_QDLDL = True
except Exception:  # pragma: no cover
    HAVE_QDLDL = False


# =========================================================================== #
#  Problem containers
# =========================================================================== #
@dataclass
class RawLP:
    """LP exactly as it comes out of the MPS file.

        min/max  c^T x + offset
        s.t.     rl <= A x <= ru
                 cl <=  x  <= cu
    """

    c: np.ndarray
    A: sp.csc_matrix
    rl: np.ndarray
    ru: np.ndarray
    cl: np.ndarray
    cu: np.ndarray
    offset: float = 0.0
    maximize: bool = False
    name: str = ""

    @property
    def shape(self):
        return self.A.shape


@dataclass
class StandardLP:
    """Canonical form used by the solver:

        min  c^T v + obj_const
        s.t. A v = b
             0 <= v_j            for j in `pos`
                  v_j <= u_j     for j in `box`  (subset of `pos`)
             v_j free            for j not in `pos`
    """

    c: np.ndarray
    A: sp.csc_matrix
    b: np.ndarray
    u: np.ndarray  # +inf where there is no upper bound
    pos: np.ndarray  # bool mask, non-free variables
    box: np.ndarray  # bool mask, variables with a finite upper bound
    # Valid caps used only by retrospective pool certificates.  Generated
    # one-sided row slacks may have finite implied caps even though their
    # explicit solver bounds remain infinite.
    cert_u: np.ndarray = None
    obj_const: float = 0.0
    # ---- data needed to map a solution back to the original variables ------
    shift: np.ndarray = None  # v -> x = shift + sign * scale * v
    sign: np.ndarray = None
    scale: np.ndarray = None  # right (column) scaling actually applied
    row_scale: np.ndarray = None  # left scaling: scaled y -> original y
    keep: np.ndarray = None  # indices, in the "rows-made-equalities" space
    n_aug: int = 0  # number of columns before fixed-column removal
    n_orig: int = 0  # number of columns of the original LP
    rows_keep: np.ndarray = None  # original row indices that survived
    m_orig: int = 0


# =========================================================================== #
#  MPS input through HiGHS
# =========================================================================== #
def _raw_from_highs_lp(lp) -> RawLP:
    """Convert a ``highspy.HighsLp`` object to the lightweight RawLP container."""
    m, n = lp.num_row_, lp.num_col_
    start = np.asarray(lp.a_matrix_.start_, dtype=np.int64)
    index = np.asarray(lp.a_matrix_.index_, dtype=np.int64)
    value = np.asarray(lp.a_matrix_.value_, dtype=np.float64)

    if str(lp.a_matrix_.format_).endswith("kColwise"):
        A = sp.csc_matrix((value, index, start), shape=(m, n))
    else:
        A = sp.csr_matrix((value, index, start), shape=(m, n)).tocsc()

    maximize = str(lp.sense_).endswith("kMaximize")
    c = np.asarray(lp.col_cost_, dtype=np.float64).copy()
    if maximize:
        c = -c

    return RawLP(
        c=c,
        A=A,
        rl=np.asarray(lp.row_lower_, dtype=np.float64).copy(),
        ru=np.asarray(lp.row_upper_, dtype=np.float64).copy(),
        cl=np.asarray(lp.col_lower_, dtype=np.float64).copy(),
        cu=np.asarray(lp.col_upper_, dtype=np.float64).copy(),
        offset=float(lp.offset_),
        maximize=maximize,
        name=str(lp.model_name_),
    )


def _read_highs_file(read_path: str, reported_path: str = None) -> RawLP:
    """Read one uncompressed model, retrying fixed-format MPS if necessary."""
    import highspy

    reported_path = reported_path or read_path
    lower_path = read_path.lower()
    parser_modes = (True, False) if lower_path.endswith((".mps", ".qps")) else (None,)

    for free_parser in parser_modes:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        if free_parser is not None:
            option_status = h.setOptionValue("mps_parser_type_free", free_parser)
            if str(option_status).endswith("kError"):
                continue
        status = h.readModel(read_path)
        if not str(status).endswith("kError"):
            return _raw_from_highs_lp(h.getLp())

    if len(parser_modes) == 2:
        detail = " with either the free-format or fixed-format MPS parser"
    else:
        detail = ""
    raise IOError(f"HiGHS could not read {reported_path}{detail}")


def read_mps(path: str) -> RawLP:
    """Read an MPS/LP file, expanding compressed input when necessary.

    Some HiGHS binary builds omit transparent gzip support.  Expanding through
    Python makes ``*.mps.gz`` and ``*.mps.bz2`` portable across those builds.
    The temporary file is placed under ``$TMPDIR`` by :mod:`tempfile`.  MPS/QPS
    input is first read with HiGHS' free-format parser, then retried with its
    fixed-format parser for instances such as ``ivu59``.
    """
    source_path = os.fspath(path)
    lower_path = source_path.lower()
    compressed = (
        (".mps.gz", gzip.open, ".mps"),
        (".lp.gz", gzip.open, ".lp"),
        (".qps.gz", gzip.open, ".qps"),
        (".mps.bz2", bz2.open, ".mps"),
        (".lp.bz2", bz2.open, ".lp"),
        (".qps.bz2", bz2.open, ".qps"),
    )
    for ending, opener, suffix in compressed:
        if lower_path.endswith(ending):
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                with opener(source_path, "rb") as source:
                    shutil.copyfileobj(source, temporary, length=8 * 1024 * 1024)
                temporary.flush()
                return _read_highs_file(temporary.name, source_path)

    return _read_highs_file(source_path)


def _highs_lp_from_raw(raw: RawLP):
    """Build a continuous HiGHS LP, deliberately dropping MPS integrality."""
    import highspy

    A = raw.A.tocsc(copy=False)
    m, n = A.shape
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = raw.c
    lp.col_lower_ = raw.cl
    lp.col_upper_ = raw.cu
    lp.row_lower_ = raw.rl
    lp.row_upper_ = raw.ru
    lp.offset_ = raw.offset
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    return lp


def presolve_raw(raw: RawLP, time_limit: float = np.inf):
    """Presolve the continuous relaxation and retain the postsolve stack."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if np.isfinite(time_limit):
        h.setOptionValue("time_limit", float(time_limit))
    h.passModel(_highs_lp_from_raw(raw))
    status = h.presolve()
    if str(status).endswith("kError"):
        raise RuntimeError(f"HiGHS presolve failed: {status}")
    pstatus = h.getModelPresolveStatus()
    if str(pstatus).endswith(("kInfeasible", "kUnboundedOrInfeasible", "kTimeout")):
        raise RuntimeError(f"HiGHS presolve stopped with status {pstatus}")
    if str(pstatus).endswith("kReducedToEmpty"):
        raise RuntimeError("HiGHS presolved the model to empty; solve via HiGHS directly")
    reduced = raw if str(pstatus).endswith("kNotReduced") else _raw_from_highs_lp(
        h.getPresolvedLp()
    )
    return reduced, h, pstatus


def read_mps_with_presolve(path: str, time_limit: float = np.inf):
    """Read a model and return ``(original_raw, presolved_raw, highs, status)``.

    The returned HiGHS object owns the presolve stack needed to postsolve a
    solution of ``presolved_raw`` back to the original model.
    """
    raw_orig = read_mps(path)
    raw_pre, h, pstatus = presolve_raw(raw_orig, time_limit=time_limit)
    return raw_orig, raw_pre, h, pstatus


# =========================================================================== #
#  Canonicalization  (replaces LP_Convert_to_Standard_Form.m)
# =========================================================================== #
def _implied_row_slack_caps(A, rl, ru, cl, cu, ineq):
    """Return valid variable-bound caps for generated one-sided row slacks."""
    caps = np.full(ineq.size, np.inf)
    if not ineq.size:
        return caps

    one_sided = (
        ((rl[ineq] > -INF) & (ru[ineq] >= INF))
        | ((rl[ineq] <= -INF) & (ru[ineq] < INF))
    )
    positions = np.flatnonzero(one_sided)
    if not positions.size:
        return caps

    rows = ineq[positions]
    Ar = A[rows, :].tocsr()
    for local, pos_in_ineq in enumerate(positions):
        start, end = Ar.indptr[local : local + 2]
        jj = Ar.indices[start:end]
        aa = Ar.data[start:end]
        row = rows[local]

        if rl[row] > -INF:
            bounds = np.where(aa > 0.0, cu[jj], cl[jj])
            finite = np.where(aa > 0.0, bounds < INF, bounds > -INF)
            if not finite.all():
                continue
            terms = aa * bounds
            cap = float(terms.sum()) - float(rl[row])
        else:
            bounds = np.where(aa > 0.0, cl[jj], cu[jj])
            finite = np.where(aa > 0.0, bounds > -INF, bounds < INF)
            if not finite.all():
                continue
            terms = aa * bounds
            cap = float(ru[row]) - float(terms.sum())

        if np.isfinite(cap):
            margin = 100.0 * np.finfo(np.float64).eps * (
                1.0 + float(np.abs(terms).sum())
            )
            cap = max(0.0, cap + margin)
            if cap < INF:
                caps[pos_in_ineq] = cap
    return caps


def to_standard_form(raw: RawLP, verbose: bool = True) -> StandardLP:
    """Bring a general bounded LP to  ``min c'v : Av = b, 0 <= v <= u``.

    * inequality / ranged rows  ``rl <= a'x <= ru``  become  ``a'x - t = 0``
      with a slack column ``t`` bounded by ``[rl, ru]``;
    * free rows are dropped;
    * fixed columns (``cl == cu``) are substituted out;
    * ``cl`` finite  ->  ``x = cl + v``,  ``v >= 0``;
    * ``cl = -inf``, ``cu`` finite -> ``x = cu - v``, ``v >= 0``;
    * both finite -> ``x = cl + v``, ``0 <= v <= cu - cl``  (kept as a *bound*,
      no extra row -- this is the modification w.r.t. the MATLAB code);
    * both infinite -> free variable.
    """
    A, c = raw.A.tocsc(), raw.c
    m0, n0 = A.shape
    rl, ru, cl, cu = raw.rl, raw.ru, raw.cl, raw.cu

    # ---- 1. rows ---------------------------------------------------------- #
    row_free = (rl <= -INF) & (ru >= INF)
    rows_keep = np.flatnonzero(~row_free)
    if row_free.any():
        A = A[rows_keep, :]
        rl, ru = rl[rows_keep], ru[rows_keep]
    m = A.shape[0]

    is_eq = rl == ru
    ineq = np.flatnonzero(~is_eq)
    n_sl = ineq.size
    slack_caps = _implied_row_slack_caps(A, rl, ru, cl, cu, ineq)
    if n_sl:
        S = sp.csc_matrix(
            (-np.ones(n_sl), (ineq, np.arange(n_sl))), shape=(m, n_sl)
        )
        A = sp.hstack([A, S], format="csc")
        c = np.concatenate([c, np.zeros(n_sl)])
        lo = np.concatenate([cl, rl[ineq]])
        up = np.concatenate([cu, ru[ineq]])
    else:
        lo, up = cl.copy(), cu.copy()
    b = np.where(is_eq, rl, 0.0)
    n_aug = A.shape[1]

    lo = np.where(lo <= -INF, -np.inf, lo)
    up = np.where(up >= INF, np.inf, up)

    # ---- 2. columns ------------------------------------------------------- #
    fixed = lo == up
    lo_fin, up_fin = np.isfinite(lo), np.isfinite(up)

    shift = np.zeros(n_aug)
    shift[lo_fin] = lo[lo_fin]
    only_up = (~lo_fin) & up_fin
    shift[only_up] = up[only_up]
    shift[fixed] = lo[fixed]

    sign = np.ones(n_aug)
    sign[only_up] = -1.0

    # b <- b - A*shift ;  obj_const <- obj_const + c'shift   (before sign flip)
    nz = shift != 0.0
    if nz.any():
        b = b - A @ np.where(nz, shift, 0.0)
    obj_const = float(c @ shift)

    # apply the sign flip  x = shift + sign * v
    if only_up.any():
        A = A @ sp.diags(sign)
        c = c * sign
        A = A.tocsc()

    u = np.where(lo_fin & up_fin, up - lo, np.inf)
    cert_u = u.copy()
    if n_sl:
        cert_u[n0:] = np.minimum(cert_u[n0:], slack_caps)
    pos = lo_fin | up_fin  # everything that is not free
    keep = np.flatnonzero(~fixed)

    if fixed.any():
        A = A[:, keep].tocsc()
        c, u, cert_u, pos = c[keep], u[keep], cert_u[keep], pos[keep]

    A.sort_indices()
    lp = StandardLP(
        c=np.ascontiguousarray(c),
        A=A,
        b=np.ascontiguousarray(b),
        u=u,
        pos=pos,
        box=pos & np.isfinite(u),
        cert_u=cert_u,
        obj_const=obj_const + raw.offset * (-1.0 if raw.maximize else 1.0),
        shift=shift,
        sign=sign,
        scale=np.ones(keep.size),
        row_scale=np.ones(m),
        keep=keep,
        n_aug=n_aug,
        n_orig=n0,
        rows_keep=rows_keep,
        m_orig=m0,
    )
    if verbose:
        print(
            f"  canonical form : m = {lp.A.shape[0]}, n = {lp.A.shape[1]}, "
            f"nnz = {lp.A.nnz}  (slacks {n_sl}, fixed removed {int(fixed.sum())}, "
            f"boxed {int(lp.box.sum())}, free {int((~lp.pos).sum())})"
        )
    return lp


# =========================================================================== #
#  Scaling  (vectorized port of Scale_the_problem.m, + Ruiz)
# =========================================================================== #
def _nextpow2(v: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v)
    good = v > 0
    out[good] = np.ceil(np.log2(v[good]))
    return out


def scale_problem(lp: StandardLP, mode: int = 3, verbose: bool = True) -> StandardLP:
    """Right (column) scaling  A <- A D,  c <- D c,  u <- D^{-1} u.

    ``mode`` follows ``Scale_the_problem.m``:
    0 none, 1 geometric, 2 equilibrium, 3 geometric rounded to a power of two,
    4 mixed.  ``mode = 5`` adds a few Ruiz equilibration sweeps on both sides
    (not in the MATLAB code, but useful on badly scaled instances).
    """
    A = lp.A
    n = A.shape[1]
    absA = abs(A)
    if mode == 0 or (absA.max() <= 10.0 and absA.data.min(initial=1.0) >= 0.1):
        if verbose:
            print("  scaling        : none")
        return lp

    # column-wise max / min of the *stored* entries (segmented reductions:
    # scipy's sparse .max() would take implicit zeros into account)
    colnnz = np.diff(absA.indptr)
    nonempty = colnnz > 0
    starts = absA.indptr[:-1][nonempty]
    colmax = np.ones(n)
    colmin = np.ones(n)
    if starts.size:
        colmax[nonempty] = np.maximum.reduceat(absA.data, starts)
        colmin[nonempty] = np.minimum.reduceat(absA.data, starts)
    prod = colmax * colmin

    D = np.ones(n)
    if mode == 1:
        ok = (prod > 1e-12) & (prod < 1e12)
        D[ok] = 1.0 / np.sqrt(prod[ok])
        tag = "geometric"
    elif mode == 2:
        ok = colmax > 1e-6
        D[ok] = 1.0 / colmax[ok]
        tag = "equilibrium"
    elif mode == 3:
        ok = (prod > 1e-12) & (prod < 1e12)
        D[ok] = 1.0 / 2.0 ** (_nextpow2(np.sqrt(prod[ok])) - 1.0)
        tag = "geometric / power of two"
    elif mode == 4:
        c1 = (colmax > 1e3) & (colmin < 1e-3)
        c2 = (~c1) & (1.0 / colmin > colmax) & (colmin > 1e-6)
        c3 = (~c1) & (~c2) & (colmax < 1e6)
        D[c1] = 1.0 / 2.0 ** (_nextpow2(np.sqrt(prod[c1])) - 1.0)
        D[c2] = 1.0 / 2.0 ** (_nextpow2(colmin[c2]) - 1.0)
        D[c3] = 1.0 / 2.0 ** (_nextpow2(colmax[c3]) - 1.0)
        tag = "mixed"
    elif mode == 5:
        tag = "Ruiz (both sides)"
        DR = np.ones(n)
        DL = np.ones(A.shape[0])
        Acur = A.copy()
        for _ in range(10):
            aa = abs(Acur)
            r = np.sqrt(np.asarray(aa.max(axis=1).todense()).ravel())
            cc = np.sqrt(np.asarray(aa.max(axis=0).todense()).ravel())
            r[r == 0] = 1.0
            cc[cc == 0] = 1.0
            Acur = sp.diags(1.0 / r) @ Acur @ sp.diags(1.0 / cc)
            DL /= r
            DR /= cc
        lp.A = Acur.tocsc()
        lp.A.sort_indices()
        lp.b = lp.b * DL
        lp.c = lp.c * DR
        lp.u = lp.u / DR
        if lp.cert_u is not None:
            lp.cert_u = lp.cert_u / DR
        lp.scale = lp.scale * DR
        lp.row_scale = DL if lp.row_scale is None else lp.row_scale * DL
        if verbose:
            print(f"  scaling        : {tag}")
        return lp
    else:
        raise ValueError("unknown scaling mode")

    lp.A = (A @ sp.diags(D)).tocsc()
    lp.A.sort_indices()
    lp.c = lp.c * D
    lp.u = lp.u / D
    if lp.cert_u is not None:
        lp.cert_u = lp.cert_u / D
    lp.scale = lp.scale * D
    if verbose:
        print(f"  scaling        : {tag}")
    return lp


# =========================================================================== #
#  Newton system  (Newton_factorization.m + Newton_backsolve.m)
# =========================================================================== #
class KKTSolveError(RuntimeError):
    """The KKT system could not be solved accurately enough to be usable."""


_NOT_PD = "not positive definite"

try:  # used to flush the *C* stdio buffers, which Python cannot reach
    import ctypes as _ctypes

    _LIBC = _ctypes.CDLL(None)
except Exception:  # pragma: no cover -- Windows, restricted builds
    _LIBC = None


def _flush_c_stdout() -> None:
    """``fflush(NULL)``: flush every C output stream, CHOLMOD's included.

    Redirecting fd 1 does not change the buffering mode that C stdio picked when
    the stream was first used, so a message printed from C can still be sitting
    in its buffer.  Without this the captured text would come up empty and the
    message would surface later, on the restored descriptor.
    """
    if _LIBC is not None:
        try:
            _LIBC.fflush(None)
        except Exception:  # pragma: no cover
            pass


@contextlib.contextmanager
def _redirect_stdout_fd():
    """Capture writes to file descriptor 1, including those made from C."""
    try:
        saved = os.dup(1)
    except OSError:  # no usable stdout -- nothing to capture
        yield None
        return
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        sys.stdout.flush()
        _flush_c_stdout()
        os.dup2(tmp.fileno(), 1)
        yield tmp
    finally:
        sys.stdout.flush()
        _flush_c_stdout()
        os.dup2(saved, 1)
        os.close(saved)


@contextlib.contextmanager
def _cholmod_call(stats):
    """Run a CHOLMOD factorization with its failure reporting under control.

    Two things need fixing up.  CHOLMOD prints ``not positive definite`` to
    *stdout* from C, once per failed pivot -- that is what interleaves a
    benchmark log with hundreds of copies of the same line.  The condition is
    already reported through the exception below, so the line is counted rather
    than printed; anything else CHOLMOD writes is passed through untouched.

    And depending on the scikit-sparse build, a non-positive-definite matrix
    surfaces either as ``CholmodNotPositiveDefiniteError`` or -- because CHOLMOD
    grades it as status 1, a *warning* -- as a ``CholmodWarning``.  In the second
    case the factor that comes back covers only a leading submatrix, and solving
    with it silently returns garbage, so the warning is promoted to an error.
    """
    caught = []
    with _redirect_stdout_fd() as tmp:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                yield
        finally:
            passthrough = ""
            if tmp is not None:
                _flush_c_stdout()  # before reading: CHOLMOD may still be buffered
                tmp.seek(0)
                text = tmp.read().decode("utf8", "replace")
                tmp.close()
                keep = []
                for line in text.splitlines():
                    if _NOT_PD in line:
                        stats["not_pd"] = stats.get("not_pd", 0) + 1
                    elif line.strip():
                        keep.append(line)
                passthrough = "\n".join(keep)
    if passthrough:
        print(passthrough)
    for wmsg in caught:
        if issubclass(wmsg.category, _cholmod.CholmodWarning) and (
            "positive definite" in str(wmsg.message).lower()
        ):
            raise _cholmod.CholmodNotPositiveDefiniteError(str(wmsg.message))


class _KKTBase:
    n_fact = 0
    n_solve = 0
    t_fact = 0.0
    t_solve = 0.0


class _CholFactor:
    """Thin compatibility wrapper over the two scikit-sparse APIs.

    scikit-sparse >= 0.5 exposes ``cho_factor`` / ``CholeskyFactor.factorize``;
    0.4.x exposes ``analyze`` / ``Factor.cholesky_inplace``.  Both keep the
    symbolic analysis and only redo the numerical factorization.
    """

    def __init__(self, M, order: str = "best", stats: Optional[dict] = None):
        self._new = hasattr(_cholmod, "cho_factor")
        self.stats = {} if stats is None else stats
        with _cholmod_call(self.stats):
            if self._new:
                self._f = _cholmod.cho_factor(M, lower=True, order=order)
            else:  # scikit-sparse 0.4.x
                self._f = _cholmod.analyze(M, ordering_method=order)
                self._f.cholesky_inplace(M)

    def refactor(self, M) -> None:
        with _cholmod_call(self.stats):
            if self._new:
                self._f.factorize(M)
            else:
                self._f.cholesky_inplace(M)

    def solve(self, b):
        return self._f.solve(b) if self._new else self._f.solve_A(b)

    @property
    def nnz(self) -> int:
        try:
            return int(self._f.nnz)
        except Exception:
            return int(self._f.L().nnz)


def _low_rank_normal_columns(
    A: sp.csc_matrix,
    max_columns: int,
    min_nnz: int,
    clique_ratio: float,
) -> np.ndarray:
    """Find columns better represented as exact low-rank updates.

    A column with ``q`` nonzeros may add ``q(q-1)/2`` structural edges to
    ``A A.T``.  Extract it when that potential clique is large relative to the
    input matrix.  Only a small number is retained because the Woodbury dense
    core has quadratic storage and cubic factorization cost in that number.
    """
    if max_columns <= 0 or A.shape[1] == 0:
        return np.empty(0, dtype=np.int64)
    counts = np.diff(A.indptr).astype(float)
    clique_edges = 0.5 * counts * np.maximum(counts - 1.0, 0.0)
    structural_scale = max(float(A.nnz), float(A.shape[0]), 1.0)
    candidates = np.flatnonzero(
        (counts >= max(1, int(min_nnz)))
        & (clique_edges >= float(clique_ratio) * structural_scale)
    )
    if candidates.size > max_columns:
        scores = clique_edges[candidates]
        keep = np.argpartition(scores, -max_columns)[-max_columns:]
        candidates = candidates[keep]
    return np.sort(candidates.astype(np.int64, copy=False))


class NormalEquations(_KKTBase):
    r"""Solve

        [ Theta   -A^T ] [dx]   [r1]
        [ A       delta][dy] = [r2]

    by eliminating ``dx = Theta^{-1}(r1 + A^T dy)`` and factorizing the SPD
    matrix ``M = A Theta^{-1} A^T + delta I`` with CHOLMOD.  Structurally
    dense columns are extracted before forming the sparse matrix and restored
    exactly with Woodbury updates.  If ``A = [A_F, U]``, the factorized matrix
    and inverse are

        M_0 = A_F Theta_F^{-1} A_F^T + delta I,
        M^{-1} = M_0^{-1} - M_0^{-1} U
                 (Theta_L + U^T M_0^{-1} U)^{-1} U^T M_0^{-1}.

    Thus extraction changes only the linear algebra representation, not the
    Newton equation.  This is especially important for the Phase I artificial
    column ``b``, whose outer product would otherwise make the normal matrix
    structurally dense.  The symbolic analysis is done once; every iteration
    only re-runs the numerical factorization.

    Two safeguards keep this usable once ``Theta^{-1}`` starts to spread over
    many orders of magnitude, which is where a plain Cholesky of ``M`` breaks
    down:

    * ``M`` is symmetrically scaled to a unit diagonal before it is handed to
      CHOLMOD.  This is an exact similarity transformation, but ``diag(M)``
      ranges from ``delta`` (a row whose variables all sit at a bound) to
      ``||a_i||^2 / rho`` (a row of free variables), i.e. over ~1e16 near
      convergence; without the scaling the pivots of the badly scaled matrix
      round to something non-positive and CHOLMOD reports "not positive
      definite" on a matrix that is provably PD.
    * Should the factorization still fail, a *tiny* diagonal shift is applied
      to the scaled matrix -- where, thanks to the unit diagonal, ``1e-14`` is
      a relative perturbation of ``1e-14`` -- instead of perturbing the model
      through ``rho``/``delta``.  The shift is then undone by iterative
      refinement against the true ``M``.

    ``solve`` verifies the first solve after every factorization by measuring
    the residual of the normal equations, and refines it.  That is what turns a
    silently wrong factorization into a reported failure.
    """

    #: diagonal shift ladder used on the *scaled* matrix (unit diagonal)
    _SHIFTS = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)

    def __init__(
        self,
        A: sp.csc_matrix,
        order: str = "best",
        refine: int = 2,
        refine_tol: float = 1e-10,
        fail_tol: float = 1e-6,
        extract_low_rank: bool = True,
        low_rank_max_columns: int = 8,
        low_rank_min_nnz: int = 1024,
        low_rank_clique_ratio: float = 2.0,
    ):
        if not HAVE_CHOLMOD:
            raise RuntimeError("scikit-sparse / CHOLMOD is required for kkt='normal'")
        self.A = A.tocsc()
        self.A.sum_duplicates()
        self.A.sort_indices()
        self.At = self.A.T.tocsr()
        self.m, self.n = A.shape
        self.low_rank_columns = (
            _low_rank_normal_columns(
                self.A,
                max_columns=int(low_rank_max_columns),
                min_nnz=int(low_rank_min_nnz),
                clique_ratio=float(low_rank_clique_ratio),
            )
            if extract_low_rank
            else np.empty(0, dtype=np.int64)
        )
        factor_mask = np.ones(self.n, dtype=bool)
        factor_mask[self.low_rank_columns] = False
        self.factor_columns = np.flatnonzero(factor_mask)
        self.A_factor = self.A[:, self.factor_columns].tocsc()
        self.A_low_rank = self.A[:, self.low_rank_columns].tocsc()
        # column index of every stored nonzero (for fast column scaling)
        self.col_of_nz = np.repeat(
            np.arange(self.A_factor.shape[1]), np.diff(self.A_factor.indptr)
        )
        self.Adata = self.A_factor.data
        self.Aidx = self.A_factor.indices
        self.Aptr = self.A_factor.indptr
        self.I = sp.eye(self.m, format="csc")
        self._factor = None
        self._pattern = None
        self._col_of_nz_M = None
        self.order = order
        self.refine = max(0, int(refine))
        self.refine_tol = float(refine_tol)
        self.fail_tol = float(fail_tol)
        self.theta = None
        self.delta = 0.0
        self._dsc = None  # Jacobi scaling of M, diag(M)^{-1/2}
        self._shift = 0.0  # diagonal shift that worked last time
        self._verify = False
        self._low_rank_basis = None
        self._low_rank_chol = None
        self.n_shifted = 0  # factorizations that needed a shift
        self.n_refine = 0  # refinement steps actually taken
        self.stats = {}  # CHOLMOD diagnostics, {"not_pd": count}
        self.n_symbolic = 0

    def _assemble(self, theta: np.ndarray, delta: float) -> sp.csc_matrix:
        factor_theta = theta[self.factor_columns]
        d = np.sqrt(1.0 / factor_theta)
        Ad = sp.csc_matrix(
            (self.Adata * d[self.col_of_nz], self.Aidx, self.Aptr),
            shape=self.A_factor.shape,
        )
        M = (Ad @ Ad.T + delta * self.I).tocsc()
        M.sort_indices()
        return M

    def factorize(self, theta: np.ndarray, delta: float) -> None:
        t0 = time.perf_counter()
        M = self._assemble(theta, delta)
        fresh = self._factor is None or not (
            M.indptr.shape == self._pattern[0].shape
            and np.array_equal(M.indptr, self._pattern[0])
            and np.array_equal(M.indices, self._pattern[1])
        )
        if fresh:
            self._col_of_nz_M = np.repeat(np.arange(self.m), np.diff(M.indptr))

        # ---- symmetric Jacobi scaling:  Ms = D M D,  D = diag(M)^{-1/2} ---- #
        diag = M.diagonal()
        dsc = 1.0 / np.sqrt(np.where(diag > 0.0, diag, 1.0))
        M.data *= dsc[M.indices] * dsc[self._col_of_nz_M]

        # start just below the shift that was needed last time, so that a
        # one-off perturbation does not stay for the rest of the solve
        start = 0
        if self._shift > 0.0:
            start = max(
                0, int(np.searchsorted(self._SHIFTS, self._shift, side="left")) - 1
            )
        last_err = None
        for k in range(start, len(self._SHIFTS)):
            eta = self._SHIFTS[k]
            Ms = M if eta == 0.0 else (M + eta * self.I).tocsc()
            if eta:
                Ms.sort_indices()
            try:
                if fresh:
                    self._factor = _CholFactor(Ms, order=self.order, stats=self.stats)
                    self._pattern = (M.indptr.copy(), M.indices.copy())
                    self.n_symbolic += 1
                else:
                    self._factor.refactor(Ms)
            except Exception as exc:  # not positive definite / out of memory
                last_err = exc
                self._factor = None if fresh else self._factor
                continue
            self._shift = eta
            if eta:
                self.n_shifted += 1
            break
        else:
            self.t_fact += time.perf_counter() - t0
            raise KKTSolveError(f"Cholesky failed even with a 1e-2 shift ({last_err})")

        self._dsc = dsc
        self.theta = theta
        self.delta = delta
        try:
            self._prepare_low_rank_correction(theta)
        except Exception:
            self.t_fact += time.perf_counter() - t0
            raise
        self._verify = True
        self.n_fact += 1
        self.t_fact += time.perf_counter() - t0

    # -- M applied exactly, i.e. with the true delta and no diagonal shift --- #
    def _matvec(self, v: np.ndarray) -> np.ndarray:
        return self.A @ ((self.At @ v) / self.theta) + self.delta * v

    def _apply_base(self, r: np.ndarray) -> np.ndarray:
        return self._dsc * self._factor.solve(self._dsc * r)

    def _prepare_low_rank_correction(self, theta: np.ndarray) -> None:
        """Factor the small exact Woodbury core for the extracted columns."""
        count = self.low_rank_columns.size
        if not count:
            self._low_rank_basis = None
            self._low_rank_chol = None
            return

        basis = np.empty((self.m, count), dtype=float, order="F")
        rhs = np.zeros(self.m, dtype=float)
        for local_column in range(count):
            rhs.fill(0.0)
            start = self.A_low_rank.indptr[local_column]
            end = self.A_low_rank.indptr[local_column + 1]
            rows = self.A_low_rank.indices[start:end]
            rhs[rows] = self.A_low_rank.data[start:end]
            basis[:, local_column] = self._apply_base(rhs)

        core = np.asarray(self.A_low_rank.T @ basis, dtype=float)
        core.flat[:: count + 1] += theta[self.low_rank_columns]
        core = 0.5 * (core + core.T)

        # The core is theoretically SPD.  A relative shift is only a guard for
        # floating-point asymmetry after a shifted or highly ill-conditioned
        # sparse factorization; outer iterative refinement uses the true M.
        scale = max(float(np.max(np.abs(np.diag(core)))), 1.0)
        last_err = None
        for relative_shift in (0.0, 1e-14, 1e-12, 1e-10, 1e-8):
            try:
                shifted = (
                    core
                    if relative_shift == 0.0
                    else core + relative_shift * scale * np.eye(count)
                )
                chol = np.linalg.cholesky(shifted)
            except np.linalg.LinAlgError as exc:
                last_err = exc
                continue
            self._low_rank_basis = basis
            self._low_rank_chol = chol
            return
        raise KKTSolveError(f"low-rank correction is not positive definite ({last_err})")

    def _apply(self, r: np.ndarray) -> np.ndarray:
        result = self._apply_base(r)
        if self._low_rank_chol is None:
            return result
        small_rhs = np.asarray(self.A_low_rank.T @ result).ravel()
        coefficient = np.linalg.solve(self._low_rank_chol, small_rhs)
        coefficient = np.linalg.solve(self._low_rank_chol.T, coefficient)
        return result - self._low_rank_basis @ coefficient

    def solve(self, r1: np.ndarray, r2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        w = r1 / self.theta
        rhs = r2 - self.A @ w
        dy = self._apply(rhs)

        # Verify (and refine) the first solve after each factorization: it is
        # the same triangular factor for every later right-hand side, so one
        # check per factorization is enough to catch a bad one, and the two
        # extra matrix-vector products stay negligible next to the Cholesky.
        if self._verify and self.refine:
            self._verify = False
            nrhs = float(np.linalg.norm(rhs))
            if nrhs > 0.0:
                nres = np.inf
                for _ in range(self.refine):
                    res = rhs - self._matvec(dy)
                    nres = float(np.linalg.norm(res))
                    if not np.isfinite(nres) or nres <= self.refine_tol * nrhs:
                        break
                    dy = dy + self._apply(res)
                    self.n_refine += 1
                    nres = np.inf  # unknown again until re-measured
                if not np.isfinite(nres):
                    nres = float(np.linalg.norm(rhs - self._matvec(dy)))
                if not np.isfinite(nres) or nres > self.fail_tol * nrhs:
                    self.t_solve += time.perf_counter() - t0
                    raise KKTSolveError(
                        f"normal equations residual {nres:.2e} vs rhs {nrhs:.2e}"
                    )

        dx = w + (self.At @ dy) / self.theta
        self.n_solve += 1
        self.t_solve += time.perf_counter() - t0
        return dx, dy

    @property
    def nnz_factor(self):
        return int(self._factor.nnz) if self._factor is not None else 0


def _column_weighted_normal(A: sp.csc_matrix, weights: np.ndarray):
    """Return ``A diag(weights) A.T`` without constructing the diagonal."""
    columns = np.repeat(np.arange(A.shape[1]), np.diff(A.indptr))
    Ad = sp.csc_matrix(
        (A.data * np.sqrt(weights)[columns], A.indices, A.indptr),
        shape=A.shape,
    )
    return (Ad @ Ad.T).tocsc()


class DynamicRegularizedNormalEquations(NormalEquations):
    r"""Factor a sparsified Schur complement with certified regularization.

    Columns whose barrier weight ``d_j = 1 / theta_j`` satisfies

        d_j || |A| |A|^T ||_inf <= reg_threshold

    are eliminated from the off-diagonal normal matrix.  Their exact diagonal
    contribution is retained, and ``max(d_N) || |A| |A|^T ||_inf I`` is added
    to the dual regularization.  This is the implementable conservative form
    of dynamic non-diagonal regularization: the dropped coupling is bounded in
    infinity norm, while the useful diagonal survives.
    """

    def __init__(
        self,
        A,
        tol,
        reg_factor=0.1,
        update_interval=2,
        **kwargs,
    ):
        # This class assembles an approximate full normal matrix itself; its
        # exact fallback still uses NormalEquations' low-rank correction.
        kwargs["extract_low_rank"] = False
        super().__init__(A, **kwargs)
        self.A2 = self.A.copy()
        self.A2.data **= 2
        self.absA = abs(self.A)
        col_l1 = np.asarray(self.absA.sum(axis=0)).ravel()
        row_bound = np.asarray(self.absA @ col_l1).ravel()
        self.norm_bound = max(float(row_bound.max(initial=0.0)), 1.0)
        self.reg_threshold = max(float(reg_factor) * float(tol), 1e-14)
        self.weight_threshold = self.reg_threshold / self.norm_bound
        self.update_interval = max(1, int(update_interval))
        self.active = np.ones(self.n, dtype=bool)
        self._approx_weights = None
        self._diag_drop = np.zeros(self.m)
        self._dynamic_shift = 0.0
        self.core_sizes = []
        self.dynamic_shifts = []
        self.full_fallbacks = 0
        self._fallback = None
        self._fallback_ready = False

    def _assemble(self, theta: np.ndarray, delta: float) -> sp.csc_matrix:
        weights = 1.0 / theta
        # Re-entry is immediate to preserve the perturbation bound.  Elimination
        # is batched so a slowly moving barrier does not trigger symbolic work at
        # every factorization.
        self.active[weights > self.weight_threshold] = True
        if self.n_fact % self.update_interval == 0:
            self.active = weights > self.weight_threshold

        core = np.flatnonzero(self.active)
        omitted = ~self.active
        if core.size:
            Ac = self.A[:, core].tocsc()
            M = _column_weighted_normal(Ac, weights[core])
        else:
            M = sp.csc_matrix((self.m, self.m))

        diag_drop = np.asarray(self.A2 @ (weights * omitted)).ravel()
        dynamic_shift = (
            float(weights[omitted].max(initial=0.0)) * self.norm_bound
            if omitted.any()
            else 0.0
        )
        M = (
            M
            + sp.diags(diag_drop + delta + dynamic_shift, format="csc")
        ).tocsc()
        M.sort_indices()
        self._approx_weights = weights
        self._diag_drop = diag_drop
        self._dynamic_shift = dynamic_shift
        self.core_sizes.append(int(core.size))
        self.dynamic_shifts.append(dynamic_shift)
        return M

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.A @ (self._approx_weights * (self.At @ v)) + self.delta * v
        ).ravel()

    def factorize(self, theta: np.ndarray, delta: float) -> None:
        super().factorize(theta, delta)
        self._fallback_ready = False

    def _fallback_solve(self, r1, r2):
        self.full_fallbacks += 1
        if self._fallback is None:
            self._fallback = NormalEquations(
                self.A,
                order=self.order,
                refine=self.refine,
                refine_tol=self.refine_tol,
                fail_tol=self.fail_tol,
            )
        if not self._fallback_ready:
            before = self._fallback.t_fact
            self._fallback.factorize(self.theta, self.delta)
            self.t_fact += self._fallback.t_fact - before
            self._fallback_ready = True
        return self._fallback.solve(r1, r2)

    def solve(self, r1: np.ndarray, r2: np.ndarray):
        """Correct the sparsified solve against the exact full Schur matrix."""
        t0 = time.perf_counter()
        fact0 = self.t_fact
        w = r1 / self.theta
        rhs = np.asarray(r2 - self.A @ w).ravel()
        dy = self._apply(rhs)
        nrhs = max(float(np.linalg.norm(rhs)), 1.0)
        relres = np.inf
        for _ in range(max(1, self.refine)):
            residual = rhs - self._matvec(dy)
            relres = float(np.linalg.norm(residual)) / nrhs
            if not np.isfinite(relres) or relres <= self.refine_tol:
                break
            dy += self._apply(residual)
            self.n_refine += 1
        if not np.isfinite(relres) or relres > self.fail_tol:
            dx, dy = self._fallback_solve(r1, r2)
        else:
            dx = w + (self.At @ dy) / self.theta
        self.n_solve += 1
        elapsed = time.perf_counter() - t0
        self.t_solve += max(0.0, elapsed - (self.t_fact - fact0))
        return dx, dy


class CorePCGNormalEquations(NormalEquations):
    r"""Solve the full Schur system by PCG with a generated-column core.

    The factorized preconditioner is

        P = A_C D_C A_C^T + diag(A_N D_N A_N^T) + delta I,

    where ``C`` is a leverage-proxy column sample.  A sampled column is
    reweighted by ``1 / p_j``; consequently the sampled normal matrix is an
    unbiased estimator of the full matrix rather than an unscaled truncation.
    PCG applies the *full* ``A D A^T + delta I`` operator, so core elimination
    does not perturb the Newton equation or the LP optimality conditions.
    """

    def __init__(
        self,
        A,
        core_factor=2.0,
        update_interval=6,
        cg_tol=1e-8,
        cg_maxit=100,
        **kwargs,
    ):
        # The sampled preconditioner has its own column representation.
        kwargs["extract_low_rank"] = False
        super().__init__(A, **kwargs)
        self.A2 = self.A.copy()
        self.A2.data **= 2
        self.core_target = min(
            self.n, max(1, int(np.ceil(float(core_factor) * self.m)))
        )
        self.update_interval = max(1, int(update_interval))
        self.cg_tol = float(cg_tol)
        self.cg_maxit = max(1, int(cg_maxit))
        self.core = None
        self.core_probability = None
        self._weights = None
        self._diag_drop = np.zeros(self.m)
        self.core_sizes = []
        self.cg_iterations = []
        self.cg_failures = 0
        self.full_fallbacks = 0
        self._fallback = None
        self._fallback_ready = False
        self._sample_count = 0

    def _choose_core(self, weights, delta):
        if self.core_target >= self.n:
            return np.arange(self.n, dtype=np.int64), np.ones(self.n)
        diagonal = np.asarray(self.A2 @ weights).ravel() + delta
        inverse_diagonal = 1.0 / np.maximum(diagonal, np.finfo(float).tiny)
        scores = weights * np.asarray(self.A2.T @ inverse_diagonal).ravel()

        # Choose alpha so sum(min(1, alpha * score_j)) equals the target sample
        # size.  These are exact column-norm scores after Jacobi row scaling;
        # their sum is at most m and they are inexpensive to refresh.
        high = 1.0
        while float(np.minimum(1.0, high * scores).sum()) < self.core_target:
            high *= 2.0
        low = 0.0
        for _ in range(40):
            middle = 0.5 * (low + high)
            if float(np.minimum(1.0, middle * scores).sum()) < self.core_target:
                low = middle
            else:
                high = middle
        probability = np.minimum(1.0, high * scores)
        rng = np.random.default_rng(1729 + self._sample_count)
        self._sample_count += 1
        selected = np.flatnonzero(rng.random(self.n) < probability)
        if not selected.size:
            selected = np.array([int(np.argmax(scores))], dtype=np.int64)
            probability[selected] = 1.0
        return selected, probability[selected]

    def _assemble(self, theta: np.ndarray, delta: float) -> sp.csc_matrix:
        weights = 1.0 / theta
        if self.core is None or self.n_fact % self.update_interval == 0:
            self.core, self.core_probability = self._choose_core(weights, delta)
        core = self.core
        Ac = self.A[:, core].tocsc()
        sampled_weights = weights[core] / np.maximum(
            self.core_probability, np.finfo(float).tiny
        )
        M = _column_weighted_normal(Ac, sampled_weights)
        diag_full = np.asarray(self.A2 @ weights).ravel()
        diag_core = np.asarray(self.A2[:, core] @ sampled_weights).ravel()
        self._diag_drop = np.maximum(diag_full - diag_core, 0.0)
        M = (M + sp.diags(self._diag_drop + delta, format="csc")).tocsc()
        M.sort_indices()
        self._weights = weights
        self.core_sizes.append(int(core.size))
        return M

    def factorize(self, theta: np.ndarray, delta: float) -> None:
        super().factorize(theta, delta)
        self._fallback_ready = False

    def _full_matvec(self, v):
        return np.asarray(
            self.A @ (self._weights * (self.At @ v)) + self.delta * v
        ).ravel()

    def _fallback_solve(self, r1, r2):
        self.full_fallbacks += 1
        if self._fallback is None:
            self._fallback = NormalEquations(
                self.A,
                order=self.order,
                refine=self.refine,
                refine_tol=self.refine_tol,
                fail_tol=self.fail_tol,
            )
        if not self._fallback_ready:
            before = self._fallback.t_fact
            self._fallback.factorize(self.theta, self.delta)
            self.t_fact += self._fallback.t_fact - before
            self._fallback_ready = True
        return self._fallback.solve(r1, r2)

    def solve(self, r1: np.ndarray, r2: np.ndarray):
        t0 = time.perf_counter()
        fact0 = self.t_fact
        w = r1 / self.theta
        rhs = np.asarray(r2 - self.A @ w).ravel()
        iterations = 0

        def count(_):
            nonlocal iterations
            iterations += 1

        operator = spla.LinearOperator((self.m, self.m), matvec=self._full_matvec)
        preconditioner = spla.LinearOperator(
            (self.m, self.m), matvec=self._apply
        )
        x0 = self._apply(rhs)
        dy, info = spla.cg(
            operator,
            rhs,
            x0=x0,
            M=preconditioner,
            rtol=self.cg_tol,
            atol=0.0,
            maxiter=self.cg_maxit,
            callback=count,
        )
        nrhs = max(float(np.linalg.norm(rhs)), 1.0)
        relres = float(np.linalg.norm(rhs - self._full_matvec(dy))) / nrhs
        self.cg_iterations.append(iterations)
        if info != 0 or not np.isfinite(relres) or relres > self.fail_tol:
            self.cg_failures += 1
            dx, dy = self._fallback_solve(r1, r2)
        else:
            dx = w + (self.At @ dy) / self.theta
        self.n_solve += 1
        elapsed = time.perf_counter() - t0
        self.t_solve += max(0.0, elapsed - (self.t_fact - fact0))
        return dx, dy


class LaggedPCGNormalEquations(NormalEquations):
    r"""Exact Schur solves with low-frequency factor updates.

    A reference factorization ``P = A D_ref A^T + delta_ref I`` is retained
    for several Newton systems.  Between refreshes, changed column weights are
    generated implicitly in the full operator

        M = P + A (D - D_ref) A^T + (delta - delta_ref) I,

    and PCG solves ``M dy = rhs`` exactly to the requested tolerance.  Thus the
    changing columns never enter a new Cholesky pattern, and a refresh is forced
    when either its age or the observed Krylov work becomes too large.
    """

    def __init__(
        self,
        A,
        max_age=8,
        refresh_cg=20,
        cg_tol=1e-8,
        cg_maxit=80,
        **kwargs,
    ):
        super().__init__(A, **kwargs)
        self.max_age = max(1, int(max_age))
        self.refresh_cg = max(1, int(refresh_cg))
        self.cg_tol = float(cg_tol)
        self.cg_maxit = max(1, int(cg_maxit))
        self.age = self.max_age
        self.reference_theta = None
        self.reference_delta = None
        self.cg_iterations = []
        self.cg_failures = 0
        self.full_fallbacks = 0
        self.refreshes = 0
        self.changed_columns = []
        self._cycle_cg_max = 0

    def _refresh(self, theta, delta):
        NormalEquations.factorize(self, theta, delta)
        self.reference_theta = theta.copy()
        self.reference_delta = float(delta)
        self.age = 0
        self._cycle_cg_max = 0
        self.refreshes += 1

    def factorize(self, theta: np.ndarray, delta: float) -> None:
        theta = np.asarray(theta, dtype=float)
        refresh = (
            self._factor is None
            or self.age >= self.max_age
            or self._cycle_cg_max >= self.refresh_cg
        )
        if refresh:
            self._refresh(theta, delta)
        else:
            ratio = theta / self.reference_theta
            changed = np.count_nonzero((ratio < 0.5) | (ratio > 2.0))
            self.changed_columns.append(int(changed))
            self.theta = theta
            self.delta = float(delta)
            self.age += 1

    def solve(self, r1: np.ndarray, r2: np.ndarray):
        t0 = time.perf_counter()
        fact0 = self.t_fact
        solve0 = self.t_solve
        w = r1 / self.theta
        rhs = np.asarray(r2 - self.A @ w).ravel()
        iterations = 0

        def count(_):
            nonlocal iterations
            iterations += 1

        operator = spla.LinearOperator((self.m, self.m), matvec=self._matvec)
        preconditioner = spla.LinearOperator(
            (self.m, self.m), matvec=self._apply
        )
        x0 = self._apply(rhs)
        dy, info = spla.cg(
            operator,
            rhs,
            x0=x0,
            M=preconditioner,
            rtol=self.cg_tol,
            atol=0.0,
            maxiter=self.cg_maxit,
            callback=count,
        )
        nrhs = max(float(np.linalg.norm(rhs)), 1.0)
        relres = float(np.linalg.norm(rhs - self._matvec(dy))) / nrhs
        self.cg_iterations.append(iterations)
        self._cycle_cg_max = max(self._cycle_cg_max, iterations)

        failed = info != 0 or not np.isfinite(relres) or relres > self.fail_tol
        expensive = iterations >= self.refresh_cg
        if failed or expensive:
            self.cg_failures += int(failed)
            self.full_fallbacks += 1
            self._refresh(self.theta, self.delta)
            dx, dy = NormalEquations.solve(self, r1, r2)
            parent_solve = self.t_solve - solve0
        else:
            dx = w + (self.At @ dy) / self.theta
            parent_solve = 0.0
            self.n_solve += 1
        elapsed = time.perf_counter() - t0
        extra = elapsed - (self.t_fact - fact0) - parent_solve
        self.t_solve += max(0.0, extra)
        return dx, dy


class AugmentedSystem(_KKTBase):
    """LDL^T of the quasi-definite augmented matrix (the MATLAB path).

    ``K = [[Theta, A^T], [A, -delta I]]``; only the upper triangle is stored.
    QDLDL performs the AMD ordering + symbolic analysis once and then only
    updates the numerical values.
    """

    def __init__(self, A: sp.csc_matrix):
        if not HAVE_QDLDL:
            raise RuntimeError("qdldl is required for kkt='augmented'")
        self.A = A.tocsc()
        self.m, self.n = A.shape
        N = self.m + self.n
        K = sp.bmat(
            [[sp.diags(np.ones(self.n)), sp.csc_matrix(self.A.T)],
             [None, -sp.eye(self.m)]],
            format="csc",
        )
        K.sort_indices()
        self.K = K
        self._solver = None
        # positions of the (1,1) and (2,2) diagonal entries in K.data
        # K is stored upper-triangular with sorted indices, so the diagonal
        # entry of column j is the last stored entry of that column.
        self.diag_pos = K.indptr[1:] - 1
        assert np.array_equal(K.indices[self.diag_pos], np.arange(N))

    def factorize(self, theta: np.ndarray, delta: float) -> None:
        t0 = time.perf_counter()
        self.K.data[self.diag_pos[: self.n]] = theta
        self.K.data[self.diag_pos[self.n:]] = -delta
        if self._solver is None:
            self._solver = _qdldl.Solver(self.K, upper=True)
        else:
            self._solver.update(self.K, upper=True)
        self.n_fact += 1
        self.t_fact += time.perf_counter() - t0

    def solve(self, r1: np.ndarray, r2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        sol = self._solver.solve(np.concatenate([r1, r2]))
        dx = sol[: self.n]
        dy = -sol[self.n:]
        self.n_solve += 1
        self.t_solve += time.perf_counter() - t0
        return dx, dy

    @property
    def nnz_factor(self):
        return -1


class PartialCondensation(_KKTBase):
    r"""Exact fill-aware partial condensation.

    Let ``C`` be columns kept explicit and ``L`` their complement.  Eliminating
    only ``dx_L`` from the regularized Newton equations gives

        [ Theta_C    A_C.T ] [ dx_C]   [r1_C             ]
        [ A_C          -S_L ] [   q ] = [r2 - A_L r1_L/t],

    where ``q = -dy`` and

        S_L = delta I + A_L diag(1 / theta_L) A_L.T.

    This is an exact symmetric quasi-definite system.  QDLDL factorizes a
    symmetrically scaled version, and every accepted direction is checked in
    the original full Newton equations.

    The explicit columns are selected once, after a short central-path warmup.
    CHOLMOD nested dissection estimates each row's separator/factor burden.  A
    column's structural score is then tilted toward a bound when the current
    multiplier is large and its complementarity product is close to ``mu``.
    This is the computational use of the draft's central-screening insight:
    likely bound-active columns are represented explicitly, not deleted.  The
    representation therefore remains exact even when the activity prediction
    is wrong.

    A symbolic gate compares the predicted factor storage of ``S_L`` plus the
    explicit coupling block with the full normal equations.  If the requested
    improvement is not attained, the class permanently delegates to the full
    CHOLMOD normal-equations solver.
    """

    def __init__(
        self,
        A,
        pos=None,
        box=None,
        explicit_columns=256,
        warmup=4,
        pool_factor=8.0,
        draft_weight=1.0,
        min_fill_ratio=2.0,
        order="best",
        refine=2,
        refine_tol=1e-10,
        fail_tol=1e-6,
    ):
        if not HAVE_CHOLMOD:
            raise RuntimeError("CHOLMOD is required for kkt='partial'")
        if not HAVE_QDLDL:
            raise RuntimeError("QDLDL is required for kkt='partial'")
        self.A = A.tocsc()
        self.A.sort_indices()
        self.At = self.A.T.tocsr()
        self.m, self.n = self.A.shape
        self.pos = (
            np.ones(self.n, dtype=bool)
            if pos is None else np.asarray(pos, dtype=bool)
        )
        self.box = (
            np.zeros(self.n, dtype=bool)
            if box is None else np.asarray(box, dtype=bool)
        )
        self.explicit_target = min(
            self.n - 1, max(0, int(explicit_columns))
        )
        self.warmup = max(1, int(warmup))
        self.pool_factor = max(1.0, float(pool_factor))
        self.draft_weight = max(0.0, float(draft_weight))
        self.min_fill_ratio = max(0.0, float(min_fill_ratio))
        self.order = order
        self.refine = max(0, int(refine))
        self.refine_tol = float(refine_tol)
        self.fail_tol = float(fail_tol)

        self._full = NormalEquations(
            self.A,
            order=order,
            refine=refine,
            refine_tol=refine_tol,
            fail_tol=fail_tol,
        )
        self.stats = self._full.stats
        self._factor_calls = 0
        self._partition_ready = False
        self._hybrid_active = False
        self._rejected = False
        self._multiplier = None
        self._centrality = None
        self._solver = None
        self._pattern = None
        self._dsc = None
        self._verify = False
        self.theta = None
        self.delta = 0.0
        self.explicit = np.empty(0, dtype=np.int64)
        self.condensed = np.arange(self.n, dtype=np.int64)
        self.AC = None
        self.AL = self.A
        self.symbolic_full_nnz = 0
        self.symbolic_condensed_nnz = 0
        self.symbolic_hybrid_nnz = 0
        self.symbolic_fill_ratio = 1.0
        self.symbolic_seconds = 0.0
        self.partition_rebuilds = 0
        self.n_refine = 0
        self.n_shifted = 0

    def observe(self, x, z, w, s, mu):
        """Store the draft-inspired bound-activity signal for the next epoch."""
        multiplier = np.asarray(z, dtype=float).copy()
        product = np.asarray(x, dtype=float) * multiplier
        if self.box.any():
            upper = np.asarray(w, dtype=float)
            choose_upper = self.box & (upper > multiplier)
            multiplier[choose_upper] = upper[choose_upper]
            product[choose_upper] = (
                np.asarray(s, dtype=float)[choose_upper] * upper[choose_upper]
            )
        multiplier[~self.pos] = 0.0
        scale = max(float(mu), np.finfo(float).tiny)
        ratio = np.maximum(product / scale, np.finfo(float).tiny)
        # Reliability is one on the central path and decreases symmetrically
        # when the complementarity product is above or below mu.
        self._multiplier = np.maximum(multiplier, 0.0)
        self._centrality = np.exp(-np.abs(np.log(ratio)))

    @staticmethod
    def _symbolic_stats(G, permutation):
        P = G[permutation, :][:, permutation].tocsc()
        result = _cholmod.symbfact(P)
        return int(result.count.sum()), np.asarray(result.count)

    def _select_partition(self, theta):
        t0 = time.perf_counter()
        q = self.explicit_target
        if q == 0:
            self._rejected = True
            self.symbolic_seconds = time.perf_counter() - t0
            return

        pattern = self.A.copy()
        pattern.data = np.ones_like(pattern.data)
        graph = (pattern @ pattern.T).tocsc()
        graph.data = np.ones_like(graph.data)
        graph.setdiag(1.0)
        graph.eliminate_zeros()
        graph.sort_indices()

        try:
            permutation, tree = _cholmod.nesdis(
                graph, return_separator=True, nd_small=200
            )
            full_nnz, counts = self._symbolic_stats(graph, permutation)
            parent = np.asarray(tree.cp)
            depth = np.zeros(parent.size, dtype=float)
            for node in range(parent.size):
                p = node
                while parent[p] >= 0:
                    depth[node] += 1.0
                    p = parent[p]
            row_depth = depth[np.asarray(tree.cmember)]
            burden = np.empty(self.m)
            burden[permutation] = counts
            row_score = burden / (1.0 + row_depth)
        except Exception:
            permutation = np.arange(self.m)
            full_nnz = int(np.diff(graph.indptr).sum())
            row_score = np.asarray(graph.sum(axis=0)).ravel()

        structural = np.asarray(pattern.T @ row_score).ravel()
        pool_size = min(
            self.n, max(q, int(np.ceil(self.pool_factor * q)))
        )
        if pool_size < self.n:
            pool = np.argpartition(structural, -pool_size)[-pool_size:]
        else:
            pool = np.arange(self.n)

        multiplier = self._multiplier
        if multiplier is None or multiplier.size != self.n:
            multiplier = np.asarray(theta, dtype=float)
            centrality = np.ones(self.n)
        else:
            centrality = self._centrality
        positive = multiplier[pool][multiplier[pool] > 0.0]
        reference = float(np.median(positive)) if positive.size else 1.0
        numerical = np.log1p(multiplier[pool] / max(reference, 1e-300))
        numerical *= np.sqrt(np.maximum(centrality[pool], 0.0))
        combined = structural[pool] * (1.0 + self.draft_weight * numerical)
        chosen = pool[np.argpartition(combined, -q)[-q:]]
        chosen.sort()

        keep = np.ones(self.n, dtype=bool)
        keep[chosen] = False
        condensed = np.flatnonzero(keep)
        reduced_pattern = pattern[:, condensed]
        reduced_graph = (reduced_pattern @ reduced_pattern.T).tocsc()
        reduced_graph.data = np.ones_like(reduced_graph.data)
        reduced_graph.setdiag(1.0)
        reduced_graph.eliminate_zeros()
        reduced_graph.sort_indices()
        try:
            reduced_perm = _cholmod.metis(reduced_graph)
            reduced_nnz, _ = self._symbolic_stats(
                reduced_graph, reduced_perm
            )
        except Exception:
            reduced_nnz, _ = self._symbolic_stats(
                reduced_graph, permutation
            )

        coupling_nnz = int(self.A[:, chosen].nnz) + int(q)
        hybrid_nnz = reduced_nnz + coupling_nnz
        ratio = full_nnz / max(hybrid_nnz, 1)
        self.symbolic_full_nnz = full_nnz
        self.symbolic_condensed_nnz = reduced_nnz
        self.symbolic_hybrid_nnz = hybrid_nnz
        self.symbolic_fill_ratio = float(ratio)
        self.symbolic_seconds = time.perf_counter() - t0
        self._partition_ready = True

        if ratio < self.min_fill_ratio:
            self._rejected = True
            return

        self.explicit = chosen
        self.condensed = condensed
        self.AC = self.A[:, chosen].tocsc()
        self.AL = self.A[:, condensed].tocsc()
        self._hybrid_active = True

    def _factorize_hybrid(self, theta, delta):
        weights = 1.0 / theta[self.condensed]
        S = (
            _column_weighted_normal(self.AL, weights)
            + delta * sp.eye(self.m, format="csc")
        ).tocsc()
        S.sort_indices()
        upper_S = sp.triu(S, format="csc")
        K = sp.bmat(
            [
                [sp.diags(theta[self.explicit], format="csc"), self.AC.T],
                [None, -upper_S],
            ],
            format="csc",
        )
        K.sort_indices()
        diagonal = np.concatenate(
            [theta[self.explicit], np.maximum(S.diagonal(), 1e-300)]
        )
        dsc = 1.0 / np.sqrt(diagonal)
        col_of_nz = np.repeat(np.arange(K.shape[1]), np.diff(K.indptr))
        K.data *= dsc[K.indices] * dsc[col_of_nz]

        fresh = self._solver is None or not (
            K.indptr.shape == self._pattern[0].shape
            and np.array_equal(K.indptr, self._pattern[0])
            and np.array_equal(K.indices, self._pattern[1])
        )
        if fresh:
            self._solver = _qdldl.Solver(K, upper=True)
            self._pattern = (K.indptr.copy(), K.indices.copy())
            self.partition_rebuilds += 1
        else:
            try:
                self._solver.update(K, upper=True)
            except Exception:
                self._solver = _qdldl.Solver(K, upper=True)
                self.partition_rebuilds += 1
        self._dsc = dsc
        self.theta = np.asarray(theta)
        self.delta = float(delta)
        self._verify = True

    def factorize(self, theta, delta):
        t0 = time.perf_counter()
        self._factor_calls += 1
        if (
            not self._partition_ready
            and not self._rejected
            and self._factor_calls >= self.warmup
        ):
            self._select_partition(theta)

        if self._hybrid_active:
            try:
                self._factorize_hybrid(theta, delta)
            except Exception:
                # Keep the experiment robust: a failed prototype factor does
                # not make the LP fail when the established full path works.
                self._hybrid_active = False
                self._rejected = True
                self._full.factorize(theta, delta)
        else:
            self._full.factorize(theta, delta)
            self.theta = np.asarray(theta)
            self.delta = float(delta)
            self.n_shifted = self._full.n_shifted
            self.n_refine = self._full.n_refine
        self.n_fact += 1
        self.t_fact += time.perf_counter() - t0

    def _solve_hybrid_once(self, r1, r2):
        q = self.explicit.size
        wL = r1[self.condensed] / self.theta[self.condensed]
        rhs = np.concatenate(
            [r1[self.explicit], np.asarray(r2 - self.AL @ wL).ravel()]
        )
        sol = self._dsc * self._solver.solve(self._dsc * rhs)
        dx = np.empty(self.n)
        dx[self.explicit] = sol[:q]
        dy = -sol[q:]
        dx[self.condensed] = wL + (
            self.AL.T @ dy
        ) / self.theta[self.condensed]
        return dx, dy

    def solve(self, r1, r2):
        t0 = time.perf_counter()
        if not self._hybrid_active:
            dx, dy = self._full.solve(r1, r2)
            self.n_refine = self._full.n_refine
        else:
            dx, dy = self._solve_hybrid_once(r1, r2)
            if self._verify and self.refine:
                self._verify = False
                nrhs = max(
                    float(np.linalg.norm(r1)) + float(np.linalg.norm(r2)), 1.0
                )
                nres = np.inf
                for _ in range(self.refine):
                    e1 = r1 - (self.theta * dx - self.At @ dy)
                    e2 = r2 - (self.A @ dx + self.delta * dy)
                    nres = float(np.linalg.norm(e1) + np.linalg.norm(e2))
                    if not np.isfinite(nres) or nres <= self.refine_tol * nrhs:
                        break
                    ddx, ddy = self._solve_hybrid_once(e1, e2)
                    dx += ddx
                    dy += ddy
                    self.n_refine += 1
                    nres = np.inf
                if not np.isfinite(nres):
                    e1 = r1 - (self.theta * dx - self.At @ dy)
                    e2 = r2 - (self.A @ dx + self.delta * dy)
                    nres = float(np.linalg.norm(e1) + np.linalg.norm(e2))
                if not np.isfinite(nres) or nres > self.fail_tol * nrhs:
                    raise KKTSolveError(
                        f"partial KKT residual {nres:.2e} vs rhs {nrhs:.2e}"
                    )
        self.n_solve += 1
        self.t_solve += time.perf_counter() - t0
        return dx, dy

    @property
    def nnz_factor(self):
        if self._hybrid_active:
            return -1
        return self._full.nnz_factor


# =========================================================================== #
#  Small helpers
# =========================================================================== #
def _step_to_boundary(v: np.ndarray, dv: np.ndarray, idx) -> float:
    """max{alpha in (0,1] : v + alpha dv >= 0} on the index set ``idx``."""
    vv, dd = v[idx], dv[idx]
    neg = dd < 0.0
    if not neg.any():
        return 1.0
    return float(min(1.0, np.min(-vv[neg] / dd[neg])))


def _all_or_idx(mask: np.ndarray):
    """`slice(None)` when the mask selects everything (avoids fancy-index copies)."""
    return slice(None) if mask.all() else np.flatnonzero(mask)


def _box_floors(u: np.ndarray, ibox) -> Tuple[np.ndarray, np.ndarray]:
    """Smallest usable distance from each of the two bounds of a boxed variable.

    ``u - x`` is not representable to better than ``eps * u``, so a slack below
    that is indistinguishable from zero however carefully it was computed.  The
    floors are additionally capped at ``u / 4`` so that they stay compatible
    with each other on very narrow boxes.
    """
    ub = np.asarray(u[ibox], dtype=float)
    quarter = 0.25 * ub
    s_floor = np.minimum(_POS_FLOOR * np.maximum(1.0, ub), quarter)
    x_floor = np.minimum(_POS_FLOOR, quarter)
    return x_floor, s_floor


def _sync_box(x, s, u, ibox, x_floor, s_floor) -> None:
    """Put ``(x, s)`` back inside the box, in place, keeping both strictly interior.

    The fraction-to-the-boundary rule keeps the *exact* slack positive, but it
    says nothing about the computed ``u - x``, which cancels to exactly zero as
    soon as ``x`` comes within one ulp of ``u`` -- and then ``w / s`` divides by
    zero and Theta fills with infinities and NaNs.  Flooring the slack at the
    last distance from the bound that doubles can still resolve is the faithful
    limit: such a variable is pinned at its upper bound, and the resulting huge
    ``w / s`` entry of Theta is precisely how the Newton system expresses that.
    """
    x[ibox] = np.clip(x[ibox], x_floor, u[ibox] - s_floor)
    s[ibox] = np.maximum(u[ibox] - x[ibox], s_floor)


@dataclass
class Solution:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    w: np.ndarray
    status: str = "unknown"
    obj: float = np.nan
    outer_iter: int = 0
    inner_iter: int = 0
    time: float = 0.0
    t_fact: float = 0.0
    t_solve: float = 0.0
    res_p: float = np.nan
    res_d: float = np.nan
    mu: float = np.nan
    history: list = field(default_factory=list)
    # linear-algebra diagnostics (normal-equations path)
    n_not_pd: int = 0  # CHOLMOD non-positive-definite reports, all recovered
    n_shifted: int = 0  # factorizations that needed a diagonal shift
    n_refine: int = 0  # iterative-refinement steps taken
    low_rank_columns: int = 0
    low_rank_candidate_edges: int = 0
    core_min: int = 0
    core_max: int = 0
    core_mean: float = 0.0
    pcg_iterations: int = 0
    pcg_max_iterations: int = 0
    pcg_failures: int = 0
    full_kkt_fallbacks: int = 0
    preconditioner_refreshes: int = 0
    preconditioner_factorizations: int = 0
    preconditioner_retained_mean: float = 0.0
    preconditioner_symbolic_analyses: int = 0
    preconditioner_retries: int = 0
    preconditioner_full: bool = False
    preconditioner_condition_max: float = 1.0
    preconditioner_spectral_error_max: float = 0.0
    preconditioner_lag_bound_max: float = 1.0
    preconditioner_core_swaps: int = 0
    residual_replacements: int = 0
    residual_checks: int = 0
    krylov_stagnations: int = 0
    minres_fallbacks: int = 0
    minres_iterations: int = 0
    krylov_breakdowns: int = 0
    krylov_iteration_limits: int = 0
    dynamic_shift_max: float = 0.0
    partial_explicit: int = 0
    partial_condensed: int = 0
    partial_fill_ratio: float = 1.0
    partial_symbolic_seconds: float = 0.0
    partial_rebuilds: int = 0
    partial_rejected: bool = False
    # Retrospective column-generation diagnostics.
    cg_iterations: int = 0
    cg_phase1_iterations: int = 0
    cg_columns_added: int = 0
    cg_columns_eliminated: int = 0
    cg_columns_reactivated: int = 0
    cg_pool_remaining: int = 0
    cg_working_columns: int = 0
    cg_max_pricing_violation: float = np.inf
    cg_artificial_value: float = np.inf
    cg_history: list = field(default_factory=list)
    cg_sparse_steps: int = 1
    cg_median_clique_burden: float = 0.0


# =========================================================================== #
#  Inner solver: one proximal evaluation   (prox_eval.m)
# =========================================================================== #
def prox_eval(
    lp_c, A, At, b, u, ipos, ibox, npos_box,
    xk, yk, zk, wk, rho, delta, tol, maxit, pc, kkt, printlevel, deadline=np.inf,
    nbox=0, tau=0.995,
):
    """Approximately minimize the proximal sub-problem

        min c'v + (rho/2)||v - xk||^2 + (delta/2)||y||^2
        s.t. A v + delta (y - yk) = b,  0 <= v <= u

    with a (predictor-corrector) infeasible IPM.  Mirrors ``prox_eval.m``.

    The upper-bound slack ``s = u - x`` is kept strictly positive through
    ``_sync_box``; the bare subtraction cancels to zero near the bound.
    """
    n = A.shape[1]
    x, y, z, w = xk.copy(), yk.copy(), zk.copy(), wk.copy()
    x_floor, s_floor = _box_floors(u, ibox) if nbox else (0.0, 0.0)
    s = np.ones(n)
    if nbox:
        _sync_box(x, s, u, ibox, x_floor, s_floor)

    sigmamin, sigmamax = 0.05, 0.95
    alpha_x = alpha_z = 0.0
    reg_extra = 0.0
    retry, max_tries = 0, 50
    stalled = 0
    res_n_best = np.inf
    iters = 0
    status = 0  # 0 maxit, 1 converged, 2 ill-conditioned

    mu = (
        (x[ipos] @ z[ipos] + s[ibox] @ w[ibox]) / npos_box if npos_box else 0.0
    )
    res_muz = np.zeros(n)
    res_muw = np.zeros(n)

    def residuals(x, y, z, w, s):
        rd = -(lp_c - At @ y - z + w + rho * (x - xk))  # = res_d
        rp = -(A @ x - b + delta * (y - yk))            # = res_p
        # natural residual of the proximal sub-problem
        g = lp_c - At @ y + rho * (x - xk)
        proj = np.minimum(np.maximum(x[ipos] - g[ipos], 0.0), u[ipos])
        gp = g.copy()
        gp[ipos] = x[ipos] - proj
        rn = np.linalg.norm(gp) + np.linalg.norm(A @ x - b + delta * (y - yk))
        return rd, rp, rn

    res_d, res_p, res_n = residuals(x, y, z, w, s)

    while iters < maxit:
        # ---- inner termination (natural residual, relative to the step) ---- #
        prox_move = min(
            1.0, np.linalg.norm(x - xk) + np.linalg.norm(y - yk)
        )
        if res_n < 1.0e4 * tol * prox_move:
            status = 1
            break
        if time.perf_counter() > deadline:
            status = 3
            break
        iters += 1

        # ---- Theta and factorization ------------------------------------- #
        # x and s are kept strictly positive by the fraction-to-the-boundary
        # rule below, so these divisions are safe; the clip only guards against
        # z/x overflowing to infinity when a variable is pinned at its bound.
        theta_min = rho + reg_extra
        theta = np.full(n, theta_min)
        theta[ipos] += z[ipos] / x[ipos]
        theta[ibox] += w[ibox] / s[ibox]
        if not np.isfinite(theta).all():
            theta = np.where(np.isfinite(theta), theta, theta_min / _EPS**2)
        np.clip(theta, theta_min, theta_min / _EPS**2, out=theta)

        try:
            if hasattr(kkt, "set_ipm_context"):
                kkt.set_ipm_context(
                    res_d=res_d,
                    res_p=res_p,
                    res_n=res_n,
                    tol=tol,
                )
            if hasattr(kkt, "observe"):
                kkt.observe(x, z, w, s, mu)
            kkt.factorize(theta, delta + reg_extra)
        except Exception:
            if retry < max_tries:
                iters -= 1
                retry += 1
                reg_extra = min(0.5, max(max(reg_extra, 1e-12) * 10.0, tol))
                continue
            status = 2
            break

        def newton(rd, rp, rmz, rmw):
            r1 = rd.copy()
            r1[ipos] += rmz[ipos] / x[ipos]
            r1[ibox] -= rmw[ibox] / s[ibox]
            try:
                dx, dy = kkt.solve(r1, rp)
            except (KKTSolveError, np.linalg.LinAlgError, FloatingPointError):
                return None
            if not (np.isfinite(dx).all() and np.isfinite(dy).all()):
                return None
            dz = np.zeros(n)
            dw = np.zeros(n)
            dz[ipos] = (rmz[ipos] - z[ipos] * dx[ipos]) / x[ipos]
            dw[ibox] = (rmw[ibox] + w[ibox] * dx[ibox]) / s[ibox]
            return dx, dy, dz, dw

        zero_p, zero_d = np.zeros(A.shape[0]), np.zeros(n)

        if pc == 1:  # ---------------------------------------- plain IPM ---
            sigma = 0.5 if iters == 1 else max(1 - alpha_x, 1 - alpha_z) ** 5
            sigma = min(max(sigma, sigmamin), sigmamax)
            res_muz[ipos] = sigma * mu - x[ipos] * z[ipos]
            res_muw[ibox] = sigma * mu - s[ibox] * w[ibox]
            out = newton(res_d, res_p, res_muz, res_muw)
            if out is None:
                if retry < max_tries:
                    iters -= 1
                    retry += 1
                    reg_extra = min(0.5, max(max(reg_extra, 1e-12) * 10.0, tol))
                    continue
                status = 2
                break
            dx, dy, dz, dw = out
            sigma_used = sigma

        else:  # ------------------------------- Mehrotra / Gondzio ---------
            if pc == 2:
                res_muz[ipos] = -x[ipos] * z[ipos]
                res_muw[ibox] = -s[ibox] * w[ibox]
            else:  # Gondzio: mildly centered predictor
                if iters > 1 or min(alpha_x, alpha_z) > 0.5:
                    mu_target, factor = 0.1 * mu, 0.05
                else:
                    mu_target, factor = 0.7 * mu, 0.1
                res_muz[ipos] = -x[ipos] * z[ipos] + factor * mu_target
                res_muw[ibox] = -s[ibox] * w[ibox] + factor * mu_target

            out = newton(res_d, res_p, res_muz, res_muw)
            if out is None:
                if retry < max_tries:
                    iters -= 1
                    retry += 1
                    reg_extra = min(0.5, max(max(reg_extra, 1e-12) * 10.0, tol))
                    continue
                status = 2
                break
            dx, dy, dz, dw = out

            ds = -dx
            alpha_x = tau * min(
                _step_to_boundary(x, dx, ipos), _step_to_boundary(s, ds, ibox)
            )
            alpha_z = tau * min(
                _step_to_boundary(z, dz, ipos), _step_to_boundary(w, dw, ibox)
            )

            if pc == 2:  # ---- single Mehrotra corrector ------------------ #
                cm = (x[ipos] + alpha_x * dx[ipos]) @ (z[ipos] + alpha_z * dz[ipos])
                cm += (s[ibox] + alpha_x * ds[ibox]) @ (w[ibox] + alpha_z * dw[ibox])
                mu_aff = cm / npos_box
                mu_t = (mu_aff / mu) ** 2 * mu_aff if mu > 0 else mu_aff
                res_muz[ipos] = mu_t - dx[ipos] * dz[ipos]
                res_muw[ibox] = mu_t - ds[ibox] * dw[ibox]
                outc = newton(zero_d, zero_p, res_muz, res_muw)
                if outc is None:
                    if retry < max_tries:
                        iters -= 1
                        retry += 1
                        reg_extra = min(0.5, max(max(reg_extra, 1e-12) * 10.0, tol))
                        continue
                    status = 2
                    break
                def trial_mu(dx_, dz_, dw_, ax_, az_):
                    ds_ = -dx_
                    val = (x[ipos] + ax_ * dx_[ipos]) @ (
                        z[ipos] + az_ * dz_[ipos]
                    )
                    val += (s[ibox] + ax_ * ds_[ibox]) @ (
                        w[ibox] + az_ * dw_[ibox]
                    )
                    return float(val / npos_box)

                # The Mehrotra corrector can be far too aggressive on boxed
                # models once a few variables are within roundoff of a bound.
                # Pick the strongest correction that still gives a usable
                # step and improves the predicted complementarity.  beta=0 is
                # the original affine predictor, so the branch always has a
                # finite fall-back without refactorizing.
                best = None
                for beta in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0):
                    dxb = dx + beta * outc[0]
                    dyb = dy + beta * outc[1]
                    dzb = dz + beta * outc[2]
                    dwb = dw + beta * outc[3]
                    axb = tau * min(
                        _step_to_boundary(x, dxb, ipos),
                        _step_to_boundary(s, -dxb, ibox),
                    )
                    azb = tau * min(
                        _step_to_boundary(z, dzb, ipos),
                        _step_to_boundary(w, dwb, ibox),
                    )
                    if axb <= 0.0 or azb <= 0.0:
                        continue
                    mub = trial_mu(dxb, dzb, dwb, axb, azb)
                    if not np.isfinite(mub) or mub < 0.0:
                        continue
                    step = min(axb, azb)
                    # Prefer lower predicted complementarity, but penalize
                    # directions that gain it only by taking a microscopic
                    # step.  This mirrors the safeguard used by the MCC path,
                    # while retaining pc=2's single-corrector character.
                    merit = mub / max(step, 1e-8)
                    if best is None or merit < best[0]:
                        best = (merit, beta, dxb, dyb, dzb, dwb)

                if best is None:
                    if retry < max_tries:
                        iters -= 1
                        retry += 1
                        reg_extra = min(0.5, max(max(reg_extra, 1e-12) * 10.0, tol))
                        continue
                    status = 2
                    break
                _, beta, dx, dy, dz, dw = best
                sigma_used = beta * (mu_t / mu) if mu > 0 else 0.0

            else:  # ---- Gondzio's multiple centrality correctors --------- #
                K_c, bmin, bmax, delta_a = 10, 1e-3, 1e3, 0.2
                sigma_used = mu_target / mu if mu > 0 else 0.0
                for _ in range(K_c):
                    ta_x = min(1.0, 1.5 * alpha_x + delta_a)
                    ta_z = min(1.0, 1.5 * alpha_z + delta_a)
                    xt = x + ta_x * dx
                    zt = z + ta_z * dz
                    st = s - ta_x * dx
                    wt = w + ta_z * dw
                    for (a_, b_, tgt) in (
                        (xt, zt, res_muz),
                        (st, wt, res_muw),
                    ):
                        idx = ipos if tgt is res_muz else ibox
                        prod = a_[idx] * b_[idx]
                        corr = np.clip(prod, bmin * mu_target, bmax * mu_target) - prod
                        corr = np.clip(corr, -bmax * mu_target, 2 * bmax * mu_target)
                        tgt[idx] = corr
                    outc = newton(zero_d, zero_p, res_muz, res_muw)
                    if outc is None:
                        break
                    dxt = dx + outc[0]
                    dyt = dy + outc[1]
                    dzt = dz + outc[2]
                    dwt = dw + outc[3]
                    a_xn = tau * min(
                        _step_to_boundary(x, dxt, ipos),
                        _step_to_boundary(s, -dxt, ibox),
                    )
                    a_zn = tau * min(
                        _step_to_boundary(z, dzt, ipos),
                        _step_to_boundary(w, dwt, ibox),
                    )
                    if a_xn < 1.01 * alpha_x and a_zn < 1.01 * alpha_z:
                        break
                    alpha_x, alpha_z = a_xn, a_zn
                    dx, dy, dz, dw = dxt, dyt, dzt, dwt

        retry = 0
        # ---- step lengths and update -------------------------------------- #
        if npos_box:
            ds = -dx
            alpha_x = tau * min(
                _step_to_boundary(x, dx, ipos), _step_to_boundary(s, ds, ibox)
            )
            alpha_z = tau * min(
                _step_to_boundary(z, dz, ipos), _step_to_boundary(w, dw, ibox)
            )
        else:
            alpha_x = alpha_z = 1.0

        x += alpha_x * dx
        y += alpha_z * dy
        z += alpha_z * dz
        w += alpha_z * dw
        if nbox:
            _sync_box(x, s, u, ibox, x_floor, s_floor)
        # a one-off perturbation should not weigh down the rest of the solve
        if reg_extra:
            reg_extra = 0.0 if reg_extra <= 1e-12 else reg_extra * 0.1

        if npos_box:
            mu = (x[ipos] @ z[ipos] + s[ibox] @ w[ibox]) / npos_box

        res_d, res_p, res_n = residuals(x, y, z, w, s)

        # Give up on a sub-problem that has stopped improving, and let the outer
        # proximal loop re-anchor at the current point.  Progress is measured on
        # the residual rather than on the step length, because a direction that
        # is mostly noise can still admit alpha ~ 1e-5 and leave every printed
        # digit of the residual unchanged; watching alpha alone lets such a
        # sub-problem grind out its whole iteration budget having stopped
        # improving forty iterations earlier.
        if res_n < 0.999 * res_n_best:
            res_n_best = res_n
            stalled = 0
        elif not np.isfinite(res_n):
            stalled += 3
        else:
            stalled += 1

        if printlevel >= 2:
            print(
                f"   {iters:4d}   {np.linalg.norm(res_p):8.2e}  "
                f"{np.linalg.norm(res_d):8.2e}  {mu:8.2e}   "
                f"{sigma_used:8.2e}  {alpha_x:6.4f}  {alpha_z:6.4f}"
            )

        if stalled >= 5:
            break

    return x, y, z, w, dict(iters=iters, status=status, stalled=stalled)


# =========================================================================== #
#  Outer proximal loop   (PPM_IPM.m)
# =========================================================================== #
def psipm_solve(
    lp: StandardLP,
    tol: float = 1e-6,
    maxit: int = 200,
    inner_maxit: int = 100,
    pc: int = 3,
    rho: Optional[float] = None,
    delta: Optional[float] = None,
    rf: float = 1.0,
    ppm_red: float = 0.5,
    kkt: str = "normal",
    stop: str = "relative",
    time_limit: float = np.inf,
    printlevel: int = 1,
    order: str = "best",
    refine: int = 2,
    low_rank_max_columns: int = 8,
    low_rank_min_nnz: int = 1024,
    low_rank_clique_ratio: float = 2.0,
    compl: str = "gap",
    tau: float = 0.995,
    diverge_tol: float = 1e4,
    initial: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    start_projection: str = "legacy",
    start_bound_fraction: float = 1e-3,
    start_linear_tolerance: Optional[float] = None,
    dynamic_reg_factor: float = 0.1,
    dynamic_update_interval: int = 2,
    core_factor: float = 2.0,
    core_update_interval: int = 6,
    core_cg_tol: float = 1e-8,
    core_cg_maxit: int = 100,
    lagged_max_age: int = 8,
    lagged_refresh_cg: int = 20,
    lagged_cg_tol: float = 1e-8,
    lagged_cg_maxit: int = 80,
    partial_explicit: int = 256,
    partial_warmup: int = 4,
    partial_pool_factor: float = 8.0,
    partial_draft_weight: float = 1.0,
    partial_min_fill_ratio: float = 2.0,
    kkt_solver=None,
) -> Solution:
    """Proximal-Stabilized IPM.

    Parameters
    ----------
    tol        : target accuracy.
    pc         : 1 = no predictor-corrector, 2 = Mehrotra, 3 = Gondzio MCC.
    rho, delta : primal / dual proximal parameters (default
                 ``rf * max(tol / ||A||_inf, 1e-8)``, as in ``Netlib_examples.m``).
    kkt        : ``"normal"`` (full CHOLMOD), ``"dynamic"`` (certified
                 dynamic non-diagonal regularization), ``"core-pcg"``
                 (generated sparse-core preconditioner for the exact full
                 Schur equation), ``"lagged-pcg"`` (exact low-frequency
                 Schur updates), or ``"augmented"`` (QDLDL).
                 ``"partial"`` uses exact fill-aware partial condensation and
                 falls back to full normal equations when its symbolic gate
                 predicts less than ``partial_min_fill_ratio`` improvement.
    stop       : ``"relative"`` -- standard scaled KKT test (comparable with
                 HiGHS); ``"psipm"`` -- the test of the MATLAB code.
    time_limit : wall-clock budget in seconds for the PS-IPM solve, including
                 the initial-point KKT factorization.  Model setup and MPS
                 reading are not counted.
    refine     : maximum iterative-refinement steps applied to the normal
                 equations after each factorization (``kkt="normal"`` only).
                 0 disables both the refinement and the residual check that
                 detects an unusable factorization.
    low_rank_* : controls exact Woodbury extraction of clique-producing
                 columns from the sparse normal factor.  Set
                 ``low_rank_max_columns=0`` to disable it.
    kkt_solver : optional pre-built object implementing ``factorize`` and
                 ``solve``.  This extension point lets experimental exact
                 matrix-free methods reuse the IPM without duplicating it.
    start_projection : ``"adaptive"`` compares the legacy central shift with
                 a feasibility-oriented bounded projection; ``"legacy"``
                 retains the original global-shift projection.
    start_bound_fraction : relative distance from finite bounds in the
                 adaptive projected candidate.
    start_linear_tolerance : optional relaxed relative tolerance for iterative
                 least-squares solves used only to construct the start.  The
                 default preserves the original all-or-nothing exact startup.
    tau        : fraction-to-the-boundary factor.  0.995 is the value in the
                 MATLAB code; HiGHS' IPX uses 0.9, which keeps a variable two
                 orders of magnitude further from its bound after ten blocking
                 steps (1e-10 against 1e-23) and so limits the ~1/x
                 amplification of the corrector target.  Lowering it costs
                 iterations on well-behaved models -- about 17% on a set of
                 generated LPs -- and is worth trying on one that breaks down
                 late.
    diverge_tol: give up when ``mu`` climbs above this multiple of the best
                 ``mu`` seen so far, reporting ``"diverged"``.  ``np.inf``
                 disables it.  ``mu`` rising by a factor of ten early on is
                 normal, so the default is deliberately loose.
    compl      : how complementarity is tested (``stop="relative"`` only).
                 ``"any"`` accepts ``mu < tol`` *or* ``gap < tol`` (the default,
                 and what the MATLAB code does); ``"mu"`` requires the average
                 pair; ``"gap"`` requires the relative duality gap.

                 These are far apart on a large model: with the residuals small
                 the gap numerator is exactly ``x'z + s'w = npos_box * mu``, so
                 ``gap`` is about ``npos_box`` times ``mu`` and ``"any"`` is
                 satisfied by whichever is weaker.  On an LP with ~5e5
                 complementarity pairs, stopping at ``mu = 8.6e-8`` left the
                 primal and dual objectives 4.7e-2 apart.  Use ``"gap"`` when
                 the objective value itself has to be trustworthy.
    """
    A = lp.A
    At = A.T.tocsr()
    c, b, u = lp.c, lp.b, lp.u
    m, n = A.shape

    ipos = _all_or_idx(lp.pos)
    ibox = _all_or_idx(lp.box)
    npos, nbox = int(lp.pos.sum()), int(lp.box.sum())
    npos_box = npos + nbox

    if start_projection not in {"adaptive", "legacy"}:
        raise ValueError("start_projection must be 'adaptive' or 'legacy'")
    if not 0.0 < start_bound_fraction < 0.5:
        raise ValueError("start_bound_fraction must lie in (0, 0.5)")
    if (
        start_linear_tolerance is not None
        and not 0.0 < start_linear_tolerance < 1.0
    ):
        raise ValueError("start_linear_tolerance must lie in (0, 1)")

    normA = abs(A).sum(axis=1).max() if A.nnz else 1.0  # ||A||_inf
    normA = float(normA)
    if rho is None:
        rho = rf * max(tol / max(normA, 1e-16), 1e-8)
    if delta is None:
        delta = rho
    sc = max(1.0, normA, float(np.abs(b).sum()), float(np.abs(c).sum()))
    dp, dd = 1.0 + float(np.abs(b).max(initial=0.0)), 1.0 + float(np.abs(c).max(initial=0.0))

    if printlevel:
        if rho == delta:
            print(f"  rho = delta    : {rho:.3e}   (||A||_inf = {normA:.3e})")
        else:
            print(
                f"  rho / delta    : {rho:.3e} / {delta:.3e}   "
                f"(||A||_inf = {normA:.3e})"
            )
        print(f"  KKT solver     : {kkt}")

    if kkt_solver is not None:
        solver = kkt_solver
        if getattr(solver, "A", A).shape != A.shape:
            raise ValueError("external KKT solver dimensions do not match the LP")
    elif kkt == "normal":
        solver = NormalEquations(
            A,
            order=order,
            refine=refine,
            low_rank_max_columns=low_rank_max_columns,
            low_rank_min_nnz=low_rank_min_nnz,
            low_rank_clique_ratio=low_rank_clique_ratio,
        )
    elif kkt == "dynamic":
        solver = DynamicRegularizedNormalEquations(
            A,
            tol=tol,
            reg_factor=dynamic_reg_factor,
            update_interval=dynamic_update_interval,
            order=order,
            refine=refine,
        )
    elif kkt == "core-pcg":
        solver = CorePCGNormalEquations(
            A,
            core_factor=core_factor,
            update_interval=core_update_interval,
            cg_tol=core_cg_tol,
            cg_maxit=core_cg_maxit,
            order=order,
            refine=refine,
        )
    elif kkt == "lagged-pcg":
        solver = LaggedPCGNormalEquations(
            A,
            max_age=lagged_max_age,
            refresh_cg=lagged_refresh_cg,
            cg_tol=lagged_cg_tol,
            cg_maxit=lagged_cg_maxit,
            order=order,
            refine=refine,
            low_rank_max_columns=low_rank_max_columns,
            low_rank_min_nnz=low_rank_min_nnz,
            low_rank_clique_ratio=low_rank_clique_ratio,
        )
    elif kkt == "augmented":
        solver = AugmentedSystem(A)
    elif kkt == "partial":
        solver = PartialCondensation(
            A,
            pos=lp.pos,
            box=lp.box,
            explicit_columns=partial_explicit,
            warmup=partial_warmup,
            pool_factor=partial_pool_factor,
            draft_weight=partial_draft_weight,
            min_fill_ratio=partial_min_fill_ratio,
            order=order,
            refine=refine,
        )
    else:
        raise ValueError(f"unknown KKT solver {kkt!r}")

    low_rank_columns = getattr(
        solver, "low_rank_columns", np.empty(0, dtype=np.int64)
    )
    low_rank_count = int(low_rank_columns.size)
    avoided_edges = 0
    if low_rank_count:
        avoided_edges = sum(
            int(q) * (int(q) - 1) // 2
            for q in np.diff(A.indptr)[low_rank_columns]
        )
    if printlevel and low_rank_count:
        print(
            f"  low-rank KKT   : {low_rank_count} column(s), "
            f"{int(avoided_edges):,} candidate row edges kept out of CHOLMOD"
        )
    x_floor, s_floor = _box_floors(u, ibox) if nbox else (0.0, 0.0)

    tic = time.perf_counter()
    deadline = tic + time_limit

    # ------------------------------------------------------------------ #
    # Starting point (HiGHS/IPX style, projected onto this bounded form)
    # ------------------------------------------------------------------ #
    t_start = time.perf_counter()
    if initial is None:
        x, y, z, w = _starting_point(
            A,
            At,
            b,
            c,
            u,
            lp,
            ipos,
            ibox,
            solver,
            kkt,
            projection=start_projection,
            bound_fraction=start_bound_fraction,
            linear_tolerance=start_linear_tolerance,
        )
        start_tag = "start point"
    else:
        if len(initial) != 4:
            raise ValueError("initial must contain (x, y, z, w)")
        x, y, z, w = (np.asarray(v, dtype=float).copy() for v in initial)
        if x.size != n or y.size != m or z.size != n or w.size != n:
            raise ValueError("warm-start dimensions do not match the LP")
        if not all(np.isfinite(v).all() for v in (x, y, z, w)):
            raise ValueError("warm start must be finite")
        lower_floor = np.full(n, 1e-12)
        if nbox:
            box_x_floor, box_s_floor = _box_floors(u, ibox)
            lower_floor[ibox] = box_x_floor
            x[ibox] = np.minimum(
                np.maximum(x[ibox], box_x_floor), u[ibox] - box_s_floor
            )
        # Use the stored Boolean masks here.  ``ipos`` is deliberately a slice
        # when every column is nonnegative, and slices cannot be complemented.
        x[lp.pos] = np.maximum(x[lp.pos], lower_floor[lp.pos])
        z[lp.pos] = np.maximum(z[lp.pos], 1e-12)
        w[lp.box] = np.maximum(w[lp.box], 1e-12)
        x[~lp.pos] = np.asarray(initial[0], dtype=float)[~lp.pos]
        z[~lp.pos] = 0.0
        w[~lp.box] = 0.0
        start_tag = "warm start"
    if printlevel:
        print(f"  {start_tag:<15}: {time.perf_counter() - t_start:.2f}s")

    xk, yk, zk, wk = x.copy(), y.copy(), z.copy(), w.copy()
    best_mu = np.inf
    sol = Solution(x=x, y=y, z=z, w=w)
    inner_tot = 0

    for it in range(1, maxit + 1):
        # ---- unregularized residuals (IPM_Res.m) ------------------------- #
        rp = b - A @ x
        rd = c - At @ y - z + w
        s = np.ones(n)
        if nbox:
            _sync_box(x, s, u, ibox, x_floor, s_floor)
        mu = (x[ipos] @ z[ipos] + s[ibox] @ w[ibox]) / npos_box if npos_box else 0.0

        nrp1, nrpi = float(np.abs(rp).sum()), float(np.abs(rp).max(initial=0.0))
        nrdi = float(np.abs(rd).max(initial=0.0))
        # min-map complementarity (STOP in PPM_IPM.m), extended to the box
        comp = np.max(
            np.minimum(
                np.minimum(np.abs(x[ipos] * z[ipos]), np.abs(z[ipos])), np.abs(x[ipos])
            ),
            initial=0.0,
        )
        if nbox:
            comp = max(
                comp,
                np.max(
                    np.minimum(
                        np.minimum(np.abs(s[ibox] * w[ibox]), np.abs(w[ibox])),
                        np.abs(s[ibox]),
                    ),
                    initial=0.0,
                ),
            )
        # Relative duality gap.  The numerator is the primal minus the dual
        # objective of the *standard form*, where every lower bound is 0 and the
        # general  b'y + l'z^l - u'z^u  collapses to  b'y - u'w.  Shifting the
        # bounds moves the primal and the dual objective by the same c'shift, so
        # the numerator is the same as it would be on the original data -- but
        # the normalization is not, and has to use the objective the solver
        # actually reports, obj_const included.  Without it, a model whose
        # columns have large lower bounds is measured against the wrong scale.
        cx = float(c @ x)
        gap = abs(cx - (float(b @ y) - float(w[ibox] @ u[ibox]))) / (
            1.0 + abs(cx + lp.obj_const)
        )

        sol.history.append(
            dict(outer=it, rp=nrpi / dp, rd=nrdi / dd, mu=mu, gap=gap, inner=inner_tot)
        )
        if printlevel:
            print(
                f"  PPM {it:3d} | rp {nrpi / dp:8.2e}  rd {nrdi / dd:8.2e}  "
                f"mu {mu:8.2e}  gap {gap:8.2e}  inner {inner_tot:4d}  "
                f"t {time.perf_counter() - tic:6.1f}s"
            )
        if printlevel >= 2:
            # Which columns hold the dual residual.  A stalled rd is usually
            # concentrated in one class -- free columns carry no z/w at all, so
            # their stationarity has to come from y alone and cannot be repaired
            # by the barrier -- and that tells you where to look.
            j = int(np.argmax(np.abs(rd)))
            cls = ("free" if not lp.pos[j] else "box" if lp.box[j] else "pos")
            parts = []
            for tag, msk in (("free", ~lp.pos), ("box", lp.box),
                             ("pos", lp.pos & ~lp.box)):
                if msk.any():
                    parts.append(f"{tag} {float(np.abs(rd[msk]).max()):8.2e}")
            print(
                f"        |rd| by class: {'  '.join(parts)}   "
                f"argmax j={j} ({cls}, |rd_j|={abs(float(rd[j])):.2e}, "
                f"x_j={float(x[j]):.2e})"
            )

        if stop == "psipm":
            done = nrp1 < tol * sc and nrdi < tol * sc and comp < tol
        else:
            if compl == "mu":
                ok_c = mu < tol
            elif compl == "gap":
                ok_c = gap < tol
            else:
                ok_c = mu < tol or gap < tol
            done = nrpi / dp < tol and nrdi / dd < tol and ok_c
        if done:
            sol.status = "optimal"
            break
        # Divergence.  Once centrality is lost the outer loop can spend hundreds
        # of iterations cycling -- the inner solve drives mu down, the next
        # proximal step throws it back up -- without ever recovering, and the
        # answer at the end is wrong rather than merely inaccurate.  IPX guards
        # this by comparing the current complementarity against the best one
        # seen; on s250r10 that fires around outer 50, where mu has climbed to
        # 5e5 times its best, instead of running to the iteration limit.
        if npos_box and mu > 0.0:
            if mu < best_mu:
                best_mu = mu
            elif best_mu > 0.0 and mu > diverge_tol * best_mu:
                sol.status = "diverged"
                break
        if time.perf_counter() > deadline:
            sol.status = "time-limit"
            break

        x, y, z, w, info = prox_eval(
            c, A, At, b, u, ipos, ibox, npos_box,
            xk, yk, zk, wk, rho, delta,
            ppm_red ** it, inner_maxit, pc, solver, printlevel, deadline,
            nbox=nbox, tau=tau,
        )
        inner_tot += info["iters"]
        if info["status"] == 2:
            sol.status = "ill-conditioned"
            break
        if info["status"] == 3:
            sol.status = "time-limit"
            xk, yk, zk, wk = x.copy(), y.copy(), z.copy(), w.copy()
            break
        xk, yk, zk, wk = x.copy(), y.copy(), z.copy(), w.copy()
    else:
        sol.status = "max-iterations"

    sol.x, sol.y, sol.z, sol.w = x, y, z, w
    sol.outer_iter = it
    sol.inner_iter = inner_tot
    sol.time = time.perf_counter() - tic
    sol.t_fact, sol.t_solve = solver.t_fact, solver.t_solve
    sol.res_p, sol.res_d, sol.mu = nrpi / dp, nrdi / dd, mu
    sol.obj = float(c @ x) + lp.obj_const
    n_not_pd = getattr(solver, "stats", {}).get("not_pd", 0)
    sol.n_not_pd = n_not_pd
    sol.n_shifted = getattr(solver, "n_shifted", 0)
    sol.n_refine = getattr(solver, "n_refine", 0)
    sol.low_rank_columns = low_rank_count
    sol.low_rank_candidate_edges = int(avoided_edges)
    core_sizes = getattr(solver, "core_sizes", [])
    if core_sizes:
        sol.core_min = min(core_sizes)
        sol.core_max = max(core_sizes)
        sol.core_mean = float(np.mean(core_sizes))
    pcg_iterations = getattr(solver, "cg_iterations", [])
    if pcg_iterations:
        sol.pcg_iterations = int(sum(pcg_iterations))
        sol.pcg_max_iterations = int(max(pcg_iterations))
    sol.pcg_failures = getattr(solver, "cg_failures", 0)
    sol.full_kkt_fallbacks = getattr(solver, "full_fallbacks", 0)
    sol.preconditioner_refreshes = getattr(solver, "refreshes", 0)
    sol.preconditioner_factorizations = getattr(solver, "n_fact", 0)
    retained = getattr(solver, "retained_fractions", [])
    if retained:
        sol.preconditioner_retained_mean = float(np.mean(retained))
    sol.preconditioner_symbolic_analyses = getattr(solver, "n_symbolic", 0)
    sol.preconditioner_retries = getattr(solver, "pcg_retries", 0)
    sol.preconditioner_full = getattr(solver, "use_full_preconditioner", False)
    condition_estimates = getattr(solver, "condition_estimates", [])
    if condition_estimates:
        finite = np.asarray(condition_estimates, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            sol.preconditioner_condition_max = float(finite.max())
    spectral_errors = getattr(solver, "spectral_error_estimates", [])
    if spectral_errors:
        finite = np.asarray(spectral_errors, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            sol.preconditioner_spectral_error_max = float(finite.max())
    lag_bounds = getattr(solver, "lag_spectral_bounds", [])
    if lag_bounds:
        finite = np.asarray(lag_bounds, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            sol.preconditioner_lag_bound_max = float(finite.max())
    sol.preconditioner_core_swaps = getattr(solver, "core_swaps", 0)
    sol.residual_replacements = getattr(solver, "residual_replacements", 0)
    sol.residual_checks = getattr(solver, "residual_checks", 0)
    sol.krylov_stagnations = getattr(solver, "stagnation_events", 0)
    sol.minres_fallbacks = getattr(solver, "minres_fallbacks", 0)
    sol.minres_iterations = getattr(solver, "minres_iterations", 0)
    sol.krylov_breakdowns = getattr(solver, "krylov_breakdowns", 0)
    sol.krylov_iteration_limits = getattr(solver, "krylov_iteration_limits", 0)
    dynamic_shifts = getattr(solver, "dynamic_shifts", [])
    if dynamic_shifts:
        sol.dynamic_shift_max = float(max(dynamic_shifts))
    if isinstance(solver, PartialCondensation):
        sol.partial_explicit = int(solver.explicit.size)
        sol.partial_condensed = int(solver.condensed.size)
        sol.partial_fill_ratio = solver.symbolic_fill_ratio
        sol.partial_symbolic_seconds = solver.symbolic_seconds
        sol.partial_rebuilds = solver.partition_rebuilds
        sol.partial_rejected = solver._rejected
    if printlevel:
        print(
            f"  -> {sol.status}: obj {sol.obj:.10e}, {sol.outer_iter} outer / "
            f"{sol.inner_iter} inner it, {sol.time:.1f}s "
            f"(fact {sol.t_fact:.1f}s, solve {sol.t_solve:.1f}s)"
        )
        if n_not_pd or sol.n_shifted or sol.n_refine:
            print(
                f"     linear algebra: {n_not_pd} non-PD pivot report(s), "
                f"{sol.n_shifted} shifted factorization(s), "
                f"{sol.n_refine} refinement step(s)"
            )
        if core_sizes:
            detail = (
                f"core {sol.core_min}/{sol.core_mean:.0f}/{sol.core_max} "
                "(min/mean/max)"
            )
            if pcg_iterations:
                detail += (
                    f", PCG {sol.pcg_iterations} total / "
                    f"{sol.pcg_max_iterations} max, "
                    f"{sol.pcg_failures} failure(s), "
                    f"{sol.full_kkt_fallbacks} full fallback(s)"
                )
            if dynamic_shifts:
                detail += f", max dynamic shift {sol.dynamic_shift_max:.2e}"
            print(f"     sparse Schur : {detail}")
            if hasattr(solver, "retained_fractions"):
                mode = "exact/full" if sol.preconditioner_full else "stable/core"
                print(
                    f"     preconditioner: {mode}, "
                    f"{sol.preconditioner_factorizations} numeric factor(s), "
                    f"{sol.preconditioner_refreshes} core refresh(es), "
                    f"{sol.preconditioner_symbolic_analyses} symbolic analysis(es), "
                    f"mean retention {sol.preconditioner_retained_mean:.3f}"
                )
        if isinstance(solver, PartialCondensation):
            decision = "rejected -> full normal" if solver._rejected else "accepted"
            print(
                f"     partial KKT  : {decision}, C {solver.explicit.size}, "
                f"symbolic ratio {solver.symbolic_fill_ratio:.3f}, "
                f"analysis {solver.symbolic_seconds:.2f}s, "
                f"{solver.partition_rebuilds} LDL build(s)"
            )
    return sol


def _restricted_master(lp, columns, costs, artificial_cost, obj_const):
    """Build an RMP with the single feasibility column ``b * theta``."""
    columns = np.asarray(columns, dtype=np.int64)
    artificial = sp.csc_matrix(lp.b.reshape(-1, 1))
    A = sp.hstack([lp.A[:, columns], artificial], format="csc")
    return StandardLP(
        c=np.concatenate([costs[columns], [float(artificial_cost)]]),
        A=A,
        b=lp.b,
        u=np.concatenate([lp.u[columns], [np.inf]]),
        pos=np.concatenate([lp.pos[columns], [True]]),
        box=np.concatenate([lp.box[columns], [False]]),
        cert_u=np.concatenate([
            (lp.u if lp.cert_u is None else lp.cert_u)[columns], [np.inf]
        ]),
        obj_const=float(obj_const),
        scale=np.concatenate([lp.scale[columns], [1.0]]),
    )


def _restricted_master_without_artificial(lp, columns, costs, obj_const):
    """Build the feasible Phase-II master after removing the Phase-I column."""
    columns = np.asarray(columns, dtype=np.int64)
    return StandardLP(
        c=np.asarray(costs[columns], dtype=np.float64),
        A=lp.A[:, columns].tocsc(),
        b=lp.b,
        u=lp.u[columns],
        pos=lp.pos[columns],
        box=lp.box[columns],
        cert_u=(lp.u if lp.cert_u is None else lp.cert_u)[columns],
        obj_const=float(obj_const),
        scale=lp.scale[columns],
    )


def _top_priced(values, mask, count):
    """Return up to ``count`` indices with largest values under ``mask``."""
    indices = np.flatnonzero(mask)
    if not indices.size:
        return indices
    count = min(int(count), indices.size)
    if count < indices.size:
        local = np.argpartition(values[indices], -count)[-count:]
        indices = indices[local]
    return indices[np.argsort(values[indices])[::-1]]


def _initial_working_mask(lp, target, l1_budget):
    """Choose structural columns while retaining every unpriceable direction."""
    m, n = lp.A.shape
    cert_u = lp.u if lp.cert_u is None else lp.cert_u
    cap_original = lp.scale * cert_u
    working = ~lp.pos
    if not np.isfinite(l1_budget):
        working |= lp.pos & ~np.isfinite(cap_original)

    # One coefficient per row gives phase I broad row coverage before pricing.
    Ar = lp.A.tocsr()
    for i in range(m):
        start, end = Ar.indptr[i : i + 2]
        jj = Ar.indices[start:end]
        if not jj.size:
            continue
        aa = Ar.data[start:end]
        available = ~working[jj]
        if not available.any():
            continue
        jj = jj[available]
        aa = aa[available]
        if lp.b[i] != 0.0:
            aligned = aa * np.sign(lp.b[i])
            positive = aligned > 0.0
            local = (
                int(np.argmax(aligned))
                if positive.any()
                else int(np.argmax(np.abs(aa)))
            )
        else:
            local = int(np.argmax(np.abs(aa)))
        working[jj[local]] = True

    target = min(n, max(int(target), int(working.sum())))
    need = target - int(working.sum())
    if need > 0:
        available = np.flatnonzero(~working)
        col_nnz = np.diff(lp.A.indptr)
        if need < available.size:
            selected = available[
                np.argpartition(col_nnz[available], -need)[-need:]
            ]
        else:
            selected = available
        working[selected] = True
    return working


def _derive_l1_budget(lp, objective_upper):
    """Derive ``||x*||_1`` from a nonnegative objective when possible."""
    if objective_upper is None or not np.isfinite(objective_upper):
        return np.inf
    if np.any(~lp.pos):
        return np.inf
    c_original = lp.c / lp.scale
    cert_u = lp.u if lp.cert_u is None else lp.cert_u
    cap_original = lp.scale * cert_u
    scale = 1.0 + float(np.abs(c_original).max(initial=0.0))
    zero = np.abs(c_original) <= 100.0 * _EPS * scale
    if np.any(c_original < -100.0 * _EPS * scale):
        return np.inf
    if np.any(zero & ~np.isfinite(cap_original)):
        return np.inf
    zero_budget = float(cap_original[zero].sum()) if zero.any() else 0.0
    positive = c_original > 100.0 * _EPS * scale
    if not positive.any():
        value = zero_budget
    else:
        rhs = max(0.0, float(objective_upper) - lp.obj_const)
        value = zero_budget + rhs / float(c_original[positive].min())
    value += 100.0 * _EPS * (1.0 + abs(value))
    return value if np.isfinite(value) else np.inf


def _pricing_budget(v_scaled, mask, cert_u, scale, l1_budget):
    """Compute the capped fractional-knapsack budget ``T(v; kappa, X)``."""
    positive = mask & (v_scaled > 0.0)
    if not positive.any():
        return 0.0
    indices = np.flatnonzero(positive)
    caps_scaled = cert_u[indices]

    if not np.isfinite(l1_budget):
        if np.any(~np.isfinite(caps_scaled)):
            return np.inf
        products = v_scaled[indices] * caps_scaled
        value = float(products.sum())
        magnitude = float(np.abs(products).sum())
    else:
        caps_original = np.minimum(
            scale[indices] * caps_scaled, float(l1_budget)
        )
        density = v_scaled[indices] / scale[indices]
        order = np.argsort(density)[::-1]
        remaining = float(l1_budget)
        value = 0.0
        magnitude = 0.0
        for local in order:
            take = min(float(caps_original[local]), remaining)
            term = float(density[local]) * take
            value += term
            magnitude += abs(term)
            remaining -= take
            if remaining <= 0.0:
                break

    value += 100.0 * _EPS * (1.0 + magnitude)
    return value if np.isfinite(value) else np.inf


def retrospective_solve(
    lp: StandardLP,
    tol: float = 1e-6,
    time_limit: float = np.inf,
    printlevel: int = 1,
    cg_initial_columns: Optional[int] = None,
    cg_pricing_batch: Optional[int] = None,
    cg_support_tol: Optional[float] = None,
    cg_pricing_tol: Optional[float] = None,
    cg_stagnation_tol: Optional[float] = None,
    cg_elim_delta: Optional[float] = None,
    cg_l1_budget: Optional[float] = None,
    cg_objective_upper: Optional[float] = None,
    cg_beta: Optional[int] = None,
    cg_max_rounds: int = 30,
    cg_full_pool_after: int = 5,
    cg_artificial_cost: Optional[float] = None,
    **solver_kwargs,
) -> Solution:
    """Option 3: support-reduction CG with retrospective pool elimination.

    Each pricing/elimination pass follows a converged restricted-master solve.
    This is required by the PDF's objective-drop theorem; ordinary infeasible
    PPM iterates do not supply a valid retrospective gap.
    """
    started = time.perf_counter()
    deadline = started + time_limit
    m, n = lp.A.shape
    scale = lp.scale
    cert_u = lp.u if lp.cert_u is None else lp.cert_u
    c_original = lp.c / scale

    pricing_tol = (
        max(tol, 1e-9) if cg_pricing_tol is None else float(cg_pricing_tol)
    )
    support_tol = (
        max(0.1 * tol, 1e-10)
        if cg_support_tol is None
        else float(cg_support_tol)
    )
    stagnation_tol = (
        max(100.0 * tol, 1e-6)
        if cg_stagnation_tol is None
        else float(cg_stagnation_tol)
    )
    elim_delta = (
        max(10.0 * tol, 1e-8)
        if cg_elim_delta is None
        else float(cg_elim_delta)
    )
    if elim_delta <= 0.0:
        raise ValueError("cg_elim_delta must be positive")

    derived_budget = _derive_l1_budget(lp, cg_objective_upper)
    l1_budget = (
        derived_budget if cg_l1_budget is None else float(cg_l1_budget)
    )
    if l1_budget < 0.0:
        raise ValueError("cg_l1_budget must be nonnegative")
    initial_columns = (
        max(256, m) if cg_initial_columns is None else int(cg_initial_columns)
    )
    pricing_batch = (
        max(1000, min(n, max(1, m // 2)))
        if cg_pricing_batch is None
        else int(cg_pricing_batch)
    )
    if pricing_batch <= 0 or cg_max_rounds <= 0 or cg_full_pool_after < 0:
        raise ValueError("pricing batch and maximum rounds must be positive")
    beta = n if cg_beta is None else int(cg_beta)
    if beta < 1:
        raise ValueError("cg_beta must be positive")

    cap_original = scale * cert_u
    if np.isfinite(l1_budget):
        common_cap = float(np.minimum(cap_original, l1_budget).max(initial=0.0))
    elif np.isfinite(cap_original).all():
        common_cap = float(cap_original.max(initial=0.0))
    else:
        common_cap = np.inf

    pool = lp.pos.copy()
    pool[~lp.pos] = True
    working = _initial_working_mask(lp, initial_columns, l1_budget)
    protected = ~lp.pos
    total_outer = total_inner = 0
    total_fact = total_solve = 0.0
    total_not_pd = total_shifted = total_refine = 0
    added_total = eliminated_total = reactivated_total = 0
    history = []
    all_history = []
    last_local = None
    last_columns = np.flatnonzero(working)
    last_phase2_local = None
    last_phase2_columns = None
    last_theta = np.inf
    last_violation = np.inf

    def solve_rmp(columns, costs, artificial_cost, obj_const):
        nonlocal total_outer, total_inner, total_fact, total_solve
        nonlocal total_not_pd, total_shifted, total_refine
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return None
        rmp = _restricted_master(
            lp, columns, costs, artificial_cost, obj_const
        )
        local = psipm_solve(
            rmp,
            tol=tol,
            time_limit=remaining,
            printlevel=max(0, printlevel - 1),
            **solver_kwargs,
        )
        total_outer += local.outer_iter
        total_inner += local.inner_iter
        total_fact += local.t_fact
        total_solve += local.t_solve
        total_not_pd += local.n_not_pd
        total_shifted += local.n_shifted
        total_refine += local.n_refine
        all_history.extend(local.history)
        return local

    # Phase I: min theta, A_W x + b theta = b.  The point (0, 1) is feasible.
    phase1_rounds = 0
    phase1_value_previous = np.inf
    phase1_support_previous = None
    phase1_stall_streak = 0
    phase1_ok = False
    zero_cost = np.zeros(n)
    for phase1_rounds in range(1, cg_max_rounds + 1):
        columns = np.flatnonzero(working)
        local = solve_rmp(columns, zero_cost, 1.0, 0.0)
        if local is None:
            last_local, last_columns = local, columns
            break
        if local.status != "optimal":
            last_local, last_columns = local, columns
            if local.status != "time-limit" and int(working.sum()) < int(pool.sum()):
                restored = pool & ~working
                added_total += int(restored.sum())
                working = pool.copy()
                working |= protected
                if printlevel:
                    print(
                        f"        phase-I RMP {local.status}; restored full pool"
                    )
                continue
            break
        last_local, last_columns = local, columns
        theta = max(0.0, float(local.x[-1]))
        last_theta = theta
        v_scaled = np.asarray(lp.A.T @ local.y).ravel()
        v_original = v_scaled / scale
        missing = pool & ~working & lp.pos
        candidates = missing & (v_original > pricing_tol)
        last_violation = float(v_original[missing].max(initial=0.0))
        real_x_original = scale[columns] * local.x[:-1]
        support = np.zeros(n, dtype=bool)
        support[columns[real_x_original > support_tol]] = True
        support |= protected
        history.append(dict(
            phase=1, round=phase1_rounds, objective=theta,
            working=int(working.sum()), pool=int(pool.sum()),
            priced=last_violation, added=0, eliminated=0,
        ))
        if printlevel:
            print(
                f"  CG phase I {phase1_rounds:2d} | theta {theta:8.2e}  "
                f"price {last_violation:8.2e}  W {int(working.sum()):7d}/"
                f"{n:<7d}  t {time.perf_counter() - started:6.1f}s"
            )
        if theta <= max(tol, 1e-9):
            phase1_ok = True
            break
        if not candidates.any():
            break
        objective_stalled = (
            phase1_support_previous is not None
            and abs(theta - phase1_value_previous)
            <= stagnation_tol * (1.0 + abs(theta))
        )
        phase1_stall_streak = (
            phase1_stall_streak + 1 if objective_stalled else 0
        )
        adaptive_batch = pricing_batch * (2 ** min(phase1_stall_streak, 6))
        entering = _top_priced(v_original, candidates, adaptive_batch)
        stagnant = objective_stalled
        next_working = (working if stagnant else support).copy()
        next_working[entering] = True
        next_working |= protected
        added_total += int(np.count_nonzero(next_working & ~working))
        history[-1]["added"] = int(np.count_nonzero(next_working & ~working))
        phase1_value_previous = theta
        phase1_support_previous = support
        working = next_working

    status = "unknown"
    phase2_rounds = 0
    previous = None
    if phase1_ok:
        max_cost = float(np.abs(c_original).max(initial=0.0))
        artificial_cost = (
            max(1e4, 100.0 * (1.0 + max_cost))
            if cg_artificial_cost is None
            else float(cg_artificial_cost)
        )
        if artificial_cost <= 0.0:
            raise ValueError("cg_artificial_cost must be positive")

        phase2_stall_streak = 0
        for phase2_rounds in range(1, cg_max_rounds + 1):
            columns = np.flatnonzero(working)
            local = solve_rmp(
                columns, lp.c, artificial_cost, lp.obj_const
            )
            if local is None:
                status = "time-limit"
                break
            last_local, last_columns = local, columns
            if local.status != "optimal":
                if (
                    local.status != "time-limit"
                    and int(working.sum()) < int(pool.sum())
                ):
                    restored = pool & ~working
                    added_total += int(restored.sum())
                    working = pool.copy()
                    working |= protected
                    previous = None
                    if printlevel:
                        print(
                            f"        phase-II RMP {local.status}; "
                            "restored full pool"
                        )
                    continue
                status = local.status
                break
            last_phase2_local = local
            last_phase2_columns = columns

            theta = max(0.0, float(local.x[-1]))
            last_theta = theta
            real_x_original = scale[columns] * local.x[:-1]
            objective = float(lp.c[columns] @ local.x[:-1]) + lp.obj_const
            v_scaled = np.asarray(lp.A.T @ local.y).ravel() - lp.c
            v_original = v_scaled / scale

            # Reversible safety check: a removed column that becomes violated
            # returns to the candidate pool before ordinary pricing.
            reactivated = (~pool) & lp.pos & (v_original > pricing_tol)
            if reactivated.any():
                count = int(reactivated.sum())
                pool[reactivated] = True
                protected[reactivated] = True
                reactivated_total += count

            missing = pool & ~working & lp.pos
            candidates = missing & (v_original > pricing_tol)
            last_violation = float(v_original[missing].max(initial=0.0))
            support = np.zeros(n, dtype=bool)
            support[columns[real_x_original > support_tol]] = True
            support |= protected

            stagnant = False
            if previous is not None:
                objective_stalled = (
                    abs(objective - previous["objective"])
                    <= stagnation_tol * (1.0 + abs(objective))
                )
                phase2_stall_streak = (
                    phase2_stall_streak + 1 if objective_stalled else 0
                )
                stagnant = objective_stalled
            restore_pool = (
                cg_full_pool_after > 0
                and phase2_stall_streak >= cg_full_pool_after
                and int(working.sum()) < int(pool.sum())
            )
            if restore_pool:
                entering = np.flatnonzero(pool & ~working)
                next_working = pool.copy()
                phase2_stall_streak = 0
            else:
                adaptive_batch = pricing_batch * (
                    2 ** min(phase2_stall_streak, 6)
                )
                entering = _top_priced(v_original, candidates, adaptive_batch)
                next_working = support.copy()
                next_working[entering] = True
                if stagnant:
                    next_working |= working
            next_working |= protected

            added = int(np.count_nonzero(next_working & ~working))
            added_total += added
            eliminated = np.zeros(n, dtype=bool)
            T = np.inf
            if theta <= max(tol, 1e-9):
                T = _pricing_budget(
                    v_scaled, pool & ~working & lp.pos,
                    cert_u, scale, l1_budget,
                )
                guard = working | next_working | protected
                if np.isfinite(T):
                    eliminated |= (
                        pool & ~guard & lp.pos
                        & (v_original <= 0.0)
                        & (-v_original > T / elim_delta)
                    )

                # Algorithm 3's retrospective pass uses the previous dual
                # point and the certified objective drop to lower-bound its gap.
                if previous is not None and np.isfinite(common_cap) and common_cap > 0.0:
                    drop_margin = 100.0 * _EPS * (
                        1.0 + abs(previous["objective"]) + abs(objective)
                    )
                    delta_lb = max(
                        0.0,
                        previous["objective"] - objective - drop_margin,
                    )
                    epsilon = previous["epsilon"]
                    G = max(
                        0.0,
                        delta_lb / common_cap - (beta - 1) * epsilon,
                    )
                    H = (
                        (beta - 1) * epsilon * common_cap - delta_lb
                    ) / elim_delta
                    pv = previous["v"]
                    retro_guard = guard | previous["working"]
                    eliminated |= (
                        pool & ~retro_guard & lp.pos
                        & (((pv > 0.0) & (pv < G))
                           | ((pv <= 0.0) & (-pv > H)))
                    )

            eliminated_count = int(eliminated.sum())
            if eliminated_count:
                pool[eliminated] = False
                next_working[eliminated] = False
                eliminated_total += eliminated_count

            history.append(dict(
                phase=2, round=phase2_rounds, objective=objective,
                theta=theta, working=int(working.sum()), pool=int(pool.sum()),
                priced=last_violation, budget=T, added=added,
                eliminated=eliminated_count,
                restored=restore_pool,
            ))
            if printlevel:
                budget_text = f"{T:8.2e}" if np.isfinite(T) else "     inf"
                print(
                    f"  CG phase II {phase2_rounds:2d} | obj {objective: .8e}  "
                    f"theta {theta:7.1e}  price {last_violation:7.1e}  "
                    f"T {budget_text}  W {int(working.sum()):7d}/{n:<7d}  "
                    f"+{added} -{eliminated_count}  "
                    f"{'restore  ' if restore_pool else ''}"
                    f"t {time.perf_counter() - started:6.1f}s"
                )

            if theta <= max(tol, 1e-9) and not candidates.any():
                status = "optimal"
                break
            if theta > max(tol, 1e-9) and not candidates.any():
                artificial_cost *= 10.0
                if not np.isfinite(artificial_cost):
                    status = "infeasible"
                    break
                continue

            previous = dict(
                objective=objective,
                support=support,
                v=v_original.copy(),
                epsilon=max(0.0, float(v_original[pool].max(initial=0.0))),
                working=working.copy(),
            )
            working = next_working
        else:
            status = "cg-round-limit"
    elif last_local is None:
        status = "time-limit"
    elif last_local.status != "optimal":
        status = last_local.status
    else:
        status = "infeasible"

    final_local = (
        last_phase2_local if last_phase2_local is not None else last_local
    )
    final_columns = (
        last_phase2_columns
        if last_phase2_columns is not None
        else last_columns
    )
    if final_local is None:
        x = np.zeros(n)
        y = np.zeros(m)
        z = np.zeros(n)
        w = np.zeros(n)
    else:
        x = np.zeros(n)
        z = np.zeros(n)
        w = np.zeros(n)
        y = final_local.y.copy()
        real_count = final_columns.size
        x[final_columns] = final_local.x[:real_count]
        z[final_columns] = final_local.z[:real_count]
        w[final_columns] = final_local.w[:real_count]
        missing = np.ones(n, dtype=bool)
        missing[final_columns] = False
        slack = lp.c - np.asarray(lp.A.T @ y).ravel()
        z[missing & lp.pos] = np.maximum(0.0, slack[missing & lp.pos])

    rp = lp.b - lp.A @ x
    rd = lp.c - lp.A.T @ y - z + w
    dp = 1.0 + float(np.abs(lp.b).max(initial=0.0))
    dd = 1.0 + float(np.abs(lp.c).max(initial=0.0))
    s = np.ones(n)
    if lp.box.any():
        s[lp.box] = np.maximum(lp.u[lp.box] - x[lp.box], 0.0)
    pairs = int(lp.pos.sum() + lp.box.sum())
    mu = (
        float(x[lp.pos] @ z[lp.pos] + s[lp.box] @ w[lp.box]) / pairs
        if pairs else 0.0
    )
    sol = Solution(x=x, y=y, z=z, w=w, status=status)
    sol.obj = float(lp.c @ x) + lp.obj_const
    sol.outer_iter = total_outer
    sol.inner_iter = total_inner
    sol.time = time.perf_counter() - started
    sol.t_fact, sol.t_solve = total_fact, total_solve
    sol.res_p = float(np.abs(rp).max(initial=0.0)) / dp
    sol.res_d = float(np.abs(rd).max(initial=0.0)) / dd
    sol.mu = mu
    sol.history = all_history
    sol.n_not_pd = total_not_pd
    sol.n_shifted = total_shifted
    sol.n_refine = total_refine
    sol.cg_iterations = phase2_rounds
    sol.cg_phase1_iterations = phase1_rounds
    sol.cg_columns_added = added_total
    sol.cg_columns_eliminated = eliminated_total
    sol.cg_columns_reactivated = reactivated_total
    sol.cg_pool_remaining = int(pool.sum())
    sol.cg_working_columns = int(final_columns.size)
    sol.cg_max_pricing_violation = last_violation
    sol.cg_artificial_value = last_theta
    sol.cg_history = history
    if printlevel:
        budget_text = f"{l1_budget:.6e}" if np.isfinite(l1_budget) else "inf"
        print(
            f"  -> retrospective {sol.status}: obj {sol.obj:.10e}, "
            f"phase I/II {phase1_rounds}/{phase2_rounds}, "
            f"{sol.time:.1f}s (L1 budget {budget_text})"
        )
        print(
            f"     pool: W {sol.cg_working_columns}/{n}, "
            f"{sol.cg_columns_added} added, {sol.cg_columns_eliminated} eliminated, "
            f"{sol.cg_columns_reactivated} reactivated"
        )
    return sol


def _warm_initial_mask(lp, target):
    """Build a feasible-looking structural support biased toward cheap columns."""
    m, n = lp.A.shape
    # A finite dummy budget suppresses the retrospective solver's rule that
    # protects every uncapped lower-only column. Such columns are priceable in
    # the warm sparse method and need not all enter the initial support.
    working = _initial_working_mask(lp, m, 0.0)
    target = min(n, max(int(target), int(working.sum())))
    need = target - int(working.sum())
    if need > 0:
        available = np.flatnonzero(~working)
        costs = lp.c[available] / lp.scale[available]
        if need < available.size:
            chosen = available[np.argpartition(costs, need - 1)[:need]]
        else:
            chosen = available
        working[chosen] = True
    return working


def _covering_crash_mask(lp):
    """Return a provably feasible support for a canonical covering LP.

    A recognized row has positive demand, nonnegative structural columns, and
    one unbounded zero-cost negative singleton surplus.  For an unbounded
    nonnegative-cost structural column, ``max_i b_i / a_ij`` over its nonzero
    rows is a valid constructive cap: reducing a larger value to that cap
    preserves coverage and cannot worsen the objective.  Selecting one column
    that can meet each demand at an explicit or constructive cap, then using
    the surplus variables for overcoverage, proves restricted-master
    feasibility without solving an artificial Phase-I problem.  Constructive
    caps are not installed as solver bounds, so this does not alter the LP.
    """
    A = lp.A.tocsc(copy=True)
    A.eliminate_zeros()
    m, n = A.shape
    if m == 0 or np.any(lp.b <= 0.0) or np.any(~lp.pos):
        return None

    col_nnz = np.diff(A.indptr)
    singleton = np.flatnonzero(col_nnz == 1)
    surplus = np.full(m, -1, dtype=np.int64)
    surplus_value = np.zeros(m)
    cost_scale = 1.0 + float(np.abs(lp.c).max(initial=0.0))
    zero_cost = np.abs(lp.c[singleton]) <= 100.0 * _EPS * cost_scale
    unbounded = ~np.isfinite(lp.u[singleton])
    candidates = singleton[zero_cost & unbounded]
    for j in candidates:
        p = A.indptr[j]
        row = int(A.indices[p])
        value = float(A.data[p])
        if value < 0.0 and (
            surplus[row] < 0 or abs(value) > abs(surplus_value[row])
        ):
            surplus[row] = j
            surplus_value[row] = value
    if np.any(surplus < 0):
        return None

    is_surplus = np.zeros(n, dtype=bool)
    is_surplus[surplus] = True
    structural = ~is_surplus
    if np.any(A.data < -100.0 * _EPS):
        coo = A.tocoo(copy=False)
        if np.any((coo.data < -100.0 * _EPS) & structural[coo.col]):
            return None

    effective_u = lp.u.copy()
    col_nonempty = col_nnz > 0
    starts = A.indptr[:-1][col_nonempty]
    implied_u = np.full(n, np.inf)
    if starts.size:
        ratios = lp.b[A.indices] / A.data
        implied_u[col_nonempty] = np.maximum.reduceat(ratios, starts)
    nonnegative_cost = lp.c >= -100.0 * _EPS * cost_scale
    implied = structural & ~np.isfinite(effective_u) & nonnegative_cost
    effective_u[implied] = implied_u[implied]
    finite_cap = structural & np.isfinite(effective_u) & (effective_u > 0.0)
    best = np.full(m, -1, dtype=np.int64)
    best_score = np.full(m, np.inf)
    Ar = A.tocsr()
    feasibility_tol = 100.0 * _EPS * (1.0 + np.abs(lp.b))
    for i in range(m):
        start, end = Ar.indptr[i : i + 2]
        jj = Ar.indices[start:end]
        aa = Ar.data[start:end]
        usable = finite_cap[jj] & (aa > 0.0)
        if not usable.any():
            return None
        jj = jj[usable]
        aa = aa[usable]
        contribution = aa * effective_u[jj]
        capable = contribution + feasibility_tol[i] >= lp.b[i]
        if not capable.any():
            return None
        jj = jj[capable]
        contribution = contribution[capable]
        cap_cost = np.maximum(lp.c[jj] * effective_u[jj], 0.0)
        score = cap_cost / np.maximum(
            np.minimum(contribution, lp.b[i]), _POS_FLOOR
        )
        local = int(np.argmin(score))
        best[i] = jj[local]
        best_score[i] = score[local]

    selected = np.unique(best)
    coverage = np.asarray(A[:, selected] @ effective_u[selected]).ravel()
    over = coverage - lp.b
    if np.any(over < -feasibility_tol):
        return None
    surplus_x = over / (-surplus_value)
    if np.any(~np.isfinite(surplus_x)) or np.any(surplus_x < 0.0):
        return None

    working = np.zeros(n, dtype=bool)
    working[selected] = True
    working[surplus] = True
    return working


def _mapped_rmp_start(
    lp,
    old_columns,
    old_solution,
    new_columns,
    old_artificial=True,
    new_artificial=True,
):
    """Map a central RMP point across support and Phase-I changes."""
    if old_solution is None or old_columns is None:
        return None

    nnew = new_columns.size + int(new_artificial)
    x = np.zeros(nnew)
    z = np.zeros(nnew)
    w = np.zeros(nnew)
    y = old_solution.y.copy()

    loc = np.searchsorted(old_columns, new_columns)
    clipped = np.minimum(loc, max(0, old_columns.size - 1))
    common = (loc < old_columns.size) & (old_columns[clipped] == new_columns)
    if common.any():
        source = loc[common]
        x[np.flatnonzero(common)] = old_solution.x[source]
        z[np.flatnonzero(common)] = old_solution.z[source]
        w[np.flatnonzero(common)] = old_solution.w[source]

    if new_artificial:
        if old_artificial:
            x[-1] = old_solution.x[-1]
            z[-1] = old_solution.z[-1]
            w[-1] = old_solution.w[-1]
        else:
            mu = max(float(old_solution.mu), 1e-10)
            x[-1] = np.sqrt(min(mu, 1.0))
            z[-1] = mu / max(x[-1], 1e-12)

    new = ~common
    if new.any():
        jj = new_columns[new]
        # In Phase I, theta is the last RMP variable.  Do not let a restricted
        # solve drive the insertion scale to zero while theta still certifies
        # substantial infeasibility: such columns would be numerically present
        # but unable to move.  The floor disappears continuously with theta.
        artificial_value = (
            max(float(old_solution.x[-1]), 0.0) if old_artificial else 0.0
        )
        feasibility_mu = 0.1 * min(1.0, artificial_value)
        mu = max(float(old_solution.mu), feasibility_mu, 1e-10)
        central = np.sqrt(min(mu, 1.0))
        xn = np.full(jj.size, max(central, 1e-8))
        boxed = lp.box[jj]
        if boxed.any():
            ub = lp.u[jj[boxed]]
            floor = np.maximum(1e-12, _POS_FLOOR * (1.0 + np.abs(ub)))
            xn[boxed] = np.minimum(np.maximum(xn[boxed], floor), 0.5 * ub)
        target = np.flatnonzero(new)
        x[target] = xn
        z[target] = mu / np.maximum(xn, 1e-12)
        if boxed.any():
            slack = lp.u[jj[boxed]] - xn[boxed]
            w[target[boxed]] = mu / np.maximum(slack, 1e-12)

    return x, y, z, w


def _expand_rmp_solution(lp, columns, local, artificial=True):
    """Expand an RMP point and give inactive lower-bound columns valid slacks."""
    n = lp.A.shape[1]
    x = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(n)
    stop = -1 if artificial else None
    x[columns] = local.x[:stop]
    z[columns] = local.z[:stop]
    w[columns] = local.w[:stop]
    reduced = lp.c - np.asarray(lp.A.T @ local.y).ravel()
    inactive = np.ones(n, dtype=bool)
    inactive[columns] = False
    z[inactive & lp.pos] = np.maximum(reduced[inactive & lp.pos], 0.0)
    return x, local.y.copy(), z, w


def _repair_warm_duals(rmp, warm, mu):
    """Recenter dual slacks after changing the restricted-master objective."""
    x, y, z, w = (v.copy() for v in warm)
    reduced = rmp.c - np.asarray(rmp.A.T @ y).ravel()
    central = np.sqrt(max(float(mu), 1e-10))
    z[:] = 0.0
    w[:] = 0.0
    lower_only = rmp.pos & ~rmp.box
    z[lower_only] = np.maximum(reduced[lower_only], 0.0) + central
    if rmp.box.any():
        # z - w = reduced exactly, while both bound multipliers stay interior.
        z[rmp.box] = np.maximum(reduced[rmp.box], 0.0) + central
        w[rmp.box] = np.maximum(-reduced[rmp.box], 0.0) + central
    return x, y, z, w


def _standard_kkt_metrics(lp, x, y, z, w):
    """Measure the unregularized full-pool KKT equations in solver scaling."""
    rp = lp.b - lp.A @ x
    rd = lp.c - lp.A.T @ y - z + w
    pden = 1.0 + float(np.abs(lp.b).max(initial=0.0))
    dden = 1.0 + float(np.abs(lp.c).max(initial=0.0))
    pinf = float(np.abs(rp).max(initial=0.0)) / pden
    dinf = float(np.abs(rd).max(initial=0.0)) / dden
    primal = float(lp.c @ x) + lp.obj_const
    dual = float(lp.b @ y) + lp.obj_const
    if lp.box.any():
        dual -= float(lp.u[lp.box] @ w[lp.box])
    gap = abs(primal - dual) / (1.0 + abs(primal))
    return pinf, dinf, gap, primal


def _sparse_step_cadence(A, requested, clique_threshold):
    """Select one or two RMP steps from predicted median Schur clique work."""
    if clique_threshold <= 0.0:
        raise ValueError("sparse step clique threshold must be positive")
    col_nnz = np.diff(A.indptr).astype(np.float64)
    clique_burden = 0.5 * col_nnz * np.maximum(col_nnz - 1.0, 0.0)
    median_burden = float(np.median(clique_burden)) if clique_burden.size else 0.0
    if requested is None or int(requested) == 0:
        steps = 2 if median_burden >= clique_threshold else 1
    else:
        steps = int(requested)
        if steps <= 0:
            raise ValueError("sparse steps must be positive or zero for automatic")
    return steps, median_burden, col_nnz, clique_burden


def warm_sparse_solve(
    lp: StandardLP,
    tol: float = 1e-6,
    time_limit: float = np.inf,
    printlevel: int = 1,
    sparse_initial_factor: float = 2.0,
    sparse_steps: Optional[int] = None,
    sparse_step_clique_threshold: float = 120.0,
    sparse_inner_maxit: int = 1,
    sparse_pricing_batch: Optional[int] = None,
    sparse_pricing_tol: Optional[float] = None,
    sparse_support_tol: Optional[float] = None,
    sparse_drop_mu: float = 1e-3,
    sparse_drop_start: Optional[float] = None,
    sparse_drop_patience: int = 3,
    sparse_drop_hysteresis: float = 10.0,
    sparse_drop_batch: Optional[int] = None,
    sparse_max_rounds: int = 200,
    sparse_fallback_full: bool = False,
    sparse_artificial_cost: Optional[float] = None,
    sparse_cover_crash: bool = True,
    sparse_fill_weight: float = 0.2,
    **solver_kwargs,
) -> Solution:
    """Warm sparse IPM: a few proximal Newton steps between support updates.

    This generalizes the SparseIPM pattern of Zanetti--Gondzio: the support is
    priced after a small number of regularized IPM steps, retained variables
    keep their primal-dual state, and new variables enter at central values.
    Full reduced costs and full-pool KKT residuals are checked every round.
    """
    started = time.perf_counter()
    deadline = started + time_limit
    m, n = lp.A.shape
    if sparse_initial_factor <= 0.0:
        raise ValueError("sparse support factor must be positive")
    if sparse_inner_maxit <= 0 or sparse_max_rounds <= 0:
        raise ValueError("sparse iteration limits must be positive")
    if sparse_fill_weight < 0.0:
        raise ValueError("sparse fill weight must be nonnegative")
    if sparse_drop_patience <= 0:
        raise ValueError("sparse drop patience must be positive")
    if sparse_drop_hysteresis < 1.0:
        raise ValueError("sparse drop hysteresis must be at least one")

    if sparse_pricing_tol is None:
        dual_denominator = 1.0 + float(np.abs(lp.c).max(initial=0.0))
        pricing_threshold = max(tol, 1e-9) * dual_denominator / lp.scale
    else:
        pricing_threshold = np.full(n, float(sparse_pricing_tol))
    if np.any(pricing_threshold <= 0.0):
        raise ValueError("sparse pricing tolerance must be positive")
    support_tol = (
        max(0.1 * tol, 1e-10)
        if sparse_support_tol is None
        else float(sparse_support_tol)
    )
    drop_start = (
        max(100.0 * tol, 1e-4)
        if sparse_drop_start is None
        else float(sparse_drop_start)
    )
    pricing_batch = (
        max(1000, m)
        if sparse_pricing_batch is None
        else int(sparse_pricing_batch)
    )
    drop_batch = (
        max(1000, m // 2)
        if sparse_drop_batch is None
        else int(sparse_drop_batch)
    )
    if drop_batch <= 0:
        raise ValueError("sparse drop batch must be positive")
    sparse_steps, median_clique, col_nnz, clique_burden = _sparse_step_cadence(
        lp.A, sparse_steps, sparse_step_clique_threshold
    )
    fill_penalty = 1.0 + float(sparse_fill_weight) * clique_burden
    initial_target = min(n, max(m, int(np.ceil(sparse_initial_factor * m))))
    crash = _covering_crash_mask(lp) if sparse_cover_crash else None
    working = (
        crash.copy() if crash is not None else _warm_initial_mask(lp, initial_target)
    )
    protected = ~lp.pos

    max_cost = float(np.abs(lp.c / lp.scale).max(initial=0.0))
    artificial_cost = (
        max(1e4, 100.0 * (1.0 + max_cost))
        if sparse_artificial_cost is None
        else float(sparse_artificial_cost)
    )
    if artificial_cost <= 0.0:
        raise ValueError("sparse artificial cost must be positive")

    zero_cost = np.zeros(n)
    phase = 2 if crash is not None else 1
    old_columns = None
    old_local = None
    last_columns = np.flatnonzero(working)
    last_local = None
    phase1_rounds = phase2_rounds = 0
    total_outer = total_inner = 0
    total_fact = total_solve = 0.0
    total_added = total_dropped = 0
    total_not_pd = total_shifted = total_refine = 0
    max_low_rank_columns = max_low_rank_edges = 0
    drop_age = np.zeros(n, dtype=np.int32)
    history = []
    status = "sparse-round-limit"
    last_price = np.inf
    last_theta = 0.0 if crash is not None else np.inf
    repair_duals = False
    old_artificial = crash is None
    last_artificial = crash is None

    if crash is not None and printlevel:
        print(
            f"  covering crash : feasible support {int(working.sum())}/{n}; "
            "skipping Phase I"
        )
    if printlevel:
        print(
            f"  pricing cadence: {sparse_steps} RMP step(s), median clique "
            f"burden {median_clique:.1f}"
        )

    local_kwargs = dict(solver_kwargs)
    local_kwargs.pop("maxit", None)
    local_kwargs.pop("inner_maxit", None)
    local_kwargs.pop("time_limit", None)
    local_kwargs.pop("initial", None)

    for round_index in range(1, sparse_max_rounds + 1):
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            status = "time-limit"
            break
        columns = np.flatnonzero(working)
        artificial = phase == 1
        warm = _mapped_rmp_start(
            lp,
            old_columns,
            old_local,
            columns,
            old_artificial=old_artificial,
            new_artificial=artificial,
        )
        if artificial:
            rmp = _restricted_master(
                lp, columns, zero_cost, 1.0, lp.obj_const
            )
        else:
            rmp = _restricted_master_without_artificial(
                lp, columns, lp.c, lp.obj_const
            )
        if repair_duals and warm is not None:
            warm = _repair_warm_duals(rmp, warm, old_local.mu)
            repair_duals = False
        local = psipm_solve(
            rmp,
            tol=tol,
            maxit=sparse_steps,
            inner_maxit=sparse_inner_maxit,
            time_limit=remaining,
            printlevel=max(0, printlevel - 1),
            initial=warm,
            **local_kwargs,
        )
        # A deliberately short local solve normally exits between outer-loop
        # checks, so Solution.mu still describes the point at the beginning of
        # its last proximal step.  Support updates need the returned point.
        pairs = int(rmp.pos.sum() + rmp.box.sum())
        local_mu = 0.0
        if pairs:
            local_mu = float(local.x[rmp.pos] @ local.z[rmp.pos])
            if rmp.box.any():
                slack = rmp.u[rmp.box] - local.x[rmp.box]
                local_mu += float(slack @ local.w[rmp.box])
            local_mu = max(0.0, local_mu / pairs)
        local.mu = local_mu
        total_outer += local.outer_iter
        total_inner += local.inner_iter
        total_fact += local.t_fact
        total_solve += local.t_solve
        total_not_pd += local.n_not_pd
        total_shifted += local.n_shifted
        total_refine += local.n_refine
        max_low_rank_columns = max(
            max_low_rank_columns, local.low_rank_columns
        )
        max_low_rank_edges = max(
            max_low_rank_edges, local.low_rank_candidate_edges
        )
        last_columns, last_local = columns, local
        last_artificial = artificial
        old_columns, old_local = columns, local
        old_artificial = artificial

        if local.status in {"time-limit", "ill-conditioned", "diverged"}:
            status = local.status
            break

        theta = max(0.0, float(local.x[-1])) if artificial else 0.0
        last_theta = theta
        if phase == 1:
            phase1_rounds += 1
            violation_scaled = np.asarray(lp.A.T @ local.y).ravel()
        else:
            phase2_rounds += 1
            violation_scaled = np.asarray(lp.A.T @ local.y).ravel() - lp.c
        violation = violation_scaled / lp.scale
        missing = ~working & lp.pos
        candidates = missing & (violation > pricing_threshold)
        last_price = float(violation[missing].max(initial=0.0))

        stop = -1 if artificial else None
        x_original = lp.scale[columns] * local.x[:stop]
        mu = max(float(local.mu), 0.0)
        pinf = dinf = gap = np.inf
        objective = np.nan
        if phase == 2:
            full_x, full_y, full_z, full_w = _expand_rmp_solution(
                lp, columns, local, artificial=False
            )
            pinf, dinf, gap, objective = _standard_kkt_metrics(
                lp, full_x, full_y, full_z, full_w
            )

        history.append(
            dict(
                phase=phase,
                round=round_index,
                working=int(working.sum()),
                theta=theta,
                mu=mu,
                pricing=last_price,
                pinf=pinf,
                dinf=dinf,
                gap=gap,
                objective=objective,
                added=0,
                dropped=0,
                drop_eligible=0,
                drop_mature=0,
            )
        )
        if printlevel:
            phase_text = "I" if phase == 1 else "II"
            quality = (
                ""
                if phase == 1
                else f" rp {pinf:7.1e} rd {dinf:7.1e} gap {gap:7.1e}"
            )
            print(
                f"  warm sparse {phase_text:>2} {round_index:3d} | "
                f"theta {theta:7.1e} mu {mu:7.1e} price {last_price:7.1e} "
                f"W {int(working.sum()):7d}/{n:<7d}{quality} "
                f"t {time.perf_counter() - started:6.1f}s"
            )

        phase_tol = max(tol, 1e-9)
        if phase == 1 and theta <= phase_tol:
            phase = 2
            repair_duals = True
            old_columns, old_local = columns, local
            continue

        if (
            phase == 2
            and theta <= phase_tol
            and not candidates.any()
            and pinf <= tol
            and dinf <= tol
            and gap <= tol
        ):
            status = "optimal"
            break

        pricing_merit = violation / fill_penalty
        entering = _top_priced(pricing_merit, candidates, pricing_batch)
        next_working = working.copy()
        next_working[entering] = True
        drop_age[entering] = 0

        dropped = np.empty(0, dtype=np.int64)
        drop_gate = (
            phase == 2
            and theta <= phase_tol
            and mu <= sparse_drop_mu
            and max(pinf, dinf, gap) <= drop_start
        )
        if drop_gate:
            eligible_local = (
                lp.pos[columns]
                & (x_original <= support_tol)
                & (
                    violation[columns]
                    < -sparse_drop_hysteresis * pricing_threshold[columns]
                )
            )
            eligible = columns[eligible_local]
            drop_age[columns[~eligible_local]] = 0
            drop_age[eligible] += 1
            removable = eligible[drop_age[eligible] >= sparse_drop_patience]
            history[-1]["drop_eligible"] = int(eligible.size)
            history[-1]["drop_mature"] = int(removable.size)
            if removable.size:
                max_drop = min(
                    drop_batch,
                    removable.size,
                    max(0, int(next_working.sum()) - max(m, entering.size)),
                )
                if max_drop > 0:
                    removable_local = np.searchsorted(columns, removable)
                    merit = x_original[removable_local]
                    if max_drop < removable.size:
                        selected = np.argpartition(merit, max_drop - 1)[:max_drop]
                        dropped = removable[selected]
                    else:
                        dropped = removable
                    next_working[dropped] = False
                    drop_age[dropped] = 0
        else:
            drop_age[columns] = 0

        next_working |= protected
        added = int(np.count_nonzero(next_working & ~working))
        removed = int(np.count_nonzero(working & ~next_working))
        history[-1]["added"] = added
        history[-1]["dropped"] = removed
        total_added += added
        total_dropped += removed

        if (
            not entering.size
            and not removed
            and phase == 1
            and local.status == "optimal"
        ):
            status = "infeasible"
            break
        working = next_working
    else:
        status = "sparse-round-limit"

    if last_local is None:
        sol = Solution(
            x=np.zeros(n), y=np.zeros(m), z=np.zeros(n), w=np.zeros(n)
        )
    else:
        x, y, z, w = _expand_rmp_solution(
            lp, last_columns, last_local, artificial=last_artificial
        )
        sol = Solution(x=x, y=y, z=z, w=w)
        sol.mu = last_local.mu

    if sparse_fallback_full and status != "optimal" and last_local is not None:
        remaining = deadline - time.perf_counter()
        if remaining > 0.0 and last_theta <= max(tol, 1e-9):
            if printlevel:
                print("        warm sparse fallback: restoring the full master")
            initial = (sol.x, sol.y, sol.z, sol.w)
            full = psipm_solve(
                lp,
                tol=tol,
                time_limit=remaining,
                printlevel=printlevel,
                initial=initial,
                **local_kwargs,
            )
            full.cg_phase1_iterations = phase1_rounds
            full.cg_iterations = phase2_rounds
            full.cg_columns_added = total_added
            full.cg_columns_eliminated = total_dropped
            full.cg_working_columns = n
            full.cg_pool_remaining = n
            full.cg_history = history
            full.cg_sparse_steps = sparse_steps
            full.cg_median_clique_burden = median_clique
            full.time = time.perf_counter() - started
            return full

    sol.status = status
    sol.outer_iter = total_outer
    sol.inner_iter = total_inner
    sol.time = time.perf_counter() - started
    sol.t_fact = total_fact
    sol.t_solve = total_solve
    sol.n_not_pd = total_not_pd
    sol.n_shifted = total_shifted
    sol.n_refine = total_refine
    sol.low_rank_columns = max_low_rank_columns
    sol.low_rank_candidate_edges = max_low_rank_edges
    sol.obj = float(lp.c @ sol.x) + lp.obj_const
    sol.cg_phase1_iterations = phase1_rounds
    sol.cg_iterations = phase2_rounds
    sol.cg_columns_added = total_added
    sol.cg_columns_eliminated = total_dropped
    sol.cg_pool_remaining = n
    sol.cg_working_columns = int(last_columns.size)
    sol.cg_final_columns = last_columns.copy()
    sol.cg_max_pricing_violation = last_price
    sol.cg_artificial_value = last_theta
    sol.cg_history = history
    sol.cg_sparse_steps = sparse_steps
    sol.cg_median_clique_burden = median_clique
    if last_local is not None:
        sol.res_p, sol.res_d, _, _ = _standard_kkt_metrics(
            lp, sol.x, sol.y, sol.z, sol.w
        )
    if printlevel:
        print(
            f"  -> warm sparse {sol.status}: obj {sol.obj:.10e}, "
            f"phase I/II {phase1_rounds}/{phase2_rounds}, "
            f"W {sol.cg_working_columns}/{n}, {sol.time:.1f}s"
        )
    return sol


def _repair_start_singletons(A, b, x, lower_only):
    """Use unique singleton lower slacks to reduce the startup residual."""
    columns = np.flatnonzero(lower_only)
    if not columns.size:
        return x
    counts = np.diff(A.indptr)[columns]
    columns = columns[counts == 1]
    if not columns.size:
        return x
    positions = A.indptr[columns]
    rows = A.indices[positions]
    values = A.data[positions]
    unique_rows, multiplicity = np.unique(rows, return_counts=True)
    unique = np.isin(rows, unique_rows[multiplicity == 1])
    columns, rows, values = columns[unique], rows[unique], values[unique]
    if not columns.size:
        return x
    residual = b - A @ x
    repaired = x[columns] + residual[rows] / values
    accepted = np.isfinite(repaired) & (repaired > _POS_FLOOR)
    x[columns[accepted]] = repaired[accepted]
    return x


def _start_primal_score(A, b, x):
    """Scale-free score used to choose between two strictly interior starts."""
    residual = np.asarray(b - A @ x).ravel()
    inf_score = float(np.abs(residual).max(initial=0.0)) / (
        1.0 + float(np.abs(b).max(initial=0.0))
    )
    two_score = float(np.linalg.norm(residual)) / (
        1.0 + float(np.linalg.norm(b))
    )
    return max(inf_score, two_score)


def _starting_point(
    A,
    At,
    b,
    c,
    u,
    lp,
    ipos,
    ibox,
    solver,
    kkt,
    projection="adaptive",
    bound_fraction=1e-3,
    linear_tolerance=None,
):
    """HiGHS/IPX-like starting point, adapted to this bounded formulation.

    HiGHS' IPX forms least-norm primal and least-squares dual estimates, shifts
    the lower/upper bound slacks positive, then levels the complementarity
    products.  IPX stores independent lower and upper slacks; here boxed
    variables are represented by one primal variable with ``s = u - x``.  For
    those columns the shifted lower/upper slacks are projected back to a
    consistent ``x`` by preserving their ratio.
    """
    m, n = A.shape
    ones_n = np.ones(n)
    zeros_n = np.zeros(n)
    pos = lp.pos
    box = lp.box
    lower_only = pos & ~box
    x, y = zeros_n.copy(), np.zeros(m)
    saved_cg_tol = getattr(solver, "current_cg_tol", None)
    relaxed_start = linear_tolerance is not None
    if relaxed_start and saved_cg_tol is not None:
        # Startup estimates need only be useful enough to reduce infeasibility.
        # Asking an iterative KKT solver for the final Newton accuracy can hit
        # its iteration limit and used to discard both estimates on large LPs.
        start_tol = min(
            float(linear_tolerance),
            float(getattr(solver, "cg_tol_max", linear_tolerance)),
        )
        solver.current_cg_tol = max(float(saved_cg_tol), start_tol)
    if relaxed_start:
        try:
            solver.factorize(ones_n, 1.0)  # M = A A^T + I, well conditioned
        except Exception:
            pass
        else:
            # Keep whichever estimate succeeds.  Primal and dual startup
            # systems are independent conveniences.
            try:
                x = At @ solver.solve(zeros_n, b)[1]
            except Exception:
                x = zeros_n.copy()
            try:
                y = solver.solve(zeros_n, A @ c)[1]
            except Exception:
                y = np.zeros(m)
    else:
        # Preserve the established trajectory by retaining the legacy
        # all-or-nothing startup when no relaxed tolerance was requested.
        # Primal and dual startup systems are deliberately coupled in this
        # compatibility path.
        try:
            solver.factorize(ones_n, 1.0)
            x = At @ solver.solve(zeros_n, b)[1]
            y = solver.solve(zeros_n, A @ c)[1]
        except Exception:
            x = zeros_n.copy()
            y = np.zeros(m)
    if relaxed_start and saved_cg_tol is not None:
        solver.current_cg_tol = saved_cg_tol
    if not np.isfinite(x).all():
        x = zeros_n.copy()
    if not np.isfinite(y).all():
        y = np.zeros(m)

    least_norm_x = x.copy()

    # --- primal shifts ----------------------------------------------------- #
    xl = np.zeros(n)
    xu = np.zeros(n)
    if pos.any():
        xl[pos] = x[pos]
    if box.any():
        xu[box] = u[box] - x[box]

    xinfeas = 0.0
    if pos.any():
        xinfeas = max(xinfeas, -float(np.min(xl[pos], initial=0.0)))
    if box.any():
        xinfeas = max(xinfeas, -float(np.min(xu[box], initial=0.0)))
    xshift1 = 1.0 + 1.5 * xinfeas
    if pos.any():
        xl[pos] += xshift1
    if box.any():
        xu[box] += xshift1

    # --- dual --------------------------------------------------------------#
    cnorm = float(np.linalg.norm(c))
    if cnorm > 0.0:
        r = c - At @ y
        if float(np.linalg.norm(r)) < 0.05 * cnorm:
            y *= 0.95
            r = c - At @ y
    else:
        r = zeros_n.copy()

    z = np.zeros(n)
    w = np.zeros(n)
    if cnorm == 0.0:
        z[pos] = 1.0
        w[box] = 1.0
    else:
        z[lower_only] = r[lower_only]
        if box.any():
            z[box] = 0.5 * r[box]
            w[box] = -0.5 * r[box]
        zinfeas = 0.0
        if pos.any():
            zinfeas = max(zinfeas, -float(np.min(z[pos], initial=0.0)))
        if box.any():
            zinfeas = max(zinfeas, -float(np.min(w[box], initial=0.0)))
        zshift1 = 1.0 + 1.5 * zinfeas
        z[pos] += zshift1
        w[box] += zshift1

    # --- second shift: level complementarity products ---------------------- #
    xsum = 1.0
    zsum = 1.0
    mu0 = 1.0
    if pos.any():
        xsum += float(np.sum(xl[pos]))
        zsum += float(np.sum(z[pos]))
        mu0 += float(xl[pos] @ z[pos])
    if box.any():
        xsum += float(np.sum(xu[box]))
        zsum += float(np.sum(w[box]))
        mu0 += float(xu[box] @ w[box])
    xshift2 = 0.5 * mu0 / zsum if zsum > 0.0 else 0.0
    zshift2 = 0.5 * mu0 / xsum if xsum > 0.0 else 0.0
    if np.isfinite(xshift2) and xshift2 > 0.0:
        xl[pos] += xshift2
        xu[box] += xshift2
    if np.isfinite(zshift2) and zshift2 > 0.0:
        z[pos] += zshift2
        w[box] += zshift2

    # Project the shifted lower/upper slacks back to the representation used by
    # this solver.  For boxed variables, preserving the HiGHS lower/upper slack
    # ratio is the closest consistent point to the IPX start.
    if lower_only.any():
        x[lower_only] = np.maximum(xl[lower_only], 1e-12)
    if box.any():
        denom = xl[box] + xu[box]
        xb = np.where(denom > 0.0, u[box] * xl[box] / denom, 0.5 * u[box])
        x_floor, s_floor = _box_floors(u, ibox)
        x[box] = np.clip(xb, x_floor, u[box] - s_floor)

        # A single extreme lower-only slack can make the global IPX shift move
        # millions of boxed variables toward their midpoints.  Build a second
        # strictly interior candidate by clipping the least-norm estimate at a
        # small relative distance from each bound.  This is especially useful
        # for large set-partitioning/cardinality relaxations, whose solutions
        # are sparse.  Select it only when it measurably improves feasibility.
        clipped = least_norm_x.copy()
        lower_margin = np.maximum(x_floor, bound_fraction * u[box])
        upper_margin = np.maximum(s_floor, bound_fraction * u[box])
        clipped[box] = np.clip(
            clipped[box], lower_margin, u[box] - upper_margin
        )
        if lower_only.any():
            clipped[lower_only] = np.maximum(
                clipped[lower_only], _POS_FLOOR
            )
        clipped = _repair_start_singletons(A, b, clipped, lower_only)
        if (
            projection == "adaptive"
            and _start_primal_score(A, b, clipped)
            < _start_primal_score(A, b, x)
        ):
            x = clipped
    return x, y, z, w


# =========================================================================== #
#  Solution recovery / KKT check in the ORIGINAL variables
# =========================================================================== #
def recover(lp: StandardLP, raw: RawLP, sol: Solution):
    """Map ``(v, y, z, w)`` back to the original LP and measure its KKT error."""
    n_aug = lp.n_aug
    v_full = np.zeros(n_aug)
    v_full[lp.keep] = lp.scale * sol.x
    x_aug = lp.shift + lp.sign * v_full
    x = x_aug[: lp.n_orig]

    y = np.zeros(raw.A.shape[0])
    row_scale = lp.row_scale if lp.row_scale is not None else 1.0
    y[lp.rows_keep] = row_scale * sol.y

    # reduced costs of the original columns
    rc_std = np.zeros(n_aug)
    rc_std[lp.keep] = (sol.z - sol.w) / lp.scale
    rc = (rc_std / lp.sign)[: lp.n_orig]

    obj = float(raw.c @ x) + raw.offset * (-1.0 if raw.maximize else 1.0)
    if raw.maximize:
        obj = -obj

    Ax = raw.A @ x
    pinf = max(
        float(np.max(np.maximum(raw.rl - Ax, Ax - raw.ru), initial=0.0)),
        float(np.max(np.maximum(raw.cl - x, x - raw.cu), initial=0.0)),
    )
    fin = np.concatenate([raw.rl[np.abs(raw.rl) < INF], raw.ru[np.abs(raw.ru) < INF]])
    pinf /= 1.0 + float(np.abs(fin).max(initial=0.0))
    dres = raw.c - raw.A.T @ y - rc
    # columns eliminated because they were fixed carry a free multiplier
    fixed_mask = np.ones(n_aug, dtype=bool)
    fixed_mask[lp.keep] = False
    dres[fixed_mask[: lp.n_orig]] = 0.0
    dinf = float(np.abs(dres).max(initial=0.0)) / (1.0 + float(np.abs(raw.c).max(initial=0.0)))
    return dict(x=x, y=y, rc=rc, obj=obj, pinf=pinf, dinf=dinf)


def _info_from_highs_solution(raw: RawLP, highs_sol):
    """Measure a postsolved HiGHS solution in the original RawLP coordinates."""
    x = np.asarray(highs_sol.col_value, dtype=np.float64)
    y = np.asarray(highs_sol.row_dual, dtype=np.float64)
    rc = np.asarray(highs_sol.col_dual, dtype=np.float64)

    obj = float(raw.c @ x) + raw.offset * (-1.0 if raw.maximize else 1.0)
    if raw.maximize:
        obj = -obj

    Ax = raw.A @ x
    pinf = max(
        float(np.max(np.maximum(raw.rl - Ax, Ax - raw.ru), initial=0.0)),
        float(np.max(np.maximum(raw.cl - x, x - raw.cu), initial=0.0)),
    )
    fin = np.concatenate([raw.rl[np.abs(raw.rl) < INF], raw.ru[np.abs(raw.ru) < INF]])
    pinf /= 1.0 + float(np.abs(fin).max(initial=0.0))
    dres = raw.c - raw.A.T @ y - rc
    dinf = float(np.abs(dres).max(initial=0.0)) / (
        1.0 + float(np.abs(raw.c).max(initial=0.0))
    )
    return dict(x=x, y=y, rc=rc, obj=obj, pinf=pinf, dinf=dinf)


def _postsolve_info(highs, raw_pre: RawLP, raw_orig: RawLP, info_pre: dict):
    """Postsolve a presolved primal/dual solution through HiGHS."""
    import highspy

    x = np.asarray(info_pre["x"], dtype=np.float64)
    y = np.asarray(info_pre["y"], dtype=np.float64)
    rc = np.asarray(info_pre["rc"], dtype=np.float64)
    hs = highspy.HighsSolution()
    hs.col_value = x.tolist()
    hs.row_value = np.asarray(raw_pre.A @ x, dtype=np.float64).tolist()
    hs.col_dual = rc.tolist()
    hs.row_dual = y.tolist()
    hs.value_valid = True
    hs.dual_valid = True
    status = highs.postsolve(hs)
    if str(status).endswith("kError"):
        raise RuntimeError(f"HiGHS postsolve failed: {status}")
    return _info_from_highs_solution(raw_orig, highs.getSolution())


# =========================================================================== #
#  Convenience driver
# =========================================================================== #
def solve_mps(
    path: str,
    tol: float = 1e-6,
    scaling: int = 3,
    printlevel: int = 1,
    presolve: bool = False,
    retrospective_pool: bool = False,
    warm_sparse: bool = False,
    cg_initial_columns: Optional[int] = None,
    cg_pricing_batch: Optional[int] = None,
    cg_support_tol: Optional[float] = None,
    cg_pricing_tol: Optional[float] = None,
    cg_stagnation_tol: Optional[float] = None,
    cg_elim_delta: Optional[float] = None,
    cg_l1_budget: Optional[float] = None,
    cg_objective_upper: Optional[float] = None,
    cg_beta: Optional[int] = None,
    cg_max_rounds: int = 30,
    cg_full_pool_after: int = 5,
    cg_artificial_cost: Optional[float] = None,
    sparse_initial_factor: float = 2.0,
    sparse_steps: Optional[int] = None,
    sparse_step_clique_threshold: float = 120.0,
    sparse_inner_maxit: int = 1,
    sparse_pricing_batch: Optional[int] = None,
    sparse_pricing_tol: Optional[float] = None,
    sparse_support_tol: Optional[float] = None,
    sparse_drop_mu: float = 1e-3,
    sparse_drop_start: Optional[float] = None,
    sparse_drop_patience: int = 3,
    sparse_drop_hysteresis: float = 10.0,
    sparse_drop_batch: Optional[int] = None,
    sparse_max_rounds: int = 200,
    sparse_fallback_full: bool = False,
    sparse_artificial_cost: Optional[float] = None,
    sparse_cover_crash: bool = True,
    sparse_fill_weight: float = 0.2,
    **kwargs,
):
    t0 = time.perf_counter()
    presolver = None
    pstatus = None
    if presolve:
        raw_orig, raw, presolver, pstatus = read_mps_with_presolve(
            path, time_limit=kwargs.get("time_limit", np.inf)
        )
    else:
        raw_orig = raw = read_mps(path)
    if printlevel:
        print(f"PS-IPM  |  {path}")
        print(
            f"  original LP    : m = {raw_orig.A.shape[0]}, n = {raw_orig.A.shape[1]}, "
            f"nnz = {raw_orig.A.nnz}   (read {time.perf_counter() - t0:.2f}s)"
        )
        if presolve:
            print(f"  HiGHS presolve : {pstatus}")
            if raw is not raw_orig:
                print(
                    f"  presolved LP   : m = {raw.A.shape[0]}, n = {raw.A.shape[1]}, "
                    f"nnz = {raw.A.nnz}"
                )
    lp = to_standard_form(raw, verbose=bool(printlevel))
    lp = scale_problem(lp, mode=scaling, verbose=bool(printlevel))
    if retrospective_pool and warm_sparse:
        raise ValueError("choose either retrospective_pool or warm_sparse")
    if warm_sparse:
        sol = warm_sparse_solve(
            lp,
            tol=tol,
            printlevel=printlevel,
            sparse_initial_factor=sparse_initial_factor,
            sparse_steps=sparse_steps,
            sparse_step_clique_threshold=sparse_step_clique_threshold,
            sparse_inner_maxit=sparse_inner_maxit,
            sparse_pricing_batch=sparse_pricing_batch,
            sparse_pricing_tol=sparse_pricing_tol,
            sparse_support_tol=sparse_support_tol,
            sparse_drop_mu=sparse_drop_mu,
            sparse_drop_start=sparse_drop_start,
            sparse_drop_patience=sparse_drop_patience,
            sparse_drop_hysteresis=sparse_drop_hysteresis,
            sparse_drop_batch=sparse_drop_batch,
            sparse_max_rounds=sparse_max_rounds,
            sparse_fallback_full=sparse_fallback_full,
            sparse_artificial_cost=sparse_artificial_cost,
            sparse_cover_crash=sparse_cover_crash,
            sparse_fill_weight=sparse_fill_weight,
            **kwargs,
        )
    elif retrospective_pool:
        sol = retrospective_solve(
            lp,
            tol=tol,
            printlevel=printlevel,
            cg_initial_columns=cg_initial_columns,
            cg_pricing_batch=cg_pricing_batch,
            cg_support_tol=cg_support_tol,
            cg_pricing_tol=cg_pricing_tol,
            cg_stagnation_tol=cg_stagnation_tol,
            cg_elim_delta=cg_elim_delta,
            cg_l1_budget=cg_l1_budget,
            cg_objective_upper=cg_objective_upper,
            cg_beta=cg_beta,
            cg_max_rounds=cg_max_rounds,
            cg_full_pool_after=cg_full_pool_after,
            cg_artificial_cost=cg_artificial_cost,
            **kwargs,
        )
    else:
        sol = psipm_solve(lp, tol=tol, printlevel=printlevel, **kwargs)
    info_pre = recover(lp, raw, sol)
    info = (
        _postsolve_info(presolver, raw, raw_orig, info_pre)
        if presolve and raw is not raw_orig
        else info_pre
    )
    if presolve and raw is not raw_orig:
        info["presolved_obj"] = info_pre["obj"]
        info["presolved_pinf"] = info_pre["pinf"]
        info["presolved_dinf"] = info_pre["dinf"]
        info["presolve_status"] = str(pstatus)
        if sol.status == "optimal" and (
            info["pinf"] > tol or info["dinf"] > tol
        ):
            sol.status = "postsolve-inaccurate"
            if printlevel:
                print(
                    "  postsolve check : rejected optimal status "
                    f"(pinf {info['pinf']:.2e}, dinf {info['dinf']:.2e})"
                )
    sol.obj = info["obj"]
    return raw_orig, lp, sol, info


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PS-IPM for LP (Python port)")
    ap.add_argument("mps")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--pc", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument(
        "--kkt",
        default="normal",
        choices=(
            "normal", "dynamic", "core-pcg", "lagged-pcg", "augmented",
            "partial",
        ),
    )
    ap.add_argument("--scaling", type=int, default=3)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument(
        "--presolve", action="store_true",
        help="run HiGHS presolve before PS-IPM and postsolve the solution",
    )
    ap.add_argument("--stop", default="relative", choices=("relative", "psipm"))
    ap.add_argument("--maxit", type=int, default=200)
    ap.add_argument(
        "--time-limit", type=float, default=2000.0,
        help="solver time limit in seconds (default: 2000)",
    )
    ap.add_argument("--printlevel", type=int, default=1)
    ap.add_argument(
        "--refine", type=int, default=2,
        help="max iterative-refinement steps on the normal equations (0 = off)",
    )
    ap.add_argument(
        "--low-rank-max-columns", type=int, default=8,
        help="maximum clique-producing columns handled by exact Woodbury updates",
    )
    ap.add_argument(
        "--low-rank-min-nnz", type=int, default=1024,
        help="minimum nonzeros required for low-rank column extraction",
    )
    ap.add_argument(
        "--low-rank-clique-ratio", type=float, default=2.0,
        help="minimum candidate-clique/input-nnz ratio for extraction",
    )
    ap.add_argument("--dynamic-reg-factor", type=float, default=0.1)
    ap.add_argument("--dynamic-update-interval", type=int, default=2)
    ap.add_argument("--core-factor", type=float, default=2.0)
    ap.add_argument("--core-update-interval", type=int, default=6)
    ap.add_argument("--core-cg-tol", type=float, default=1e-8)
    ap.add_argument("--core-cg-maxit", type=int, default=100)
    ap.add_argument("--lagged-max-age", type=int, default=8)
    ap.add_argument("--lagged-refresh-cg", type=int, default=20)
    ap.add_argument("--lagged-cg-tol", type=float, default=1e-8)
    ap.add_argument("--lagged-cg-maxit", type=int, default=80)
    ap.add_argument("--partial-explicit", type=int, default=256)
    ap.add_argument("--partial-warmup", type=int, default=4)
    ap.add_argument("--partial-pool-factor", type=float, default=8.0)
    ap.add_argument("--partial-draft-weight", type=float, default=1.0)
    ap.add_argument(
        "--partial-min-fill-ratio", type=float, default=2.0,
        help="minimum predicted full/hybrid symbolic fill ratio; 0 forces the prototype",
    )
    ap.add_argument(
        "--diverge-tol", type=float, default=1e4,
        help="stop when mu exceeds this multiple of the best mu seen "
             "(inf disables)",
    )
    ap.add_argument(
        "--tau", type=float, default=0.995,
        help="fraction-to-the-boundary factor (MATLAB uses 0.995, IPX 0.9); "
             "lower keeps variables further from their bounds",
    )
    ap.add_argument(
        "--compl", default="gap", choices=("any", "mu", "gap"),
        help="complementarity test: 'gap' = relative duality gap (default), "
             "'any' = mu or gap, 'mu' = the "
             "average pair only, 'gap' = the relative duality gap only. "
             "gap is ~npos_box times mu, so 'any' can stop substantially earlier",
    )
    ap.add_argument(
        "--retrospective-pool", action="store_true",
        help="use option-3 support-reduction column generation with current "
             "and retrospective pool elimination",
    )
    ap.add_argument(
        "--warm-sparse", action="store_true",
        help="use warm sparse-IPM support generation/elimination",
    )
    ap.add_argument("--cg-initial-columns", type=int, default=None)
    ap.add_argument("--cg-pricing-batch", type=int, default=None)
    ap.add_argument("--cg-support-tol", type=float, default=None)
    ap.add_argument("--cg-pricing-tol", type=float, default=None)
    ap.add_argument("--cg-stagnation-tol", type=float, default=None)
    ap.add_argument("--cg-elim-delta", type=float, default=None)
    ap.add_argument(
        "--cg-l1-budget", type=float, default=None,
        help="valid upper bound on the canonical optimal solution's L1 norm",
    )
    ap.add_argument(
        "--cg-objective-upper", type=float, default=None,
        help="valid transformed-objective upper bound; derives an L1 budget "
             "when all canonical costs are nonnegative",
    )
    ap.add_argument("--cg-beta", type=int, default=None)
    ap.add_argument("--cg-max-rounds", type=int, default=30)
    ap.add_argument("--cg-full-pool-after", type=int, default=5)
    ap.add_argument("--cg-artificial-cost", type=float, default=None)
    ap.add_argument("--sparse-initial-factor", type=float, default=2.0)
    ap.add_argument(
        "--sparse-steps", type=int, default=0,
        help="RMP steps per pricing pass; 0 selects one or two from clique burden",
    )
    ap.add_argument("--sparse-step-clique-threshold", type=float, default=120.0)
    ap.add_argument("--sparse-inner-maxit", type=int, default=1)
    ap.add_argument("--sparse-pricing-batch", type=int, default=None)
    ap.add_argument("--sparse-pricing-tol", type=float, default=None)
    ap.add_argument("--sparse-support-tol", type=float, default=None)
    ap.add_argument("--sparse-drop-mu", type=float, default=1e-3)
    ap.add_argument("--sparse-drop-start", type=float, default=None)
    ap.add_argument("--sparse-drop-patience", type=int, default=3)
    ap.add_argument("--sparse-drop-hysteresis", type=float, default=10.0)
    ap.add_argument("--sparse-drop-batch", type=int, default=None)
    ap.add_argument("--sparse-max-rounds", type=int, default=200)
    ap.add_argument("--sparse-fallback-full", action="store_true")
    ap.add_argument("--sparse-artificial-cost", type=float, default=None)
    ap.add_argument(
        "--sparse-fill-weight",
        type=float,
        default=0.2,
        help="penalize clique-producing columns when ordering valid entrants",
    )
    ap.add_argument(
        "--no-sparse-cover-crash",
        dest="sparse_cover_crash",
        action="store_false",
        help="disable the provably feasible covering support crash",
    )
    a = ap.parse_args()
    raw, lp, sol, info = solve_mps(
        a.mps, tol=a.tol, scaling=a.scaling, pc=a.pc, kkt=a.kkt,
        stop=a.stop, maxit=a.maxit, time_limit=a.time_limit,
        rho=a.rho, delta=a.delta,
        printlevel=a.printlevel, refine=a.refine,
        low_rank_max_columns=a.low_rank_max_columns,
        low_rank_min_nnz=a.low_rank_min_nnz,
        low_rank_clique_ratio=a.low_rank_clique_ratio,
        compl=a.compl, tau=a.tau, diverge_tol=a.diverge_tol,
        dynamic_reg_factor=a.dynamic_reg_factor,
        dynamic_update_interval=a.dynamic_update_interval,
        core_factor=a.core_factor,
        core_update_interval=a.core_update_interval,
        core_cg_tol=a.core_cg_tol,
        core_cg_maxit=a.core_cg_maxit,
        lagged_max_age=a.lagged_max_age,
        lagged_refresh_cg=a.lagged_refresh_cg,
        lagged_cg_tol=a.lagged_cg_tol,
        lagged_cg_maxit=a.lagged_cg_maxit,
        partial_explicit=a.partial_explicit,
        partial_warmup=a.partial_warmup,
        partial_pool_factor=a.partial_pool_factor,
        partial_draft_weight=a.partial_draft_weight,
        partial_min_fill_ratio=a.partial_min_fill_ratio,
        retrospective_pool=a.retrospective_pool,
        warm_sparse=a.warm_sparse,
        cg_initial_columns=a.cg_initial_columns,
        cg_pricing_batch=a.cg_pricing_batch,
        cg_support_tol=a.cg_support_tol,
        cg_pricing_tol=a.cg_pricing_tol,
        cg_stagnation_tol=a.cg_stagnation_tol,
        cg_elim_delta=a.cg_elim_delta,
        cg_l1_budget=a.cg_l1_budget,
        cg_objective_upper=a.cg_objective_upper,
        cg_beta=a.cg_beta,
        cg_max_rounds=a.cg_max_rounds,
        cg_full_pool_after=a.cg_full_pool_after,
        cg_artificial_cost=a.cg_artificial_cost,
        sparse_initial_factor=a.sparse_initial_factor,
        sparse_steps=a.sparse_steps,
        sparse_step_clique_threshold=a.sparse_step_clique_threshold,
        sparse_inner_maxit=a.sparse_inner_maxit,
        sparse_pricing_batch=a.sparse_pricing_batch,
        sparse_pricing_tol=a.sparse_pricing_tol,
        sparse_support_tol=a.sparse_support_tol,
        sparse_drop_mu=a.sparse_drop_mu,
        sparse_drop_start=a.sparse_drop_start,
        sparse_drop_patience=a.sparse_drop_patience,
        sparse_drop_hysteresis=a.sparse_drop_hysteresis,
        sparse_drop_batch=a.sparse_drop_batch,
        sparse_max_rounds=a.sparse_max_rounds,
        sparse_fallback_full=a.sparse_fallback_full,
        sparse_artificial_cost=a.sparse_artificial_cost,
        sparse_cover_crash=a.sparse_cover_crash,
        sparse_fill_weight=a.sparse_fill_weight,
        presolve=a.presolve,
    )
    print(
        f"\nobjective {info['obj']:.10e}   pinf {info['pinf']:.2e}   "
        f"dinf {info['dinf']:.2e}   time {sol.time:.2f}s"
    )
