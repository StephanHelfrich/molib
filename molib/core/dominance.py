# -*- coding: utf-8 -*-
"""
==================
Dominance relations
==================

Componentwise comparisons and Pareto dominance with a numerical tolerance, and
filters that extract the nondominated elements of a finite set of images.

Unless stated otherwise, all functions assume that every objective is to be
minimized (``sense="min"``). For maximization problems, pass ``sense="max"``.
"""
from typing import Iterable, List, Sequence, TypeVar

import numpy as np

TOL = 1e-6

T = TypeVar("T")


def _oriented(a, sense: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if sense == "min":
        return a
    if sense == "max":
        return -a
    raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")


# ---------------------------------------------------------------------------
# componentwise orders (minimization), used by the local upper bound machinery
# ---------------------------------------------------------------------------
def strict_leq(a, b, tol: float = TOL) -> bool:
    """a < b componentwise (strictly, with tolerance)."""
    return bool(np.all(np.asarray(a, dtype=float) < np.asarray(b, dtype=float) - tol))


def leqq(a, b, tol: float = TOL) -> bool:
    """a <= b componentwise (with tolerance)."""
    return bool(np.all(np.asarray(a, dtype=float) <= np.asarray(b, dtype=float) + tol))


def leq(a, b, tol: float = TOL) -> bool:
    """a <= b componentwise and a != b, i.e., a dominates b in the Pareto sense."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return leqq(a, b, tol) and bool(np.any(a < b - tol))


# ---------------------------------------------------------------------------
# dominance with sense
# ---------------------------------------------------------------------------
def weakly_dominates(a, b, sense: str = "min", tol: float = TOL) -> bool:
    """True iff a is componentwise at least as good as b."""
    return leqq(_oriented(a, sense), _oriented(b, sense), tol)


def dominates(a, b, sense: str = "min", tol: float = TOL) -> bool:
    """True iff a weakly dominates b and is strictly better in at least one component."""
    return leq(_oriented(a, sense), _oriented(b, sense), tol)


def strictly_dominates(a, b, sense: str = "min", tol: float = TOL) -> bool:
    """True iff a is strictly better than b in every component."""
    return strict_leq(_oriented(a, sense), _oriented(b, sense), tol)


def nondominated_mask(Y, sense: str = "min", tol: float = TOL) -> np.ndarray:
    """Boolean mask of the nondominated rows of an (n x p) array.

    Duplicate rows are all kept, since none of them dominates another.
    """
    Y = _oriented(Y, sense)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array of shape (n, p)")
    keep = np.ones(len(Y), dtype=bool)
    for i in range(len(Y)):
        dominated_by = np.all(Y <= Y[i] + tol, axis=1) & np.any(Y < Y[i] - tol, axis=1)
        keep[i] = not np.any(dominated_by)
    return keep


def nondominated(images: Iterable[T], sense: str = "min", tol: float = TOL) -> List[T]:
    """Return the nondominated elements of ``images`` (the objects themselves, e.g. ``Image``)."""
    images = list(images)
    if not images:
        return []
    mask = nondominated_mask(np.array([np.asarray(y, dtype=float) for y in images]), sense, tol)
    return [y for y, keep in zip(images, mask) if keep]


def segment_weakly_dominates_point(p: Sequence[float], q: Sequence[float], b: Sequence[float],
                                   tolerances=0.0) -> bool:
    """True iff some point of the segment [p, q] weakly dominates b (minimization).

    ``tolerances`` is a scalar or a vector of absolute tolerances per objective, i.e.,
    the test is whether some mu in [0, 1] satisfies mu * p + (1 - mu) * q <= b + tolerances.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    b = np.asarray(b, dtype=float)
    tol = np.broadcast_to(np.asarray(tolerances, dtype=float), b.shape)

    lo, hi = 0.0, 1.0
    for d, rhs in zip(p - q, (b + tol) - q):
        if abs(d) <= 1e-12:
            if rhs < -1e-12:
                return False
        elif d > 0:
            hi = min(hi, rhs / d)
        else:
            lo = max(lo, rhs / d)
    return lo <= hi + 1e-12