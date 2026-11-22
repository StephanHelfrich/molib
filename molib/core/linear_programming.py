# -*- coding: utf-8 -*-
"""
=========================================
Linear programs used by several algorithms
=========================================

Small linear programs on finite sets of images, solved with ``scipy.optimize.linprog`` (HiGHS).
They replace the former Gurobi models, so that no commercial solver is required.
"""
from typing import Optional

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "max_last_coordinate_in_hull",
    "weighted_sum_ratio_on_cell",
    "convex_approximation_factor",
]


def _solve(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None)):
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")


def max_last_coordinate_in_hull(points, prefix) -> Optional[float]:
    """Maximum of the last coordinate over all z in conv(points) with z[:-1] = prefix.

    Parameters
    ----------
    points : array of shape (n, d)
    prefix : array of shape (d - 1,)

    Returns
    -------
    float or None
        The maximum, or None if no point of conv(points) has the first d - 1 coordinates ``prefix``.
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    prefix = np.asarray(prefix, dtype=float)
    n = P.shape[0]
    A_eq = np.vstack([P[:, :-1].T, np.ones((1, n))])
    b_eq = np.append(prefix, 1.0)
    res = _solve(-P[:, -1], A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
    if res.status == 2:          # infeasible
        return None
    if res.status != 0:
        raise RuntimeError(f"linear program could not be solved: {res.message}")
    return float(-res.fun)


def weighted_sum_ratio_on_cell(images, reference, y_index: int, sense: str) -> Optional[float]:
    r"""Largest weighted sum ratio between ``images`` and ``reference`` over the weights for which
    ``reference[y_index]`` is optimal among ``reference``.

    For minimization, the value is

        sup { min_{q in images} w^T q / w^T y :  w >= 0, w^T y <= w^T y' for all y' in reference },

    and for maximization

        sup { w^T y / max_{q in images} w^T q :  w >= 0, w^T y >= w^T y' for all y' in reference },

    where y = reference[y_index]. Both are linear-fractional programs that are homogeneous in w, so
    the normalization w^T y = 1 turns them into linear programs. Requires positive images.

    Returns
    -------
    float or None
        The supremum (np.inf if unbounded), or None if no weight vector w >= 0 with w^T y > 0
        makes y optimal (y is not supported).
    """
    Q = np.atleast_2d(np.asarray(images, dtype=float))
    Y = np.atleast_2d(np.asarray(reference, dtype=float))
    y = Y[y_index]
    p = Y.shape[1]
    others = np.delete(Y, y_index, axis=0)

    if sense == "min":
        # variables (w, t): max t  s.t.  t - w^T q <= 0,  w^T (y - y') <= 0,  w^T y = 1,  w >= 0
        c = np.append(np.zeros(p), -1.0)
        A_ub = np.vstack([
            np.hstack([-Q, np.ones((len(Q), 1))]),
            np.hstack([y - others, np.zeros((len(others), 1))]),
        ])
        b_ub = np.zeros(A_ub.shape[0])
        A_eq = np.append(y, 0.0)[None, :]
        res = _solve(c, A_ub, b_ub, A_eq, [1.0], bounds=[(0, None)] * p + [(None, None)])
        if res.status == 2:
            return None
        if res.status == 3:
            return np.inf
        if res.status != 0:
            raise RuntimeError(f"linear program could not be solved: {res.message}")
        return float(-res.fun)

    if sense == "max":
        # variables (w, s): min s  s.t.  w^T q - s <= 0,  w^T (y' - y) <= 0,  w^T y = 1,  w >= 0;
        # the ratio is 1 / s
        c = np.append(np.zeros(p), 1.0)
        A_ub = np.vstack([
            np.hstack([Q, -np.ones((len(Q), 1))]),
            np.hstack([others - y, np.zeros((len(others), 1))]),
        ])
        b_ub = np.zeros(A_ub.shape[0])
        A_eq = np.append(y, 0.0)[None, :]
        res = _solve(c, A_ub, b_ub, A_eq, [1.0], bounds=[(0, None)] * p + [(0, None)])
        if res.status == 2:
            return None
        if res.status != 0:
            raise RuntimeError(f"linear program could not be solved: {res.message}")
        return np.inf if res.fun <= 0 else float(1.0 / res.fun)

    raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")


def convex_approximation_factor(images, y, sense: str) -> float:
    r"""Smallest factor t such that a convex combination z of ``images`` satisfies z <= t * y
    (minimization) or t * z >= y (maximization).

    For minimization, this is the LP  min t  s.t.  Q^T theta <= t y,  sum(theta) = 1,  theta >= 0.
    For maximization, the substitution phi = t * theta yields the LP
    min sum(phi)  s.t.  Q^T phi >= y,  phi >= 0. Requires positive images.
    """
    Q = np.atleast_2d(np.asarray(images, dtype=float))
    y = np.asarray(y, dtype=float)
    n, p = Q.shape

    if sense == "min":
        # variables (theta, t)
        c = np.append(np.zeros(n), 1.0)
        A_ub = np.hstack([Q.T, -y[:, None]])
        A_eq = np.append(np.ones(n), 0.0)[None, :]
        res = _solve(c, A_ub, np.zeros(p), A_eq, [1.0], bounds=[(0, None)] * (n + 1))
    elif sense == "max":
        # variables phi
        res = _solve(np.ones(n), -Q.T, -y, bounds=[(0, None)] * n)
    else:
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")

    if res.status == 2:
        return np.inf
    if res.status != 0:
        raise RuntimeError(f"linear program could not be solved: {res.message}")
    return float(res.fun)