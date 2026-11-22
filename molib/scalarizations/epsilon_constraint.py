# -*- coding: utf-8 -*-
"""Epsilon-constraint scalarization."""
import numpy as np

from .base import Scalarization, _as_vector, _check_length, _check_sense

__all__ = ["EpsilonConstraint"]


class EpsilonConstraint(Scalarization):
    r"""Epsilon-constraint scalarization.

    For a minimization problem,  min f_k(x)  s.t.  f_j(x) <= b_j  for all j != k.
    For a maximization problem,  max f_k(x)  s.t.  f_j(x) >= b_j  for all j != k.

    Parameters
    ----------
    objective : int
        Index k of the objective that is optimized.
    bounds : sequence of float
        Bounds b_j, one per objective. The entry of objective k is ignored. Use np.inf
        (minimization) or -np.inf (maximization) for objectives without constraint.
    """

    def __init__(self, objective: int, bounds):
        self.objective = int(objective)
        self.bounds = _as_vector(bounds, "bounds")
        if not 0 <= self.objective < len(self.bounds):
            raise ValueError(f"objective must be an index in 0, ..., {len(self.bounds) - 1}")

    def direction(self, sense: str) -> str:
        _check_sense(sense)
        return sense

    def value(self, Y, sense: str) -> np.ndarray:
        _check_sense(sense)
        return np.atleast_2d(np.asarray(Y, dtype=float))[:, self.objective]

    def feasible(self, Y, sense: str) -> np.ndarray:
        _check_sense(sense)
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        others = np.arange(Y.shape[1]) != self.objective
        if sense == "min":
            return np.all(Y[:, others] <= self.bounds[others], axis=1)
        return np.all(Y[:, others] >= self.bounds[others], axis=1)

    def validate(self, nr_obj: int) -> None:
        _check_length(self.bounds, nr_obj, "bounds")

    def __repr__(self):
        return f"EpsilonConstraint(objective={self.objective}, bounds={self.bounds.tolist()})"