# -*- coding: utf-8 -*-
"""Base class of all scalarizations."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = ["Scalarization"]


def _check_sense(sense: str) -> None:
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")


class Scalarization(ABC):
    """Single-objective substitute problem of a multiobjective optimization problem.

    The meaning of a scalarization depends on the optimization sense ("min" or "max") of the
    problem it is applied to. For example, ``WeightedSum(w)`` minimizes w^T f(x) for a
    minimization problem and maximizes w^T f(x) for a maximization problem. Hence, every
    method takes the sense of the problem as argument. Problems are never negated, so that
    multiplicative guarantees keep their meaning.

    Subclasses implement ``direction``, ``value`` and, if the scalarization has additional
    constraints, ``feasible``.
    """

    @abstractmethod
    def direction(self, sense: str) -> str:
        """Optimization direction ("min" or "max") of the scalarization for a problem with ``sense``."""

    @abstractmethod
    def value(self, Y, sense: str) -> np.ndarray:
        """Objective values of the scalarization for the images in the rows of Y."""

    def feasible(self, Y, sense: str) -> np.ndarray:
        """Feasibility of the images in the rows of Y with respect to additional constraints."""
        return np.ones(np.atleast_2d(Y).shape[0], dtype=bool)

    def validate(self, nr_obj: int) -> None:
        """Raise a ValueError if the parameters do not fit a problem with ``nr_obj`` objectives."""

    # ------------------------------------------------------------------ helpers
    def is_better(self, a: float, b: float, sense: str, tol: float = 0.0) -> bool:
        """True iff value a is better than value b by more than tol."""
        return a < b - tol if self.direction(sense) == "min" else a > b + tol

    def best_index(self, Y, sense: str) -> Optional[int]:
        """Index of an optimal row of Y (enumeration), or None if no row is feasible.

        Ties are broken lexicographically with respect to the objectives (in the sense of the
        problem), so the selected image is nondominated among the optimal images.
        """
        _check_sense(sense)
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        candidates = np.flatnonzero(self.feasible(Y, sense))
        if len(candidates) == 0:
            return None
        values = self.value(Y[candidates], sense)
        oriented_values = values if self.direction(sense) == "min" else -values
        oriented_images = Y[candidates] if sense == "min" else -Y[candidates]
        keys = np.c_[oriented_values, oriented_images]
        idx = np.arange(len(candidates))
        for k in range(keys.shape[1]):
            col = keys[idx, k]
            idx = idx[col == col.min()]
            if len(idx) < 2:
                break
        return int(candidates[idx[0]])


def _as_vector(values, name: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return v


def _check_length(v: np.ndarray, nr_obj: int, name: str) -> None:
    if len(v) != nr_obj:
        raise ValueError(f"{name} has length {len(v)}, but the problem has {nr_obj} objectives")