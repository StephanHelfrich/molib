# -*- coding: utf-8 -*-
"""Weighted sum scalarization."""
import numpy as np

from .base import Scalarization, _as_vector, _check_length, _check_sense

__all__ = ["WeightedSum"]


class WeightedSum(Scalarization):
    r"""Weighted sum scalarization  opt_x  \sum_i w_i f_i(x)  with opt = min or max as the problem.

    Parameters
    ----------
    weights : sequence of float
        Nonnegative weights, one per objective.
    """

    def __init__(self, weights):
        self.weights = _as_vector(weights, "weights")
        if np.any(self.weights < 0):
            raise ValueError("weights must be nonnegative")

    def direction(self, sense: str) -> str:
        _check_sense(sense)
        return sense

    def value(self, Y, sense: str) -> np.ndarray:
        _check_sense(sense)
        return np.atleast_2d(np.asarray(Y, dtype=float)) @ self.weights

    def validate(self, nr_obj: int) -> None:
        _check_length(self.weights, nr_obj, "weights")

    def __repr__(self):
        return f"WeightedSum(weights={self.weights.tolist()})"