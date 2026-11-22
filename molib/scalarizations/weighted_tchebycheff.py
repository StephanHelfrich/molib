# -*- coding: utf-8 -*-
"""(Augmented) weighted Tchebycheff scalarization."""
import numpy as np

from .base import Scalarization, _as_vector, _check_length, _check_sense

__all__ = ["WeightedTchebycheff"]


class WeightedTchebycheff(Scalarization):
    r"""(Augmented) weighted Tchebycheff scalarization (always minimized).

    For a minimization problem with reference point r below all images,

        min_x  max_i w_i (f_i(x) - r_i)  +  augmentation * \sum_i w_i (f_i(x) - r_i).

    For a maximization problem with reference point r above all images, f_i(x) - r_i is replaced
    by r_i - f_i(x). With augmentation > 0, every optimal solution is properly efficient.

    Parameters
    ----------
    weights : sequence of float
        Nonnegative weights, one per objective.
    reference_point : sequence of float
        Reference point r.
    augmentation : float
        Augmentation parameter rho >= 0.
    """

    def __init__(self, weights, reference_point, augmentation: float = 0.0):
        self.weights = _as_vector(weights, "weights")
        self.reference_point = _as_vector(reference_point, "reference_point")
        self.augmentation = float(augmentation)
        if np.any(self.weights < 0):
            raise ValueError("weights must be nonnegative")
        if self.augmentation < 0:
            raise ValueError("augmentation must be nonnegative")
        if len(self.weights) != len(self.reference_point):
            raise ValueError("weights and reference_point must have the same length")

    def direction(self, sense: str) -> str:
        _check_sense(sense)
        return "min"

    def deviations(self, Y, sense: str) -> np.ndarray:
        """f(x) - r for minimization and r - f(x) for maximization (row-wise)."""
        _check_sense(sense)
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        return Y - self.reference_point if sense == "min" else self.reference_point - Y

    def value(self, Y, sense: str) -> np.ndarray:
        D = self.deviations(Y, sense)
        return np.max(self.weights * D, axis=1) + self.augmentation * (D @ self.weights)

    def validate(self, nr_obj: int) -> None:
        _check_length(self.weights, nr_obj, "weights")

    def __repr__(self):
        return (f"WeightedTchebycheff(weights={self.weights.tolist()}, "
                f"reference_point={self.reference_point.tolist()}, augmentation={self.augmentation:g})")