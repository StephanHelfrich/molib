# -*- coding: utf-8 -*-
from molib.core.problem import Problem, Image
from molib.core.solve import implements
from molib.scalarizations import EpsilonConstraint, WeightedSum, WeightedTchebycheff
import numpy as np


def _lexargmin(x: np.ndarray) -> int:
    """Index of the lexicographically smallest row of x."""
    idx = np.arange(x.shape[0])
    for k in range(x.shape[1]):
        col = x[idx, k]
        idx = idx[col == col.min()]
        if len(idx) < 2:
            break
    return int(idx[0])


class _EnumerationSolvers:
    """Exact solvers by enumeration of the rows of self.Y for the weighted Tchebycheff and the
    epsilon-constraint scalarization.

    Ties are broken lexicographically, so the returned image is nondominated also for augmentation = 0.
    """

    def weighted_tchebycheff(self, weights, reference_point, augmentation: float = 0.0) -> Image:
        w = np.asarray(weights, dtype=float)
        r = np.asarray(reference_point, dtype=float)
        Y = np.asarray(self.Y, dtype=float)
        D = Y - r if self.type == "min" else r - Y
        values = np.max(w * D, axis=1) + augmentation * (D @ w)
        oriented = Y if self.type == "min" else -Y
        i = _lexargmin(np.c_[values, oriented])
        return Image(self.Y[i], solution=i)                             # unverändert

    @implements(WeightedTchebycheff)
    def _solve_weighted_tchebycheff(self, s: WeightedTchebycheff) -> Image:
        return self.weighted_tchebycheff(s.weights, s.reference_point, s.augmentation)

    @implements(EpsilonConstraint)
    def _solve_epsilon_constraint(self, s: EpsilonConstraint):
        i = s.best_index(self.Y, self.type)
        return None if i is None else Image(self.Y[i], solution=i)
    
class ImageSet(_EnumerationSolvers, Problem):
    """
        Problem instance is given by an explicit list of points in the image space,
        weighted sum return an (arbitrary) image that approximates the image minimizing or maximizing the weighted sum.
        Solution quality of weighted sum can be specified. Note that solution quality = 1 is also possible.
    """
    def __init__(self, Y: np.ndarray, type = "min", solution_quality_weighted_sum = 2):
        
        self.Y = Y

        super().__init__(type, self.Y.shape[1])

        self.LB = np.min(Y[Y>0])
        self.UB = np.max(Y[Y>0])
 
        self.solution_quality_weighted_sum = solution_quality_weighted_sum

    def weighted_sum(self, weights: list) -> int:
        # print("run with ", weights)
        weights = np.array(weights)
        values = np.matmul(self.Y, weights)

        if self.type == "min":
            opt_value = np.min(values)
            i = np.random.choice(np.where(values <= self.solution_quality_weighted_sum * opt_value)[0])
        elif self.type == "max":
            opt_value = np.max(values)
            i = np.random.choice(np.where(values >= 1 / self.solution_quality_weighted_sum * opt_value)[0])
        
        image = Image(self.Y[i], solution = i)
        return image

        @implements(WeightedSum, guarantee="solution_quality_weighted_sum")
        def _solve_weighted_sum(self, s: WeightedSum) -> Image:
            return self.weighted_sum(s.weights)

class ImageSet_withLexWS(_EnumerationSolvers, Problem):
    """
        Problem instance is given by an explicit list of points in the image space,
        weighted sum return an image minimizing or maximizing a lexicographic version of the weighted sum scalarization. Hence, only extreme supported nondominated images are returned.
    """
    def __init__(self, Y: np.ndarray,type = "min"):
        self.Y = Y
        super().__init__(type,self.Y.shape[1] )

        self.LB = np.min(Y[Y>0])
        self.UB = np.max(Y[Y>0])
        self.solution_quality_weighted_sum = 1


    def weighted_sum(self, weights: list) -> int:
        # print("run with ", weights)
        weights = np.array(weights)
        values = np.matmul(self.Y, weights)
        
        if self.type == "min":
            i = self.lexargmin(np.c_[values, self.Y])
        elif self.type == "max":
            i = self.lexargmax(np.c_[values, self.Y])
        
        image = Image(self.Y[i], solution = i)
        return image
    @implements(WeightedSum)
    def _solve_weighted_sum(self, s: WeightedSum) -> Image:
        return self.weighted_sum(s.weights)

    def ixmin(self,x, k=0, idx=None):
        col = x[idx, k] if idx is not None else x[:, k]
        z = np.where(col == col.min())[0]
        return z if idx is None else idx[z]

    def lexargmin(self,x):
        idx = None
        for k in range(x.shape[1]):
            idx = self.ixmin(x, k, idx)
            if len(idx) < 2:
                break
        return idx[0]  

    def ixmax(self,x, k=0, idx=None):
        col = x[idx, k] if idx is not None else x[:, k]
        z = np.where(col == col.max())[0]
        return z if idx is None else idx[z]

    def lexargmax(self,x):
        idx = None
        for k in range(x.shape[1]):
            idx = self.ixmax(x, k, idx)
            if len(idx) < 2:
                break
        return idx[0]  

    