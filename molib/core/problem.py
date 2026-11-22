# -*- coding: utf-8 -*-
import numpy as np
from .image import *

class Problem:
    def __init__(self, type, nr_obj):
        '''
        Parameters
        ----------
        type:   str
            "min" or "max: it is assumed that all objectives are of the same type
        nr_obj:  int
            number of objectives
        '''
        self.type = type
        self.nr_obj = nr_obj

        self._ideal_point, self._nadir_point = None, None # np.array(), np.array()

        self.solution_quality_weighted_sum = float('inf') # guaranteed quality of solution for the weighted sum scalarization
        
    def print(self):
        print() 

    def weighted_sum(self, weight: list) -> Image:
        '''
        Parameters
        ----------
        weight: np.array
            numpy array  containing the weight values for an application of a weighted sum scalarization, number of weights should match the number of objectives
        ouput:  

        Returns
        -------
        Image
            Image with solution such that the weighted sum objective is not worse that self.solution_qualitiy_weighted_sum times the optimal objective.
        '''
        pass

    def weighted_tchebycheff(self, weights, reference_point, augmentation: float = 0.0) -> Image:
        r'''(Augmented) weighted Tchebycheff scalarization.

        For a minimization problem, return an image of an optimal solution of

            min_x  max_i w_i (f_i(x) - r_i)  +  augmentation * sum_i w_i (f_i(x) - r_i),

        where r is the reference point. For a maximization problem, f_i(x) - r_i is replaced by
        r_i - f_i(x). With augmentation > 0, every optimal solution is properly efficient.
        With augmentation = 0, the returned image is expected to be nondominated (e.g., by
        lexicographic tie-breaking).

        Returns
        -------
        Image or None
            Image (with solution) of an optimal solution, or None if no solution was found.
        '''
        raise NotImplementedError("weighted_tchebycheff is not implemented for this problem")

    def get_bounds(self):
        """Return (LB, UB) as vectors of length nr_obj, or raise if they are not set."""
        LB = getattr(self, "LB", None)
        UB = getattr(self, "UB", None)
        if LB is None or UB is None:
            raise AttributeError("The problem does not provide the attributes LB and UB")
        LB = np.broadcast_to(np.asarray(LB, dtype=float), (self.nr_obj,)).copy()
        UB = np.broadcast_to(np.asarray(UB, dtype=float), (self.nr_obj,)).copy()
        return LB, UB
