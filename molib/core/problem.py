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
