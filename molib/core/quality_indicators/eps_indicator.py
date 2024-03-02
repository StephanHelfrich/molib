# -*- coding: utf-8 -*-
"""
=======================================================
Epsilon Indicator for Multiobjective Approxiamtion Sets 
=======================================================

Implementation of the epsilon indicator as a quality indicator for 
multiobjective approximations sets.

For further information, see 
E. Zitzler, L. Thiele, M. Laumanns, C.M. Fonseca, V.G. da Fonseca, 
Performance assessment of multiobjective optimizers: an analysis and review, 
IEEE Transactions on Evolutionary Computation, 7(2), 117-132, 2003.
"""
from molib.core.image import Image
from typing import List,Set

def get_epsilon_indicator(images: Set[Image], Y_N: Set[Image], type: str) -> float:
    ''' epsilon indicator

    Determine the epsilon indicator of a sets of images to a reference set that contains all nondomianted images.
        
    Parameters
    ----------
    images : Set[Image] of List[Image]
        Images of approximation set
        
    Y_N : Set[Image] or List[Image]
        reference set that that contains all nondomianted images
    
    type : str (either "min" of "max")
        string that specifies if the epsilon indicator should be determined for a minimization problem ("min") or a maximization problem ("max")

    Returns
    -------
    float
        epsilon indicator of the approxiamtion set. I.e., smallest '\alpha' such that the set 'images' constitutes an 'alpha'-approxiamtion set.

    '''  
    solutions = list(images)
    exact_solutions = list(Y_N)

    nr_obj = len(solutions)
    approx_quality = 1
    for exact_sol in exact_solutions:
        # find best solution wrt approximation
        min_approx = float('inf')
        for sol in solutions:
            alpha = 1
            for k in range(nr_obj):
                if type == "min":
                    if sol[k] / exact_sol[k] > alpha:
                        alpha = sol[k] / exact_sol[k]
                if type == "max":
                    if exact_sol[k] / sol[k]  > alpha:
                        alpha = exact_sol[k] / sol[k] 

            if min_approx > alpha:
                min_approx = alpha
        if min_approx > approx_quality:
            approx_quality = min_approx
    return approx_quality