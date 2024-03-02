# -*- coding: utf-8 -*-
"""
=========================================================================
General Approximation Methods to obtain multiobjective approximation sets
=========================================================================

Implementation of multiobjective approximation algorithms that are applicable
under mild assumptions. Given a multiobjective optimization problem, these methods 
seek to find a set of solutions that, for each possible image, contains a solution
whose image is component-wise at least as good up to a multiplicative factor.

For further information, see 
Hezel, Thielen, Ruzika. "Approximation Methods for Multiobjective Optimization Problems: A Survey"
"""
from typing import Set
import itertools
import math

from molib.core.image import Image
from molib.core.problem import Problem

def Papadimitriou_and_Yannakakis(problem: Problem, epsilon: float) -> Set[Image]:
    """General Approximation Algorithm of Papadimitriou and Yannakakis

    Obtain a multiobjective approximation set by the General Approximation Algorithm of Papadimitriou and Yannakakis presented in [1]_.

    Parameters
    ----------
    problem : Problem
        The instance of the problem for the approxiamtion set should be calculated. It is required that the method 'gap' is implemented 
        and the attribute solution_quality_gap is set to the appropriate float value greater than or equal to one that is guaranteed by gap.

    epsilon : float
        Indicates the additional loss on the guaranteed approximtion factor. Required to be greater than 1.


    Returns
    -------
    Set[Image]
        Set of images (with solutions) that constitutes a multiobjcective '(1 + epsilon) * solution_quality_gap'-approximation set.

    Notes
    -----
        This function assumes that the method 'gap' of the problem objective is implemented and the attribute
        solution_quality_gap is set to a foat value greater or equal to one.


    Raises
    ------



    References
    ----------
    .. [1] Papadimitriou, C. & Yannakakis, M. "On the Approximability of Trade-offs and Optimal Access of Web Sources."
    Proceedings of the 41st Annual IEEE Symposium on the Foundations of Computer Science (FOCS), IEEE, 2000, 86-92
    """

    K = int(math.ceil(math.log( problem.UB / problem.LB, base = 1 + epsilon)) ) 

    
    approximation_set = set()

    for bound_exp in itertools.product(range(K), repeat = problem.nr_obj):
        bounds = [ (1 + epsilon)**bound_exp[i] * problem.LB for i in range(problem.nr_obj)]
        image = problem.gap(bounds)
        approximation_set.add(image)

    return approximation_set
