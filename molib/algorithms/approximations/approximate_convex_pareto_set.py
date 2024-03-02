# -*- coding: utf-8 -*-
"""
=================================================================
General Approximation Methods to obtain convex approximation sets
=================================================================

Implementation of multiobjective convex approximation algorithms that are applicable
under mild assumptions. Given a multiobjective optimization problem, these methods 
seek to find a set of solutions that, for each possible image, contains some convex 
combination of finitely many images of solutions that is component-wise at least as 
good up to a multiplicative factor. In case of multiobjective minimization and maximization problems, 
convex approxiamtion sets are equivalent to sets that contain, for each possible 
weighted sum scalarization, a solution that is approximately optimal.

For further information, see
Helfrich, S. "Approximation and Scalarization in Multiobjective Optimization". PhD Thesis, RPTU Kaiserslautern-Landau.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection
import math
import itertools
from typing import List,Set

from molib.core.image import Image
from molib.core.problem import Problem


## VARIABLES
TOL = 1e-6



def approximate_dichotomy(problem: Problem, epsilon : float) -> Set[Image]:
    """General Approximation Algorithm of Bazgan et al. for biobjective optimization problems

    Obtain a multiobjective convex approximation set by the General Convex Approximation Algorithm of Bazgan et al. [1]_. 
    This algorithm is tailored to biobjective optimization problems!

    Parameters
    ----------
    problem : Problem
        The instance of a biobjective(!) problem for the approximation set should be calculated. It is required that the method 'weighted_sum' is implemented 
        and the attribute 'solution_quality_weighted_sum' is set to the appropriate float value greater than or equal to one that is guaranteed by weighted_sum.

    epsilon : float
        Indicates the additional loss on the guaranteed convex approximtion factor. Required to be greater than 1.


    Returns
    -------
    set
        Set of images (with solutions) that constitutes a multiobjective '(1 + epsilon) * solution_quality_weighted_sum'-convex approximation set.


    Raises
    ------

    
    References
    ----------
    .. [1] Bazgan, C.; Herzel, A.; Ruzika, S.; Thielen, C. & Vanderpooten, D. "An Approximation Algorithm for a General Class of Parametric Optimization Problems."
    Journal of Combinatorial Optimization 43, 2022, pp. 1328-1358.
    """

    assert problem.nr_obj == 2, f"number of objectives of the problem is not equal to 2."


    def mu(image_l, image_r):
        weight = (image_r[0] - image_l[0]) / (image_l[1] - image_r[1])
        return weight

    convexParetoSet = set()

    c = (epsilon * problem.LB) / (problem.solution_quality_weighted_sum * problem.UB)

    image_under_star = problem.weighted_sum([1, c])
    image_above_star = problem.weighted_sum([1, 1/c])

    Q = [(c,1/c,image_under_star,image_above_star)]

    convexParetoSet.add(image_under_star)
    convexParetoSet.add(image_above_star)

    while len(Q) > 0:
        weight_l,weight_r,image_l,image_r = Q.pop(0)

        if np.array([1,weight_r]) @ image_l <= np.array([1,weight_r]) @ image_r:
            convexParetoSet.add(image_l)
        elif np.array([1,weight_l]) @ image_r <= np.array([1,weight_l]) @ image_l:
            convexParetoSet.add(image_r)
        else:
            weight_m = mu(image_l,image_r)
            if np.array([1,weight_m]) @ image_l <= (1 + epsilon) * ( (weight_r - weight_m) / (weight_r - weight_l) * (np.array([1,weight_l]) @ image_l ) + (weight_m - weight_l) / (weight_r - weight_l) * (np.array([1,weight_r]) @ image_r) ):
                convexParetoSet.add(image_l)
                convexParetoSet.add(image_r)
            else:
                image_m = problem.weighted_sum(weight_m)
                Q = Q + [ (weight_l,weight_m,image_l,image_m),(weight_m,weight_r,image_m,image_r)  ]

    return convexParetoSet


def DandY(problem: Problem, epsilon: float) -> Set[Image]:
    """General Approximation Algorithm of Diakonikolas

    Obtain a multiobjective convex approximation set by the General Convex Approximation Algorithm of Diakonikolas [1]_.

    Parameters
    ----------
    problem : Problem
        The instance of the problem for the approximation set should be calculated. It is required that the method 'weighted_sum' is implemented 
        and the attribute 'solution_quality_weighted_sum' is set to the appropriate float value greater than or equal to one that is guaranteed by weighted_sum.

    epsilon : float
        Indicates the additional loss on the guaranteed convex approximtion factor. Required to be greater than 1.


    Returns
    -------
    set
        Set of images (with solutions) that constitutes a multiobjcetive '(1 + epsilon) * solution_quality_weighted_sum'-convex approximation set.


    Raises
    ------

    
    References
    ----------
    .. [1] Diakonikolas, I. "Approximation of Multiobjective Optimization Problems ."
    PhD Thesis, Columbia University, 2011.
    """
    #get/compute import constants
    p = problem.nr_obj
    delta1 = problem.solution_quality_weighted_sum - 1

    
    delta2 = (1 + epsilon)/(1 + delta1) - 1
    if delta2 < 0:
        raise ValueError("weighted sum of problem must guarantee approximation quality less or equal to 1 + epsilon")

    M = int(math.ceil(2 * (p-1)/delta2))

    m = int(math.ceil(max(problem.UB, - problem.LB))) # m best possible such that 2^{-m} \leq LB \leq f(x) \leq UB \leq 2^m


    grid1R = np.arange(0,2*m ,1)
    grid1Wbal = np.arange(1,M+1)/M

    convexParetoSet = set()

    # iteration over r in R
    for r in itertools.product(grid1R, repeat = p - 1):
        for dim in range(p):
            r2 = list(r[:dim]) + [1] + list(r[dim:])
            #iteration over omega in Wbal
            for omega in itertools.product(grid1Wbal,repeat = p-1):
                for dim2 in range(p):
                    omega2 = list(omega[:dim2]) + [1] + list(omega[dim2:])

                    weight = [r2[i] * omega2[i] for i in range(p)]
                    y = problem.weighted_sum(weight)

                    convexParetoSet.add(y)
    
    return convexParetoSet



def GridFPTAA(problem: Problem, epsilon : float) -> Set[Image]:
    """General Approximation Algorithm of Helfrich

    Obtain a multiobjective convex approximation set by the General Convex Approximation Algorithm of Helfrich [1]_
    following a grid approach.

    Parameters
    ----------
    problem : Problem
        The instance of the problem for the approximation set should be calculated. It is required that the method 'weighted_sum' is implemented 
        and the attribute 'solution_quality_weighted_sum' is set to the appropriate float value greater than or equal to one that is guaranteed by weighted_sum.

    epsilon : float
        Indicates the additional loss on the guaranteed convex approximtion factor. Required to be greater than 1.


    Returns
    -------
    set
        Set of images (with solutions) that constitutes a multiobjcetive '(1 + epsilon) * solution_quality_weighted_sum'-convex approximation set.


    Raises
    ------

    
    References
    ----------
    .. [1] Helfrich, S.; Herzel, A.; Ruzika, S. & Thielen, C. "An Approximation Algorithm for a General Class of Multi-Parametric Optimization Problems."
    Journal of Combinatorial Optimization 44, 2022, pp. 1459-1494
    """

    # helper function to check if in interior
    def check_if_in_W(weight:List[float],c,epsilon_dash) -> bool:
        w = np.sort((list(weight)))
        for k in range(1, len(w)-1):
            if sum(w[:k]) < c*w[k] / (1 + epsilon_dash):
                return 0
        return 1

    # get/compute important constants
    p = problem.nr_obj
    epsilon_dash = math.sqrt(1 + epsilon) - 1
    alpha = problem.solution_quality_weighted_sum
    beta = (1 + epsilon_dash) * alpha
    c = (epsilon_dash * problem.LB) / (beta  * problem.UB)

    convexParetoSet = set()

    lb = int(np.floor(    np.log((c**(p - 1))/(math.factorial(p)))   /  np.log(1 + epsilon_dash) ))
    ub = int(np.ceil(    np.log( (math.factorial(p)) / (c**(p - 1)) )  /  np.log(1 +epsilon_dash) ))

    grid1d = np.arange(lb,ub +1 ,1)
    for w in itertools.product(grid1d, repeat = p - 1):
        weights = [1] + [(1 + epsilon_dash)**(i) for i in w ]
    
        # check if the weight is contained in Wapprox, if yes, call the solver
        if check_if_in_W(weights,c,epsilon_dash): 
            
            obj_values = problem.weighted_sum(weights)
           
            convexParetoSet.add(obj_values)

    return convexParetoSet


def FPTOAA(problem: Problem, epsilon: float) -> Set[Image]:
    """General Approximation Algorithm of Helfrich via dual Benson

    Obtain a multiobjective convex approximation set by the General Convex Approximation Algorithm of Helfrich [1]_
    following a grid approach combined with the dual varaint of Benson's Outer Approxiamtion Algorithm.

    Parameters
    ----------
    problem : Problem
        The instance of the problem for the approximation set should be calculated. It is required that the method 'weighted_sum' is implemented 
        and the attribute 'solution_quality_weighted_sum' is set to the appropriate float value greater than or equal to one that is guaranteed by weighted_sum.

    epsilon : float
        Indicates the additional loss on the guaranteed convex approximtion factor. Required to be greater than 1.


    Returns
    -------
    set
        Set of images (with solutions) that constitutes a multiobjcetive '(1 + epsilon) * solution_quality_weighted_sum'-convex approximation set.


    Raises
    ------

    
    References
    ----------
    .. [1] Helfrich, S. "Approximation and Scalarization in Multiobjective Optimization". PhD Thesis, RPTU Kaiserslautern-Landau.
    """

    # helper functions, boundary rounding and grid rounding
    def boundary_rounding(weight: List[float], c) -> tuple():

        sigma = np.argsort(weight)
        w_roof = np.array([weight[sigma[i]] for i in range(0, len(sigma))])

        flag = False
        for k in range(1, len(weight)):
            if sum(w_roof[:k]) < (c*w_roof[k]):
                flag = True
                temp_sum_w_roof = sum(w_roof[:k])
                for i in range(0, k):
                    if temp_sum_w_roof > 0:
                        w_roof[i] = (w_roof[i] / sum(w_roof[:k])) * c * w_roof[k]
                    else:
                        w_roof[i] = (c / (k+1)) * w_roof[k]
        w_dash = np.array([w_roof[list(sigma).index(j)] for j in range(0, len(weight))])
        norm_w_dash = sum([abs(x) for x in w_dash])
        w_dash = w_dash / norm_w_dash

        return list(w_dash),flag
    
    def grid_rounding_get_exp(weight,epsilon_dash,epsilon,n): #n = 1 if applied for (1 + \varepsilon'), n = 2 if applied for (1 + \varepsilon)
        z = []
        for i in range(0, len(weight)):
            if n == 1:
                # z.append( n* math.ceil(    math.log(weight[i], (1+self.epsilon_dash))  ) )
                z.append(  math.ceil(    math.log(weight[i], (1+epsilon_dash))  ) )
            else:
                z.append( n * math.ceil(   math.log(weight[i], (1+epsilon))  ) )
        return tuple([i - z[0] for i in z])

    # get/compute important constants
    p = problem.nr_obj
    epsilon_dash = 0.5*epsilon
    alpha = problem.solution_quality_weighted_sum
    beta = (1 + epsilon_dash) * alpha
    c = (epsilon_dash * problem.LB) / (beta  * problem.UB)

    y = problem.weighted_sum([1 for _ in range(p)])    
    
    convexParetoSet = set()
    convexParetoSet.add(y)


    ''' initialize outer approximation with initial solution'''
    A = []

    if problem.type == "min":
        A.append([-i + y[-1] for i in y[:-1]] + [1] + [-y[-1]])
    if problem.type == "max":
        A.append([i - y[-1] for i in y[:-1]] + [-1] + [y[-1]])

    # add that the w's can only add up to 1
    A.append([1 for _ in range(0, p-1)] + [0, -1])

    # add w__i >= 0
    for i in range(0, p-1):
        A.append([0 for _ in range(i)] + [-1] + [0 for _ in range(i+1,p - 1)] + [0,0])
        
    if problem.type == "min":
        # add that y >= 0
        A.append([0 for i in range(0, p-1)] + [-1, 0])
    elif problem.type == "max":
        # add that y <= UB
        A.append([0 for _ in range(p - 1)] + [1, -problem.UB])

    # make A np array
    A = np.array(A)

    if problem.type == "min":
        # determine interior point: take y = f(x,w)/2 and w = (1/K,...(1/K))
        interior_point = np.full((p-1), 1/(p))
        interior_point = np.append(interior_point, np.matmul(np.append(interior_point, 0), y)/2)
        interior_point = np.array(interior_point)
    elif problem.type == "max":
        # determine interior point: take y = f(x,y) + (UB - f(x,w))/2 and w = (1/K,...(1/K))
        interior_point = np.full((p-1), 1/(p))
        interior_point = np.append(interior_point, np.matmul(np.append(interior_point, 0), y)/2 + problem.UB/2)
        interior_point = np.array(interior_point)

    outer_approximation = HalfspaceIntersection(A, interior_point, incremental=True)
    
    # get extreme points of outer_approximation as list
    vertices = outer_approximation.intersections.tolist()

    # initialize list of investigated weights as kdTree with dimension number_criteria - 1
    L = set()
    L.add(tuple([ 0 for _ in range(0, p -1)]))

    while len(vertices) > 0: 
        next_vertex = vertices.pop()

        # check if vertex is subset of y == 0 or y == UB -> if yes, skip
        if next_vertex[-1] < TOL and problem.type == "min":
            continue
        if next_vertex[-1] > problem.UB - TOL and problem.type == "max":
            continue

        # correct vertice
        for i in range(len(next_vertex)-1):
            if next_vertex[i] < 0:
                next_vertex[i] = 0.0
            elif next_vertex[i] > 1:
                next_vertex[i] = 1.0 

        # boundary rounding
        w_line, flag = boundary_rounding(next_vertex[:-1] + [1 - sum(i for i in next_vertex[:-1])] ,c)

        # grid rounding
        if flag == True:
            z = grid_rounding_get_exp(w_line, epsilon_dash, epsilon, 1)
        else:
            z = grid_rounding_get_exp(w_line, epsilon_dash, epsilon, 2)       

        # skip if z in L
        if z in L:
            continue

        w_line = [(1+epsilon_dash)**(z[i]) for i in range(p)]

        #solve routine
        y = problem.weighted_sum( w_line )
        convexParetoSet.add(y)
        L.add(z)

        # add hyperplane for obtained solution
        if problem.type == "min":
            # f(x,w) >= y for minimization
            outer_approximation.add_halfspaces(np.array([ [-i + y[-1] for i in y[:-1]] + [1] + [-y[-1]] ] ))
        elif problem.type == "max":
            # f(x,w) <= y for maximization
           outer_approximation.add_halfspaces(np.array([  [i - y[-1] for i in y[:-1]] + [-1] + [y[-1]] ] ))

        vertices = outer_approximation.intersections.tolist()

    return convexParetoSet



