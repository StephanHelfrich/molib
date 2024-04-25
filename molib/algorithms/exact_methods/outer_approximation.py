# -*- coding: utf-8 -*-
"""
=========================================================================
General Exact Methods to obtain all optimal image sets for the weighted sum scalarization.
=========================================================================

Implementation of Outer Approximation Algorithms to compute optimal image sets for the weighted sum scalarization. More precisely,
Given a multiobjective optimization problem, these methods seek to find a set of images that contain, for each possible weight vecotrs 
$w \in \R^d_\geq$, an image $y^*$ such that$ w^\top y^* = min_{y \in Y} w^\top y.

For further information, see
H. P. Benson. "An outer approximation algorithm for generating all efficient extreme points in the outcome set of a multiple objective linear programming
problem." Journal of Global Optimization, 13(1):1-24, 1998

and 

F. Bokler and P. Mutzel. "Output-sensitive algorithms for enumerating the extreme nondominated points of multiobjective combinatorial optimization problems." 
In: Algorithms - ESA 2015, pp 288-299, 2015.

and 

M. Ehrgott, A. L ohne, and L. Shao. "A dual variant of Benson's outer approx-
imation algorithm for multiple objective linear programming."
Journal of Global Optimization, 52(4):757-778, 2012.
"""
import sys
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from typing import List,Set,Tuple

from molib.core.image import Image
from molib.core.problem import Problem



from molib.core.kdTree import make_kd_tree,get_nearest,add_point


TOL = 1e-6

def dual_outer_approximation(problem:Problem) -> Set[Image]:
    """dual varaint of Benson's Outer Approximation Algorithm

    Obtain an optimal solution set by the dual variant of Benson's Outer Approxiamtion Algorithm presented in Bökler et al. [1]_. 
    
    It is assumed that the image set of the multiobjective optimziation problem is contained in the positive orthant. If weighted sum algorithm returns solution that is optimal for lexicographic variant lexmin \lambda^T F(x), F_1(x), \dots, F_k(X), the algorithm
    returns extreme supported nondominated images, only.

    Parameters
    ----------
    problem : Problem
        The instance of a mo problem for which the optimal solution set for the weighted sum scalarization should be calculated. It is required that the method 'weighted_sum' is implemented 
        and the attribute 'solution_quality_weighted_sum' is set to 1.


    Returns
    -------
    set
        Set of images (with solutions) that constitutes an optimal image set for the weighted sum scalarization. In Particular, this set contains all extreme supported nondominated images.


    Raises
    ------

    
    References
    ----------
    .. [1] F. Bokler and P. Mutzel. "Output-sensitive algorithms for enumerating the extreme nondominated points of multiobjective combinatorial optimization problems." In: Algorithms - ESA 2015, pp 288-299, 2015.
    """

    def tcheb(a, b):
            value = max([abs(a[i] - b[i]) for i in range(len(a))])
            return value

    def already_checked(next_vertex,L):
        # get nearest weights in L with respect to weighted Tchebycheff distance. If tcheb_dist <= TOL, skip vertex
        distance, _ = get_nearest(L, next_vertex[:-1], len(next_vertex)-1, lambda a,b: tcheb(a,b))

        if distance <= problem.nr_obj * TOL:
            return True
        else:
            return False

    # compute important constants

    p = problem.nr_obj

    supported_images = set()

    y = problem.weighted_sum([1 for _ in range(p)])
    supported_images.add(y)

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
        interior_point = np.append(interior_point, np.matmul(np.append(interior_point, 1/(p)), y) - 100)
        interior_point = np.array(interior_point)
    elif problem.type == "max":
        # determine interior point: take y = f(x,w) + (UB - f(x,w))/2 and w = (1/K,...(1/K))
        interior_point = np.full((p-1), 1/(p))
        interior_point = np.append(interior_point, np.matmul(np.append(interior_point, 1/(p)), y) + 1)
        interior_point = np.array(interior_point)

    oa = HalfspaceIntersection(A, interior_point, incremental=True)

    ''' main loop ''' 
    vertices = oa.intersections.tolist()

    # initialize list of investigated weights as kdTree with dimension number_criteria - 1
    L = make_kd_tree([[1/p for i in range(0, p -1)]],p-1)

    while len(vertices) > 0:
        next_vertex = vertices.pop() 
        # print(len(supported_images), next_vertex)
        
        # if next_vertex[-1] == float('inf') or next_vertex[-1] == -float('inf'):
        #     # print('skip vertex because y <= -inf')
        #     continue
        # if next_vertex[-1] >=  float('inf') and problem.type == "max":
        #     #'skip vertex because y >= inf'
        #     continue
        if next_vertex[-1] <= TOL and problem.type == "min":
            #skip vertex because it is vertex on lower boundary
            continue
        if next_vertex[-1] >= problem.UB - TOL and problem.type == "max":
            # skip vertex becaues it is vertex aon upper boundary
            # print('skip vertex because its upper boundary vertex')
            continue

        # correct vertice
        for i in range(len(next_vertex)-1):
            if next_vertex[i] < 0 +  TOL:
                next_vertex[i] = 0
            elif next_vertex[i] > 1 - TOL:
                next_vertex[i] = 1

        if already_checked(next_vertex,L):
            # print('skip vertex because its already checked')
            continue

        # solve weighted sum
        # print('solve weighted sum')
        obj_values = problem.weighted_sum( next_vertex[:-1] + [1 - sum(i for i in next_vertex[:-1])]     )
        supported_images.add(obj_values)
        add_point(L,next_vertex[:-1],p-1)
        # L.append(next_vertex[:-1])

        
        if abs( sum( obj_values[i]*next_vertex[i] for i in range(len(obj_values)-1)) + (1 - sum(j for j in next_vertex[:-1])) * obj_values[-1] - next_vertex[-1]) < TOL:
            # print('weighted sum has same objective than current vertice, skip')
            continue
        
        # add halfspace to outer approximation
        if problem.type == "min":
            halfspace = [  obj_values[-1] - i for i in obj_values[:-1] ] + [1,-obj_values[-1]]
        elif problem.type == "max":
            halfspace = [  -obj_values[-1] + i for i in obj_values[:-1] ] + [-1,obj_values[-1]]
        else:
            print('somethings wrong')
        oa.add_halfspaces([halfspace])
        vertices = oa.intersections.tolist() # better routine needed to avoid redundant tests!!!

    oa.close()
    return supported_images


def dual_outer_approximation_double_description(problem:Problem) -> Set[Image]:
    """double description cariant of dual varaint of Benson's Outer Approximation Algorithm

    Obtain an optimal solution set by the dual variant of Benson's Outer Approxiamtion Algorithm presented in Bökler et al. [1]_. 
    
    It is assumed that the image set of the multiobjective optimziation problem is contained in the positive orthant. If weighted sum algorithm returns solution that is optimal for lexicographic variant lexmin \lambda^T F(x), F_1(x), \dots, F_k(X), the algorithm
    returns extreme supported nondominated images, only.

    Parameters
    ----------
    problem : Problem
        The instance of a mo problem for which the optimal solution set for the weighted sum scalarization should be calculated. It is required that the method 'weighted_sum' is implemented 
        and the attribute 'solution_quality_weighted_sum' is set to 1.


    Returns
    -------
    set
        Set of images (with solutions) that constitutes an optimal image set for the weighted sum scalarization. In Particular, this set contains all extreme supported nondominated images.


    Raises
    ------

    
    References
    ----------
    .. [1] F. Bokler and P. Mutzel. "Output-sensitive algorithms for enumerating the extreme nondominated points of multiobjective combinatorial optimization problems." In: Algorithms - ESA 2015, pp 288-299, 2015.
    """
    try: 
        import gurobipy as gp
    except:
        print('gurobi not found')
        return
    
    def tcheb(a, b):
            value = max([abs(a[i] - b[i]) for i in range(len(a))])
            return value

    def already_checked(next_vertex,L):
        # get nearest weights in L with respect to weighted Tchebycheff distance. If tcheb_dist <= TOL, skip vertex
        distance, _ = get_nearest(L, next_vertex[:-1], len(next_vertex)-1, lambda a,b: tcheb(a,b))

        if distance <= problem.nr_obj * TOL:
            return True
        else:
            return False
        
    def distanzConvexHull(points, point):
        # points: np.array of dimension (nr_point, nr_criteria)
        # point: np.array of dimension [nr_criteria,]
        # output distance point, conv(points)

        # print('Calculate distance to inner approximation polytope..........')
        # Create Model
        model = gp.Model('convex hull')
        model.Params.LogToConsole = 0
        # create variable per point in points
        x = model.addVars(points.shape[0], lb = 0, ub = 1, name = 'x')
        # create obj variable
        y = model.addVar(lb = 0, name = 'y')

        # add summ constraint
        model.addConstr( gp.quicksum( x[i] for i in range(len(x))  ) == 1 )

        # add pont constraints
        for j in range(points.shape[1] - 1):
            model.addConstr( gp.quicksum( points[i,j] * x[i] for i in range(len(x)) )  == point[j] )
        model.addConstr( gp.quicksum(points[i,-1] * x[i] for i in range(len(x)) ) == y )

        # set objective
        model.setObjective(y, gp.GRB.MAXIMIZE  )
        model.optimize()

        # print(f"optimal value = {model.objVal:.2f}")
        # for v in model.getVars():
        #     print(f"optimal {v.varName} = {v.x:.2f}")
    
        if model.status == gp.GRB.OPTIMAL:
            # print('Distance sucessfully caluclated ' )
            return model.objVal
        if model.status == gp.GRB.INF_OR_UNBD:
            # print('Unbbounded Solution in Convex Hull Distance Calculation')
            sys.exit()
        elif model.status != gp.GRB.INFEASIBLE:
            # print('Optimization was stopped with status %d' % model.status)
            sys.exit(0) 

    # compute important constants

    p = problem.nr_obj

    supported_images = set()

    y = problem.weighted_sum([1 for _ in range(p)])
    supported_images.add(y)

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
    
    ''' initalize inner approximation with initial solution -> syipy.spatial class ConvexHull to reduce number of points given to lp solver'''

    if problem.type == "min":
        # add extreme points of weight set
        I = [ [0 for _ in range(p)] ]
        for i in range(p -1):
            I.append( [0 for _ in range(i)] + [1] + [0 for _ in range(i+1, p-1)] + [0] )
        
    elif problem.type == "max":
        I = [ [0 for _ in range(p - 1)] + [problem.UB] ]
        for i in range(p -1):
            I.append( [0 for _ in range(i)] + [1] + [0 for _ in range(i+1, p-1)] + [problem.UB] )

    # add point implied by found image y
    I.append([  1/p for _ in range(p -1) ] +  [sum(i for i in y)/( p)] )
    
    inner_approximation = ConvexHull(I, incremental=True)


    ''' main loop ''' 
    print('Start outer approximation....')

    # get extreme points of outer_approximation as list
    vertices = outer_approximation.intersections.tolist()

    # initialize list of investigated weights as kdTree with dimension number_criteria - 1
    L = make_kd_tree([[1/p for i in range(0, p -1)]],p-1)

    while len(vertices) > 0: 
        next_vertex = vertices.pop()

        if next_vertex[-1] < TOL and problem.type == "min":
            continue
        if next_vertex[-1] > problem.UB - TOL and problem.type == "max":
            continue
        
        if already_checked(next_vertex):
            continue

        # correct vertice
        for i in range(len(next_vertex)-1):
            if next_vertex[i] < 0:
                next_vertex[i] = 0
            elif next_vertex[i] > 1:
                next_vertex[i] = 1

        # check distance to inner approximation
        distance = distanzConvexHull(inner_approximation.points[inner_approximation.vertices] ,  np.array(next_vertex))
        
        if abs(distance - next_vertex[-1]) < TOL:
            #approximation error is alrady achive a vertice -> go to next vertice
            continue

        #solve routine and update outer and inner approximation
        # get new objectives
        obj_values = problem.weighted_sum( next_vertex[:-1] + [1 - sum(i for i in next_vertex[:-1])]     )

        L.append(next_vertex[:-1])

        supported_images.add(obj_values)

        if abs( sum( obj_values[i] *next_vertex[i] for i in range(len(obj_values)-1)) + (1 - sum(j for j in next_vertex[:-1])) * obj_values[-1] - next_vertex[-1]) < TOL:
            continue
        
        # build halfspace and add it to outer_approximation
        halfspace = [  obj_values[-1] - i for i in obj_values[:-1] ] + [1,-obj_values[-1]]
        outer_approximation.add_halfspaces([halfspace])
        vertices = outer_approximation.intersections.tolist()[1:] # better routine needed to avoid redundant tests!!!

        # update inner_approximation
        inner_approximation.add_points(   [[ i for i in next_vertex[:-1]] + [  sum(next_vertex[i]*obj_values[i] for i in range(len(obj_values)-1)) + (1 - sum(i for i in next_vertex[:-1]))*obj_values[-1]   ] ])
    return supported_images
    