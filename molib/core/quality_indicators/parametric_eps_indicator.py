"""
==============================================================
Epsilon Convex Indicator for Multiobjective Approximation Sets 
==============================================================

Implementation of the epsilon convex indicator as a quality indicator for 
multiobjective approximations sets.

For further information, see 
S. Helfrich. "Approximation and Scalarization in Multiobjective Optimization". PhD Thesis, RPTU Kaiserslautern-Landau. 2023.
"""
from molib.core.image import Image


from typing import List,Set
from itertools import product
import numpy as np


def get_epsilon_convex_indicator(images: Set[Image], Y_ESN: Set[Image], type: str, method = "grid", n: int = 100) -> float:
    '''Get Convex Epsilon Indicator

    Determine the epsilon indicator of a sets of images to a reference set that contains all extreme supported nondominated images.
    This is done by computing

    \sup_{w \in \R^p_>}  max{1 , \inf_{y' \in Y'} w^T y' /  (inf_{y \in Y} w^T y )} in the case  that all objectives are to be minimized
    \sup_{w \in \R^p_>}  max{1 , (sup_{y \in Y} w^T y ) / \sup_{y' \in Y'}w^T y' } in the case that all objectives are to be maximized
        
    Parameters
    ----------
    images : Set[Image] of List[Image]
        Images of approximation set
        
    Y_ESN : Set[Image] or List[Image]
        reference set that that contains all extreme supported nondomianted images
    
    type : str (either "min" of "max")
        string that specifies if the epsilon indicator should be determined for a minimization problem ("min") or a maximization problem ("max")

    method : str (either "grid" or "gurobi")
        method that is applied to obtain epsilon convex indicator. 
        "grid" : \sup_{w \in \R^p_>} ... is approxiamted by an additive grid over the weight set
        "gurobi" \sup_{w \in \R^p_>} ... is determined by an appropriate nonlinear mixed-integer program solved with gurobi.

    n : int
        relevant if method "grid" is chosen. Specifies the densitiy of the grid. Computation time and accuracy increases, if n is chosen larger. 

    Returns
    -------
    float
        epsilon convex indicator of the approxiamtion set. I.e., smallest '\alpha' such that the set 'images' constitutes an 'alpha'-convex approximation set.

    Comment
    -------
    if method "gurobi" is chosen, the Gurobi solver must be available
    '''  
    assert method in ["grid","gurobi"]

    if method == "grid":
        approx_quality = 1.0

        solutions = list(images)
        exact_solutions = list(Y_ESN)


        nr_criteria = len(solutions[0])
        for k in product(np.arange(0.0, 1.0+1/n, 1/n), repeat = nr_criteria-1):
            if sum(k[:-1]) > 1:
                continue
            values = [sum([k[s]*solution[s] for s in range(0, nr_criteria-1)]) + (1-sum(k[:-1]))*solution[-1] for solution in solutions]
            exact_values = [sum([k[s]*solution[s] for s in range(0, nr_criteria-1)]) + (1-sum(k[:-1]))*solution[-1] for solution in exact_solutions]

            if type == "min":
                values = [sum([k[s]*solution[s] for s in range(0, nr_criteria-1)]) + (1-sum(k[:-1]))*solution[-1] for solution in solutions]
                exact_values = [sum([k[s]*solution[s] for s in range(0, nr_criteria-1)]) + (1-sum(k[:-1]))*solution[-1] for solution in exact_solutions]
                if min(values)/min(exact_values) > approx_quality:
                    approx_quality = min(values)/min(exact_values)

            if type == "max": 
                if max(exact_values)/max(values) > approx_quality:
                    approx_quality = max(exact_values)/max(values)

        return approx_quality
    
    if method == "gurobi":
        try:
            import gurobipy as gp
            import sys
        except:
            print('gurobi not found')
            return float('inf')
        
        Q = list(images)
        Y = list(Y_ESN)


        model = gp.Model('parametric epsilon indicator')
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 0
        model.params.NonConvex = 2

        # create variable for parameter
        lam = model.addVars(len(Q[0]), name="lambda",lb = 0, ub = 1)

        # create variables z per point y \in Y which represents (lambda_1,\ldots, lambda_d)^top y
        z = model.addVars(len(Y), name = "z")

        # create variables z per point hat_y \in images which represents (1,lambda_1,\ldots, lambda_K)^top hat_y 
        z2 = model.addVars(len(Q), name = "z2")

        # create variable for min/max z
        m = model.addVar(name = "m",lb=0)

        # create variable for min z2
        m2 = model.addVar(name = "m2",lb=0)

        # creat obj variable
        t = model.addVar(name = 't',lb = 1, ub = 5)

        # add weight vector constraint
        model.addConstr(gp.quicksum(lam[i] for i in range(len(lam)) )== 1)

        # add constraint that z,z2 represent weighted sum
        for j, y in enumerate(Y):
            model.addConstr(gp.quicksum( lam[i] * y[i] for i in range(len(lam))  ) == z[j])

        for j, y in enumerate(Q):
            model.addConstr(gp.quicksum( lam[i] * y[i] for i in range(len(lam))  ) == z2[j])


        # add that m = min/max z, m2 = min/max z2, t = m/m2 or t = m2/m
        if type == "min":
            model.addConstr(m == gp.min_(z))
            model.addConstr(m2 == gp.min_(z2))
            model.addConstr(t * m == m2)


        elif type == "max":
            model.addConstr(m == gp.max_(z))
            model.addConstr(m2 == gp.max_(z2))
            model.addConstr(t * m2 == m)
        else:
            print('type not specified correctly')
            return float('inf')
        
        model.setObjective(t, gp.GRB.MAXIMIZE)

        model.update()

        model.optimize()

        if model.status == gp.GRB.OPTIMAL:

            opt_value = model.objVal
            return opt_value
        if model.status == gp.GRB.INF_OR_UNBD:
            print('Unbounded Solution in Convex Hull Distance Calculation')
            sys.exit()
        elif model.status == gp.GRB.INFEASIBLE:
            sys.exit(0) 
        elif model.status == gp.GRB.SUBOPTIMAL:
            print('Optimization was stopped with status %d' % model.status)
            
            opt_value = model.objVal

            print('Real approximation error is between ', model.opt_value, ' and ', model.objBoundC, '. Return upper bound')
            return model.objBoundC
        
        sys.exit(0)


def get_epsilon_convex_indicator_with_nondominated_set(images: Set[Image], Y_N: Set[Image], type: str) -> float:
    '''Get Convex Epsilon Indicator

    Determine the epsilon indicator of a sets of images to a reference set that contains all nondominated images.
    This is done by computing

    \max__{i=1,...p} \max_{y \in Y_N} inf_{y' \in conv(Y')} max{1, y'_i / y_i} in the case that all objectives are to be minimized
    \max__{i=1,...p} \max_{y \in Y_N} inf_{y' \in conv(Y')} max{1, y_i / y'_i} in the case that all objectives are to be maximized
        
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
        epsilon convex indicator of the approxiamtion set. I.e., smallest '\alpha' such that the set 'images' constitutes an 'alpha'-convex approximation set.

    Comment
    -------
    Gurobi solver must be available
    '''

    approx_quality = 1.0

    solutions = list(images)
    exact_solutions = list(Y_N)

    # build gurobi model based on solutions
    try:
        import gurobipy as gp
        import sys
    except:
        print('gurobi not found')
        return float('inf')

    model = gp.Model('convex hull')
    model.Params.LogToConsole = 0
    model.Params.OutputFlag = 0
    # create variable per point in points
    theta = model.addVars(len(solutions), lb = 0, ub = 1, name = 'theta')

    # creat convex combination variables
    y = model.addVars(len(solutions[0]), name= "y") 

    # create obj variable
    t = model.addVar(lb = 0, name = 't')

    # add summ constraint
    model.addConstr( gp.quicksum( theta[i] for i in range(len(theta))  ) == 1 )

    # add constraints
    a = model.addConstrs( (gp.quicksum(solutions[i][dim] * theta[i] for i in range(len(theta)) ) == y[dim]  for dim  in range(len(solutions[0])) ) , name = "a" )
    if type == "max":
        c = model.addConstrs( (t * y[dim]  >= 0 for dim in range(len(solutions[0])) ), name = 'c' )

    if type == "min":
        c = model.addConstrs( (t * 0 - y[dim]  >= 0 for dim in range(len(solutions[0])) ), name = 'c' )

    # set objective
    model.setObjective(t, gp.GRB.MINIMIZE  )

    model.update()

    flag = False
    for exact_sol in exact_solutions:
        #add exact solution to gurobi model
        if type == "max":
            for dim in range(len(exact_sol)):
                c[dim].setAttr("qcrhs", exact_sol[dim])

        if type == "min":
            for dim in range(len(exact_sol)):
                model.chgCoeff(c[dim],t,exact_sol[dim])

        model.update()
        model.reset()
        #solve for best coonvex combinations of solutions with respect to approximation factors
        model.optimize()
        if model.status == gp.GRB.OPTIMAL:
            # print('Distance sucessfully caluclated ' )
            opt_value = model.objVal
        if model.status == gp.GRB.INF_OR_UNBD:
            print('Unbounded Solution in Convex Hull Distance Calculation')
            sys.exit()
        elif model.status == gp.GRB.INFEASIBLE:
            print('Optimization was stopped with status %d' % model.status)
            sys.exit(0) 
        elif model.status == gp.GRB.SUBOPTIMAL:
            flag = True
            opt_value = model.objVal
            print('suboptimal problem with MIPGap')

        #store worst approx quality
        if approx_quality < opt_value:
            approx_quality = opt_value

    if flag:
        print('There has been a suboptimal problem -> approximation quality might not be exact')
    return approx_quality

