r"""
==============================================================
Epsilon Convex Indicator for Multiobjective Approximation Sets 
==============================================================

Implementation of the epsilon convex indicator as a quality indicator for 
multiobjective approximations sets.

For further information, see 
S. Helfrich. "Approximation and Scalarization in Multiobjective Optimization". PhD Thesis, RPTU Kaiserslautern-Landau. 2023.
"""
from molib.core.image import Image
from molib.core.linear_programming import convex_approximation_factor, weighted_sum_ratio_on_cell


from typing import List,Set
from itertools import product
import warnings
import numpy as np


def get_epsilon_convex_indicator(images: Set[Image], Y_ESN: Set[Image], type: str, method = "grid", n: int = 100) -> float:
    r'''Get Convex Epsilon Indicator

    Determine the epsilon indicator of a sets of images to a reference set that contains all extreme supported nondominated images.
    This is done by computing::

        sup_{w > 0}  max{1, min_{y' in Y'} w^T y' / min_{y in Y} w^T y}    (minimization)
        sup_{w > 0}  max{1, max_{y in Y} w^T y / max_{y' in Y'} w^T y'}    (maximization)

    Parameters
    ----------
    ...
    method : str ("grid" or "exact")
        Method that is applied to obtain the epsilon convex indicator:

        - "grid": the supremum over the weight set is approximated by an additive grid (lower bound).
        - "exact": the supremum is determined exactly by one linear program per image y of Y_ESN,
          which maximizes the ratio over the weights for which y is optimal (scipy/HiGHS).
        - "gurobi": deprecated alias of "exact".
    ...
    Notes
    -----
    All images are assumed to be positive.
    '''
    if method == "gurobi":
        warnings.warn('method "gurobi" is deprecated and computed with method "exact" (no Gurobi required)',
                      DeprecationWarning, stacklevel=2)
        method = "exact"
    if method not in ("grid", "exact"):
        raise ValueError(f'method must be "grid" or "exact", got {method!r}')

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
    
    # method == "exact": one linear program per image of the reference set
    Q = np.array([np.asarray(q, dtype=float) for q in images])
    Y = np.array([np.asarray(y, dtype=float) for y in Y_ESN])
    approx_quality = 1.0
    for j in range(len(Y)):
        ratio = weighted_sum_ratio_on_cell(Q, Y, j, type)
        if ratio is not None and ratio > approx_quality:
            approx_quality = ratio
    return approx_quality


def get_epsilon_convex_indicator_with_nondominated_set(images: Set[Image], Y_N: Set[Image], type: str) -> float:
    r'''Get Convex Epsilon Indicator

    Determine the epsilon indicator of a sets of images to a reference set that contains all nondominated images.
    This is done by computing

    \max_{y \in Y_N} \min { t >= 1 : z <= t * y  for some z \in conv(Y') }   in the case that all objectives are to be minimized
    \max_{y \in Y_N} \min { t >= 1 : t * z >= y  for some z \in conv(Y') }   in the case that all objectives are to be maximized

    with one linear program per image y (scipy/HiGHS).

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
    All images are assumed to be positive.
    '''
    if type not in ("min", "max"):
        raise ValueError(f'type must be "min" or "max", got {type!r}')

    Q = np.array([np.asarray(q, dtype=float) for q in images])
    approx_quality = 1.0
    for y in Y_N:
        factor = convex_approximation_factor(Q, y, type)
        if factor > approx_quality:
            approx_quality = factor
    return approx_quality