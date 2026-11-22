# -*- coding: utf-8 -*-
"""
==========================
Generic upper bound method
==========================

Computation of the nondominated set of a multiobjective minimization problem by successively
exploring the search zones of local upper bounds with an (augmented) weighted Tchebycheff
scalarization, cf.

K. Klamroth, R. Lacour, D. Vanderpooten. "On the representation of the search region in
multi-objective optimization." European Journal of Operational Research 245(3):767-778, 2015.

With positive ``approximation_thresholds``, search zones whose box is small enough are not
explored, and the method returns a representation of the nondominated set whose quality is
controlled by the thresholds. With ``initial_images`` (e.g., the result of a weighted sum sweep)
and a ``time_limit``, the method can be used as an anytime heuristic.
"""
import time
from typing import Iterable, Optional, Sequence, Set, Tuple, Union

import numpy as np

from molib.core.dominance import leqq, nondominated, strict_leq
from molib.core.image import Image
from molib.core.local_upper_bound import LocalUpperBoundSet
from molib.core.problem import Problem

__all__ = ["generic_upper_bound"]


def _as_vector(value, p: int, name: str) -> np.ndarray:
    try:
        return np.broadcast_to(np.asarray(value, dtype=float), (p,)).copy()
    except ValueError as err:
        raise ValueError(f"{name} must be a scalar or a vector of length {p}") from err


def generic_upper_bound(
    problem: Problem,
    LB: Optional[Union[float, Sequence[float]]] = None,
    UB: Optional[Union[float, Sequence[float]]] = None,
    approximation_thresholds: Optional[Union[float, Sequence[float]]] = None,
    initial_images: Optional[Iterable] = None,
    augmentation: float = 1e-6,
    margin: float = 1.0,
    update_method: str = "ra",
    largest_first: bool = True,
    time_limit: Optional[float] = None,
    return_upper_bound_set: bool = False,
) -> Union[Set[Image], Tuple[Set[Image], LocalUpperBoundSet]]:
    r"""Generic upper bound method of Klamroth, Lacour and Vanderpooten (2015).

    The box [LB - margin, UB + margin) is described by a local upper bound set. In each
    iteration, a non-verified local upper bound u is chosen, and the augmented weighted
    Tchebycheff scalarization with reference point r = LB - margin and weights
    w_i = 1 / (u_i - r_i) is solved. If the returned image z lies in the search zone of u
    (z < u), the local upper bound set is updated with z. Otherwise, u is verified.

    For augmentation = 0 and an exact scalarization, the minimizer lies in the search zone of u
    whenever the search zone contains an image, since max_i w_i (z_i - r_i) < 1 holds exactly
    for the images z < u. For augmentation > 0, this holds if the augmentation is sufficiently
    small relative to the differences of the Tchebycheff values.

    Parameters
    ----------
    problem : Problem
        Minimization problem that implements ``weighted_tchebycheff(weights, reference_point,
        augmentation)``.
    LB, UB : float or sequence of float, optional
        Componentwise lower and upper bounds on all nondominated images. Default: problem.LB and
        problem.UB. UB has to be an upper bound on the nadir point; for three or more objectives,
        the maxima of the lexicographically optimal images are in general no valid upper bounds.
    approximation_thresholds : float or sequence of float, optional
        A local upper bound whose box widths are componentwise at most these values is not
        explored. Default: 0 (exact computation).
    initial_images : iterable, optional
        Images that are inserted before the main loop, e.g., from a weighted sum sweep. Dominated
        images are allowed; they are filtered from the result.
    augmentation : float
        Augmentation parameter rho of the weighted Tchebycheff scalarization.
    margin : float
        The box is enlarged by ``margin`` in every component, so that images attaining LB_i or
        UB_i still refine the local upper bound set.
    update_method : str
        Update method of the local upper bound set ('ra', 're' or 're_advanced').
    largest_first : bool
        Explore the local upper bound with the largest box first (default) or the smallest.
    time_limit : float, optional
        Time limit in seconds. If it is reached, the method stops and returns the images found so
        far; the non-verified local upper bounds describe the unexplored region.
    return_upper_bound_set : bool
        If True, additionally return the final local upper bound set.

    Returns
    -------
    Set[Image] or (Set[Image], LocalUpperBoundSet)
        Nondominated images found (with solutions).
    """
    if problem.type != "min":
        raise NotImplementedError("The generic upper bound method is implemented for minimization problems only")

    p = problem.nr_obj
    if LB is None or UB is None:
        LB_problem, UB_problem = problem.get_bounds()
        LB = LB_problem if LB is None else LB
        UB = UB_problem if UB is None else UB
    LB = _as_vector(LB, p, "LB")
    UB = _as_vector(UB, p, "UB")
    if np.any(LB > UB):
        raise ValueError("LB must be componentwise at most UB")

    if approximation_thresholds is None:
        thresholds = np.zeros(p)
    else:
        thresholds = _as_vector(approximation_thresholds, p, "approximation_thresholds")

    reference_point = LB - margin
    upper_bound_set = LocalUpperBoundSet(reference_point, UB + margin, update_method=update_method)

    found: Set[Image] = set()
    for y in (initial_images if initial_images is not None else []):
        y = y if isinstance(y, Image) else Image(y)
        if np.any(np.asarray(y) < LB - 1e-9) or np.any(np.asarray(y) > UB + 1e-9):
            raise ValueError(f"Initial image {np.asarray(y).tolist()} is not contained in [LB, UB]")
        upper_bound_set.update(y)
        found.add(y)

    start = time.perf_counter()
    while True:
        if time_limit is not None and time.perf_counter() - start >= time_limit:
            break

        u = upper_bound_set.get_non_verified_upper_bound(largest_first=largest_first)
        if u is None:
            break

        if np.any(thresholds > 0) and leqq(u.widths, thresholds):
            u.set_verified()
            continue

        weights = 1.0 / (np.asarray(u, dtype=float) - reference_point)
        weights = weights / weights.sum()

        z = problem.weighted_tchebycheff(weights.tolist(), reference_point.tolist(), augmentation)
        if z is not None and strict_leq(z, u):
            z = z if isinstance(z, Image) else Image(z)
            upper_bound_set.update(z)
            found.add(z)
        else:
            u.set_verified()

    result = set(nondominated(found))
    if return_upper_bound_set:
        return result, upper_bound_set
    return result