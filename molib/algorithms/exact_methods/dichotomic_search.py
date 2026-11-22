# -*- coding: utf-8 -*-
"""
=================
Dichotomic search
=================

Computation of all extreme supported nondominated images of a biobjective optimization problem
by the dichotomic search of Aneja and Nair, see

Y. P. Aneja and K. P. K. Nair. "Bicriteria Transportation Problem."
Management Science 25(1):73-78, 1979.
"""
from typing import Set

import numpy as np

from molib.core.image import Image
from molib.core.problem import Problem

TOL = 1e-6

__all__ = ["dichotomic_search"]


def dichotomic_search(problem: Problem, tol: float = TOL) -> Set[Image]:
    """Dichotomic search of Aneja and Nair for biobjective problems.

    Starting from the images obtained for the weights (1, 0) and (0, 1), the weighted sum
    scalarization is solved recursively with the normal vector of the segment between two
    adjacent images. A new image is accepted if its weighted sum value is smaller (larger for
    maximization) than the value of the segment by more than ``tol`` (relative to the magnitude
    of the value).

    Parameters
    ----------
    problem : Problem
        Biobjective problem for which an exact weighted sum algorithm is implemented, i.e.,
        ``solution_quality_weighted_sum == 1``. The weighted sum algorithm is assumed to return
        nondominated images also for weights with a zero entry (e.g., by lexicographic
        tie-breaking); otherwise, the two endpoints may be weakly nondominated only.
    tol : float
        Relative tolerance for comparisons of weighted sum values and images.

    Returns
    -------
    Set[Image]
        Images (with solutions) that contain all extreme supported nondominated images,
        including the two lexicographically optimal images.
    """
    if problem.nr_obj != 2:
        raise ValueError(f"dichotomic search requires exactly two objectives, got {problem.nr_obj}")
    sign = 1.0 if problem.type == "min" else -1.0

    def close(a, b) -> bool:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return bool(np.all(np.abs(a - b) <= tol * np.maximum(1.0, np.abs(a))))

    y_l = problem.weighted_sum([1.0, 0.0])
    y_r = problem.weighted_sum([0.0, 1.0])

    images: Set[Image] = {y_l}
    if close(y_l, y_r):
        return images
    images.add(y_r)

    stack = [(y_l, y_r)]
    while stack:
        a, b = stack.pop()
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)

        # normal vector of the segment [a, b], oriented such that it is positive
        w = sign * np.array([a_arr[1] - b_arr[1], b_arr[0] - a_arr[0]])
        if np.any(w <= 0.0):
            continue
        w = w / w.sum()

        y = problem.weighted_sum(w.tolist())
        if close(y, a) or close(y, b):
            continue

        reference = float(w @ a_arr)
        if sign * float(w @ np.asarray(y, dtype=float)) < sign * reference - tol * max(1.0, abs(reference)):
            images.add(y)
            stack.append((a, y))
            stack.append((y, b))

    return images