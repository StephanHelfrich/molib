import numpy as np
import pytest

from molib.algorithms.exact_methods import dichotomic_search, dual_outer_approximation
from molib.problems.image_set import ImageSet_withLexWS

from helpers import as_tuples, nondominated_set, random_images


def weighted_sum_optimality_violations(result, Y, sense="min", nr_weights=500, seed=0):
    """Number of random weights for which the best weighted sum value over `result` differs from
    the optimal value over Y, i.e., result is not an optimal image set for the weighted sum."""
    rng = np.random.default_rng(seed)
    R = np.array([np.asarray(y, dtype=float) for y in result])
    W = rng.dirichlet(np.ones(Y.shape[1]), size=nr_weights)
    if sense == "min":
        best_result, best = (R @ W.T).min(axis=0), (Y @ W.T).min(axis=0)
    else:
        best_result, best = (R @ W.T).max(axis=0), (Y @ W.T).max(axis=0)
    return int(np.sum(np.abs(best_result - best) > 1e-6 * np.maximum(1, np.abs(best))))


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_dual_outer_approximation_min(p, seed):
    # images below 100: the former interior point f(x,w) - 100 made qhull fail here
    Y = random_images(50, p, seed)
    result = dual_outer_approximation(ImageSet_withLexWS(Y))
    assert as_tuples(result) <= nondominated_set(Y)
    assert weighted_sum_optimality_violations(result, Y) == 0
    if p == 2:
        assert as_tuples(result) == as_tuples(dichotomic_search(ImageSet_withLexWS(Y)))


@pytest.mark.parametrize("p", [2, 3])
def test_dual_outer_approximation_max(p):
    Y = random_images(50, p, seed=11)
    result = dual_outer_approximation(ImageSet_withLexWS(Y, type="max"))
    assert as_tuples(result) <= nondominated_set(Y, "max")
    assert weighted_sum_optimality_violations(result, Y, sense="max") == 0