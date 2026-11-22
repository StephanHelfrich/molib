import warnings

import numpy as np
import pytest

from molib.algorithms.exact_methods import dual_outer_approximation
from molib.core.linear_programming import max_last_coordinate_in_hull
from molib.core.quality_indicators import (get_epsilon_convex_indicator,
                                           get_epsilon_convex_indicator_with_nondominated_set)
from molib.problems.image_set import ImageSet_withLexWS

from helpers import nondominated_set, random_images


def reference_sets(seed, p, sense):
    Y = random_images(40, p, seed)
    N = np.array(sorted(nondominated_set(Y, sense)))
    ESN = np.array([np.asarray(y) for y in dual_outer_approximation(ImageSet_withLexWS(Y, type=sense))])
    return N, ESN


def test_hand_example():
    # w = (1, 0): 2 / 1 = 2 is the worst ratio
    assert np.isclose(get_epsilon_convex_indicator([[2, 2]], [[1, 3], [3, 1]], "min", method="exact"), 2)
    assert np.isclose(get_epsilon_convex_indicator_with_nondominated_set([[2, 2]], [[1, 3], [3, 1]], "min"), 2)
    assert np.isclose(get_epsilon_convex_indicator([[2, 2]], [[1, 3], [3, 1]], "max", method="exact"), 1.5)
    assert np.isclose(get_epsilon_convex_indicator_with_nondominated_set([[2, 2]], [[1, 3], [3, 1]], "max"), 1.5)


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_exact_methods_agree_and_bound_grid(sense, p, seed):
    N, ESN = reference_sets(seed, p, sense)
    rng = np.random.default_rng(seed)
    Q = N[rng.choice(len(N), size=max(1, len(N) // 3), replace=False)]

    exact = get_epsilon_convex_indicator(Q, ESN, sense, method="exact")
    via_nondominated_set = get_epsilon_convex_indicator_with_nondominated_set(Q, N, sense)
    grid = get_epsilon_convex_indicator(Q, ESN, sense, method="grid", n=60 if p == 3 else 200)

    assert np.isclose(exact, via_nondominated_set, rtol=1e-9)
    assert grid <= exact + 1e-9                     # the grid yields a lower bound
    assert np.isclose(get_epsilon_convex_indicator(ESN, ESN, sense, method="exact"), 1)
    assert np.isclose(get_epsilon_convex_indicator_with_nondominated_set(N, N, sense), 1)


def test_gurobi_method_is_deprecated_alias():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = get_epsilon_convex_indicator([[2, 2]], [[1, 3], [3, 1]], "min", method="gurobi")
    assert np.isclose(value, 2) and caught[0].category is DeprecationWarning
    with pytest.raises(ValueError):
        get_epsilon_convex_indicator([[2, 2]], [[1, 3]], "min", method="xyz")


def test_max_last_coordinate_in_hull():
    points = np.array([[0.0, 1.0], [1.0, 3.0], [1.0, 2.0]])
    assert np.isclose(max_last_coordinate_in_hull(points, [0.5]), 2.0)
    assert max_last_coordinate_in_hull(points, [2.0]) is None