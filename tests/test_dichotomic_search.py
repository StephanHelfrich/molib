import numpy as np
import pytest

from molib.algorithms.exact_methods import dichotomic_search
from molib.problems.image_set import ImageSet_withLexWS

from helpers import as_tuples, nondominated_set, random_images


def lower_left_convex_hull(N):
    """Extreme supported points of a biobjective nondominated set (minimization)."""
    hull = []
    for q in sorted(N):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (q[1] - y1) - (y2 - y1) * (q[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(q)
    return set(hull)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("sense", ["min", "max"])
def test_dichotomic_search(seed, sense):
    Y = random_images(80, 2, seed)
    result = dichotomic_search(ImageSet_withLexWS(Y, type=sense))
    if sense == "min":
        expected = lower_left_convex_hull(nondominated_set(Y))
    else:
        negated = {(-a, -b) for a, b in nondominated_set(Y, "max")}
        expected = {(-a, -b) for a, b in lower_left_convex_hull(negated)}
    assert as_tuples(result) == expected


def test_dichotomic_search_contains_lexicographic_endpoints():
    Y = np.array([[2, 62], [5, 21], [15, 11], [50, 1], [3, 70], [60, 1]], dtype=float)
    result = as_tuples(dichotomic_search(ImageSet_withLexWS(Y)))
    assert (2.0, 62.0) in result and (50.0, 1.0) in result


def test_dichotomic_search_number_of_weighted_sum_calls():
    class Counting(ImageSet_withLexWS):
        calls = 0

        def weighted_sum(self, weights):
            self.calls += 1
            return super().weighted_sum(weights)

    Y = random_images(500, 2, seed=0, high=1000)
    problem = Counting(Y)
    result = dichotomic_search(problem)
    assert problem.calls == 2 * len(result) - 1


def test_dichotomic_search_single_image():
    Y = np.array([[1, 1], [2, 3]], dtype=float)
    assert as_tuples(dichotomic_search(ImageSet_withLexWS(Y))) == {(1.0, 1.0)}


def test_dichotomic_search_requires_two_objectives():
    with pytest.raises(ValueError):
        dichotomic_search(ImageSet_withLexWS(random_images(10, 3, seed=0)))