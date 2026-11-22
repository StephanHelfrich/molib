import numpy as np
import pytest

from molib.algorithms.exact_methods import generic_upper_bound
from molib.problems.image_set import ImageSet_withLexWS

from helpers import as_tuples, nondominated_set, random_images


class CountingImageSet(ImageSet_withLexWS):
    """ImageSet that counts the calls of the weighted Tchebycheff scalarization."""

    def __init__(self, Y, type="min"):
        super().__init__(Y, type)
        self.calls = 0

    def weighted_tchebycheff(self, *args, **kwargs):
        self.calls += 1
        return super().weighted_tchebycheff(*args, **kwargs)


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("augmentation", [0.0, 1e-6])
def test_exact(sense, p, seed, augmentation):
    Y = random_images(60, p, seed)
    problem = CountingImageSet(Y, type=sense)
    result = generic_upper_bound(problem, LB=Y.min(axis=0), UB=Y.max(axis=0), augmentation=augmentation)
    assert as_tuples(result) == nondominated_set(Y, sense)
    assert problem.calls >= len(result)
    for y in result:
        assert np.array_equal(Y[y.solutions[0]], np.asarray(y))


@pytest.mark.parametrize("sense", ["min", "max"])
def test_problem_bounds_and_initial_images(images_3d, sense):
    problem = ImageSet_withLexWS(images_3d, type=sense)
    result = generic_upper_bound(problem, initial_images=images_3d[:10])  # scalar problem.LB / UB
    assert as_tuples(result) == nondominated_set(images_3d, sense)


def test_approximation_thresholds(images_3d):
    problem = ImageSet_withLexWS(images_3d)
    approx, lubs = generic_upper_bound(problem, LB=images_3d.min(axis=0), UB=images_3d.max(axis=0),
                                       approximation_thresholds=30, return_upper_bound_set=True)
    assert as_tuples(approx) <= nondominated_set(images_3d)
    assert lubs.nr_nonverified_upper_bounds() == 0


def test_time_limit(images_3d):
    problem = CountingImageSet(images_3d)
    result, lubs = generic_upper_bound(problem, LB=1, UB=99, time_limit=0.0, return_upper_bound_set=True)
    assert result == set() and problem.calls == 0
    assert lubs.nr_nonverified_upper_bounds() == 1


def test_initial_image_outside_bounds(images_3d):
    with pytest.raises(ValueError):
        generic_upper_bound(ImageSet_withLexWS(images_3d), LB=1, UB=99, initial_images=[[0, 50, 50]])