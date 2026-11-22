import numpy as np
import pytest

from molib.scalarizations import EpsilonConstraint, WeightedSum, WeightedTchebycheff

Y = np.array([[1.0, 5.0], [2.0, 2.0], [5.0, 1.0], [3.0, 3.0]])


def test_weighted_sum_follows_problem_sense():
    s = WeightedSum([1, 1])
    assert np.allclose(s.value(Y, "min"), [6, 4, 6, 6])
    assert s.direction("min") == "min" and s.direction("max") == "max"
    assert s.best_index(Y, "min") == 1
    assert s.best_index(Y, "max") == 2      # tie (value 6) broken lexicographically: (5, 1) is largest


def test_weighted_tchebycheff_min_and_max():
    s = WeightedTchebycheff([0.5, 0.5], [0, 0])
    assert np.allclose(s.value(Y, "min"), [2.5, 1.0, 2.5, 1.5])
    assert s.direction("max") == "min"      # always minimized
    t = WeightedTchebycheff([0.5, 0.5], [6, 6])
    assert np.allclose(t.value(Y, "max"), [2.5, 2.0, 2.5, 1.5])
    assert t.best_index(Y, "max") == 3


def test_weighted_tchebycheff_augmentation():
    s = WeightedTchebycheff([1, 1], [0, 0], augmentation=0.1)
    assert np.allclose(s.value([[2, 1]], "min"), [2 + 0.1 * 3])


def test_epsilon_constraint():
    s = EpsilonConstraint(0, [np.inf, 2.5])
    assert np.array_equal(s.feasible(Y, "min"), [False, True, True, False])
    assert s.best_index(Y, "min") == 1
    assert EpsilonConstraint(0, [np.inf, 10]).best_index(Y, "min") == 0
    assert EpsilonConstraint(0, [np.inf, 0.5]).best_index(Y, "min") is None
    m = EpsilonConstraint(1, [2.5, -np.inf])
    assert np.array_equal(m.feasible(Y, "max"), [False, False, True, True])
    assert m.best_index(Y, "max") == 3


def test_validation():
    with pytest.raises(ValueError):
        WeightedSum([1, -1])
    with pytest.raises(ValueError):
        WeightedTchebycheff([1, 1], [0, 0, 0])
    with pytest.raises(ValueError):
        WeightedTchebycheff([1, 1], [0, 0], augmentation=-1)
    with pytest.raises(ValueError):
        EpsilonConstraint(2, [1, 1])
    with pytest.raises(ValueError):
        WeightedSum([1, 1]).validate(3)
    with pytest.raises(ValueError):
        WeightedSum([1, 1]).value(Y, "minimize")