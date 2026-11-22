import numpy as np
import pytest

from molib.algorithms.exact_methods import dichotomic_search, generic_upper_bound
from molib.core import (EXACT, Guarantee, GuaranteeNotSatisfiedError, Image, Problem, ScalarizationResult,
                        UnsupportedScalarizationError, available_solvers, implements, record_solves, solve,
                        solver_info)
from molib.problems.image_set import ImageSet, ImageSet_withLexWS
from molib.scalarizations import EpsilonConstraint, WeightedSum, WeightedTchebycheff

from helpers import random_images


# --------------------------------------------------------------------------- test problems
class LegacyProblem(Problem):
    """Only the legacy method weighted_sum (as e.g. Knapsack and TravelingSalesman)."""

    def __init__(self, Y, quality=1.5):
        super().__init__("min", Y.shape[1])
        self.Y = Y
        self.solution_quality_weighted_sum = quality

    def weighted_sum(self, weights):
        i = int(np.argmin(self.Y @ np.asarray(weights)))
        return Image(self.Y[i], solution=i)


class BoundedSolverProblem(Problem):
    """Registered solver that returns a ScalarizationResult with dual bound (as a MIP solver)."""

    def __init__(self, Y):
        super().__init__("min", Y.shape[1])
        self.Y = Y

    @implements(WeightedSum, guarantee=Guarantee.multiplicative(1.1), name="mip", lexicographic=True)
    def _solve(self, s):
        values = self.Y @ s.weights
        i = int(np.argsort(values)[1])          # second best image, bound = optimal value
        return ScalarizationResult(image=Image(self.Y[i]), status="time_limit",
                                   dual_bound=float(values.min()), guarantee=Guarantee.none())


# --------------------------------------------------------------------------- dispatch
def test_registered_solvers_of_image_sets():
    Y = random_images(30, 3, seed=0)
    for problem, ws_guarantee in ((ImageSet(Y, solution_quality_weighted_sum=2), Guarantee.multiplicative(2)),
                                  (ImageSet_withLexWS(Y), EXACT)):
        infos = {info.scalarization: info for info in available_solvers(problem)}
        assert set(infos) == {WeightedSum, WeightedTchebycheff, EpsilonConstraint}
        assert infos[WeightedSum].guarantee == ws_guarantee
        assert infos[WeightedTchebycheff].guarantee == EXACT


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("scalarization", [
    WeightedSum([0.2, 0.3, 0.5]),
    WeightedTchebycheff([0.2, 0.3, 0.5], [0, 0, 0], augmentation=1e-6),
    EpsilonConstraint(0, [np.inf, 60, 60]),
])
def test_solve_matches_brute_force(sense, scalarization):
    Y = random_images(50, 3, seed=1)
    if sense == "max":
        if isinstance(scalarization, WeightedTchebycheff):
            scalarization = WeightedTchebycheff(scalarization.weights, [100, 100, 100], 1e-6)
        if isinstance(scalarization, EpsilonConstraint):
            scalarization = EpsilonConstraint(0, [-np.inf, 40, 40])
    problem = ImageSet_withLexWS(Y, type=sense)
    result = solve(problem, scalarization)

    feasible = scalarization.feasible(Y, sense)
    values = scalarization.value(Y[feasible], sense)
    best = values.min() if scalarization.direction(sense) == "min" else values.max()
    assert result.found and result.status == "solved" and result.guarantee == EXACT
    assert np.isclose(result.value, best)
    assert np.array_equal(Y[result.image.solutions[0]], np.asarray(result.image))
    assert result.direction == scalarization.direction(sense)
    assert result.scalarization is scalarization and result.runtime >= 0


def test_infeasible_epsilon_constraint():
    Y = random_images(20, 2, seed=2)
    result = solve(ImageSet_withLexWS(Y), EpsilonConstraint(0, [np.inf, 0]))
    assert not result.found and result.status == "infeasible" and result.value is None


def test_legacy_fallback_uses_solution_quality():
    problem = LegacyProblem(random_images(20, 2, seed=3), quality=1.5)
    info = solver_info(problem, WeightedSum([1, 1]))
    assert info.name == "weighted_sum" and info.guarantee == Guarantee.multiplicative(1.5)
    assert solve(problem, WeightedSum([1, 1])).found
    with pytest.raises(UnsupportedScalarizationError):
        solve(problem, WeightedTchebycheff([1, 1], [0, 0]))
    with pytest.raises(UnsupportedScalarizationError):
        solve(problem, EpsilonConstraint(0, [np.inf, 50]))


def test_require_guarantee():
    problem = LegacyProblem(random_images(20, 2, seed=3), quality=1.5)
    assert solve(problem, WeightedSum([1, 1]), require=Guarantee.multiplicative(2)).found
    with pytest.raises(GuaranteeNotSatisfiedError):
        solve(problem, WeightedSum([1, 1]), require=EXACT)


def test_wrong_dimension_and_type():
    problem = ImageSet_withLexWS(random_images(10, 2, seed=0))
    with pytest.raises(ValueError):
        solve(problem, WeightedSum([1, 1, 1]))
    with pytest.raises(TypeError):
        solve(problem, [1, 1])


def test_result_of_solver_with_dual_bound():
    Y = random_images(20, 2, seed=4)
    problem = BoundedSolverProblem(Y)
    info = solver_info(problem, WeightedSum([1, 1]))
    assert info.name == "mip" and info.capabilities == {"lexicographic": True}

    result = solve(problem, WeightedSum([1, 1]))
    values = np.sort(Y @ np.ones(2))
    assert result.status == "time_limit"
    assert result.guarantee == Guarantee.none()          # set by the solver, not overwritten
    assert np.isclose(result.realized_factor, values[1] / values[0])


def test_subclass_overrides_registered_solver():
    class Inexact(ImageSet_withLexWS):
        @implements(WeightedSum, guarantee=Guarantee.multiplicative(3))
        def _solve_weighted_sum(self, s):
            return super()._solve_weighted_sum(s)

    problem = Inexact(random_images(10, 2, seed=0))
    assert solver_info(problem, WeightedSum([1, 1])).guarantee == Guarantee.multiplicative(3)


# --------------------------------------------------------------------------- logging
def test_record_solves_counts_calls_of_algorithms():
    Y = random_images(200, 2, seed=5, high=1000)
    with record_solves() as outer:
        with record_solves() as log:
            result = dichotomic_search(ImageSet_withLexWS(Y))
        generic_upper_bound(ImageSet_withLexWS(Y), LB=Y.min(axis=0), UB=Y.max(axis=0))
    assert log.nr_calls == 2 * len(result) - 1
    assert log.calls_by_scalarization() == {"WeightedSum": log.nr_calls}
    counts = outer.calls_by_scalarization()
    assert counts["WeightedSum"] == log.nr_calls and counts["WeightedTchebycheff"] > 0
    assert outer.total_runtime >= log.total_runtime >= 0
    # no logging outside of the block
    solve(ImageSet_withLexWS(Y), WeightedSum([1, 1]))
    assert outer.nr_calls == sum(counts.values())