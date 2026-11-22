# -*- coding: utf-8 -*-
"""
==========================
Solving scalarizations
==========================

Central entry point ``solve(problem, scalarization)`` for all algorithms. It selects a solver
for the combination of problem and scalarization, checks a required guarantee, measures the
running time, and returns a ``ScalarizationResult``.

Solvers are methods of problem classes that are registered with the decorator ``implements``::

    class MyProblem(Problem):
        @implements(WeightedSum, guarantee=Guarantee.multiplicative(2))
        def _solve_weighted_sum(self, scalarization):
            ...                     # return an Image, None (infeasible) or a ScalarizationResult

For problem classes that only implement the methods ``weighted_sum(weights)`` or
``weighted_tchebycheff(weights, reference_point, augmentation)`` of ``Problem``, these methods
are used as solvers (legacy interface). The guarantee of ``weighted_sum`` is then taken from
``solution_quality_weighted_sum``.
"""
import contextlib
import contextvars
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np

from .guarantees import EXACT, Guarantee
from .image import Image
from .problem import Problem
from molib.scalarizations import Scalarization, WeightedSum, WeightedTchebycheff

__all__ = [
    "ScalarizationResult",
    "SolverInfo",
    "SolveLog",
    "UnsupportedScalarizationError",
    "GuaranteeNotSatisfiedError",
    "implements",
    "solve",
    "solver_info",
    "available_solvers",
    "record_solves",
]

GuaranteeSpec = Union[Guarantee, str, Callable[[Any], Guarantee]]


class UnsupportedScalarizationError(NotImplementedError):
    """The problem provides no solver for the scalarization."""


class GuaranteeNotSatisfiedError(ValueError):
    """The solver of the problem does not provide the required guarantee."""


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class ScalarizationResult:
    """Result of solving a scalarization.

    Attributes
    ----------
    image : Image or None
        Image (with solutions) of the returned solution; None if no solution was found.
    status : str
        "solved" or "infeasible" for native solvers that return an image or None; solvers that
        return a ScalarizationResult may use further values (e.g. "optimal", "time_limit").
    value : float or None
        Value of the scalarization at ``image``.
    dual_bound : float or None
        Bound on the optimal value of the scalarization, if the solver provides one.
    guarantee : Guarantee
        Guarantee of this result. Filled with the guarantee of the registered solver unless the
        solver sets it itself (e.g. Guarantee.none() if a time limit was reached).
    runtime : float
        Running time of the solver in seconds.
    solver : str
        Name of the solver.
    scalarization : Scalarization or None
        The scalarization that was solved.
    """
    image: Optional[Image]
    status: str = "solved"
    value: Optional[float] = None
    dual_bound: Optional[float] = None
    guarantee: Optional[Guarantee] = None
    runtime: float = 0.0
    solver: str = ""
    scalarization: Optional[Scalarization] = None
    direction: Optional[str] = None     # optimization direction of the scalarization

    @property
    def found(self) -> bool:
        return self.image is not None

    @property
    def realized_factor(self) -> Optional[float]:
        """A posteriori multiplicative quality from value and dual bound, or None.

        For minimization value / dual_bound, for maximization dual_bound / value. Requires
        positive values.
        """
        if self.value is None or self.dual_bound is None or self.direction is None:
            return None
        if self.direction == "min":
            return self.value / self.dual_bound if self.dual_bound > 0 else None
        return self.dual_bound / self.value if self.value > 0 else None


@dataclass(frozen=True)
class SolverInfo:
    """Registration of a solver for a type of scalarization.

    Attributes
    ----------
    scalarization : type
        Type of scalarization (instances of subclasses are also accepted).
    guarantee : Guarantee
        Guarantee of the solver (already resolved for the problem instance in ``solver_info``).
    name : str
        Name of the solver (method name by default).
    capabilities : Mapping[str, Any]
        Further properties of the solver that algorithms may require (e.g. a lexicographic
        variant). Not used by ``solve`` itself.
    """
    scalarization: type
    guarantee: GuaranteeSpec = EXACT
    name: str = ""
    capabilities: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self, problem) -> "SolverInfo":
        g = self.guarantee
        if isinstance(g, str):
            g = Guarantee.from_factor(getattr(problem, g))
        elif callable(g) and not isinstance(g, Guarantee):
            g = g(problem)
        if not isinstance(g, Guarantee):
            raise TypeError(f"guarantee of solver {self.name!r} must resolve to a Guarantee, got {g!r}")
        return SolverInfo(self.scalarization, g, self.name, self.capabilities)


def implements(scalarization: type, guarantee: GuaranteeSpec = EXACT, name: Optional[str] = None,
               **capabilities):
    """Register a method of a problem class as solver for a type of scalarization.

    Parameters
    ----------
    scalarization : type
        Subclass of Scalarization.
    guarantee : Guarantee, str or callable
        Guarantee of the solver. A string is the name of an attribute of the problem instance
        that holds a multiplicative factor (interpreted with ``Guarantee.from_factor``); a
        callable receives the problem instance and returns a Guarantee.
    name : str, optional
        Name of the solver; default: the name of the method.
    **capabilities
        Further properties of the solver.

    The decorated method is called as ``method(self, scalarization)`` and returns an Image, None
    (no feasible solution), or a ScalarizationResult.
    """
    if not (isinstance(scalarization, type) and issubclass(scalarization, Scalarization)):
        raise TypeError("implements expects a subclass of Scalarization")

    def decorator(method):
        method._molib_solver = SolverInfo(scalarization, guarantee, name or method.__name__,
                                          dict(capabilities))
        return method

    return decorator


# ---------------------------------------------------------------------------
# solver lookup
# ---------------------------------------------------------------------------
def _registered_solvers(problem) -> List[Tuple[SolverInfo, Callable]]:
    """Registered solvers of the problem class, subclasses first."""
    found, seen = [], set()
    for cls in type(problem).__mro__:
        for attr in vars(cls):
            if attr in seen:
                continue
            seen.add(attr)
            method = getattr(problem, attr, None)
            info = getattr(getattr(method, "__func__", method), "_molib_solver", None)
            if info is not None:
                found.append((info, method))
    return found


def _overrides(problem, method_name: str) -> bool:
    return getattr(type(problem), method_name, None) is not getattr(Problem, method_name)


def _legacy_solvers(problem) -> List[Tuple[SolverInfo, Callable]]:
    """Adapters for the methods weighted_sum and weighted_tchebycheff of the legacy interface."""
    solvers = []
    if _overrides(problem, "weighted_sum"):
        solvers.append((
            SolverInfo(WeightedSum, "solution_quality_weighted_sum", "weighted_sum"),
            lambda s: problem.weighted_sum(s.weights.tolist()),
        ))
    if _overrides(problem, "weighted_tchebycheff"):
        solvers.append((
            SolverInfo(WeightedTchebycheff,
                       lambda p: Guarantee.from_factor(getattr(p, "solution_quality_weighted_tchebycheff", math.inf)),
                       "weighted_tchebycheff"),
            lambda s: problem.weighted_tchebycheff(s.weights.tolist(), s.reference_point.tolist(), s.augmentation),
        ))
    return solvers


def _all_solvers(problem) -> List[Tuple[SolverInfo, Callable]]:
    """Registered solvers, followed by legacy adapters for scalarization types without a
    registered solver."""
    registered = _registered_solvers(problem)
    covered = {info.scalarization for info, _ in registered}
    legacy = [(info, m) for info, m in _legacy_solvers(problem) if info.scalarization not in covered]
    return registered + legacy


def _find_solver(problem, scalarization: Scalarization) -> Optional[Tuple[SolverInfo, Callable]]:
    for info, method in _all_solvers(problem):
        if isinstance(scalarization, info.scalarization):
            return info.resolve(problem), method
    return None


def solver_info(problem, scalarization: Scalarization) -> Optional[SolverInfo]:
    """Solver that ``solve`` would use for the scalarization (with resolved guarantee), or None."""
    found = _find_solver(problem, scalarization)
    return None if found is None else found[0]


def available_solvers(problem) -> List[SolverInfo]:
    """All solvers of the problem (registered ones first), with resolved guarantees."""
    return [info.resolve(problem) for info, _ in _all_solvers(problem)]


# ---------------------------------------------------------------------------
# logging of solves
# ---------------------------------------------------------------------------
@dataclass
class SolveLog:
    """Record of all calls of ``solve`` within a ``record_solves()`` block."""
    results: List[ScalarizationResult] = field(default_factory=list)

    @property
    def nr_calls(self) -> int:
        return len(self.results)

    @property
    def total_runtime(self) -> float:
        return float(sum(r.runtime for r in self.results))

    def calls_by_scalarization(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.results:
            key = type(r.scalarization).__name__
            counts[key] = counts.get(key, 0) + 1
        return counts


_ACTIVE_LOGS: contextvars.ContextVar = contextvars.ContextVar("molib_solve_logs", default=())


@contextlib.contextmanager
def record_solves() -> Iterator[SolveLog]:
    """Record every call of ``solve`` in the block, also within algorithms::

        with record_solves() as log:
            images = dichotomic_search(problem)
        print(log.nr_calls, log.total_runtime)

    Blocks can be nested; each active log records all calls.
    """
    log = SolveLog()
    token = _ACTIVE_LOGS.set(_ACTIVE_LOGS.get() + (log,))
    try:
        yield log
    finally:
        _ACTIVE_LOGS.reset(token)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
def solve(problem, scalarization: Scalarization, require: Optional[Guarantee] = None) -> ScalarizationResult:
    """Solve a scalarization for a problem.

    Parameters
    ----------
    problem : Problem
        The multiobjective optimization problem.
    scalarization : Scalarization
        The scalarization to be solved, interpreted in the sense (min or max) of the problem.
    require : Guarantee, optional
        Guarantee that the solver has to provide; otherwise, GuaranteeNotSatisfiedError is raised
        before the solver is called.

    Returns
    -------
    ScalarizationResult

    Raises
    ------
    UnsupportedScalarizationError
        If the problem provides no solver for the scalarization.
    """
    if not isinstance(scalarization, Scalarization):
        raise TypeError(f"expected a Scalarization, got {type(scalarization).__name__}")
    scalarization.validate(problem.nr_obj)

    found = _find_solver(problem, scalarization)
    if found is None:
        raise UnsupportedScalarizationError(
            f"{type(problem).__name__} provides no solver for {type(scalarization).__name__}")
    info, method = found
    if require is not None and not info.guarantee.satisfies(require):
        raise GuaranteeNotSatisfiedError(
            f"solver {info.name!r} of {type(problem).__name__} guarantees {info.guarantee}, "
            f"but {require} is required")

    start = time.perf_counter()
    output = method(scalarization)
    runtime = time.perf_counter() - start

    result = _to_result(output, problem, scalarization, info, runtime)
    for log in _ACTIVE_LOGS.get():
        log.results.append(result)
    return result


def _to_result(output, problem, scalarization, info: SolverInfo, runtime: float) -> ScalarizationResult:
    if isinstance(output, ScalarizationResult):
        result = output
    elif output is None:
        result = ScalarizationResult(image=None, status="infeasible")
    else:
        image = output if isinstance(output, Image) else Image(output)
        result = ScalarizationResult(image=image, status="solved")

    result.scalarization = scalarization
    result.direction = scalarization.direction(problem.type)
    if not result.solver:
        result.solver = info.name
    if result.guarantee is None:
        result.guarantee = info.guarantee
    if not result.runtime:
        result.runtime = runtime
    if result.image is not None and result.value is None:
        result.value = float(scalarization.value(np.asarray(result.image, dtype=float), problem.type)[0])
    return result