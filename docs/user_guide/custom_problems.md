# Implementing a problem

A new problem is a subclass of {class}`~molib.core.problem.Problem`. Solvers for scalarizations
are methods that are registered with the decorator {func}`~molib.core.solve.implements`:

```python
import numpy as np
from molib.core import Guarantee, Image, Problem, implements
from molib.scalarizations import WeightedSum


class MyProblem(Problem):
    def __init__(self, data):
        super().__init__("min", nr_obj=2)
        self.data = data
        self.LB, self.UB = 1, 1000

    @implements(WeightedSum, guarantee=Guarantee.multiplicative(2))
    def _solve_weighted_sum(self, scalarization: WeightedSum):
        weights = scalarization.weights
        ...                                   # problem-specific algorithm
        return Image(objective_vector, solution=solution)
```

A registered method is called as `method(self, scalarization)` and returns

- an {class}`~molib.core.image.Image` (with the solution in `solutions`),
- `None` if the scalarization has no feasible solution, or
- a {class}`~molib.core.solve.ScalarizationResult`, e.g. to report a status, a dual bound, or a
  weaker guarantee for this particular call.

The argument `guarantee` of `implements` is a {class}`~molib.core.guarantees.Guarantee`, the name
of an attribute of the problem that holds a multiplicative factor (e.g.
`"solution_quality_weighted_sum"`), or a function that receives the problem and returns a
guarantee. Further keyword arguments of `implements` are stored as `capabilities` of the solver
and can be queried by algorithms.

## Subclasses

A subclass can override a registered solver by registering a method of the same name. For
example, a subclass with a heuristic for the weighted sum:

```python
class MyProblemHeuristic(MyProblem):
    @implements(WeightedSum, guarantee=Guarantee.none())
    def _solve_weighted_sum(self, scalarization):
        ...
```

## Legacy interface

Problem classes that only implement the methods `weighted_sum(weights)` or
`weighted_tchebycheff(weights, reference_point, augmentation)` of
{class}`~molib.core.problem.Problem` keep working without changes: `solve` uses these methods if
no solver is registered for the scalarization. The guarantee of `weighted_sum` is then taken from
the attribute `solution_quality_weighted_sum` (1 is exact, a finite factor greater than 1 is
multiplicative, infinity is no guarantee). New problem classes should register their solvers
with `implements`.