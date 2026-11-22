# Solving scalarizations and guarantees

## The function `solve`

All algorithms of molib solve scalarizations exclusively through
{func}`~molib.core.solve.solve`:

```python
from molib.core import solve
from molib.scalarizations import WeightedSum

result = solve(problem, WeightedSum([0.5, 0.5]))
result.image          # Image (with solutions) or None
result.value          # value of the scalarization at the image
result.guarantee      # guarantee of the solver
result.runtime        # running time in seconds
```

`solve` selects a solver of the problem for the type of the scalarization, checks a required
guarantee (argument `require`), measures the running time, and returns a
{class}`~molib.core.solve.ScalarizationResult`. If the problem provides no solver for the
scalarization, {class}`~molib.core.solve.UnsupportedScalarizationError` is raised.

{func}`~molib.core.solve.available_solvers` and {func}`~molib.core.solve.solver_info` show which
solvers a problem provides:

```python
from molib.core import available_solvers

for info in available_solvers(problem):
    print(info.scalarization.__name__, info.guarantee, info.name)
```

## Guarantees

A guarantee is a property of a solver for a combination of problem and scalarization. Let OPT be
the optimal value of the scalarization and VAL the value of the returned image, both in the
optimization direction of the scalarization.

| Guarantee | Minimization | Maximization |
|---|---|---|
| `Guarantee.exact()` | VAL = OPT | VAL = OPT |
| `Guarantee.multiplicative(alpha)` | VAL $\le \alpha \cdot$ OPT | VAL $\ge$ OPT $/ \alpha$ |
| `Guarantee.additive(delta)` | VAL $\le$ OPT $+ \delta$ | VAL $\ge$ OPT $- \delta$ |
| `Guarantee.none()` | no statement | no statement |

Multiplicative guarantees require nonnegative values and are defined in the original
optimization direction; they are not invariant under negation of the objectives. For this
reason, molib never negates a problem.

An algorithm that relies on a guarantee passes it to `solve`:

```python
from molib.core import Guarantee

solve(problem, WeightedSum([1, 1]), require=Guarantee.multiplicative(2))
```

If the solver of the problem does not provide the required guarantee,
{class}`~molib.core.solve.GuaranteeNotSatisfiedError` is raised before the solver is called.

## A posteriori quality

A solver may return a {class}`~molib.core.solve.ScalarizationResult` with a dual bound, e.g. a MIP
solver that stopped at a time limit. Then `result.realized_factor` is the multiplicative quality
that was actually achieved (value / dual bound for minimization, dual bound / value for
maximization).

## Recording calls

```python
from molib.core import record_solves

with record_solves() as log:
    images = dichotomic_search(problem)
log.nr_calls, log.total_runtime, log.calls_by_scalarization()
```

`record_solves` records every call of `solve` in the block, including the calls inside
algorithms. Blocks can be nested.