# Problems

A multiobjective optimization problem is an instance of a subclass of
{class}`~molib.core.problem.Problem`. Every problem has

- `type`: `"min"` or `"max"`; all objectives are minimized or all are maximized,
- `nr_obj`: the number of objectives,
- `LB`, `UB`: lower and upper bounds on every component of every nondominated image (scalars or
  vectors of length `nr_obj`); {meth}`~molib.core.problem.Problem.get_bounds` returns them as
  vectors.

Images are instances of {class}`~molib.core.image.Image`, a read-only NumPy array with the
additional attribute `solutions`.

## Problems shipped with molib

| Class | Sense | Solvers | Guarantee of the weighted sum |
|---|---|---|---|
| {class}`~molib.problems.image_set.ImageSet_withLexWS` | min or max | weighted sum, weighted Tchebycheff, epsilon-constraint (enumeration) | exact |
| {class}`~molib.problems.image_set.ImageSet` | min or max | as above, but a random image within the factor `solution_quality_weighted_sum` for the weighted sum | multiplicative (default 2) |
| {class}`~molib.problems.knapsack.Knapsack_Exact` | max | weighted sum (dynamic programming) | exact |
| {class}`~molib.problems.knapsack.Knapsack_Greedy` | max | weighted sum (greedy) | multiplicative(2) |
| `TravelingSalesman_*` | min | weighted sum (Gurobi, double tree, Christofides, ...) | depends on the class |

The image set problems are mainly intended for experiments and tests: they solve every
scalarization by enumeration, so results can be compared with brute force.

```python
import numpy as np
from molib.problems import knapsack as kp

profits, weights, capacity = kp.get_knapsack_instance(30, 2, seed=0)
problem = kp.Knapsack_Exact(profits, weights, capacity)
```