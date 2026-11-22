# Scalarizations

A scalarization is a single-objective substitute problem. In molib, a scalarization object only
holds its parameters. It knows its value and feasibility in the image space, but not how it is
solved for a concrete problem; this is the task of {func}`~molib.core.solve.solve`.

| Class | Minimization problem | Maximization problem |
|---|---|---|
| {class}`~molib.scalarizations.WeightedSum` | $\min_x \sum_i w_i f_i(x)$ | $\max_x \sum_i w_i f_i(x)$ |
| {class}`~molib.scalarizations.WeightedTchebycheff` | $\min_x \max_i w_i (f_i(x) - r_i) + \rho \sum_i w_i (f_i(x) - r_i)$ | $\min_x \max_i w_i (r_i - f_i(x)) + \rho \sum_i w_i (r_i - f_i(x))$ |
| {class}`~molib.scalarizations.EpsilonConstraint` | $\min_x f_k(x)$ s.t. $f_j(x) \le b_j$, $j \ne k$ | $\max_x f_k(x)$ s.t. $f_j(x) \ge b_j$, $j \ne k$ |

## Interpretation in the sense of the problem

A scalarization is always interpreted in the optimization sense of the problem it is applied to:
`WeightedSum(w)` minimizes for a minimization problem and maximizes for a maximization problem.
Problems are never negated. The method `direction(sense)` returns the optimization direction of
the scalarization itself; the weighted Tchebycheff scalarization is always minimized.

```python
import numpy as np
from molib.scalarizations import WeightedSum, WeightedTchebycheff

Y = np.array([[1.0, 5.0], [2.0, 2.0], [5.0, 1.0]])
s = WeightedTchebycheff([0.5, 0.5], reference_point=[0, 0])
s.value(Y, "min")          # array([2.5, 1. , 2.5])
s.best_index(Y, "min")     # 1, ties are broken lexicographically
```

The reason for not negating problems is that multiplicative approximation guarantees are not
invariant under negation (see {doc}`solving`).

## Implementing a new scalarization

A new scalarization is a subclass of {class}`~molib.scalarizations.Scalarization` that implements
`direction`, `value`, and, if it has additional constraints, `feasible` and `validate`. No
problem or algorithm has to be changed; a problem supports the new scalarization as soon as it
registers a solver for it (see {doc}`custom_problems`).