# Getting started

This page computes the nondominated images of a small problem with three exact methods and
explains what they return.

## A problem given by its images

The simplest problem class, {class}`~molib.problems.image_set.ImageSet_withLexWS`, is defined by
an explicit array of images (one row per feasible solution). Its solutions are the row indices.
It solves every scalarization exactly by enumeration and is therefore well suited to try
algorithms and to test them against brute force.

```python
import numpy as np
from molib.problems.image_set import ImageSet_withLexWS

rng = np.random.default_rng(1)
Y = rng.integers(1, 100, size=(80, 2)).astype(float)
problem = ImageSet_withLexWS(Y, type="min")     # all objectives are minimized
```

## Extreme supported nondominated images

The dichotomic search (biobjective problems) and the dual variant of Benson's outer approximation
algorithm (any number of objectives) compute all extreme supported nondominated images with the
weighted sum scalarization:

```python
from molib.algorithms.exact_methods import dichotomic_search, dual_outer_approximation

extreme = dichotomic_search(problem)
same = dual_outer_approximation(problem)
```

Both return a set of {class}`~molib.core.image.Image` objects. An image is a read-only NumPy array
whose attribute `solutions` holds the solutions that were found for it:

```python
for y in sorted(extreme, key=lambda y: y[0]):
    print(np.asarray(y), "solution:", y.solutions)
```

## All nondominated images

The generic upper bound method of Klamroth, Lacour and Vanderpooten (2015) computes all
nondominated images with the augmented weighted Tchebycheff scalarization:

```python
from molib.algorithms.exact_methods import generic_upper_bound

nondominated = generic_upper_bound(problem, LB=Y.min(axis=0), UB=Y.max(axis=0))
```

`LB` and `UB` are componentwise bounds on all nondominated images; if they are omitted, the
attributes `problem.LB` and `problem.UB` are used.

## Counting scalarizations

Every algorithm solves scalarizations through {func}`~molib.core.solve.solve`. The context
manager {func}`~molib.core.solve.record_solves` records all of these calls:

```python
from molib.core import record_solves

with record_solves() as log:
    generic_upper_bound(problem, LB=Y.min(axis=0), UB=Y.max(axis=0))
print(log.nr_calls, log.calls_by_scalarization(), f"{log.total_runtime:.3f} s")
```

The {doc}`user_guide/index` explains problems, scalarizations, guarantees, and how to implement
your own problem.