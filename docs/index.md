# molib

**molib** is a Python library of general exact and approximation methods for multiobjective
optimization problems. Its algorithms only access a problem through *scalarizations*, i.e.,
single-objective substitute problems such as the weighted sum or the weighted Tchebycheff
scalarization. Hence, a new problem becomes usable with every algorithm of the library as soon
as it provides solvers for the scalarizations that the algorithm requires.

```python
import numpy as np
from molib.problems.image_set import ImageSet_withLexWS
from molib.algorithms.exact_methods import generic_upper_bound

Y = np.random.default_rng(0).integers(1, 100, size=(60, 3)).astype(float)
problem = ImageSet_withLexWS(Y)            # explicitly given images, minimization
nondominated = generic_upper_bound(problem)
```

The library distinguishes

- **problems** ({class}`~molib.core.problem.Problem`), which provide solvers for scalarizations,
- **scalarizations** ({mod}`molib.scalarizations`), which only hold parameters,
- the central function {func}`~molib.core.solve.solve`, which selects a solver, checks
  guarantees, and returns a {class}`~molib.core.solve.ScalarizationResult`, and
- **algorithms** ({mod}`molib.algorithms`), which only call `solve`.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
getting_started
user_guide/index
auto_examples/index
api/index
```