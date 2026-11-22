"""
Nondominated set and search region with the generic upper bound method
======================================================================

The generic upper bound method computes all nondominated images of a biobjective minimization
problem with the augmented weighted Tchebycheff scalarization. With positive approximation
thresholds, search zones that are small enough are not explored; the method then returns a
representation of the nondominated set with fewer scalarizations.
"""
import matplotlib.pyplot as plt
import numpy as np

from molib.algorithms.exact_methods import generic_upper_bound
from molib.core import record_solves
from molib.problems.image_set import ImageSet_withLexWS

# images with conflicting objectives: f_2 decreases with f_1 up to noise
rng = np.random.default_rng(4)
f1 = rng.integers(1, 100, size=300)
f2 = np.clip(110 - f1 + rng.integers(0, 40, size=300), 1, None)
Y = np.column_stack([f1, f2]).astype(float)
problem = ImageSet_withLexWS(Y)
LB, UB = Y.min(axis=0), Y.max(axis=0)

# %%
# Exact computation and a representation with thresholds of 15 in both objectives.
with record_solves() as log_exact:
    exact, lubs = generic_upper_bound(problem, LB=LB, UB=UB, return_upper_bound_set=True)
with record_solves() as log_repr:
    representation = generic_upper_bound(problem, LB=LB, UB=UB, approximation_thresholds=15)
print(f"exact:          {len(exact)} images, {log_exact.nr_calls} scalarizations")
print(f"representation: {len(representation)} images, {log_repr.nr_calls} scalarizations")

# %%
# The local upper bounds of the exact run describe the (empty) remaining search region: every
# point that is not weakly dominated by a computed image lies below some local upper bound.
N = np.array(sorted(tuple(np.asarray(y)) for y in exact))
R = np.array(sorted(tuple(np.asarray(y)) for y in representation))
U = lubs.as_array()
U = U[np.all(U <= UB + 0.5, axis=1)]           # omit the local upper bounds at the box boundary

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(Y[:, 0], Y[:, 1], s=8, color="lightgray", label="all images")
ax.plot(N[:, 0], N[:, 1], "o", label="nondominated images")
ax.plot(R[:, 0], R[:, 1], "x", markersize=9, label="representation")
ax.plot(U[:, 0], U[:, 1], "^", markersize=5, label="local upper bounds")
ax.set_xlabel("$f_1$ (minimized)")
ax.set_ylabel("$f_2$ (minimized)")
ax.legend()
fig.tight_layout()