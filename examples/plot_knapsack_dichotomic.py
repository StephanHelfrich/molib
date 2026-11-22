"""
Extreme supported images of a biobjective knapsack problem
==========================================================

The dichotomic search computes all extreme supported nondominated images of a biobjective
knapsack problem with the exact dynamic programming solver of
:class:`~molib.problems.knapsack.Knapsack_Exact`. The same search with the greedy solver of
:class:`~molib.problems.knapsack.Knapsack_Greedy`, whose weighted sum guarantee is
multiplicative(2), returns other images; the convex epsilon indicator measures how well they
approximate the exact ones.
"""
import matplotlib.pyplot as plt
import numpy as np

from molib.algorithms.exact_methods import dichotomic_search
from molib.core import available_solvers, record_solves
from molib.core.quality_indicators import get_epsilon_convex_indicator
from molib.problems import knapsack as kp

profits, weights, capacity = kp.get_knapsack_instance(40, 2, seed=3)

exact = kp.Knapsack_Exact(profits, weights, capacity)
greedy = kp.Knapsack_Greedy(profits, weights, capacity)
for problem in (exact, greedy):
    print(type(problem).__name__, [(i.scalarization.__name__, str(i.guarantee)) for i in available_solvers(problem)])

# %%
# Run the dichotomic search with both solvers and record the scalarizations.
with record_solves() as log_exact:
    Y_exact = dichotomic_search(exact)
with record_solves() as log_greedy:
    Y_greedy = dichotomic_search(greedy)
print(f"exact:  {len(Y_exact)} images, {log_exact.nr_calls} weighted sums")
print(f"greedy: {len(Y_greedy)} images, {log_greedy.nr_calls} weighted sums")

# %%
# Approximation quality of the greedy images with respect to the extreme supported images.
alpha = get_epsilon_convex_indicator(Y_greedy, Y_exact, "max", method="exact")
print(f"convex epsilon indicator of the greedy images: {alpha:.4f}")

# %%
# Plot both sets of images.
E = np.array(sorted(tuple(np.asarray(y)) for y in Y_exact))
G = np.array(sorted(tuple(np.asarray(y)) for y in Y_greedy))
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(E[:, 0], E[:, 1], "o-", label="exact (extreme supported)")
ax.plot(G[:, 0], G[:, 1], "s", fillstyle="none", label="greedy")
ax.set_xlabel("$f_1$ (maximized)")
ax.set_ylabel("$f_2$ (maximized)")
ax.legend()
fig.tight_layout()