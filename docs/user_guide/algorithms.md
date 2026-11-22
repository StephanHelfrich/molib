# Algorithms

| Function | Module | Scalarization | Result |
|---|---|---|---|
| {func}`~molib.algorithms.exact_methods.dichotomic_search.dichotomic_search` | exact methods | weighted sum (exact) | all extreme supported nondominated images (two objectives) |
| {func}`~molib.algorithms.exact_methods.outer_approximation.dual_outer_approximation` | exact methods | weighted sum (exact) | all extreme supported nondominated images |
| {func}`~molib.algorithms.exact_methods.generic_upper_bound.generic_upper_bound` | exact methods | weighted Tchebycheff (exact) | all nondominated images; a representation with `approximation_thresholds` |
| {func}`~molib.algorithms.approximations.approximate_convex_pareto_set.approximate_dichotomy` | approximations | weighted sum ($\alpha$-approximate) | $(1+\varepsilon)\alpha$-convex approximation set (two objectives) |
| {func}`~molib.algorithms.approximations.approximate_convex_pareto_set.GridFPTAA` | approximations | weighted sum ($\alpha$-approximate) | $(1+\varepsilon)\alpha$-convex approximation set |
| {func}`~molib.algorithms.approximations.approximate_convex_pareto_set.FPTOAA` | approximations | weighted sum ($\alpha$-approximate) | $(1+\varepsilon)\alpha$-convex approximation set |
| {func}`~molib.algorithms.approximations.approximate_pareto_set.Papadimitriou_and_Yannakakis` | approximations | gap problem | $(1+\varepsilon)$-approximation set |

The exact methods solve their scalarizations with {func}`~molib.core.solve.solve`. The
approximation algorithms still call the method `weighted_sum` of the problem directly and read
the factor $\alpha$ from `solution_quality_weighted_sum`.

## Exact methods

The **dichotomic search** of Aneja and Nair (1979) and the **dual outer approximation** of
Ehrgott, Löhne and Shao (2012) and Bökler and Mutzel (2015) assume that the weighted sum solver
returns nondominated images also for weights with a zero entry, e.g. by lexicographic
tie-breaking. Otherwise, the images found for such weights may be weakly nondominated only.

The **generic upper bound method** of Klamroth, Lacour and Vanderpooten (2015) explores the
search zones of local upper bounds with the augmented weighted Tchebycheff scalarization. It
supports minimization and maximization, approximation thresholds, initial images (e.g. from a
weighted sum sweep), and a time limit. With `return_upper_bound_set=True`, it also returns the
final {class}`~molib.core.local_upper_bound.LocalUpperBoundSet`, whose non-verified local upper
bounds describe the region that has not been explored.

The bounds `LB` and `UB` of the generic upper bound method must contain all nondominated images.
For three or more objectives, the componentwise maximum of the lexicographically optimal images
is in general not an upper bound on the nadir point.