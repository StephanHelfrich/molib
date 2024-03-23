Collection of instances used in

- S. Helfrich, A. Herzel, S. Ruzika, and C. Thielen. "Efficiently Constructing Convex Approximation Sets in Multiobjective Optimization Problems" (in preparation)

## Knapsack Instances (Knapsack)

We followed [[1]](#1) and constructed two types of instances.

The first type are uniform instances for which the weight and the $i$th costs of each item are independently and uniformly sampled as integers in the interval $[0,1000]$, and the capacity is set to half of the total weight of all items rounded up to the nearest integer.

The second type of instances are conflicting instances for which the weight of each item is independently and uniformly sampled as an integer in the interval $[0,1000]$, the capacity is set to half of the total weight of all items rounded up to the nearest integer, and the costs of each item are sampled to be negatively correlated to each other. More precisely, for each item $e$, $f_1(e)$ is an integer uniformly generated in $[0,1000]$, $f_2(e)$ is an integer uniformly distributed in $[1000 - f_1(e)]$, and $f_3(e)$ is an integer uniformly generated in $[\max \{ 900 - f_1(e) - f_2(e), 0  \}, \min \{ 1100 - f_1(e) - f_2(e), 1000 -  f_1(e)\}]$.

For each $n \in \{10,20,\ldots,250\}$ and each type, five $3$-objective knapsack instances are generated.

The files are formatted as follows:

```
d - integer specifying the number of objectives
n - integer specifying the number of items
C - integer specifying the capacity
the next d lines specify the costs of the items as a list of size n, from profit function 1 to d
the last line specifies the weights of the items as a list of size n
```

## Symmetric Metric Traveling Salesman Instances (TSP)

We used the portgen of the DIMACS TSP instance generator (http://archive.dimacs.rutgers.edu/Challenges/TSP/) to obtain, for each $i = 1,2,3$, integer coordinates of cities on a $1000 \times 1000$ square, on the basis of which the $i$th cost between each two cities is chosen to be the Euclidean distance of their $i$th coordinates.

For each $n \in \{10,20,\ldots,100\}$, five $3$-objective symmetric metric traveling salesman instances are generated.

The files are formatted as follows:

```
Each line contains d+2 values:

city_i city_j c_1(i,j) ... c_d(i,j)
```

## References

<a id="1">[1]</a>
C. Bazgan, H. Hugot & D. Vanderpooten (2009).
Implementing an Efficient {FPTAS} for the 0-1 Multi-objective Knapsack Problem.
European Journal of Operational Research, 198(1), pp. 47-56.

<a id="2">[2]</a>
M. Cornu, T. Cazenave, and D. Vanderpooten (2017).
Perturbed decomposition algorithm applied to the multi-objective traveling salesman problem.
Computers & Operations Research 79, pp. 314–330.

<a id="3">[3]</a>
K. Florios and G. Mavrotas (2014).
Generation of the exact pareto set in multi-objective traveling salesman and set covering problems.
Applied Mathematics and Computation 237, pp. 1–19.

<a id="4">[4]</a>
T. Lust and J. Teghem (2009).
Two-phase pareto local search for the biobjective traveling salesman problem.
Journal of Heuristics 16.3, pp. 475–510.

<a id="5">[5]</a>
L. Paquete and T. Stützle (2010).
On the performance of local search for the biobjective traveling salesman problem.
Advances in Multi-Objective Nature Inspired Computing. pp. 143–165.
