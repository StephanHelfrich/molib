# Quality indicators

The module {mod}`molib.core.quality_indicators` measures the quality of approximation sets.

The **convex epsilon indicator** of a set of images $Y'$ with respect to a set $Y$ is the smallest
$\alpha \ge 1$ such that $Y'$ is an $\alpha$-convex approximation set, i.e., for every $y \in Y$
there is a convex combination $z$ of images in $Y'$ with $z \le \alpha y$ (minimization). It is
available in two equivalent forms:

- {func}`~molib.core.quality_indicators.parametric_eps_indicator.get_epsilon_convex_indicator`
  with a reference set that contains all extreme supported nondominated images. With
  `method="exact"`, one linear program is solved per reference image; `method="grid"` evaluates
  the weight set on a grid and yields a lower bound.
- {func}`~molib.core.quality_indicators.parametric_eps_indicator.get_epsilon_convex_indicator_with_nondominated_set`
  with a reference set that contains all nondominated images; one linear program per reference
  image.

```python
from molib.core.quality_indicators import get_epsilon_convex_indicator

alpha = get_epsilon_convex_indicator(approximation, extreme_supported, "max", method="exact")
```

The **epsilon indicator**
{func}`~molib.core.quality_indicators.eps_indicator.get_epsilon_indicator` measures the
(non-convex) multiplicative approximation quality with respect to all nondominated images.

All indicators assume positive images. The linear programs are solved with
{func}`scipy.optimize.linprog` (HiGHS).