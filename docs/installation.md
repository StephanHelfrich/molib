# Installation

molib requires Python 3.10 or newer, NumPy and SciPy. Install it from the repository:

```bash
git clone https://github.com/StephanHelfrich/molib
cd molib
pip install -e .
```

Optional dependencies are available as extras:

| Extra | Installs | Required for |
|---|---|---|
| `gurobi` | `gurobipy` | `TravelingSalesman_Gurobi` |
| `test` | `pytest` | running the test suite |
| `docs` | Sphinx and extensions | building this documentation |

```bash
pip install -e ".[test]"   # then run: pytest
pip install -e ".[docs]"   # then run: python -m sphinx -b html docs docs/_build/html
```