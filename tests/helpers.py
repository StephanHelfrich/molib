"""Helper functions for the tests: random instances and brute-force reference solutions."""
import numpy as np

from molib.core.dominance import nondominated_mask


def random_images(n, p, seed, low=1, high=100):
    """n random integer images in [low, high)^p."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=(n, p)).astype(float)


def nondominated_set(Y, sense="min"):
    """Nondominated rows of Y as a set of tuples (brute force)."""
    return {tuple(y) for y in Y[nondominated_mask(Y, sense)]}


def as_tuples(images):
    """Set of tuples, e.g. to compare a set of Image objects with nondominated_set."""
    return {tuple(np.asarray(y, dtype=float).tolist()) for y in images}