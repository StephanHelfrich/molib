import numpy as np

from molib.core.dominance import (dominates, nondominated, nondominated_mask,
                                  segment_weakly_dominates_point, strictly_dominates, weakly_dominates)


def test_dominance_relations_min_and_max():
    assert dominates([1, 2], [1, 3])
    assert not dominates([1, 3], [1, 3])
    assert weakly_dominates([1, 3], [1, 3])
    assert strictly_dominates([0, 2], [1, 3])
    assert not strictly_dominates([1, 2], [1, 3])
    assert dominates([1, 3], [1, 2], sense="max")


def test_nondominated_mask_matches_brute_force():
    rng = np.random.default_rng(3)
    Y = rng.integers(0, 20, size=(80, 3)).astype(float)
    mask = nondominated_mask(Y)
    for i, y in enumerate(Y):
        dominated = any(np.all(z <= y) and np.any(z < y) for z in Y)
        assert mask[i] == (not dominated)
    assert len(nondominated(list(Y))) == mask.sum()


def test_segment_dominance():
    assert segment_weakly_dominates_point([0, 2], [2, 0], [1, 1])
    assert not segment_weakly_dominates_point([0, 2], [2, 0], [0.9, 0.9])
    assert segment_weakly_dominates_point([0, 2], [2, 0], [0.9, 0.9], tolerances=0.1)