import numpy as np
import pytest

from molib.core.image import Image
from molib.core.local_upper_bound import (LocalUpperBoundSet, update_local_lower_bounds,
                                          update_local_upper_bounds)

from helpers import nondominated_set, random_images

METHODS = ("ra", "re", "re_advanced")


def build(N, method, LB, UB, order_seed=0):
    """Local upper bound set of N, images inserted in random order."""
    rng = np.random.default_rng(order_seed)
    lubs = LocalUpperBoundSet(LB, UB, update_method=method)
    for y in rng.permutation(N):
        lubs.update(Image(y))
    return lubs


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("seed", range(3))
def test_update_methods_agree_and_describe_search_region(p, seed):
    Y = random_images(40, p, seed, high=30)
    N = np.array(sorted(nondominated_set(Y)))
    LB, UB = np.zeros(p), np.full(p, 31.0)

    results = {m: build(N, m, LB, UB, order_seed=seed) for m in METHODS}
    U = {m: {tuple(np.asarray(u).tolist()) for u in results[m]} for m in METHODS}
    assert U["ra"] == U["re"] == U["re_advanced"]

    A = results["ra"].as_array()
    # no inserted image lies in a search zone
    assert not np.any(np.all(N[:, None, :] < A[None, :, :] - 1e-9, axis=2))
    # minimality: no local upper bound is dominated by another one
    for i, u in enumerate(A):
        others = np.delete(A, i, axis=0)
        assert not np.any(np.all(u <= others, axis=1) & np.any(u < others, axis=1))
    # characterization of the search region on sample points
    rng = np.random.default_rng(seed)
    S = rng.uniform(LB, UB, size=(2000, p))
    in_region = np.any(np.all(S[:, None, :] < A[None, :, :], axis=2), axis=1)
    not_dominated = ~np.any(np.all(N[None, :, :] <= S[:, None, :], axis=2), axis=1)
    assert np.array_equal(in_region, not_dominated)


def test_dummy_images_and_search_region():
    lubs = LocalUpperBoundSet([0, 0], [10, 10])
    assert len(lubs) == 1 and np.array_equal(np.asarray(lubs.upper_bounds[0]), [10, 10])
    lubs.update([3, 4])
    assert {tuple(np.asarray(y)) for y in lubs.nondummy_images} == {(3.0, 4.0)}
    assert lubs.in_search_region([2, 9]) and not lubs.in_search_region([5, 5])


def test_update_keeps_image_solutions():
    lubs = LocalUpperBoundSet([0, 0], [10, 10])
    lubs.update(Image([3, 4], solution="x"))
    (image,) = lubs.nondummy_images
    assert image.solutions == ["x"]


def test_invalid_bounds_and_update_method():
    with pytest.raises(ValueError):
        LocalUpperBoundSet([0, 5], [10, 5])
    with pytest.raises(ValueError):
        LocalUpperBoundSet([0, 0], [10, 10], update_method="xyz")


@pytest.mark.parametrize("p", [2, 3])
def test_array_updates_match_class(p):
    Y = random_images(40, p, seed=7, high=30)
    N = np.array(sorted(nondominated_set(Y)))
    LB, UB = np.zeros(p), np.full(p, 31.0)

    lubs = build(N, "re", LB, UB)
    U = LocalUpperBoundSet(LB, UB).as_array()
    for y in N:
        U = update_local_upper_bounds(U, y)
    assert {tuple(u) for u in U} == {tuple(np.asarray(u).tolist()) for u in lubs}

    # local lower bounds of N (maximization) are the negated local upper bounds of -N
    lubs_negated = build(-N, "re", -UB, -LB)
    L = -LocalUpperBoundSet(-UB, -LB).as_array()
    for y in N:
        L = update_local_lower_bounds(L, y)
    assert {tuple(l) for l in L} == {tuple((-np.asarray(u)).tolist()) for u in lubs_negated}