# -*- coding: utf-8 -*-
r"""
=====================================
Local upper bounds and search regions
=====================================

Implementation of local upper bound sets for multiobjective minimization problems, cf.

K. Klamroth, R. Lacour, D. Vanderpooten. "On the representation of the search region in
multi-objective optimization." European Journal of Operational Research 245(3):767-778, 2015.

Given a finite set N of mutually nondominated images inside the box [LB, UB), the search region
S(N) is the set of points of the box that are not weakly dominated by any image of N. It is
described by a finite set U(N) of local upper bounds via

    S(N) = \bigcup_{u \in U(N)} \{ y : y < u \}.

Three update procedures are available:

- ``'ra'``: redundancy avoidance via defining sets (Algorithm 5 in Klamroth et al.),
- ``'re'``: redundancy elimination (filter projections against all candidates),
- ``'re_advanced'``: redundancy elimination with advanced filtering (Algorithm 3).

All three produce the same set U(N). In addition, array-based update functions for local
upper bounds and (for maximization) local lower bounds are provided.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .image import Image
from .dominance import TOL, leq, leqq, strict_leq

__all__ = [
    "LocalUpperBound",
    "LocalUpperBoundSet",
    "update_local_upper_bounds",
    "update_local_lower_bounds",
]


class LocalUpperBound(np.ndarray):
    """Local upper bound u, stored as read-only ndarray.

    Attributes
    ----------
    images : list of Image
        Images that lie on the boundary of the search zone {y : y < u} (defining images).
    verified : bool
        Flag for algorithms: True if the search zone of u is known to contain no further
        nondominated image (or is not to be explored further).
    """

    def __new__(cls, input_array, images=None, verified: bool = False):
        obj = np.asarray(input_array, dtype=float).view(cls)
        obj.flags.writeable = False

        if images:
            coords = np.asarray(input_array, dtype=float)
            for image_nr, image in enumerate(images):
                if not np.any(np.asarray(image, dtype=float) >= coords - TOL):
                    raise ValueError(f"Image {image_nr} lies in the search zone of the upper bound")
            obj.images = list(images)
        else:
            obj.images = []

        obj.verified = verified
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.images = getattr(obj, "images", [])
        self.verified = getattr(obj, "verified", False)

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        return bool(np.ndarray.__eq__(self, other).all())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (f"{np.asarray(self).tolist()} images={[np.asarray(y).tolist() for y in self.images]} "
                f"measure={self.measure}")

    def set_verified(self):
        self.verified = True

    @property
    def widths(self) -> np.ndarray:
        """u minus the componentwise minimum of its defining images (edge lengths of its box)."""
        if not self.images:
            return np.zeros(self.shape[0])
        pointwise_min = np.min(np.asarray([np.asarray(y, dtype=float) for y in self.images]), axis=0)
        return np.asarray(self, dtype=float) - pointwise_min

    @property
    def measure(self) -> float:
        """Largest edge length of the box below u."""
        if not self.images:
            return 0.0
        return float(np.max(np.abs(self.widths)))

    @classmethod
    def from_images(cls, images: List[Image]) -> "LocalUpperBound":
        """Local upper bound given by the componentwise maximum of ``images``."""
        coords = np.max(np.asarray([np.asarray(y, dtype=float) for y in images]), axis=0)
        return cls(coords, images=images)


UpdateResult = Tuple[Set[LocalUpperBound], Set[LocalUpperBound]]


class LocalUpperBoundSet:
    """Set of local upper bounds U(N) of a set N of images, cf. Klamroth et al. (2015).

    Parameters
    ----------
    LB : sequence of float
        Componentwise lower bounds on all (nondominated) images.
    UB : sequence of float
        Componentwise upper bounds. Images z with z_i >= UB_i for some i never lie in a search
        zone, so UB must be strictly greater than every image that should refine the set.
    update_method : str
        ``'ra'`` (default), ``'re'`` or ``'re_advanced'``.
    """

    def __init__(self, LB: Sequence[float], UB: Sequence[float], update_method: str = "ra") -> None:
        if update_method not in ("ra", "re", "re_advanced"):
            raise ValueError(f"Invalid update method {update_method!r}")

        self.LB = np.asarray(LB, dtype=float)
        self.UB = np.asarray(UB, dtype=float)
        if self.LB.shape != self.UB.shape or self.LB.ndim != 1:
            raise ValueError("LB and UB must be vectors of the same length")
        if np.any(self.LB >= self.UB):
            raise ValueError("LB must be componentwise strictly smaller than UB")

        self.dimension = len(self.LB)
        self.update_method = update_method

        self.images: Set[Image] = set()
        self.upper_bounds: List[LocalUpperBound] = []

        # defining sets Z^j(u), maintained by the redundancy-avoidance update only
        self._Zdef: Dict[LocalUpperBound, List[List[Image]]] = {}

        # dummy images (UB_i in component i, LB elsewhere)
        self.dummy_images: List[Image] = []
        for i in range(self.dimension):
            coords = self.LB.copy()
            coords[i] = self.UB[i]
            self.dummy_images.append(Image(coords))
        self.images.update(self.dummy_images)

        u = LocalUpperBound.from_images(self.dummy_images)
        self._Zdef[u] = [[self.dummy_images[i]] for i in range(self.dimension)]
        self.upper_bounds.append(u)

    @classmethod
    def from_images(cls, images: Iterable, LB=None, UB=None, margin: float = 1.0,
                    update_method: str = "ra") -> "LocalUpperBoundSet":
        """Create the local upper bound set of ``images``.

        If LB is not given, it is set to the componentwise minimum of the images minus ``margin``.
        If UB is not given, it is set to the componentwise maximum of the images plus ``margin``.
        """
        images = [y if isinstance(y, Image) else Image(y) for y in images]
        Y = np.array([np.asarray(y, dtype=float) for y in images])
        if LB is None:
            LB = Y.min(axis=0) - margin
        if UB is None:
            UB = Y.max(axis=0) + margin
        lubs = cls(LB, UB, update_method=update_method)
        for y in images:
            lubs.update(y)
        return lubs

    def __str__(self):
        return f"{len(self.upper_bounds)} local upper bound(s). Update method: {self.update_method}."

    def __len__(self):
        return len(self.upper_bounds)

    def __iter__(self):
        return iter(self.upper_bounds)

    @property
    def nondummy_images(self) -> Set[Image]:
        """Images inserted by the user (without the dummy images)."""
        return self.images - set(self.dummy_images)

    def as_array(self) -> np.ndarray:
        return np.array([np.asarray(u, dtype=float) for u in self.upper_bounds])

    def nr_nonverified_upper_bounds(self) -> int:
        return sum(1 for u in self.upper_bounds if not u.verified)

    def get_non_verified_upper_bound(self, largest_first: bool = False) -> Optional[LocalUpperBound]:
        """Non-verified upper bound with the smallest (or largest) measure, or None."""
        non_verified = [u for u in self.upper_bounds if not u.verified]
        if not non_verified:
            return None
        key = lambda u: u.measure
        return max(non_verified, key=key) if largest_first else min(non_verified, key=key)

    def in_search_region(self, y) -> bool:
        """True iff y lies in the search region, i.e., y < u for some local upper bound u."""
        return any(strict_leq(y, u) for u in self.upper_bounds)

    def update(self, z) -> UpdateResult:
        """Insert image z. Returns (removed upper bounds, added upper bounds)."""
        if not isinstance(z, Image):
            z = Image(z)
        if self.update_method == "ra":
            return self._update_redundancy_avoidance(z)
        if self.update_method == "re_advanced":
            return self._update_redundancy_elimination_advanced_filtering(z)
        return self._update_redundancy_elimination(z)

    # ---------------- helpers ----------------
    def _search_zones_containing(self, z: Image) -> List[LocalUpperBound]:
        return [u for u in self.upper_bounds if strict_leq(z, u)]

    @staticmethod
    def _project(u: LocalUpperBound, z: Image, j: int) -> LocalUpperBound:
        """Projection (z_j, u_-j); inherits the images of u that remain below it."""
        coords = np.array(u, dtype=float)
        coords[j] = z[j]
        p = LocalUpperBound(coords)
        p.images = [y for y in u.images if leqq(y, p)]
        return p

    @staticmethod
    def _unique(candidates: Iterable[LocalUpperBound]) -> Set[LocalUpperBound]:
        """Merge upper bounds with identical coordinates, joining their images."""
        merged: Dict[LocalUpperBound, LocalUpperBound] = {}
        for p in candidates:
            if p in merged:
                q = merged[p]
                q.images = q.images + [y for y in p.images if y not in q.images]
            else:
                merged[p] = p
        return set(merged.values())

    def _finalize(self, z: Image, A: List[LocalUpperBound], P: Set[LocalUpperBound]) -> UpdateResult:
        """(U \\ A) ∪ P, register z, and attach z to every upper bound above it."""
        A_set = set(A)
        self.upper_bounds = [u for u in self.upper_bounds if u not in A_set] + list(P)
        self.images.add(z)
        for u in self.upper_bounds:
            if leqq(z, u) and z not in u.images:
                u.images.append(z)
        return A_set, P

    # ---------------- RE (redundancy elimination) ----------------
    def _update_redundancy_elimination(self, z: Image) -> UpdateResult:
        A = self._search_zones_containing(z)
        if not A:
            return self._finalize(z, A, set())
        A_set = set(A)

        # B = {u in U \ A : z <= u}; only these can dominate a projection of z
        B = [u for u in self.upper_bounds if u not in A_set and leqq(z, u)]
        P = [self._project(u, z, j) for u in A for j in range(self.dimension)]

        PB = P + B
        P_nonred = self._unique(p for p in P if not any(leq(p, q) for q in PB))
        return self._finalize(z, A, P_nonred)

    # ---------------- RE with advanced filtering (Algorithm 3) ----------------
    def _update_redundancy_elimination_advanced_filtering(self, z: Image) -> UpdateResult:
        A = self._search_zones_containing(z)
        if not A:
            return self._finalize(z, A, set())

        P_nonred: List[LocalUpperBound] = []
        for j in range(self.dimension):
            # B_j = {u in U : z_j = u_j and z_-j < u_-j}
            Bj = [u for u in self.upper_bounds
                  if abs(z[j] - u[j]) <= TOL and strict_leq(np.delete(z, j), np.delete(u, j))]
            Pj = [self._project(u, z, j) for u in A]
            PB = Pj + Bj
            P_nonred.extend(p for p in Pj if not any(leq(p, q) for q in PB))

        return self._finalize(z, A, self._unique(P_nonred))

    # ---------------- RA (Algorithm 5, general case) ----------------
    def _update_redundancy_avoidance(self, z: Image) -> UpdateResult:
        p = self.dimension

        # update Z^j(u) if z lies on the boundary of the search zone of u in component j
        for u in self.upper_bounds:
            for j in range(p):
                if abs(z[j] - u[j]) <= TOL and strict_leq(np.delete(z, j), np.delete(u, j)):
                    self._Zdef[u][j].append(z)

        A = self._search_zones_containing(z)
        if not A:
            return self._finalize(z, A, set())

        P: Set[LocalUpperBound] = set()
        for u in A:
            for j in range(p):
                zj_max = max(min(y[j] for y in self._Zdef[u][k]) for k in range(p) if k != j)
                if z[j] > zj_max:
                    uj = self._project(u, z, j)
                    P.add(uj)
                    self._Zdef[uj] = [
                        [z] if k == j else [y for y in self._Zdef[u][k] if y[j] < z[j]]
                        for k in range(p)
                    ]

        for u in A:
            del self._Zdef[u]

        return self._finalize(z, A, P)


# ---------------------------------------------------------------------------
# array-based variants
# ---------------------------------------------------------------------------
def update_local_upper_bounds(U: np.ndarray, z, tol: float = TOL) -> np.ndarray:
    """Update an (m x p) array of local upper bounds with image z (minimization).

    Redundancy elimination with advanced filtering (Klamroth et al. 2015, Algorithm 3),
    operating on plain arrays without defining images. Returns the new array of local
    upper bounds, with duplicate rows removed.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    z = np.asarray(z, dtype=float)
    p = z.shape[0]

    in_A = np.all(z < U - tol, axis=1)
    A = U[in_A]
    if len(A) == 0:
        return U

    new = []
    for j in range(p):
        others = np.arange(p) != j
        # B_j: upper bounds with u_j = z_j and z_-j < u_-j
        Bj = U[(np.abs(U[:, j] - z[j]) <= tol) & np.all(z[others] < U[:, others] - tol, axis=1)]
        Pj = A.copy()
        Pj[:, j] = z[j]
        PB = np.vstack([Pj, Bj]) if len(Bj) else Pj
        for candidate in Pj:
            dominated = np.all(candidate <= PB + tol, axis=1) & np.any(candidate < PB - tol, axis=1)
            if not np.any(dominated):
                new.append(candidate)

    remaining = U[~in_A]
    stacked = np.vstack([remaining] + ([np.array(new)] if new else []))
    return np.unique(stacked, axis=0)


def update_local_lower_bounds(L: np.ndarray, z, tol: float = TOL) -> np.ndarray:
    """Update an (m x p) array of local lower bounds with image z (maximization).

    Equivalent to :func:`update_local_upper_bounds` applied to the negated problem.
    """
    return -update_local_upper_bounds(-np.atleast_2d(np.asarray(L, dtype=float)),
                                      -np.asarray(z, dtype=float), tol)