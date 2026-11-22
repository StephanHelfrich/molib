# -*- coding: utf-8 -*-
"""
==========
Guarantees
==========

Quality guarantees of algorithms that solve a scalarization. A guarantee is a property of a
solver for a combination of problem and scalarization, not of the problem or the scalarization
alone. Let OPT denote the optimal value of the scalarization and VAL the value of the returned
image, both in the optimization direction of the scalarization. Then

- ``exact``:                   VAL = OPT,
- ``multiplicative`` (alpha):  VAL <= alpha * OPT (minimization) or VAL >= OPT / alpha (maximization),
                               which requires nonnegative values,
- ``additive`` (delta):        VAL <= OPT + delta (minimization) or VAL >= OPT - delta (maximization),
- ``none``:                    no guarantee (heuristic, time limit, unknown).

Multiplicative guarantees are defined in the original optimization direction. They are not
invariant under negation of the objectives.
"""
from dataclasses import dataclass
import math

__all__ = ["Guarantee", "EXACT", "NO_GUARANTEE"]

_KINDS = ("exact", "multiplicative", "additive", "none")


@dataclass(frozen=True)
class Guarantee:
    """Quality guarantee of a solver for a scalarization.

    Use the constructors ``Guarantee.exact()``, ``Guarantee.multiplicative(alpha)``,
    ``Guarantee.additive(delta)``, ``Guarantee.none()`` or ``Guarantee.from_factor(alpha)``.
    """
    kind: str
    factor: float = 1.0

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        if self.kind == "multiplicative" and not self.factor >= 1.0:
            raise ValueError(f"a multiplicative guarantee requires a factor >= 1, got {self.factor}")
        if self.kind == "additive" and not self.factor >= 0.0:
            raise ValueError(f"an additive guarantee requires a factor >= 0, got {self.factor}")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def exact(cls) -> "Guarantee":
        return cls("exact", 1.0)

    @classmethod
    def multiplicative(cls, alpha: float) -> "Guarantee":
        return cls("multiplicative", float(alpha))

    @classmethod
    def additive(cls, delta: float) -> "Guarantee":
        return cls("additive", float(delta))

    @classmethod
    def none(cls) -> "Guarantee":
        return cls("none", math.inf)

    @classmethod
    def from_factor(cls, alpha: float) -> "Guarantee":
        """Guarantee for a multiplicative factor as in ``Problem.solution_quality_weighted_sum``:
        1 is exact, a finite factor > 1 is multiplicative, infinity is no guarantee."""
        alpha = float(alpha)
        if alpha == 1.0:
            return cls.exact()
        if math.isinf(alpha):
            return cls.none()
        return cls.multiplicative(alpha)

    # ------------------------------------------------------------------ queries
    @property
    def is_exact(self) -> bool:
        return self.kind == "exact"

    def satisfies(self, required: "Guarantee") -> bool:
        """True iff a solver with this guarantee meets the requirement ``required``.

        An exact guarantee satisfies every requirement. A multiplicative (additive) guarantee
        satisfies a multiplicative (additive) requirement with a factor at least as large.
        Every guarantee satisfies the requirement ``none``.
        """
        if required.kind == "none" or self.kind == "exact":
            return True
        if self.kind != required.kind:
            return False
        return self.factor <= required.factor

    def __str__(self):
        if self.kind in ("exact", "none"):
            return self.kind
        return f"{self.kind}({self.factor:g})"


EXACT = Guarantee.exact()
NO_GUARANTEE = Guarantee.none()