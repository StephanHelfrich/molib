import math

import pytest

from molib.core.guarantees import EXACT, NO_GUARANTEE, Guarantee


def test_constructors_and_from_factor():
    assert Guarantee.from_factor(1) == EXACT
    assert Guarantee.from_factor(2) == Guarantee.multiplicative(2)
    assert Guarantee.from_factor(math.inf) == NO_GUARANTEE
    assert str(Guarantee.multiplicative(1.5)) == "multiplicative(1.5)"


@pytest.mark.parametrize("kind,factor", [("multiplicative", 0.5), ("additive", -1.0), ("other", 1.0)])
def test_invalid_guarantees(kind, factor):
    with pytest.raises(ValueError):
        Guarantee(kind, factor)


def test_satisfies():
    assert EXACT.satisfies(Guarantee.multiplicative(1.1))
    assert EXACT.satisfies(Guarantee.additive(0.0))
    assert Guarantee.multiplicative(1.5).satisfies(Guarantee.multiplicative(2))
    assert not Guarantee.multiplicative(2).satisfies(Guarantee.multiplicative(1.5))
    assert not Guarantee.multiplicative(1.5).satisfies(Guarantee.additive(10))
    assert not Guarantee.multiplicative(1.5).satisfies(EXACT)
    assert not NO_GUARANTEE.satisfies(Guarantee.multiplicative(100))
    assert NO_GUARANTEE.satisfies(NO_GUARANTEE)