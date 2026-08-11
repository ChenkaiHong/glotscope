from __future__ import annotations

from fractions import Fraction

import pytest

from glotscope.metrics import gini


def test_gini_hardcoded_ascending_value_and_range() -> None:
    result = gini({str(index): float(cost) for index, cost in enumerate([1, 2, 3, 4, 5])})

    assert result.value == float(Fraction(4, 15))
    assert 0.0 <= result.value <= 1.0
    assert result.cost_unit == "tokens_per_line"
    assert gini({"a": 7.0, "b": 7.0, "c": 7.0}).value == 0.0


@pytest.mark.parametrize(
    ("costs", "message"),
    [
        ({}, "at least one language"),
        ({"a": -1.0}, "finite and non-negative"),
        ({"a": float("inf")}, "finite and non-negative"),
    ],
)
def test_gini_rejects_invalid_costs(costs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gini(costs)
