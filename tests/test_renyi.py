from __future__ import annotations

import pytest

from glotscope.enums import RenyiNormalizer
from glotscope.metrics import renyi_efficiency


@pytest.mark.reference
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pick @@ed pick @@l @@ed pick @@les", 0.8265064834225245),
        ("pick @@e @@d pick @@l @@e @@d pick @@l @@e @@s", 0.9204840242168807),
    ],
)
def test_renyi_power_2_5_uses_the_stated_formula(text: str, expected: float) -> None:
    result = renyi_efficiency(text.split(), alpha=2.5)

    assert result.value == pytest.approx(expected, abs=1e-9)
    assert result.alpha == 2.5
    assert result.normalizer is RenyiNormalizer.OBSERVED


@pytest.mark.reference
@pytest.mark.parametrize(
    ("text", "documented"),
    [
        ("pick @@ed pick @@l @@ed pick @@les", 0.8031528501359657),
        ("pick @@e @@d pick @@l @@e @@d pick @@l @@e @@s", 0.9105681923824472),
    ],
)
def test_zouhar_documented_values_reproduce_at_power_3_only(text: str, documented: float) -> None:
    result = renyi_efficiency(text.split(), alpha=3.0)

    assert result.value == pytest.approx(documented, abs=1e-9)


def test_renyi_handles_shannon_and_nominal_normalization() -> None:
    shannon = renyi_efficiency(["a", "b"], alpha=1.0)
    nominal = renyi_efficiency(
        ["a", "b"],
        alpha=2.0,
        normalizer=RenyiNormalizer.NOMINAL,
        nominal_vocab_size=4,
    )

    assert shannon.value == 1.0
    assert shannon.entropy_bits == 1.0
    assert nominal.value == 0.5


@pytest.mark.parametrize(
    ("tokens", "alpha", "normalizer", "nominal_vocab_size", "message"),
    [
        ([], 2.0, RenyiNormalizer.OBSERVED, None, "at least one token"),
        (["a"], 0.0, RenyiNormalizer.OBSERVED, None, "alpha must be positive"),
        (["a"], 2.0, RenyiNormalizer.NOMINAL, None, "nominal_vocab_size is required"),
        (["a", "b"], 2.0, RenyiNormalizer.NOMINAL, 1, "cannot be smaller"),
    ],
)
def test_renyi_rejects_invalid_parameters(
    tokens: list[str],
    alpha: float,
    normalizer: RenyiNormalizer,
    nominal_vocab_size: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        renyi_efficiency(
            tokens,
            alpha=alpha,
            normalizer=normalizer,
            nominal_vocab_size=nominal_vocab_size,
        )
