from __future__ import annotations

import pytest

from glotscope.enums import RenyiNormalizer
from glotscope.metrics import renyi_efficiency, renyi_efficiency_from_counts
from glotscope.results import require_comparable


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


def test_renyi_normalizes_string_enums_and_records_the_nominal_denominator() -> None:
    observed = renyi_efficiency(["a", "a", "b"], alpha=2.5, normalizer="observed")
    nominal = renyi_efficiency(
        ["a", "a", "b"], alpha=2.5, normalizer="nominal", nominal_vocab_size=32
    )

    assert observed.normalizer is RenyiNormalizer.OBSERVED
    assert nominal.normalizer is RenyiNormalizer.NOMINAL
    assert nominal.nominal_vocab_size == 32
    with pytest.raises(Exception, match="nominal_vocab_size"):
        require_comparable(
            nominal,
            renyi_efficiency(
                ["a", "a", "b"], alpha=2.5, normalizer="nominal", nominal_vocab_size=16
            ),
        )


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), float("-inf")])
def test_renyi_refuses_nonfinite_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        renyi_efficiency(["a"], alpha=alpha)


def test_renyi_refuses_zero_count_types() -> None:
    with pytest.raises(ValueError, match="positive"):
        renyi_efficiency_from_counts({"a": 1, "unused": 0}, alpha=2.5)


def test_a_single_type_vocabulary_is_zero_efficiency_not_a_division_by_zero() -> None:
    # Found by mutation: forcing the division broke no test, so nothing exercised
    # the degenerate case at all. log2(1) is 0, so the guard is what stands
    # between a published metric and a ZeroDivisionError — and 0.0 is the right
    # answer, since a one-type vocabulary carries no information to distribute.
    result = renyi_efficiency(["a", "a", "a"], alpha=2.5)

    assert result.value == 0.0
    assert result.entropy_bits == pytest.approx(0.0)


def test_the_nominal_size_is_recorded_only_under_the_nominal_normalizer() -> None:
    # Also from mutation: forcing the nominal branch broke nothing. The field is
    # part of RenyiResult's comparability key, so recording it under the observed
    # normalizer would make two results differ on a parameter that did not apply.
    observed = renyi_efficiency(
        ["a", "b", "b"], alpha=2.5, normalizer=RenyiNormalizer.OBSERVED, nominal_vocab_size=32000
    )
    nominal = renyi_efficiency(
        ["a", "b", "b"], alpha=2.5, normalizer=RenyiNormalizer.NOMINAL, nominal_vocab_size=32000
    )

    assert observed.nominal_vocab_size is None
    assert nominal.nominal_vocab_size == 32000
