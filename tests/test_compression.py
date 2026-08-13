"""The compression family (PRD §7.2).

Every quantity here is a ratio of totals, never a mean of per-document ratios.
The two differ whenever documents have unequal length, and the difference is
invisible in any single number — so the tests pick inputs where they disagree
rather than inputs where both happen to give the same answer.
"""

from __future__ import annotations

import pytest

from glotscope.aggregate import aggregate_documents
from glotscope.compression import BYTES, CHARS, compression


def test_cpt_and_bpt_are_ratios_of_totals_not_means_of_ratios() -> None:
    stats = aggregate_documents([[1, 2], [3]], char_lengths=[8, 2], byte_lengths=[10, 2])

    result = compression(stats, unit_lengths=[10, 2], language="eng_Latn")

    # Ratio of totals: 10 chars over 3 tokens. The mean of per-document ratios
    # would be (4.0 + 2.0) / 2 = 3.0, which is the wrong answer this pins.
    assert result.cpt == pytest.approx(10 / 3)
    assert result.bpt == pytest.approx(12 / 3)
    assert result.cpt != pytest.approx(3.0)


def test_ctc_is_the_corpus_token_count() -> None:
    stats = aggregate_documents([[1, 2, 3], [4]], char_lengths=[6, 2], byte_lengths=[6, 2])

    result = compression(stats, unit_lengths=[6, 2], language="eng_Latn")

    assert result.ctc == 4
    assert result.ctc == stats.total_tokens


def test_default_compression_rate_is_numerically_identical_to_bpt() -> None:
    # U1, frozen from TokEval source: the default measurement unit is UTF-8
    # bytes and CR is the ratio of totals, so with no excluded records CR and
    # BPT are the same number. A CR that differs here means the exclusion rule
    # fired when it should not have.
    stats = aggregate_documents([[1, 2], [3, 4, 5]], char_lengths=[4, 9], byte_lengths=[7, 11])

    result = compression(stats, unit_lengths=[7, 11], language="eng_Latn")

    assert result.compression_rate == pytest.approx(result.bpt)
    assert result.compression_rate_unit == BYTES


def test_compression_rate_excludes_empty_tokenizations_and_zero_unit_records() -> None:
    # TokEval skips blank text, zero-unit text and empty tokenizations, then
    # divides the surviving totals. Here the second record has units but no
    # tokens and the third has tokens but no units; only the first survives.
    stats = aggregate_documents(
        [[1, 2], [], [3]],
        char_lengths=[8, 4, 0],
        byte_lengths=[8, 4, 0],
    )

    result = compression(stats, unit_lengths=[8, 4, 0], language="eng_Latn")

    assert result.compression_rate == pytest.approx(4.0)
    # BPT keeps every record, so the two quantities are computed over different
    # denominators: 8/2 against 12/3.
    assert result.bpt == pytest.approx(4.0)
    assert result.ctc == 3


def test_compression_rate_excludes_whitespace_only_records() -> None:
    stats = aggregate_documents([[1, 2], [3]], char_lengths=[4, 16], byte_lengths=[4, 16])

    result = compression(stats, unit_lengths=[4, 16], is_blank=[False, True], language="eng_Latn")

    assert result.compression_rate == pytest.approx(2.0)


def test_excluded_records_can_move_the_rate_away_from_bpt() -> None:
    stats = aggregate_documents(
        [[1, 2], [3, 4, 5, 6]],
        char_lengths=[8, 0],
        byte_lengths=[8, 0],
    )

    result = compression(stats, unit_lengths=[8, 0], language="eng_Latn")

    assert result.compression_rate == pytest.approx(4.0)
    assert result.bpt == pytest.approx(8 / 6)
    assert result.compression_rate != pytest.approx(result.bpt)


def test_compression_rate_can_be_measured_in_characters() -> None:
    stats = aggregate_documents([[1, 2]], char_lengths=[6], byte_lengths=[10])

    result = compression(stats, unit_lengths=[6], unit=CHARS, language="eng_Latn")

    assert result.compression_rate == pytest.approx(3.0)
    assert result.compression_rate_unit == CHARS
    assert result.comparability_key() == {"compression_rate_unit": CHARS}


def test_results_measured_in_different_units_are_not_comparable() -> None:
    stats = aggregate_documents([[1, 2]], char_lengths=[6], byte_lengths=[10])

    in_bytes = compression(stats, unit_lengths=[10], language="eng_Latn")
    in_chars = compression(stats, unit_lengths=[6], unit=CHARS, language="eng_Latn")

    assert in_bytes.comparability_key() != in_chars.comparability_key()


def test_compression_rejects_an_unknown_unit() -> None:
    stats = aggregate_documents([[1]], char_lengths=[1], byte_lengths=[1])

    with pytest.raises(ValueError, match="unit must be"):
        compression(stats, unit_lengths=[1], unit="graphemes", language="eng_Latn")


def test_compression_rejects_a_unit_array_of_the_wrong_length() -> None:
    stats = aggregate_documents([[1], [2]], char_lengths=[1, 1], byte_lengths=[1, 1])

    with pytest.raises(ValueError, match="equal length"):
        compression(stats, unit_lengths=[1], language="eng_Latn")


def test_compression_rejects_a_corpus_with_no_tokens() -> None:
    stats = aggregate_documents([[], []], char_lengths=[0, 0], byte_lengths=[0, 0])

    with pytest.raises(ValueError, match="no tokens"):
        compression(stats, unit_lengths=[0, 0], language="eng_Latn")


def test_compression_refuses_when_every_record_is_excluded() -> None:
    # Every record has tokens but no measured units, so CR has no denominator.
    # Returning 0.0 would publish a compression rate for a corpus that supports
    # none.
    stats = aggregate_documents([[1], [2]], char_lengths=[0, 0], byte_lengths=[0, 0])

    with pytest.raises(ValueError, match="no records"):
        compression(stats, unit_lengths=[0, 0], language="eng_Latn")


def test_compression_rejects_negative_unit_lengths() -> None:
    stats = aggregate_documents([[1]], char_lengths=[1], byte_lengths=[1])

    with pytest.raises(ValueError, match="negative"):
        compression(stats, unit_lengths=[-1], language="eng_Latn")
