"""Fertility and continuation rate (PRD §7.1, §12.2).

The arithmetic is two divisions. Everything that can go wrong is in what gets
counted: which words are in the denominator, whether the language should have
been dropped at all, and whether the result can say what produced its word
boundaries.
"""

from __future__ import annotations

from typing import Any

import pytest

from glotscope.aggregate import aggregate_words
from glotscope.enums import Segmenter
from glotscope.errors import UnkRateExceededError
from glotscope.fertility import UNK_EXCLUSION_THRESHOLD, fertility
from glotscope.results import FertilityResult


def _fertility(encodings: list[list[int]], **overrides: Any) -> FertilityResult:
    arguments: dict[str, Any] = {
        "language": "eng_Latn",
        "segmenter": Segmenter.WHITESPACE,
        "segmenter_model_version": None,
        "leading_space": True,
        "unk_char_rate": 0.0,
    }
    arguments.update(overrides)
    return fertility(aggregate_words(encodings), **arguments)


def test_fertility_is_tokens_over_words() -> None:
    result = _fertility([[1, 2], [3], [4, 5, 6], [7]])

    assert result.fertility == pytest.approx(7 / 4)
    assert result.p_continued == pytest.approx(2 / 4)


def test_a_word_that_encoded_to_nothing_stays_in_the_denominator() -> None:
    # §12.2: a normalizer can strip a word to empty — soft hyphen, ZWSP, some
    # ZWJ and RTL marks — so the naive invariant "fertility >= 1" fails there.
    # Dropping those words would restore the invariant by changing what was
    # measured, so they are counted and reported instead.
    result = _fertility([[1, 2], [], [3]])

    assert result.n_zero_length_words == 1
    assert result.fertility == pytest.approx(3 / 3)
    assert result.fertility < 2.0


def test_p_continued_counts_words_needing_two_or_more_tokens() -> None:
    result = _fertility([[1], [2], [3, 4], [5, 6, 7]])

    assert result.p_continued == pytest.approx(0.5)


def test_a_language_above_the_unk_threshold_is_dropped_not_reported() -> None:
    # Petrov's convention. The number is not a worse measurement, it is a
    # measurement of the wrong thing: UNK-collapsing tokenizers score *better*
    # as their coverage gets worse.
    with pytest.raises(UnkRateExceededError, match=r"11\.0%"):
        _fertility([[1, 2], [3]], unk_char_rate=0.11)


def test_the_threshold_itself_is_admitted() -> None:
    # "More than 10%" excludes; exactly 10% does not. Worth pinning because an
    # off-by-one here silently changes which languages a leaderboard reports.
    result = _fertility([[1, 2], [3]], unk_char_rate=UNK_EXCLUSION_THRESHOLD)

    assert result.unk_char_rate == pytest.approx(0.10)


@pytest.mark.parametrize("rate", [-0.01, 1.5])
def test_an_unk_rate_that_is_not_a_fraction_is_refused(rate: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        _fertility([[1]], unk_char_rate=rate)


def test_an_empty_word_set_is_refused_rather_than_returning_zero() -> None:
    with pytest.raises(ValueError, match="undefined"):
        _fertility([])


def test_a_model_backed_segmenter_must_record_its_version() -> None:
    # FertilityResult allows a null model version only for WHITESPACE and
    # UD_GOLD. Enforced rather than trusted, because the alternative is a
    # published number whose segmentation cannot be reproduced.
    with pytest.raises(ValueError, match="must be recorded"):
        _fertility([[1]], segmenter=Segmenter.JIEBA, segmenter_model_version=None)


def test_whitespace_may_pin_no_model_version() -> None:
    result = _fertility([[1]], segmenter=Segmenter.WHITESPACE, segmenter_model_version=None)

    assert result.segmenter_model_version is None


def test_the_comparability_key_carries_what_makes_results_incomparable() -> None:
    # Segmenter, model version and leading-space convention. Two fertility
    # numbers differing in any of them are not on the same scale, and §7.1
    # rule 3 requires the comparison API to refuse rather than table them.
    result = _fertility(
        [[1, 2]],
        segmenter=Segmenter.MECAB,
        segmenter_model_version="unidic-lite 1.0.8",
        leading_space=False,
    )

    assert result.comparability_key() == {
        "segmenter": Segmenter.MECAB,
        "segmenter_model_version": "unidic-lite 1.0.8",
        "leading_space": False,
    }
