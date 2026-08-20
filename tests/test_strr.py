"""Single-token retention rate (PRD §7.6).

The measure is only meaningful as a pair. The leading-space convention can move
STRR by tens of points and the source paper does not specify which it used, so
every test here checks that both conventions survive to the caller.
"""

from __future__ import annotations

import pytest

from glotscope.aggregate import aggregate_words
from glotscope.strr import strr


def test_strr_is_the_share_of_words_that_survive_as_one_token() -> None:
    bare = aggregate_words([[1], [2], [3, 4]])
    leading_space = aggregate_words([[1], [2, 5], [3, 4]])

    result = strr(bare, leading_space, language="eng_Latn", lowercased=False)

    assert result.bare == pytest.approx(200 / 3)
    assert result.leading_space == pytest.approx(100 / 3)
    assert result.n_words == 3
    assert result.language == "eng_Latn"


def test_strr_is_a_percentage_because_the_formula_says_so() -> None:
    # §7.6 is `(1/n) sum 1(|T(w)| = 1) x 100`, and the scale is part of the
    # formula rather than presentation. This returned a fraction while
    # Tier1Report.to_dict published the number unchanged, so the same §9 field
    # could carry 0.25 or 25.0 depending on which path built the pair — and a
    # reader checking it against the source paper's table would be two orders of
    # magnitude out with nothing to flag it.
    one_of_four = aggregate_words([[1], [2, 3], [4, 5], [6, 7]])

    result = strr(one_of_four, one_of_four, language="eng_Latn", lowercased=False)

    assert result.bare == pytest.approx(25.0)


def test_both_conventions_are_reported_and_can_disagree_sharply() -> None:
    # The exact failure §7.6 exists to prevent: one convention retains every
    # word, the other none. A single unqualified STRR would be either 1.0 or
    # 0.0 depending on a choice nobody recorded.
    bare = aggregate_words([[1], [2], [3]])
    leading_space = aggregate_words([[1, 9], [2, 9], [3, 9]])

    result = strr(bare, leading_space, language="deu_Latn", lowercased=True)

    assert result.bare == 100.0
    assert result.leading_space == 0.0
    assert result.lowercased is True


def test_casing_is_recorded_on_the_result() -> None:
    words = aggregate_words([[1], [2, 3]])

    lowered = strr(words, words, language="tur_Latn", lowercased=True)
    as_written = strr(words, words, language="tur_Latn", lowercased=False)

    assert lowered.lowercased is True
    assert as_written.lowercased is False
    assert lowered.comparability_key() != as_written.comparability_key()


def test_zero_length_words_stay_in_the_denominator() -> None:
    # A word that encodes to nothing is not a retained word, and dropping it
    # would inflate the rate. §12.2 requires it counted, so it dilutes.
    bare = aggregate_words([[1], []])
    leading_space = aggregate_words([[1], []])

    result = strr(bare, leading_space, language="tha_Thai", lowercased=False)

    assert result.bare == pytest.approx(50.0)
    assert result.n_words == 2


def test_strr_refuses_two_word_lists_of_different_sizes() -> None:
    # The two conventions must be the same word list tokenized twice. Different
    # sizes mean they are not the same list, and the pair would be meaningless.
    bare = aggregate_words([[1], [2]])
    leading_space = aggregate_words([[1]])

    with pytest.raises(ValueError, match="same word list"):
        strr(bare, leading_space, language="eng_Latn", lowercased=False)


def test_strr_refuses_an_empty_word_list() -> None:
    empty = aggregate_words([])

    with pytest.raises(ValueError, match="at least one word"):
        strr(empty, empty, language="eng_Latn", lowercased=False)
