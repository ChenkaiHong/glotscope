"""Morphological alignment, all three measures (PRD §7.7, D11).

The two calibration rows are Poelman, Bauwens & de Lhoneux (arXiv:2511.01380),
and they behave differently:

    word         tokenized        MorphScore   their F1   reproduces?
    gathered     g/a/t/h/e/r/e/d     1.0         0.25      yes, exactly
    arabalari    araba/lari          1.0         0.5       no — 2/3

``gathered`` is a hard gate: 0.25 falls straight out of one gold boundary
against seven predicted ones, and any drift in the definition moves it.

``arabalari`` does **not** reproduce, and it is not tuned to. Against gold
``araba/lar/i`` a single predicted boundary gives TP=1, predicted=1, reference=2,
so F1 is 2/3 under micro-aggregated exact-boundary scoring — and the paper's own
0.5 equals recall, which contradicts its ``gathered`` row where 0.25 is neither
recall (1.0) nor precision (1/7). The discrepancy is recorded in
``docs/divergences.md``; §17 calls tuning a value until it matches misconduct.
"""

from __future__ import annotations

import pytest

from glotscope.enums import MorphologicalType, TypologicalScope
from glotscope.morphology import AlignedWord, boundaries, morphology, pieces_from_offsets
from glotscope.results import MorphologyResult

_GATHERED = AlignedWord(
    morphemes=("gather", "ed"),
    tokens=("g", "a", "t", "h", "e", "r", "e", "d"),
)
_ARABALARI = AlignedWord(morphemes=("araba", "lar", "i"), tokens=("araba", "lari"))


def _scored(word: AlignedWord, morphological_type: MorphologicalType) -> MorphologyResult:
    return morphology(
        [word],
        language="tur_Latn",
        morphological_type=morphological_type,
        frequency_weighted=False,
        include_single_token_words=False,
    )


def test_boundaries_are_the_cumulative_offsets_between_pieces() -> None:
    # The final offset is the end of the word, not a boundary between anything.
    assert boundaries(("araba", "lar", "i")) == frozenset({5, 8})
    assert boundaries(("gathered",)) == frozenset()


def test_gathered_scores_one_quarter_under_full_alignment() -> None:
    # The reproduction gate. One gold boundary at 6, seven predicted, so
    # precision 1/7 and recall 1 give F1 = 0.25 exactly.
    # Arrange / Act
    result = _scored(_GATHERED, MorphologicalType.FUSIONAL)

    # Assert
    assert result.full_alignment is not None
    assert result.full_alignment.f1 == pytest.approx(0.25, abs=1e-12)
    assert result.full_alignment.precision == pytest.approx(1 / 7, abs=1e-12)
    assert result.full_alignment.recall == pytest.approx(1.0, abs=1e-12)


def test_gathered_scores_a_perfect_one_under_morphscore_v1() -> None:
    # Which is the whole point of the row: MorphScore calls a character-level
    # split of `gathered` perfect, because the one boundary it looks at is there.
    result = _scored(_GATHERED, MorphologicalType.FUSIONAL)

    assert result.morphscore_v1 == pytest.approx(1.0, abs=1e-12)


def test_the_turkish_row_gives_two_thirds_and_is_not_tuned_to_the_published_half() -> None:
    # Documented in docs/divergences.md. Tuning until it matched would be the
    # misconduct §17 names.
    result = _scored(_ARABALARI, MorphologicalType.AGGLUTINATIVE)

    assert result.full_alignment is not None
    assert result.full_alignment.f1 == pytest.approx(2 / 3, abs=1e-12)
    assert result.full_alignment.f1 != pytest.approx(0.5, abs=1e-3)


def test_the_turkish_row_is_perfect_under_morphscore_which_is_the_papers_point() -> None:
    # MorphScore looks only at the stem-suffix boundary, which `araba/lari` gets
    # right, so it cannot see the missed `lar/i` boundary at all.
    result = _scored(_ARABALARI, MorphologicalType.AGGLUTINATIVE)

    assert result.morphscore_v1 == pytest.approx(1.0, abs=1e-12)
    assert result.morphscore_v2 is not None
    assert result.morphscore_v2.f1 == pytest.approx(1.0, abs=1e-12)


def test_full_alignment_sees_the_suffix_suffix_boundary_that_morphscore_omits() -> None:
    # The one-sentence case for shipping (c) at all.
    result = _scored(_ARABALARI, MorphologicalType.AGGLUTINATIVE)

    assert result.morphscore_v2 is not None
    assert result.full_alignment is not None
    assert result.full_alignment.f1 < result.morphscore_v2.f1


@pytest.mark.parametrize(
    "morphological_type",
    [MorphologicalType.NON_CONCATENATIVE, MorphologicalType.ISOLATING],
)
def test_out_of_scope_languages_carry_no_numbers_at_all(
    morphological_type: MorphologicalType,
) -> None:
    # §7.7 rule 2, and a deliberate divergence: Arnett et al.'s released tables
    # publish Hebrew and Mandarin anyway, Mandarin at precision 0.98 / recall
    # 1.00, which is a single-token artifact rather than alignment.
    # Arrange / Act
    result = _scored(_GATHERED, morphological_type)

    # Assert
    assert result.scope is TypologicalScope.OUT_OF_SCOPE
    assert result.morphscore_v1 is None
    assert result.morphscore_v2 is None
    assert result.full_alignment is None


def test_a_batch_emptied_by_the_filter_is_refused_not_scored_as_zero() -> None:
    # The empty-batch refusal guards the *input*; `scored` can still be emptied
    # afterwards, when a vocabulary emits every gold word whole and the caller
    # excluded one-token words. `_counts` over nothing returns
    # BoundaryCounts(0, 0, 0), whose precision, recall and F1 all read 0.0 — the
    # "this tokenizer aligns nothing" claim the refusal exists to prevent,
    # published where a reader cannot tell it apart from a measurement.
    whole = AlignedWord(morphemes=("cat", "s"), tokens=("cats",))

    with pytest.raises(ValueError, match="emitted every gold word whole"):
        morphology(
            [whole],
            language="eng_Latn",
            morphological_type=MorphologicalType.FUSIONAL,
            frequency_weighted=False,
            include_single_token_words=False,
        )


def test_a_character_level_tokenizer_earns_perfect_recall_and_poor_precision() -> None:
    # D11's reason for making precision non-optional: recall alone rewards
    # oversegmentation, and this is what that looks like.
    result = _scored(_GATHERED, MorphologicalType.FUSIONAL)

    assert result.full_alignment is not None
    assert result.full_alignment.recall == 1.0
    assert result.full_alignment.precision < 0.2


def test_single_token_words_are_excluded_unless_asked_for() -> None:
    # A word the tokenizer emitted whole predicts no boundaries at all. Counting
    # it scores every gold boundary as a miss and drags recall down for a
    # property of the vocabulary rather than of the alignment.
    # Arrange
    whole = AlignedWord(morphemes=("cat", "s"), tokens=("cats",))

    # Act
    excluded = morphology(
        [whole, _GATHERED],
        language="eng_Latn",
        morphological_type=MorphologicalType.FUSIONAL,
        frequency_weighted=False,
        include_single_token_words=False,
    )
    included = morphology(
        [whole, _GATHERED],
        language="eng_Latn",
        morphological_type=MorphologicalType.FUSIONAL,
        frequency_weighted=False,
        include_single_token_words=True,
    )

    # Assert
    assert excluded.full_alignment is not None
    assert included.full_alignment is not None
    assert included.full_alignment.false_negative == excluded.full_alignment.false_negative + 1


def test_morphscore_v1_always_excludes_single_token_words() -> None:
    # Not a parameter: §7.7(a) defines v1 that way, so honouring the flag here
    # would compute something that is not MorphScore and call it MorphScore.
    # Arrange — one scorable word and one the tokenizer emitted whole.
    words = [_GATHERED, AlignedWord(morphemes=("cat", "s"), tokens=("cats",))]

    # Act
    result = morphology(
        words,
        language="eng_Latn",
        morphological_type=MorphologicalType.FUSIONAL,
        frequency_weighted=False,
        include_single_token_words=True,
    )

    # Assert — 1.0 from `gathered` alone, not 0.5 averaged with the whole word.
    assert result.morphscore_v1 == pytest.approx(1.0, abs=1e-12)


def test_a_word_whose_tokens_do_not_spell_it_is_refused() -> None:
    # Offsets into two different strings are not comparable, and the scores
    # computed from them would look entirely reasonable.
    with pytest.raises(ValueError, match="spell"):
        AlignedWord(morphemes=("gather", "ed"), tokens=("gath", "er"))


def test_the_recorded_parameters_reach_the_comparability_key() -> None:
    # §7.7 rule 4: both change tokenizer rankings and the v2 paper could not
    # choose defaults, so a result that does not carry them is not comparable.
    result = morphology(
        [_GATHERED],
        language="eng_Latn",
        morphological_type=MorphologicalType.FUSIONAL,
        frequency_weighted=True,
        include_single_token_words=True,
    )

    assert result.comparability_key() == {
        "frequency_weighted": True,
        "include_single_token_words": True,
    }


def test_scoring_no_words_at_all_is_refused_rather_than_scored_as_zero() -> None:
    # An F1 of 0.0 over an empty batch reads as "this tokenizer aligns nothing",
    # which is a claim about the tokenizer rather than about the input.
    with pytest.raises(ValueError, match="no words"):
        morphology(
            [],
            language="eng_Latn",
            morphological_type=MorphologicalType.FUSIONAL,
            frequency_weighted=False,
            include_single_token_words=False,
        )


def test_morphscore_is_none_rather_than_zero_when_no_word_qualifies() -> None:
    # Every word came back whole, so v1 has nothing to score. 0.0 would read as
    # "the tokenizer aligned none of them", which is a different claim from
    # "none of them was measurable" — and the second is what happened.
    # Arrange
    whole = [
        AlignedWord(morphemes=("cat", "s"), tokens=("cats",)),
        AlignedWord(morphemes=("dog", "s"), tokens=("dogs",)),
    ]

    # Act
    result = morphology(
        whole,
        language="eng_Latn",
        morphological_type=MorphologicalType.FUSIONAL,
        frequency_weighted=False,
        include_single_token_words=True,
    )

    # Assert
    assert result.morphscore_v1 is None
    assert result.full_alignment is not None
    assert result.full_alignment.false_negative == 2


def test_offsets_become_the_pieces_the_word_is_scored_from() -> None:
    # What `tokenizers` reports for a four-character word split per character.
    assert pieces_from_offsets("cats", [(0, 1), (1, 2), (2, 3), (3, 4)]) == ("c", "a", "t", "s")


def test_the_leading_space_marker_is_shifted_away_rather_than_scored() -> None:
    # Encoding " cats" puts the space in its own token spanning (0, 1). Left in,
    # its zero-length piece would put a boundary at offset 0 — the start of the
    # word, which is not a boundary between anything, and one free false
    # positive per word.
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    pieces = pieces_from_offsets("cats", offsets, shift=1)

    assert pieces == ("c", "a", "t", "s")
    assert boundaries(pieces) == frozenset({1, 2, 3})


def test_tokens_inside_one_character_claim_no_boundary() -> None:
    # Two byte tokens covering one two-byte character both report that
    # character's span. Counting them as pieces would score a boundary inside a
    # character, which is a claim about UTF-8 rather than about morphology.
    offsets = [(0, 1), (0, 1), (1, 2), (1, 2)]

    assert pieces_from_offsets("ab", offsets) == ("a", "b")


def test_offsets_that_do_not_tile_the_word_are_refused() -> None:
    # Each of these would otherwise produce boundaries measured against a string
    # the tokenizer never saw, and the scores would look entirely reasonable.
    assert pieces_from_offsets("cats", []) is None
    assert pieces_from_offsets("cats", [(0, 1), (1, 2)]) is None
    assert pieces_from_offsets("cats", [(0, 3), (0, 1), (3, 4)]) is None
