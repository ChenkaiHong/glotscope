"""The four aggregation folds (PRD §13, D3).

These are the v1↔v2 FFI boundary: ints in, frozen struct out, one call per
batch. The tests assert that shape as much as the arithmetic, because a fold
that silently drops a record corrupts every ratio downstream and nothing later
in the pipeline can detect it.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from glotscope.aggregate import (
    aggregate_documents,
    aggregate_words,
    align_boundaries,
    attribute_scripts,
)

_ENCODINGS = st.lists(st.lists(st.integers(min_value=0, max_value=50), max_size=6), max_size=8)


# -- aggregate_documents ----------------------------------------------------


def test_document_totals_are_sums_over_the_batch() -> None:
    stats = aggregate_documents(
        [[1, 2, 3], [4, 4], [5]],
        char_lengths=[10, 6, 2],
        byte_lengths=[14, 6, 5],
    )

    assert stats.n_documents == 3
    assert stats.total_tokens == 6
    assert stats.total_chars == 18
    assert stats.total_bytes == 25
    assert stats.per_document_tokens == (3, 2, 1)


def test_type_counts_are_a_frequency_distribution_over_token_ids() -> None:
    stats = aggregate_documents([[7, 7, 9], [7]], char_lengths=[3, 1], byte_lengths=[3, 1])

    assert dict(stats.type_counts) == {7: 3, 9: 1}
    assert sum(stats.type_counts.values()) == stats.total_tokens


def test_type_counts_are_read_only() -> None:
    stats = aggregate_documents([[1]], char_lengths=[1], byte_lengths=[1])

    # Renyi efficiency derives p_Delta from this mapping; a caller mutating it
    # would silently change a published number.
    with pytest.raises(TypeError):
        stats.type_counts[2] = 1  # type: ignore[index]


def test_empty_documents_are_counted_rather_than_dropped() -> None:
    # Normalization can strip a document to nothing (U+00AD, U+200B, some ZWJ
    # and RTL marks). §12.2 requires these be reported, not silently discarded:
    # dropping them would shorten one language's line count and break the
    # ratio-of-means identity parity depends on.
    stats = aggregate_documents([[1, 2], [], [3]], char_lengths=[4, 0, 2], byte_lengths=[4, 0, 2])

    assert stats.n_documents == 3
    assert stats.n_empty_documents == 1
    assert stats.per_document_tokens == (2, 0, 1)


def test_document_fold_rejects_unequal_input_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        aggregate_documents([[1], [2]], char_lengths=[1], byte_lengths=[1, 1])


def test_document_fold_rejects_negative_lengths() -> None:
    with pytest.raises(ValueError, match="negative"):
        aggregate_documents([[1]], char_lengths=[-1], byte_lengths=[1])


def test_empty_batch_yields_zeroed_stats() -> None:
    stats = aggregate_documents([], char_lengths=[], byte_lengths=[])

    assert stats.n_documents == 0
    assert stats.total_tokens == 0
    assert stats.per_document_tokens == ()


@pytest.mark.property
@given(encodings=_ENCODINGS)
def test_total_tokens_always_equals_the_sum_of_per_document_tokens(
    encodings: list[list[int]],
) -> None:
    lengths = [len(encoding) for encoding in encodings]

    stats = aggregate_documents(encodings, char_lengths=lengths, byte_lengths=lengths)

    assert stats.total_tokens == sum(stats.per_document_tokens)
    assert len(stats.per_document_tokens) == stats.n_documents
    assert sum(stats.type_counts.values()) == stats.total_tokens


# -- aggregate_words --------------------------------------------------------


def test_word_counts_split_by_token_count() -> None:
    stats = aggregate_words([[1], [2, 3], [], [4, 5, 6], [7]])

    assert stats.n_words == 5
    assert stats.total_tokens == 7
    assert stats.n_single_token == 2
    assert stats.n_continued == 2
    assert stats.n_zero_length == 1


def test_zero_length_words_are_counted_not_dropped() -> None:
    # "fertility >= 1.0" is false in their presence, so §12.2 requires them
    # counted separately rather than removed from the denominator.
    stats = aggregate_words([[], []])

    assert stats.n_words == 2
    assert stats.n_zero_length == 2
    assert stats.total_tokens == 0


@pytest.mark.property
@given(encodings=_ENCODINGS)
def test_word_classes_partition_the_word_list(encodings: list[list[int]]) -> None:
    stats = aggregate_words(encodings)

    assert stats.n_zero_length + stats.n_single_token + stats.n_continued == stats.n_words


# -- attribute_scripts ------------------------------------------------------


def test_scripts_are_counted_per_script_code() -> None:
    counts = attribute_scripts([10, 11, 12, 13], script_ids=[1, 1, 2, 3])

    assert dict(counts) == {1: 2, 2: 1, 3: 1}


def test_script_attribution_rejects_unequal_input_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        attribute_scripts([1, 2], script_ids=[1])


def test_script_counts_are_read_only() -> None:
    counts = attribute_scripts([1], script_ids=[9])

    with pytest.raises(TypeError):
        counts[0] = 1  # type: ignore[index]


def test_script_attribution_of_an_empty_vocabulary_is_empty() -> None:
    assert dict(attribute_scripts([], script_ids=[])) == {}


# -- align_boundaries -------------------------------------------------------


@pytest.mark.reference
def test_gathered_reproduces_boundary_f1_of_one_quarter() -> None:
    # PRD §7.7(c): "gathered" tokenized g/a/t/h/e/r/e/d against gold gather|ed.
    # Seven predicted boundaries, one of them correct, one gold boundary.
    # MorphScore scores this 1.0; full alignment scores 0.25, and that gap is
    # the whole reason the third measure exists.
    counts = align_boundaries([[1, 2, 3, 4, 5, 6, 7]], gold=[[6]])

    assert counts.true_positive == 1
    assert counts.false_positive == 6
    assert counts.false_negative == 0
    assert counts.precision == pytest.approx(1 / 7)
    assert counts.recall == 1.0
    assert counts.f1 == pytest.approx(0.25)


def test_turkish_derived_value_is_two_thirds_not_the_published_half() -> None:
    # arabaları tokenized araba|ları against gold araba|lar|ı. One true  # noqa: RUF003
    # boundary, one predicted, two gold -> F1 2/3. The source table's 0.5 is an
    # upstream inconsistency; see docs/divergences.md. Tested as the derived
    # value, never tuned toward the published one.
    counts = align_boundaries([[5]], gold=[[5, 8]])

    assert counts.f1 == pytest.approx(2 / 3)


def test_boundary_counts_are_micro_aggregated_across_the_batch() -> None:
    counts = align_boundaries([[1, 2], [3]], gold=[[1], [3, 4]])

    assert counts.true_positive == 2
    assert counts.false_positive == 1
    assert counts.false_negative == 1


def test_duplicate_boundaries_within_a_word_are_counted_once() -> None:
    counts = align_boundaries([[2, 2, 2]], gold=[[2]])

    assert counts.true_positive == 1
    assert counts.false_positive == 0


def test_a_word_with_no_predicted_boundaries_scores_zero_precision() -> None:
    counts = align_boundaries([[]], gold=[[3]])

    assert counts.true_positive == 0
    assert counts.precision == 0.0
    assert counts.recall == 0.0
    assert counts.f1 == 0.0


def test_alignment_rejects_unequal_input_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        align_boundaries([[1], [2]], gold=[[1]])


@pytest.mark.property
@given(predicted=st.lists(st.sets(st.integers(min_value=1, max_value=12)), max_size=6))
def test_identical_segmentation_scores_perfectly(predicted: list[set[int]]) -> None:
    # Only meaningful when at least one boundary exists: with no boundaries at
    # all there is nothing to find, and 0.0 is the documented convention.
    segmentation = [sorted(word) for word in predicted]

    counts = align_boundaries(segmentation, gold=segmentation)

    assert counts.false_positive == 0
    assert counts.false_negative == 0
    if counts.true_positive:
        assert counts.f1 == 1.0
