"""The Tier 2 reference-set fallback chain (PRD §7.9).

Three links, tried in order: ``<unused…>``-style vocabulary entries, then
embedding rows above ``|V|``, then single bytes ``0xF5``-``0xFF`` which are never
valid in UTF-8. Exhausting the chain is a typed refusal, never an empty mean —
Land & Bartolo document tokenizers holding exactly the 243 bytes UTF-8 uses, and
StarCoder2 additionally misses ``0xF1``, so "there is always a spare byte" is
false on real checkpoints.
"""

from __future__ import annotations

import pytest

from glotscope.enums import ReferenceSource, TokenizerFamily
from glotscope.errors import NoReferenceSetError
from glotscope.lint import byte_to_unicode
from glotscope.reference_set import ReferenceSet, resolve_reference_set

_BYTE_CHAR = byte_to_unicode()


def _byte_level_vocab(byte_values: range | tuple[int, ...]) -> dict[str, int]:
    """A byte-level vocabulary holding exactly the given byte values."""
    return {_BYTE_CHAR[value]: index for index, value in enumerate(byte_values)}


def test_unused_token_entries_are_the_first_link() -> None:
    # Arrange
    vocab = {"hello": 0, "<unused0>": 1, "<unused1>": 2, "world": 3}

    # Act
    reference = resolve_reference_set(
        vocab,
        TokenizerFamily.BYTE_LEVEL,
        vocab_size=4,
        n_rows=4,
        checkpoint="acme/model",
        tied=True,
    )

    # Assert
    assert reference.source is ReferenceSource.UNUSED_TOKENS
    assert reference.token_ids == (1, 2)


def test_padding_rows_are_the_second_link() -> None:
    # No unused entries, but the embedding matrix is wider than the vocabulary.
    # Arrange
    vocab = {"hello": 0, "world": 1}

    # Act
    reference = resolve_reference_set(
        vocab,
        TokenizerFamily.BYTE_LEVEL,
        vocab_size=2,
        n_rows=5,
        checkpoint="acme/model",
        tied=True,
    )

    # Assert
    assert reference.source is ReferenceSource.PADDING_ROWS
    assert reference.token_ids == (2, 3, 4)


def test_unused_entries_are_preferred_over_padding_rows() -> None:
    # Both available. The chain is ordered, not a union: rows above |V| are
    # padding by inference, while an <unused…> entry says so in the vocabulary.
    # Arrange
    vocab = {"hello": 0, "<unused0>": 1}

    # Act
    reference = resolve_reference_set(
        vocab,
        TokenizerFamily.BYTE_LEVEL,
        vocab_size=2,
        n_rows=9,
        checkpoint="acme/model",
        tied=True,
    )

    # Assert
    assert reference.source is ReferenceSource.UNUSED_TOKENS
    assert reference.token_ids == (1,)


def test_non_utf8_single_bytes_are_the_third_link() -> None:
    # Arrange — a full byte-level vocabulary, so 0xF5-0xFF are all present.
    vocab = _byte_level_vocab(range(256))

    # Act
    reference = resolve_reference_set(
        vocab,
        TokenizerFamily.BYTE_LEVEL,
        vocab_size=256,
        n_rows=256,
        checkpoint="acme/model",
        tied=True,
    )

    # Assert
    assert reference.source is ReferenceSource.UNUSED_BYTES
    assert reference.token_ids == tuple(range(0xF5, 0x100))


def test_a_byte_fallback_vocabulary_spells_the_same_bytes_differently() -> None:
    # <0xF5> rather than the byte-level unicode surrogate for 0xF5.
    # Arrange
    vocab = {f"<0x{value:02X}>": value for value in range(256)}

    # Act
    reference = resolve_reference_set(
        vocab,
        TokenizerFamily.BYTE_FALLBACK,
        vocab_size=256,
        n_rows=256,
        checkpoint="acme/model",
        tied=True,
    )

    # Assert
    assert reference.source is ReferenceSource.UNUSED_BYTES
    assert reference.token_ids == tuple(range(0xF5, 0x100))


def test_a_tokenizer_holding_only_utf8_valid_bytes_exhausts_the_chain() -> None:
    # The documented real case: exactly the 243 bytes UTF-8 uses, nothing spare.
    # Arrange
    vocab = _byte_level_vocab(tuple(range(0xF5)))

    # Act / Assert
    with pytest.raises(NoReferenceSetError) as excinfo:
        resolve_reference_set(
            vocab,
            TokenizerFamily.BYTE_LEVEL,
            vocab_size=0xF5,
            n_rows=0xF5,
            checkpoint="bigcode/starcoder2",
            tied=True,
        )
    assert excinfo.value.tied is True
    assert "cannot be analysed" in str(excinfo.value)


def test_an_untied_checkpoint_is_told_it_can_still_degrade() -> None:
    # Untied has L2(E_in), which needs no reference set at all. The refusal has
    # to say so, or a caller reads it as "Tier 2 is impossible here".
    # Arrange
    vocab = _byte_level_vocab(tuple(range(0xF5)))

    # Act / Assert
    with pytest.raises(NoReferenceSetError) as excinfo:
        resolve_reference_set(
            vocab,
            TokenizerFamily.BYTE_LEVEL,
            vocab_size=0xF5,
            n_rows=0xF5,
            checkpoint="bigcode/starcoder2",
            tied=False,
        )
    assert "L2(E_in)" in str(excinfo.value)


@pytest.mark.parametrize(
    "token",
    ["<unused0>", "<unused_token7>", "<|reserved_special_token_3|>"],
)
def test_the_unused_spellings_the_roster_actually_uses_are_recognised(token: str) -> None:
    # Arrange
    vocab = {"hello": 0, token: 1}

    # Act
    reference = resolve_reference_set(
        vocab,
        TokenizerFamily.BYTE_LEVEL,
        vocab_size=2,
        n_rows=2,
        checkpoint="acme/model",
        tied=True,
    )

    # Assert
    assert reference.token_ids == (1,)


def test_t5_sentinels_are_not_treated_as_unused() -> None:
    # `<extra_id_N>` reads like a reserved slot and is not one. T5's denoising
    # objective emits those sentinels in every training target, so their E_out
    # rows are among the best-trained in the vocabulary. Averaging them into
    # u_ref measures similarity-to-trained under the name of its opposite, and
    # nothing raises: the manifest warning still reads "unused_tokens".
    # Arrange
    vocab = {"hello": 0, "<extra_id_0>": 1, "<extra_id_1>": 2, "<extra_id_99>": 3}

    # Act / Assert
    with pytest.raises(NoReferenceSetError):
        resolve_reference_set(
            vocab,
            TokenizerFamily.BYTE_LEVEL,
            vocab_size=4,
            n_rows=4,
            checkpoint="google/mt5-base",
            tied=True,
        )


def test_an_empty_reference_set_cannot_be_constructed_at_all() -> None:
    # `resolve_reference_set` raises before it could build one, so this guards
    # the type against a future second caller rather than against that path.
    # u_ref over an empty set is a zero vector that every cosine ranks against
    # happily, which is the shape of failure §7.9 is emphatic about.
    with pytest.raises(ValueError, match="cannot be empty"):
        ReferenceSet((), ReferenceSource.UNUSED_TOKENS)


def test_an_ordinary_angle_bracket_token_is_not_mistaken_for_unused() -> None:
    # <s>, </s> and <pad> are special, not unused. Treating them as the
    # reference set would compute u_ref from trained rows and silently move the
    # yardstick.
    # Arrange
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "hello": 3}

    # Act / Assert
    with pytest.raises(NoReferenceSetError):
        resolve_reference_set(
            vocab,
            TokenizerFamily.BYTE_LEVEL,
            vocab_size=4,
            n_rows=4,
            checkpoint="acme/model",
            tied=True,
        )
