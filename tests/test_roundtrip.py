"""Round-trip losslessness (PRD §7.8).

``Roundtrip(T; D)`` is the share of documents satisfying
``decode(encode(d)) == d``. It is the one Tier 1 measure with a correct answer
rather than a distribution: a byte-level tokenizer should score exactly 1.0, and
anything less is a defect in the tokenizer rather than a property of the
language.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.lint import byte_to_unicode
from glotscope.roundtrip import roundtrip_rate


def _byte_level() -> BackendTokenizer:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return backend


def _wordpiece() -> BackendTokenizer:
    backend = BackendTokenizer(
        models.WordPiece(vocab={"[UNK]": 0, "the": 1, "cat": 2}, unk_token="[UNK]")
    )
    backend.decoder = decoders.WordPiece()
    return backend


def test_a_byte_level_tokenizer_round_trips_every_script() -> None:
    documents = ["hello", "こんにちは", "नमस्ते", "مرحبا", "🙂 mixed 日本"]

    assert roundtrip_rate(_byte_level(), documents) == 1.0


def test_an_out_of_vocabulary_word_breaks_the_round_trip() -> None:
    # WordPiece maps anything it cannot cover to [UNK], and the original text is
    # unrecoverable from that. Two of three documents survive.
    documents = ["the", "cat", "aardvark"]

    assert roundtrip_rate(_wordpiece(), documents) == pytest.approx(2 / 3)


def test_the_rate_counts_documents_not_characters() -> None:
    # One long failing document weighs exactly as much as one short passing one:
    # the measure is a mean over documents, so a corpus of few long lines and a
    # corpus of many short ones are not interchangeable.
    documents = ["the", "aardvark " * 50]

    assert roundtrip_rate(_wordpiece(), documents) == pytest.approx(0.5)


def test_roundtrip_refuses_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        roundtrip_rate(_byte_level(), [])
