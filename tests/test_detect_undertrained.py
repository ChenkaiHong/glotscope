"""Tier 2 end to end: tokenizer + embeddings to a ``Tier2Report`` (PRD §7.9).

The pieces are already tested apart — ``reference_set`` for the fallback chain,
``detect`` for the indicators. What is under test here is the wiring, and the
wiring is where §7.9's traps live: Stage 1 must exclude exactly three sets, the
reference set must come from the chain rather than from an assumption, and
``top_pct`` must apply after the exclusion and not before.

Embeddings are constructed directly rather than written to safetensors and read
back. The file format is tested in ``test_embeddings_loading.py``; putting it in
the way here would make every assertion below depend on a parser that has
nothing to do with the claim being made.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.embeddings import Embeddings
from glotscope.enums import Confidence, Indicator, TokenClass
from glotscope.errors import NoReferenceSetError, UnsupportedCheckpointError
from glotscope.lint import byte_to_unicode
from glotscope.tokenizer import Tokenizer

_BYTE_CHAR = byte_to_unicode()


def _byte_level(path: Path, *, byte_values: range) -> Tokenizer:
    """A byte-level BPE over the given bytes, plus one special token."""
    vocab = {_BYTE_CHAR[value]: index for index, value in enumerate(byte_values)}
    vocab["<s>"] = len(vocab)
    backend = BackendTokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<s>"])
    backend.save(str(path))
    return Tokenizer.from_file(path)


def _embeddings(
    tokenizer: Tokenizer,
    *,
    n_rows: int | None = None,
    tied: bool = True,
) -> Embeddings:
    """Distinct row norms, ascending by id, so the ranking is predictable."""
    vocab_size = tokenizer.lint().vocab_size
    rows = n_rows if n_rows is not None else vocab_size
    scale = np.arange(1, rows + 1, dtype=np.float64).reshape(rows, 1)
    e_in = scale * np.full((rows, 4), 0.5)
    return Embeddings(
        e_in=e_in,
        e_out=None if tied else e_in[::-1].copy(),
        tied=tied,
        dtype="float32",
        shard_sha256="0" * 64,
        checkpoint="acme/model",
        n_rows=rows,
        vocab_size=vocab_size,
    )


def test_a_tied_checkpoint_produces_a_report_from_the_cosine_indicator(tmp_path: Path) -> None:
    # Arrange — 256 bytes present, so 0xF5-0xFF supply the reference set.
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))

    # Act
    report = tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=10.0)

    # Assert
    assert report.indicator is Indicator.COSINE_TO_UNUSED_MEAN
    assert report.tied is True
    assert report.indicator_agreement is None


def test_top_pct_is_applied_after_stage_one_and_both_counts_are_published(
    tmp_path: Path,
) -> None:
    # §7.9: the ordering is what makes a published candidate-set size reproducible.
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))
    tier0 = tokenizer.lint()
    surviving = tier0.vocab_size - len(tier0.stage1_exclusions())

    # Act
    report = tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=10.0)

    # Assert
    assert report.candidates_pre_exclusion == tier0.vocab_size
    assert report.candidates_post_exclusion == surviving
    assert report.candidate_count == int(surviving * 10.0 / 100.0)


def test_no_stage_one_exclusion_is_ever_offered_as_a_candidate(tmp_path: Path) -> None:
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))
    excluded = tokenizer.lint().stage1_exclusions()

    # Act
    report = tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=100.0)

    # Assert
    assert not {candidate.token_id for candidate in report.candidates} & excluded


def test_candidates_are_ranked_from_one_and_carry_their_utf8_class(tmp_path: Path) -> None:
    # A candidate without its class is a bare integer: a reader cannot tell an
    # under-trained word piece from a stray byte, and §14.3 regresses on that
    # distinction.
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))

    # Act
    report = tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=5.0)

    # Assert
    assert [candidate.rank for candidate in report.candidates] == list(
        range(1, report.candidate_count + 1)
    )
    assert all(candidate.token_class in set(TokenClass) for candidate in report.candidates)


def test_the_reference_set_source_is_named_in_the_warnings(tmp_path: Path) -> None:
    # Which link of the chain supplied t_ref changes the numbers, and §9 has no
    # field for it. The warnings array is where a contested choice goes.
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))

    # Act
    report = tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=5.0)

    # Assert
    assert any("unused_bytes" in warning for warning in report.warnings)


def test_padding_rows_supply_the_reference_set_when_the_vocabulary_has_no_spares(
    tmp_path: Path,
) -> None:
    # Arrange — only UTF-8-valid bytes, but the embedding matrix is wider than |V|.
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(0xF5))

    # Act
    report = tokenizer.detect_undertrained(
        _embeddings(tokenizer, n_rows=tokenizer.lint().vocab_size + 8), top_pct=5.0
    )

    # Assert
    assert any("padding_rows" in warning for warning in report.warnings)
    assert report.candidates_pre_exclusion == tokenizer.lint().vocab_size


def test_an_untied_checkpoint_with_no_reference_set_degrades_rather_than_refusing(
    tmp_path: Path,
) -> None:
    # §7.9's table: L2(E_in) needs no reference set, so untied has somewhere to go.
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(0xF5))

    # Act
    report = tokenizer.detect_undertrained(_embeddings(tokenizer, tied=False), top_pct=5.0)

    # Assert
    assert report.indicator is Indicator.L2_E_IN
    assert report.confidence is Confidence.LOW_CONFIDENCE
    assert report.indicator_agreement is None


def test_a_tied_checkpoint_with_no_reference_set_is_refused(tmp_path: Path) -> None:
    # Nothing to degrade to: the only available indicator is defined against u_ref.
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(0xF5))

    # Act / Assert
    with pytest.raises(NoReferenceSetError):
        tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=5.0)


def test_an_embedding_matrix_smaller_than_the_vocabulary_is_refused(tmp_path: Path) -> None:
    # Row i of E_in is token i or it is nothing. A matrix with fewer rows than
    # the vocabulary has no row for the tail of it, so ranking would index off
    # the end — or, worse, quietly score a truncated domain.
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))
    built = _embeddings(tokenizer)
    truncated = Embeddings(
        e_in=built.e_in[:-3],
        e_out=None,
        tied=True,
        dtype="float32",
        shard_sha256="0" * 64,
        checkpoint="acme/other",
        n_rows=built.n_rows - 3,
        vocab_size=built.vocab_size,
    )

    # Act / Assert
    with pytest.raises(UnsupportedCheckpointError, match="no row"):
        tokenizer.detect_undertrained(truncated)


def test_a_config_vocab_size_above_the_tokenizers_is_analysed_not_refused(
    tmp_path: Path,
) -> None:
    # The Qwen3 shape: config declares 151936, the tokenizer holds 151669. The
    # gap is not a mismatch to refuse — it *is* chain link 2. TokenizerManifest
    # records both fields precisely because "Tier 2's reference-set chain uses
    # the gap", so refusing on the difference disables the padding-row fallback
    # on exactly the checkpoints that have one.
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))
    vocab_size = tokenizer.lint().vocab_size
    padded = _embeddings(tokenizer, n_rows=vocab_size + 8)
    declared = Embeddings(
        e_in=padded.e_in,
        e_out=None,
        tied=True,
        dtype="float32",
        shard_sha256="0" * 64,
        checkpoint="Qwen/Qwen3-shaped",
        n_rows=vocab_size + 8,
        # What config.json says, which is the row count and not the token count.
        vocab_size=vocab_size + 8,
    )

    # Act
    report = tokenizer.detect_undertrained(declared, top_pct=5.0)

    # Assert
    # The domain is the tokenizer's vocabulary, and the 8 rows above it were
    # spent as t_ref rather than ranked as candidates.
    assert report.candidates_pre_exclusion == vocab_size
    assert any("padding_rows" in warning for warning in report.warnings)


def test_a_unigram_tokenizer_is_analysed_but_warned_about(tmp_path: Path) -> None:
    # §7.9's scope limit is explicit: BPE only, Unigram-LM untested. Untested is
    # not the same as refused, so this runs and says so.
    # Arrange
    path = tmp_path / "tokenizer.json"
    pieces = [(_BYTE_CHAR[value], -float(value + 1)) for value in range(256)]
    backend = BackendTokenizer(models.Unigram(pieces, 0, byte_fallback=False))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.save(str(path))
    tokenizer = Tokenizer.from_file(path)

    # Act
    report = tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=5.0)

    # Assert
    assert any("unigram" in warning.lower() for warning in report.warnings)


def test_the_document_block_carries_the_counts_a_reader_needs(tmp_path: Path) -> None:
    # Arrange
    tokenizer = _byte_level(tmp_path / "tokenizer.json", byte_values=range(256))

    # Act
    block = tokenizer.detect_undertrained(_embeddings(tokenizer), top_pct=5.0).to_dict()

    # Assert
    assert json.loads(json.dumps(block))["indicator"] == "cosine_to_unused_mean"
    assert block["tied"] is True
    assert block["candidates_pre_exclusion"] > block["candidates_post_exclusion"]
