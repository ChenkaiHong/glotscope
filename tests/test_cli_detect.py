"""``glotscope detect`` — Tier 2 from the command line (PRD §8.2, §7.9).

Executed through ``main`` rather than by calling the handler, because the exit
code is part of the interface: 0 produced a document, 1 is a typed refusal, 2 is
scheduled but not built. A test that calls the handler directly cannot tell those
apart, and the difference is what tells a reader whether to fix their argument or
wait for a release.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from _safetensors import f32, write_safetensors
from glotscope.cli import main
from glotscope.lint import byte_to_unicode

_BYTE_CHAR = byte_to_unicode()


def _tokenizer(path: Path, *, byte_values: range) -> int:
    """Write a byte-level BPE and return its vocabulary size."""
    vocab = {_BYTE_CHAR[value]: index for index, value in enumerate(byte_values)}
    vocab["<s>"] = len(vocab)
    backend = BackendTokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<s>"])
    backend.save(str(path))
    return len(vocab)


def _weights(path: Path, rows: int) -> Path:
    """Ascending row norms, so the ranking is determined rather than arbitrary."""
    scale = np.arange(1, rows + 1, dtype=np.float32).reshape(rows, 1)
    return write_safetensors(path, {"wte.weight": f32(scale * np.full((rows, 4), 0.5))})


def test_detect_writes_a_document_spanning_tier_zero_and_tier_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # G2: one document spans every tier that ran. Tier 2 needs no corpus, so
    # this is the two-tier case and the corpus block must be absent, not null.
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights(tmp_path / "model.safetensors", rows)

    # Act
    code = main(["detect", str(tokenizer), "--weights", str(weights), "--top-pct", "5.0"])

    # Assert
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert set(document) >= {"schema_version", "manifest", "tier0", "tier2", "warnings"}
    assert "corpus" not in document["manifest"]
    assert "tier1" not in document


def test_the_weights_block_records_what_was_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # §9's weights block is the audit trail for the quantization refusal.
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights(tmp_path / "model.safetensors", rows)

    # Act
    main(["detect", str(tokenizer), "--weights", str(weights)])

    # Assert
    block = json.loads(capsys.readouterr().out)["manifest"]["weights"]
    assert block["dtype"] == "float32"
    assert block["tied_embeddings"] is True
    assert len(block["shard_sha256"]) == 64


def test_top_pct_and_both_exclusion_counts_reach_the_parameter_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # §7.9: recording only the post-exclusion count makes the denominator
    # unreproducible, which is the whole reason the ordering is normative.
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights(tmp_path / "model.safetensors", rows)

    # Act
    main(["detect", str(tokenizer), "--weights", str(weights), "--top-pct", "10.0"])

    # Assert
    parameters = json.loads(capsys.readouterr().out)["manifest"]["parameters"]
    assert parameters["top_pct"] == 10.0
    assert parameters["candidates_pre_exclusion"] > parameters["candidates_post_exclusion"]
    assert parameters["first_pc_removed"] is False


def test_a_tied_checkpoint_with_no_reference_set_is_a_typed_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Exit 1, not exit 2: this is a real answer about this checkpoint, not a
    # feature that has yet to be written.
    # Arrange — only the bytes UTF-8 uses, so the chain is exhausted.
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(0xF5))
    weights = _weights(tmp_path / "model.safetensors", rows)

    # Act
    code = main(["detect", str(tokenizer), "--weights", str(weights)])

    # Assert
    assert code == 1
    assert "reference set" in capsys.readouterr().err


def test_a_hub_identifier_for_the_weights_is_reported_as_unbuilt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Exit 2. Saying "no such file" would send the reader after the wrong fix.
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    _tokenizer(tokenizer, byte_values=range(256))

    # Act
    code = main(["detect", str(tokenizer), "--weights", "google/gemma-2b"])

    # Assert
    assert code == 2
    assert "not implemented" in capsys.readouterr().err.lower()


def test_a_mistyped_weights_path_is_a_wrong_argument_not_a_missing_feature(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    _tokenizer(tokenizer, byte_values=range(256))

    # Act
    code = main(["detect", str(tokenizer), "--weights", str(tmp_path / "absent.safetensors")])

    # Assert
    assert code == 1
    assert "absent.safetensors" in capsys.readouterr().err


def test_the_document_can_be_written_to_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights(tmp_path / "model.safetensors", rows)
    out = tmp_path / "result.json"

    # Act
    code = main(["detect", str(tokenizer), "--weights", str(weights), "--out", str(out)])

    # Assert
    assert code == 0
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text(encoding="utf-8"))["tier2"]["tied"] is True


def test_first_principal_component_removal_is_off_unless_asked_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # D9. Recorded either way, because it changes every value.
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights(tmp_path / "model.safetensors", rows)

    # Act
    main(["detect", str(tokenizer), "--weights", str(weights), "--remove-first-pc"])

    # Assert
    document = json.loads(capsys.readouterr().out)
    assert document["manifest"]["parameters"]["first_pc_removed"] is True
    assert document["tier2"]["first_pc_removed"] is True


def test_the_warnings_array_carries_the_reference_set_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Which chain link supplied t_ref moves every cosine value and §9 has no
    # field for it, so the warnings array is where it has to appear.
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights(tmp_path / "model.safetensors", rows)

    # Act
    main(["detect", str(tokenizer), "--weights", str(weights)])

    # Assert
    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("unused_bytes" in warning for warning in warnings)
