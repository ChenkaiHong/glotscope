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


def _weights_multi_direction(path: Path, rows: int) -> Path:
    """Ascending row norms that do not all lie along one direction.

    ``_weights`` is rank 1 by construction, which is what makes its ranking easy
    to reason about. It is unusable for the first-principal-component flag: every
    row of a rank-1 matrix is a multiple of the same vector, so projecting that
    direction out leaves the zero matrix and there is no reference direction left
    to measure against.
    """
    index = np.arange(1, rows + 1, dtype=np.float32)
    columns = np.stack(
        [index, index % 7, index % 3, np.ones(rows, dtype=np.float32)],
        axis=1,
    )
    return write_safetensors(path, {"wte.weight": f32(columns)})


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


def test_stderr_carries_every_warning_the_document_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The warnings that decide how a Tier 2 number should be read — which link of
    # the t_ref chain supplied the reference set, and LOW_CONFIDENCE when the two
    # indicators disagree — come from the tier report, not from the tokenizer.
    # Echoing only the tokenizer's would leave stderr silent on a degraded run,
    # and silence reads as "nothing to report".
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights(tmp_path / "model.safetensors", rows)

    code = main(["detect", str(tokenizer), "--weights", str(weights), "--top-pct", "5.0"])

    assert code == 0
    captured = capsys.readouterr()
    published = json.loads(captured.out)["warnings"]
    assert published, "this fixture is expected to emit at least one warning"
    emitted = [line.removeprefix("warning: ") for line in captured.err.splitlines()]
    assert emitted == published


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


def test_a_bare_repo_id_is_routed_to_the_hub_and_not_to_the_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The routing is the claim: a bare name is a Hub identifier, and reporting
    # it as a missing local file would send the reader after the wrong fix. The
    # Hub is mocked at the file-fetch seam so this asserts routing without
    # making the suite depend on huggingface.co being up.
    # Arrange
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    _weights(tmp_path / "model.safetensors", rows)
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": rows}), encoding="utf-8")
    asked: list[str] = []

    def _download(repo_id: str, filename: str, *, revision: str | None = None) -> str:
        from huggingface_hub.errors import EntryNotFoundError

        asked.append(f"{repo_id}/{filename}")
        if filename == "model.safetensors.index.json":
            raise EntryNotFoundError(filename)
        return str(tmp_path / filename)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda repo_id, revision=None: type("Info", (), {"card_data": {"license": "mit"}})(),
    )

    # Act
    code = main(["detect", str(tokenizer), "--weights", "acme/tiny"])

    # Assert
    assert code == 0
    assert "acme/tiny/config.json" in asked
    assert json.loads(capsys.readouterr().out)["manifest"]["weights"]["license_spdx"] == "mit"


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
    # Arrange — `_weights` is rank 1, and removing the first principal component
    # from a rank-1 matrix leaves exactly the zero matrix, which detect() refuses
    # rather than ranking by token id. That refusal is the point of the guard, so
    # this test needs weights that still span a direction afterwards.
    tokenizer = tmp_path / "tokenizer.json"
    rows = _tokenizer(tokenizer, byte_values=range(256))
    weights = _weights_multi_direction(tmp_path / "model.safetensors", rows)

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
