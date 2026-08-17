"""Reading ``E_in``/``E_out`` out of a safetensors file (PRD §7.9, §8.1).

Fixtures are written by hand rather than through ``safetensors.numpy.save_file``
because the format's own header is what these tests are about, and because two
of §7.9's three reference checkpoints store BF16 — a dtype numpy does not have,
so ``safetensors.numpy`` cannot round-trip it and a fixture built through that
API could not reach the code path the real checkpoints take.

The tensor names are not invented. They were read from the published safetensors
headers of the three reference checkpoints without downloading any weights:

* ``google/gemma-2b`` — ``model.embed_tokens.weight``, BF16, no ``lm_head`` (tied)
* ``openai-community/gpt2-medium`` — ``wte.weight``, F32, tied
* ``ai21labs/Jamba-v0.1`` — ``model.embed_tokens.weight`` + ``lm_head.weight``, BF16
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from _safetensors import bf16, f32, write_safetensors
from glotscope.embeddings import Embeddings
from glotscope.errors import UnsupportedCheckpointError


def test_a_checkpoint_without_a_separate_head_is_read_as_tied(tmp_path: Path) -> None:
    # Arrange — gemma-2b's shape: one embedding tensor and no lm_head.
    rows = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = write_safetensors(
        tmp_path / "model.safetensors", {"model.embed_tokens.weight": f32(rows)}
    )

    # Act
    embeddings = Embeddings.from_file(path, vocab_size=4)

    # Assert
    assert embeddings.tied
    assert embeddings.e_out is None
    assert embeddings.n_rows == 4
    assert embeddings.dtype == "float32"
    np.testing.assert_array_equal(embeddings.e_in, rows)


def test_a_separate_lm_head_is_read_as_untied(tmp_path: Path) -> None:
    # Arrange — Jamba-v0.1's shape.
    e_in = np.arange(12, dtype=np.float32).reshape(4, 3)
    e_out = np.arange(12, 24, dtype=np.float32).reshape(4, 3)
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {"model.embed_tokens.weight": f32(e_in), "lm_head.weight": f32(e_out)},
    )

    # Act
    embeddings = Embeddings.from_file(path, vocab_size=4)

    # Assert
    assert not embeddings.tied
    assert embeddings.e_out is not None
    np.testing.assert_array_equal(embeddings.e_out, e_out)


def test_the_gpt2_embedding_name_resolves(tmp_path: Path) -> None:
    # Arrange — gpt2-medium calls it wte.weight, not model.embed_tokens.weight.
    rows = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = write_safetensors(tmp_path / "model.safetensors", {"wte.weight": f32(rows)})

    # Act
    embeddings = Embeddings.from_file(path, vocab_size=2)

    # Assert
    np.testing.assert_array_equal(embeddings.e_in, rows)
    assert embeddings.tied


def test_bfloat16_is_upcast_but_recorded_as_the_checkpoint_dtype(tmp_path: Path) -> None:
    # numpy has no bfloat16, so the array must be widened to work on at all —
    # but the manifest records what the checkpoint IS, not what we loaded it as.
    # Arrange
    rows = np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype=np.float32)
    path = write_safetensors(
        tmp_path / "model.safetensors", {"model.embed_tokens.weight": bf16(rows)}
    )

    # Act
    embeddings = Embeddings.from_file(path, vocab_size=2)

    # Assert
    assert embeddings.dtype == "bfloat16"
    assert embeddings.e_in.dtype == np.float32
    # These values are exactly representable in bfloat16, so truncation is lossless.
    np.testing.assert_array_equal(embeddings.e_in, rows)


def test_a_quantized_checkpoint_is_refused_rather_than_warned_about(tmp_path: Path) -> None:
    # Arrange
    payload = np.arange(6, dtype=np.int8).tobytes()
    path = write_safetensors(
        tmp_path / "model.safetensors", {"model.embed_tokens.weight": ("I8", [2, 3], payload)}
    )

    # Act / Assert
    with pytest.raises(UnsupportedCheckpointError) as excinfo:
        Embeddings.from_file(path, vocab_size=2)
    assert "int8" in str(excinfo.value).lower()


def test_rows_above_the_vocabulary_are_reported_as_padding(tmp_path: Path) -> None:
    # Link two of the §7.9 reference-set chain.
    # Arrange
    rows = np.zeros((6, 3), dtype=np.float32)
    path = write_safetensors(
        tmp_path / "model.safetensors", {"model.embed_tokens.weight": f32(rows)}
    )

    # Act
    embeddings = Embeddings.from_file(path, vocab_size=4)

    # Assert
    assert embeddings.n_rows == 6
    assert embeddings.padding_rows == (4, 5)


def test_the_recorded_digest_is_of_the_file_on_disk(tmp_path: Path) -> None:
    # The manifest pins the artifact by hash and by nothing else.
    # Arrange
    rows = np.zeros((2, 3), dtype=np.float32)
    path = write_safetensors(
        tmp_path / "model.safetensors", {"model.embed_tokens.weight": f32(rows)}
    )
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    # Act
    embeddings = Embeddings.from_file(path, vocab_size=2)

    # Assert
    assert embeddings.shard_sha256 == expected


def test_a_file_with_no_embedding_tensor_names_what_it_looked_for(tmp_path: Path) -> None:
    # Arrange
    rows = np.zeros((2, 3), dtype=np.float32)
    path = write_safetensors(
        tmp_path / "model.safetensors", {"model.layers.0.mlp.weight": f32(rows)}
    )

    # Act / Assert
    with pytest.raises(UnsupportedCheckpointError) as excinfo:
        Embeddings.from_file(path, vocab_size=2)
    assert "embed_tokens" in str(excinfo.value)


def test_the_manifest_records_what_a_reader_needs_to_refuse_the_file_again(
    tmp_path: Path,
) -> None:
    # §9's weights block exists so a reader can tell a bf16 checkpoint from a
    # 4-bit mirror republished under the same name.
    # Arrange
    rows = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = write_safetensors(tmp_path / "model.safetensors", {"wte.weight": bf16(rows)})

    # Act
    manifest = Embeddings.from_file(path, vocab_size=2).manifest

    # Assert
    assert manifest.dtype == "bfloat16"
    assert manifest.tied_embeddings is True
    assert manifest.shard_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_local_file_declares_no_license_rather_than_borrowing_one(tmp_path: Path) -> None:
    # A safetensors file on disk carries no license metadata, and the repository
    # it came from is not knowable from the bytes. Recording anything else would
    # put an unverifiable claim in the field a `--license-filter` run trusts.
    # Arrange
    rows = np.zeros((2, 3), dtype=np.float32)
    path = write_safetensors(tmp_path / "model.safetensors", {"wte.weight": f32(rows)})

    # Act
    manifest = Embeddings.from_file(path, vocab_size=2).manifest

    # Assert
    assert manifest.license_spdx == "UNKNOWN"


def test_an_untied_checkpoint_says_so_in_its_manifest(tmp_path: Path) -> None:
    # Arrange
    rows = np.zeros((2, 3), dtype=np.float32)
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {"model.embed_tokens.weight": bf16(rows), "lm_head.weight": bf16(rows)},
    )

    # Act
    manifest = Embeddings.from_file(path, vocab_size=2).manifest

    # Assert
    assert manifest.tied_embeddings is False
