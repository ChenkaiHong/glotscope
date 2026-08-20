"""Resolving a checkpoint from the Hub without loading the model (PRD §6, §7.9).

The shard layout here is not invented. It was read from the three §7.9 reference
checkpoints' published metadata with no weights downloaded:

* ``openai-community/gpt2-medium`` — one ``model.safetensors``, no index
* ``google/gemma-2b`` — ``model.embed_tokens.weight`` in shard 1 of 2
* ``ai21labs/Jamba-v0.1`` — ``model.embed_tokens.weight`` in shard **1 of 21**
  and ``lm_head.weight`` in shard **21 of 21**

Jamba is why the index is read at all: the two shards holding the embeddings are
8.94 GB of a 96.06 GB checkpoint, so resolving through ``weight_map`` is the
difference between a Tier 2 run being cheap and being impractical.

The Hub itself is mocked at exactly one seam — the function that turns
``(repo, filename)`` into a local path. Everything below that is the real
parser, the real merge and the real refusals, so these tests fail if the format
handling breaks. Reaching the network would make the suite a test of
huggingface.co's uptime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _safetensors import bf16, f32, write_safetensors
from glotscope.embeddings import Embeddings, embedding_shards
from glotscope.errors import UnsupportedCheckpointError

_GEMMA_MAP = {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.9.mlp.up_proj.weight": "model-00002-of-00002.safetensors",
}

_JAMBA_MAP = {
    "model.embed_tokens.weight": "model-00001-of-00021.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00021.safetensors",
    "lm_head.weight": "model-00021-of-00021.safetensors",
}


def test_only_the_shards_holding_the_embeddings_are_selected() -> None:
    # Jamba: 2 shards out of 21, which is 8.94 GB instead of 96.06 GB.
    assert embedding_shards(_JAMBA_MAP) == (
        "model-00001-of-00021.safetensors",
        "model-00021-of-00021.safetensors",
    )


def test_a_tied_checkpoint_needs_one_shard() -> None:
    # Gemma has no lm_head at all, so shard 2 is never fetched.
    assert embedding_shards(_GEMMA_MAP) == ("model-00001-of-00002.safetensors",)


def test_one_shard_holding_both_tensors_is_named_once() -> None:
    # Downloading the same shard twice would double the cost of the whole run.
    weight_map = {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00001-of-00002.safetensors",
    }

    assert embedding_shards(weight_map) == ("model-00001-of-00002.safetensors",)


def test_an_index_naming_no_embedding_tensor_is_refused() -> None:
    # Better than downloading every shard on the chance one holds it.
    with pytest.raises(UnsupportedCheckpointError, match="embed"):
        embedding_shards({"model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors"})


class _FakeHub:
    """Serves files from a directory, in place of the Hub."""

    def __init__(self, root: Path, *, present: set[str]) -> None:
        self.root = root
        self.present = present
        self.fetched: list[str] = []

    def download(self, repo_id: str, filename: str, *, revision: str | None = None) -> str:
        from huggingface_hub.errors import EntryNotFoundError

        if filename not in self.present:
            raise EntryNotFoundError(f"{filename} not found in {repo_id}")
        self.fetched.append(filename)
        return str(self.root / filename)


def _install(
    monkeypatch: pytest.MonkeyPatch, hub: _FakeHub, *, license_value: str | None = "apache-2.0"
) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.download)

    def _model_info(repo_id: str, *, revision: str | None = None) -> Any:
        return type("Info", (), {"card_data": {"license": license_value}})()

    monkeypatch.setattr(huggingface_hub, "model_info", _model_info)


def _write_config(root: Path, vocab_size: int) -> None:
    (root / "config.json").write_text(json.dumps({"vocab_size": vocab_size}), encoding="utf-8")


def test_a_sharded_untied_checkpoint_is_assembled_from_two_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Jamba shape: E_in in the first shard, lm_head in the last.
    # Arrange
    _write_config(tmp_path, 4)
    rows = np.arange(8, dtype=np.float32).reshape(4, 2)
    write_safetensors(
        tmp_path / "model-00001-of-00021.safetensors", {"model.embed_tokens.weight": bf16(rows)}
    )
    write_safetensors(
        tmp_path / "model-00021-of-00021.safetensors", {"lm_head.weight": bf16(rows[::-1].copy())}
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": _JAMBA_MAP}), encoding="utf-8"
    )
    hub = _FakeHub(
        tmp_path,
        present={
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00021.safetensors",
            "model-00021-of-00021.safetensors",
        },
    )
    _install(monkeypatch, hub)

    # Act
    embeddings = Embeddings.from_checkpoint("ai21labs/Jamba-v0.1")

    # Assert
    assert embeddings.tied is False
    assert embeddings.e_out is not None
    assert embeddings.dtype == "bfloat16"
    assert embeddings.vocab_size == 4
    assert "model-00002-of-00021.safetensors" not in hub.fetched


def test_a_checkpoint_with_no_index_falls_back_to_the_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gpt2-medium shape.
    # Arrange
    _write_config(tmp_path, 4)
    write_safetensors(
        tmp_path / "model.safetensors", {"wte.weight": f32(np.zeros((4, 2), dtype=np.float32))}
    )
    hub = _FakeHub(tmp_path, present={"config.json", "model.safetensors"})
    _install(monkeypatch, hub, license_value="mit")

    # Act
    embeddings = Embeddings.from_checkpoint("openai-community/gpt2-medium")

    # Assert
    assert embeddings.tied is True
    assert embeddings.license_spdx == "mit"
    assert "model.safetensors.index.json" not in hub.fetched


def test_the_checkpoint_field_records_the_repo_id_not_a_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local cache path is machine-specific and would make the record
    # unreadable on any other machine. §9 identifies artifacts by hash.
    # Arrange
    _write_config(tmp_path, 4)
    write_safetensors(
        tmp_path / "model.safetensors", {"wte.weight": f32(np.zeros((4, 2), dtype=np.float32))}
    )
    _install(monkeypatch, _FakeHub(tmp_path, present={"config.json", "model.safetensors"}))

    # Act
    embeddings = Embeddings.from_checkpoint("openai-community/gpt2-medium")

    # Assert
    assert embeddings.checkpoint == "openai-community/gpt2-medium"


def test_a_multi_shard_digest_is_deterministic_and_covers_every_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One shard's digest would name half the evidence, and a digest that varied
    # by fetch order would fail `glotscope verify` for no reason.
    # Arrange
    _write_config(tmp_path, 4)
    rows = np.arange(8, dtype=np.float32).reshape(4, 2)
    write_safetensors(
        tmp_path / "model-00001-of-00021.safetensors", {"model.embed_tokens.weight": bf16(rows)}
    )
    write_safetensors(tmp_path / "model-00021-of-00021.safetensors", {"lm_head.weight": bf16(rows)})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": _JAMBA_MAP}), encoding="utf-8"
    )
    present = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00021.safetensors",
        "model-00021-of-00021.safetensors",
    }
    _install(monkeypatch, _FakeHub(tmp_path, present=present))

    # Act
    first = Embeddings.from_checkpoint("ai21labs/Jamba-v0.1").shard_sha256
    second = Embeddings.from_checkpoint("ai21labs/Jamba-v0.1").shard_sha256

    # Assert
    assert first == second
    assert len(first) == 64

    # And it moves when either shard's bytes move.
    write_safetensors(
        tmp_path / "model-00021-of-00021.safetensors",
        {"lm_head.weight": bf16(rows[::-1].copy())},
    )
    assert Embeddings.from_checkpoint("ai21labs/Jamba-v0.1").shard_sha256 != first


def test_a_quantized_checkpoint_is_refused_before_anything_is_ranked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — an int8 embedding, as a community mirror would republish it.
    _write_config(tmp_path, 4)
    payload = np.zeros((4, 2), dtype=np.int8).tobytes()
    write_safetensors(tmp_path / "model.safetensors", {"wte.weight": ("I8", [4, 2], payload)})
    _install(monkeypatch, _FakeHub(tmp_path, present={"config.json", "model.safetensors"}))

    # Act / Assert
    with pytest.raises(UnsupportedCheckpointError, match="int8"):
        Embeddings.from_checkpoint("acme/gpt2-medium-int8")


def test_a_checkpoint_whose_card_declares_no_license_says_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `--license-filter=commercial` reads this field, so an absent declaration
    # has to stay absent rather than defaulting to something permissive.
    # Arrange
    _write_config(tmp_path, 4)
    write_safetensors(
        tmp_path / "model.safetensors", {"wte.weight": f32(np.zeros((4, 2), dtype=np.float32))}
    )
    _install(
        monkeypatch,
        _FakeHub(tmp_path, present={"config.json", "model.safetensors"}),
        license_value=None,
    )

    # Act
    embeddings = Embeddings.from_checkpoint("acme/unlicensed")

    # Assert
    assert embeddings.license_spdx == "UNKNOWN"


def test_padding_rows_are_visible_when_the_matrix_is_wider_than_the_config_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Chain link 2 of §7.9's reference set. None of the three reference
    # checkpoints has padding, so this is the case a mocked fixture is for.
    # Arrange
    _write_config(tmp_path, 4)
    write_safetensors(
        tmp_path / "model.safetensors", {"wte.weight": f32(np.zeros((6, 2), dtype=np.float32))}
    )
    _install(monkeypatch, _FakeHub(tmp_path, present={"config.json", "model.safetensors"}))

    # Act
    embeddings = Embeddings.from_checkpoint("acme/padded")

    # Assert
    assert embeddings.n_rows == 6
    assert embeddings.padding_rows == (4, 5)
