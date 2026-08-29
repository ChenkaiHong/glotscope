"""Loading a tokenizer from the Hub (PRD §8.1, §9, §11).

The Hub is mocked at the fetch seam, so these assert *provenance* without making
the suite depend on huggingface.co being up — the same seam
``tests/test_cli_detect.py`` uses for the weights path.

Provenance is the whole point of this loader. §11 requires every leaderboard row
pinned by commit revision and mirror-sourced rows visibly labelled, and §16.1's
nightly job exists to catch upstream tokenizers changing under a published
number. A loader that fetched the artifact and recorded "huggingface" would
satisfy none of that.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.errors import TokenizerLoadError
from glotscope.lint import byte_to_unicode
from glotscope.tokenizer import Tokenizer

_RESOLVED = "a" * 40
"""What the Hub resolves a branch name to. Forty hex characters, like a real one."""


class _Card:
    """Stand-in for ``huggingface_hub``'s ``ModelInfo``, carrying what is read."""

    def __init__(self, license_spdx: str | None) -> None:
        self.card_data = {"license": license_spdx} if license_spdx else {}
        self.sha = _RESOLVED


def _write_repo(tmp_path: Path, *, with_config: bool = True) -> Path:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.save(str(tmp_path / "tokenizer.json"))
    if with_config:
        (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 300}), encoding="utf-8")
    return tmp_path


def _mock_hub(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    *,
    license_spdx: str | None = "apache-2.0",
    missing: tuple[str, ...] = (),
) -> list[tuple[str, str, str | None]]:
    """Patch the Hub seam and record every fetch as ``(repo, filename, revision)``."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    asked: list[tuple[str, str, str | None]] = []

    def _download(repo_id: str, filename: str, *, revision: str | None = None, **_: Any) -> str:
        asked.append((repo_id, filename, revision))
        if filename in missing:
            raise EntryNotFoundError(filename)
        return str(repo / filename)

    def _model_info(repo_id: str, *, revision: str | None = None, **_: Any) -> _Card:
        return _Card(license_spdx)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr(huggingface_hub, "model_info", _model_info)
    return asked


def test_the_resolved_commit_is_recorded_not_the_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unpinned run still produces a pinned manifest: the branch is resolved and
    # the SHA it resolved to is recorded. §11 pins rows by revision, and a
    # manifest saying "main" would name something that moves.
    repo = _write_repo(tmp_path)
    _mock_hub(monkeypatch, repo)

    tokenizer = Tokenizer.from_pretrained("acme/model")

    assert tokenizer.manifest.revision == _RESOLVED
    assert tokenizer.manifest.id == "acme/model"


def test_an_unpinned_run_says_so_in_the_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The manifest stays honest either way, but an unpinned run is not a
    # reproducible one — it recorded where a branch pointed at one moment.
    repo = _write_repo(tmp_path)
    _mock_hub(monkeypatch, repo)

    tokenizer = Tokenizer.from_pretrained("acme/model")

    assert any("did not pin" in warning for warning in tokenizer.warnings)


def test_a_pinned_run_carries_no_such_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_repo(tmp_path)
    asked = _mock_hub(monkeypatch, repo)

    tokenizer = Tokenizer.from_pretrained("acme/model", revision="b" * 40)

    assert not any("did not pin" in warning for warning in tokenizer.warnings)
    assert tokenizer.manifest.revision == "b" * 40
    assert all(revision == "b" * 40 for _, _, revision in asked)


def test_the_hash_is_of_the_bytes_that_were_analysed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same guarantee `from_file` makes: the digest in the manifest is the digest
    # of what was parsed, not of whatever the path pointed at a moment later.
    repo = _write_repo(tmp_path)
    _mock_hub(monkeypatch, repo)
    expected = hashlib.sha256((repo / "tokenizer.json").read_bytes()).hexdigest()

    tokenizer = Tokenizer.from_pretrained("acme/model")

    assert tokenizer.manifest.tokenizer_json_sha256 == expected


def test_a_mirror_is_labelled_because_the_leaderboard_must_show_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # §11: a leaderboard that silently uses unofficial re-uploads is a legitimate
    # line of attack, so the flag travels in the manifest rather than in a note.
    repo = _write_repo(tmp_path)
    _mock_hub(monkeypatch, repo)

    tokenizer = Tokenizer.from_pretrained("someone/llama-mirror", is_mirror=True)

    assert tokenizer.manifest.source_is_mirror is True
    assert tokenizer.manifest.source == "huggingface"


def test_the_config_vocab_size_is_read_because_tier_2_uses_the_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # vocab_size_config differs from the tokenizer's count on real models — Qwen3
    # reports 151936 against 151669 — and §7.9's reference chain reads the gap.
    repo = _write_repo(tmp_path)
    _mock_hub(monkeypatch, repo)

    tokenizer = Tokenizer.from_pretrained("acme/model")

    assert tokenizer.manifest.vocab_size_config == 300
    assert tokenizer.manifest.vocab_size_tokenizer == 256


def test_a_repo_without_a_config_records_null_rather_than_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Back-filling from the tokenizer's own count would be a *claim* that the
    # embedding matrix has no padding rows, and that claim is what chain link 2
    # reads.
    repo = _write_repo(tmp_path, with_config=False)
    _mock_hub(monkeypatch, repo, missing=("config.json",))

    tokenizer = Tokenizer.from_pretrained("acme/model")

    assert tokenizer.manifest.vocab_size_config is None


def test_the_license_comes_from_the_model_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _write_repo(tmp_path)
    _mock_hub(monkeypatch, repo, license_spdx="apache-2.0")

    assert Tokenizer.from_pretrained("acme/model").manifest.license_spdx == "apache-2.0"


def test_an_unlicensed_card_records_unknown_rather_than_a_guess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --license-filter reads this field, so an unverifiable guess would be worse
    # than an explicit unknown.
    repo = _write_repo(tmp_path)
    _mock_hub(monkeypatch, repo, license_spdx=None)

    assert Tokenizer.from_pretrained("acme/model").manifest.license_spdx == "UNKNOWN"


def test_a_repo_without_a_tokenizer_json_is_a_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not every model publishes one — SentencePiece-only repos are common — and
    # converting is out of scope (§3.2 forbids implementing a tokenizer).
    repo = _write_repo(tmp_path, with_config=False)
    _mock_hub(monkeypatch, repo, missing=("tokenizer.json",))

    with pytest.raises(TokenizerLoadError) as excinfo:
        Tokenizer.from_pretrained("acme/sentencepiece-only")

    assert "tokenizer.json" in str(excinfo.value)
