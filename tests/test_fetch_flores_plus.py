"""Fetching FLORES+ into the layout ``Corpus.load`` reads (PRD §10.1, D12).

The Hub is mocked at the same seam ``tests/test_from_pretrained.py`` uses. That
is not only about keeping the suite offline: FLORES+ is **gated**, so a test that
really fetched would pass on one machine and 403 on every other, including CI.

The assertions are about refusals. A fetch that silently produced misaligned
languages is the failure worth preventing here — parity is a ratio of means, and
a language with one extra line still produces a number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts import fetch_flores_plus

from glotscope.corpus import Corpus

_RESOLVED = "b" * 40


class _Sibling:
    def __init__(self, name: str) -> None:
        self.rfilename = name


class _Info:
    def __init__(self, files: list[str]) -> None:
        self.sha = _RESOLVED
        self.siblings = [_Sibling(name) for name in files]


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _release(
    tmp_path: Path,
    payloads: dict[str, str],
    *,
    split: str = "devtest",
) -> Path:
    """Write a stand-in release and return the directory holding it."""
    source = tmp_path / "release" / split
    source.mkdir(parents=True)
    for language, text in payloads.items():
        (source / f"{language}.jsonl").write_text(text, encoding="utf-8")
    return source


def _mock_hub(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    split: str = "devtest",
    fail: Exception | None = None,
) -> list[tuple[str, str | None]]:
    """Patch the Hub seam, recording every ``(filename, revision)`` fetched."""
    fetched: list[tuple[str, str | None]] = []

    def _dataset_info(repo: str, revision: str | None = None) -> _Info:
        if fail is not None:
            raise fail
        return _Info([f"{split}/{path.stem}.jsonl" for path in sorted(source.glob("*.jsonl"))])

    def _download(
        repo: str, filename: str, repo_type: str = "model", revision: str | None = None
    ) -> str:
        fetched.append((filename, revision))
        return str(source / Path(filename).name)

    monkeypatch.setattr(fetch_flores_plus, "_hub", lambda: (_dataset_info, _download))
    return fetched


_TWO_LANGUAGES = {
    "eng_Latn": _jsonl([{"id": 1, "text": "The cat sat."}, {"id": 2, "text": "It rained."}]),
    "hin_Deva": _jsonl([{"id": 1, "text": "बिल्ली बैठी।"}, {"id": 2, "text": "बारिश हुई।"}]),
}


def test_it_writes_the_layout_corpus_load_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point: what this writes must load without a further step."""
    fetched = _mock_hub(monkeypatch, _release(tmp_path, _TWO_LANGUAGES))
    root = tmp_path / "corpora"

    manifest = fetch_flores_plus.fetch(root, split="devtest", version="2024.08")

    assert [name for name, _ in fetched] == ["devtest/eng_Latn.jsonl", "devtest/hin_Deva.jsonl"]
    written = root / "flores_plus" / "2024.08" / "devtest" / "eng_Latn.txt"
    assert written.read_text(encoding="utf-8") == "The cat sat.\nIt rained.\n"

    loaded = Corpus.flores_plus(["eng_Latn", "hin_Deva"]).load(root)
    assert loaded.lines["eng_Latn"] == ("The cat sat.", "It rained.")
    # The digest the fetch reports is the one a pinned run will check against.
    assert loaded.corpus.sha256 == manifest["corpus_sha256"]


def test_the_revision_is_resolved_before_anything_is_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every file must come from the commit the manifest names, not from a
    branch that may have moved between the two calls."""
    fetched = _mock_hub(monkeypatch, _release(tmp_path, _TWO_LANGUAGES))

    manifest = fetch_flores_plus.fetch(tmp_path / "corpora", split="devtest")

    assert manifest["revision"] == _RESOLVED
    assert manifest["revision_was_pinned"] is False
    assert {revision for _, revision in fetched} == {_RESOLVED}


def test_a_gated_repository_says_so(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """403 is not 404. Reporting it as a missing dataset sends the reader to
    check their spelling instead of accepting the terms."""
    _mock_hub(
        monkeypatch,
        _release(tmp_path, _TWO_LANGUAGES),
        fail=RuntimeError("403 Client Error"),
    )

    with pytest.raises(SystemExit) as caught:
        fetch_flores_plus.fetch(tmp_path / "corpora", split="devtest")

    message = str(caught.value)
    assert "gated" in message
    assert "HF_TOKEN" in message


def test_unequal_document_counts_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = dict(_TWO_LANGUAGES)
    payloads["hin_Deva"] = _jsonl([{"id": 1, "text": "बिल्ली बैठी।"}])
    _mock_hub(monkeypatch, _release(tmp_path, payloads))

    with pytest.raises(SystemExit) as caught:
        fetch_flores_plus.fetch(tmp_path / "corpora", split="devtest")

    assert "unequal document counts" in str(caught.value)


def test_equal_counts_with_different_sentence_ids_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal counts are not alignment. Two languages can hold the same number of
    *different* sentences, and every parity number would be computed across
    them."""
    payloads = dict(_TWO_LANGUAGES)
    payloads["hin_Deva"] = _jsonl([{"id": 7, "text": "बिल्ली बैठी।"}, {"id": 8, "text": "बारिश हुई।"}])
    _mock_hub(monkeypatch, _release(tmp_path, payloads))

    with pytest.raises(SystemExit) as caught:
        fetch_flores_plus.fetch(tmp_path / "corpora", split="devtest")

    assert "sentence ids differ" in str(caught.value)


def test_a_document_holding_a_newline_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = dict(_TWO_LANGUAGES)
    payloads["eng_Latn"] = _jsonl([{"id": 1, "text": "Two\nlines."}, {"id": 2, "text": "One."}])
    _mock_hub(monkeypatch, _release(tmp_path, payloads))

    with pytest.raises(SystemExit) as caught:
        fetch_flores_plus.fetch(tmp_path / "corpora", split="devtest")

    assert "newline" in str(caught.value)


def test_a_renamed_text_field_names_what_it_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Discovered, not assumed. A release that renames the column is an error
    listing the keys it carries, because guessing would put an unknown column
    under every number."""
    payloads = {
        "eng_Latn": _jsonl([{"id": 1, "sentence": "The cat sat."}]),
        "hin_Deva": _jsonl([{"id": 1, "sentence": "बिल्ली बैठी।"}]),
    }
    _mock_hub(monkeypatch, _release(tmp_path, payloads))

    with pytest.raises(SystemExit) as caught:
        fetch_flores_plus.fetch(tmp_path / "corpora", split="devtest")

    assert "'sentence'" in str(caught.value)

    manifest = fetch_flores_plus.fetch(tmp_path / "corpora", split="devtest", text_field="sentence")
    assert manifest["text_field"] == "sentence"


def test_a_variety_absent_from_this_split_is_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The splits do not carry the same set — dev has 227 varieties and devtest
    221 in the release checked on 29 Aug 2026 — so this is a real answer about
    the input rather than a typo."""
    _mock_hub(monkeypatch, _release(tmp_path, _TWO_LANGUAGES))

    with pytest.raises(SystemExit) as caught:
        fetch_flores_plus.fetch(
            tmp_path / "corpora", split="devtest", languages=["eng_Latn", "wuu_Hans"]
        )

    assert "wuu_Hans" in str(caught.value)


def test_the_cli_writes_a_manifest_beside_the_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _mock_hub(monkeypatch, _release(tmp_path, _TWO_LANGUAGES))
    root = tmp_path / "corpora"

    assert fetch_flores_plus.main(["--root", str(root), "--split", "devtest"]) == 0

    manifest = json.loads(
        (root / "flores_plus" / "2024.08" / "devtest.fetch.json").read_text(encoding="utf-8")
    )
    assert manifest["repo"] == "openlanguagedata/flores_plus"
    assert manifest["languages"]["eng_Latn"]["documents"] == 2
    captured = capsys.readouterr()
    assert "corpus_sha256" in captured.out
    # An unpinned fetch is reproducible against a commit nobody chose, and says so.
    assert "--revision was not given" in captured.err
