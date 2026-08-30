"""Stanza and UDPipe (PRD §7.1, §10.3, D6).

The two adapters that were scheduled rather than refused, and the last
``NotImplementedError`` in the package. Both differ from every other segmenter
in one way that shapes the whole design: they are **trained models**, so the
boundaries depend on an artifact that is not the package.

Two consequences, and both are tested here rather than commented:

* **The model is an explicit path the caller pins.** Never a download on first
  use, which would put an unrecorded artifact behind a published number — and
  §7.1 requires the segmenter *model* version recorded, not a treebank release.
* **The recorded version is a digest of the file that produced the boundaries.**
  A package version says nothing about where a word ends.

The third-party modules are faked. Real Stanza models are hundreds of megabytes
and downloading one is exactly what the design forbids, so a suite that used them
would be a suite that cannot run in CI and would test the wrong thing anyway.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from glotscope.enums import Segmenter
from glotscope.errors import SegmenterUnavailableError
from glotscope.segmenters import get_segmenter

_SENTENCE = "The cat sat on the mat."


def _model_file(tmp_path: Path, name: str = "english.udpipe") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not a real model, but a real file with a real digest")
    return path


@pytest.fixture
def fake_udpipe(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A stand-in for ``ufal.udpipe`` recording how it was called."""
    calls: dict[str, Any] = {}

    class _Word:
        def __init__(self, form: str) -> None:
            self.form = form

    class _Sentence:
        def __init__(self, text: str) -> None:
            # A deliberately un-whitespace-like split, so a silent fallback to
            # whitespace segmentation cannot pass these tests.
            self.words = [_Word("")] + [_Word(part) for part in text.replace(".", " .").split()]

    class _Model:
        DEFAULT = "default"

        @staticmethod
        def load(path: str) -> Any:
            calls["loaded"] = path
            return _Model()

        def newTokenizer(self, options: str) -> Any:  # noqa: N802 - upstream spelling
            calls["tokenizer_options"] = options
            return _Tokenizer()

    class _Tokenizer:
        def setText(self, text: str) -> None:  # noqa: N802 - upstream spelling
            self._text = text

        def nextSentence(self, sentence: Any, error: Any) -> bool:  # noqa: N802
            if getattr(self, "_done", False):
                return False
            self._done = True
            sentence.words = _Sentence(self._text).words
            return True

    ufal = ModuleType("ufal")
    udpipe = ModuleType("ufal.udpipe")
    udpipe.Model = _Model  # type: ignore[attr-defined]
    udpipe.Sentence = lambda: SimpleNamespace(words=[])  # type: ignore[attr-defined]
    udpipe.ProcessingError = lambda: None  # type: ignore[attr-defined]
    ufal.udpipe = udpipe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ufal", ufal)
    monkeypatch.setitem(sys.modules, "ufal.udpipe", udpipe)
    return calls


@pytest.fixture
def fake_stanza(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A stand-in for ``stanza`` recording the arguments it was built with."""
    calls: dict[str, Any] = {}

    class _Token:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Doc:
        def __init__(self, text: str) -> None:
            self.sentences = [
                SimpleNamespace(tokens=[_Token(part) for part in text.replace(".", " .").split()])
            ]

    class _Pipeline:
        def __init__(self, **kwargs: Any) -> None:
            calls.update(kwargs)

        def __call__(self, text: str) -> _Doc:
            return _Doc(text)

    stanza = ModuleType("stanza")
    stanza.Pipeline = _Pipeline  # type: ignore[attr-defined]
    stanza.__version__ = "1.10.1"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stanza", stanza)
    return calls


# -- the model is pinned, never downloaded ----------------------------------


@pytest.mark.parametrize("segmenter", [Segmenter.UDPIPE, Segmenter.STANZA])
def test_a_trained_segmenter_without_a_model_is_refused(segmenter: Segmenter) -> None:
    """No default model. Picking one would put an artifact nobody chose behind
    every fertility number, and §7.1 requires the model recorded."""
    with pytest.raises(ValueError) as caught:
        get_segmenter(segmenter, language="eng_Latn")

    message = str(caught.value)
    assert "model" in message
    assert "download" in message


@pytest.mark.parametrize("segmenter", [Segmenter.UDPIPE, Segmenter.STANZA])
def test_a_model_path_that_does_not_exist_names_it(segmenter: Segmenter, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        get_segmenter(segmenter, language="eng_Latn", model=tmp_path / "absent.bin")

    assert "absent.bin" in str(caught.value)


def test_stanza_is_built_so_it_cannot_download(tmp_path: Path, fake_stanza: dict[str, Any]) -> None:
    """The decision this adapter was waiting on, asserted rather than described.

    A pipeline left free to download would fetch a model on first use, and the
    published number would rest on an artifact the manifest never saw.
    """
    model = _model_file(tmp_path, "en_tokenize.pt")

    get_segmenter(Segmenter.STANZA, language="eng_Latn", model=model)

    assert fake_stanza["download_method"] is None
    assert fake_stanza["processors"] == "tokenize"
    assert Path(fake_stanza["tokenize_model_path"]) == model
    assert fake_stanza["lang"] == "en"


def test_udpipe_loads_the_file_it_was_given(tmp_path: Path, fake_udpipe: dict[str, Any]) -> None:
    model = _model_file(tmp_path)

    get_segmenter(Segmenter.UDPIPE, language="eng_Latn", model=model)

    assert Path(fake_udpipe["loaded"]) == model


# -- what gets recorded ------------------------------------------------------


@pytest.mark.parametrize(
    ("segmenter", "fixture", "name"),
    [
        (Segmenter.UDPIPE, "fake_udpipe", "english.udpipe"),
        (Segmenter.STANZA, "fake_stanza", "en.pt"),
    ],
)
def test_the_recorded_version_digests_the_model(
    segmenter: Segmenter,
    fixture: str,
    name: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """§7.1: record the segmenter model version, not a treebank release.

    A package version says nothing about where a word ends, and a model file
    name can be anything. The digest is the only identity that cannot be wrong.
    """
    request.getfixturevalue(fixture)
    model = _model_file(tmp_path, name)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()

    adapter = get_segmenter(segmenter, language="eng_Latn", model=model)

    assert adapter.model_version is not None
    assert digest in adapter.model_version
    assert name in adapter.model_version
    assert adapter.segmenter is segmenter


# -- segmentation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("segmenter", "fixture"),
    [(Segmenter.UDPIPE, "fake_udpipe"), (Segmenter.STANZA, "fake_stanza")],
)
def test_it_returns_the_model_boundaries_and_not_whitespace(
    segmenter: Segmenter, fixture: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Never falls back to whitespace (D6). The fakes split the full stop off,
    which whitespace does not, so a fallback cannot pass."""
    request.getfixturevalue(fixture)
    adapter = get_segmenter(segmenter, language="eng_Latn", model=_model_file(tmp_path))

    words = adapter.segment(_SENTENCE)

    assert words == ("The", "cat", "sat", "on", "the", "mat", ".")
    assert "mat." not in words
    assert "" not in words


@pytest.mark.parametrize("segmenter", [Segmenter.UDPIPE, Segmenter.STANZA])
def test_a_missing_extra_names_the_package(
    segmenter: Segmenter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither ships in a core install, and neither may fall back to another
    segmenter when absent."""
    for name in ("stanza", "ufal", "ufal.udpipe"):
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(SegmenterUnavailableError) as caught:
        get_segmenter(segmenter, language="eng_Latn", model=_model_file(tmp_path))

    assert "segmenters" in str(caught.value)


def test_a_file_udpipe_cannot_read_is_refused(
    tmp_path: Path, fake_udpipe: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Model.load`` returns ``None`` for a file that is not a model — it does
    not raise. Returning an adapter around that would segment every document
    into nothing and report the result as a measurement."""
    # Reached through sys.modules rather than imported: the real package is not
    # installed here, and a bare import would fail type checking as well as
    # collection on a core install.
    udpipe = sys.modules["ufal.udpipe"]
    monkeypatch.setattr(udpipe.Model, "load", staticmethod(lambda path: None))

    with pytest.raises(FileNotFoundError, match="could not read"):
        get_segmenter(Segmenter.UDPIPE, language="eng_Latn", model=_model_file(tmp_path))
