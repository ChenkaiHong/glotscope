from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))


# -- the trained segmenters, faked ------------------------------------------
#
# Real Stanza and UDPipe models are hundreds of megabytes and downloading one is
# exactly what the adapters forbid, so the third-party modules are stood in for.
# Shared here rather than kept beside the adapter tests because ``verify`` has to
# regenerate a result computed under them, and a fixture only one module can see
# would leave that path untested.


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
