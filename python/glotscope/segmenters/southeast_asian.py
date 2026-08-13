"""Thai and Khmer segmentation (PRD §10.3).

Both scripts run words together without spaces, so whitespace segmentation
returns roughly one "word" per sentence and fertility for these languages
becomes a measurement of sentence length. Both adapters are scoped to their
language for the same reason the East Asian ones are: outside it they still
return something.
"""

from __future__ import annotations

from dataclasses import dataclass

from glotscope.enums import Segmenter
from glotscope.segmenters._support import import_or_refuse, package_version, require_scope

__all__ = [
    "KHMER",
    "THAI",
    "KhmerNltkSegmenter",
    "PyThaiNlpSegmenter",
    "load_khmer_nltk",
    "load_pythainlp",
]

THAI = frozenset({"tha", "th"})
KHMER = frozenset({"khm", "km"})

_DEFAULT_THAI_ENGINE = "newmm"
"""PyThaiNLP's default word-tokenization engine. Named explicitly and recorded,
because the engine — not just the package version — decides the boundaries, and
a result that does not say which one ran cannot be reproduced."""


@dataclass(frozen=True, slots=True)
class PyThaiNlpSegmenter:
    """PyThaiNLP word tokenization."""

    package: str
    engine: str = _DEFAULT_THAI_ENGINE
    segmenter: Segmenter = Segmenter.PYTHAINLP

    @property
    def model_version(self) -> str:
        return f"pythainlp {self.package} (engine: {self.engine})"

    def segment(self, text: str) -> tuple[str, ...]:
        tokenize = import_or_refuse(Segmenter.PYTHAINLP, "pythainlp.tokenize", "pythainlp")
        words = tokenize.word_tokenize(text, engine=self.engine)
        return tuple(word for word in words if word.strip())


@dataclass(frozen=True, slots=True)
class KhmerNltkSegmenter:
    """khmer-nltk word tokenization, which is a bundled CRF model."""

    package: str
    segmenter: Segmenter = Segmenter.KHMER_NLTK

    @property
    def model_version(self) -> str:
        return f"khmer-nltk {self.package} (bundled CRF model)"

    def segment(self, text: str) -> tuple[str, ...]:
        khmernltk = import_or_refuse(Segmenter.KHMER_NLTK, "khmernltk", "khmer-nltk")
        return tuple(word for word in khmernltk.word_tokenize(text) if word.strip())


def load_pythainlp(language: str) -> PyThaiNlpSegmenter:
    """Build the PyThaiNLP segmenter, refusing any language but Thai.

    Raises:
        SegmenterScopeError: for a non-Thai language.
        SegmenterUnavailableError: if pythainlp is not installed.
    """
    require_scope(Segmenter.PYTHAINLP, language, THAI)
    import_or_refuse(Segmenter.PYTHAINLP, "pythainlp.tokenize", "pythainlp")
    return PyThaiNlpSegmenter(package=package_version("pythainlp"))


def load_khmer_nltk(language: str) -> KhmerNltkSegmenter:
    """Build the khmer-nltk segmenter, refusing any language but Khmer.

    Raises:
        SegmenterScopeError: for a non-Khmer language.
        SegmenterUnavailableError: if khmer-nltk is not installed.
    """
    require_scope(Segmenter.KHMER_NLTK, language, KHMER)
    import_or_refuse(Segmenter.KHMER_NLTK, "khmernltk", "khmer-nltk")
    return KhmerNltkSegmenter(package=package_version("khmer-nltk"))
