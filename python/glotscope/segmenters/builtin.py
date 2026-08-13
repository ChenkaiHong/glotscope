"""The two unscoped segmenters: whitespace and ICU (PRD §10.3).

Unscoped for opposite reasons. ICU implements Unicode text segmentation for
every script, so it is the generic fallback. Whitespace is unscoped because its
failure is the property it is kept for: it is offered *only* for comparability
with tools that use it, and never as a default.
"""

from __future__ import annotations

from dataclasses import dataclass

from glotscope.enums import Segmenter
from glotscope.segmenters._support import import_or_refuse

__all__ = ["IcuSegmenter", "WhitespaceSegmenter", "load_icu", "load_whitespace"]


@dataclass(frozen=True, slots=True)
class WhitespaceSegmenter:
    """``str.split()``, and nothing else.

    Degenerate for Chinese, Japanese, Thai, Khmer, Lao and Tibetan: a whole
    clause becomes one "word" and fertility explodes. TokEval and most
    reimplementations use exactly this, which is why glotscope offers it — a
    number that cannot be compared against the existing literature is its own
    kind of unusable — and why it can never be the default (D6).
    """

    segmenter: Segmenter = Segmenter.WHITESPACE

    @property
    def model_version(self) -> str | None:
        """``None``: no model is applied, so there is no version to pin."""
        return None

    def segment(self, text: str) -> tuple[str, ...]:
        return tuple(text.split())


@dataclass(frozen=True, slots=True)
class IcuSegmenter:
    """ICU word boundaries (UAX #29) for any language.

    The version recorded is ICU's own — the data is what decides where a Thai
    word ends, and it moves independently of the PyICU wrapper around it.
    """

    locale: str
    icu_version: str
    segmenter: Segmenter = Segmenter.ICU

    @property
    def model_version(self) -> str:
        return f"icu {self.icu_version} ({self.locale})"

    def segment(self, text: str) -> tuple[str, ...]:
        icu = import_or_refuse(Segmenter.ICU, "icu", "PyICU")
        iterator = icu.BreakIterator.createWordInstance(icu.Locale(self.locale))
        iterator.setText(text)
        words: list[str] = []
        start = 0
        for boundary in iterator:
            candidate = text[start:boundary]
            # ICU's boundaries include the whitespace runs between words.
            # Dropping the blank spans leaves words; dropping punctuation too
            # would change what W(D) means, so it is kept.
            if candidate.strip():
                words.append(candidate)
            start = boundary
        return tuple(words)


def load_whitespace(language: str) -> WhitespaceSegmenter:
    """Build the whitespace segmenter. Accepts every language, by design."""
    del language
    return WhitespaceSegmenter()


def load_icu(language: str) -> IcuSegmenter:
    """Build the ICU segmenter for ``language``.

    Raises:
        SegmenterUnavailableError: if PyICU is not installed. It needs system
            ICU to build, which is the friction §10.3 warns about.
    """
    icu = import_or_refuse(Segmenter.ICU, "icu", "PyICU")
    return IcuSegmenter(locale=language.split("_", 1)[0], icu_version=str(icu.ICU_VERSION))
