"""The gold-annotation segmenter (PRD §7.1 rule 1, §10.3).

Every other adapter here predicts: hand it a string and it applies a rule or a
model. This one cannot, and the difference is normative rather than an
implementation detail. ``UD_GOLD`` reports boundaries a human annotated, and
those exist only for the sentences inside a treebank — so it is a **lookup**,
and a document nobody annotated has no gold segmentation rather than a guessable
one.

That is why a miss refuses instead of falling back to whitespace. A fallback
would mix annotated and predicted boundaries inside a single fertility number,
which is the conflation §7.1 rule 1 exists to prevent and which nothing in the
published result would reveal.

``model_version`` reports the **treebank**. §7.1's instruction to record a model
version rather than a treebank release is aimed at ``UDPIPE`` and ``STANZA``,
which have a model; this has none, so the treebank release *is* the provenance
of these boundaries. §10.3 requires exactly that: UD Korean treebanks disagree
among themselves — Kaist segments morphologically, GSD by eojeol — so "UD" alone
does not identify what was measured.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from glotscope.enums import Segmenter
from glotscope.errors import CorpusIntegrityError

__all__ = ["UdGoldSegmenter"]

_PREVIEW = 60
"""Characters of the offending document quoted in a refusal — enough to find the
line, short enough not to paste a paragraph into a traceback."""


@dataclass(frozen=True, slots=True)
class UdGoldSegmenter:
    """Gold word boundaries read out of a treebank, keyed by sentence text."""

    gold: Mapping[str, tuple[str, ...]]
    """Sentence text to the words annotated for it, from
    :attr:`~glotscope.conllu.GoldSentences.by_text`."""

    treebank: str
    """Treebank identifier and release, e.g. ``UD_English-EWT 2.18``. Recorded
    as the model version, because it is what produced these boundaries."""

    segmenter: Segmenter = Segmenter.UD_GOLD

    @property
    def model_version(self) -> str | None:
        """The treebank, since no model produced these boundaries."""
        return self.treebank

    def segment(self, text: str) -> tuple[str, ...]:
        """The annotated words for ``text``.

        Raises:
            CorpusIntegrityError: if ``text`` was never annotated. Predicting a
                fallback would put two segmentation conventions inside one
                fertility number with nothing in the result to show it.
        """
        try:
            return self.gold[text]
        except KeyError:
            raise CorpusIntegrityError(
                "universal_dependencies",
                f"{self.treebank} carries no gold segmentation for a document it "
                f"was asked to segment ({text[:_PREVIEW]!r}). UD_GOLD reads "
                f"annotation rather than predicting it, so an un-annotated "
                f"document has no gold answer, and a fallback here would mix "
                f"annotated and predicted boundaries inside one fertility number",
            ) from None
