"""Tier reports and the ``Report`` object that spans them (PRD §6, §8.1, G2).

G2 is the requirement that a *single* ``Report`` holds vocabulary lint, corpus
fairness metrics, and weight-based under-training indicators for one
(tokenizer, checkpoint) pair. No competing tool spans that range, and this class
is where the claim is either true or not.

Two structural decisions here encode PRD traps that comments alone would not:

* :meth:`Tier0Report.stage1_exclusions` unions exactly the three sets §7.9 Stage 1
  excludes. The full ill-formed set is deliberately not reachable from a method
  named for the exclusion, because handing it to Stage 1 over-excludes and the
  published candidate-set denominators stop reproducing.
* :attr:`Tier1Report.fertility` is a property that raises when no segmenter was
  supplied, so a caller cannot get a fertility number without having chosen one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glotscope.enums import (
    Confidence,
    Indicator,
    RenyiNormalizer,
    Segmenter,
    TokenClass,
    TokenizerFamily,
)
from glotscope.errors import SegmenterRequiredError
from glotscope.manifest import Manifest, canonical_json
from glotscope.results import (
    CorpusMetrics,
    GiniResult,
    LanguageMetrics,
    ParityResult,
    RenyiResult,
)

__all__ = [
    "Report",
    "Tier0Report",
    "Tier1Report",
    "Tier2Report",
    "TokenCandidate",
]


@dataclass(frozen=True, slots=True)
class Tier0Report:
    """Vocabulary introspection. Tokenizer only, milliseconds (PRD §7.8, §6)."""

    vocab_size: int
    family: TokenizerFamily
    ill_formed_vocab_rate: float
    """``IllFormedVocab(T)``: the union of ``PARTIAL_UTF8`` and
    ``ILL_FORMED_NOT_PARTIAL`` over ``|V|``. Reported as a rate beside the family
    classifier, never as a binary alarm — UTF-8 Plumbing's Theorem 2 is a
    possibility statement true of essentially the entire §11 roster, and a
    predicate true of every input carries no information."""

    token_classes: Mapping[TokenClass, int]
    """Counts per disjoint class. Must partition ``|V|`` exactly."""

    partial_utf8_tokens: tuple[int, ...]
    unreachable_tokens: tuple[int, ...]
    """``encode(decode(id)) != [id]``."""

    special_tokens: tuple[int, ...]
    byte_fallback_coverage: int
    """How many of the 256 byte values the vocabulary contains. Distinguishes the
    full 256 from the 243 that UTF-8 actually uses — the gap is a Tier 2
    reference-set source, and StarCoder2 additionally misses 0xF1."""

    @property
    def unreachable_count(self) -> int:
        return len(self.unreachable_tokens)

    def stage1_exclusions(self) -> frozenset[int]:
        """Token ids excluded before Tier 2 ranking (PRD §7.9 Stage 1).

        Exactly three sets: partial-UTF-8 sequences, unreachable tokens, and
        special tokens. **Not** the full ill-formed set — Firestone et al. state
        the nesting explicitly, partial-UTF-8 is a special case of ill-formed, and
        excluding the wider set breaks the candidate-set reproduction that §14.3's
        validation floor depends on.
        """
        return frozenset(
            (*self.partial_utf8_tokens, *self.unreachable_tokens, *self.special_tokens)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "family": self.family.value,
            "ill_formed_vocab_rate": self.ill_formed_vocab_rate,
            "token_classes": {k.value: v for k, v in sorted(self.token_classes.items())},
            "unreachable_count": self.unreachable_count,
            "byte_fallback_coverage": self.byte_fallback_coverage,
        }


@dataclass(frozen=True, slots=True)
class Tier1Report:
    """Corpus-based metrics. Tokenizer plus corpus (PRD §7.1-§7.7, §6)."""

    per_language: Mapping[str, LanguageMetrics]
    corpus_level: CorpusMetrics
    segmenter: Segmenter | None
    """``None`` is legal and common: parity, Gini, Renyi and the compression
    family are all segmenter-free, and §14.3 step 1 runs parity over all 229
    FLORES+ varieties precisely because it costs nothing. Word-level metrics raise
    instead of defaulting."""

    warnings: tuple[str, ...] = ()

    @property
    def fertility(self) -> Mapping[str, float]:
        """Per-language fertility (PRD §7.1).

        Raises:
            SegmenterRequiredError: if no segmenter was supplied. This is the
                enforcement point for D6 — there is no default, because ``W(D)``
                is the single largest source of silent incomparability in this
                literature.
        """
        if self.segmenter is None:
            raise SegmenterRequiredError("fertility")
        return {
            language: metrics.fertility.fertility
            for language, metrics in self.per_language.items()
            if metrics.fertility is not None
        }

    def parity(self, reference: str) -> ParityResult:
        """Corpus-level parity against ``reference`` (PRD §7.3).

        Raises:
            CapabilityError: if the corpus does not declare ``parallel``.
        """
        raise NotImplementedError

    def gini(self) -> GiniResult:
        """Cross-lingual cost inequality over the evaluated language set (§7.4)."""
        raise NotImplementedError

    def renyi_efficiency(
        self,
        alpha: float,
        normalizer: RenyiNormalizer = RenyiNormalizer.OBSERVED,
    ) -> RenyiResult:
        """Renyi efficiency at an explicit ``alpha`` (PRD §7.5, D17).

        ``alpha`` has no default on purpose: calling the reference implementation
        without specifying it silently uses 3.0 and reproduces nobody.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    """One under-trained-token candidate (PRD §7.9)."""

    token_id: int
    token_repr: str
    """A safe printable rendering. Candidates are frequently mojibake, base64
    fragments or partial byte sequences, so this is for display only and never
    for re-encoding."""

    indicator: Indicator
    indicator_value: float
    """Low implies under-trained for both indicators."""

    rank: int
    token_class: TokenClass
    script: str | None
    """UAX #24 script of the decoded token, with ``Common``/``Inherited`` resolved
    via Script_Extensions. ``None`` when no script applies. This is the attribution
    §14.3 regresses on (D14)."""


@dataclass(frozen=True, slots=True)
class Tier2Report:
    """Weight-based under-training indicators (PRD §7.9, §6).

    Needs only ``E_in`` and optionally ``E_out`` — two tensors, read from
    ``safetensors`` without instantiating the model. Tier 2 being this cheap is
    why unifying it with Tier 1 is practical, and nobody noticing that is the
    opportunity §2.2 identifies.
    """

    candidates: tuple[TokenCandidate, ...]
    indicator: Indicator
    tied: bool
    top_pct: float
    candidates_pre_exclusion: int
    candidates_post_exclusion: int
    confidence: Confidence
    indicator_agreement: float | None = None
    """Spearman rho between the two indicators. ``None`` when embeddings are tied,
    since only the cosine indicator is available. Measuring agreement is better
    than assuming weight decay either way, because applied weight decay is
    frequently undocumented (D10)."""

    first_pc_removed: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator.value,
            "tied": self.tied,
            "indicator_agreement": self.indicator_agreement,
            "confidence": self.confidence.value,
            "candidate_count": self.candidate_count,
            "top_pct": self.top_pct,
            "candidates_pre_exclusion": self.candidates_pre_exclusion,
            "candidates_post_exclusion": self.candidates_post_exclusion,
            "first_pc_removed": self.first_pc_removed,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """One (tokenizer, corpus, checkpoint) result spanning every tier run (G2)."""

    tier0: Tier0Report
    manifest: Manifest
    tier1: Tier1Report | None = None
    tier2: Tier2Report | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Assemble the §9 document.

        The ``warnings`` array aggregates this report's own warnings plus every
        tier's. It is load-bearing, not decorative: any parameter choice known to
        be contested emits one.
        """
        document: dict[str, Any] = self.manifest.to_dict()
        document["tier0"] = self.tier0.to_dict()
        collected = list(self.warnings)
        if self.tier1 is not None:
            document["tier1"] = self.tier1.to_dict()
            collected.extend(self.tier1.warnings)
        if self.tier2 is not None:
            document["tier2"] = self.tier2.to_dict()
            collected.extend(self.tier2.warnings)
        document["warnings"] = collected
        return document

    def to_json(self, path: str | Path) -> None:
        """Write ``result.json`` deterministically (PRD §8.1, G4).

        Uses :func:`~glotscope.manifest.canonical_json` so that ``glotscope
        verify`` can assert bit-identical regeneration. Any nondeterminism
        introduced here — dict ordering, a timestamp, a float format that varies
        by platform — turns the CI job that delivers G4 permanently red.
        """
        Path(path).write_text(canonical_json(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> Report:
        """Load a result document, validating ``schema_version``."""
        raise NotImplementedError
