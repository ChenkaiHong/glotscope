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

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from glotscope.aggregate import DocumentStats
from glotscope.enums import (
    Capability,
    Confidence,
    Indicator,
    RenyiNormalizer,
    Segmenter,
    TokenClass,
    TokenizerFamily,
)
from glotscope.errors import CapabilityError, SegmenterRequiredError
from glotscope.manifest import CorpusManifest, Manifest, ParameterManifest, canonical_json
from glotscope.metrics import gini as _gini
from glotscope.metrics import parity as _parity
from glotscope.metrics import renyi_efficiency_from_counts as _renyi_from_counts
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


def _no_stats() -> Mapping[str, DocumentStats]:
    """The empty default, as a factory rather than as a shared constant.

    A read-only mapping rather than a fresh ``dict``: these reports are frozen,
    and a mutable default would be the one way to change a result after it was
    produced. It has to be a ``default_factory`` because ``dataclasses`` on
    Python 3.10 and 3.11 rejects any default whose ``__hash__`` is ``None``, and
    ``mappingproxy``'s is — 3.12 relaxed the check to list/dict/set only.
    Assigning the constant directly imports fine on 3.13 and fails half the CI
    matrix at import time, which is the shape of bug the 12-cell matrix exists
    to catch.
    """
    return MappingProxyType({})


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
    byte_fallback_coverage: int | None
    """How many of the 256 byte values the vocabulary contains. ``None`` means
    byte fallback is not applicable for this code-point tokenizer. Distinguishes the
    full 256 from the 243 that UTF-8 actually uses — the gap is a Tier 2
    reference-set source, and StarCoder2 additionally misses 0xF1."""

    non_utf8_byte_values: tuple[int, ...] | None = None
    """Present bytes in the 0xF5-0xFF range; ``None`` when not applicable."""

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
            "non_utf8_byte_values": list(self.non_utf8_byte_values)
            if self.non_utf8_byte_values is not None
            else None,
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

    document_stats: Mapping[str, DocumentStats] = field(default_factory=_no_stats)
    """Folded statistics per language, kept so the corpus-level metrics can be
    *computed* rather than looked up. ``alpha`` is chosen at call time in §8.1,
    and a report storing only a finished Renyi number could not answer a second
    alpha without re-encoding the corpus."""

    corpus: CorpusManifest | None = None
    """The §9 corpus block, which is also what the capability gate reads. One
    field rather than a loose identifier beside a loose capability set: gating is
    on capability and never on corpus identity (D5), and a report that can name
    what was missing must have been told by the corpus itself."""

    parameters: ParameterManifest | None = None
    """Every contested parameter the run was performed under (§9). Carried on the
    report because these are properties of the *analysis*, and a caller
    assembling a :class:`~glotscope.manifest.Manifest` from a report should not
    have to restate them — a second copy in user code is a second thing to drift.

    Renyi's alpha and normalizer are absent here on purpose: they are chosen per
    call in §8.1, and :class:`~glotscope.results.RenyiResult` carries the ones it
    was computed under."""

    def __post_init__(self) -> None:
        parameters = self.parameters
        if parameters is not None and parameters.segmenter is not self.segmenter:
            raise ValueError(
                f"the report's segmenter ({self.segmenter}) and its recorded "
                f"parameters ({parameters.segmenter}) disagree. The segmenter "
                f"appears twice by necessity — here because it gates the fertility "
                f"refusal, and in the parameter block because that is what gets "
                f"published — so disagreement is a construction error rather than "
                f"something a reader has to notice."
            )

    @property
    def corpus_id(self) -> str:
        return self.corpus.id if self.corpus is not None else ""

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.corpus.capabilities if self.corpus is not None else frozenset()

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

        Computed as a **ratio of means**, not a mean of per-sentence ratios (D7):
        only the ratio of means maps to what an API actually bills.

        Raises:
            CapabilityError: if the corpus does not declare ``parallel``.
            ValueError: if the report carries no folded statistics.
        """
        if Capability.PARALLEL not in self.capabilities:
            raise CapabilityError(
                "parity",
                Capability.PARALLEL.value,
                self.corpus_id or "<unidentified corpus>",
                (capability.value for capability in self.capabilities),
            )
        stats = self._require_stats("parity")
        token_counts = {
            language: language_stats.per_document_tokens
            for language, language_stats in stats.items()
        }
        return _parity(token_counts, reference=reference)

    def gini(self) -> GiniResult:
        """Cross-lingual cost inequality over the evaluated language set (§7.4).

        The cost unit is tokens per aligned line, so the comparison is only
        meaningful across a fixed language set — a Gini computed over a different
        set is a different number and :class:`GiniResult` records the set for
        exactly that reason.
        """
        stats = self._require_stats("gini")
        if Capability.PARALLEL not in self.capabilities:
            raise CapabilityError(
                "gini",
                Capability.PARALLEL.value,
                self.corpus_id or "<unidentified corpus>",
                (capability.value for capability in self.capabilities),
            )
        if len(stats) < 2:
            raise ValueError("gini requires at least two parallel languages")
        line_counts = {language: value.n_documents for language, value in stats.items()}
        if len(set(line_counts.values())) != 1:
            raise ValueError(f"parallel languages have unequal line counts: {line_counts}")
        costs = {
            language: language_stats.total_tokens / language_stats.n_documents
            for language, language_stats in stats.items()
            if language_stats.n_documents
        }
        if not costs:
            raise ValueError("gini requires at least one language with at least one line")
        return _gini(costs)

    def renyi_efficiency(
        self,
        alpha: float,
        normalizer: RenyiNormalizer = RenyiNormalizer.OBSERVED,
        nominal_vocab_size: int | None = None,
    ) -> RenyiResult:
        """Renyi efficiency at an explicit ``alpha`` (PRD §7.5, D17).

        ``alpha`` has no default on purpose: calling the reference implementation
        without specifying it silently uses 3.0 and reproduces nobody.

        Computed from the stored type-count distributions, merged across the
        evaluated languages, so a second ``alpha`` costs a fold rather than a
        re-encode. Ship the result as supplementary, never headline (D8): two
        published constructions raise Renyi efficiency while lowering BLEU.
        """
        stats = self._require_stats("renyi_efficiency")
        merged: Counter[int] = Counter()
        for language_stats in stats.values():
            merged.update(language_stats.type_counts)
        return _renyi_from_counts(
            merged,
            alpha=alpha,
            normalizer=normalizer,
            nominal_vocab_size=nominal_vocab_size,
        )

    def _require_stats(self, metric: str) -> Mapping[str, DocumentStats]:
        if not self.document_stats:
            raise ValueError(
                f"{metric!r} needs the folded corpus statistics, and this report "
                f"carries no stored document statistics"
            )
        return self.document_stats

    def to_dict(self) -> dict[str, Any]:
        """Assemble the §9 ``tier1`` block.

        A metric that was not computed is *absent*, never zero: ``None`` means
        "not computed", and a zero would tabulate beside real measurements as
        though it were one.
        """
        per_language: dict[str, Any] = {}
        for language, language_metrics in self.per_language.items():
            compression = language_metrics.compression
            entry: dict[str, Any] = {
                "cpt": compression.cpt,
                "bpt": compression.bpt,
                "ctc": compression.ctc,
                "compression_rate": compression.compression_rate,
                "compression_rate_unit": compression.compression_rate_unit,
            }
            if language_metrics.fertility is not None:
                entry["fertility"] = language_metrics.fertility.fertility
                # §7.1 defines P_cont beside fertility, and the two answer
                # different questions: a tokenizer can hold fertility down while
                # splitting most words, and only P_cont shows it.
                entry["p_continued"] = language_metrics.fertility.p_continued
            if language_metrics.strr is not None:
                entry["strr_bare"] = language_metrics.strr.bare
                entry["strr_leading_space"] = language_metrics.strr.leading_space
            if language_metrics.parity_vs_reference is not None:
                entry["parity_vs_reference"] = language_metrics.parity_vs_reference
            if language_metrics.roundtrip_rate is not None:
                entry["roundtrip_rate"] = language_metrics.roundtrip_rate
            per_language[language] = entry

        corpus_level: dict[str, Any] = {}
        if self.corpus_level.gini is not None:
            corpus_level["gini"] = self.corpus_level.gini.value
        if self.corpus_level.renyi is not None:
            corpus_level["renyi_efficiency"] = self.corpus_level.renyi.value
            corpus_level["renyi_alpha"] = self.corpus_level.renyi.alpha
            corpus_level["renyi_normalizer"] = self.corpus_level.renyi.normalizer.value
            if self.corpus_level.renyi.nominal_vocab_size is not None:
                corpus_level["renyi_nominal_vocab_size"] = (
                    self.corpus_level.renyi.nominal_vocab_size
                )
        if self.corpus_level.parity is not None:
            parity = self.corpus_level.parity
            corpus_level["parity"] = {
                "reference_language": parity.reference_language,
                "macro_parity": parity.macro_parity,
                "worst_case_language": parity.worst_case_language,
                "worst_case_parity": parity.worst_case_parity,
                "per_language": dict(parity.per_language),
            }

        return {
            "per_language": per_language,
            "corpus_level": corpus_level,
            "segmenter": self.segmenter.value if self.segmenter is not None else None,
        }


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
