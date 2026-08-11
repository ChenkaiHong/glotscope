"""Metric results that carry their own comparability parameters.

The design rule: **any metric with a comparability hazard returns a struct that
carries the parameters defining its comparability.** Renyi efficiency carries its
alpha and normalizer, Gini carries its language set, parity carries its reference
language. That makes §8.2's promise — ``compare`` refuses to table results
computed under different parameters — mechanical rather than aspirational, and it
makes the parameter impossible to lose in transit.

The second rule: a metric that must never be reported as one number does not
expose one. :class:`StrrPair` has no single ``value``, and
:class:`~glotscope.aggregate.BoundaryCounts` has no recall without precision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from glotscope.aggregate import BoundaryCounts
from glotscope.enums import RenyiNormalizer, Segmenter, TypologicalScope
from glotscope.errors import IncomparableError

__all__ = [
    "Comparable",
    "CompressionResult",
    "CorpusMetrics",
    "FertilityResult",
    "GiniResult",
    "LanguageMetrics",
    "MorphologyResult",
    "ParityResult",
    "RenyiResult",
    "StrrPair",
    "require_comparable",
    "sorted_costs",
]


@runtime_checkable
class Comparable(Protocol):
    """A result that knows what makes it comparable to another of its kind."""

    def comparability_key(self) -> Mapping[str, object]:
        """Parameters that must match for two results to share a table."""
        ...


def require_comparable(left: Comparable, right: Comparable) -> None:
    """Raise :class:`IncomparableError` unless two results are on the same scale.

    This is the enforcement point for PRD §7.1 rule 3, §7.4 and §8.2. It fails on
    the *first* differing field, sorted for determinism, so the error message
    names one concrete cause rather than a diff.
    """
    left_key = left.comparability_key()
    right_key = right.comparability_key()
    for field in sorted(set(left_key) | set(right_key)):
        left_value = left_key.get(field)
        right_value = right_key.get(field)
        if left_value != right_value:
            raise IncomparableError(field, left_value, right_value)


@dataclass(frozen=True, slots=True)
class FertilityResult:
    """Fertility and continuation rate for one language (PRD §7.1).

    A within-language, cross-tokenizer diagnostic. For cross-lingual claims use
    :class:`ParityResult` instead — the docs must say this above the fold, because
    Ahia et al. and Arnett et al. abandoned fertility entirely over the
    segmentation problem.
    """

    language: str
    fertility: float
    p_continued: float
    segmenter: Segmenter
    segmenter_model_version: str | None
    """``None`` only for :attr:`Segmenter.UD_GOLD` and :attr:`Segmenter.WHITESPACE`,
    which read annotation or apply no model. Everything else pins a model
    version, not a treebank release."""

    leading_space: bool
    n_zero_length_words: int
    """Non-zero means normalization stripped words to nothing; the fertility
    denominator choice becomes load-bearing and a warning is required (§12.2)."""

    unk_char_rate: float
    """Fraction of input characters mapping to ``[UNK]``. Above 0.10 the language
    is excluded per Petrov's convention (§7.1 rule 6) — without this,
    UNK-collapsing tokenizers fake good scores, and FlanT5 fails it for 42% of
    FLORES-200 languages."""

    def comparability_key(self) -> Mapping[str, object]:
        return {
            "segmenter": self.segmenter,
            "segmenter_model_version": self.segmenter_model_version,
            "leading_space": self.leading_space,
        }


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """The compression family for one language (PRD §7.2).

    Never present :attr:`cpt` or :attr:`bpt` as a cross-script efficiency
    comparison: UTF-8 charges 1 byte for ASCII and 3 for CJK, so BPT structurally
    favours CJK. And never claim compression predicts quality — Goldman et al.
    report Pearson -0.71 to -0.996 varying only training-corpus size, while
    Schmidt et al. varying algorithm across 54 models find +0.241 and an
    inverted U.
    """

    language: str
    cpt: float
    """Characters per token, ratio of totals."""

    bpt: float
    """UTF-8 bytes per token, ratio of totals. Preferred over CPT for
    portability, because "character" is itself ambiguous."""

    ctc: int
    """Corpus token count."""

    compression_rate: float
    """TokEval's ``CR``: a mean of per-segment ratios, and therefore *not* the
    ratio of the means. By Jensen's inequality it over-weights short segments."""

    compression_rate_unit: str
    """``"chars"`` or ``"bytes"``. Recorded because the source's normalization
    unit is the highest-risk ambiguity in §7 — a form circulating without a
    numerator is dimensionally wrong, and must be resolved from TokEval's source
    before the spec freeze."""

    def comparability_key(self) -> Mapping[str, object]:
        return {"compression_rate_unit": self.compression_rate_unit}


@dataclass(frozen=True, slots=True)
class ParityResult:
    """Corpus-level parity against a reference language (PRD §7.3, D7).

    Ratio of means, equivalently ratio of corpus totals — not the mean of
    per-sentence ratios. They differ, and only the ratio of means maps to actual
    API cost.

    Note the notation discipline: ``premium_{A|B}`` is the per-sentence-pair ratio
    and appears only when discussing the definition. This class is
    ``parity_l``, the corpus-level quantity everything downstream computes.
    """

    reference_language: str
    per_language: Mapping[str, float]
    macro_parity: float
    worst_case_language: str
    worst_case_parity: float
    """``max_l parity_l``. Reported alongside the English-relative figures because
    giving only the first is the standard way to understate the problem."""

    n_lines_per_language: Mapping[str, int]
    """Asserted equal across languages: the ratio-of-means equals
    ratio-of-totals identity breaks if a line is dropped for one language and not
    another (§12.2)."""

    def comparability_key(self) -> Mapping[str, object]:
        return {
            "reference_language": self.reference_language,
            "languages": frozenset(self.per_language),
        }


@dataclass(frozen=True, slots=True)
class GiniResult:
    """Cross-lingual cost inequality (PRD §7.4).

    Scale-invariant, so always report it jointly with compression as a Pareto
    pair. Reference values: classical byte-level BPE 0.064, UnigramLM 0.094,
    parity-aware BPE 0.011 at 30 languages and 128k vocab.
    """

    value: float
    languages: frozenset[str]
    """Gini depends on the language set and is **not comparable across different
    sets** — :meth:`comparability_key` is what enforces that."""

    cost_unit: str
    """Always ``"tokens_per_line"``. Tokens-per-byte favours CJK and
    tokens-per-word reintroduces the segmentation problem, so the unit is the
    whole ballgame."""

    def comparability_key(self) -> Mapping[str, object]:
        return {"languages": self.languages, "cost_unit": self.cost_unit}


@dataclass(frozen=True, slots=True)
class RenyiResult:
    """Renyi efficiency (PRD §7.5, D8, D17).

    **Supplementary, never headline.** Cognetta et al. give two constructions
    that provably raise efficiency while lowering BLEU (DUPLICATION BPE: 0.444 to
    0.612 efficiency, 33.94 to 31.60 BLEU), and Schmidt et al. find it correlates
    at -0.891 with CTC — nearly redundant with compression. The docs must cite
    both counterexamples next to the metric.
    """

    value: float
    alpha: float
    """Required explicitly, never defaulted: the paper says 2.5, Cognetta et al.
    report the optimum as 2.7, and the reference implementation defaults to 3.0.
    ``alpha == 1`` is special-cased to Shannon entropy, since ``1/(1-alpha)``
    diverges."""

    normalizer: RenyiNormalizer
    entropy_bits: float
    """The unnormalized ``H_alpha``. Kept because Foroutan et al. report entropy
    in bits under the label "Renyi Entropy (alpha=2.5)", which therefore cannot
    serve as an efficiency reference."""

    def comparability_key(self) -> Mapping[str, object]:
        return {"alpha": self.alpha, "normalizer": self.normalizer}


@dataclass(frozen=True, slots=True)
class StrrPair:
    """Single-token retention rate under both leading-space conventions (§7.6).

    Deliberately exposes no single ``value``. The convention can move STRR by
    tens of points and the source paper does not specify it, so glotscope refuses
    to emit one unqualified number.

    Known confound to document rather than fix: the word lists are
    English-frequency lists *translated*, so they are not per-language frequency
    lists. Hindi scores lowest across every tokenizer tested despite 128k-255k
    vocabularies.
    """

    language: str
    bare: float
    leading_space: float
    lowercased: bool
    """Casing is a recorded parameter."""

    n_words: int

    def comparability_key(self) -> Mapping[str, object]:
        return {"lowercased": self.lowercased, "n_words": self.n_words}


@dataclass(frozen=True, slots=True)
class MorphologyResult:
    """All three morphological-alignment measures side by side (PRD §7.7).

    Presented as a *descriptive linguistic property*, never a quality proxy:
    Arnett et al. report that morphological alignment "does not explain very much
    variance in model performance", and Arnett & Bergen found no significant
    correlation with perplexity, F(1,13)=0.323, p=0.580.
    """

    language: str
    scope: TypologicalScope
    morphscore_v1: float | None
    """Binary accuracy on one annotated stem-suffix boundary, excluding
    single-token words. Backward comparability only."""

    morphscore_v2: BoundaryCounts | None
    """Boundary and subword P/R/F1 over 70 languages."""

    full_alignment: BoundaryCounts | None
    """Full-alignment P/R/F1 against MorphyNet — the measure Poelman et al. argue
    for and which nobody has shipped. Scores *all* morpheme boundaries including
    suffix-suffix."""

    frequency_weighted: bool
    """Recorded parameter with no default: it changes tokenizer rankings and the
    v2 paper explicitly could not choose a default (§7.7 rule 4)."""

    include_single_token_words: bool
    """Likewise a recorded parameter with no default."""

    def __post_init__(self) -> None:
        measures = (self.morphscore_v1, self.morphscore_v2, self.full_alignment)
        if self.scope is TypologicalScope.OUT_OF_SCOPE:
            if any(m is not None for m in measures):
                raise ValueError(
                    f"{self.language!r} is OUT_OF_SCOPE for morphological alignment "
                    f"but carries measure values. Semitic root-and-pattern morphology "
                    f"has no linear boundary to score and isolating languages lack "
                    f"affixation; returning a number here is the artifact §7.7 rule 2 "
                    f"exists to refuse."
                )
        elif all(m is None for m in measures):
            raise ValueError(f"{self.language!r} is IN_SCOPE but carries no measure values")

    def comparability_key(self) -> Mapping[str, object]:
        return {
            "frequency_weighted": self.frequency_weighted,
            "include_single_token_words": self.include_single_token_words,
        }


@dataclass(frozen=True, slots=True)
class LanguageMetrics:
    """Every per-language Tier 1 result for one language.

    Optional members are ``None`` when the metric was not requested or the corpus
    could not support it. ``None`` here always means "not computed", never "zero"
    — a metric that could not be computed raises at request time rather than
    landing a placeholder.
    """

    language: str
    compression: CompressionResult
    fertility: FertilityResult | None = None
    strr: StrrPair | None = None
    morphology: MorphologyResult | None = None
    parity_vs_reference: float | None = None
    roundtrip_rate: float | None = None


@dataclass(frozen=True, slots=True)
class CorpusMetrics:
    """Tier 1 results defined over the whole language set rather than one language."""

    gini: GiniResult | None = None
    renyi: RenyiResult | None = None
    parity: ParityResult | None = None


def sorted_costs(costs: Sequence[float]) -> tuple[float, ...]:
    """Ascending sort for the Gini computation (PRD §7.4).

    Extracted and named because the ascending-sort sign error is the most common
    implementation bug in this formula, and because the order-invariance property
    test does *not* catch it: the formula sorts internally, so a descending sort
    is equally order-invariant and simply returns the negation. The
    ``value in [0, 1]`` range check is what catches it.
    """
    return tuple(sorted(costs))
