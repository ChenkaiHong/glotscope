"""Fertility and continuation rate (PRD §7.1).

```
Fertility(T; D) = ( Σ_{w ∈ W(D)} |τ(w)| ) / |W(D)|
P_cont(T; D)    = |{ w ∈ W(D) : |τ(w)| ≥ 2 }| / |W(D)|
```

A **within-language, cross-tokenizer** diagnostic. For cross-lingual claims use
parity (§7.3) instead: fertility compares tokenizers on one language, and the
work that tried to use it across languages — Ahia et al., Arnett et al. —
abandoned it over the segmentation problem this module's inputs make explicit.

Nothing here chooses a segmenter. Words arrive already segmented and already
encoded, because ``W(D)`` is the load-bearing decision and a default would
manufacture exactly the incomparability the library exists to surface (D6).
"""

from __future__ import annotations

from glotscope.aggregate import WordStats
from glotscope.enums import Segmenter
from glotscope.errors import UnkRateExceededError
from glotscope.results import FertilityResult

__all__ = ["UNK_EXCLUSION_THRESHOLD", "fertility"]

UNK_EXCLUSION_THRESHOLD = 0.10
"""Petrov et al.'s convention (§7.1 rule 6): drop a language when more than 10%
of its characters map to ``[UNK]``.

Without it, UNK-collapsing tokenizers fake good scores — every unrepresentable
character collapses into one token, so fertility *falls* as coverage gets worse.
FlanT5 fails this for 42% of FLORES-200 languages."""

_NO_MODEL_NEEDED = frozenset({Segmenter.WHITESPACE, Segmenter.UD_GOLD})
"""The two segmenters that legitimately pin no model version: one applies no
model, the other reads annotation."""


def fertility(
    stats: WordStats,
    *,
    language: str,
    segmenter: Segmenter,
    segmenter_model_version: str | None,
    leading_space: bool,
    unk_char_rate: float,
) -> FertilityResult:
    """Fertility and continuation rate for one language (§7.1).

    Args:
        stats: folded per-word statistics, from
            :func:`~glotscope.aggregate.aggregate_words`.
        language: the language code these numbers describe.
        segmenter: which convention produced ``W(D)``. Recorded, never defaulted.
        segmenter_model_version: what produced the boundaries. ``None`` is legal
            only for ``WHITESPACE`` and ``UD_GOLD``.
        leading_space: whether words were encoded as ``tau(" the")`` rather than
            ``tau("the")``. A recorded parameter: it moves the result and the
            source papers do not state it.
        unk_char_rate: fraction of this language's characters mapping to
            ``[UNK]``.

    Raises:
        UnkRateExceededError: if ``unk_char_rate`` exceeds 10% (§7.1 rule 6).
        ValueError: if ``W(D)`` is empty, if the rate is not a fraction, or if a
            model-backed segmenter pins no version.
    """
    if not 0.0 <= unk_char_rate <= 1.0:
        raise ValueError(f"unk_char_rate must be a fraction in [0, 1], got {unk_char_rate!r}")
    if unk_char_rate > UNK_EXCLUSION_THRESHOLD:
        raise UnkRateExceededError(language, unk_char_rate, UNK_EXCLUSION_THRESHOLD)
    if stats.n_words == 0:
        raise ValueError(
            f"fertility is undefined for {language!r}: the segmenter returned no "
            f"words, so the denominator |W(D)| is zero. An empty result would be "
            f"indistinguishable from a tokenizer that emits nothing"
        )
    if segmenter not in _NO_MODEL_NEEDED and not segmenter_model_version:
        raise ValueError(
            f"the {segmenter.value!r} segmenter applies a model, so its version "
            f"must be recorded (§7.1 rule 1). Only WHITESPACE and UD_GOLD may "
            f"pin none, because they apply no model and read annotation"
        )

    return FertilityResult(
        language=language,
        fertility=stats.total_tokens / stats.n_words,
        p_continued=stats.n_continued / stats.n_words,
        segmenter=segmenter,
        segmenter_model_version=segmenter_model_version,
        leading_space=leading_space,
        # Carried rather than filtered: a normalizer can strip a word to nothing
        # (soft hyphen, ZWSP, some ZWJ and RTL marks), and §12.2 requires those
        # be counted and reported rather than silently dropped — dropping them
        # would change the denominator without saying so.
        n_zero_length_words=stats.n_zero_length,
        unk_char_rate=unk_char_rate,
    )
