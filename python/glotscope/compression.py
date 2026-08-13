"""The compression family: CPT, BPT, CTC and TokEval's CR (PRD §7.2).

Every quantity here is a **ratio of totals**. The mean of per-document ratios is
a different number whenever documents differ in length, and it is the number an
implementation reaches for by accident — §7.2 records that the earlier
mean-of-per-segment-ratios reading of CR was simply wrong.

Two things this module will not do, because the sources do not support them:

* Present CPT or BPT as a cross-script efficiency comparison. UTF-8 charges one
  byte for ASCII and three for most CJK, so BPT structurally favours CJK. These
  are within-language, cross-tokenizer diagnostics.
* Imply that compression predicts quality. Goldman et al. report Pearson -0.71 to
  -0.996 while varying only training-corpus size; Schmidt et al., varying the
  algorithm across 54 models, find +0.241 and an inverted U.
"""

from __future__ import annotations

from collections.abc import Sequence

from glotscope.aggregate import DocumentStats
from glotscope.results import CompressionResult

__all__ = ["BYTES", "CHARS", "compression"]

BYTES = "bytes"
"""TokEval's default measurement unit, which makes default CR identical to BPT."""

CHARS = "chars"

_UNITS = (BYTES, CHARS)


def compression(
    stats: DocumentStats,
    *,
    unit_lengths: Sequence[int],
    is_blank: Sequence[bool] | None = None,
    language: str,
    unit: str = BYTES,
) -> CompressionResult:
    """Compute the compression family for one language (PRD §7.2).

    ``unit_lengths`` is the per-document measurement in ``unit``, in the order
    the stats were folded from. It is required separately because CR is not
    derivable from :class:`~glotscope.aggregate.DocumentStats` totals alone: its
    exclusion rule is per-record, and a summed struct cannot say which records
    had zero units.

    CR is frozen from TokEval source revision
    ``798063302bc1d96a75fb963bcb91c9aab53ee9a1`` rather than from paper prose
    (U1): skip blank text, zero-unit text and empty tokenizations, accumulate
    measured units and tokens over what survives, then divide the totals.

    Args:
        stats: folded document statistics for this language.
        unit_lengths: per-document measurement in ``unit``, same order and
            length as the batch the stats were folded from.
        language: the language code these numbers describe.
        unit: ``"bytes"`` (TokEval's default) or ``"chars"``. Recorded on the
            result, because byte and character normalization are not
            interchangeable and results measured under different units are not
            comparable.

    Raises:
        ValueError: if ``unit`` is unknown, if ``unit_lengths`` does not match
            the batch, if a measurement is negative, if the corpus produced no
            tokens at all, or if every record was excluded from CR. The last is
            a refusal rather than a ``0.0``: a rate over an empty denominator
            would be a number the corpus does not support.
    """
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {list(_UNITS)}, got {unit!r}")
    if len(unit_lengths) != stats.n_documents:
        raise ValueError(
            f"unit_lengths and the folded batch must be of equal length: "
            f"got {len(unit_lengths)} and {stats.n_documents}"
        )
    if any(length < 0 for length in unit_lengths):
        raise ValueError("unit_lengths cannot be negative")
    if is_blank is not None and len(is_blank) != stats.n_documents:
        raise ValueError(
            f"is_blank and the folded batch must be of equal length: "
            f"got {len(is_blank)} and {stats.n_documents}"
        )
    if stats.total_tokens == 0:
        raise ValueError(
            "compression is undefined for a corpus with no tokens: every ratio "
            "here divides by the token count"
        )

    kept_units = 0
    kept_tokens = 0
    blanks = is_blank if is_blank is not None else (False,) * stats.n_documents
    for units, tokens, blank in zip(unit_lengths, stats.per_document_tokens, blanks, strict=True):
        if blank or units == 0 or tokens == 0:
            continue
        kept_units += units
        kept_tokens += tokens

    if kept_tokens == 0:
        raise ValueError(
            "no records survived the compression-rate exclusion rule: every "
            "document was blank, zero-unit, or tokenized to nothing"
        )

    return CompressionResult(
        language=language,
        cpt=stats.total_chars / stats.total_tokens,
        bpt=stats.total_bytes / stats.total_tokens,
        ctc=stats.total_tokens,
        compression_rate=kept_units / kept_tokens,
        compression_rate_unit=unit,
    )
