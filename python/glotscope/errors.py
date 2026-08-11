"""Typed refusals.

The PRD's central claim is that this library refuses to emit a number it cannot
justify, rather than silently computing a meaningless one (§6, D5). Each error
here corresponds to a specific class of wrong result that existing tools in this
space produce quietly.

Nothing in this module is a fallback path. If an error here is ever caught and
replaced with a default value, that is the bug the error exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "CapabilityError",
    "GlotscopeError",
    "IncomparableError",
    "NoReferenceSetError",
    "SegmenterRequiredError",
    "TokenizerLoadError",
    "UnsupportedCheckpointError",
]


class GlotscopeError(Exception):
    """Base class for every refusal glotscope raises deliberately."""


class TokenizerLoadError(GlotscopeError):
    """A tokenizer could not be loaded from the source it was asked for.

    Typed rather than left as a bare ``OSError`` or JSON failure so that a
    leaderboard run can distinguish "this row could not be loaded" from "this
    row produced a number", and never conflate the two. The offending source is
    carried on the exception; it is deliberately not copied into the manifest,
    which must contain no filesystem paths (§9).
    """

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(f"tokenizer {source!r} could not be read: {reason}")


class CapabilityError(GlotscopeError):
    """A metric was requested that the corpus cannot support (PRD §6, D5).

    Gating is on corpus *capabilities*, never corpus identity: requesting parity
    on a monolingual corpus raises this rather than returning a number computed
    against a reference that does not exist.
    """

    def __init__(
        self,
        metric: str,
        required: str,
        corpus_id: str,
        available: Iterable[str],
    ) -> None:
        self.metric = metric
        self.required = required
        self.corpus_id = corpus_id
        self.available = tuple(sorted(available))
        super().__init__(
            f"{metric!r} requires a corpus declaring {required!r}, but "
            f"{corpus_id!r} declares {list(self.available)}. This is a refusal, "
            f"not a missing feature: the number would not mean anything."
        )


class SegmenterRequiredError(GlotscopeError):
    """A word-level metric was requested without a segmenter (PRD §7.1, D6).

    There is no default segmenter and there will not be one. ``W(D)`` is the
    load-bearing choice in fertility and the field silently disagrees on it;
    a default would manufacture exactly the incomparability this library exists
    to surface.
    """

    def __init__(self, metric: str) -> None:
        self.metric = metric
        super().__init__(
            f"{metric!r} is word-level and requires an explicit segmenter. "
            f"Pass segmenter=Segmenter.<X> to analyze(). There is no default "
            f"(PRD §7.1 rule 2, D6): whitespace segmentation is degenerate for "
            f"Chinese, Japanese, Thai, Khmer, Lao and Tibetan."
        )


class IncomparableError(GlotscopeError):
    """Results computed under different parameters were asked to share a table.

    Raised by comparison APIs when the operands' comparability parameters differ
    (PRD §7.1 rule 3, §7.4, §8.2). Segmenter, leading-space convention,
    normalization form, Renyi alpha and normalizer, and the evaluated language
    set all make results incomparable.
    """

    def __init__(self, field: str, left: object, right: object) -> None:
        self.field = field
        self.left = left
        self.right = right
        super().__init__(
            f"cannot compare results with differing {field}: {left!r} vs {right!r}. "
            f"These numbers are not on the same scale, and tabling them together "
            f"would be the error this check exists to prevent. Recompute one side "
            f"with matching parameters."
        )


class NoReferenceSetError(GlotscopeError):
    """The Tier 2 reference-set fallback chain was exhausted (PRD §7.9).

    The chain is: unused-token entries, then embedding rows above ``|V|``, then
    single bytes 0xF5-0xFF. Link three is convenient, not universal — tokenizers
    exist that include exactly the 243 bytes UTF-8 uses and omit the rest, and
    StarCoder2 additionally misses 0xF1. For those checkpoints the cosine
    indicator has no reference at all.

    Never fall through to an empty mean. Callers may degrade to ``L2(E_in)``
    alone with a warning when embeddings are untied; tied checkpoints cannot be
    analysed at all without a reference set.
    """

    def __init__(self, checkpoint: str, *, tied: bool) -> None:
        self.checkpoint = checkpoint
        self.tied = tied
        remedy = (
            "tied embeddings leave no alternative indicator; this checkpoint "
            "cannot be analysed at Tier 2"
            if tied
            else "degrade to the L2(E_in) indicator alone and emit a warning"
        )
        super().__init__(
            f"no reference set found for {checkpoint!r}: no unused-token entries, "
            f"no padding rows above |V|, and no unused single-byte tokens. "
            f"Remedy: {remedy}."
        )


class UnsupportedCheckpointError(GlotscopeError):
    """The checkpoint cannot support Tier 2 (PRD §8.1, §14.2).

    Most importantly: quantized weights. A 4-bit ``E_in`` destroys the L2-norm
    indicator, which is the exact signal Tier 2 depends on. Community mirrors
    routinely republish 4-bit, GGUF or merged variants under near-identical
    names, so this is a hard refusal rather than a warning.
    """

    def __init__(self, checkpoint: str, reason: str, *, dtype: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.reason = reason
        self.dtype = dtype
        detail = f" (dtype={dtype})" if dtype is not None else ""
        super().__init__(
            f"{checkpoint!r} cannot be used for Tier 2{detail}: {reason}. "
            f"Tier 2 requires original-dtype embedding tensors."
        )
