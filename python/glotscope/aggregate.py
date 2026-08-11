"""The v1-to-v2 FFI boundary (PRD §13, D3).

Every function in this module is batch-oriented: it takes whole sequences of
integers and returns a frozen struct. None takes a ``str``, and none is intended
to be called inside a per-document or per-word loop.

**This is a v1 constraint, not a v2 one.** §13 is explicit that v1 must be
written against the boundary v2 will implement — per-call FFI crossings would
erase the speedup entirely, and discovering that after the metric layer is
written means rewriting the metric layer.

Rules for anything added here:

1. Inputs are sequences of ints, or parallel sequences of ints. Never ``str``.
2. One call handles the whole batch. Callers pre-collect, then call once.
3. Returns an immutable struct, never a mutable accumulator the caller updates.

What is deliberately *not* on this boundary: the Tier 2 embedding sweeps. A
cosine pass over a 250k x 4096 matrix is already vectorised in numpy and gains
nothing from crossing into Rust. The genuinely loop-bound work — per-word
aggregation, per-token script attribution, morpheme boundary alignment — is what
lives here, and it is what §13 means by "the aggregation is not fast".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "BoundaryCounts",
    "DocumentStats",
    "WordStats",
    "aggregate_documents",
    "aggregate_words",
    "align_boundaries",
    "attribute_scripts",
]


@dataclass(frozen=True, slots=True)
class DocumentStats:
    """Document-level totals, sufficient for the whole compression family.

    Holds everything §7.2 and §7.5 need from a single pass, so no metric has to
    re-encode the corpus:

    * ``CTC`` is :attr:`total_tokens` (PRD §7.2).
    * ``CPT``/``BPT`` are :attr:`total_chars`/:attr:`total_bytes` over
      :attr:`total_tokens` — ratios of totals, not means of ratios.
    * ``CR`` (TokEval's compression rate) is total measured units over total
      tokens after invalid records are excluded. Its default unit is UTF-8 bytes,
      making the default CR numerically identical to BPT.
    * Renyi efficiency needs :attr:`type_counts`, the token-type frequency
      distribution.
    """

    n_documents: int
    total_tokens: int
    total_chars: int
    total_bytes: int
    type_counts: Mapping[int, int]
    """Token id to occurrence count. Read-only; the distribution ``p_Delta`` in
    §7.5 is derived from this."""

    per_document_tokens: tuple[int, ...]
    """``|tau(d)|`` per document, in input order. Required by ``CR``; a sum-only
    struct cannot reconstruct it."""

    n_empty_documents: int
    """Documents whose encoding was empty. Counted, never silently dropped —
    normalization can strip a document to nothing (U+00AD, U+200B, some ZWJ and
    RTL marks) and §12.2 requires these be reported separately."""


@dataclass(frozen=True, slots=True)
class WordStats:
    """Word-level counts for fertility, continuation rate and STRR.

    Requires a segmenter upstream: ``W(D)`` is the load-bearing choice and there
    is no default (PRD §7.1 rule 2, D6). This struct does not know which
    segmenter produced its input, which is precisely why the segmenter must be
    recorded in the manifest by the caller.
    """

    n_words: int
    """All segmented words, *including* those that encoded to zero tokens."""

    total_tokens: int
    n_continued: int
    """``|tau(w)| >= 2``. Divided by :attr:`n_words` this is ``P_cont`` (§7.1)."""

    n_single_token: int
    """``|tau(w)| == 1``. The basis of STRR (§7.6). Note STRR is defined over a
    word *list*, type-level, not over corpus tokens."""

    n_zero_length: int
    """``|tau(w)| == 0``. The naive invariant "fertility >= 1.0" is false in the
    presence of these, so §12.2 requires they be counted and reported separately
    rather than dropped. The metric layer must state whether its fertility
    denominator includes them and emit a warning when this is non-zero."""


@dataclass(frozen=True, slots=True)
class BoundaryCounts:
    """Confusion counts for morphological alignment (PRD §7.7).

    Precision is structurally non-optional (D11): there is no way to read recall
    off this struct without precision also being available. Accuracy and recall
    reward oversegmentation — a character-level tokenizer scores perfect recall,
    and Llama2's 0.56 recall against 0.13 precision is misinformation if only the
    first is reported.
    """

    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        """Predicted boundaries that are correct.

        Convention: 0.0 when nothing was predicted. Undefined in principle, but
        0.0 is the standard convention and keeps the return type non-optional.
        """
        predicted = self.true_positive + self.false_positive
        if predicted == 0:
            return 0.0
        return self.true_positive / predicted

    @property
    def recall(self) -> float:
        """Gold boundaries that were found. 0.0 when there are no gold boundaries."""
        actual = self.true_positive + self.false_negative
        if actual == 0:
            return 0.0
        return self.true_positive / actual

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        if p + r == 0.0:
            return 0.0
        return 2 * p * r / (p + r)


def aggregate_documents(
    encodings: Sequence[Sequence[int]],
    char_lengths: Sequence[int],
    byte_lengths: Sequence[int],
) -> DocumentStats:
    """Fold a batch of document encodings into :class:`DocumentStats`.

    ``char_lengths`` and ``byte_lengths`` are passed in as parallel int arrays
    rather than recomputed from text, so that no string crosses the boundary.
    The caller measures them once on the normalized text.

    Character counts are ambiguous by nature — codepoint versus grapheme cluster,
    and Devanagari combining marks are separate codepoints — so §7.2 prefers BPT
    over CPT for portability. Whichever the caller measured, it must record the
    unit in the manifest.

    Args:
        encodings: one token-id sequence per document.
        char_lengths: character count per document, same length and order.
        byte_lengths: UTF-8 byte count per document, same length and order.

    Raises:
        ValueError: if the three sequences have differing lengths. A dropped
            document would silently corrupt every ratio downstream, and §12.2
            requires equal line counts across languages for the
            ratio-of-means-equals-ratio-of-totals identity to hold.
    """
    raise NotImplementedError


def aggregate_words(encodings: Sequence[Sequence[int]]) -> WordStats:
    """Fold a batch of per-word encodings into :class:`WordStats`.

    One entry per segmented word, in corpus order. The leading-space convention
    (``tau("the")`` versus ``tau(" the")``) is applied by the caller before this
    point and recorded in the manifest: it can move STRR by tens of points
    (§7.6), and the source paper does not specify it.
    """
    raise NotImplementedError


def attribute_scripts(
    token_ids: Sequence[int],
    script_ids: Sequence[int],
) -> Mapping[int, int]:
    """Count vocabulary entries per Unicode script (PRD §14.3 step 3, D14).

    Script attribution is the *primary* method for the paper, not corpus
    attribution. Under-trained tokens are by construction tokens the model
    essentially never saw, and FLORES+ devtest is clean translated prose, so
    corpus attribution against it returns approximately zero for every language
    and makes the correlation undefined. Gemma 2B's confirmed Devanagari
    candidate illustrates it: script attribution finds it, FLORES+ Hindi never
    emits it.

    ``script_ids`` are resolved by the caller from the decoded token via UAX #24,
    with ``Common`` and ``Inherited`` resolved through Script_Extensions. Passing
    integer script codes rather than names keeps the boundary int-only.

    Args:
        token_ids: vocabulary entries under consideration.
        script_ids: script code per entry, same length and order.

    Returns:
        Script code to entry count. ``UTR_l`` is defined over this partition —
        denominator = all script-attributed vocabulary entries, independent of
        corpus emission — which is what decouples it from ``parity_l``'s
        numerator.
    """
    raise NotImplementedError


def align_boundaries(
    predicted: Sequence[Sequence[int]],
    gold: Sequence[Sequence[int]],
) -> BoundaryCounts:
    """Score predicted morpheme boundaries against gold, over a whole batch.

    Boundaries are character offsets within a word. Full alignment means *all*
    morpheme boundaries including suffix-suffix, which is what MorphScore omits
    and what matters for agglutinative languages (PRD §7.7(c)).

    Two calibration notes, both load-bearing:

    * ``gathered`` tokenized ``g/a/t/h/e/r/e/d`` must yield boundary-F1 0.25
      against MorphScore's 1.0. That reproduces exactly and is a regression test.
    * ``arabalari`` tokenized ``araba/lari`` against gold ``araba/lar/i`` does
      *not* reproduce Poelman et al.'s reported 0.5 under either standard reading
      — boundary-F1 gives 0.667, span-F1 gives 0.4. Their exact definition must
      be recovered from ``LAGoM-NLP/ConfoundingFactors`` before this is gated;
      until then it is a documented divergence, not a test.

    Raises:
        ValueError: if ``predicted`` and ``gold`` have differing lengths.
    """
    raise NotImplementedError


def _count_types(encodings: Sequence[Sequence[int]]) -> Mapping[int, int]:
    """Token-type frequency over a batch, as a read-only mapping.

    Kept separate so the Rust implementation has one obvious thing to replace,
    and so the Python reference stays a single pass.
    """
    counter: Counter[int] = Counter()
    for encoding in encodings:
        counter.update(encoding)
    return MappingProxyType(dict(counter))
