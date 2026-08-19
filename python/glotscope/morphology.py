"""Morphological alignment — three measures side by side (PRD §7.7).

    (a) MorphScore v1   binary accuracy on one annotated stem-suffix boundary
    (b) MorphScore v2   P/R/F1 over that same single boundary
    (c) full alignment  P/R/F1 over *all* morpheme boundaries, suffix-suffix
                        included — the measure Poelman, Bauwens & de Lhoneux
                        argue for and which nobody has shipped

What separates (c) from (b) is one thing: which gold boundaries are in the
reference set. MorphScore looks at the stem-suffix boundary alone, so
``araba/lari`` against gold ``araba/lar/i`` scores a perfect 1.0 under (a) and
(b) while missing a boundary that (c) counts. For agglutinative languages that
omission is most of the morphology.

**Never report recall without precision** (D11). Accuracy and recall reward
oversegmentation — a character-level tokenizer scores perfect recall, and
``gathered`` split into eight characters is exactly that case. The return type
makes precision non-optional, so this is enforced by the shape of the data
rather than by a warning somebody can ignore.

**This is a descriptive linguistic property, never a quality proxy.** Arnett et
al. report that morphological alignment "does not explain very much variance in
model performance", and Arnett & Bergen found no significant correlation with
perplexity (F(1,13)=0.323, p=0.580). The null result travels with the number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from glotscope.aggregate import BoundaryCounts, align_boundaries
from glotscope.enums import MorphologicalType, TypologicalScope
from glotscope.results import MorphologyResult

__all__ = ["AlignedWord", "boundaries", "morphology", "pieces_from_offsets"]


def boundaries(pieces: Sequence[str]) -> frozenset[int]:
    """Character offsets where one piece ends and the next begins.

    The cumulative lengths, excluding the last: the end of the word is not a
    boundary between anything, and counting it would hand every tokenizer one
    free true positive per word.
    """
    offsets: list[int] = []
    running = 0
    for piece in pieces[:-1]:
        running += len(piece)
        offsets.append(running)
    return frozenset(offsets)


def pieces_from_offsets(
    word: str,
    offsets: Sequence[tuple[int, int]],
    *,
    shift: int = 0,
) -> tuple[str, ...] | None:
    """Slice a word into the pieces a tokenizer's character offsets describe.

    Offsets rather than decoded tokens, because decoding is where the two
    families break in opposite directions: a byte-level vocabulary spells a space
    ``Ġ`` and a byte-fallback vocabulary spells one byte ``<0xNN>``, and neither
    string's length is the word's character offset.

    ``tokenizers`` reports offsets in **characters** into the input it was given,
    and tokens covering less than one character all report that character's span.
    So a two-byte Cyrillic character encoded as two byte tokens yields one piece
    and one empty one; the empty piece is dropped, and no boundary is claimed
    inside a character. That is deliberate: counting sub-character splits as
    boundaries is what earns a byte-fallback tokenizer artificially high recall
    on non-Latin scripts (§7.7 rule 1), and it would be a claim about UTF-8
    rather than about morphology.

    Args:
        word: the word the offsets index into.
        offsets: ``(start, end)`` per token, in token order.
        shift: characters prepended before encoding — 1 under the leading-space
            convention (§7.1 rule 5), which shifts every offset by one.

    Returns:
        The pieces, which concatenate to ``word``, or ``None`` if the offsets do
        not tile it. ``None`` is a refusal rather than a fallback: a pretokenizer
        that reorders or drops spans cannot be scored as boundaries, and the
        caller counts what it dropped.
    """
    if not offsets:
        return None
    length = len(word)
    cut = 0
    pieces: list[str] = []
    for _, raw_end in offsets:
        end = min(max(raw_end - shift, 0), length)
        if end < cut:
            return None
        piece = word[cut:end]
        if piece:
            pieces.append(piece)
        cut = end
    if cut != length:
        return None
    return tuple(pieces)


@dataclass(frozen=True, slots=True)
class AlignedWord:
    """One word, segmented two ways: by the gold annotation and by the tokenizer.

    Both are sequences of pieces that concatenate back to the same string. That
    is checked rather than assumed — boundaries are offsets, and offsets into two
    different strings produce scores that look entirely reasonable and mean
    nothing.

    Tokens are the *decoded* pieces. A byte-level vocabulary's ``Ġ`` markers and
    a byte-fallback vocabulary's ``<0xNN>`` spellings are the tokenizer's
    internal representation, and their lengths are not the word's character
    offsets.
    """

    morphemes: tuple[str, ...]
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        gold = "".join(self.morphemes)
        predicted = "".join(self.tokens)
        if gold != predicted:
            raise ValueError(
                f"morphemes and tokens must spell the same word: "
                f"{gold!r} against {predicted!r}. Boundaries are character "
                f"offsets, so offsets into two different strings are not "
                f"comparable and the scores computed from them are meaningless."
            )

    @property
    def is_single_token(self) -> bool:
        """Whether the tokenizer emitted the whole word as one piece."""
        return len(self.tokens) == 1

    @property
    def gold_boundaries(self) -> frozenset[int]:
        return boundaries(self.morphemes)

    @property
    def predicted_boundaries(self) -> frozenset[int]:
        return boundaries(self.tokens)

    @property
    def stem_suffix_boundary(self) -> frozenset[int]:
        """The single boundary MorphScore annotates, as a set of zero or one.

        The first one: MorphScore scores the stem-suffix split, and everything
        after it is the suffix-suffix structure that §7.7(c) exists to see.
        """
        gold = sorted(self.gold_boundaries)
        return frozenset(gold[:1])


def _accuracy(words: Sequence[AlignedWord]) -> float | None:
    """MorphScore v1: did the tokenizer put a boundary where the annotation is?

    Binary per word, averaged. Single-token words are excluded here regardless of
    what the caller asked for, because §7.7(a) defines the measure that way —
    honouring the flag would compute something that is not MorphScore and publish
    it under MorphScore's name.

    ``None`` when no word qualifies, which is not the same as 0.0: one says
    nothing was measurable, the other says nothing aligned.
    """
    scorable = [word for word in words if not word.is_single_token and word.gold_boundaries]
    if not scorable:
        return None
    hits = sum(1 for word in scorable if word.stem_suffix_boundary <= word.predicted_boundaries)
    return hits / len(scorable)


def _counts(
    predicted: Sequence[Sequence[int]],
    gold: Sequence[Sequence[int]],
) -> BoundaryCounts:
    """Micro-aggregate through the §13 boundary, which is where the fold lives.

    Counts are summed across the batch and the ratio taken once. Averaging
    per-word F1 would weight a one-morpheme word as heavily as a six-morpheme
    one, which is exactly the comparison agglutinative languages exist to break.
    """
    return align_boundaries(predicted, gold)


def morphology(
    words: Sequence[AlignedWord],
    *,
    language: str,
    morphological_type: MorphologicalType,
    frequency_weighted: bool,
    include_single_token_words: bool,
) -> MorphologyResult:
    """Score all three measures for one language (PRD §7.7).

    Args:
        words: gold segmentation paired with the tokenizer's, per word.
        language: the code recorded in the result.
        morphological_type: decides scope. Semitic root-and-pattern morphology
            has no linear boundary to score and isolating languages lack
            affixation, so both return :attr:`TypologicalScope.OUT_OF_SCOPE`
            carrying no numbers — **even though the reference implementation
            publishes numbers there** (a deliberate divergence, logged in
            ``docs/divergences.md``).
        frequency_weighted: whether ``words`` is a stream of occurrences rather
            than a list of types. Recorded, not applied: glotscope weights
            nothing here, and the flag records which the caller passed rather
            than asserting anything about it. §7.7 rule 4 makes it a recorded
            parameter with no default, because it changes tokenizer rankings and
            the v2 paper explicitly could not choose one.
        include_single_token_words: whether words the tokenizer emitted whole
            count. They predict no boundaries, so including them scores every
            gold boundary as a miss — a property of the vocabulary rather than
            of the alignment. Also a recorded parameter with no default.

    Raises:
        ValueError: if ``words`` is empty and the language is in scope. An F1 of
            0.0 over nothing reads as "this tokenizer aligns nothing", which is a
            claim about the tokenizer rather than about the input. An
            out-of-scope language returns before this check, since no batch would
            change the answer.
    """
    # Scope first, and deliberately before the empty check: it is a property of
    # the language, not of the batch. Semitic root-and-pattern morphology has
    # nothing to score however many words arrive, so demanding a non-empty batch
    # before saying so would make the caller tokenize a corpus to be told the
    # measure does not apply to it.
    scope = TypologicalScope.for_type(morphological_type)
    if scope is TypologicalScope.OUT_OF_SCOPE:
        return MorphologyResult(
            language=language,
            morphological_type=morphological_type,
            scope=scope,
            morphscore_v1=None,
            morphscore_v2=None,
            full_alignment=None,
            frequency_weighted=frequency_weighted,
            include_single_token_words=include_single_token_words,
        )

    if not words:
        raise ValueError(
            f"{language!r}: no words to score. An F1 of 0.0 over an empty batch "
            f"reads as a finding about the tokenizer when it is a fact about "
            f"the input."
        )

    scored = [word for word in words if include_single_token_words or not word.is_single_token]
    predicted = [sorted(word.predicted_boundaries) for word in scored]
    return MorphologyResult(
        language=language,
        morphological_type=morphological_type,
        scope=scope,
        morphscore_v1=_accuracy(words),
        morphscore_v2=_counts(predicted, [sorted(w.stem_suffix_boundary) for w in scored]),
        full_alignment=_counts(predicted, [sorted(w.gold_boundaries) for w in scored]),
        frequency_weighted=frequency_weighted,
        include_single_token_words=include_single_token_words,
    )
