"""CoNLL-U as gold word boundaries (PRD §7.1 rule 1, §10.3).

:attr:`~glotscope.enums.Segmenter.UD_GOLD` is not a segmentation model and must
never be conflated with ``UDPIPE`` or ``STANZA``. Those predict boundaries on
arbitrary text using a model with its own per-language accuracy and its own
version; this reads boundaries a human annotated, which exist only for the
sentences inside a treebank. §7.1 rule 1 makes that distinction a refusal —
requesting ``UD_GOLD`` on FLORES+ raises rather than quietly predicting.

The consequence for this module is that a "gold segmenter" is a **lookup**, not
a function of the string. What it produces is a table from sentence text to the
words annotated for that text, and every entry satisfies one invariant:

    the words, re-joined under the annotation's own spacing, spell the text

Fertility divides by a word count (§7.1). A sentence whose word list does not
reconstruct its own text describes a different string from the one the tokenizer
encodes, and a ratio between the two has no referent. So sentences failing that
check are dropped and **counted** — the discipline :mod:`glotscope.morphynet`
already applies to canonical segmentations.

Three CoNLL-U constructs each offer a way to miscount words while still
producing a plausible figure:

===================  =========================================================
Construct            Why it is not one word each
===================  =========================================================
``1-2  al``          A multiword token. ``al`` is the surface word; the ``a``
                     and ``el`` beneath it are syntactic words no tokenizer
                     ever sees. Counting both inflates the denominator for
                     every language that writes contractions.
``5.1  _``           An empty node — an ellipsis in the enhanced graph. It has
                     no surface form at all.
``SpaceAfter=No``    Governs the reconstruction. Ignoring it inserts spaces the
                     sentence never had, and then every punctuated sentence
                     fails the invariant above for a reason that is not real.
===================  =========================================================
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from glotscope.errors import CorpusIntegrityError

__all__ = ["GoldSentence", "GoldSentences", "parse_conllu"]

CORPUS_ID = "universal_dependencies"
"""The registry id, so a refusal names the resource the way §9 does."""

_COLUMNS = 10
"""ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC."""

_ID, _FORM, _MISC = 0, 1, 9

_COMMENT = "#"
_TEXT_PREFIX = "# text ="
_RANGE = "-"
_EMPTY_NODE = "."
_NO_SPACE_AFTER = "SpaceAfter=No"
_MISC_SEPARATOR = "|"


@dataclass(frozen=True, slots=True)
class GoldSentence:
    """One annotated sentence and the surface words it was segmented into."""

    text: str
    words: tuple[str, ...]
    """Surface words in order. A multiword token appears once, as the token —
    never as the syntactic words beneath it."""


@dataclass(frozen=True, slots=True)
class GoldSentences:
    """Gold segmentation for one treebank, with what it cost to get it.

    The counters are not diagnostics. Fertility computed over the usable subset
    describes that subset, and a reader who is not told how much was dropped
    will read it as a statement about the treebank.
    """

    treebank: str
    sentences: tuple[GoldSentence, ...]
    """Sentences whose words reconstruct their own text, in file order."""

    n_sentences: int
    """Sentences read, before any drop."""

    n_text_mismatch: int
    """Sentences whose words do not spell their own ``# text``."""

    @property
    def by_text(self) -> Mapping[str, tuple[str, ...]]:
        """Sentence text to its words — the segmenter's lookup table.

        A text carrying two different segmentations is dropped rather than
        resolved. Korean treebanks disagree with one another — Kaist segments
        morphologically, GSD by eojeol — so "first one seen" would make
        fertility depend on the order of the file rather than on the annotation.
        """
        readings: dict[str, set[tuple[str, ...]]] = {}
        for sentence in self.sentences:
            readings.setdefault(sentence.text, set()).add(sentence.words)
        return MappingProxyType(
            {text: next(iter(seen)) for text, seen in readings.items() if len(seen) == 1}
        )

    @property
    def n_ambiguous(self) -> int:
        """Texts carrying two or more conflicting segmentations."""
        distinct = {sentence.text for sentence in self.sentences}
        return len(distinct) - len(self.by_text)

    @property
    def coverage(self) -> float:
        """Usable share of the sentences read. 0.0 over none."""
        if self.n_sentences == 0:
            return 0.0
        return len(self.sentences) / self.n_sentences

    def warning(self) -> str:
        """The §9 warnings entry naming what was dropped and why."""
        return (
            f"{self.treebank}: {len(self.sentences)} of {self.n_sentences} CoNLL-U "
            f"sentences carry gold word boundaries that spell their own text "
            f"({self.coverage:.2%}), giving {len(self.by_text)} distinct sentences "
            f"to segment. {self.n_text_mismatch} do not reconstruct their '# text' "
            f"and {self.n_ambiguous} texts carry conflicting segmentations. "
            f"Fertility below describes that usable subset, not the treebank."
        )


def _fields(row: str, treebank: str) -> Sequence[str]:
    fields = row.split("\t")
    if len(fields) != _COLUMNS:
        raise CorpusIntegrityError(
            CORPUS_ID,
            f"a CoNLL-U row for {treebank!r} has {_COLUMNS} tab-separated columns "
            f"(ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC); "
            f"found {len(fields)}",
        )
    return fields


def _no_space_after(misc: str) -> bool:
    return _NO_SPACE_AFTER in misc.split(_MISC_SEPARATOR)


def _covered(token_id: str) -> range:
    """The syntactic word ids a multiword range subsumes."""
    start, _, end = token_id.partition(_RANGE)
    return range(int(start), int(end) + 1)


def _sentence(block: Sequence[str], treebank: str) -> GoldSentence | None:
    """Build one sentence, or ``None`` when its words do not spell its text."""
    declared: str | None = None
    words: list[str] = []
    spaced: list[bool] = []
    covered: set[int] = set()

    for row in block:
        if row.startswith(_COMMENT):
            if row.startswith(_TEXT_PREFIX):
                declared = row[len(_TEXT_PREFIX) :].strip()
            continue
        fields = _fields(row, treebank)
        token_id = fields[_ID]
        if _EMPTY_NODE in token_id:
            continue
        if _RANGE in token_id:
            covered.update(_covered(token_id))
        elif int(token_id) in covered:
            continue
        words.append(fields[_FORM])
        spaced.append(not _no_space_after(fields[_MISC]))

    if not words:
        return None

    rebuilt = "".join(
        word + (" " if follows and index < len(words) - 1 else "")
        for index, (word, follows) in enumerate(zip(words, spaced, strict=True))
    )
    if declared is not None and declared != rebuilt:
        return None
    return GoldSentence(text=rebuilt if declared is None else declared, words=tuple(words))


def parse_conllu(lines: Iterable[str], *, treebank: str = CORPUS_ID) -> GoldSentences:
    """Parse a ``.conllu`` file into gold word boundaries.

    Args:
        lines: rows of a CoNLL-U file, blank lines separating sentences.
        treebank: recorded on the result and named in every refusal. Record the
            treebank rather than "UD": §10.3 notes that UD Korean treebanks
            disagree among themselves about what a word is.

    Returns:
        The usable subset, with the counts that describe it.

    Raises:
        CorpusIntegrityError: if a token row does not have ten columns, or if no
            sentence survives — a gold segmenter with an empty table reports
            every document as un-annotated, which reads as a finding about the
            corpus rather than about the file it was handed.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        row = line.rstrip("\n")
        if not row.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(row)
    if current:
        blocks.append(current)

    kept: list[GoldSentence] = []
    n_sentences = 0
    n_text_mismatch = 0
    for block in blocks:
        if all(row.startswith(_COMMENT) for row in block):
            # Comments with no token rows are metadata, not a sentence.
            continue
        n_sentences += 1
        sentence = _sentence(block, treebank)
        if sentence is None:
            n_text_mismatch += 1
            continue
        kept.append(sentence)

    if not kept:
        raise CorpusIntegrityError(
            CORPUS_ID,
            f"no usable gold segmentation for {treebank!r} in {n_sentences} "
            f"sentences: {n_text_mismatch} have words that do not spell their own "
            f"'# text'. Fertility divides by a word count, so a word list "
            f"describing a different string cannot be scored against a tokenization",
        )

    return GoldSentences(
        treebank=treebank,
        sentences=tuple(kept),
        n_sentences=n_sentences,
        n_text_mismatch=n_text_mismatch,
    )
