"""Closed vocabularies.

Every value here is serialized verbatim into ``manifest.json`` (PRD §9), so the
string values are part of the on-disk schema. Changing one is a schema break and
requires bumping ``schema_version``.

All of these are ``str`` enums so that ``json.dumps`` handles them without a
custom encoder and so manifest round-tripping is exact.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Algorithm",
    "Backend",
    "Capability",
    "Confidence",
    "Indicator",
    "MorphologicalType",
    "Normalization",
    "ReferenceSource",
    "RenyiNormalizer",
    "Segmenter",
    "TokenClass",
    "TokenizerFamily",
    "TypologicalScope",
]


class Segmenter(str, Enum):
    """Word-segmentation convention (PRD §7.1 rule 1).

    There is deliberately no default member and no ``DEFAULT`` alias. ``W(D)`` is
    the single largest source of silent incomparability in this literature (D6).

    ``UD_GOLD`` is *not* the same operation as ``UDPIPE``/``STANZA`` and must not
    be conflated with them. UD supplies gold word boundaries only for sentences
    inside the treebanks; applying "UD segmentation" to arbitrary text requires a
    trained model with its own per-language accuracy and its own version. Note
    also that UD Korean treebanks disagree among themselves — Kaist segments
    morphologically, GSD by eojeol — so record the treebank, not just "UD".
    """

    UD_GOLD = "ud_gold"
    UDPIPE = "udpipe"
    STANZA = "stanza"
    MECAB = "mecab"
    JIEBA = "jieba"
    PYTHAINLP = "pythainlp"
    KHMER_NLTK = "khmer_nltk"
    ICU = "icu"
    WHITESPACE = "whitespace"
    """Provided for comparability with existing tools only. Never a default."""

    @property
    def requires_gold_segmentation(self) -> bool:
        """Whether this segmenter reads annotation rather than predicting it.

        ``UD_GOLD`` is legal only on a corpus declaring
        :attr:`Capability.WORD_SEGMENTATION`; requesting it on FLORES+ raises
        :class:`~glotscope.errors.CapabilityError`.
        """
        return self is Segmenter.UD_GOLD


class Capability(str, Enum):
    """What a corpus can support (PRD §6, D5).

    Tier 1 gates on capabilities, not corpus identity. This single decision
    prevents the most common category of wrong result in this space.
    """

    PARALLEL = "parallel"
    WORD_SEGMENTATION = "word_segmentation"
    MORPH_GOLD = "morph_gold"
    WORDLIST = "wordlist"


class TokenClass(str, Enum):
    """UTF-8 classification of a vocabulary entry (PRD §7.8).

    These three classes are **disjoint**, and the sources nest them in a way that
    breaks downstream counts if collapsed. ``IllFormedVocab`` is the union of
    ``PARTIAL_UTF8`` and ``ILL_FORMED_NOT_PARTIAL``; Stage 1 of §7.9 excludes
    only ``PARTIAL_UTF8``. Handing the full ill-formed set to Stage 1
    over-excludes and the published candidate-set denominators stop reproducing.
    """

    PARTIAL_UTF8 = "partial_utf8"
    """Holds part of a UTF-8 encoding form. Land & Bartolo's class."""

    ILL_FORMED_NOT_PARTIAL = "ill_formed_not_partial"
    """Ill-formed but contains no part of any valid form: C0, C1, F5-FF."""

    WELL_FORMED = "well_formed"

    @property
    def is_ill_formed(self) -> bool:
        """Whether this class counts toward ``IllFormedVocab(T)``."""
        return self in (TokenClass.PARTIAL_UTF8, TokenClass.ILL_FORMED_NOT_PARTIAL)


class TokenizerFamily(str, Enum):
    """Byte-handling family (PRD §7.8).

    Reported alongside a quantitative ill-formed rate rather than as a binary
    alarm. UTF-8 Plumbing's Theorem 2 applies to byte-level and byte-fallback
    equally, so a predicate that is true of the entire §11 roster carries no
    information; the rate is what a user can act on.
    """

    BYTE_LEVEL = "byte_level"
    BYTE_FALLBACK = "byte_fallback"
    CODE_POINT = "code_point"


class Algorithm(str, Enum):
    """Tokenizer construction algorithm, recorded in the manifest (PRD §9)."""

    BYTE_LEVEL_BPE = "byte_level_bpe"
    BYTE_FALLBACK_BPE = "byte_fallback_bpe"
    UNIGRAM_LM = "unigram_lm"
    WORDPIECE = "wordpiece"
    TIKTOKEN = "tiktoken"
    BYTE = "byte"
    """Degenerate byte- or character-level model, e.g. ByT5 (vocab 384)."""

    UNKNOWN = "unknown"
    """Recorded honestly rather than guessed. Tier 2 scope limits key off this:
    §7.9 is validated on BPE only, and Unigram-LM models are untested."""


class ReferenceSource(str, Enum):
    """Which link of the §7.9 fallback chain supplied ``t_ref``.

    Recorded rather than inferred, because the three links are not equally
    strong evidence. An ``<unused…>`` entry is the vocabulary *stating* that a
    token was never trained; a row above ``|V|`` is padding by inference; an
    unused byte is a token that exists and merely should never appear in
    well-formed text. A reader comparing two checkpoints needs to know which
    yardstick each number was measured against.
    """

    UNUSED_TOKENS = "unused_tokens"
    PADDING_ROWS = "padding_rows"
    UNUSED_BYTES = "unused_bytes"


class MorphologicalType(str, Enum):
    """Morphological type of a language (PRD §10.2)."""

    FUSIONAL = "fusional"
    AGGLUTINATIVE = "agglutinative"
    NON_CONCATENATIVE = "non_concatenative"
    ISOLATING = "isolating"


class TypologicalScope(str, Enum):
    """Whether morphological alignment is meaningful for a language (PRD §7.7).

    MorphScore covers only fusional and agglutinative languages. Semitic
    root-and-pattern morphology (``kataba``/``kātib``) has no linear boundary to
    score; isolating languages lack affixation.

    glotscope returns ``OUT_OF_SCOPE`` where the reference implementation
    publishes a number — Arnett et al.'s prose excludes these languages but their
    released tables include Hebrew and Mandarin anyway, the latter at precision
    0.98 / recall 1.00, which is a single-token artifact rather than
    morphological alignment. That divergence is deliberate and belongs in
    ``docs/divergences.md``.
    """

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"

    @classmethod
    def for_type(cls, morphology: MorphologicalType) -> TypologicalScope:
        in_scope = (MorphologicalType.FUSIONAL, MorphologicalType.AGGLUTINATIVE)
        return cls.IN_SCOPE if morphology in in_scope else cls.OUT_OF_SCOPE


class RenyiNormalizer(str, Enum):
    """Denominator for Renyi efficiency (PRD §7.5 landmine 2, D17).

    Zouhar writes ``H_0 = log|Delta|``, but that identity holds only under full
    support. The reference implementation uses the *observed* type count, so
    ``OBSERVED`` is the default: it is what the only published implementation
    does, and ``NOMINAL`` would match no reference value in §12.1.

    Pass ``NOMINAL`` explicitly for cross-tokenizer comparison at fixed ``|V|``.
    """

    OBSERVED = "observed"
    NOMINAL = "nominal"


class Normalization(str, Enum):
    """Unicode normalization form, recorded because it shifts results (PRD §8.1).

    NFKC vs NFC changes CPT, STRR and round-trip losslessness.
    """

    NFC = "NFC"
    NFD = "NFD"
    NFKC = "NFKC"
    NFKD = "NFKD"
    NONE = "none"


class Indicator(str, Enum):
    """Tier 2 under-training indicator (PRD §7.9).

    Low values imply under-trained in both cases. The cosine direction is
    verified correct in Appendix B: low distance to the mean *unused* embedding
    means under-trained, and there is no sign error.
    """

    COSINE_TO_UNUSED_MEAN = "cosine_to_unused_mean"
    """``C(E_out, u_ref)``. Requires a reference set. The only indicator
    available for tied embeddings."""

    L2_E_IN = "l2_e_in"
    """``||E_in,i||``. Untied embeddings only; needs no reference set, but per
    D10 it is never run alone except as a documented degradation."""


class Confidence(str, Enum):
    """Tier 2 result confidence (PRD §8.1, D10).

    ``LOW_CONFIDENCE`` when the two indicators disagree beyond threshold. Applied
    weight decay is frequently undocumented, so measuring agreement between
    indicators is better than assuming it either way.
    """

    HIGH = "HIGH"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class Backend(str, Enum):
    """Compute backend, recorded in every manifest (PRD §9, §13 rule 5)."""

    PYTHON = "python"
    RUST = "rust"
