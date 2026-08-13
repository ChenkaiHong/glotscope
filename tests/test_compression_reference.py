"""The §7.2 compression family against an independent implementation (§12.1).

The row this closes: *CPT/BPT/CTC on a fixed corpus vs. an independent
implementation, tolerance ±1e-6, dependency none.*

The independent implementation is TokEval — ``cimeister/tokenizer-intrinsic-evals``
at revision ``798063302bc1d96a75fb963bcb91c9aab53ee9a1``, MIT — which is the same
revision §7.2 freezes CR from (U1). The values below were produced by executing
that revision's ``InformationTheoreticMetrics.compute_compression_rate``
(``tokenizer_analysis/metrics/information_theoretic.py``, lines 241-324) on the
corpus and tokenizer defined here, once, in a throwaway environment. Nothing of
upstream is vendored and nothing is fetched at test time, which is what keeps
this row's dependency "none".

**Why the numbers are hard-coded rather than recomputed.** Re-deriving them from
the same source lines that produced ``compression.py`` would be an oracle written
by the mental model it is meant to falsify. A recorded output of the other
implementation is evidence; a second reading of the spec is not.

**Reproducing them.** Install that revision, build the tokenizer below, wrap each
document in upstream's ``TokenizedData(tokenizer_name, language, tokens, text)``,
and call ``compute_compression_rate`` with
``TextMeasurementConfig(method=NormalizationMethod.BYTES)`` and again with
``NormalizationMethod.CHARACTERS``. Its default byte counting is
``ByteCountingMethod.UTF8``, i.e. ``len(text.encode("utf-8"))`` — the
``hf_bytelevel`` path exists but is not the default and is not what §7.2 freezes.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.aggregate import aggregate_documents
from glotscope.compression import BYTES, CHARS, compression
from glotscope.lint import byte_to_unicode

pytestmark = pytest.mark.reference

CORPUS: dict[str, tuple[str, ...]] = {
    "eng_Latn": ("The cat sat.", "It rained.", "   "),
    "hin_Deva": ("बिल्ली बैठी।", "बारिश हुई।", "\t\n "),
    "jpn_Jpan": ("猫が座った。", "雨が降った。"),
}
"""The fixed corpus, byte for byte what upstream was run on.

Three scripts, so bytes and characters genuinely disagree — UTF-8 charges one
byte for ASCII and three for Devanagari and Kana. Two whitespace-only records,
because the blank-text exclusion is the clause the two implementations could
most plausibly differ on, and a corpus without one would not test it.
"""

MERGES: list[tuple[str, str]] = [
    ("à", "¤"),
    ("à", "¥"),
    ("ã", "ģ"),
    ("a", "t"),
    ("t", "Ġ"),
    ("¤", "¬"),
]
"""Six merges over the byte-level alphabet, fixed by hand rather than trained.

Without them the model emits exactly one token per byte, every BPT is exactly
1.0, and a ±1e-6 assertion on 1.0 would pass for any implementation that counts
bytes and tokens at all. These are the most frequent adjacent pairs in the corpus
above and they cover all three scripts, so tokens and bytes disagree everywhere.
Written out rather than trained because a trainer's output is a function of the
``tokenizers`` version, and this is a value pinned to 1e-6.
"""

UPSTREAM_RATE_BYTES: dict[str, float] = {
    "eng_Latn": 1.1,
    "hin_Deva": 1.4761904761904763,
    "jpn_Jpan": 1.2,
}
UPSTREAM_RATE_CHARS: dict[str, float] = {
    "eng_Latn": 1.1,
    "hin_Deva": 0.5238095238095238,
    "jpn_Jpan": 0.4,
}
UPSTREAM_POOLED_BYTES = 1.3043478260869565
UPSTREAM_POOLED_CHARS = 0.6086956521739131
UPSTREAM_TOTAL_TOKENS = 92
UPSTREAM_TOTAL_BYTES = 120
UPSTREAM_TOTAL_CHARS = 56
UPSTREAM_TEXTS_ANALYZED = 6
"""Upstream's own totals. It analysed 6 of the 8 records: both whitespace-only
documents were excluded, which is the rule §7.2 states and the one glotscope
reproduces through ``is_blank``."""

TOLERANCE = 1e-6
"""§12.1's tolerance for this row."""


def _tokenizer() -> BackendTokenizer:
    mapping = byte_to_unicode()
    vocab = {mapping[value]: value for value in range(256)}
    for index, (left, right) in enumerate(MERGES):
        vocab[left + right] = 256 + index
    backend = BackendTokenizer(models.BPE(vocab=vocab, merges=MERGES))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return backend


def _measure(texts: tuple[str, ...]) -> tuple[list[list[int]], list[int], list[int], list[bool]]:
    """Encode and measure one batch exactly as ``analyze`` would."""
    backend = _tokenizer()
    encodings = [backend.encode(text).ids for text in texts]
    char_lengths = [len(text) for text in texts]
    byte_lengths = [len(text.encode("utf-8")) for text in texts]
    is_blank = [not text.strip() for text in texts]
    return encodings, char_lengths, byte_lengths, is_blank


@pytest.mark.parametrize("language", sorted(CORPUS))
def test_compression_rate_in_bytes_reproduces_tokeval(language: str) -> None:
    encodings, char_lengths, byte_lengths, is_blank = _measure(CORPUS[language])
    stats = aggregate_documents(encodings, char_lengths=char_lengths, byte_lengths=byte_lengths)

    result = compression(
        stats,
        unit_lengths=byte_lengths,
        is_blank=is_blank,
        language=language,
        unit=BYTES,
    )

    assert result.compression_rate == pytest.approx(UPSTREAM_RATE_BYTES[language], abs=TOLERANCE)


@pytest.mark.parametrize("language", sorted(CORPUS))
def test_compression_rate_in_characters_reproduces_tokeval(language: str) -> None:
    encodings, char_lengths, byte_lengths, is_blank = _measure(CORPUS[language])
    stats = aggregate_documents(encodings, char_lengths=char_lengths, byte_lengths=byte_lengths)

    result = compression(
        stats,
        unit_lengths=char_lengths,
        is_blank=is_blank,
        language=language,
        unit=CHARS,
    )

    assert result.compression_rate == pytest.approx(UPSTREAM_RATE_CHARS[language], abs=TOLERANCE)


def test_cpt_bpt_and_ctc_reproduce_tokeval_over_the_surviving_records() -> None:
    """CPT, BPT and CTC against upstream's totals.

    Compared over the records upstream actually analysed, because CPT and BPT
    divide by *every* document's tokens while CR applies the exclusion rule. On
    this corpus those differ — BPT would be 126/98 and CR is 120/92 — so feeding
    the blank records here would compare two different quantities and call the
    mismatch a divergence. §7.2's "default CR is numerically identical to BPT"
    is exact only when no record is excluded.
    """
    surviving = tuple(text for texts in CORPUS.values() for text in texts if text.strip())
    encodings, char_lengths, byte_lengths, _ = _measure(surviving)
    stats = aggregate_documents(encodings, char_lengths=char_lengths, byte_lengths=byte_lengths)

    result = compression(stats, unit_lengths=byte_lengths, language="mixed", unit=BYTES)

    assert len(surviving) == UPSTREAM_TEXTS_ANALYZED
    assert stats.total_bytes == UPSTREAM_TOTAL_BYTES
    assert stats.total_chars == UPSTREAM_TOTAL_CHARS
    assert result.ctc == UPSTREAM_TOTAL_TOKENS
    assert result.bpt == pytest.approx(UPSTREAM_POOLED_BYTES, abs=TOLERANCE)
    assert result.cpt == pytest.approx(UPSTREAM_POOLED_CHARS, abs=TOLERANCE)
    # With nothing excluded, CR and BPT are the same number, which is the
    # identity §7.2 states and the tests above deliberately do not assume.
    assert result.compression_rate == pytest.approx(result.bpt, abs=TOLERANCE)


def test_the_pooled_rate_reproduces_tokeval_across_all_three_languages() -> None:
    # Upstream pools by summing units and tokens across languages, not by
    # averaging the per-language rates. The two differ here — the mean of 1.1,
    # 1.476... and 1.2 is 1.259, not 1.304 — so this pins the aggregation as
    # well as the arithmetic.
    texts = tuple(text for texts in CORPUS.values() for text in texts)
    encodings, char_lengths, byte_lengths, is_blank = _measure(texts)
    stats = aggregate_documents(encodings, char_lengths=char_lengths, byte_lengths=byte_lengths)

    result = compression(
        stats,
        unit_lengths=byte_lengths,
        is_blank=is_blank,
        language="mixed",
        unit=BYTES,
    )
    mean_of_rates = sum(UPSTREAM_RATE_BYTES.values()) / len(UPSTREAM_RATE_BYTES)

    assert result.compression_rate == pytest.approx(UPSTREAM_POOLED_BYTES, abs=TOLERANCE)
    assert result.compression_rate != pytest.approx(mean_of_rates, abs=TOLERANCE)
