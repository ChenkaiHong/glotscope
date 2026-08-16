"""STRR through ``analyze`` rather than in isolation (PRD §7.6, §6, D5).

``tests/test_strr.py`` covers the arithmetic against hand-built ``WordStats``.
This covers the part that was missing: nothing populated
``LanguageMetrics.strr``, so the module was correct and unreachable — the same
shape as the CLI defects earlier in this milestone, where reviewed code had
simply never been executed end to end.

STRR is **type-level, over a word list**, not over corpus tokens. So the wiring
gates on :attr:`~glotscope.enums.Capability.WORDLIST` rather than running
always: computing a retention rate over sentences would return a number for a
quantity nobody defined.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.corpus import Corpus
from glotscope.lint import byte_to_unicode
from glotscope.tokenizer import Tokenizer

_WORDS = ("the", "cat", "sat", "rain")
_HINDI_WORDS = ("बिल्ली", "बारिश")


def _tokenizer(tmp_path: Path) -> Tokenizer:
    mapping = byte_to_unicode()
    vocab = {mapping[value]: value for value in range(256)}
    # Two merges, so some words survive as one token and others do not: a
    # fixture where every word needs the same number of tokens would give STRR
    # 0.0 or 1.0 and hide any denominator mistake.
    merges = [("t", "h"), ("th", "e")]
    for index, (left, right) in enumerate(merges):
        vocab[left + right] = 256 + index
    backend = BackendTokenizer(models.BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return Tokenizer.from_file(path)


def _wordlist(root: Path, **languages: tuple[str, ...]) -> Corpus:
    corpus = Corpus.resolve("strr_wordlists", list(languages), version="1", split="train")
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    for language, words in languages.items():
        (directory / f"{language}.txt").write_text(
            "".join(f"{word}\n" for word in words), encoding="utf-8"
        )
    return corpus


def test_a_wordlist_corpus_produces_both_strr_conventions(tmp_path: Path) -> None:
    corpus = _wordlist(tmp_path, eng_Latn=_WORDS)

    report = _tokenizer(tmp_path).analyze(corpus.load(tmp_path))
    strr = report.per_language["eng_Latn"].strr

    assert strr is not None
    # Both, never one: the conventions disagree and §7.6 refuses to publish a
    # single unqualified number.
    assert 0.0 <= strr.bare <= 1.0
    assert 0.0 <= strr.leading_space <= 1.0
    assert strr.n_words == len(_WORDS)


def test_the_two_conventions_can_disagree(tmp_path: Path) -> None:
    # The reason the pair exists. "the" is one token bare and two with a leading
    # space under this vocabulary, so a fixture asserting they always match
    # would be asserting the bug §7.6 warns about.
    corpus = _wordlist(tmp_path, eng_Latn=_WORDS)

    strr = _tokenizer(tmp_path).analyze(corpus.load(tmp_path)).per_language["eng_Latn"].strr

    assert strr is not None
    assert strr.bare != strr.leading_space


def test_a_document_corpus_gets_no_strr(tmp_path: Path) -> None:
    # FLORES+ is sentences. A retention rate over sentences is not a quantity
    # §7.6 defines, so the gate is on the declared capability rather than on
    # whether the lines happen to look like words.
    corpus = Corpus.flores_plus(["eng_Latn"])
    directory = tmp_path / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True)
    (directory / "eng_Latn.txt").write_text("The cat sat.\nIt rained.\n", encoding="utf-8")

    report = _tokenizer(tmp_path).analyze(corpus.load(tmp_path))

    assert report.per_language["eng_Latn"].strr is None


def test_strr_needs_no_segmenter(tmp_path: Path) -> None:
    # Each line is already one word, so there is no W(D) choice to make and no
    # SegmenterRequiredError to raise. Worth pinning: STRR sits next to
    # fertility in §7 and fertility refuses without a segmenter.
    corpus = _wordlist(tmp_path, eng_Latn=_WORDS, hin_Deva=_HINDI_WORDS)

    report = _tokenizer(tmp_path).analyze(corpus.load(tmp_path))

    assert report.segmenter is None
    assert report.per_language["hin_Deva"].strr is not None
    with pytest.raises(Exception, match="segmenter"):
        _ = report.fertility
