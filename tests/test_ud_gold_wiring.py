"""Gold word boundaries reaching a §9 document (PRD §7.1 rule 1, §9, §10.3).

A ``UD_GOLD`` run is its own run: the corpus *is* the treebank, so the single §9
corpus block pins the annotation that produced the word counts. The sentences
parsed out of the CoNLL-U become the documents every other Tier 1 metric sees —
which is why CPT here describes annotated sentences rather than prose. That is
what was measured, and the manifest says so.

The fixture is the byte-level tokenizer the rest of the suite uses: 256
single-byte tokens, no merges, so a word costs one token per byte. Fertility is
therefore hand-checkable, and the point of this file is not the arithmetic but
*which* word list the arithmetic divides by.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.corpus import UD_VERSION, Corpus, LoadedCorpus
from glotscope.enums import Segmenter
from glotscope.errors import CorpusIntegrityError
from glotscope.lint import byte_to_unicode
from glotscope.tokenizer import Tokenizer

_TREEBANK = "UD_English-EWT"

_ROWS = """\
# text = Hi, there
1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\tSpaceAfter=No
2\t,\t,\tPUNCT\t_\t_\t1\tpunct\t_\t_
3\tthere\tthere\tADV\t_\t_\t1\tadvmod\t_\t_

# text = Go
1\tGo\tgo\tVERB\t_\t_\t0\troot\t_\t_
"""
"""Two sentences. The first is the case separating gold from whitespace: `Hi,
there` is three gold words and two whitespace ones."""


def _tokenizer(tmp_path: Path) -> Tokenizer:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return Tokenizer.from_file(path)


def _treebank(root: Path, rows: str = _ROWS, treebank: str = _TREEBANK) -> LoadedCorpus:
    corpus = Corpus.universal_dependencies([treebank])
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{treebank}.txt").write_text(rows, encoding="utf-8")
    return corpus.load(root)


def test_fertility_divides_by_the_annotated_words(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _treebank(tmp_path), segmenter=Segmenter.UD_GOLD, leading_space=False
    )

    result = report.per_language[_TREEBANK].fertility
    assert result is not None
    # Gold words: Hi | , | there | Go. One token per byte, so 2 + 1 + 5 + 2 = 10
    # tokens over 4 words. Whitespace would have found 3 words and reported
    # 10/3 — the same tokenization against a different denominator, which is the
    # incomparability §7.1 rule 1 exists to prevent.
    assert result.fertility == pytest.approx(10 / 4)


def test_the_treebank_and_its_release_are_recorded_as_the_model_version(
    tmp_path: Path,
) -> None:
    # §10.3: record the treebank, not "UD". Korean treebanks disagree among
    # themselves, and the identifier alone still does not say which annotation
    # was read — the release pins that.
    report = _tokenizer(tmp_path).analyze(
        _treebank(tmp_path), segmenter=Segmenter.UD_GOLD, leading_space=False
    )

    result = report.per_language[_TREEBANK].fertility
    assert result is not None
    assert result.segmenter is Segmenter.UD_GOLD
    assert result.segmenter_model_version == f"{_TREEBANK} {UD_VERSION}"


def test_the_documents_are_the_sentences_not_the_conllu_rows(tmp_path: Path) -> None:
    # Compression over `1\tHi\thi\tINTJ...` would describe the file format.
    report = _tokenizer(tmp_path).analyze(
        _treebank(tmp_path), segmenter=Segmenter.UD_GOLD, leading_space=False
    )

    stats = report.document_stats[_TREEBANK]
    assert stats.n_documents == 2
    assert stats.total_chars == len("Hi, there") + len("Go")


def test_the_loader_warning_travels_with_the_numbers(tmp_path: Path) -> None:
    rows = _ROWS + "\n# text = Don’t\n1\tDon\tdo\tAUX\t_\t_\t0\troot\t_\t_\n"

    report = _tokenizer(tmp_path).analyze(
        _treebank(tmp_path, rows), segmenter=Segmenter.UD_GOLD, leading_space=False
    )

    assert any("2 of 3" in warning for warning in report.warnings)


def test_a_treebank_with_nothing_usable_is_refused(tmp_path: Path) -> None:
    rows = "# text = Don’t\n1\tDon\tdo\tAUX\t_\t_\t0\troot\t_\t_\n"

    with pytest.raises(CorpusIntegrityError):
        _tokenizer(tmp_path).analyze(
            _treebank(tmp_path, rows), segmenter=Segmenter.UD_GOLD, leading_space=False
        )


def test_the_document_publishes_the_gold_segmenter(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _treebank(tmp_path), segmenter=Segmenter.UD_GOLD, leading_space=False
    )

    document = report.to_dict()
    # The Tier 1 document names the segmenter once, for the run. Which treebank
    # supplied the boundaries is recoverable from §9's corpus block, whose
    # languages *are* the treebank identifiers and whose version is the release —
    # so a UD_GOLD result says what annotated it without a per-language field.
    assert document["segmenter"] == "ud_gold"
    assert document["per_language"][_TREEBANK]["fertility"] == pytest.approx(10 / 4)
