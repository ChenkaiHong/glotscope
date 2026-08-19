"""Morphological alignment reaching a §9 document (PRD §7.7, §9).

A morphology run is its own run: the corpus *is* MorphyNet, so the single §9
corpus block pins the gold that produced the numbers. The surface forms parsed
out of the TSV become the documents every other Tier 1 metric sees, which is why
CPT here describes a word list rather than prose — that is what was measured, and
the manifest says so.

The fixture is the same byte-level tokenizer the rest of the suite uses: 256
single-byte tokens, no merges, so it splits every word at every character. That
makes it the oversegmentation case §7.7 rule 1 is about — perfect recall, poor
precision — and every number below is derivable by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.corpus import Corpus, LoadedCorpus
from glotscope.enums import MorphologicalType, TypologicalScope
from glotscope.lint import byte_to_unicode
from glotscope.tokenizer import Tokenizer

_VERSION = "9c9dbf1"
"""MorphyNet is pinned by commit, not by release (§10.1)."""

_ENGLISH = (
    "cat\tcats\tN|PL\tcat|s",
    "dog\tdogs\tN|PL\tdog|s",
    "eat\tate\tV;PST\t-",
)
"""Two usable rows and one suppletive form, in MorphyNet's real four-column
shape."""


def _tokenizer(tmp_path: Path) -> Tokenizer:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return Tokenizer.from_file(path)


def _gold_corpus(root: Path, **languages: tuple[str, ...]) -> LoadedCorpus:
    corpus = Corpus.resolve("morphynet", list(languages), version=_VERSION)
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    for language, rows in languages.items():
        (directory / f"{language}.txt").write_text(
            "".join(f"{row}\n" for row in rows), encoding="utf-8"
        )
    return corpus.load(root)


def test_analyze_scores_all_three_measures_against_the_gold(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, eng=_ENGLISH),
        morphological_types={"eng": MorphologicalType.FUSIONAL},
        frequency_weighted=False,
        include_single_token_words=False,
    )

    result = report.per_language["eng"].morphology
    assert result is not None
    assert result.scope is TypologicalScope.IN_SCOPE
    # A character-level split puts a boundary everywhere, so the one annotated
    # stem-suffix boundary is always among them.
    assert result.morphscore_v1 == pytest.approx(1.0)
    # "cats" and "dogs": predicted {1, 2, 3} each, gold {3} each. TP 2, FP 4,
    # FN 0 -- recall 1.0 against precision 1/3, which is the oversegmentation
    # artifact D11 exists to keep visible.
    full = result.full_alignment
    assert full is not None
    assert (full.true_positive, full.false_positive, full.false_negative) == (2, 4, 0)
    assert full.recall == pytest.approx(1.0)
    assert full.precision == pytest.approx(1 / 3)


def test_the_documents_are_the_surface_forms_not_the_tsv_rows(tmp_path: Path) -> None:
    # Compression over `cat\tcats\tN|PL\tcat|s` would describe the file format.
    report = _tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, eng=_ENGLISH),
        morphological_types={"eng": MorphologicalType.FUSIONAL},
        frequency_weighted=False,
        include_single_token_words=False,
    )

    stats = report.document_stats["eng"]
    assert stats.n_documents == 2
    assert stats.total_chars == len("cats") + len("dogs")


def test_the_loader_warning_travels_with_the_numbers(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, eng=_ENGLISH),
        morphological_types={"eng": MorphologicalType.FUSIONAL},
        frequency_weighted=False,
        include_single_token_words=False,
    )

    assert any("2 of 3" in warning for warning in report.warnings)


def test_sub_character_splits_do_not_invent_boundaries(tmp_path: Path) -> None:
    # Cyrillic is two bytes per character, so this tokenizer emits ten tokens for
    # a six-character word. Boundaries come from character offsets, so the pair
    # inside one character contributes no boundary -- reading piece counts
    # instead would score five boundaries that are not there and inflate recall
    # exactly where the PRD says Llama tokenizers do.
    rows = ("далай\tдалайд\tN;DAT\tдалай|д",)
    report = _tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, mon=rows),
        morphological_types={"mon": MorphologicalType.AGGLUTINATIVE},
        frequency_weighted=False,
        include_single_token_words=False,
    )

    result = report.per_language["mon"].morphology
    assert result is not None and result.full_alignment is not None
    counts = result.full_alignment
    assert (counts.true_positive, counts.false_positive, counts.false_negative) == (1, 4, 0)


def test_an_out_of_scope_language_carries_no_numbers(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, eng=_ENGLISH),
        morphological_types={"eng": MorphologicalType.NON_CONCATENATIVE},
        frequency_weighted=False,
        include_single_token_words=False,
    )

    result = report.per_language["eng"].morphology
    assert result is not None
    assert result.scope is TypologicalScope.OUT_OF_SCOPE
    assert result.morphscore_v1 is None
    assert result.morphscore_v2 is None
    assert result.full_alignment is None


def test_the_recorded_parameters_have_no_defaults(tmp_path: Path) -> None:
    # §7.7 rule 4: frequency weighting and one-token-word inclusion change
    # tokenizer rankings and the v2 paper could not choose defaults. Picking one
    # here would publish a ranking under a convention nobody stated.
    with pytest.raises(ValueError, match="frequency_weighted"):
        _tokenizer(tmp_path).analyze(
            _gold_corpus(tmp_path, eng=_ENGLISH),
            morphological_types={"eng": MorphologicalType.FUSIONAL},
            include_single_token_words=False,
        )


def test_a_language_without_a_declared_type_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eng"):
        _tokenizer(tmp_path).analyze(
            _gold_corpus(tmp_path, eng=_ENGLISH),
            morphological_types={},
            frequency_weighted=False,
            include_single_token_words=False,
        )


def test_morphology_parameters_against_a_corpus_without_gold_are_refused(tmp_path: Path) -> None:
    # Recording a parameter that never applied is the failure mode §9 exists to
    # prevent: the manifest would describe a measurement nobody made.
    corpus = Corpus.flores_plus(["eng_Latn"])
    directory = tmp_path / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "eng_Latn.txt").write_text("The cat sat.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="morph_gold"):
        _tokenizer(tmp_path).analyze(
            corpus.load(tmp_path),
            morphological_types={"eng_Latn": MorphologicalType.FUSIONAL},
            frequency_weighted=False,
            include_single_token_words=False,
        )


def test_the_document_publishes_the_morphology_block(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, eng=_ENGLISH),
        morphological_types={"eng": MorphologicalType.FUSIONAL},
        frequency_weighted=False,
        include_single_token_words=False,
    )

    block = report.to_dict()["per_language"]["eng"]["morphology"]
    assert block["scope"] == "in_scope"
    assert block["morphscore_v1"] == pytest.approx(1.0)
    assert block["full_alignment"]["precision"] == pytest.approx(1 / 3)
    assert block["full_alignment"]["recall"] == pytest.approx(1.0)
    assert block["full_alignment"]["f1"] == pytest.approx(0.5)
    # Recorded because they have no defaults and change the ranking.
    assert block["frequency_weighted"] is False
    assert block["include_single_token_words"] is False


def test_an_out_of_scope_document_publishes_the_scope_and_no_measures(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, eng=_ENGLISH),
        morphological_types={"eng": MorphologicalType.ISOLATING},
        frequency_weighted=True,
        include_single_token_words=True,
    )

    block = report.to_dict()["per_language"]["eng"]["morphology"]
    assert block["scope"] == "out_of_scope"
    assert "morphscore_v1" not in block
    assert "full_alignment" not in block


def _latin_only_tokenizer(tmp_path: Path) -> Tokenizer:
    """A vocabulary of four Latin letters and no UNK.

    ``tokenizers`` drops what such a model cannot represent, so a word outside
    the vocabulary comes back as **zero tokens** — no offsets, nothing to align.
    That is a real tokenizer state rather than a contrived one, and it is what
    the drop-and-count path exists for.
    """
    backend = BackendTokenizer(models.BPE(vocab={c: i for i, c in enumerate("cats")}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "latin.json"
    backend.save(str(path))
    return Tokenizer.from_file(path)


def test_a_word_the_tokenizer_cannot_represent_is_dropped_and_counted(tmp_path: Path) -> None:
    rows = ("cat\tcats\tN|PL\tcat|s", "далай\tдалайд\tN;DAT\tдалай|д")

    report = _latin_only_tokenizer(tmp_path).analyze(
        _gold_corpus(tmp_path, eng=rows),
        morphological_types={"eng": MorphologicalType.FUSIONAL},
        frequency_weighted=False,
        include_single_token_words=False,
    )

    assert any("1 of 2 gold words dropped" in warning for warning in report.warnings)
    # The word that did encode is still scored: dropping is per word, and a
    # partial corpus with a stated count beats a refusal that hides which part.
    result = report.per_language["eng"].morphology
    assert result is not None and result.full_alignment is not None
    assert result.full_alignment.true_positive == 1
