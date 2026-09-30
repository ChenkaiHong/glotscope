"""Tier 1 end to end: a loaded corpus in, a ``Tier1Report`` out (PRD §8.1).

The fixture is a byte-level tokenizer holding all 256 byte values with no merges,
which makes every number here derivable by hand: one token per UTF-8 byte, so BPT
is exactly 1.0 and CPT is exactly the corpus's characters-per-byte ratio. These
are assertions about the wiring, not about anyone's trained vocabulary.

§8.1 calls ``parity``, ``gini`` and ``renyi_efficiency`` on the *report*, not on
``analyze``. So ``analyze`` folds the corpus once and stores the statistics, and
the corpus-level metrics stay a parameter of the request — which is the only way
``renyi_efficiency(alpha=2.5)`` and ``renyi_efficiency(alpha=3.0)`` can both be
answered without re-encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors

from glotscope import __version__, backend
from glotscope.corpus import Corpus, LoadedCorpus
from glotscope.enums import Capability, Normalization, Segmenter
from glotscope.errors import CapabilityError, SegmenterRequiredError
from glotscope.lint import byte_to_unicode
from glotscope.manifest import Manifest, canonical_json, environment
from glotscope.report import Report
from glotscope.tokenizer import Tokenizer

_ENGLISH = ("The cat sat.", "It rained.")
_HINDI = ("बिल्ली बैठी।", "बारिश हुई।")
_SAMPLE = "sample-2026-08"
"""FineWeb2 names its own sample; see ``test_corpus_loading``."""

_DECOMPOSED = "é"
"""``e`` plus COMBINING ACUTE ACCENT: two characters and three UTF-8 bytes, which
NFC folds to one character and two bytes. The pair is what makes the recorded
normalization form observable in CPT."""


def _tokenizer(tmp_path: Path) -> Tokenizer:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return Tokenizer.from_file(path)


def _write(root: Path, corpus: Corpus, **languages: tuple[str, ...]) -> None:
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    for language, lines in languages.items():
        text = "".join(f"{line}\n" for line in lines)
        (directory / f"{language}.txt").write_text(text, encoding="utf-8")


def _parallel(root: Path) -> LoadedCorpus:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    _write(root, corpus, eng_Latn=_ENGLISH, hin_Deva=_HINDI)
    return corpus.load(root)


def _monolingual(root: Path, *lines: str) -> LoadedCorpus:
    corpus = Corpus.fineweb2(["eng_Latn"], version=_SAMPLE)
    _write(root, corpus, eng_Latn=tuple(lines))
    return corpus.load(root)


def _utf8_bytes(lines: tuple[str, ...]) -> int:
    return sum(len(line.encode("utf-8")) for line in lines)


def test_analyze_computes_the_compression_family_for_every_language(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(_parallel(tmp_path))

    english = report.per_language["eng_Latn"].compression
    hindi = report.per_language["hin_Deva"].compression

    # 12 + 10 ASCII characters, one token per byte.
    assert english.ctc == 22
    assert english.cpt == pytest.approx(1.0)
    assert english.bpt == pytest.approx(1.0)
    # Devanagari costs three bytes per character, so CPT falls while BPT cannot:
    # exactly the reason §7.2 refuses to present either as a cross-script
    # efficiency comparison.
    assert hindi.bpt == pytest.approx(1.0)
    assert hindi.cpt < 0.5
    assert hindi.ctc == _utf8_bytes(_HINDI)


def test_the_default_compression_rate_is_measured_in_bytes(tmp_path: Path) -> None:
    # TokEval's default unit is UTF-8 bytes, which makes CR numerically identical
    # to BPT. Recording the unit is what keeps that from looking like a coincidence.
    report = _tokenizer(tmp_path).analyze(_parallel(tmp_path))

    english = report.per_language["eng_Latn"].compression
    assert english.compression_rate_unit == "bytes"
    assert english.compression_rate == pytest.approx(english.bpt)


def test_analyze_stores_the_folded_statistics_so_parity_recomputes(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(_parallel(tmp_path))

    result = report.parity(reference="eng_Latn")

    # Ratio of means, equivalently ratio of totals over equal line counts (D7).
    expected = _utf8_bytes(_HINDI) / _utf8_bytes(_ENGLISH)
    assert result.per_language["eng_Latn"] == pytest.approx(1.0)
    assert result.per_language["hin_Deva"] == pytest.approx(expected)
    assert result.worst_case_language == "hin_Deva"


def test_the_report_answers_two_renyi_alphas_without_re_encoding(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(_parallel(tmp_path))

    at_two_five = report.renyi_efficiency(alpha=2.5)
    at_three = report.renyi_efficiency(alpha=3.0)

    assert at_two_five.alpha == 2.5
    assert at_three.alpha == 3.0
    assert at_two_five.value != at_three.value
    assert 0.0 <= report.gini().value <= 1.0


def test_analyze_reports_round_trip_losslessness_per_language(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(_parallel(tmp_path))

    # A byte-level tokenizer round-trips every script; anything below 1.0 here
    # would be a defect in the tokenizer rather than a property of the language.
    assert report.per_language["eng_Latn"].roundtrip_rate == 1.0
    assert report.per_language["hin_Deva"].roundtrip_rate == 1.0


def test_analyze_measures_the_normalized_text(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    composed = tokenizer.analyze(_monolingual(tmp_path, _DECOMPOSED))
    decomposed = tokenizer.analyze(
        _monolingual(tmp_path, _DECOMPOSED), normalization=Normalization.NFD
    )

    # NFC folds e + U+0301 to one character in two bytes; NFD keeps two
    # characters in three. NFKC versus NFC shifts CPT the same way, which is why
    # the form is a recorded parameter rather than an implementation detail.
    assert composed.per_language["eng_Latn"].compression.cpt == pytest.approx(0.5)
    assert decomposed.per_language["eng_Latn"].compression.cpt == pytest.approx(2 / 3)


def test_analyze_records_every_contested_parameter(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _parallel(tmp_path),
        leading_space=False,
        normalization=Normalization.NFKC,
        add_special_tokens=True,
    )

    parameters = report.parameters
    assert parameters is not None
    assert parameters.leading_space is False
    assert parameters.normalization is Normalization.NFKC
    assert parameters.add_special_tokens is True
    assert parameters.segmenter is None


def test_analyze_carries_the_corpus_digest_into_the_report(tmp_path: Path) -> None:
    loaded = _parallel(tmp_path)

    report = _tokenizer(tmp_path).analyze(loaded)

    assert report.corpus is not None
    # The digest computed at load time, not the empty one that was asked for:
    # a manifest naming a corpus it did not read is worse than no manifest.
    assert report.corpus.sha256 == loaded.corpus.sha256
    assert report.corpus.id == "flores_plus"
    assert report.corpus.capabilities == frozenset({Capability.PARALLEL})
    assert report.capabilities == frozenset({Capability.PARALLEL})
    assert report.corpus_id == "flores_plus"


def test_analyze_warns_about_documents_that_encoded_to_nothing(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(_monolingual(tmp_path, "hello", "", "world"))

    # §12.2: empty encodings are counted and reported, never silently dropped.
    # Normalization alone can strip a document to nothing.
    assert report.document_stats["eng_Latn"].n_empty_documents == 1
    assert any("eng_Latn" in warning and "zero tokens" in warning for warning in report.warnings)


def test_analyze_warns_when_normalization_is_switched_off(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(
        _monolingual(tmp_path, _DECOMPOSED), normalization=Normalization.NONE
    )

    assert any("normalization" in warning for warning in report.warnings)


def test_analyze_is_quiet_on_a_clean_corpus(tmp_path: Path) -> None:
    # The warnings array is load-bearing, so it must not fill with noise that
    # applies to every run: a reader who learns to skip it loses the signal.
    assert _tokenizer(tmp_path).analyze(_parallel(tmp_path)).warnings == ()


def test_analyze_refuses_a_corpus_that_was_never_loaded(tmp_path: Path) -> None:
    # glotscope ships no corpora (D12), so a bare Corpus carries no text at all.
    with pytest.raises(ValueError, match="load"):
        _tokenizer(tmp_path).analyze(Corpus.flores_plus(["eng_Latn"]))


def test_analyze_refuses_a_corpus_with_no_languages(tmp_path: Path) -> None:
    corpus = Corpus.flores_plus([])
    _write(tmp_path, corpus)

    with pytest.raises(ValueError, match="no languages"):
        _tokenizer(tmp_path).analyze(corpus.load(tmp_path))


def test_analyze_refuses_a_language_with_no_documents(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no documents"):
        _tokenizer(tmp_path).analyze(_monolingual(tmp_path))


def test_ud_gold_is_refused_on_a_corpus_without_gold_word_boundaries(tmp_path: Path) -> None:
    # §7.1 rule 1: UD supplies gold boundaries only inside its own treebanks.
    # Applying "UD segmentation" to FLORES+ is a different operation performed by
    # a trained model with its own accuracy and its own version.
    with pytest.raises(CapabilityError, match="word_segmentation"):
        _tokenizer(tmp_path).analyze(_parallel(tmp_path), segmenter=Segmenter.UD_GOLD)


def test_a_trained_segmenter_without_its_model_is_refused_rather_than_recorded(
    tmp_path: Path,
) -> None:
    # Stanza and UDPipe are built now, and the decision they were waiting on is
    # made: the model is an explicit path the caller pins, never a download on
    # first use, which would put an unrecorded artifact behind a published
    # number. So this is a ValueError naming the missing input rather than a
    # NotImplementedError — the adapter exists; the model is the caller's.
    # Recording a segmenter that never ran would be a false claim in the
    # manifest, and an empty fertility mapping the silently-plausible wrong
    # answer D6 prevents.
    with pytest.raises(ValueError, match="model"):
        _tokenizer(tmp_path).analyze(_parallel(tmp_path), segmenter=Segmenter.STANZA)


def _wordpiece(tmp_path: Path) -> Tokenizer:
    backend_tokenizer = BackendTokenizer(
        models.WordPiece(vocab={"[UNK]": 0, "the": 1, "cat": 2}, unk_token="[UNK]")
    )
    backend_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "wordpiece.json"
    backend_tokenizer.save(str(path))
    return Tokenizer.from_file(path)


def test_a_language_the_tokenizer_cannot_represent_is_dropped(tmp_path: Path) -> None:
    """§7.1 rule 6, exercised through ``analyze`` rather than only in isolation.

    The byte-level fixture everything else here uses has no ``[UNK]`` at all —
    every byte is representable — so the exclusion rule could never fire against
    it and the wiring would go untested. This uses a WordPiece vocabulary that
    covers two English words and nothing else, which is the shape of tokenizer
    the rule was written for: FlanT5 fails it on 42% of FLORES-200 languages.

    Hindi is over the line and English is not, and the rule drops the
    *language*. Raising out of ``analyze`` instead threw English's numbers away
    with it, which no reading of "drop" allows.
    """
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    _write(tmp_path, corpus, eng_Latn=("the cat", "cat the"), hin_Deva=_HINDI)

    report = _wordpiece(tmp_path).analyze(corpus.load(tmp_path), segmenter=Segmenter.WHITESPACE)

    assert set(report.fertility) == {"eng_Latn"}
    hindi = report.per_language["hin_Deva"]
    assert hindi.fertility is None
    # Only fertility is excluded; the language's other numbers still stand.
    assert hindi.compression.ctc > 0
    assert "fertility" not in report.to_dict()["per_language"]["hin_Deva"]
    [warning] = [w for w in report.warnings if "hin_Deva" in w and "[UNK]" in w]
    assert "§7.1 rule 6" in warning


def test_a_run_where_every_language_is_dropped_still_returns(tmp_path: Path) -> None:
    corpus = Corpus.fineweb2(["hin_Deva"], version=_SAMPLE)
    _write(tmp_path, corpus, hin_Deva=_HINDI)

    report = _wordpiece(tmp_path).analyze(corpus.load(tmp_path), segmenter=Segmenter.WHITESPACE)

    assert report.fertility == {}
    assert sum("[UNK]" in warning for warning in report.warnings) == 1


def test_a_built_segmenter_produces_fertility(tmp_path: Path) -> None:
    # Whitespace needs no extra, so this runs everywhere the core install does.
    # It is the wiring test: an adapter with nothing consuming it is untested by
    # construction, which is how the CLI defects earlier in this milestone got in.
    report = _tokenizer(tmp_path).analyze(_parallel(tmp_path), segmenter=Segmenter.WHITESPACE)

    assert set(report.fertility) == {"eng_Latn", "hin_Deva"}
    assert all(value > 0 for value in report.fertility.values())
    english = report.per_language["eng_Latn"].fertility
    assert english is not None
    assert english.segmenter is Segmenter.WHITESPACE
    # No model applies, so none is pinned — the one case where null is legal.
    assert english.segmenter_model_version is None


def test_a_run_assembles_into_a_result_document(tmp_path: Path) -> None:
    # What the report carries has to be *sufficient* to build the §9 manifest,
    # or the caller ends up restating the parameters by hand and G4's promise
    # rests on user code getting a second copy right.
    tokenizer = _tokenizer(tmp_path)
    loaded = _parallel(tmp_path)
    tier1 = tokenizer.analyze(loaded)
    assert tier1.parameters is not None and tier1.corpus is not None

    report = Report(
        tier0=tokenizer.lint(),
        tier1=tier1,
        manifest=Manifest(
            tokenizer=tokenizer.manifest,
            parameters=tier1.parameters,
            environment=environment(),
            backend=backend(),
            glotscope_version=__version__,
            corpus=tier1.corpus,
        ),
        warnings=tokenizer.warnings,
    )
    output = tmp_path / "result.json"
    report.to_json(output)

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["manifest"]["corpus"]["sha256"] == loaded.corpus.sha256
    assert document["manifest"]["parameters"]["normalization"] == "NFC"
    assert document["tier1"]["per_language"]["hin_Deva"]["ctc"] == _utf8_bytes(_HINDI)
    # Deterministic serialization is what makes `glotscope verify` possible at
    # all: the same run must produce the same bytes.
    assert output.read_text(encoding="utf-8") == canonical_json(report.to_dict()) + "\n"


def test_word_level_metrics_still_refuse_without_a_segmenter(tmp_path: Path) -> None:
    report = _tokenizer(tmp_path).analyze(_parallel(tmp_path))

    with pytest.raises(SegmenterRequiredError):
        _ = report.fertility


def _unigram(tmp_path: Path, *, unk_id: int | None) -> Tokenizer:
    pieces = [("a", -1.0), ("b", -1.0), ("c", -1.0), ("abc", -0.5)]
    vocab = [("<unk>", 0.0), *pieces] if unk_id is not None else pieces
    backend_tokenizer = BackendTokenizer(models.Unigram(vocab, unk_id=unk_id))
    backend_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    path = tmp_path / "unigram.json"
    backend_tokenizer.save(str(path))
    return Tokenizer.from_file(path)


def test_a_unigram_tokenizer_reports_the_characters_it_maps_to_unk(tmp_path: Path) -> None:
    """§7.1 rule 6 for the family it was written for.

    A Unigram ``tokenizer.json`` stores its UNK as ``unk_id`` and names no
    ``unk_token``: T5, mT5, Flan-T5 and XLM-R all look like this. Reading only the
    name left every one of them at an UNK rate of 0.0, so the exclusion could
    never fire for FlanT5 — the tokenizer the rule is quoted against.
    """
    tokenizer = _unigram(tmp_path, unk_id=0)

    # Thirteen characters, of which "x" alone is unrepresentable: 1/13, under 10%.
    report = tokenizer.analyze(
        _monolingual(tmp_path, "abc abc abc x"), segmenter=Segmenter.WHITESPACE
    )

    english = report.per_language["eng_Latn"].fertility
    assert english is not None
    assert english.unk_char_rate == pytest.approx(1 / 13)


def test_a_unigram_tokenizer_without_an_unk_records_none(tmp_path: Path) -> None:
    # `unk_id: null` is legal. It must read as "no UNK", not crash on the lookup.
    report = _unigram(tmp_path, unk_id=None).analyze(
        _monolingual(tmp_path, "abc abc"), segmenter=Segmenter.WHITESPACE
    )

    english = report.per_language["eng_Latn"].fertility
    assert english is not None
    assert english.unk_char_rate == 0.0


def _prepending(tmp_path: Path) -> Tokenizer:
    """A Llama-2-style tokenizer: its normalizer prepends ``▁`` to every input."""
    backend_tokenizer = BackendTokenizer(
        models.Unigram([("<unk>", 0.0), ("▁", -2.0), ("▁the", -1.0), ("▁cat", -1.0)], unk_id=0)
    )
    backend_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")]
    )
    path = tmp_path / "prepending.json"
    backend_tokenizer.save(str(path))
    return Tokenizer.from_file(path)


def test_a_tokenizer_that_prepends_its_own_boundary_gets_no_second_one(tmp_path: Path) -> None:
    """§7.1 rule 5 on a Llama-2-style tokenizer.

    The leading-space convention asks for the word as it appears mid-sentence.
    A byte-level tokenizer needs a literal space for that; this one prepends ``▁``
    to every input itself, so a literal space became a second marker —
    ``" the"`` encoded as ``['▁', '▁the']`` — and fertility rose by exactly one.
    """
    report = _prepending(tmp_path).analyze(
        _monolingual(tmp_path, "the cat", "cat the"), segmenter=Segmenter.WHITESPACE
    )

    english = report.per_language["eng_Latn"].fertility
    assert english is not None
    assert english.fertility == 1.0
    assert english.leading_space is True
    assert sum("prepends '▁'" in warning for warning in report.warnings) == 1


def test_without_a_leading_space_a_prepending_tokenizer_is_left_alone(tmp_path: Path) -> None:
    report = _prepending(tmp_path).analyze(
        _monolingual(tmp_path, "the cat"), segmenter=Segmenter.WHITESPACE, leading_space=False
    )

    english = report.per_language["eng_Latn"].fertility
    assert english is not None
    assert english.fertility == 1.0
    assert not any("prepends '▁'" in warning for warning in report.warnings)


def test_strr_on_a_prepending_tokenizer_counts_no_phantom_boundary(tmp_path: Path) -> None:
    # τ(" the") on this tokenizer is ['▁', '▁the'] only because the word was
    # encoded alone; the leading-space convention is the mid-sentence form, ['▁the'].
    corpus = Corpus.resolve("strr_wordlists", ["eng_Latn"], version="1", split="train")
    _write(tmp_path, corpus, eng_Latn=("the", "cat"))

    report = _prepending(tmp_path).analyze(corpus.load(tmp_path))

    strr = report.per_language["eng_Latn"].strr
    assert strr is not None
    assert (strr.bare, strr.leading_space) == (100.0, 100.0)


def _bos_template(tmp_path: Path) -> Tokenizer:
    """The byte-level fixture with a template that prepends ``<s>`` to every input."""
    mapping = byte_to_unicode()
    backend_tokenizer = BackendTokenizer(
        models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[])
    )
    backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend_tokenizer.decoder = decoders.ByteLevel()
    backend_tokenizer.add_special_tokens(["<s>"])
    backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 256)]
    )
    path = tmp_path / "bos.json"
    backend_tokenizer.save(str(path))
    return Tokenizer.from_file(path)


def test_round_trip_is_scored_on_the_text_not_on_the_specials_a_template_adds(
    tmp_path: Path,
) -> None:
    """§7.8: a rate below 1.0 is a defect in the tokenizer. With
    ``add_special_tokens`` the BOS a template prepends decoded back as ``<s>``,
    so every document "failed" and the rate read 0.0 for a lossless tokenizer."""
    report = _bos_template(tmp_path).analyze(_parallel(tmp_path), add_special_tokens=True)

    for language, lines in (("eng_Latn", _ENGLISH), ("hin_Deva", _HINDI)):
        metrics = report.per_language[language]
        assert metrics.roundtrip_rate == 1.0
        # The specials still count where they are tokens a model is fed.
        assert metrics.compression.ctc == _utf8_bytes(lines) + len(lines)
