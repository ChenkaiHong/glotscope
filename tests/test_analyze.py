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
from tokenizers import decoders, models, pre_tokenizers

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


def test_a_segmenter_that_cannot_run_yet_is_refused_rather_than_recorded(tmp_path: Path) -> None:
    # The segmenter adapters are not built. Recording a segmenter that never ran
    # would put a false claim in the manifest, and returning an empty fertility
    # mapping would be the silently-plausible wrong answer D6 exists to prevent.
    with pytest.raises(NotImplementedError, match="segmenter"):
        _tokenizer(tmp_path).analyze(_parallel(tmp_path), segmenter=Segmenter.STANZA)


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
