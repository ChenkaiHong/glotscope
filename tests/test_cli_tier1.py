"""The ``lint`` and ``analyze`` subcommands (PRD §8.2).

Exit codes carry meaning and are asserted rather than assumed:

* ``0`` — the command produced its document.
* ``1`` — a typed refusal. The library declined to emit a number, and the CLI's
  job is to surface that refusal rather than to soften it into a default.
* ``2`` — a path that is scheduled but not built, which is a different thing and
  must not be confused with a refusal by a script reading the exit status.

``analyze`` writes the §9 document through ``canonical_json``, so its output is
byte-stable — that is what makes ``glotscope verify`` possible at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.cli import main
from glotscope.corpus import Corpus
from glotscope.lint import byte_to_unicode

_ENGLISH = ("The cat sat.", "It rained.")
_HINDI = ("बिल्ली बैठी।", "बारिश हुई।")


def _tokenizer_json(tmp_path: Path) -> Path:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return path


def _write(root: Path, corpus: Corpus, **languages: tuple[str, ...]) -> None:
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    for language, lines in languages.items():
        text = "".join(f"{line}\n" for line in lines)
        (directory / f"{language}.txt").write_text(text, encoding="utf-8")


def _flores(root: Path) -> Corpus:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    _write(root, corpus, eng_Latn=_ENGLISH, hin_Deva=_HINDI)
    return corpus


def _analyze_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "analyze",
        str(_tokenizer_json(tmp_path)),
        "--corpus",
        "flores_plus",
        "--corpus-root",
        str(tmp_path),
        "--languages",
        "eng_Latn,hin_Deva",
        *extra,
    ]


def test_lint_prints_the_tier0_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lint", str(_tokenizer_json(tmp_path))]) == 0

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["vocab_size"] == 256
    assert document["family"] == "byte_level"
    assert document["byte_fallback_coverage"] == 256
    # Provenance gaps go to stderr, so stdout stays a clean document a pipe can
    # consume — and the gaps are still impossible to miss.
    assert "no upstream revision" in captured.err


def test_lint_refuses_a_hub_identifier_as_unbuilt_rather_than_invalid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Exit 2, not 1: from_pretrained is scheduled, and reporting "refused" here
    # would tell a script the input was wrong when it was not.
    assert main(["lint", "acme/tokenizer"]) == 2
    assert "from_pretrained" in capsys.readouterr().err


def test_lint_reports_a_mistyped_path_as_a_refusal_not_as_unbuilt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Exit 1, not 2. A typo in a path is a wrong argument; answering "scheduled
    # for a later release" sends the reader looking for a feature when what they
    # need is to fix the path.
    assert main(["lint", str(tmp_path / "typo.json")]) == 1

    captured = capsys.readouterr()
    assert "from_pretrained" not in captured.err
    assert "typo.json" in captured.err


def test_lint_reads_a_tokenizer_json_out_of_a_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A local checkout of a model directory is a legitimate local source: the
    # file it holds is the same artifact, so refusing it as "needs
    # from_pretrained" reports an unbuilt feature for something already built.
    _tokenizer_json(tmp_path)

    assert main(["lint", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["vocab_size"] == 256


def test_lint_reports_a_directory_without_a_tokenizer_json_as_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["lint", str(tmp_path)]) == 1
    assert "tokenizer.json" in capsys.readouterr().err


def test_lint_refuses_a_revision_it_cannot_resolve(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lint", "tokenizer.json", "--revision", "a" * 40]) == 2
    assert "revision" in capsys.readouterr().err


def test_analyze_writes_a_result_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _flores(tmp_path)
    output = tmp_path / "result.json"

    assert main(_analyze_argv(tmp_path, "--out", str(output))) == 0

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["manifest"]["corpus"]["id"] == "flores_plus"
    assert len(document["manifest"]["corpus"]["sha256"]) == 64
    assert document["manifest"]["parameters"]["normalization"] == "NFC"
    assert document["tier1"]["per_language"]["eng_Latn"]["ctc"] == 22
    # Tier 0 costs milliseconds and needs nothing extra, so a Tier 1 run carries
    # it: G2's claim is that one document spans the tiers that were run.
    assert document["tier0"]["family"] == "byte_level"
    assert capsys.readouterr().out == ""


def test_analyze_writes_to_stdout_when_no_output_path_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _flores(tmp_path)

    assert main(_analyze_argv(tmp_path)) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["tier1"]["per_language"]["hin_Deva"]["bpt"] == pytest.approx(1.0)


def test_analyze_serializes_requested_corpus_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _flores(tmp_path)

    assert (
        main(
            _analyze_argv(
                tmp_path,
                "--parity-reference",
                "eng_Latn",
                "--gini",
                "--renyi-alpha",
                "2.5",
            )
        )
        == 0
    )

    corpus_level = json.loads(capsys.readouterr().out)["tier1"]["corpus_level"]
    assert corpus_level["parity"]["reference_language"] == "eng_Latn"
    assert "gini" in corpus_level
    assert corpus_level["renyi_alpha"] == 2.5


def test_the_leading_space_convention_can_actually_be_switched_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # It is a recorded contested parameter that can move STRR by tens of points,
    # so a flag that cannot express False is a defect rather than a default.
    _flores(tmp_path)

    assert main(_analyze_argv(tmp_path, "--no-leading-space")) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["manifest"]["parameters"]["leading_space"] is False


def test_analyze_defaults_the_corpus_version_to_the_pinned_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _flores(tmp_path)

    assert main(_analyze_argv(tmp_path)) == 0

    document = json.loads(capsys.readouterr().out)
    assert document["manifest"]["corpus"]["version"] == corpus.version
    assert document["manifest"]["corpus"]["split"] == "devtest"


def test_analyze_refuses_ud_gold_against_flores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _flores(tmp_path)

    assert main(_analyze_argv(tmp_path, "--segmenter", "ud_gold")) == 1
    assert "word_segmentation" in capsys.readouterr().err


def test_analyze_reports_a_segmenter_it_cannot_run_as_unbuilt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _flores(tmp_path)

    assert main(_analyze_argv(tmp_path, "--segmenter", "stanza")) == 2
    # Exit 2 and a message naming what is actually missing: stanza needs a
    # pinned model, which is a decision rather than an install. Exit 1 would
    # tell a script its argument was wrong when it was not.
    assert "explicit local path" in capsys.readouterr().err


def test_analyze_runs_a_built_segmenter_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Whitespace needs no extra, so this exercises the CLI's segmenter path on
    # every platform the core install supports.
    _flores(tmp_path)

    assert main(_analyze_argv(tmp_path, "--segmenter", "whitespace")) == 0

    document = json.loads(capsys.readouterr().out)["tier1"]

    assert document["per_language"]["eng_Latn"]["fertility"] > 0
    assert document["per_language"]["hin_Deva"]["fertility"] > 0
    # The segmenter is published beside the number it produced. A fertility
    # value without one is not interpretable, and §7.1 rule 3 makes two values
    # under different segmenters incomparable.
    assert document["segmenter"] == "whitespace"


def test_analyze_surfaces_a_missing_corpus_file_as_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    _write(tmp_path, corpus, eng_Latn=_ENGLISH)

    assert main(_analyze_argv(tmp_path)) == 1
    assert "hin_Deva" in capsys.readouterr().err


def test_analyze_surfaces_the_license_filter_as_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = Corpus.from_registry("europarl", ["eng_Latn"], split="train", version="v7", sha256="")
    _write(tmp_path, corpus, eng_Latn=_ENGLISH)

    exit_code = main(
        [
            "analyze",
            str(_tokenizer_json(tmp_path)),
            "--corpus",
            "europarl",
            "--corpus-root",
            str(tmp_path),
            "--languages",
            "eng_Latn",
            "--license-filter",
            "commercial",
        ]
    )

    assert exit_code == 1
    assert "commercial" in capsys.readouterr().err


def test_an_unknown_corpus_is_rejected_by_the_parser(capsys: pytest.CaptureFixture[str]) -> None:
    # The registry is a closed set and its ids are what land in the manifest, so
    # a typo is caught before anything is read from disk.
    with pytest.raises(SystemExit):
        main(
            ["analyze", "t.json", "--corpus", "flores+", "--corpus-root", ".", "--languages", "en"]
        )
    assert "invalid choice" in capsys.readouterr().err
