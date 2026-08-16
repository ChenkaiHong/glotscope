"""``glotscope verify``: regenerate a committed result and compare (PRD §12.3, G4).

G4 is the differentiator — *every result carries a manifest, and verify
regenerates the numbers bit-identically*. Without this command that promise is an
untested aspiration, which is why §12.3 puts it in v1 rather than v2.

The design question it had to answer: §12.3 wants regeneration, so verify must
re-run from the manifest, but §9 forbids filesystem paths in a manifest — so a
tokenizer recorded as ``source="file"`` has nothing to resolve from. The answer
taken here is that the caller supplies the artifact and the manifest's SHA-256
decides whether it is the right one. The path stays out of the manifest; the
identity check stays real.

What must match bit-for-bit is the *numbers*. Environment differs between
machines by construction — that is what makes it worth recording — so a verify
demanding it match would fail everywhere but the machine that produced the file,
and nobody would run it.
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


def _tokenizer_json(path: Path, *, vocab_size: int = 256) -> Path:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(
        models.BPE(vocab={mapping[value]: value for value in range(vocab_size)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.save(str(path))
    return path


def _corpus(root: Path) -> Corpus:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    for language, lines in (("eng_Latn", _ENGLISH), ("hin_Deva", _HINDI)):
        (directory / f"{language}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
    return corpus


def _produce(tmp_path: Path, *extra: str) -> tuple[Path, Path]:
    """Run ``analyze``, returning the tokenizer path and the result path."""
    tokenizer = _tokenizer_json(tmp_path / "tokenizer.json")
    _corpus(tmp_path)
    result = tmp_path / "result.json"
    argv = [
        "analyze",
        str(tokenizer),
        "--corpus",
        "flores_plus",
        "--corpus-root",
        str(tmp_path),
        "--languages",
        "eng_Latn,hin_Deva",
        "--out",
        str(result),
        *extra,
    ]
    assert main(argv) == 0
    return tokenizer, result


def _verify(tokenizer: Path, result: Path, root: Path) -> list[str]:
    return ["verify", str(result), "--tokenizer", str(tokenizer), "--corpus-root", str(root)]


def test_a_result_verifies_against_the_inputs_that_produced_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer, result = _produce(tmp_path)

    assert main(_verify(tokenizer, result, tmp_path)) == 0
    assert "reproduced" in capsys.readouterr().out


def test_a_tampered_number_fails_and_names_where(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The case the command exists for. A published number that no longer
    # regenerates is the failure G4 promises to catch, and the message has to
    # say which number rather than only that something moved.
    tokenizer, result = _produce(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    document["tier1"]["per_language"]["eng_Latn"]["bpt"] = 999.0
    result.write_text(json.dumps(document), encoding="utf-8")

    assert main(_verify(tokenizer, result, tmp_path)) == 1

    captured = capsys.readouterr()
    assert "bpt" in captured.err
    assert "eng_Latn" in captured.err


def test_a_different_tokenizer_is_refused_before_anything_is_recomputed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The manifest pins the artifact by SHA-256 and nothing else, which is what
    # lets §9 keep filesystem paths out. Handing verify a different tokenizer
    # must fail on identity rather than by producing different numbers —
    # otherwise the error says "your results moved" when the truth is "wrong
    # file".
    _, result = _produce(tmp_path)
    other = _tokenizer_json(tmp_path / "other.json", vocab_size=255)

    assert main(_verify(other, result, tmp_path)) == 1
    assert "sha256" in capsys.readouterr().err.lower()


def test_environment_differences_do_not_fail_a_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The load-bearing design decision. Environment is recorded *because* it
    # differs between machines; demanding it match would mean a result could
    # only ever be verified on the machine that produced it, and the CI job
    # would never go green.
    tokenizer, result = _produce(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    document["manifest"]["environment"]["python"] = "3.0.0-not-a-real-python"
    result.write_text(json.dumps(document), encoding="utf-8")

    assert main(_verify(tokenizer, result, tmp_path)) == 0
    assert "environment" in capsys.readouterr().out


def test_a_result_from_another_release_still_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Comparing glotscope_version would make every release invalidate every
    # result published before it: an upgrade would fail a verify whose numbers
    # had not moved. G4 promises the *numbers* regenerate, so the version is
    # reported and the cross-release reproduction is the interesting part.
    tokenizer, result = _produce(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    document["glotscope_version"] = "0.0.1-an-older-release"
    result.write_text(json.dumps(document), encoding="utf-8")

    assert main(_verify(tokenizer, result, tmp_path)) == 0
    assert "regenerate across releases" in capsys.readouterr().out


def test_a_result_from_another_backend_still_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The v2 case, pinned now while it is cheap: "the Rust backend reproduces
    # the Python numbers" is the backend-parity evidence §13 needs, and a verify
    # that refused on a backend difference could never produce it.
    tokenizer, result = _produce(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    document["backend"] = "rust"
    result.write_text(json.dumps(document), encoding="utf-8")

    assert main(_verify(tokenizer, result, tmp_path)) == 0
    assert "backend parity holds" in capsys.readouterr().out


def test_a_schema_change_still_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The line between provenance and content. A schema change changes the
    # document, so it must fail where a version change must not.
    tokenizer, result = _produce(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    document["schema_version"] = "0.9"
    result.write_text(json.dumps(document), encoding="utf-8")

    assert main(_verify(tokenizer, result, tmp_path)) == 1
    assert "schema_version" in capsys.readouterr().err


def test_corpus_bytes_that_changed_are_caught_by_the_pinned_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer, result = _produce(tmp_path)
    corpus = Corpus.flores_plus(["eng_Latn"])
    path = tmp_path / corpus.spec.id / corpus.version / corpus.split / "eng_Latn.txt"
    path.write_text("Different text entirely.\nIt rained.\n", encoding="utf-8")

    assert main(_verify(tokenizer, result, tmp_path)) == 1
    assert "digest" in capsys.readouterr().err


def test_a_document_without_a_manifest_is_a_refusal_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `glotscope lint` emits a Tier 0 document with no manifest, so pointing
    # verify at one is an easy mistake to make.
    tokenizer = _tokenizer_json(tmp_path / "tokenizer.json")
    result = tmp_path / "lint.json"
    result.write_text(json.dumps({"vocab_size": 256}), encoding="utf-8")

    assert main(_verify(tokenizer, result, tmp_path)) == 1
    assert "manifest" in capsys.readouterr().err


def test_verify_regenerates_the_corpus_level_metrics_the_document_carries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Which corpus-level metrics ran is not a parameter the caller repeats: it
    # is readable from the document itself. A verify that silently skipped them
    # would pass a file whose gini or renyi no longer reproduces.
    tokenizer, result = _produce(
        tmp_path, "--gini", "--renyi-alpha", "2.5", "--parity-reference", "eng_Latn"
    )
    document = json.loads(result.read_text(encoding="utf-8"))
    assert "gini" in document["tier1"]["corpus_level"]

    assert main(_verify(tokenizer, result, tmp_path)) == 0

    document["tier1"]["corpus_level"]["gini"] = 0.5
    result.write_text(json.dumps(document), encoding="utf-8")
    assert main(_verify(tokenizer, result, tmp_path)) == 1
    assert "gini" in capsys.readouterr().err
