"""``analyze`` and ``verify`` over a gold-morphology corpus (PRD §8.2, §7.7, G4).

The claim this file exists to check is the one G4 makes: a published morphology
number regenerates from its own document. That is harder than it is for the rest
of Tier 1, because §7.7's three recorded parameters are not in the manifest's
parameter block — they live in the per-language morphology block, and ``verify``
has to read them back from there. A round trip is the only assertion that catches
it if that link breaks.

Exit codes carry meaning: ``0`` produced a document, ``1`` is a typed refusal.
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

_VERSION = "9c9dbf1"
"""MorphyNet is pinned by commit, not by release (§10.1)."""

_ROWS = (
    "cat\tcats\tN|PL\tcat|s",
    "dog\tdogs\tN|PL\tdog|s",
    "eat\tate\tV;PST\t-",
)


def _tokenizer_json(tmp_path: Path) -> Path:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return path


def _gold(root: Path) -> Corpus:
    corpus = Corpus.resolve("morphynet", ["eng"], version=_VERSION)
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "eng.txt").write_text("".join(f"{row}\n" for row in _ROWS), encoding="utf-8")
    return corpus


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    _gold(tmp_path)
    return [
        "analyze",
        str(_tokenizer_json(tmp_path)),
        "--corpus",
        "morphynet",
        "--corpus-root",
        str(tmp_path),
        "--corpus-version",
        _VERSION,
        "--languages",
        "eng",
        *extra,
    ]


def _complete(tmp_path: Path, out: Path) -> list[str]:
    return _argv(
        tmp_path,
        "--morphological-type",
        "eng=fusional",
        "--frequency-weighted",
        "false",
        "--include-single-token-words",
        "false",
        "--out",
        str(out),
    )


def test_analyze_publishes_the_morphology_block(tmp_path: Path) -> None:
    out = tmp_path / "result.json"

    assert main(_complete(tmp_path, out)) == 0

    document = json.loads(out.read_text(encoding="utf-8"))
    block = document["tier1"]["per_language"]["eng"]["morphology"]
    assert block["morphological_type"] == "fusional"
    assert block["scope"] == "in_scope"
    # Character-level splits: every gold boundary found, two thirds of the
    # predicted ones wrong.
    assert block["full_alignment"]["recall"] == pytest.approx(1.0)
    assert block["full_alignment"]["precision"] == pytest.approx(1 / 3)


def test_the_morphology_document_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "result.json"
    assert main(_complete(tmp_path, out)) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "verify",
            str(out),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--corpus-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert "reproduced" in captured.out


def test_a_missing_recorded_parameter_is_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(_argv(tmp_path, "--morphological-type", "eng=fusional"))

    assert exit_code == 1
    assert "frequency_weighted" in capsys.readouterr().err


def test_a_malformed_type_argument_is_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        _argv(
            tmp_path,
            "--morphological-type",
            "fusional",
            "--frequency-weighted",
            "false",
            "--include-single-token-words",
            "false",
        )
    )

    assert exit_code == 1
    assert "LANG=TYPE" in capsys.readouterr().err


def test_an_unknown_type_names_the_closed_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        _argv(
            tmp_path,
            "--morphological-type",
            "eng=polysynthetic",
            "--frequency-weighted",
            "false",
            "--include-single-token-words",
            "false",
        )
    )

    assert exit_code == 1
    assert "agglutinative" in capsys.readouterr().err
