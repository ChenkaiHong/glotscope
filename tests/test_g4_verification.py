"""The committed G4 fixture still verifies (PRD §12.3, G4).

``tests/test_verify.py`` exercises the command against results it produces in a
temporary directory. This checks the *committed* artifact — the one the CI job
runs against, and the only one that can rot silently: a change to a metric, a
serializer or the manifest schema invalidates it without touching a test.

Kept as a test as well as a CI step so the failure arrives during `pytest`,
where the change was made, rather than three minutes later in a workflow log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glotscope.cli import main

FIXTURE = Path(__file__).resolve().parents[1] / "verification"
RESULT = FIXTURE / "result.json"
TOKENIZER = FIXTURE / "tokenizer.json"
CORPUS_ROOT = FIXTURE / "corpus"

pytestmark = pytest.mark.skipif(
    not RESULT.is_file(),
    reason=(
        "the G4 fixture lives in the repository, not in the distribution: "
        "tests/ ships in the sdist and verification/ deliberately does not, so "
        "these run from a checkout and skip from an unpacked release"
    ),
)


def test_the_committed_result_reproduces(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "verify",
            str(RESULT),
            "--tokenizer",
            str(TOKENIZER),
            "--corpus-root",
            str(CORPUS_ROOT),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    assert "reproduced" in captured.out


def test_the_fixture_records_the_corpus_it_actually_used() -> None:
    # The fixture is four invented sentences. Naming it after a real corpus
    # would put a false claim in a manifest that ships in this repository as an
    # example of manifests being trustworthy.
    document = json.loads(RESULT.read_text(encoding="utf-8"))

    assert document["manifest"]["corpus"]["id"] == "verification_fixture"
    assert document["manifest"]["corpus"]["license"] == "CC0-1.0"


def test_the_fixture_exercises_more_than_the_default_path() -> None:
    # A result carrying only the always-on metrics would verify while leaving
    # the segmenter, parity, gini and Renyi paths unchecked — and those are the
    # ones with contested parameters that a refactor is most likely to move.
    document = json.loads(RESULT.read_text(encoding="utf-8"))
    corpus_level = document["tier1"]["corpus_level"]

    assert document["tier1"]["segmenter"] == "whitespace"
    assert "gini" in corpus_level
    assert corpus_level["renyi_alpha"] == 2.5
    assert corpus_level["parity"]["reference_language"] == "eng_Latn"
    assert document["tier1"]["per_language"]["hin_Deva"]["fertility"] > 0


def test_the_corpus_files_use_lf_endings() -> None:
    # The digest is over the bytes on disk, so a CRLF checkout on Windows would
    # fail the verify job for a reason unrelated to any number. .gitattributes
    # marks the fixture -text; this asserts the working tree agrees.
    for path in sorted(CORPUS_ROOT.rglob("*.txt")):
        assert b"\r" not in path.read_bytes(), f"{path} carries CR bytes"
