"""``glotscope compare`` (PRD §8.2).

The subcommand reads the ``result.json`` documents ``analyze`` writes, not
tokenizers. §8.2 sketches the positional as a tokenizer, but its own requirement
— that compare *refuse* results computed under different segmenters, alpha
values, normalizers or language sets — is unreachable that way: tokenizers
analyzed together in one invocation share one set of flags and can never
disagree. Only a published document records the parameters its numbers were
produced under, so only a published document can be checked against another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glotscope.cli import main

FIXTURE = Path(__file__).resolve().parents[1] / "verification"
RESULT = FIXTURE / "result.json"

pytestmark = pytest.mark.skipif(
    not RESULT.is_file(),
    reason=(
        "the G4 fixture lives in the repository, not in the distribution: "
        "tests/ ships in the sdist and verification/ deliberately does not, so "
        "these run from a checkout and skip from an unpacked release"
    ),
)


def _pair(tmp_path: Path) -> tuple[str, str]:
    """Two documents differing only in which artifact produced them."""
    document: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    left = tmp_path / "a.json"
    left.write_text(json.dumps(document), encoding="utf-8")
    document["manifest"]["tokenizer"]["tokenizer_json_sha256"] = "b" * 64
    right = tmp_path / "b.json"
    right.write_text(json.dumps(document), encoding="utf-8")
    return str(left), str(right)


def test_a_markdown_table_names_both_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    left, right = _pair(tmp_path)

    # Act
    exit_code = main(["compare", left, right, "--metric", "fertility"])
    captured = capsys.readouterr()

    # Assert
    assert exit_code == 0, captured.err
    assert "local@49697ba047fd" in captured.out
    assert "local@bbbbbbbbbbbb" in captured.out
    assert "hin_Deva" in captured.out


def test_json_format_emits_the_table_as_a_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    left, right = _pair(tmp_path)

    # Act
    exit_code = main(["compare", left, right, "--metric", "cpt", "--format", "json"])
    captured = capsys.readouterr()

    # Assert
    assert exit_code == 0, captured.err
    table = json.loads(captured.out)
    assert table["metric"] == "cpt"
    assert table["rows"]["eng_Latn"] == [1.0681818181818181, 1.0681818181818181]


def test_csv_format_writes_a_header_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    left, right = _pair(tmp_path)

    # Act
    exit_code = main(["compare", left, right, "--metric", "gini", "--format", "csv"])
    captured = capsys.readouterr()

    # Assert
    assert exit_code == 0, captured.err
    assert captured.out.splitlines()[0] == "gini,local@49697ba047fd,local@bbbbbbbbbbbb"


def test_incomparable_results_exit_one_and_say_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    document: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    left = tmp_path / "a.json"
    left.write_text(json.dumps(document), encoding="utf-8")
    document["manifest"]["tokenizer"]["tokenizer_json_sha256"] = "b" * 64
    document["manifest"]["parameters"]["segmenter"] = "icu"
    document["tier1"]["segmenter"] = "icu"
    right = tmp_path / "b.json"
    right.write_text(json.dumps(document), encoding="utf-8")

    # Act
    exit_code = main(["compare", str(left), str(right), "--metric", "fertility"])
    captured = capsys.readouterr()

    # Assert
    assert exit_code == 1
    assert "segmenter" in captured.err
    assert "not on the same scale" in captured.err


def test_a_tokenizer_json_is_reported_as_the_wrong_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The likeliest mistake, given §8.2 sketches the positional as a tokenizer.
    # Arrange
    left, _ = _pair(tmp_path)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"model": {"type": "BPE"}}), encoding="utf-8")

    # Act
    exit_code = main(["compare", left, str(tokenizer), "--metric", "cpt"])
    captured = capsys.readouterr()

    # Assert
    assert exit_code == 1
    assert "analyze" in captured.err
