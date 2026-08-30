"""``glotscope leaderboard`` end to end (PRD §8.2, §16.1).

The command's contract is an exit code and two files. Exit 2 meant *scheduled but
not built*, and this is the change that makes it wrong: after this, a
leaderboard that cannot run is a real answer about the input, not a missing
feature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glotscope.cli import main
from glotscope.corpus import Corpus


@pytest.fixture
def toy_encoding(monkeypatch: pytest.MonkeyPatch) -> Any:
    import tiktoken

    ranks = {bytes([value]): value for value in range(256)}
    ranks[b"the"] = 256
    encoding = tiktoken.Encoding(
        name="toy",
        pat_str=r"\s+|\S+",
        mergeable_ranks=ranks,
        special_tokens={"<|end|>": 257},
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)
    return encoding


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    directory = tmp_path / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True)
    (directory / "eng_Latn.txt").write_text("The cat sat.\nIt rained.\n", encoding="utf-8")
    (directory / "hin_Deva.txt").write_text("बिल्ली बैठी।\nबारिश हुई।\n", encoding="utf-8")

    config = tmp_path / "leaderboard.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "corpus": {"id": "flores_plus", "languages": ["eng_Latn", "hin_Deva"]},
                "parameters": {"parity_reference": "eng_Latn", "gini": True},
                "roster": [{"id": "tiktoken:toy", "label": "toy encoding"}],
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path


def test_it_writes_the_json_and_the_table(
    tmp_path: Path, toy_encoding: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    config, root = _fixture(tmp_path)
    out = tmp_path / "results"

    assert (
        main(
            ["leaderboard", "--config", str(config), "--out", str(out), "--corpus-root", str(root)]
        )
        == 0
    )

    document = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert document["rows"][0]["label"] == "toy encoding"
    assert document["corpus"]["sha256"]
    table = (out / "leaderboard.md").read_text(encoding="utf-8")
    assert "toy encoding" in table
    assert "n/a (tokenizer-only)" in table


def test_a_malformed_config_is_exit_1_not_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 is reserved for a path that is scheduled and unbuilt. Once the
    command exists, reporting a bad config as 2 would send the reader to wait for
    a release instead of fixing their file."""
    config = tmp_path / "leaderboard.json"
    config.write_text(json.dumps({"corpus": {"id": "nope", "languages": ["x"]}}), encoding="utf-8")

    assert main(["leaderboard", "--config", str(config), "--out", str(tmp_path / "out")]) == 1
    assert "nope" in capsys.readouterr().err


def test_the_output_is_canonical_so_a_rerun_diffs_cleanly(
    tmp_path: Path, toy_encoding: Any
) -> None:
    """§16.1's nightly job compares a regenerated board against the published
    one. Key order drifting between runs would make every night a diff."""
    config, root = _fixture(tmp_path)
    first, second = tmp_path / "a", tmp_path / "b"

    for out in (first, second):
        assert (
            main(
                [
                    "leaderboard",
                    "--config",
                    str(config),
                    "--out",
                    str(out),
                    "--corpus-root",
                    str(root),
                ]
            )
            == 0
        )

    assert (first / "leaderboard.json").read_text(encoding="utf-8") == (
        second / "leaderboard.json"
    ).read_text(encoding="utf-8")
