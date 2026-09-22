"""Re-checking a published board (PRD §12.3, §16.1).

§16.1 requires a nightly re-run against pinned revisions that **fails if any
published number moves**. Pinning does not make that redundant, which is the
thing worth stating: a tiktoken encoding is pinned by the installed library
version rather than by content, a repository can be deleted or newly gated under
a revision that still resolves, and our own code can change a number without
anyone noticing.

What must *not* fail the check is everything that legitimately varies — the
environment, the glotscope version, the backend. `verify` already drew that line
for a single result, and drawing it differently here would make a nightly job
that goes red on a release rather than on a regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glotscope.corpus import Corpus
from glotscope.leaderboard import check_board, load_config, run_leaderboard


@pytest.fixture
def toy_encoding(monkeypatch: pytest.MonkeyPatch) -> Any:
    import tiktoken

    ranks = {bytes([value]): value for value in range(256)}
    ranks[b"the"] = 256
    encoding = tiktoken.Encoding(
        name="toy", pat_str=r"\s+|\S+", mergeable_ranks=ranks, special_tokens={"<|end|>": 257}
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)
    return encoding


def _board(tmp_path: Path) -> dict[str, Any]:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    directory = tmp_path / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "eng_Latn.txt").write_text("The cat sat.\nIt rained.\n", encoding="utf-8")
    (directory / "hin_Deva.txt").write_text("बिल्ली बैठी।\nबारिश हुई।\n", encoding="utf-8")

    config = tmp_path / "board.json"
    config.write_text(
        json.dumps(
            {
                "corpus": {"id": "flores_plus", "languages": ["eng_Latn", "hin_Deva"]},
                "parameters": {"parity_reference": "eng_Latn", "gini": True},
                "roster": [{"id": "tiktoken:toy"}],
            }
        ),
        encoding="utf-8",
    )
    return run_leaderboard(load_config(config), corpus_root=tmp_path).to_dict()


def test_a_board_checked_against_itself_reports_no_movement(
    tmp_path: Path, toy_encoding: Any
) -> None:
    published = _board(tmp_path)

    assert check_board(published, published) == []


def test_a_moved_number_is_reported_with_both_values(tmp_path: Path, toy_encoding: Any) -> None:
    """The point of the job. A silent upstream change is invisible in a table;
    naming the path and both values is what makes it actionable."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"][0]["result"]["tier0"]["vocab_size"] = 999

    differences = check_board(published, regenerated)

    assert len(differences) == 1
    assert "vocab_size" in differences[0]
    assert "999" in differences[0]


def test_the_environment_is_not_compared(tmp_path: Path, toy_encoding: Any) -> None:
    """Recorded *because* it varies. Comparing it would make the nightly job go
    red every time a runner image changes, which trains everyone to ignore it."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["environment"] = {"python": "9.9.9", "platform": "elsewhere"}
    regenerated["rows"][0]["result"]["backend"] = "rust"
    regenerated["rows"][0]["result"]["glotscope_version"] = "99.0.0"
    regenerated["glotscope_version"] = "99.0.0"

    assert check_board(published, regenerated) == []


def test_a_schema_change_is_reported(tmp_path: Path, toy_encoding: Any) -> None:
    """A schema change changes the document, which is precisely what a reader
    comparing two published boards needs told."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"][0]["result"]["schema_version"] = "9.9"

    assert check_board(published, regenerated)


def test_a_row_that_stopped_running_is_reported(tmp_path: Path, toy_encoding: Any) -> None:
    """A repository deleted or newly gated under a revision that still resolves
    is exactly what the nightly job exists to catch, and it shows up as a row
    that used to publish and now skips."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"][0]["result"] = None
    regenerated["rows"][0]["skipped"] = "TokenizerLoadError: gated"

    differences = check_board(published, regenerated)

    assert any("stopped publishing" in difference for difference in differences)


def test_a_row_that_started_running_is_reported_too(tmp_path: Path, toy_encoding: Any) -> None:
    """Not an error, but not silent either: the published board no longer
    describes what the tool now produces, and someone has to regenerate it."""
    published = _board(tmp_path)
    published["rows"][0]["result"] = None
    published["rows"][0]["skipped"] = "TokenizerLoadError: gated"
    regenerated = _board(tmp_path)

    assert any("now publishes" in difference for difference in check_board(published, regenerated))


def test_a_row_that_disappeared_from_the_roster_is_reported(
    tmp_path: Path, toy_encoding: Any
) -> None:
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"] = []

    assert any("missing" in difference for difference in check_board(published, regenerated))


def test_tier_0_only_comparison_ignores_the_corpus_columns(
    tmp_path: Path, toy_encoding: Any
) -> None:
    """The nightly job runs anonymously where FLORES+ is gated, so it can
    regenerate Tier 0 and nothing else. Comparing only what was recomputed is
    what keeps that run honest: a Tier 1 column it never measured must not be
    reported as unchanged *or* as moved."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"][0]["result"].pop("tier1")
    regenerated["rows"][0]["result"]["tier0"]["vocab_size"] = 12345

    differences = check_board(published, regenerated, tiers=("tier0",))

    assert len(differences) == 1
    assert "vocab_size" in differences[0]


def test_a_tier_0_run_reads_no_corpus_at_all(tmp_path: Path, toy_encoding: Any) -> None:
    """What makes the nightly job possible on an anonymous runner.

    FLORES+ is gated, so there is no corpus to read there. This was not merely
    untested but broken: the runner loaded the corpus before the loop
    unconditionally, so the nightly job would have failed on a missing corpus
    rather than on a moved number — a red job that says nothing about the board.
    """
    config_path = tmp_path / "board.json"
    config_path.write_text(
        json.dumps(
            {
                "corpus": {"id": "flores_plus", "languages": ["eng_Latn", "hin_Deva"]},
                "parameters": {"parity_reference": "eng_Latn"},
                "roster": [{"id": "tiktoken:toy"}],
            }
        ),
        encoding="utf-8",
    )

    board = run_leaderboard(
        load_config(config_path), corpus_root=tmp_path / "no-corpus-here", tiers=("tier0",)
    )

    row = board.rows[0]
    assert row.skipped is None
    assert row.result is not None
    assert "tier0" in row.result
    assert "tier1" not in row.result
    # The manifest does not claim a corpus this run never opened.
    assert row.result["manifest"].get("corpus") is None
    assert board.corpus_sha256 == ""


def test_a_tier_0_run_checks_clean_against_a_full_board(tmp_path: Path, toy_encoding: Any) -> None:
    """The nightly job compares a Tier 0 run against a board that carries Tier 1
    columns, and must report nothing moved — the missing tiers were not measured,
    not changed."""
    published = _board(tmp_path)

    config_path = tmp_path / "board.json"
    regenerated = run_leaderboard(
        load_config(config_path), corpus_root=tmp_path / "gone", tiers=("tier0",)
    ).to_dict()

    assert check_board(published, regenerated, tiers=("tier0",)) == []


def test_a_key_only_one_side_carries_is_reported(tmp_path: Path, toy_encoding: Any) -> None:
    """Inside a tier both boards carry, a field appearing or vanishing is a
    change to the document and is reported as one."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"][0]["result"]["tier0"]["a_new_field"] = 1
    del regenerated["rows"][0]["result"]["tier0"]["vocab_size"]

    differences = check_board(published, regenerated)

    assert any("absent from the published board" in d for d in differences)
    assert any("not regenerated" in d for d in differences)


def test_a_list_that_changed_length_is_reported(tmp_path: Path, toy_encoding: Any) -> None:
    """An unreachable-id list gaining an entry is a moved number even though no
    scalar changed."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"][0]["result"]["tier0"]["non_utf8_byte_values"] = [1, 2, 3]

    assert check_board(published, regenerated)


def test_a_row_the_published_board_never_had_is_reported(tmp_path: Path, toy_encoding: Any) -> None:
    """A roster gaining a row without the board being regenerated: not a
    regression, but the published board no longer describes the configuration."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    extra = json.loads(json.dumps(regenerated["rows"][0]))
    extra["id"] = "tiktoken:other"
    regenerated["rows"].append(extra)

    assert any("not in the published board" in d for d in check_board(published, regenerated))


def test_a_reworded_warning_is_not_a_moved_number(tmp_path: Path, toy_encoding: Any) -> None:
    """Warnings carry provenance commentary — which link of the reference-set
    chain supplied a set, whether a revision was pinned — and their wording is
    ours to improve. A rewording must not turn the nightly job red."""
    published = _board(tmp_path)
    regenerated = json.loads(json.dumps(published))
    regenerated["rows"][0]["result"]["tier1"]["warnings"] = ["reworded, same meaning"]
    published["rows"][0]["result"]["tier1"]["warnings"] = ["original wording"]

    assert check_board(published, regenerated) == []


def test_a_row_skipped_on_both_sides_is_not_a_difference(tmp_path: Path, toy_encoding: Any) -> None:
    """A permanently unreachable row — a SentencePiece-only repository, say —
    skips every night. Reporting that as movement would make the job red
    forever, which is the same as having no job."""
    published = _board(tmp_path)
    published["rows"][0]["result"] = None
    published["rows"][0]["skipped"] = "TokenizerLoadError: no tokenizer.json"
    regenerated = json.loads(json.dumps(published))

    assert check_board(published, regenerated) == []
