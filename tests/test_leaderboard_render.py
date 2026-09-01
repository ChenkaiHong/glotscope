"""Rendering the board (PRD §16.1).

The renderer must never invent a number it was not given. Two cells carry that
weight: a skipped row, which has no numbers at all, and the Tier 2 column, which
§16.1 says must read ``n/a (tokenizer-only)`` rather than be left visually empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glotscope.leaderboard import TOKENIZER_ONLY, LeaderboardRow, RosterEntry, render_markdown


class _Board:
    """The narrow surface the renderer reads, built directly.

    Rendering is tested against hand-built rows rather than a real run: a
    renderer that only works on output it produced itself is untested against the
    case that matters, which is the row that failed.
    """

    def __init__(self, rows: tuple[LeaderboardRow, ...], **document: Any) -> None:
        self.rows = rows
        self._document = {
            "corpus": {
                "id": "flores_plus",
                "version": "2024.08",
                "split": "devtest",
                "languages": ["eng_Latn"],
                "sha256": "a" * 64,
            },
            "parameters": {
                "segmenter": None,
                "parity_reference": "eng_Latn",
                "renyi_alpha": 2.5,
                "renyi_normalizer": "observed",
                "normalization": "NFC",
                "leading_space": True,
                "add_special_tokens": False,
                "gini": True,
            },
            "glotscope_version": "0.1.0",
            "backend": "python",
            "published": sum(1 for row in rows if row.skipped is None),
            "skipped": sum(1 for row in rows if row.skipped is not None),
            **document,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._document, "rows": [row.to_dict() for row in self.rows]}


def _result(*, vocab: int, cpt: float) -> dict[str, Any]:
    """The shape a real run produces.

    Hand-built, and therefore only as good as the key paths it asserts — which
    is why ``test_a_real_run_fills_every_numeric_cell`` renders an actual
    document instead. Two columns shipped empty because this fixture guessed a
    nesting the document does not have, and every test here passed while the
    renderer could not read a real board.
    """
    return {
        "tier0": {"vocab_size": vocab, "ill_formed_vocab_rate": 0.0068, "family": "byte_level"},
        "tier1": {
            "per_language": {"eng_Latn": {"cpt": cpt}},
            "corpus_level": {
                "parity": {"worst_case_parity": 2.5, "worst_case_language": "shn_Mymr"},
                "gini": 0.12,
            },
        },
        "manifest": {"tokenizer": {"revision": "b" * 40, "vocab_size_tokenizer": vocab}},
    }


def test_a_tokenizer_only_row_says_so_in_its_tier_2_cell() -> None:
    board = _Board(
        (
            LeaderboardRow(
                entry=RosterEntry(id="tiktoken:o200k_base"),
                result=_result(vocab=200019, cpt=4.1),
            ),
        )
    )

    table = render_markdown(board.to_dict())

    assert TOKENIZER_ONLY in table
    assert "o200k_base" in table


def test_a_skipped_row_appears_with_its_reason_and_no_numbers() -> None:
    """A board that dropped skipped rows would look complete while being short,
    and a reader could not tell which model was missing."""
    board = _Board(
        (
            LeaderboardRow(
                entry=RosterEntry(id="google/gemma-2b", revision="c" * 40),
                skipped="TokenizerLoadError: gated; accept the terms and set HF_TOKEN",
            ),
        )
    )

    table = render_markdown(board.to_dict())

    assert "gemma-2b" in table
    assert "skipped" in table.lower()
    assert "HF_TOKEN" in table


def test_a_multi_line_skip_reason_stays_in_one_table_row() -> None:
    """A Hub 404 embeds a blank line in its message. Spliced into a cell as it
    came, it terminated the CommonMark table, and every row after it rendered
    as loose paragraphs — which is what the first published board did for its
    three SentencePiece-only rows."""
    board = _Board(
        (
            LeaderboardRow(
                entry=RosterEntry(id="google/mt5-base", revision="c" * 40),
                skipped=(
                    "TokenizerLoadError: 404 (Request ID: Root=1-abc)\n\n"
                    "Entry Not Found for url: x | y"
                ),
            ),
        )
    )

    table = render_markdown(board.to_dict())

    row = next(line for line in table.splitlines() if "mt5-base" in line)
    assert row.startswith("| ") and row.endswith(" |")
    assert "Entry Not Found" in row
    # The pipe inside the reason is escaped, so the row still has its eight
    # columns rather than a ninth.
    assert len(row.split(" | ")) == 8


def test_a_mirror_row_is_visibly_labelled() -> None:
    """§11: a leaderboard silently using unofficial re-uploads is a line of
    attack, so the label has to be in the rendered table and not only in the
    JSON a reader may never open."""
    board = _Board(
        (
            LeaderboardRow(
                entry=RosterEntry(
                    id="unsloth/Meta-Llama-3.1-8B-Instruct",
                    revision="d" * 40,
                    is_mirror=True,
                    note="ungated re-upload of a manually gated repository",
                ),
                result=_result(vocab=128256, cpt=3.9),
            ),
        )
    )

    table = render_markdown(board.to_dict())

    assert "mirror" in table.lower()
    assert "ungated re-upload" in table


def test_the_header_records_what_the_board_was_computed_under() -> None:
    """A table without its parameters is a table nobody can compare against."""
    board = _Board(())

    table = render_markdown(board.to_dict())

    assert "flores_plus" in table
    assert "devtest" in table
    assert "2.5" in table  # the Renyi alpha, which §7.5 requires recorded


def test_the_caveat_travels_with_the_table() -> None:
    """§7 carries contradicting evidence on downstream quality in three places.
    A ranked table is exactly the artifact that invites the causal reading, so
    the disclaimer is part of the rendering rather than a docs page."""
    board = _Board(())

    table = render_markdown(board.to_dict())

    assert "diagnostic" in table.lower()
    assert "quality" in table.lower()


def test_a_missing_measurement_renders_as_a_dash_not_a_zero() -> None:
    """The one thing a renderer must never do is invent a number. A row whose
    result carries no Tier 1 block has nothing to show, and 0.000 would read as
    a measurement that came out at zero."""
    board = _Board(
        (
            LeaderboardRow(
                entry=RosterEntry(id="tiktoken:toy", note="Tier 0 only"),
                result={"tier0": {"vocab_size": 258}, "manifest": {}},
            ),
        )
    )

    table = render_markdown(board.to_dict())
    row = next(line for line in table.splitlines() if line.startswith("| tiktoken:toy"))

    assert row.count("—") >= 3
    assert "0.000" not in row
    assert "Tier 0 only" in row


def test_a_row_with_no_result_at_all_still_renders() -> None:
    """Neither run nor skipped is a state the runner does not produce — but the
    renderer is given documents from disk, including ones written by an older
    release, and a crash there loses the whole table rather than one row."""
    board = _Board((LeaderboardRow(entry=RosterEntry(id="tiktoken:toy")),))

    table = render_markdown(board.to_dict())

    assert "tiktoken:toy" in table


def test_an_empty_per_language_block_does_not_average_nothing() -> None:
    board = _Board(
        (
            LeaderboardRow(
                entry=RosterEntry(id="tiktoken:toy"),
                result={"tier0": {"vocab_size": 258}, "tier1": {"per_language": {}}},
            ),
        )
    )

    assert "nan" not in render_markdown(board.to_dict()).lower()


def test_a_real_run_fills_every_numeric_cell(tmp_path: Path, monkeypatch: Any) -> None:
    """Render a document the runner actually produced, not one written here.

    This is the test the hand-built fixtures above could not be: they assert the
    key paths *this file* believes in, so a renderer reading the wrong path
    passes every one of them and still emits a table of dashes. That is exactly
    what happened — CPT and parity shipped empty because the fixture nested
    `cpt` under a `compression` block the document does not have.
    """
    import json

    import tiktoken

    from glotscope.corpus import Corpus
    from glotscope.leaderboard import load_config, run_leaderboard

    ranks = {bytes([value]): value for value in range(256)}
    ranks[b"the"] = 256
    encoding = tiktoken.Encoding(
        name="toy", pat_str=r"\s+|\S+", mergeable_ranks=ranks, special_tokens={"<|end|>": 257}
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)

    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    directory = tmp_path / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True)
    (directory / "eng_Latn.txt").write_text("The cat sat.\nIt rained.\n", encoding="utf-8")
    (directory / "hin_Deva.txt").write_text("बिल्ली बैठी।\nबारिश हुई।\n", encoding="utf-8")

    config_path = tmp_path / "board.json"
    config_path.write_text(
        json.dumps(
            {
                "corpus": {"id": "flores_plus", "languages": ["eng_Latn", "hin_Deva"]},
                "parameters": {"parity_reference": "eng_Latn", "gini": True},
                "roster": [{"id": "tiktoken:toy"}],
            }
        ),
        encoding="utf-8",
    )

    board = run_leaderboard(load_config(config_path), corpus_root=tmp_path)
    row = next(
        line
        for line in render_markdown(board.to_dict()).splitlines()
        if line.startswith("| tiktoken:toy")
    )

    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert "—" not in cells[1:6], f"a real run left a measurement cell empty: {cells}"
