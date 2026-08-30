"""Running the leaderboard (PRD §16.1, §11).

Every roster row here is a tiktoken encoding or a local file, and the corpus is
written into ``tmp_path``, so the whole board runs offline. That is not only
convenience: §11's roster spans gated repositories, and a test suite that needed
them would be a suite only one machine can run.

The behaviour under test is what happens when a row *cannot* run. §11 says gated
resources skip with a message and never fail the run — so a board with one
unreachable row must still publish, and must say which row is missing and why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.corpus import Corpus
from glotscope.errors import TokenizerLoadError
from glotscope.leaderboard import load_config, run_leaderboard
from glotscope.lint import byte_to_unicode

_ENGLISH = ("The cat sat on the mat.", "It rained all afternoon.")
_HINDI = ("बिल्ली चटाई पर बैठी।", "दोपहर भर बारिश हुई।")


@pytest.fixture
def toy_encoding(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A synthetic tiktoken encoding, so no row touches a vendor CDN."""
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


def _corpus_root(tmp_path: Path) -> Path:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    directory = tmp_path / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True)
    for language, lines in (("eng_Latn", _ENGLISH), ("hin_Deva", _HINDI)):
        (directory / f"{language}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
    return tmp_path


def _local_tokenizer(tmp_path: Path) -> Path:
    mapping = byte_to_unicode()
    backend = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path / "local.json"
    backend.save(str(path))
    return path


def _config(tmp_path: Path, roster: list[dict[str, Any]]) -> Any:
    document = {
        "version": 1,
        "corpus": {"id": "flores_plus", "languages": ["eng_Latn", "hin_Deva"]},
        "parameters": {"parity_reference": "eng_Latn", "gini": True, "renyi_alpha": 2.5},
        "roster": roster,
    }
    path = tmp_path / "leaderboard.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_config(path)


def test_every_row_carries_its_own_manifest(tmp_path: Path, toy_encoding: Any) -> None:
    """§16.1: every row carries its manifest. A table of numbers whose rows
    cannot be traced to an artifact is the thing this library exists not to be."""
    root = _corpus_root(tmp_path)
    config = _config(tmp_path, [{"id": "tiktoken:toy"}, {"id": str(_local_tokenizer(tmp_path))}])

    board = run_leaderboard(config, corpus_root=root)

    assert len(board.rows) == 2
    for row in board.rows:
        assert row.skipped is None
        assert row.result is not None
        manifest = row.result["manifest"]["tokenizer"]
        assert manifest["tokenizer_json_sha256"]
        assert manifest["revision"]


def test_an_unreachable_row_skips_and_the_board_still_publishes(
    tmp_path: Path, toy_encoding: Any
) -> None:
    """§11: gated resources skip with a message and never fail the run."""
    root = _corpus_root(tmp_path)
    config = _config(
        tmp_path,
        [{"id": "tiktoken:toy"}, {"id": str(tmp_path / "absent.json")}],
    )

    board = run_leaderboard(config, corpus_root=root)

    first, second = board.rows
    assert first.skipped is None
    assert second.skipped is not None
    assert "absent.json" in second.skipped
    assert board.published == 1
    assert board.skipped == 1


def test_a_broken_corpus_fails_the_run_rather_than_skipping(
    tmp_path: Path, toy_encoding: Any
) -> None:
    """A row that cannot load is one row; a corpus that cannot load is every
    number on the board. Skipping it would publish an empty table as a success."""
    config = _config(tmp_path, [{"id": "tiktoken:toy"}])

    with pytest.raises(Exception) as caught:
        run_leaderboard(config, corpus_root=tmp_path / "nothing")

    assert "eng_Latn" in str(caught.value)


def test_a_tokenizer_only_row_says_so_rather_than_leaving_tier_2_empty(
    tmp_path: Path, toy_encoding: Any
) -> None:
    """§16.1 is explicit: roughly half the roster is tokenizer-only, and the
    Tier 2 column must say so rather than look like a missing measurement."""
    root = _corpus_root(tmp_path)
    config = _config(tmp_path, [{"id": "tiktoken:toy"}])

    row = run_leaderboard(config, corpus_root=root).rows[0]

    assert row.tier2_status == "n/a (tokenizer-only)"
    assert row.result is not None
    assert "tier2" not in row.result


def test_the_board_records_what_it_was_computed_under(tmp_path: Path, toy_encoding: Any) -> None:
    """One parameter set for the whole board, recorded once. ``compare`` refuses
    to table results computed under different parameters, and a leaderboard is
    that comparison rendered."""
    root = _corpus_root(tmp_path)
    config = _config(tmp_path, [{"id": "tiktoken:toy"}])

    board = run_leaderboard(config, corpus_root=root)
    document = board.to_dict()

    assert document["corpus"]["id"] == "flores_plus"
    assert document["corpus"]["languages"] == ["eng_Latn", "hin_Deva"]
    assert document["parameters"]["parity_reference"] == "eng_Latn"
    assert document["glotscope_version"]
    assert len(document["rows"]) == 1


def test_a_mirror_row_travels_labelled(tmp_path: Path, toy_encoding: Any) -> None:
    """The flag has to reach the *document*, not only the config. A reader of
    results/leaderboard.json cannot see the file it was generated from."""
    root = _corpus_root(tmp_path)
    config = _config(
        tmp_path,
        [
            {
                "id": str(_local_tokenizer(tmp_path)),
                "is_mirror": True,
                "note": "stand-in for an ungated re-upload",
            }
        ],
    )

    row = run_leaderboard(config, corpus_root=root).to_dict()["rows"][0]

    assert row["is_mirror"] is True
    assert row["note"]


def test_the_corpus_is_read_once_for_the_whole_board(
    tmp_path: Path, toy_encoding: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """221 varieties by 1012 documents is the real shape. Re-reading it per row
    would make an eighteen-model board eighteen corpus reads."""
    root = _corpus_root(tmp_path)
    config = _config(tmp_path, [{"id": "tiktoken:toy"}, {"id": str(_local_tokenizer(tmp_path))}])

    loads = 0
    original = Corpus.load

    def counting(self: Corpus, *args: Any, **kwargs: Any) -> Any:
        nonlocal loads
        loads += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Corpus, "load", counting)
    run_leaderboard(config, corpus_root=root)

    assert loads == 1


def test_a_row_that_raises_an_unexpected_error_is_not_swallowed(
    tmp_path: Path, toy_encoding: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping is for artifacts that cannot be reached, not for bugs. A board
    that swallowed a TypeError would publish a shorter table and call it a
    gated-resource skip."""
    root = _corpus_root(tmp_path)
    config = _config(tmp_path, [{"id": "tiktoken:toy"}])

    def exploding(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("a bug, not a missing artifact")

    monkeypatch.setattr("glotscope.leaderboard.run.load_tokenizer", exploding)

    with pytest.raises(TypeError):
        run_leaderboard(config, corpus_root=root)


def test_a_typed_load_refusal_is_reported_with_its_reason(
    tmp_path: Path, toy_encoding: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _corpus_root(tmp_path)
    config = _config(tmp_path, [{"id": "tiktoken:toy"}])

    def refusing(*args: Any, **kwargs: Any) -> Any:
        raise TokenizerLoadError("toy", "gated: accept the terms and set HF_TOKEN")

    monkeypatch.setattr("glotscope.leaderboard.run.load_tokenizer", refusing)

    row = run_leaderboard(config, corpus_root=root).rows[0]

    assert row.skipped is not None
    assert "HF_TOKEN" in row.skipped


def test_a_row_with_weights_is_measured_at_tier_2(
    tmp_path: Path, toy_encoding: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The board spans the tiers a row can reach. Tier 2 is the differentiator
    (§2.2), so a row that declares weights must carry a tier2 block rather than
    the tokenizer-only label."""
    import numpy as np

    from glotscope.embeddings import Embeddings

    root = _corpus_root(tmp_path)
    config = _config(tmp_path, [{"id": "tiktoken:toy", "weights": "acme/model"}])

    rows = 258
    generator = np.random.default_rng(0)
    embeddings = Embeddings(
        e_in=generator.normal(size=(rows, 8)).astype(np.float32),
        e_out=None,
        tied=True,
        dtype="float32",
        shard_sha256="f" * 64,
        checkpoint="acme/model",
        n_rows=rows,
        vocab_size=rows,
    )
    monkeypatch.setattr("glotscope.leaderboard.run.load_embeddings", lambda *a, **k: embeddings)

    row = run_leaderboard(config, corpus_root=root).rows[0]

    assert row.skipped is None
    assert row.tier2_status == "measured"
    assert row.result is not None
    assert "tier2" in row.result
