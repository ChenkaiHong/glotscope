from __future__ import annotations

import numpy as np
import pytest

import glotscope
from glotscope.cli import main
from glotscope.embeddings import ALLOWED_DTYPES, Embeddings


def test_cli_version_reports_package_version_and_selected_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--version"]) == 0

    captured = capsys.readouterr()
    assert captured.out == f"glotscope {glotscope.__version__} (backend: python)\n"
    assert captured.err == ""


def test_cli_without_a_command_prints_help_and_returns_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 1

    captured = capsys.readouterr()
    assert "usage: glotscope" in captured.out
    assert "Multilingual tokenizer diagnostics" in captured.out
    assert captured.err == ""


def test_every_subcommand_is_built() -> None:
    """`detect` was here, then `leaderboard`; both are implemented now — see
    tests/test_cli_detect.py and tests/test_cli_leaderboard.py.

    Exit 2 means *scheduled but not built*, and no subcommand is any more. The
    assertion is kept rather than deleted because the empty table is the claim:
    the whole §8.2 surface exists, and a subcommand added later without a handler
    would surface here rather than as a 2 nobody expected.
    """
    from glotscope.cli import _HANDLERS, _MILESTONES

    assert _MILESTONES == {}
    assert set(_HANDLERS) == {"lint", "analyze", "detect", "compare", "leaderboard", "verify"}


def test_a_scheduled_path_inside_a_built_command_still_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The distinction survives the last unbuilt subcommand.

    ``STANZA`` and ``UDPIPE`` are written as scheduled rather than refused, so a
    built command can still reach an unbuilt path — and that must stay a 2, which
    sends the reader to wait for a release rather than to fix their arguments.
    """
    from glotscope.cli import _HANDLERS

    def scheduled(args: object) -> int:
        raise NotImplementedError("stanza: the adapter is scheduled but not written")

    monkeypatch.setitem(_HANDLERS, "lint", scheduled)

    assert main(["lint", "anything"]) == 2
    assert "scheduled" in capsys.readouterr().err


def test_cli_rejects_an_unknown_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["unknown"])

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "invalid choice: 'unknown'" in captured.err


def test_embeddings_expose_padding_rows_without_treating_them_as_vocab() -> None:
    embeddings = Embeddings(
        e_in=np.empty((6, 2), dtype=np.float32),
        e_out=None,
        tied=True,
        dtype="float16",
        shard_sha256="a" * 64,
        checkpoint="acme/model",
        n_rows=6,
        vocab_size=4,
    )

    assert embeddings.padding_rows == (4, 5)
    assert embeddings.e_out is None
    assert embeddings.tied


def test_embeddings_report_no_padding_rows_when_dimensions_match() -> None:
    embeddings = Embeddings(
        e_in=np.empty((4, 2), dtype=np.float32),
        e_out=np.empty((4, 2), dtype=np.float32),
        tied=False,
        dtype="float32",
        shard_sha256="b" * 64,
        checkpoint="acme/model",
        n_rows=4,
        vocab_size=4,
    )

    assert embeddings.padding_rows == ()
    assert embeddings.e_out is not None
    assert not embeddings.tied


def test_embeddings_only_advertise_original_precision_float_dtypes() -> None:
    assert frozenset({"float16", "float32", "float64", "bfloat16"}) == ALLOWED_DTYPES
    assert {"int4", "int8", "uint8"}.isdisjoint(ALLOWED_DTYPES)


# `from_file`, `manifest` and `from_checkpoint` are all implemented now — see
# tests/test_embeddings_loading.py and tests/test_embeddings_checkpoint.py.
