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


def test_the_one_unbuilt_subcommand_refuses_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # `detect` used to be here and is now implemented — see tests/test_cli_detect.py.
    assert main(["leaderboard", "--config", "board.toml", "--out", "board.json"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "glotscope leaderboard: not implemented in this release. Scheduled for M3.\n"
    )


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


def test_the_hub_path_explicitly_refuses_until_implemented() -> None:
    # `from_file` and `manifest` are implemented — see
    # tests/test_embeddings_loading.py. Resolving a checkpoint from the Hub is
    # not, and says so rather than returning something empty.
    with pytest.raises(NotImplementedError):
        Embeddings.from_checkpoint("acme/model", revision="deadbeef")
