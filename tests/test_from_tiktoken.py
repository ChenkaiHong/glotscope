"""Loading an OpenAI encoding (PRD §8.1, §9, §11).

Every test builds a **synthetic** ``tiktoken.Encoding`` in process and patches
``tiktoken.get_encoding`` to return it. Nothing here calls the real registry,
because ``tiktoken.get_encoding("o200k_base")`` downloads its BPE ranks from
``openaipublic.blob.core.windows.net`` on a cold cache — and a suite that fetches
from a vendor CDN is a suite that goes red when that CDN does. The same reasoning
put the Hub behind a mock in ``tests/test_from_pretrained.py``.

Provenance is what needs asserting. A tiktoken encoding has no ``tokenizer.json``
and no commit to pin, so §9's two identity fields have to be filled from
something else: a digest over the four things that determine every number the
encoding can produce, and the version of the library that defines them.
"""

from __future__ import annotations

import builtins
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from glotscope.enums import Algorithm, Segmenter, TokenizerFamily
from glotscope.errors import TokenizerLoadError
from glotscope.lint import byte_to_unicode
from glotscope.morphology import pieces_from_offsets
from glotscope.tokenizer import Tokenizer

_TOY_PATTERN = r"""\s+|\S+"""
"""Whitespace-or-run. Not GPT-2's regex — the split rule is irrelevant to every
assertion here, and a short one keeps the expected token sequences readable."""


def _toy_ranks() -> dict[bytes, int]:
    """All 256 single bytes, plus one merge. Byte coverage is deliberate: §7.8's
    ``byte_fallback_coverage`` counts exactly this, and a vocabulary missing
    bytes would make the interesting assertion accidental."""
    ranks = {bytes([value]): value for value in range(256)}
    ranks[b"ab"] = 256
    return ranks


def _encoding(
    *,
    name: str = "toy",
    pat_str: str = _TOY_PATTERN,
    ranks: Mapping[bytes, int] | None = None,
    specials: Mapping[str, int] | None = None,
) -> Any:
    import tiktoken

    return tiktoken.Encoding(
        name=name,
        pat_str=pat_str,
        mergeable_ranks=dict(ranks if ranks is not None else _toy_ranks()),
        special_tokens=dict(specials if specials is not None else {"<|end|>": 257}),
    )


def _patch_registry(monkeypatch: pytest.MonkeyPatch, encoding: Any) -> list[str]:
    """Patch the registry seam, recording every name asked for."""
    import tiktoken

    asked: list[str] = []

    def _get_encoding(name: str) -> Any:
        asked.append(name)
        if name != encoding.name:
            raise ValueError(f"Unknown encoding {name}")
        return encoding

    monkeypatch.setattr(tiktoken, "get_encoding", _get_encoding)
    return asked


def _version() -> str:
    import importlib.metadata

    return importlib.metadata.version("tiktoken")


# -- provenance -------------------------------------------------------------


def test_manifest_records_the_encoding_rather_than_inventing_a_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _patch_registry(monkeypatch, _encoding())

    tok = Tokenizer.from_tiktoken("toy")

    assert asked == ["toy"]
    manifest = tok.manifest
    assert manifest.id == "toy"
    assert manifest.source == "tiktoken"
    assert manifest.source_is_mirror is False
    assert manifest.algorithm is Algorithm.TIKTOKEN
    assert manifest.revision == f"tiktoken/{_version()}"
    assert manifest.vocab_size_tokenizer == 258
    # No config.json and no weights, so both stay null rather than being
    # back-filled — equal values are a claim §7.9's chain would read at link 2.
    assert manifest.vocab_size_config is None
    assert manifest.embedding_rows is None


def test_the_warning_says_identity_rests_on_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_registry(monkeypatch, _encoding())

    tok = Tokenizer.from_tiktoken("toy")

    joined = " ".join(tok.warnings)
    assert "tokenizer_json_sha256" in joined
    assert "tiktoken" in joined


def test_digest_covers_every_input_that_changes_a_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranks, specials, the split pattern and the vocabulary size all move it.

    Anything that changes what the encoding does must change its recorded
    identity, or two different tokenizers table as the same artifact.
    """

    def digest_of(**kwargs: Any) -> str:
        _patch_registry(monkeypatch, _encoding(**kwargs))
        return Tokenizer.from_tiktoken(kwargs.get("name", "toy")).manifest.tokenizer_json_sha256

    baseline = digest_of()
    assert digest_of() == baseline, "the digest must not depend on when it was taken"

    changed_ranks = _toy_ranks()
    changed_ranks[b"ba"] = 257
    assert digest_of(ranks=changed_ranks, specials={"<|end|>": 258}) != baseline
    assert digest_of(pat_str=r"\S+|\s+") != baseline
    assert digest_of(specials={"<|eot|>": 257}) != baseline


def test_the_digest_does_not_depend_on_the_encoding_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The name is recorded in ``id``; the digest is over the artifact.

    Two names for identical ranks *are* the same tokenizer, and a digest that
    disagreed would make ``verify`` reject a document that reproduces exactly.
    """
    _patch_registry(monkeypatch, _encoding(name="toy"))
    first = Tokenizer.from_tiktoken("toy").manifest.tokenizer_json_sha256
    _patch_registry(monkeypatch, _encoding(name="other"))
    second = Tokenizer.from_tiktoken("other").manifest.tokenizer_json_sha256

    assert first == second


def test_an_unknown_encoding_is_a_typed_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_registry(monkeypatch, _encoding())

    with pytest.raises(TokenizerLoadError) as caught:
        Tokenizer.from_tiktoken("o200k_nope")

    assert "o200k_nope" in str(caught.value)


def test_a_missing_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A core install promises Tier 0 and Tier 1 and must say which extra is
    missing, not fail with a bare ModuleNotFoundError from an import line."""
    real_import = builtins.__import__

    def _no_tiktoken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tiktoken":
            raise ModuleNotFoundError("No module named 'tiktoken'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(__import__("sys").modules, "tiktoken", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_tiktoken)

    with pytest.raises(ModuleNotFoundError) as caught:
        Tokenizer.from_tiktoken("o200k_base")

    message = str(caught.value)
    assert "tiktoken" in message
    assert "glotscope[tiktoken]" in message


# -- Tier 0 -----------------------------------------------------------------


def test_lint_reads_the_bytes_the_encoding_actually_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tiktoken keys its vocabulary by ``bytes``, so §7.8 needs no recovery
    heuristic: byte-level coverage is countable rather than inferred."""
    _patch_registry(monkeypatch, _encoding())

    report = Tokenizer.from_tiktoken("toy").lint()

    assert report.family is TokenizerFamily.BYTE_LEVEL
    assert report.byte_fallback_coverage == 256
    # 128 of the 256 single-byte entries are lone non-ASCII bytes and cannot
    # stand alone as UTF-8; b"ab" and the special token are well-formed.
    assert report.partial_utf8_tokens
    assert 257 in report.special_tokens


def test_a_lone_continuation_byte_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``decode`` replaces an unpaired byte with U+FFFD, so re-encoding cannot
    return the id it came from. That is the §7.8 signal, not a bug to paper."""
    _patch_registry(monkeypatch, _encoding())

    report = Tokenizer.from_tiktoken("toy").lint()

    assert 0x80 in report.unreachable_tokens


def test_the_adapter_answers_the_whole_tokenizers_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every method and keyword glotscope reaches for, including the ones no
    current call site passes.

    A keyword accepted but ignored is the failure this locks down: ``tokenizers``
    answers ``with_added_tokens=False`` and ``skip_special_tokens=True`` with a
    *different* vocabulary, and an adapter that quietly returned the same one
    would put special tokens into a denominator that excludes them.
    """
    _patch_registry(monkeypatch, _encoding())
    backend = Tokenizer.from_tiktoken("toy")._backend

    assert backend.get_vocab_size(with_added_tokens=True) == 258
    assert backend.get_vocab_size(with_added_tokens=False) == 257
    assert "<|end|>" in backend.get_vocab(with_added_tokens=True)
    assert "<|end|>" not in backend.get_vocab(with_added_tokens=False)

    assert backend.token_to_id("<|end|>") == 257
    assert backend.token_to_id("nothing spells this") is None

    assert backend.decode([257], skip_special_tokens=False) == "<|end|>"
    assert backend.decode([257], skip_special_tokens=True) == ""
    assert backend.decode_batch([[257], [99]], skip_special_tokens=True) == ["", "c"]

    (empty,) = backend.encode_batch([""], add_special_tokens=False)
    assert empty.ids == []
    assert empty.offsets == []


def test_add_special_tokens_is_accepted_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tiktoken has no post-processor template, so no encoding here gains a BOS
    or an EOS. Saying so beats a keyword that looks honoured."""
    _patch_registry(monkeypatch, _encoding())
    backend = Tokenizer.from_tiktoken("toy")._backend

    (without,) = backend.encode_batch(["ab"], add_special_tokens=False)
    (with_them,) = backend.encode_batch(["ab"], add_special_tokens=True)

    assert without.ids == with_them.ids == [256]


# -- Tier 1 -----------------------------------------------------------------


def test_offsets_claim_no_boundary_inside_a_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A character split across two tokens must yield one piece and one empty
    one — counting sub-character splits as boundaries is what inflates §7.7
    recall on non-Latin scripts."""
    _patch_registry(monkeypatch, _encoding())
    tok = Tokenizer.from_tiktoken("toy")

    (encoding,) = tok._backend.encode_batch(["語"], add_special_tokens=False)

    assert len(encoding.ids) == 3, "three raw bytes, so three tokens"
    assert pieces_from_offsets("語", encoding.offsets) == ("語",)


def test_offsets_tile_a_multi_token_word(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_registry(monkeypatch, _encoding())
    tok = Tokenizer.from_tiktoken("toy")

    (encoding,) = tok._backend.encode_batch(["abc"], add_special_tokens=False)

    assert pieces_from_offsets("abc", encoding.offsets) == ("ab", "c")


def test_offsets_agree_with_the_tokenizers_backend_character_for_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same vocabulary, loaded through both libraries, must report the same
    spans.

    This is the assertion that keeps a leaderboard honest. tiktoken publishes no
    offsets, so they are reconstructed here — and a reconstruction that put the
    empty piece on the other side of a split character would score identically
    under ``pieces_from_offsets`` while disagreeing on every span it reported.
    Both backends hold all 256 bytes plus one merge, and the strings are
    single words so the two libraries' split rules cannot separate them.
    """
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers import decoders, models, pre_tokenizers

    mapping = byte_to_unicode()
    reference = BackendTokenizer(
        models.BPE(
            vocab={**{mapping[value]: value for value in range(256)}, "ab": 256},
            merges=[("a", "b")],
        )
    )
    reference.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    reference.decoder = decoders.ByteLevel()

    _patch_registry(monkeypatch, _encoding())
    adapter = Tokenizer.from_tiktoken("toy")._backend

    words = ("abc", "語", "aば", "日本語", "ábc", "🎯", "ｦｧabｨ")
    theirs = reference.encode_batch(list(words), add_special_tokens=False)
    ours = adapter.encode_batch(words, add_special_tokens=False)

    for word, expected, actual in zip(words, theirs, ours, strict=True):
        assert list(actual.ids) == list(expected.ids), word
        assert actual.offsets == list(expected.offsets), word


def test_offsets_survive_an_unpaired_byte(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hand-built id list can hold a byte that is not valid UTF-8 alone.

    Decoding replaces it with U+FFFD, so the decoded text has *more* bytes than
    the tokens that produced it and the spans describe that decoded text. They
    stay in range and stop short of nothing — the replacement character is a
    character, and reporting it as one is what §7.8 says a lossy decode looks
    like.
    """
    _patch_registry(monkeypatch, _encoding())
    backend = Tokenizer.from_tiktoken("toy")._backend

    encoding = backend.encode_batch(["a"], add_special_tokens=False)[0]
    encoding.ids.append(0x80)

    assert encoding.offsets == [(0, 1), (1, 2)]
    assert backend.decode(encoding.ids) == "a\ufffd"


def test_analyze_produces_a_tier_1_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The adapter has to satisfy the Tier 1 path, not only Tier 0: fertility
    reads ids, and the compression family reads them per document."""
    from glotscope.corpus import Corpus

    _patch_registry(monkeypatch, _encoding())
    tok = Tokenizer.from_tiktoken("toy")

    corpus = Corpus.fineweb2(["eng_Latn"], version="sample-10BT")
    directory = tmp_path / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True)
    (directory / "eng_Latn.txt").write_text("ab ab c\nc ab\n", encoding="utf-8")

    report = tok.analyze(corpus.load(tmp_path), segmenter=Segmenter.WHITESPACE, leading_space=False)

    assert set(report.fertility) == {"eng_Latn"}
    assert all(value > 0 for value in report.fertility.values())
    metrics = report.per_language["eng_Latn"]
    assert metrics.compression.cpt > 0
    assert metrics.roundtrip_rate == 1.0, "byte-level encodings are lossless"


# -- CLI --------------------------------------------------------------------


def test_cli_routes_the_tiktoken_prefix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from glotscope.cli import main

    _patch_registry(monkeypatch, _encoding())

    assert main(["lint", "tiktoken:toy"]) == 0

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["family"] == "byte_level"
    assert document["byte_fallback_coverage"] == 256
    # The provenance warnings reach the operator, not only the document.
    assert "tokenizer_json_sha256" in captured.err


def test_cli_refuses_a_revision_on_an_encoding(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1, not 2: the feature exists, and the argument is the wrong one.

    An encoding is defined by the installed library, not by a commit, so a
    revision beside it would pin nothing.
    """
    from glotscope.cli import main

    _patch_registry(monkeypatch, _encoding())

    assert main(["lint", "tiktoken:toy", "--revision", "a" * 40]) == 1
    assert "revision" in capsys.readouterr().err
