"""Tier 0 vocabulary lint (PRD §7.8).

The fixtures are hand-built rather than downloaded: every count asserted here is
derivable from the UTF-8 grammar alone, so the tests stay hermetic and stay a
statement about the classifier rather than about whoever trained a real
tokenizer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope.enums import Algorithm, TokenClass, TokenizerFamily
from glotscope.errors import GlotscopeError
from glotscope.lint import (
    byte_to_unicode,
    classify_family,
    detect_algorithm,
    lint_backend,
    token_bytes,
    unreachable_ids,
)
from glotscope.tokenizer import Tokenizer

# Derivable from Unicode 16.0 Table 3-7, for a vocabulary holding all 256 bytes
# as separate entries: 0x00-0x7F decode alone; 0xC2-0xF4 are lead bytes and are
# therefore truncated forms; everything else is ill-formed without being partial.
_WELL_FORMED_BYTES = 128
_PARTIAL_BYTES = 115
_ILL_FORMED_NOT_PARTIAL_BYTES = 256 - _WELL_FORMED_BYTES - _PARTIAL_BYTES


def _write_byte_level(path: Path) -> BackendTokenizer:
    """A byte-level BPE holding all 256 byte values and one special token."""
    mapping = byte_to_unicode()
    vocab = {mapping[value]: value for value in range(256)}
    vocab["<s>"] = 256
    backend = BackendTokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<s>"])
    backend.save(str(path))
    return backend


def _write_byte_fallback(path: Path) -> BackendTokenizer:
    """A byte-fallback BPE: bytes appear as ``<0xNN>`` pieces, not as raw bytes."""
    vocab = {f"<0x{value:02X}>": value for value in range(256)}
    vocab["▁the"] = 256
    backend = BackendTokenizer(models.BPE(vocab=vocab, merges=[], byte_fallback=True))
    backend.decoder = decoders.ByteFallback()
    backend.save(str(path))
    return backend


def _write_wordpiece(path: Path) -> BackendTokenizer:
    vocab = {"[UNK]": 0, "the": 1, "##ing": 2, "é": 3}
    backend = BackendTokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    backend.save(str(path))
    return backend


def _spec(path: Path) -> dict[str, object]:
    loaded: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def test_byte_to_unicode_is_a_bijection_over_all_256_values() -> None:
    mapping = byte_to_unicode()

    assert len(mapping) == 256
    assert len(set(mapping.values())) == 256


def test_byte_level_token_bytes_recovers_the_original_byte() -> None:
    mapping = byte_to_unicode()

    for value in range(256):
        assert token_bytes(mapping[value], TokenizerFamily.BYTE_LEVEL) == bytes([value])


def test_byte_fallback_token_bytes_reads_the_hex_escape() -> None:
    assert token_bytes("<0xF5>", TokenizerFamily.BYTE_FALLBACK) == b"\xf5"
    assert token_bytes("▁the", TokenizerFamily.BYTE_FALLBACK) == b" the"


def test_code_point_token_bytes_is_plain_utf8() -> None:
    assert token_bytes("é", TokenizerFamily.CODE_POINT) == b"\xc3\xa9"


def test_family_and_algorithm_are_read_from_the_tokenizer_json(tmp_path: Path) -> None:
    byte_level = tmp_path / "byte_level.json"
    byte_fallback = tmp_path / "byte_fallback.json"
    wordpiece = tmp_path / "wordpiece.json"
    _write_byte_level(byte_level)
    _write_byte_fallback(byte_fallback)
    _write_wordpiece(wordpiece)

    assert classify_family(_spec(byte_level)) is TokenizerFamily.BYTE_LEVEL
    assert classify_family(_spec(byte_fallback)) is TokenizerFamily.BYTE_FALLBACK
    assert classify_family(_spec(wordpiece)) is TokenizerFamily.CODE_POINT

    assert detect_algorithm(_spec(byte_level)) is Algorithm.BYTE_LEVEL_BPE
    assert detect_algorithm(_spec(byte_fallback)) is Algorithm.BYTE_FALLBACK_BPE
    assert detect_algorithm(_spec(wordpiece)) is Algorithm.WORDPIECE


def test_unknown_model_type_is_recorded_as_unknown_rather_than_guessed() -> None:
    assert detect_algorithm({"model": {"type": "SomethingNew"}}) is Algorithm.UNKNOWN
    assert classify_family({"model": {"type": "SomethingNew"}}) is TokenizerFamily.CODE_POINT


def test_byte_level_lint_partitions_the_vocabulary_by_utf8_class(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    backend = _write_byte_level(path)

    report = lint_backend(backend, _spec(path))

    assert report.vocab_size == 257
    assert report.family is TokenizerFamily.BYTE_LEVEL
    assert report.token_classes[TokenClass.PARTIAL_UTF8] == _PARTIAL_BYTES
    assert report.token_classes[TokenClass.ILL_FORMED_NOT_PARTIAL] == _ILL_FORMED_NOT_PARTIAL_BYTES
    # The special token is the only well-formed entry beyond the 128 ASCII bytes.
    assert report.token_classes[TokenClass.WELL_FORMED] == _WELL_FORMED_BYTES + 1
    assert sum(report.token_classes.values()) == report.vocab_size
    assert report.byte_fallback_coverage == 256


def test_byte_level_lint_reports_the_special_token_and_ill_formed_rate(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    backend = _write_byte_level(path)

    report = lint_backend(backend, _spec(path))

    assert report.special_tokens == (256,)
    expected = (_PARTIAL_BYTES + _ILL_FORMED_NOT_PARTIAL_BYTES) / 257
    assert report.ill_formed_vocab_rate == expected


def test_non_ascii_single_bytes_are_unreachable_in_a_byte_level_vocab(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    backend = _write_byte_level(path)

    report = lint_backend(backend, _spec(path))

    # Decoding one non-ASCII byte alone yields U+FFFD, which never re-encodes to
    # the same id, so every such entry fails encode(decode(id)) == [id].
    assert set(report.unreachable_tokens) >= set(range(0x80, 0x100))
    assert 0x41 not in report.unreachable_tokens
    assert report.unreachable_count == len(report.unreachable_tokens)


def test_stage1_exclusion_stays_narrower_than_the_ill_formed_set(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    backend = _write_byte_level(path)

    report = lint_backend(backend, _spec(path))
    excluded = report.stage1_exclusions()

    # Continuation bytes and truncated lead bytes are partial UTF-8. C0 remains
    # ill-formed-not-partial and is excluded here only because it is unreachable.
    assert 0xC2 in excluded
    assert set(report.partial_utf8_tokens) == set(range(0x80, 0xC0)) | set(range(0xC2, 0xF5))


def test_byte_fallback_lint_finds_the_hex_escaped_bytes(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    backend = _write_byte_fallback(path)

    report = lint_backend(backend, _spec(path))

    assert report.family is TokenizerFamily.BYTE_FALLBACK
    assert report.byte_fallback_coverage == 256
    assert report.token_classes[TokenClass.PARTIAL_UTF8] == _PARTIAL_BYTES


def test_from_file_lints_and_records_provenance(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    _write_byte_level(path)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    tokenizer = Tokenizer.from_file(path)
    report = tokenizer.lint()
    manifest = tokenizer.manifest

    assert report.vocab_size == 257
    assert manifest.tokenizer_json_sha256 == expected_sha
    assert manifest.vocab_size_tokenizer == 257
    assert manifest.algorithm is Algorithm.BYTE_LEVEL_BPE
    assert manifest.source == "file"
    assert manifest.source_is_mirror is False


def test_from_file_leaves_unknown_manifest_fields_null_and_warns(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    _write_byte_level(path)

    tokenizer = Tokenizer.from_file(path)
    manifest = tokenizer.manifest

    # A bare tokenizer.json carries no config.json and no weights. Refusing to
    # invent these is the point: filling them from vocab_size_tokenizer would
    # silently claim the embedding matrix has no padding rows, and Tier 2's
    # reference-set chain reads exactly that gap.
    assert manifest.vocab_size_config is None
    assert manifest.embedding_rows is None
    assert manifest.to_dict()["vocab_size_config"] is None
    assert manifest.to_dict()["embedding_rows"] is None
    assert any("vocab_size_config" in warning for warning in tokenizer.warnings)


def test_from_file_records_no_upstream_revision(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    _write_byte_level(path)

    tokenizer = Tokenizer.from_file(path)

    assert tokenizer.manifest.revision == "local"
    assert any("revision" in warning for warning in tokenizer.warnings)


def test_from_file_accepts_an_explicit_identity_and_license(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    _write_byte_level(path)

    tokenizer = Tokenizer.from_file(path, tokenizer_id="fixture", license_spdx="Apache-2.0")

    assert tokenizer.manifest.id == "fixture"
    assert tokenizer.manifest.license_spdx == "Apache-2.0"


def test_byte_level_added_tokens_are_read_as_literal_text() -> None:
    # A byte-level vocabulary stores added tokens verbatim rather than
    # byte-mapped, and CJK characters are outside the map entirely. Force-decoding
    # them through the map would raise; reading them as UTF-8 is the honest answer.
    assert token_bytes("日本", TokenizerFamily.BYTE_LEVEL) == "日本".encode()
    assert token_bytes("", TokenizerFamily.BYTE_LEVEL) == b""


def test_algorithm_is_inferred_when_the_model_type_is_absent() -> None:
    assert detect_algorithm({}) is Algorithm.UNKNOWN
    assert detect_algorithm({"model": {}}) is Algorithm.UNKNOWN
    assert detect_algorithm({"model": {"type": 7}}) is Algorithm.UNKNOWN
    assert (
        detect_algorithm({"model": {"merges": []}, "decoder": {"type": "ByteLevel"}})
        is Algorithm.BYTE_LEVEL_BPE
    )
    assert detect_algorithm({"model": {"continuing_subword_prefix": "##"}}) is Algorithm.WORDPIECE
    # BPE without either byte convention has an unknown byte convention.
    assert detect_algorithm({"model": {"type": "BPE"}}) is Algorithm.UNKNOWN
    assert detect_algorithm({"model": {"type": "Unigram"}}) is Algorithm.UNIGRAM_LM


def test_unreachable_ids_on_an_empty_vocabulary_is_empty() -> None:
    assert unreachable_ids(object(), []) == ()


def test_wordpiece_lint_reports_the_code_point_family(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    backend = _write_wordpiece(path)

    report = lint_backend(backend, _spec(path))

    assert report.family is TokenizerFamily.CODE_POINT
    assert report.token_classes[TokenClass.WELL_FORMED] == 4
    assert report.ill_formed_vocab_rate == 0.0
    assert report.byte_fallback_coverage is None
    assert report.non_utf8_byte_values is None


def test_from_file_warns_when_the_algorithm_cannot_be_identified(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    backend = BackendTokenizer(models.WordLevel(vocab={"the": 0, "[UNK]": 1}, unk_token="[UNK]"))
    backend.save(str(path))

    tokenizer = Tokenizer.from_file(path)

    # WordLevel is a real, loadable model with no §9 algorithm of its own. It is
    # recorded as unknown and warned about rather than filed under a neighbouring
    # algorithm, because Tier 2's scope limits read this field.
    assert tokenizer.manifest.algorithm is Algorithm.UNKNOWN
    assert any("unknown" in warning for warning in tokenizer.warnings)


def test_from_file_refuses_a_malformed_document(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(GlotscopeError, match=r"not a valid tokenizer\.json"):
        Tokenizer.from_file(path)


def test_from_file_refuses_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(GlotscopeError, match="could not be read"):
        Tokenizer.from_file(tmp_path / "absent.json")


def test_direct_construction_is_still_refused() -> None:
    with pytest.raises(TypeError, match="provenance"):
        Tokenizer()
