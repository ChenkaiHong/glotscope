"""Tier 0 vocabulary lint (PRD §7.8).

:mod:`glotscope.utf8` classifies a byte vocabulary; this module is what produces
one. That split is not cosmetic — recovering the *bytes* a vocabulary entry
stands for is family-specific, and getting it wrong is the quiet way to make
every §7.8 and §7.9 count wrong:

* A byte-level BPE stores each byte as a printable stand-in character (GPT-2's
  ``bytes_to_unicode``). Reading those token strings as UTF-8 reports zero
  ill-formed entries for a vocabulary that is half ill-formed.
* A byte-fallback model stores raw bytes as ``<0xNN>`` pieces. Read literally,
  each is six well-formed ASCII characters, and ``byte_fallback_coverage``
  collapses to zero.

``decode([id])`` is not an escape from this: it is lossy by design — invalid
bytes come back as U+FFFD — which is exactly the signal Tier 0 exists to report.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from glotscope.enums import Algorithm, TokenizerFamily
from glotscope.report import Tier0Report
from glotscope.utf8 import build_utf8_report

__all__ = [
    "byte_to_unicode",
    "classify_family",
    "detect_algorithm",
    "lint_backend",
    "special_ids",
    "token_bytes",
    "unreachable_ids",
    "vocab_bytes",
]

_HEX_BYTE = re.compile(r"^<0[xX]([0-9a-fA-F]{2})>$")

_SENTENCEPIECE_SPACE = "▁"
"""LOWER ONE EIGHTH BLOCK: SentencePiece's word-boundary marker, standing for a
literal space in the piece it prefixes."""

_ALGORITHM_BY_MODEL_TYPE = {
    "Unigram": Algorithm.UNIGRAM_LM,
    "WordPiece": Algorithm.WORDPIECE,
}


def byte_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte-to-printable-character map.

    Byte-level BPE cannot store raw control bytes in a JSON vocabulary, so each
    of the 256 byte values is mapped to a distinct printable codepoint. The map
    is a fixed part of the format rather than a per-tokenizer choice, which is
    why it is reconstructed here instead of read from ``tokenizer.json``.
    """
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    codepoints = list(printable)
    spare = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            codepoints.append(256 + spare)
            spare += 1
    return {value: chr(code) for value, code in zip(printable, codepoints, strict=True)}


_UNICODE_TO_BYTE = {character: value for value, character in byte_to_unicode().items()}


def token_bytes(token: str, family: TokenizerFamily) -> bytes:
    """Recover the byte sequence one vocabulary entry stands for.

    Added tokens are the case where a byte-level vocabulary holds ordinary text:
    ``<s>`` is stored verbatim, not byte-mapped. An entry whose characters are
    not all in the byte map is therefore read as literal UTF-8 rather than being
    force-decoded into nonsense.
    """
    if family is TokenizerFamily.BYTE_LEVEL:
        if token and all(character in _UNICODE_TO_BYTE for character in token):
            return bytes(_UNICODE_TO_BYTE[character] for character in token)
        return token.encode("utf-8")

    if family is TokenizerFamily.BYTE_FALLBACK:
        escaped = _HEX_BYTE.match(token)
        if escaped is not None:
            return bytes([int(escaped.group(1), 16)])
        return token.replace(_SENTENCEPIECE_SPACE, " ").encode("utf-8")

    return token.encode("utf-8")


def classify_family(spec: Mapping[str, Any]) -> TokenizerFamily:
    """Classify byte handling from a parsed ``tokenizer.json`` (PRD §7.8).

    Reported beside a quantitative rate, never as a binary alarm: UTF-8
    Plumbing's Table 2 puts essentially the whole §11 roster in the first two
    families, and a predicate true of every input carries no information.
    """
    if _mentions_type(spec.get("pre_tokenizer"), "ByteLevel") or _mentions_type(
        spec.get("decoder"), "ByteLevel"
    ):
        return TokenizerFamily.BYTE_LEVEL

    model = spec.get("model") or {}
    if model.get("byte_fallback") or _mentions_type(spec.get("decoder"), "ByteFallback"):
        return TokenizerFamily.BYTE_FALLBACK

    return TokenizerFamily.CODE_POINT


def detect_algorithm(spec: Mapping[str, Any]) -> Algorithm:
    """Read the construction algorithm off a parsed ``tokenizer.json`` (PRD §9).

    Unrecognized model types record :attr:`~glotscope.enums.Algorithm.UNKNOWN`
    rather than a guess. Tier 2's scope limits key off this field — §7.9 is
    validated on BPE only — so a wrong value here is worse than an absent one.
    """
    model = spec.get("model") or {}
    model_type = model.get("type")
    if not isinstance(model_type, str):
        if "merges" in model:
            family = classify_family(spec)
            if family is TokenizerFamily.BYTE_LEVEL:
                return Algorithm.BYTE_LEVEL_BPE
            if family is TokenizerFamily.BYTE_FALLBACK:
                return Algorithm.BYTE_FALLBACK_BPE
        if "continuing_subword_prefix" in model:
            return Algorithm.WORDPIECE
        if "unk_id" in model and isinstance(model.get("vocab"), list):
            return Algorithm.UNIGRAM_LM
        return Algorithm.UNKNOWN
    if model_type != "BPE":
        return _ALGORITHM_BY_MODEL_TYPE.get(model_type, Algorithm.UNKNOWN)

    family = classify_family(spec)
    if family is TokenizerFamily.BYTE_LEVEL:
        return Algorithm.BYTE_LEVEL_BPE
    if family is TokenizerFamily.BYTE_FALLBACK:
        return Algorithm.BYTE_FALLBACK_BPE
    return Algorithm.UNKNOWN


def vocab_bytes(vocab: Mapping[str, int], family: TokenizerFamily) -> dict[int, bytes]:
    """Map token id to the bytes it stands for, across a whole vocabulary."""
    return {token_id: token_bytes(token, family) for token, token_id in vocab.items()}


def unreachable_ids(backend: Any, token_ids: Sequence[int]) -> tuple[int, ...]:
    """Ids failing ``encode(decode(id)) == [id]`` (PRD §7.8).

    Batched through ``encode_batch`` rather than looped one id at a time: a
    250k-entry vocabulary is a real cost, and per-call crossings are what §13
    rules out.
    """
    if not token_ids:
        return ()
    texts = [backend.decode([token_id], skip_special_tokens=False) for token_id in token_ids]
    encodings = backend.encode_batch(texts, add_special_tokens=False)
    return tuple(
        token_id
        for token_id, encoding in zip(token_ids, encodings, strict=True)
        if list(encoding.ids) != [token_id]
    )


def special_ids(backend: Any) -> tuple[int, ...]:
    """Ids the tokenizer itself declares special (PRD §7.9 Stage 1).

    Read from the added-tokens table rather than pattern-matched on ``<...>`` or
    ``[...]``: a byte-fallback vocabulary spells ordinary bytes ``<0xNN>``, and
    a pattern would drop 256 unremarkable entries out of the candidate set.
    """
    added = backend.get_added_tokens_decoder()
    return tuple(sorted(token_id for token_id, token in added.items() if token.special))


def lint_backend(
    backend: Any,
    spec: Mapping[str, Any],
    *,
    family: TokenizerFamily | None = None,
) -> Tier0Report:
    """Run the whole Tier 0 pass over a loaded backend tokenizer (PRD §7.8).

    Args:
        backend: a loaded ``tokenizers.Tokenizer``, or an adapter presenting the
            same surface.
        spec: the parsed ``tokenizer.json`` it was loaded from. Passed in rather
            than re-serialized from ``backend``, so that the family, the
            algorithm and the manifest's SHA-256 all describe the same bytes.
        family: the byte-handling family, where the loader knows it outright
            instead of inferring it. A ``tokenizer.json`` names its own
            pre-tokenizer and decoder, so inference is the only evidence there;
            a tiktoken encoding is byte-level *by format*, and synthesizing a
            spec that says so would put a fabricated document under a field
            ``verify`` trusts.
    """
    resolved = family if family is not None else classify_family(spec)
    by_id = vocab_bytes(backend.get_vocab(with_added_tokens=True), resolved)
    return build_utf8_report(
        by_id,
        unreachable_tokens=unreachable_ids(backend, sorted(by_id)),
        special_tokens=special_ids(backend),
        family=resolved,
    )


def _mentions_type(component: Any, wanted: str) -> bool:
    """Whether a ``tokenizer.json`` component is, or contains, ``wanted``.

    Components nest: a ``Sequence`` decoder holding ``ByteFallback`` is as
    byte-fallback as a bare one, and Llama-style tokenizers are routinely
    serialized the nested way.
    """
    if not isinstance(component, Mapping):
        return False
    if component.get("type") == wanted:
        return True
    nested = component.get("decoders") or component.get("pretokenizers") or []
    return any(_mentions_type(child, wanted) for child in nested)
