"""UTF-8 vocabulary classification for Tier 0."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from glotscope.enums import TokenClass, TokenizerFamily
from glotscope.report import Tier0Report

__all__ = ["build_utf8_report", "classify_utf8_token"]


def classify_utf8_token(token_bytes: bytes) -> TokenClass:
    """Classify one byte token into the three disjoint PRD §7.8 classes."""
    try:
        token_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data":
            return TokenClass.PARTIAL_UTF8
        return TokenClass.ILL_FORMED_NOT_PARTIAL
    return TokenClass.WELL_FORMED


def build_utf8_report(
    vocab: Mapping[int, bytes],
    *,
    unreachable_tokens: Iterable[int] = (),
    special_tokens: Iterable[int] = (),
    family: TokenizerFamily,
) -> Tier0Report:
    """Classify a byte vocabulary and assemble its Tier 0 integrity report."""
    if not vocab:
        raise ValueError("UTF-8 classification requires a non-empty vocabulary")

    counts = dict.fromkeys(TokenClass, 0)
    partial: list[int] = []
    for token_id, token_bytes in vocab.items():
        token_class = classify_utf8_token(token_bytes)
        counts[token_class] += 1
        if token_class is TokenClass.PARTIAL_UTF8:
            partial.append(token_id)

    ill_formed = counts[TokenClass.PARTIAL_UTF8] + counts[TokenClass.ILL_FORMED_NOT_PARTIAL]
    byte_values = {token[0] for token in vocab.values() if len(token) == 1}
    return Tier0Report(
        vocab_size=len(vocab),
        family=family,
        ill_formed_vocab_rate=ill_formed / len(vocab),
        token_classes=counts,
        partial_utf8_tokens=tuple(sorted(partial)),
        unreachable_tokens=tuple(sorted(set(unreachable_tokens))),
        special_tokens=tuple(sorted(set(special_tokens))),
        byte_fallback_coverage=len(byte_values),
    )
