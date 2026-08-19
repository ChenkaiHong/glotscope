"""The Tier 2 reference-set fallback chain (PRD §7.9).

``u_ref`` is the mean ``E_out`` row over a set of tokens the model is believed
never to have trained on. The whole cosine indicator is measured against it, so
where that set comes from decides what the number means.

§7.9 specifies a **chain, not an assumption**, tried in order:

1. ``<unused…>``-style vocabulary entries — the vocabulary stating outright that
   a slot was reserved and never filled.
2. Embedding rows above ``|V|`` — padding, inferred from the shape mismatch.
3. Single bytes ``0xF5``-``0xFF``, which no well-formed UTF-8 text contains.

Link 3 is convenient, not universal: Land & Bartolo document tokenizers holding
exactly the 243 bytes UTF-8 uses, and StarCoder2 additionally misses ``0xF1``.
For those checkpoints the chain is genuinely exhausted, and this module raises
:class:`~glotscope.errors.NoReferenceSetError` rather than taking the mean of an
empty set — which would be ``nan`` at best and a silent zero vector at worst.

Note the interaction with §7.9 Stage 1, which is deliberate and easy to misread:
``<unused0>`` is a *special* token, so Stage 1 excludes it from the candidate
set. It is still the best available reference. The reference set is the
yardstick, not a candidate, and the two are allowed to sit differently in the
vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from glotscope.enums import ReferenceSource, TokenizerFamily
from glotscope.errors import NoReferenceSetError
from glotscope.lint import vocab_bytes

__all__ = ["ReferenceSet", "resolve_reference_set"]

_UNUSED_TOKEN = re.compile(
    r"^<\|?(?:unused(?:[-_]?token)?|reserved[-_]special[-_]token)[-_]?\d+\|?>$",
    re.IGNORECASE,
)
"""Spellings the §11 roster actually uses.

Gemma writes ``<unused0>``, Llama 3 writes ``<|reserved_special_token_0|>``.
Matching bare ``<...>`` instead would sweep in ``<s>``, ``</s>`` and ``<pad>`` —
trained tokens, every one of them — and computing ``u_ref`` from trained rows
moves the yardstick with no error raised anywhere.

**T5's ``<extra_id_N>`` is deliberately absent**, though it reads like a
reserved slot and was matched here at first. It is not one: span corruption
emits those sentinels in every training target, so across T5, mT5, Flan-T5 and
UL2 their ``E_out`` rows are among the *best*-trained in the vocabulary.
Averaging them into ``u_ref`` inverts the indicator — it then measures
similarity-to-trained — while the manifest goes on reporting ``unused_tokens``
as the source. A T5-family checkpoint now falls through to padding rows or
spare bytes, or exhausts the chain and refuses, all three of which are honest
answers where the fourth was not."""

_NON_UTF8_BYTES = range(0xF5, 0x100)
"""Byte values that never appear in well-formed UTF-8 (§7.8, §7.9 chain link 3)."""


@dataclass(frozen=True, slots=True)
class ReferenceSet:
    """The tokens ``u_ref`` is averaged over, and where they came from."""

    token_ids: tuple[int, ...]
    source: ReferenceSource
    """Published in the manifest. Two checkpoints measured against different
    links of the chain were measured against different yardsticks, and a reader
    tabling them together needs to be able to see that."""

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError(
                "a reference set cannot be empty; the caller should have raised "
                "NoReferenceSetError rather than constructing one"
            )


def _unused_token_ids(vocab: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(
        sorted(token_id for token, token_id in vocab.items() if _UNUSED_TOKEN.match(token))
    )


def _unused_byte_ids(vocab: Mapping[str, int], family: TokenizerFamily) -> tuple[int, ...]:
    """Ids standing for a single byte in ``0xF5``-``0xFF``.

    Resolved through :func:`~glotscope.lint.vocab_bytes` so the byte-level and
    byte-fallback spellings are handled by the code that already knows the
    difference — ``0xF5`` is one surrogate character in the first family and the
    six ASCII characters ``<0xF5>`` in the second.
    """
    by_id = vocab_bytes(vocab, family)
    wanted = {bytes([value]) for value in _NON_UTF8_BYTES}
    return tuple(sorted(token_id for token_id, raw in by_id.items() if raw in wanted))


def resolve_reference_set(
    vocab: Mapping[str, int],
    family: TokenizerFamily,
    *,
    vocab_size: int,
    n_rows: int,
    checkpoint: str,
    tied: bool,
) -> ReferenceSet:
    """Walk the §7.9 chain and return the first link that yields tokens.

    Args:
        vocab: token string to id, as the tokenizer reports it.
        family: decides how a single byte is spelled in this vocabulary.
        vocab_size: ``|V|``.
        n_rows: rows in the embedding matrix; may exceed ``|V|``.
        checkpoint: named in the refusal, so a failure says which model.
        tied: changes what the caller can do about exhaustion, and therefore
            what the refusal should tell them.

    Raises:
        NoReferenceSetError: if all three links are empty. Never returns an
            empty set — the mean of nothing is the failure this exists to stop.
    """
    unused = _unused_token_ids(vocab)
    if unused:
        return ReferenceSet(unused, ReferenceSource.UNUSED_TOKENS)

    if n_rows > vocab_size:
        return ReferenceSet(tuple(range(vocab_size, n_rows)), ReferenceSource.PADDING_ROWS)

    spare_bytes = _unused_byte_ids(vocab, family)
    if spare_bytes:
        return ReferenceSet(spare_bytes, ReferenceSource.UNUSED_BYTES)

    raise NoReferenceSetError(checkpoint, tied=tied)
