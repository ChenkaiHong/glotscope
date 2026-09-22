"""An OpenAI encoding, presented through the surface Tier 0 and Tier 1 read.

``tiktoken`` and ``tokenizers`` describe the same kind of object in two
incompatible shapes: one keys its vocabulary by ``bytes`` and returns bare id
lists, the other keys by ``str`` and returns encodings carrying character
offsets. Everything in :mod:`glotscope.tokenizer` and :mod:`glotscope.lint` is
written against the second, so this module supplies it — an adapter, not a
conversion.

**Why not convert.** The alternative was building a ``tokenizers`` BPE from
``mergeable_ranks`` and translating tiktoken's split regex into a pre-tokenizer.
That is implementing a tokenizer, §3.2's first non-goal, and it fails quietly:
a mistranslated split rule still encodes, still round-trips, and changes every
Tier 1 number by a few percent with nothing to notice it by.

**The vocabulary is presented byte-mapped.** Ranks are keyed by raw bytes, and
:func:`glotscope.lint.token_bytes` recovers bytes from a *string* per family. So
each entry is spelled through GPT-2's byte-to-printable map, which is a bijection
on all 256 values — the recovery is exact rather than heuristic, and §7.8 gets
the bytes the encoding actually holds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from glotscope.lint import byte_to_unicode

__all__ = ["TiktokenBackend", "encoding_digest"]

_BYTE_TO_UNICODE = byte_to_unicode()


def _spell(raw: bytes) -> str:
    """Spell a token's bytes the way a byte-level vocabulary spells them."""
    return "".join(_BYTE_TO_UNICODE[value] for value in raw)


@dataclass(frozen=True, slots=True)
class _AddedToken:
    """One row of the added-tokens table, as ``special_ids`` reads it."""

    content: str
    special: bool


class _Encoding:
    """One encoded text: ids always, character offsets only if asked.

    Offsets cost a decode of the whole text, and the Tier 1 hot path — fertility,
    compression, round-trip — reads ids alone. Morphology is the only caller that
    needs offsets, over one word at a time, so they are computed on access rather
    than for every document in a corpus.
    """

    __slots__ = ("_encoding", "_ids")

    def __init__(self, ids: Sequence[int], encoding: Any) -> None:
        self._ids = list(ids)
        self._encoding = encoding

    @property
    def ids(self) -> list[int]:
        return self._ids

    @property
    def offsets(self) -> list[tuple[int, int]]:
        """``(start, end)`` per token, in **characters**, spelled as ``tokenizers``
        spells it.

        The rule is HF's: a token's span runs from the character holding its
        first byte to the end of the character holding its last one. A token
        ending inside a character therefore reports that whole character, and a
        character split across two tokens yields the character once and an empty
        piece once — which is why
        :func:`glotscope.morphology.pieces_from_offsets` claims no boundary
        inside a character.

        Computed from **byte** lengths rather than from
        ``decode_with_offsets``'s starting characters alone. Those starts cannot
        distinguish a token that ended on a character boundary from one that
        ended inside the next character, and the two get different spans:
        checked against HF's GPT-2 on Devanagari, where ``" ह"`` is one token
        covering a space plus one byte of the next character. Numbers must not
        depend on which library loaded the tokenizer.
        """
        if not self._ids:
            return []
        text = self._encoding.decode(self._ids)
        char_of_byte: list[int] = []
        for index, character in enumerate(text):
            char_of_byte.extend([index] * len(character.encode("utf-8")))

        # Indexing is safe without a bound check: decoding is byte-for-byte for
        # text that was encoded from a string, and for a hand-built id list
        # holding an unpaired byte it is *longer* — a maximal invalid subpart is
        # at most three bytes and U+FFFD is exactly three. So the decoded text
        # never has fewer bytes than the tokens that produced it, and the spans
        # simply stop short of its tail, which is a refusal downstream rather
        # than a crash here.
        spans: list[tuple[int, int]] = []
        cursor = 0
        for token_id in self._ids:
            size = len(self._encoding.decode_single_token_bytes(token_id))
            spans.append((char_of_byte[cursor], char_of_byte[cursor + size - 1] + 1))
            cursor += size
        return spans


class TiktokenBackend:
    """A ``tiktoken.Encoding`` behind the six methods glotscope calls.

    Held by :class:`~glotscope.tokenizer.Tokenizer` in place of a
    ``tokenizers.Tokenizer``. The class is deliberately not a subclass of
    anything: the surface is small, and a structural adapter cannot inherit a
    method that would silently answer from the wrong vocabulary.
    """

    __slots__ = ("_by_id", "_encoding", "_ids", "_specials")

    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding
        self._specials: dict[str, int] = dict(encoding._special_tokens)
        by_id: dict[int, str] = {
            token_id: _spell(raw) for raw, token_id in encoding._mergeable_ranks.items()
        }
        # Special tokens are stored as the text they stand for, not byte-mapped:
        # that is how a byte-level vocabulary holds an added token, and it is
        # what ``token_bytes`` expects to read back.
        by_id.update({token_id: text for text, token_id in self._specials.items()})
        self._by_id = by_id
        self._ids = {token: token_id for token_id, token in by_id.items()}

    # -- the surface --------------------------------------------------------

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        if with_added_tokens:
            return dict(self._ids)
        return {
            token: token_id for token, token_id in self._ids.items() if token not in self._specials
        }

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """``n_vocab``, which counts reserved ids the ranks table does not hold.

        Reserved gaps are real vocabulary slots — an embedding matrix has a row
        for each — so counting only the entries with tokens would understate the
        vocabulary and put a wrong denominator under every §7.8 rate.
        """
        if with_added_tokens:
            return int(self._encoding.n_vocab)
        return len(self._encoding._mergeable_ranks)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        wanted = (
            [token_id for token_id in ids if token_id not in self._specials.values()]
            if skip_special_tokens
            else list(ids)
        )
        return str(self._encoding.decode(wanted))

    def decode_batch(
        self, batch: Sequence[Sequence[int]], skip_special_tokens: bool = False
    ) -> list[str]:
        """Decode a batch, which is what §7.6's round-trip check reads.

        Routed through :meth:`decode` per row rather than tiktoken's own
        ``decode_batch``, because that one takes no special-token filter and the
        round-trip check is the one caller that must see exactly what a single
        decode would have produced."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def encode_batch(
        self, texts: Sequence[str], add_special_tokens: bool = False
    ) -> list[_Encoding]:
        """Encode a batch.

        ``add_special_tokens`` is accepted and has no effect, which is the honest
        answer rather than a silent one: tiktoken has no post-processor template,
        so no encoding here ever gains a BOS or EOS. A special token *written in
        the text* is still matched — that is what ``tokenizers`` does with an
        added token, and refusing here would raise on ordinary corpus text.
        """
        del add_special_tokens
        batch = self._encoding.encode_batch(list(texts), allowed_special="all")
        return [_Encoding(ids, self._encoding) for ids in batch]

    def get_added_tokens_decoder(self) -> dict[int, _AddedToken]:
        return {
            token_id: _AddedToken(content=text, special=True)
            for text, token_id in self._specials.items()
        }

    def token_to_id(self, token: str) -> int | None:
        return self._ids.get(token)


def encoding_digest(encoding: Any) -> str:
    """SHA-256 over what determines every number the encoding can produce.

    A tiktoken encoding has no ``tokenizer.json`` and no commit, so §9's identity
    field is filled from the artifact itself: the merge ranks, the special
    tokens, the split pattern and the vocabulary size. Nothing else is in scope —
    the *name* is deliberately excluded, because two names for identical ranks
    are the same tokenizer and a digest that disagreed would make ``verify``
    reject a document that reproduces exactly.

    Every field is length-prefixed before hashing, so no two different
    definitions can serialize to the same byte string.
    """
    digest = hashlib.sha256()

    def feed(chunk: bytes) -> None:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)

    feed(b"glotscope/tiktoken/1")
    feed(str(encoding._pat_str).encode("utf-8"))
    feed(str(int(encoding.n_vocab)).encode("ascii"))

    ranks: Mapping[bytes, int] = encoding._mergeable_ranks
    feed(str(len(ranks)).encode("ascii"))
    for raw, token_id in sorted(ranks.items(), key=lambda item: item[1]):
        feed(str(token_id).encode("ascii"))
        feed(raw)

    specials: Mapping[str, int] = encoding._special_tokens
    feed(str(len(specials)).encode("ascii"))
    for text, token_id in sorted(specials.items(), key=lambda item: item[1]):
        feed(str(token_id).encode("ascii"))
        feed(text.encode("utf-8"))

    return digest.hexdigest()
