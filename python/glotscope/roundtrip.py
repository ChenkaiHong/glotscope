"""Round-trip losslessness (PRD §7.8).

``Roundtrip(T; D) = mean over d of 1(decode(encode(d)) == d)``.

Unlike the rest of Tier 1 this measure has a *correct* answer: a byte-level
tokenizer round-trips everything, so a rate below 1.0 is a defect in the
tokenizer rather than a property of the language. That makes it the one number
here a user can act on directly.

Comparison is against the text as handed in. Normalization is the caller's
decision and is recorded in the manifest, because NFKC folds distinctions NFC
keeps and a corpus normalized under NFKC can round-trip at a visibly different
rate — normalizing inside this function would hide that behind an average.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["roundtrip_rate", "roundtrip_rate_from_ids"]


def roundtrip_rate(
    backend: Any,
    documents: Sequence[str],
    *,
    add_special_tokens: bool = False,
) -> float:
    """Share of documents surviving ``decode(encode(d))`` unchanged (§7.8).

    Batched through ``encode_batch``/``decode_batch``: per-document calls are
    what §13 rules out, and this runs over a whole corpus.

    Args:
        backend: a loaded ``tokenizers.Tokenizer``.
        documents: the corpus, already normalized by the caller.
        add_special_tokens: recorded in the manifest either way. Left ``False``
            by default because appended specials do not decode back to the
            source text, and would depress the rate for a reason that has
            nothing to do with losslessness.

    Raises:
        ValueError: if ``documents`` is empty. A mean over nothing is not 1.0,
            and returning 1.0 would report a perfect score for a corpus that was
            never examined.
    """
    if not documents:
        raise ValueError("round-trip losslessness requires at least one document")

    encodings = backend.encode_batch(list(documents), add_special_tokens=add_special_tokens)
    return roundtrip_rate_from_ids(backend, documents, [encoding.ids for encoding in encodings])


def roundtrip_rate_from_ids(
    backend: Any,
    documents: Sequence[str],
    token_ids: Sequence[Sequence[int]],
) -> float:
    """As :func:`roundtrip_rate`, for a caller that already encoded the batch.

    Tier 1 encodes the corpus once for the aggregation fold, and re-encoding it
    here would double the dominant cost of an ``analyze()`` run for a measure
    that only needs the decode half.

    ``documents`` must be the exact text the ids were produced from — including
    whatever normalization the caller applied. Comparing a normalized encoding
    against unnormalized source would report a round-trip failure caused by the
    pipeline rather than by the tokenizer.

    Raises:
        ValueError: if ``documents`` is empty, or if the two sequences differ in
            length. The pairing is positional, so a mismatch compares one
            document against another document's decoding.
    """
    if not documents:
        raise ValueError("round-trip losslessness requires at least one document")
    if len(documents) != len(token_ids):
        raise ValueError(
            f"documents and token_ids must be of equal length: "
            f"got {len(documents)} and {len(token_ids)}"
        )

    decoded = backend.decode_batch([list(ids) for ids in token_ids], skip_special_tokens=False)
    survived = sum(
        1 for original, restored in zip(documents, decoded, strict=True) if original == restored
    )
    return survived / len(documents)
