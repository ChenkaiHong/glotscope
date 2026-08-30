"""Petrov's published token premium, reproduced on FLORES+ (PRD §12.1, §7.3).

**Gated, and skips rather than fails.** FLORES+ needs accepted terms and an
`HF_TOKEN`, so anonymous CI cannot fetch it. Point ``GLOTSCOPE_CORPUS_ROOT`` at a
download made by ``scripts/fetch_flores_plus.py``; without one, these skip with
a message naming the command.

The row this closes is the one §12.1 flags as ambiguous: *"Compute both
aggregations first — Petrov never states which he used, and at premium 15-19
they can differ by more than the tolerance."* So both are computed here, and both
are asserted, because the interesting result is which one is closer. It is the
ratio of means — by a factor of four on both tokenizers — which is independent
evidence for D7 rather than a restatement of it.

Shan is the extreme cell of Petrov's table and therefore the discriminating one:
a bug that shifted every language a little would be invisible at 1.2x and obvious
at 18x.
"""

from __future__ import annotations

import os
from pathlib import Path
from statistics import mean

import pytest

from glotscope.corpus import Corpus
from glotscope.tokenizer import Tokenizer

CORPUS_ROOT = Path(os.environ.get("GLOTSCOPE_CORPUS_ROOT", Path.home() / "corpora"))
_DEVTEST = CORPUS_ROOT / "flores_plus" / "2024.08" / "devtest"

pytestmark = [
    pytest.mark.reference,
    pytest.mark.gated,
    pytest.mark.skipif(
        not (_DEVTEST / "shn_Mymr.txt").is_file(),
        reason=(
            "FLORES+ is gated and ships with nothing (D12). Accept the terms, "
            "export HF_TOKEN, and run: python scripts/fetch_flores_plus.py "
            "--root <root> --split devtest --revision "
            "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06 — then point "
            "GLOTSCOPE_CORPUS_ROOT at <root>."
        ),
    ),
]

_TOLERANCE = 0.10
"""§12.1's tolerance for this row: ±10%."""

_PUBLISHED = {"gpt2": 18.76, "cl100k": 15.05}
"""Petrov Table 1 / Appendix C: the Shan premium against English."""


def _counts(tokenizer: Tokenizer, documents: tuple[str, ...]) -> list[int]:
    return [
        len(encoding.ids)
        for encoding in tokenizer._backend.encode_batch(list(documents), add_special_tokens=False)
    ]


@pytest.fixture(scope="module")
def shan_and_english() -> tuple[tuple[str, ...], tuple[str, ...]]:
    loaded = Corpus.flores_plus(["eng_Latn", "shn_Mymr"]).load(CORPUS_ROOT)
    return loaded.lines["eng_Latn"], loaded.lines["shn_Mymr"]


@pytest.mark.parametrize(
    ("label", "encoding"),
    [("gpt2", "r50k_base"), ("cl100k", "cl100k_base")],
)
def test_the_shan_premium_reproduces_under_the_ratio_of_means(
    label: str,
    encoding: str,
    shan_and_english: tuple[tuple[str, ...], tuple[str, ...]],
) -> None:
    english, shan = shan_and_english
    tokenizer = Tokenizer.from_tiktoken(encoding)

    premium = mean(_counts(tokenizer, shan)) / mean(_counts(tokenizer, english))

    published = _PUBLISHED[label]
    assert abs(premium - published) / published < _TOLERANCE, (
        f"{label}: ratio of means {premium:.3f} against Petrov's {published}"
    )


@pytest.mark.parametrize(
    ("label", "encoding"),
    [("gpt2", "r50k_base"), ("cl100k", "cl100k_base")],
)
def test_the_ratio_of_means_is_the_closer_aggregation(
    label: str,
    encoding: str,
    shan_and_english: tuple[tuple[str, ...], tuple[str, ...]],
) -> None:
    """Both aggregations land inside the tolerance; only one is close.

    §7.3 keeps the ratio of means because that is what maps to API cost (D7).
    That argument stands on its own — but Petrov's own numbers turn out to agree
    with it, and a run that silently switched aggregation would still pass the
    tolerance above. This is what would catch it.
    """
    english, shan = shan_and_english
    tokenizer = Tokenizer.from_tiktoken(encoding)
    english_counts = _counts(tokenizer, english)
    shan_counts = _counts(tokenizer, shan)

    ratio_of_means = mean(shan_counts) / mean(english_counts)
    mean_of_ratios = mean(s / e for s, e in zip(shan_counts, english_counts, strict=True))

    published = _PUBLISHED[label]
    assert abs(mean_of_ratios - published) / published < _TOLERANCE, "both are in tolerance"
    assert abs(ratio_of_means - published) < abs(mean_of_ratios - published), (
        f"{label}: ratio of means {ratio_of_means:.3f} and mean of ratios "
        f"{mean_of_ratios:.3f} against {published}"
    )


def test_the_two_backends_count_identically(
    shan_and_english: tuple[tuple[str, ...], tuple[str, ...]],
) -> None:
    """GPT-2 through ``tokenizers`` and ``r50k_base`` through tiktoken are the
    same BPE, so they must produce the same counts on real multilingual text.

    Needs the Hub as well as the corpus, and both are already required by the
    skip above being satisfied."""
    english, shan = shan_and_english
    hub = Tokenizer.from_pretrained("openai-community/gpt2")
    encoding = Tokenizer.from_tiktoken("r50k_base")

    for documents in (english, shan):
        assert _counts(hub, documents) == _counts(encoding, documents)
