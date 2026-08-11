"""Pure Tier 1 metric calculations with no tokenizer or corpus I/O."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

from glotscope.enums import RenyiNormalizer
from glotscope.results import GiniResult, ParityResult, RenyiResult, sorted_costs

__all__ = ["gini", "parity", "renyi_efficiency", "renyi_efficiency_from_counts"]

_TokenType = TypeVar("_TokenType", bound=Hashable)
"""Whatever the caller counts by — token ids from the fold, token strings from a
sequence. ``Mapping`` is invariant in its key type, so a bare ``Hashable`` key
would reject the ``Counter[int]`` the aggregation layer produces."""


def renyi_efficiency(
    tokens: Sequence[Hashable],
    *,
    alpha: float,
    normalizer: RenyiNormalizer = RenyiNormalizer.OBSERVED,
    nominal_vocab_size: int | None = None,
) -> RenyiResult:
    """Compute Rényi entropy and efficiency over a token sequence (PRD §7.5)."""
    if not tokens:
        raise ValueError("Rényi efficiency requires at least one token")

    return renyi_efficiency_from_counts(
        Counter(tokens),
        alpha=alpha,
        normalizer=normalizer,
        nominal_vocab_size=nominal_vocab_size,
    )


def renyi_efficiency_from_counts(
    counts: Mapping[_TokenType, int],
    *,
    alpha: float,
    normalizer: RenyiNormalizer = RenyiNormalizer.OBSERVED,
    nominal_vocab_size: int | None = None,
) -> RenyiResult:
    """Compute Rényi efficiency from a token-type frequency distribution (§7.5).

    The distribution is the only input the formula needs, and it is what
    :class:`~glotscope.aggregate.DocumentStats` already carries. Taking counts
    directly avoids materializing a corpus-sized token sequence purely to count
    it again — across 229 FLORES+ varieties that is the difference between a
    fold and a re-encode.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if any(count < 0 for count in counts.values()):
        raise ValueError("type counts cannot be negative")

    total = sum(counts.values())
    if total == 0:
        raise ValueError("Rényi efficiency requires at least one token")
    probabilities = tuple(count / total for count in counts.values())

    if alpha == 1.0:
        entropy_bits = -sum(probability * math.log2(probability) for probability in probabilities)
    else:
        power_sum = sum(probability**alpha for probability in probabilities)
        entropy_bits = math.log2(power_sum) / (1.0 - alpha)

    if normalizer is RenyiNormalizer.OBSERVED:
        vocabulary_size = len(counts)
    else:
        if nominal_vocab_size is None:
            raise ValueError("nominal_vocab_size is required with the nominal normalizer")
        if nominal_vocab_size < len(counts):
            raise ValueError("nominal_vocab_size cannot be smaller than the observed vocabulary")
        vocabulary_size = nominal_vocab_size

    value = 0.0 if vocabulary_size == 1 else entropy_bits / math.log2(vocabulary_size)
    return RenyiResult(
        value=value,
        alpha=alpha,
        normalizer=normalizer,
        entropy_bits=entropy_bits,
    )


def parity(
    token_counts: Mapping[str, Sequence[int]],
    *,
    reference: str,
) -> ParityResult:
    """Compute corpus parity as a ratio of per-language token-count means (§7.3)."""
    if not token_counts:
        raise ValueError("parity requires at least one language")
    if reference not in token_counts:
        raise KeyError(f"reference language {reference!r} is not present")

    line_counts = {language: len(counts) for language, counts in token_counts.items()}
    expected_lines = line_counts[reference]
    if expected_lines == 0:
        raise ValueError("parity requires at least one aligned line")
    if any(count != expected_lines for count in line_counts.values()):
        raise ValueError(f"parallel languages have unequal line counts: {line_counts}")
    if any(count < 0 for counts in token_counts.values() for count in counts):
        raise ValueError("token counts cannot be negative")

    reference_total = sum(token_counts[reference])
    if reference_total == 0:
        raise ValueError("reference language has zero tokens")

    per_language = {
        language: sum(counts) / reference_total for language, counts in token_counts.items()
    }
    worst_case_language = max(per_language, key=per_language.__getitem__)
    return ParityResult(
        reference_language=reference,
        per_language=per_language,
        macro_parity=sum(per_language.values()) / len(per_language),
        worst_case_language=worst_case_language,
        worst_case_parity=per_language[worst_case_language],
        n_lines_per_language=line_counts,
    )


def gini(costs_by_language: Mapping[str, float]) -> GiniResult:
    """Compute Gini over ascending per-language tokens-per-line costs (§7.4)."""
    if not costs_by_language:
        raise ValueError("Gini requires at least one language")
    if any(not math.isfinite(cost) or cost < 0.0 for cost in costs_by_language.values()):
        raise ValueError("Gini costs must be finite and non-negative")

    costs = sorted_costs(tuple(costs_by_language.values()))
    total = sum(costs)
    if total == 0.0:
        value = 0.0
    else:
        n = len(costs)
        absolute_differences = sum(abs(left - right) for left in costs for right in costs)
        value = absolute_differences / (2 * n * total)

    return GiniResult(
        value=value,
        languages=frozenset(costs_by_language),
        cost_unit="tokens_per_line",
    )
