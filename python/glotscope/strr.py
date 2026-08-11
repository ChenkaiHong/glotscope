"""Single-token retention rate (PRD §7.6).

STRR is the share of a *word list* that survives tokenization as exactly one
token. It is type-level, defined over a list rather than over corpus tokens, so
a frequent word counts once.

The function returns a pair and has no single-value counterpart. ``tau("the")``
and ``tau(" the")`` are different tokenizations of the same word, the choice
between them can move STRR by tens of points, and the source paper does not say
which it used — so glotscope refuses to emit one unqualified number rather than
picking a convention and hoping the reader shares it.

Known confound, documented rather than fixed: the published word lists are
English-frequency lists *translated*, so they are not per-language frequency
lists. Hindi scores lowest across every tokenizer tested despite 128k-255k
vocabularies, and that is a property of the list as much as of the tokenizers.
"""

from __future__ import annotations

from glotscope.aggregate import WordStats
from glotscope.results import StrrPair

__all__ = ["strr"]


def strr(
    bare: WordStats,
    leading_space: WordStats,
    *,
    language: str,
    lowercased: bool,
) -> StrrPair:
    """Compute STRR under both leading-space conventions (PRD §7.6).

    Args:
        bare: words encoded as written, ``tau("the")``.
        leading_space: the same word list encoded with a leading space,
            ``tau(" the")``.
        language: the language code this word list describes.
        lowercased: whether the list was lowercased before encoding. Recorded
            because casing is a parameter of the measurement, and results taken
            under different casing are not comparable.

    Returns:
        Both rates, never one. Words that encoded to zero tokens stay in the
        denominator: such a word was not retained as a single token, and
        dropping it would inflate the rate.

    Raises:
        ValueError: if the two arguments describe word lists of different sizes
            — they must be one list tokenized twice — or if the list is empty.
    """
    if bare.n_words != leading_space.n_words:
        raise ValueError(
            f"both conventions must describe the same word list: got "
            f"{bare.n_words} bare words against {leading_space.n_words} with a "
            f"leading space"
        )
    if bare.n_words == 0:
        raise ValueError("STRR requires at least one word")

    return StrrPair(
        language=language,
        bare=bare.n_single_token / bare.n_words,
        leading_space=leading_space.n_single_token / leading_space.n_words,
        lowercased=lowercased,
        n_words=bare.n_words,
    )
