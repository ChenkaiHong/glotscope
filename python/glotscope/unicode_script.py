"""UAX #24 script attribution for a decoded token (PRD §14.3, D14).

D14 makes script attribution **primary** and corpus attribution a demoted sanity
check, after the earlier design was found to return ≈0 for every language:
under-trained tokens are by construction tokens the model never saw, and FLORES+
devtest is clean translated prose, so the intersection is empty. Script
attribution finds Gemma 2B's Devanagari example where FLORES+ Hindi never emits
it.

That makes this module the paper's independent variable, so two properties
matter more than convenience.

**The table is pinned, not read from the runtime.** ``unicodedata`` exposes no
script property, and the supported matrix bundles four different Unicode
versions — 13.0 on Python 3.10 through 15.1 on 3.13. Any runtime lookup would
attribute tokens differently on different CI cells, so a published ``script``
field would depend on which interpreter ran the analysis. G4 promises numbers
regenerate bit-identically across OS and Python; the committed table is what
makes that true, and ``scripts/generate_script_table.py`` regenerates it from
digest-pinned UCD files.

**A token gets one script or none.** ``Common`` and ``Inherited`` are not
scripts — they are the absence of one — so they never decide the answer, and a
token mixing two real scripts is refused rather than assigned by majority vote.
Mojibake and intermediate BPE junk are precisely what Tier 2 surfaces, and
filing them under whichever script has more characters would inflate a real
language's ``UTR_l`` with something no reader would call theirs.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib.resources import files
from typing import Any

__all__ = ["UNICODE_VERSION", "script_of", "scripts_in", "scripts_of_tokens"]

_TABLE_FILE = "data/unicode-scripts.json"

_COMMON = "Zyyy"
"""ISO 15924 for ``Common``. Not a script: the characters shared by all of them —
digits, spaces, most punctuation."""

_INHERITED = "Zinh"
"""ISO 15924 for ``Inherited``: combining marks, which take the script of
whatever they attach to."""

_UNKNOWN = "Zzzz"
"""ISO 15924 for ``Unknown``: unassigned and private-use codepoints."""

_NOT_A_SCRIPT = frozenset({_COMMON, _INHERITED, _UNKNOWN})

_TABLE: Mapping[str, Any] = json.loads(
    files("glotscope").joinpath(_TABLE_FILE).read_text(encoding="utf-8")
)

UNICODE_VERSION: str = _TABLE["unicode_version"]
"""The pinned release every attribution in this library was computed under.

Recorded beside the numbers rather than assumed: a different release assigns some
codepoints differently, and that would move ``UTR_l``."""

_RANGES: list[list[Any]] = _TABLE["ranges"]
_STARTS: list[int] = [start for start, _, _ in _RANGES]

_EXTENSIONS: list[tuple[int, int, tuple[str, ...]]] = sorted(
    (int(span.split("..")[0], 16), int(span.split("..")[1], 16), tuple(codes))
    for span, codes in _TABLE["extensions"].items()
)
_EXTENSION_STARTS: list[int] = [start for start, _, _ in _EXTENSIONS]


def _script_of_codepoint(codepoint: int) -> str:
    """The Script property of one codepoint, as an ISO 15924 code.

    No lower guard: the table's first range starts at ``U+0000``, so the bisect
    always lands on a real row. The gaps between assigned ranges are what the
    bounds check below catches, and they are the ``Unknown`` codepoints.
    """
    start, end, code = _RANGES[bisect_right(_STARTS, codepoint) - 1]
    return str(code) if start <= codepoint <= end else _UNKNOWN


def _extensions_of_codepoint(codepoint: int) -> tuple[str, ...]:
    """Script_Extensions for one codepoint; empty where it has none."""
    index = bisect_right(_EXTENSION_STARTS, codepoint) - 1
    if index < 0:
        return ()
    start, end, codes = _EXTENSIONS[index]
    return codes if start <= codepoint <= end else ()


@lru_cache(maxsize=8192)
def scripts_in(text: str) -> frozenset[str]:
    """Every real script the characters of ``text`` could belong to.

    ``Common`` and ``Inherited`` characters contribute their Script_Extensions
    set rather than themselves, which is what UAX #24 defines that property for:
    ``U+00B7`` MIDDLE DOT is ``Common`` and used by a dozen scripts, and treating
    it as a script of its own would take every token containing one out of the
    partition ``UTR_l`` is defined over.

    Characters belonging to no script at all — unassigned, private use, plain
    punctuation with no extensions — contribute nothing, so a token made only of
    those yields an empty set.
    """
    found: set[str] = set()
    for character in text:
        codepoint = ord(character)
        code = _script_of_codepoint(codepoint)
        if code not in _NOT_A_SCRIPT:
            found.add(code)
            continue
        found.update(
            extension
            for extension in _extensions_of_codepoint(codepoint)
            if extension not in _NOT_A_SCRIPT
        )
    return frozenset(found)


def script_of(text: str) -> str | None:
    """The single script ``text`` belongs to, or ``None``.

    ``None`` covers three cases that are the same answer for §14's purposes —
    this token contributes to no script's statistics:

    * nothing scriptable at all: ``"..."``, ``" "``, ``"123"``, the empty string;
    * unassigned or private-use characters, which byte-level vocabularies reach
      when decoding unmapped bytes;
    * **two or more real scripts in one token**, refused rather than resolved by
      majority. Mixed-script tokens are mojibake and BPE junk, which is what Tier
      2 surfaces; attributing them to whichever script has more characters would
      inflate a real language's numbers with them.

    A character that merely *widens* the candidate set does not cause a refusal:
    ``" the"`` is Latin because the space carries no script, and ``"a·b"`` is
    Latin because ``U+00B7`` is shared rather than Latin's alone — so the
    decision falls back to the characters carrying a script outright.
    """
    candidates = scripts_in(text)
    if len(candidates) == 1:
        return next(iter(candidates))
    if not candidates:
        return None
    outright = {
        code
        for code in (_script_of_codepoint(ord(character)) for character in text)
        if code not in _NOT_A_SCRIPT
    }
    return next(iter(outright)) if len(outright) == 1 else None


def scripts_of_tokens(tokens: Sequence[str]) -> tuple[str | None, ...]:
    """:func:`script_of` over a batch, in order.

    Batch-shaped for §13's boundary: attributing a whole vocabulary is per-token
    Python work over hundreds of thousands of small strings, which is exactly the
    cache-bound loop v2 is meant to take over.
    """
    return tuple(script_of(token) for token in tokens)
