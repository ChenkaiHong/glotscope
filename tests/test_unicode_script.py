"""UAX #24 script attribution (PRD §14.3, D14).

D14 makes this the paper's **independent variable**: ``UTR_l`` is defined over
the vocabulary partitioned by script, so a token attributed to the wrong script
moves a published number rather than a display string.

Two things carry the design, and both are tested here rather than assumed:

* the table is **pinned to one Unicode version**, because the supported matrix
  bundles four different ones and a runtime lookup would attribute differently
  on different CI cells;
* ``Common`` and ``Inherited`` resolve through **Script_Extensions**, because a
  real token is rarely one clean script — ``"word."`` is Latin letters plus a
  full stop that belongs to every script at once.
"""

from __future__ import annotations

import pytest

from glotscope.unicode_script import (
    UNICODE_VERSION,
    script_of,
    scripts_in,
    scripts_of_tokens,
)


def test_the_table_is_pinned_to_one_unicode_version() -> None:
    # The whole reason the table is committed rather than read from the runtime.
    # Python 3.10 bundles Unicode 13.0 and 3.13 bundles 15.1, so a runtime lookup
    # would attribute tokens differently across the 12-cell matrix — and the
    # paper's independent variable would depend on which interpreter ran it.
    assert UNICODE_VERSION == "17.0.0"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", "Latn"),
        ("Привет", "Cyrl"),
        ("नमस्ते", "Deva"),
        ("こんにちは", "Hira"),
        ("漢字", "Hani"),
        ("مرحبا", "Arab"),
        ("שלום", "Hebr"),
        ("Γειά", "Grek"),
    ],
)
def test_a_single_script_token_is_attributed_to_it(text: str, expected: str) -> None:
    assert script_of(text) == expected


def test_punctuation_alone_has_no_script_of_its_own() -> None:
    # `Common` is not a script — it is the absence of one. Giving punctuation its
    # own bucket would create a partition of UTR_l that no language inherits.
    assert script_of("...") is None
    assert script_of(" ") is None
    assert script_of("123") is None


def test_common_characters_do_not_outvote_the_script_they_accompany() -> None:
    # The ordinary shape of a real token: a word plus a leading space or a full
    # stop. Byte-level vocabularies are full of these, and letting the Common
    # characters decide would strand them outside every script.
    assert script_of(" the") == "Latn"
    assert script_of("word.") == "Latn"
    assert script_of("的。") == "Hani"


def test_a_mixed_script_token_is_refused_rather_than_guessed() -> None:
    # Mojibake and intermediate BPE junk mix scripts, and those are exactly the
    # tokens Tier 2 surfaces. Taking a majority vote would file them under a real
    # language and inflate that language's UTR_l with something no reader would
    # recognise as theirs.
    assert script_of("aП") is None
    assert script_of("漢a") is None


def test_script_extensions_resolve_a_shared_character() -> None:
    # U+00B7 MIDDLE DOT is Common, and Script_Extensions names the scripts that
    # actually use it. Beside a Latin letter it must not break attribution.
    assert "Latn" in scripts_in("·")
    assert script_of("a·b") == "Latn"


def test_an_empty_token_has_no_script() -> None:
    assert script_of("") is None


def test_a_private_use_character_belongs_to_no_script() -> None:
    # Byte-level vocabularies decode unmapped bytes into odd places, so this is
    # reachable rather than theoretical.
    assert script_of("") is None


def test_the_batch_form_preserves_order_and_the_refusals() -> None:
    # §13's boundary shape: attribution over a whole vocabulary is per-token work
    # over hundreds of thousands of small strings, so the batch call is the one
    # v2 will implement. Order is load-bearing — the caller zips it against token
    # ids — and the Nones must survive rather than being filtered out.
    tokens = ["hello", "...", "漢字", "aП", ""]

    assert scripts_of_tokens(tokens) == ("Latn", None, "Hani", None, None)
