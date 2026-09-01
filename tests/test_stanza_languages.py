"""Stanza's language codes, from corpus ones (PRD §7.1, §10.2).

Stanza names a language by its ISO 639-1 code where one exists; FLORES+ names
every variety by ISO 639-3 and script. The two agree only where a language has
no two-letter code, so a table is the only correct bridge — and a table is only
as good as its coverage of the languages the library is built for, which is
what these tests pin.
"""

from __future__ import annotations

import pytest

from glotscope.segmenters.stanza_languages import STANZA_LANGUAGE_CODES, stanza_language

# §10.2's core set: what Stanza calls each, or the code unchanged where Stanza
# ships no model — Amharic, Swahili, Shan and Santali are not mapped to a
# neighbour, and Stanza's own resources file is what refuses them.
_CORE_SET = [
    ("eng_Latn", "en"),
    ("spa_Latn", "es"),
    ("rus_Cyrl", "ru"),
    ("fin_Latn", "fi"),
    ("tur_Latn", "tr"),
    ("hin_Deva", "hi"),
    ("arb_Arab", "ar"),
    ("kor_Hang", "ko"),
    ("cmn_Hans", "zh-hans"),
    ("jpn_Jpan", "ja"),
    ("tha_Thai", "th"),
    ("swh_Latn", "swh"),
    ("amh_Ethi", "amh"),
    ("shn_Mymr", "shn"),
    ("sat_Olck", "sat"),
]


@pytest.mark.parametrize(("language", "expected"), _CORE_SET)
def test_every_core_set_language_resolves_to_what_stanza_calls_it(
    language: str, expected: str
) -> None:
    assert stanza_language(language) == expected


def test_the_script_decides_for_chinese() -> None:
    """One ISO 639-3 code, two Stanza models: the script half of the FLORES+
    code is what picks."""
    assert stanza_language("zho_Hans") == "zh-hans"
    assert stanza_language("zho_Hant") == "zh-hant"
    assert stanza_language("cmn_Hant") == "zh-hant"


def test_a_bare_two_letter_code_passes_through() -> None:
    """Other corpora use ISO 639-1 already, and Stanza resolves its own aliases
    (``no`` for ``nb``), so nothing is rewritten."""
    assert stanza_language("en") == "en"
    assert stanza_language("no") == "no"


def test_an_unknown_language_is_not_guessed() -> None:
    """A code absent from the table reaches Stanza unchanged: refusing is
    Stanza's job, and a nearest-neighbour mapping would put a model for one
    language behind a number published for another."""
    assert stanza_language("xxx_Latn") == "xxx"


def test_the_table_is_keyed_by_lowercase_iso_codes() -> None:
    for code, target in STANZA_LANGUAGE_CODES.items():
        assert code == code.lower(), code
        assert target == target.lower(), target
        assert len(code.split("_", 1)[0]) == 3, code
