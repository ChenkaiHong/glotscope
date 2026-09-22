"""Stanza's language codes, from corpus ones (PRD §7.1, §10.2).

Stanza names a language by its ISO 639-1 code where one exists and by ISO 639-3
otherwise, with Chinese split by script into ``zh-hans`` and ``zh-hant``.
FLORES+ names every variety by ISO 639-3 and script. The two agree only where a
language has no two-letter code, and the first two letters of a three-letter
code are not it: ``spa`` is ``es``, ``jpn`` is ``ja``, ``tur`` is ``tr``, and
``cmn`` is ``zh-hans`` or ``zh-hant`` depending on the script half of the code.

This table is the bridge for every language Stanza 1.14 ships a tokenizer for,
keyed by ISO 639-3 — the individual-language code FLORES+ uses (``arb``,
``cmn``, ``pes``, ``swh``) and the macrolanguage code where one exists — or by
the whole lowercased FLORES+ code where the script decides. A language absent
from it reaches Stanza unchanged, so Stanza's own resources file is what refuses
it. Nothing here guesses: a variety Stanza has no model for — Amharic, Swahili,
Santali, Shan — is not mapped to a neighbour, because a model for one language
behind a number published for another is the mistake this layer exists to make
visible.

``scripts/audit_stanza_languages.py`` checks the table against the resources
file Stanza publishes. Re-run it when the pinned Stanza version moves.
"""

from __future__ import annotations

from types import MappingProxyType

from glotscope.segmenters._support import language_prefix

__all__ = ["STANZA_LANGUAGE_CODES", "stanza_language"]

STANZA_LANGUAGE_CODES: MappingProxyType[str, str] = MappingProxyType(
    {
        "abk": "ab",
        "afr": "af",
        "als": "sq",  # Tosk Albanian: FLORES+'s code for Albanian
        "ara": "ar",
        "arb": "ar",  # Modern Standard Arabic: FLORES+'s code
        "bel": "be",
        "bul": "bg",
        "cat": "ca",
        "ces": "cs",
        "chu": "cu",
        "cmn_hans": "zh-hans",  # Mandarin, Simplified: FLORES+'s code
        "cmn_hant": "zh-hant",
        "cym": "cy",
        "dan": "da",
        "deu": "de",
        "ekk": "et",  # Standard Estonian: FLORES+'s code
        "ell": "el",
        "eng": "en",
        "est": "et",
        "eus": "eu",
        "fao": "fo",
        "fas": "fa",
        "fin": "fi",
        "fra": "fr",
        "gla": "gd",
        "gle": "ga",
        "glg": "gl",
        "glv": "gv",
        "heb": "he",
        "hin": "hi",
        "hrv": "hr",
        "hun": "hu",
        "hye": "hy",
        "ind": "id",
        "isl": "is",
        "ita": "it",
        "jpn": "ja",
        "kat": "ka",
        "kaz": "kk",
        "kir": "ky",
        "kor": "ko",
        "lat": "la",
        "lav": "lv",
        "lit": "lt",
        "lvs": "lv",  # Standard Latvian: FLORES+'s code
        "mar": "mr",
        "mlt": "mt",
        "mya": "my",
        "nld": "nl",
        "nno": "nn",
        "nob": "nb",
        "ori": "or",
        "ory": "or",  # Odia: FLORES+'s code
        "pes": "fa",  # Western Persian: FLORES+'s code
        "pol": "pl",
        "por": "pt",
        "ron": "ro",
        "rus": "ru",
        "san": "sa",
        "slk": "sk",
        "slv": "sl",
        "snd": "sd",
        "spa": "es",
        "sqi": "sq",
        "srp": "sr",
        "swe": "sv",
        "tam": "ta",
        "tel": "te",
        "tha": "th",
        "tur": "tr",
        "uig": "ug",
        "ukr": "uk",
        "urd": "ur",
        "vie": "vi",
        "wol": "wo",
        "zho_hans": "zh-hans",
        "zho_hant": "zh-hant",
    }
)
"""ISO 639-3 code, or whole FLORES+ code where the script decides, to the code
Stanza's resources file keys the language by. Read-only: the table is a closed
vocabulary, and the audit script is how it changes."""


def stanza_language(language: str) -> str:
    """What Stanza calls ``language``.

    Looked up by the whole code first, so that ``cmn_Hans`` and ``cmn_Hant``
    reach different models, then by the ISO 639-3 prefix. A code the table does
    not know — a bare ISO 639-1 code from another corpus, or a language Stanza
    has no model for — is returned unchanged for Stanza to accept or refuse.
    """
    whole = language.lower()
    if whole in STANZA_LANGUAGE_CODES:
        return STANZA_LANGUAGE_CODES[whole]
    prefix = language_prefix(language)
    return STANZA_LANGUAGE_CODES.get(prefix, prefix)
