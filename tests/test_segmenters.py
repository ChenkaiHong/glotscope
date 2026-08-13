"""Word segmentation adapters (PRD §7.1, §10.3, D6).

``W(D)`` is the single largest source of silent incomparability in this
literature, so these tests are mostly about refusals: a missing extra, a
segmenter used outside the language it was built for, and a segmenter that is
not written yet must each fail loudly and differently. The one thing none of
them may do is quietly fall back to whitespace.

Every adapter is exercised against the real library. A test that mocked the
segmenter would assert that glotscope calls a function, which is not the part
that goes wrong — what goes wrong is the version pinned in the manifest and the
segmentation the library actually returns.
"""

from __future__ import annotations

import pytest

from glotscope.enums import Segmenter
from glotscope.errors import SegmenterScopeError, SegmenterUnavailableError
from glotscope.segmenters import available, get_segmenter

pytestmark = pytest.mark.segmenter

ENGLISH = "The cat sat."
CHINESE = "我爱北京天安门"
JAPANESE = "猫が座った。"
THAI = "ฉันรักกรุงเทพ"
KHMER = "ខ្ញុំស្រលាញ់"


def _require(segmenter: Segmenter) -> None:
    if not available(segmenter):
        pytest.skip(f"the {segmenter.value} extra is not installed")


def test_whitespace_needs_no_extra_and_pins_no_model() -> None:
    # The one segmenter that is always available, and the one that must never be
    # a default: it is degenerate for Chinese, Japanese, Thai, Khmer, Lao and
    # Tibetan, where a whole clause becomes one "word".
    words = get_segmenter(Segmenter.WHITESPACE, language="eng_Latn")

    assert words.segment(ENGLISH) == ("The", "cat", "sat.")
    assert words.model_version is None


def test_whitespace_is_degenerate_on_a_no_whitespace_language() -> None:
    # Recorded rather than worked around. This is why the segmenter is a
    # required parameter, and a test that avoided the case would hide it.
    words = get_segmenter(Segmenter.WHITESPACE, language="zho_Hans")

    assert words.segment(CHINESE) == (CHINESE,)


@pytest.mark.parametrize(
    ("segmenter", "language", "text", "expected"),
    [
        (Segmenter.JIEBA, "zho_Hans", CHINESE, ("我", "爱", "北京", "天安门")),
        (Segmenter.MECAB, "jpn_Jpan", JAPANESE, ("猫", "が", "座っ", "た", "。")),
        (Segmenter.PYTHAINLP, "tha_Thai", THAI, ("ฉัน", "รัก", "กรุงเทพ")),
        (Segmenter.KHMER_NLTK, "khm_Khmr", KHMER, ("ខ្ញុំ", "ស្រលាញ់")),
        (Segmenter.ICU, "tha_Thai", THAI, ("ฉัน", "รัก", "กรุงเทพ")),
    ],
)
def test_each_adapter_segments_its_language(
    segmenter: Segmenter, language: str, text: str, expected: tuple[str, ...]
) -> None:
    _require(segmenter)

    assert get_segmenter(segmenter, language=language).segment(text) == expected


@pytest.mark.parametrize(
    "segmenter",
    [Segmenter.JIEBA, Segmenter.MECAB, Segmenter.PYTHAINLP, Segmenter.KHMER_NLTK, Segmenter.ICU],
)
def test_every_model_backed_adapter_pins_a_version(segmenter: Segmenter) -> None:
    # FertilityResult allows a null model version only for UD_GOLD and
    # WHITESPACE, which read annotation or apply no model. Everything else has
    # to say what produced the boundaries: a result whose segmentation cannot be
    # reproduced is not reproducible whatever else the manifest records.
    _require(segmenter)
    language = {
        Segmenter.JIEBA: "zho_Hans",
        Segmenter.MECAB: "jpn_Jpan",
        Segmenter.PYTHAINLP: "tha_Thai",
        Segmenter.KHMER_NLTK: "khm_Khmr",
        Segmenter.ICU: "eng_Latn",
    }[segmenter]

    version = get_segmenter(segmenter, language=language).model_version

    assert version
    assert any(character.isdigit() for character in version)


@pytest.mark.parametrize(
    ("segmenter", "language"),
    [
        (Segmenter.MECAB, "hin_Deva"),
        (Segmenter.JIEBA, "eng_Latn"),
        (Segmenter.PYTHAINLP, "jpn_Jpan"),
        (Segmenter.KHMER_NLTK, "tha_Thai"),
    ],
)
def test_a_language_scoped_segmenter_refuses_other_languages(
    segmenter: Segmenter, language: str
) -> None:
    # jieba on English degenerates to something whitespace-like and MeCab on
    # Hindi hands back the input, so both produce a number that looks fine in a
    # table. §10.3 scopes each of these to one language; using them elsewhere is
    # a refusal rather than a warning.
    _require(segmenter)

    with pytest.raises(SegmenterScopeError, match=language):
        get_segmenter(segmenter, language=language)


def test_icu_is_the_generic_fallback_and_accepts_any_language() -> None:
    _require(Segmenter.ICU)

    assert get_segmenter(Segmenter.ICU, language="eng_Latn").segment(ENGLISH) == (
        "The",
        "cat",
        "sat",
        ".",
    )


def test_icu_and_whitespace_disagree_about_punctuation_on_the_same_english() -> None:
    # Not a defect in either: UAX #29 gives the full stop its own boundary,
    # str.split() leaves it attached to "sat". It changes |W(D)| and therefore
    # fertility on identical input, which is exactly the incomparability the
    # recorded segmenter parameter exists to surface — and it is worth pinning
    # because English is where a reader would assume any two segmenters agree.
    _require(Segmenter.ICU)
    icu = get_segmenter(Segmenter.ICU, language="eng_Latn").segment(ENGLISH)
    whitespace = get_segmenter(Segmenter.WHITESPACE, language="eng_Latn").segment(ENGLISH)

    assert len(icu) == 4
    assert len(whitespace) == 3


@pytest.mark.parametrize("segmenter", [Segmenter.STANZA, Segmenter.UDPIPE, Segmenter.UD_GOLD])
def test_the_unwritten_segmenters_are_reported_as_unbuilt_not_as_refusals(
    segmenter: Segmenter,
) -> None:
    # NotImplementedError, never SegmenterUnavailableError: these are scheduled,
    # and telling a caller to install an extra that would not help sends them
    # after the wrong fix. STANZA and UDPIPE additionally need a pinned model,
    # and UD_GOLD needs gold boundaries the corpus layer does not carry yet.
    with pytest.raises(NotImplementedError):
        get_segmenter(segmenter, language="eng_Latn")


def test_an_unavailable_extra_names_the_install_command() -> None:
    # The error a user actually hits. It must name the pip extra, because the
    # alternative — falling back to whitespace — is the silent incomparability
    # the enum exists to surface.
    missing = [s for s in (Segmenter.JIEBA, Segmenter.MECAB, Segmenter.ICU) if not available(s)]
    if not missing:
        pytest.skip("every optional segmenter is installed here")

    language = {Segmenter.JIEBA: "zho_Hans", Segmenter.MECAB: "jpn_Jpan", Segmenter.ICU: "eng_Latn"}
    with pytest.raises(SegmenterUnavailableError, match="segmenters"):
        get_segmenter(missing[0], language=language[missing[0]])


def test_available_is_honest_about_whitespace() -> None:
    assert available(Segmenter.WHITESPACE)
