from __future__ import annotations

import pytest

from glotscope.enums import TokenClass, TokenizerFamily
from glotscope.utf8 import build_utf8_report, classify_utf8_token


def test_hand_built_vocab_classification_and_stage1() -> None:
    report = build_utf8_report(
        {
            0: b"a",
            1: b"\xe2",
            2: b"\xff",
            3: b"<s>",
            4: b"b",
        },
        unreachable_tokens=(4,),
        special_tokens=(3,),
        family=TokenizerFamily.BYTE_LEVEL,
    )

    assert report.token_classes == {
        TokenClass.WELL_FORMED: 3,
        TokenClass.PARTIAL_UTF8: 1,
        TokenClass.ILL_FORMED_NOT_PARTIAL: 1,
    }
    assert report.ill_formed_vocab_rate == 2 / 5
    assert report.partial_utf8_tokens == (1,)
    assert report.stage1_exclusions() == frozenset({1, 3, 4})
    assert 2 not in report.stage1_exclusions()


def test_utf8_classifier_distinguishes_invalid_from_truncated() -> None:
    assert classify_utf8_token("é".encode()) is TokenClass.WELL_FORMED
    assert classify_utf8_token(b"\xe2\x82") is TokenClass.PARTIAL_UTF8
    assert classify_utf8_token(b"\x80") is TokenClass.PARTIAL_UTF8
    assert classify_utf8_token(b"\xc0") is TokenClass.ILL_FORMED_NOT_PARTIAL
    assert classify_utf8_token(b"\xe0\x80\x80") is TokenClass.PARTIAL_UTF8


def test_utf8_report_rejects_empty_vocab() -> None:
    with pytest.raises(ValueError, match="non-empty vocabulary"):
        build_utf8_report({}, family=TokenizerFamily.BYTE_LEVEL)
