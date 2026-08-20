"""``Segmenter.UD_GOLD`` end to end (PRD §7.1 rule 1, §10.3).

The distinction this file exists to hold: ``UD_GOLD`` reads boundaries a human
annotated, while ``UDPIPE`` and ``STANZA`` predict them with a model. §7.1 rule 1
forbids reporting one under the other's name, and the failure mode is silent — a
predicted fallback returns plausible words, and nothing in the published result
would say the number mixes two conventions.

So most of what follows is about what it refuses.
"""

from __future__ import annotations

import pytest

from glotscope.conllu import parse_conllu
from glotscope.enums import Segmenter
from glotscope.errors import CorpusIntegrityError
from glotscope.segmenters import available, get_segmenter
from glotscope.segmenters.gold import UdGoldSegmenter

_TREEBANK = """\
# text = Hi, there
1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\tSpaceAfter=No
2\t,\t,\tPUNCT\t_\t_\t1\tpunct\t_\t_
3\tthere\tthere\tADV\t_\t_\t1\tadvmod\t_\t_
"""


def _gold() -> UdGoldSegmenter:
    parsed = parse_conllu(_TREEBANK.splitlines(), treebank="UD_English-EWT 2.18")
    segmenter = get_segmenter(Segmenter.UD_GOLD, language="eng_Latn", gold=parsed)
    assert isinstance(segmenter, UdGoldSegmenter)
    return segmenter


def test_an_annotated_sentence_segments_to_its_gold_words() -> None:
    assert _gold().segment("Hi, there") == ("Hi", ",", "there")


def test_the_treebank_is_recorded_as_the_model_version() -> None:
    # §7.1 says record the model version rather than a treebank release, and that
    # is aimed at UDPIPE and STANZA, which have a model. This has none, so the
    # treebank *is* the provenance — and §10.3 requires it by name, because UD
    # Korean treebanks disagree with each other about what a word is.
    segmenter = _gold()

    assert segmenter.segmenter is Segmenter.UD_GOLD
    assert segmenter.model_version == "UD_English-EWT 2.18"


def test_an_un_annotated_document_is_refused_rather_than_predicted() -> None:
    # The whole point. Whitespace would return ("Good", "morning") here and
    # fertility would average annotated and predicted boundaries together.
    with pytest.raises(CorpusIntegrityError) as excinfo:
        _gold().segment("Good morning")

    message = str(excinfo.value)
    assert "Good morning" in message
    assert "UD_English-EWT 2.18" in message


def test_requesting_ud_gold_without_annotation_says_what_to_pass() -> None:
    with pytest.raises(ValueError, match="parse_conllu"):
        get_segmenter(Segmenter.UD_GOLD, language="eng_Latn")


def test_ud_gold_is_not_reported_as_available() -> None:
    # `available` answers "is the dependency importable". Gold annotation is not
    # a dependency and cannot be installed, so False stays the honest answer even
    # now the adapter is built — a caller checks by having a treebank.
    assert available(Segmenter.UD_GOLD) is False


def test_gold_is_ignored_by_the_segmenters_that_predict() -> None:
    # The parameter exists for one member. Passing it to another must not change
    # what that one does, or the signature would be a trap.
    parsed = parse_conllu(_TREEBANK.splitlines())

    segmenter = get_segmenter(Segmenter.WHITESPACE, language="eng_Latn", gold=parsed)

    assert segmenter.segmenter is Segmenter.WHITESPACE
    assert segmenter.segment("Hi, there") == ("Hi,", "there")
