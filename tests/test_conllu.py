"""Parsing CoNLL-U into gold word boundaries (PRD §7.1 rule 1, §10.3).

``Segmenter.UD_GOLD`` is not a segmentation *model*. It reads boundaries a human
annotated, which exist only for the sentences inside a treebank — applying "UD
segmentation" to arbitrary text is a different operation with its own accuracy
and its own version, and conflating the two is what §7.1 rule 1 forbids.

So the parser's job is narrow and its refusals matter more than its coverage:
every sentence it keeps must have words that spell that sentence, because
fertility divides by a word count, and a word list that does not reconstruct its
own text is measuring something else.

Rows here are in real CoNLL-U shape — ten tab-separated fields, comment lines,
multiword-token ranges, empty nodes — because each of those is a way to get the
word count wrong while still producing a plausible number.
"""

from __future__ import annotations

import pytest

from glotscope.conllu import GoldSentences, parse_conllu
from glotscope.errors import CorpusIntegrityError

_ENGLISH = """\
# sent_id = weblog-1
# text = From the AP comes this story :
1\tFrom\tfrom\tADP\tIN\t_\t3\tcase\t3:case\t_
2\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t3\tdet\t3:det\t_
3\tAP\tAP\tPROPN\tNNP\tNumber=Sing\t4\tobl\t4:obl\t_
4\tcomes\tcome\tVERB\tVBZ\tNumber=Sing\t0\troot\t0:root\t_
5\tthis\tthis\tDET\tDT\tNumber=Sing\t6\tdet\t6:det\t_
6\tstory\tstory\tNOUN\tNN\tNumber=Sing\t4\tnsubj\t4:nsubj\t_
7\t:\t:\tPUNCT\t:\t_\t4\tpunct\t4:punct\t_
"""
"""UD_English-EWT's shape, trimmed to the fields that matter here."""


def test_a_sentence_yields_its_surface_words() -> None:
    gold = parse_conllu(_ENGLISH.splitlines())

    assert len(gold.sentences) == 1
    sentence = gold.sentences[0]
    assert sentence.text == "From the AP comes this story :"
    assert sentence.words == ("From", "the", "AP", "comes", "this", "story", ":")


def test_a_multiword_token_is_one_word_not_its_syntactic_parts() -> None:
    # Spanish `al` is one surface word annotated as two syntactic words. Fertility
    # divides by words a tokenizer could have seen, and no tokenizer sees `a` and
    # `el` separately — counting them would inflate the denominator and depress
    # fertility for every language that writes contractions.
    rows = """\
# text = Vámonos al mar
1-2\tVámonos\t_\t_\t_\t_\t_\t_\t_\t_
1\tVamos\tir\tVERB\t_\t_\t0\troot\t_\t_
2\tnos\tnosotros\tPRON\t_\t_\t1\tobj\t_\t_
3-4\tal\t_\t_\t_\t_\t_\t_\t_\t_
3\ta\ta\tADP\t_\t_\t5\tcase\t_\t_
4\tel\tel\tDET\t_\t_\t5\tdet\t_\t_
5\tmar\tmar\tNOUN\t_\t_\t1\tobl\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert gold.sentences[0].words == ("Vámonos", "al", "mar")


def test_an_empty_node_is_not_a_word() -> None:
    # `5.1` is an ellipsis node in the enhanced representation. It has no surface
    # form at all, so counting it would add a word nobody wrote.
    rows = """\
# text = Sue likes coffee and Bill tea
1\tSue\tSue\tPROPN\t_\t_\t2\tnsubj\t_\t_
2\tlikes\tlike\tVERB\t_\t_\t0\troot\t_\t_
3\tcoffee\tcoffee\tNOUN\t_\t_\t2\tobj\t_\t_
4\tand\tand\tCCONJ\t_\t_\t5\tcc\t_\t_
5\tBill\tBill\tPROPN\t_\t_\t2\tconj\t_\t_
5.1\t_\t_\t_\t_\t_\t_\t_\t2:conj\t_
6\ttea\ttea\tNOUN\t_\t_\t5\torphan\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert gold.sentences[0].words == ("Sue", "likes", "coffee", "and", "Bill", "tea")


def test_space_after_no_reconstructs_the_text() -> None:
    # Without honouring SpaceAfter the reconstruction gains a space before the
    # comma, stops matching `# text`, and the sentence is dropped as a mismatch.
    rows = """\
# text = Hi, there
1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\tSpaceAfter=No
2\t,\t,\tPUNCT\t_\t_\t1\tpunct\t_\t_
3\tthere\tthere\tADV\t_\t_\t1\tadvmod\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert gold.sentences[0].words == ("Hi", ",", "there")
    assert gold.n_text_mismatch == 0


def test_a_sentence_whose_words_do_not_spell_its_text_is_dropped_and_counted() -> None:
    # A treebank whose FORM column has been normalized away from the raw text —
    # a curly apostrophe straightened, say — cannot be used for fertility: the
    # word count would describe one string while the tokenizer encodes another.
    rows = """\
# text = Don’t go
1\tDon\tdo\tAUX\t_\t_\t3\taux\t_\tSpaceAfter=No
2\t't\tnot\tPART\t_\t_\t3\tadvmod\t_\t_
3\tgo\tgo\tVERB\t_\t_\t0\troot\t_\t_

# text = Hi there
1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_
2\tthere\tthere\tADV\t_\t_\t1\tadvmod\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert len(gold.sentences) == 1
    assert gold.sentences[0].text == "Hi there"
    assert gold.n_text_mismatch == 1


def test_a_sentence_without_a_text_comment_is_reconstructed_from_its_words() -> None:
    # `# text` is required by the format's own validator, but older releases and
    # hand-built files omit it. Reconstructing is safe precisely because the
    # mismatch check above is the only thing that comment was being used for.
    rows = """\
1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\tSpaceAfter=No
2\t!\t!\tPUNCT\t_\t_\t1\tpunct\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert gold.sentences[0].text == "Hi!"
    assert gold.sentences[0].words == ("Hi", "!")


def test_a_row_with_the_wrong_column_count_is_refused_by_name() -> None:
    with pytest.raises(CorpusIntegrityError) as excinfo:
        parse_conllu(["1\tHi\thi\tINTJ"])

    message = str(excinfo.value)
    assert "10" in message
    assert "found 4" in message


def test_a_file_with_no_usable_sentence_refuses_rather_than_segmenting_nothing() -> None:
    # A gold segmenter with an empty table reports every document as
    # un-annotated, which reads as a finding about the corpus.
    with pytest.raises(CorpusIntegrityError) as excinfo:
        parse_conllu(["# text = Don’t go", "1\tDon\tdo\tAUX\t_\t_\t0\troot\t_\t_"])

    assert "spell" in str(excinfo.value)


def test_blank_lines_separate_sentences_and_do_not_create_them() -> None:
    rows = """\

# text = A
1\tA\ta\tDET\t_\t_\t0\troot\t_\t_


# text = B
1\tB\tb\tNOUN\t_\t_\t0\troot\t_\t_

"""

    gold = parse_conllu(rows.splitlines())

    assert [sentence.text for sentence in gold.sentences] == ["A", "B"]


def test_a_repeated_sentence_with_one_segmentation_is_not_ambiguous() -> None:
    rows = """\
# text = A
1\tA\ta\tDET\t_\t_\t0\troot\t_\t_

# text = A
1\tA\ta\tDET\t_\t_\t0\troot\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert gold.n_sentences == 2
    assert dict(gold.by_text) == {"A": ("A",)}
    assert gold.n_ambiguous == 0


def test_one_text_with_two_segmentations_is_dropped_rather_than_guessed() -> None:
    # Real within Korean UD, where treebanks disagree about eojeol versus
    # morphological segmentation. Taking the first seen would make fertility
    # depend on file order rather than on the annotation.
    rows = """\
# text = ab
1\tab\tab\tX\t_\t_\t0\troot\t_\t_

# text = ab
1\ta\ta\tX\t_\t_\t0\troot\t_\tSpaceAfter=No
2\tb\tb\tX\t_\t_\t1\tdep\t_\t_

# text = c
1\tc\tc\tX\t_\t_\t0\troot\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert dict(gold.by_text) == {"c": ("c",)}
    assert gold.n_ambiguous == 1


def test_the_warning_reports_what_was_dropped() -> None:
    rows = """\
# text = Don’t go
1\tDon\tdo\tAUX\t_\t_\t0\troot\t_\t_

# text = Hi
1\tHi\thi\tINTJ\t_\t_\t0\troot\t_\t_
"""

    gold = parse_conllu(rows.splitlines(), treebank="UD_English-EWT")

    warning = gold.warning()
    assert "UD_English-EWT" in warning
    assert "1 of 2" in warning


def test_a_block_of_only_comments_is_metadata_and_not_a_sentence() -> None:
    # `# newdoc` and `# newpar` open a block that carries no token rows. Counting
    # it would put a sentence in the denominator that has no words in it.
    rows = """\
# newdoc id = weblog

# text = A
1\tA\ta\tDET\t_\t_\t0\troot\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert gold.n_sentences == 1
    assert gold.coverage == 1.0


def test_a_sentence_of_only_empty_nodes_has_no_words_and_is_dropped() -> None:
    # Every row an ellipsis node, so there is no surface form anywhere in the
    # block. Emitting a sentence with zero words would divide fertility by zero
    # further down; it is dropped and counted like any other unusable sentence.
    rows = """\
# text = x
1.1\t_\t_\t_\t_\t_\t_\t_\t0:root\t_

# text = A
1\tA\ta\tDET\t_\t_\t0\troot\t_\t_
"""

    gold = parse_conllu(rows.splitlines())

    assert [sentence.text for sentence in gold.sentences] == ["A"]
    assert gold.n_sentences == 2
    assert gold.n_text_mismatch == 1


def test_coverage_over_no_sentences_is_zero_rather_than_a_division() -> None:
    empty = GoldSentences(treebank="UD_X", sentences=(), n_sentences=0, n_text_mismatch=0)

    assert empty.coverage == 0.0
