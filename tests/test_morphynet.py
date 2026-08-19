"""Parsing MorphyNet's inflectional TSV into character-offset gold (PRD §7.7c).

Every row here is either copied verbatim from the upstream files or built to the
shape they actually have, because the trap this module exists for is invisible
from the format description: MorphyNet's segmentation column is **canonical**,
not surface. ``microtome|ing`` spells ``microtomeing`` and the inflected form is
``microtoming``; Mongolian ``далай|аар`` spells ``далайаар`` against the form
``далайгаар``. Boundaries are character offsets, so a loader that trusts the
column indexes into a string that was never tokenized and every number computed
from it is meaningless.

Measured over the real files: of 649,593 English rows, 66.05% carry ``-`` and a
further 9.75% do not spell their own surface form, leaving 24.20% usable. Of
30,129 Mongolian rows none carry ``-`` and 69.59% mismatch, leaving 30.41%.
Dropping three quarters of a corpus is a fact about coverage that belongs in the
warnings array, not in a silent filter.
"""

from __future__ import annotations

import pytest

from glotscope.errors import CorpusIntegrityError
from glotscope.morphynet import GoldSegmentations, parse_morphynet

_ENGLISH_ROWS = (
    "microtome\tmicrotomes\tN|PL\tmicrotome|s",
    "microtome\tmicrotoming\tV|V.PTCP;PRS\tmicrotome|ing",
    "eat\tate\tV;PST\t-",
    "eat\teaten\tV|V.PTCP;PST\teat|en",
)
"""Verbatim from ``eng/eng.inflectional.v1.tsv``: one usable row, one canonical
mismatch, one suppletive form with no segmentation, one usable row."""


def test_parses_a_usable_row_into_its_morphemes() -> None:
    gold = parse_morphynet(["microtome\tmicrotomes\tN|PL\tmicrotome|s"], language="eng")

    assert dict(gold.segmentations) == {"microtomes": ("microtome", "s")}
    assert gold.n_rows == 1


def test_the_dash_sentinel_is_counted_and_dropped() -> None:
    gold = parse_morphynet(
        ["eat\tate\tV;PST\t-", "eat\teaten\tV|V.PTCP;PST\teat|en"], language="eng"
    )

    assert dict(gold.segmentations) == {"eaten": ("eat", "en")}
    assert gold.n_unsegmented == 1
    assert gold.n_surface_mismatch == 0


def test_a_canonical_segmentation_that_does_not_spell_the_form_is_dropped() -> None:
    # ``microtome`` + ``ing`` is ``microtomeing``; the form is ``microtoming``.
    gold = parse_morphynet(
        [
            "microtome\tmicrotoming\tV|V.PTCP;PRS\tmicrotome|ing",
            "microtome\tmicrotomes\tN|PL\tmicrotome|s",
        ],
        language="eng",
    )

    assert dict(gold.segmentations) == {"microtomes": ("microtome", "s")}
    assert gold.n_surface_mismatch == 1
    assert gold.n_unsegmented == 0


def test_mongolian_allomorphy_is_the_same_mismatch() -> None:
    # Verbatim from ``mon/mon.inflectional.v1.tsv``. 69.59% of that file is this
    # case: the buffer consonant in the surface form is absent from the gold.
    gold = parse_morphynet(
        ["далай\tдалайгаар\tN;INS\tдалай|аар", "далай\tдалайд\tN;DAT\tдалай|д"],
        language="mon",
    )

    assert dict(gold.segmentations) == {"далайд": ("далай", "д")}
    assert gold.n_surface_mismatch == 1


def test_repeated_identical_rows_collapse_to_one_type() -> None:
    row = "microtome\tmicrotomes\tN|PL\tmicrotome|s"
    gold = parse_morphynet([row, row], language="eng")

    assert dict(gold.segmentations) == {"microtomes": ("microtome", "s")}
    assert gold.n_rows == 2
    assert gold.n_ambiguous == 0
    # Types and rows are counted separately on purpose. Both rows were usable, so
    # coverage is 1.0 — dividing the one *form* by the two *rows* would report
    # half the file as unusable when nothing was dropped, and that number goes
    # into the §9 warnings array where a reader takes it at face value.
    assert gold.n_usable == 1
    assert gold.n_usable_rows == 2
    assert gold.coverage == pytest.approx(1.0)


def test_an_ambiguous_form_takes_all_of_its_rows_out_of_the_row_count() -> None:
    # Two readings of `unlocks` across two rows, plus one usable row. The
    # ambiguous form is dropped whole, so neither of its rows may count as usable
    # — otherwise coverage would credit rows nothing was scored from.
    gold = parse_morphynet(
        [
            "unlock\tunlocks\tV|PRS;3;SG\tun|lock|s",
            "unlock\tunlocks\tN|PL\tunlock|s",
            "cat\tcats\tN|PL\tcat|s",
        ],
        language="eng",
    )

    assert gold.n_rows == 3
    assert gold.n_usable_rows == 1
    assert gold.coverage == pytest.approx(1 / 3)


def test_a_form_with_two_different_segmentations_is_dropped_rather_than_guessed() -> None:
    gold = parse_morphynet(
        [
            "unlock\tunlocks\tV|PRS;3;SG\tun|lock|s",
            "unlock\tunlocks\tN|PL\tunlock|s",
            "cat\tcats\tN|PL\tcat|s",
        ],
        language="eng",
    )

    assert dict(gold.segmentations) == {"cats": ("cat", "s")}
    assert gold.n_ambiguous == 1


def test_an_empty_morpheme_is_dropped_rather_than_producing_a_phantom_boundary() -> None:
    # ``a||b`` would give cumulative offsets 1 and 1, which collapse in a set, so
    # a two-boundary claim silently becomes one.
    gold = parse_morphynet(["ab\tab\tN\ta||b", "cat\tcats\tN|PL\tcat|s"], language="eng")

    assert dict(gold.segmentations) == {"cats": ("cat", "s")}
    assert gold.n_empty_morpheme == 1


def test_a_single_morpheme_row_is_kept() -> None:
    # No gold boundary is not the same as no data: a monomorphemic word the
    # tokenizer splits is a false positive, and dropping it would hide that.
    gold = parse_morphynet(["cat\tcat\tN\tcat"], language="eng")

    assert dict(gold.segmentations) == {"cat": ("cat",)}


def test_blank_lines_are_not_rows() -> None:
    gold = parse_morphynet(["", "cat\tcat\tN\tcat", ""], language="eng")

    assert gold.n_rows == 1


def test_the_derivational_file_is_refused_by_name() -> None:
    # Six columns, verbatim from ``eng/eng.derivational.v1.tsv``. Pointing the
    # loader at the wrong MorphyNet file is the likeliest user error, and the
    # sixth column would otherwise be silently ignored.
    with pytest.raises(CorpusIntegrityError) as excinfo:
        parse_morphynet(["sense\tnonsense\tN\tN\tnon\tprefix"], language="eng")

    message = str(excinfo.value)
    assert "derivational" in message
    assert "4" in message


def test_coverage_and_the_warning_report_what_was_dropped() -> None:
    gold = parse_morphynet(_ENGLISH_ROWS, language="eng")

    assert dict(gold.segmentations) == {
        "microtomes": ("microtome", "s"),
        "eaten": ("eat", "en"),
    }
    assert gold.n_rows == 4
    assert gold.n_usable == 2
    assert gold.coverage == pytest.approx(0.5)

    warning = gold.warning()
    assert "eng" in warning
    assert "2 of 4" in warning
    assert "canonical" in warning


def test_a_file_with_nothing_usable_refuses_rather_than_scoring_nothing() -> None:
    # An F1 over zero words reads as a finding about the tokenizer. §7.7's own
    # refusal is downstream in ``morphology()``; this one names the cause.
    with pytest.raises(CorpusIntegrityError) as excinfo:
        parse_morphynet(["eat\tate\tV;PST\t-"], language="eng")

    assert "eng" in str(excinfo.value)


def test_a_row_that_is_neither_file_still_names_the_expected_shape() -> None:
    with pytest.raises(CorpusIntegrityError) as excinfo:
        parse_morphynet(["cat\tcats\tN|PL"], language="eng")

    message = str(excinfo.value)
    assert "found 3" in message
    assert "derivational" not in message


def test_coverage_over_no_rows_is_zero_rather_than_a_division() -> None:
    # Unreachable through `parse_morphynet`, which refuses an empty file, and
    # guarded anyway: the counters are public and this one divides by them.
    empty = GoldSegmentations(
        language="eng",
        segmentations={},
        n_rows=0,
        n_unsegmented=0,
        n_surface_mismatch=0,
        n_empty_morpheme=0,
        n_ambiguous=0,
        n_usable_rows=0,
    )

    assert empty.coverage == 0.0
