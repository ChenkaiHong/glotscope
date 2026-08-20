"""MorphyNet's inflectional TSV as character-offset gold (PRD §7.7c, §10.1).

§7.7(c) scores full alignment against MorphyNet, and :mod:`glotscope.morphology`
computes it — over :class:`~glotscope.morphology.AlignedWord`, whose gold and
predicted pieces must spell the same string, because boundaries are **character
offsets** and offsets into two different strings are not comparable.

MorphyNet does not hand you that. Its segmentation column is *canonical*, not
surface:

===============  ==================  ==========================================
Inflected form   Gold column         Why it cannot be used as offsets
===============  ==================  ==========================================
``microtoming``  ``microtome|ing``   spells ``microtomeing`` — the stem's silent
                                     ``e`` is deleted in the surface form
``далайгаар``    ``далай|аар``       spells ``далайаар`` — the buffer consonant
                                     is in the form and not in the gold
``ate``          ``-``               suppletive; no segmentation is given at all
===============  ==================  ==========================================

Measured over the published files rather than assumed: of **649,593** English
rows 66.05% carry ``-`` and a further 9.75% do not spell their own surface form,
leaving **24.20%** usable; of **30,129** Mongolian rows none carry ``-`` and
69.59% mismatch, leaving **30.41%**. A loader that trusts the column publishes an
F1 computed against offsets into a string nobody tokenized, and it looks exactly
like a successful run.

So this module drops what it cannot use and **counts every drop**. Discarding
three quarters of a corpus is a fact about coverage, and it belongs in the
warnings array beside the number it constrains (§9).

Inflectional only. The derivational file names one affix per (source, target)
pair rather than a segmentation, so composing it into full boundaries means
chaining derivations recursively — a different piece of work with its own
correctness argument, and §7.7(c)'s measure is defined over the inflectional
gold. Handed the derivational file, this module says so by name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from glotscope.errors import CorpusIntegrityError

__all__ = ["GoldSegmentations", "parse_morphynet"]

CORPUS_ID = "morphynet"
"""The registry id, so the refusals name the resource the same way §9 does."""

_COLUMNS = 4
"""lemma, inflected form, morphological features, morpheme segmentation."""

_DERIVATIONAL_COLUMNS = 6
"""source, target, source POS, target POS, morpheme, prefix|suffix."""

_NO_SEGMENTATION = "-"
"""What MorphyNet writes where a form has no segmentation. Two thirds of the
English file. Read as a one-morpheme word it would score every gold boundary as
absent, which is a claim about the tokenizer rather than about the datum."""

_SEPARATOR = "|"


@dataclass(frozen=True, slots=True)
class GoldSegmentations:
    """Gold segmentations for one language, with what it cost to get them.

    The counters are not diagnostics. They are the denominator of every number
    computed downstream: ``full_alignment`` describes the usable subset, and a
    reader who is not told the subset is a quarter of the file will read it as a
    statement about the language.
    """

    language: str
    segmentations: Mapping[str, tuple[str, ...]]
    """Surface form to its morphemes, in file order. Read-only. Every entry
    satisfies ``"".join(morphemes) == form``, which is what makes the offsets
    computed from it comparable to the tokenizer's."""

    n_rows: int
    """Non-blank rows read, before any drop."""

    n_unsegmented: int
    """Rows whose segmentation column is ``-``."""

    n_surface_mismatch: int
    """Rows whose morphemes do not spell the inflected form."""

    n_empty_morpheme: int
    """Rows carrying a zero-length morpheme. ``a||b`` yields cumulative offsets
    1 and 1, which collapse in a set, so a two-boundary claim silently becomes a
    one-boundary one."""

    n_ambiguous: int
    """Forms carrying two or more conflicting segmentations, dropped whole. There
    is no gold answer to score against, and taking the first seen would make the
    result depend on row order rather than on the data."""

    n_usable_rows: int
    """Rows that survived every filter, counted **as rows**.

    Not the same as :attr:`n_usable`, and the difference is why this field
    exists. A form is scored once however many rows carry it, so ``n_usable``
    counts types while every drop counter above counts rows. Dividing one by the
    other would report a file whose usable rows repeat a form as mostly
    unusable — a claim about coverage the data does not make."""

    @property
    def n_usable(self) -> int:
        """Distinct forms available to score — the gold set's size."""
        return len(self.segmentations)

    @property
    def coverage(self) -> float:
        """Usable share of the rows read, on a row basis. 0.0 over no rows."""
        if self.n_rows == 0:
            return 0.0
        return self.n_usable_rows / self.n_rows

    def warning(self) -> str:
        """The §9 warnings entry naming what was dropped and why.

        Both counts appear because they answer different questions: the row share
        says how much of the file survived, and the type count is the actual
        denominator of the alignment scores it accompanies.
        """
        return (
            f"{self.language}: {self.n_usable_rows} of {self.n_rows} MorphyNet rows are "
            f"usable as character offsets ({self.coverage:.2%}), giving "
            f"{self.n_usable} distinct forms to score. "
            f"{self.n_unsegmented} carry the '-' sentinel, {self.n_surface_mismatch} "
            f"give a canonical segmentation that does not spell the surface form, "
            f"{self.n_empty_morpheme} contain an empty morpheme, and "
            f"{self.n_ambiguous} forms carry conflicting segmentations. "
            f"Morphological alignment below describes that usable subset, not the "
            f"language and not the file."
        )


def parse_morphynet(lines: Iterable[str], *, language: str) -> GoldSegmentations:
    """Parse MorphyNet inflectional rows into gold segmentations.

    Args:
        lines: rows of a ``<lang>.inflectional.v1.tsv``, blank lines allowed.
        language: recorded on the result and named in every refusal.

    Returns:
        The usable subset with the drop counts that describe it.

    Raises:
        CorpusIntegrityError: if a row does not have four columns — the
            derivational file has six and is named explicitly, because pointing
            the loader at it is the likeliest mistake and its sixth column would
            otherwise be silently ignored — or if no row survives, since an F1
            over nothing reads as a finding about the tokenizer.
    """
    n_rows = 0
    n_unsegmented = 0
    n_surface_mismatch = 0
    n_empty_morpheme = 0
    candidates: dict[str, set[tuple[str, ...]]] = {}
    rows_per_form: dict[str, int] = {}
    """How many rows carried each form, so the row share can be reported without
    conflating it with the number of forms actually scored."""

    for line in lines:
        row = line.rstrip("\n")
        if not row:
            continue
        n_rows += 1
        fields = row.split("\t")
        if len(fields) != _COLUMNS:
            raise CorpusIntegrityError(CORPUS_ID, _column_count_reason(language, len(fields)))

        form, segmentation = fields[1], fields[3]
        if segmentation == _NO_SEGMENTATION:
            n_unsegmented += 1
            continue
        morphemes = tuple(segmentation.split(_SEPARATOR))
        if any(not morpheme for morpheme in morphemes):
            n_empty_morpheme += 1
            continue
        if "".join(morphemes) != form:
            n_surface_mismatch += 1
            continue
        candidates.setdefault(form, set()).add(morphemes)
        rows_per_form[form] = rows_per_form.get(form, 0) + 1

    segmentations = {
        form: next(iter(readings)) for form, readings in candidates.items() if len(readings) == 1
    }
    n_ambiguous = len(candidates) - len(segmentations)
    n_usable_rows = sum(rows_per_form[form] for form in segmentations)

    if not segmentations:
        raise CorpusIntegrityError(
            CORPUS_ID,
            f"no usable gold segmentation for {language!r} in {n_rows} rows: "
            f"{n_unsegmented} carry the '-' sentinel, {n_surface_mismatch} do not "
            f"spell their own surface form, {n_empty_morpheme} contain an empty "
            f"morpheme, {n_ambiguous} forms conflict. Boundaries are character "
            f"offsets, so none of those can be scored against a tokenization",
        )

    return GoldSegmentations(
        language=language,
        segmentations=MappingProxyType(segmentations),
        n_rows=n_rows,
        n_unsegmented=n_unsegmented,
        n_surface_mismatch=n_surface_mismatch,
        n_empty_morpheme=n_empty_morpheme,
        n_ambiguous=n_ambiguous,
        n_usable_rows=n_usable_rows,
    )


def _column_count_reason(language: str, found: int) -> str:
    reason = (
        f"a MorphyNet inflectional row for {language!r} has {_COLUMNS} "
        f"tab-separated columns (lemma, inflected form, features, segmentation); "
        f"found {found}"
    )
    if found == _DERIVATIONAL_COLUMNS:
        reason += (
            ". That is the derivational file, which names one affix per "
            "source/target pair rather than a segmentation — §7.7(c) is defined "
            "over the inflectional gold"
        )
    return reason
