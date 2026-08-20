"""Tabling published results side by side (PRD §8.2).

``compare`` refuses to table metrics computed under different segmenters, alpha
values, normalizers or language sets. §8.2 says the error message must explain
that the refusal is deliberate, and :class:`~glotscope.errors.IncomparableError`
does.

**Comparability is scoped per metric**, which is the same rule
:mod:`glotscope.results` already applies: every result type declares its own
``comparability_key``, and a Renyi alpha that differs makes two Renyi numbers
incomparable while saying nothing about two compression numbers. Enforcement
goes through :func:`~glotscope.results.require_comparable` rather than through a
second implementation of the same check.

Each metric's key mirrors the corresponding class in :mod:`glotscope.results`,
plus three things those classes leave out because they cannot vary inside a
single process and can vary between two published documents: the corpus
identity, the Unicode normalization form, and whether special tokens were added.

**STRR is deliberately absent.** :class:`~glotscope.results.StrrPair` is
comparable only at a fixed word list — its key is ``lowercased`` and
``n_words`` — and §9 publishes neither. Offering an STRR column here would mean
comparing two numbers whose comparability cannot be checked, which is the exact
failure this module exists to prevent. Publishing those two fields is the
prerequisite, and that is a schema change.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from glotscope.document import LoadedResult
from glotscope.results import require_comparable

__all__ = ["METRICS", "ComparisonTable", "compare"]

_PER_LANGUAGE_METRICS = (
    "fertility",
    "p_continued",
    "cpt",
    "bpt",
    "ctc",
    "compression_rate",
    "roundtrip_rate",
)
_CORPUS_LEVEL_METRICS = ("gini", "renyi_efficiency")
_TIER0_METRICS = (
    "vocab_size",
    "ill_formed_vocab_rate",
    "unreachable_count",
    "byte_fallback_coverage",
)

METRICS: tuple[str, ...] = (
    *_TIER0_METRICS,
    *_PER_LANGUAGE_METRICS,
    "parity",
    *_CORPUS_LEVEL_METRICS,
)
"""Every metric ``compare`` can table, in the §9 document's own names."""

_CORPUS_ROW = "corpus"
"""Row label for a metric that has one value per result rather than one per
language. Gini and Renyi efficiency are properties of the whole evaluated
language set, and giving them a per-language row would invent structure."""

_VOCAB_ROW = "vocab"
"""Row label for a Tier 0 metric, which describes the vocabulary and no corpus."""


@dataclass(frozen=True, slots=True)
class _Key:
    """Adapter making a plain mapping satisfy the ``Comparable`` protocol."""

    fields: Mapping[str, object]

    def comparability_key(self) -> Mapping[str, object]:
        return self.fields


@dataclass(frozen=True, slots=True)
class ComparisonTable:
    """One metric across several results (PRD §8.2)."""

    metric: str
    columns: tuple[str, ...]
    """One label per result, ``<tokenizer id>@<first 12 of the artifact SHA>``.
    The hash is part of the label rather than decoration: two rows can carry the
    same model id and different artifacts, and §11 requires mirror-sourced
    tokenizers to be distinguishable."""

    rows: Mapping[str, tuple[float | None, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "columns": list(self.columns),
            "rows": {row: list(values) for row, values in self.rows.items()},
        }

    def to_markdown(self) -> str:
        header = f"| | {' | '.join(self.columns)} |"
        rule = f"|---|{'---|' * len(self.columns)}"
        lines = [header, rule]
        for row, values in self.rows.items():
            rendered = " | ".join("" if value is None else str(value) for value in values)
            lines.append(f"| {row} | {rendered} |")
        return "\n".join(lines)

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow([self.metric, *self.columns])
        for row, values in self.rows.items():
            writer.writerow([row, *("" if value is None else value for value in values)])
        return buffer.getvalue()


def _label(result: LoadedResult) -> str:
    tokenizer = result.manifest.tokenizer
    return f"{tokenizer.id}@{tokenizer.tokenizer_json_sha256[:12]}"


def _tier1(result: LoadedResult) -> Mapping[str, Any]:
    if result.tier1 is None:
        raise ValueError(
            f"{_label(result)} carries no tier1 block, so it has no corpus "
            f"metric to compare. `glotscope lint` writes a Tier 0 document; "
            f"`glotscope analyze` writes the one compare reads."
        )
    return result.tier1


def _per_language(result: LoadedResult) -> Mapping[str, Any]:
    languages: Mapping[str, Any] = _tier1(result).get("per_language", {})
    return languages


def _corpus_level(result: LoadedResult) -> Mapping[str, Any]:
    level: Mapping[str, Any] = _tier1(result).get("corpus_level", {})
    return level


def _named(member: Enum | None) -> str | None:
    """The enum's published spelling, for a message a reader can act on.

    Comparison is unaffected — every enum here subclasses ``str``, so the member
    and its value are equal — but ``IncomparableError`` renders its operands with
    ``repr``, and ``<Normalization.NFC: 'NFC'>`` is noise where ``'NFC'`` is the
    string the caller passed on the command line.
    """
    return None if member is None else str(member.value)


def _shared_key(result: LoadedResult) -> dict[str, object]:
    """What must match for any two documents to be tabled at all.

    Absent from every ``comparability_key`` in :mod:`glotscope.results` because
    a single process holds them fixed. Two published documents do not.
    """
    corpus = result.manifest.corpus
    parameters = result.manifest.parameters
    return {
        "corpus": None
        if corpus is None
        else (corpus.id, corpus.version, corpus.split, corpus.sha256),
        "languages": frozenset() if corpus is None else frozenset(corpus.languages),
        "normalization": _named(parameters.normalization),
        "add_special_tokens": parameters.add_special_tokens,
    }


def _compression_unit(result: LoadedResult) -> frozenset[object]:
    """The units the compression numbers in this document were normalized by.

    Mirrors :meth:`glotscope.results.CompressionResult.comparability_key`: bytes
    and characters are not interchangeable, and UTF-8 charges 1 byte for ASCII
    against 3 for CJK.

    A set rather than one value, because reading the first language's unit and
    assuming the rest agree is the kind of assumption this module exists to
    check. A document with mixed units compares equal only to another document
    with the same mixture.
    """
    return frozenset(block.get("compression_rate_unit") for block in _per_language(result).values())


def _metric_key(result: LoadedResult, metric: str) -> Mapping[str, object]:
    if metric in _TIER0_METRICS:
        # Nothing. A Tier 0 number is a property of the tokenizer artifact and
        # of nothing else — no corpus, no segmenter, no normalization — which is
        # the definition of the tier (§6). Requiring the corpora to agree before
        # two vocabularies may be tabled would refuse a comparison that is
        # always valid, and an over-refusing tool gets routed around.
        return {}
    key = _shared_key(result)
    parameters = result.manifest.parameters
    if metric in ("fertility", "p_continued"):
        key["segmenter"] = _named(parameters.segmenter)
        key["segmenter_model_version"] = parameters.segmenter_model_version
        key["leading_space"] = parameters.leading_space
    elif metric in ("cpt", "bpt", "ctc", "compression_rate"):
        key["compression_rate_unit"] = _compression_unit(result)
    elif metric == "parity":
        parity: Mapping[str, Any] = _corpus_level(result).get("parity", {})
        key["reference_language"] = parity.get("reference_language")
    elif metric == "renyi_efficiency":
        corpus_level = _corpus_level(result)
        key["alpha"] = corpus_level.get("renyi_alpha")
        key["normalizer"] = corpus_level.get("renyi_normalizer")
        key["nominal_vocab_size"] = corpus_level.get("renyi_nominal_vocab_size")
    return key


def _values(result: LoadedResult, metric: str) -> Mapping[str, float | None]:
    if metric in _TIER0_METRICS:
        return {_VOCAB_ROW: result.tier0.get(metric)}
    if metric == "parity":
        parity: Mapping[str, Any] = _corpus_level(result).get("parity", {})
        per_language: Mapping[str, Any] = parity.get("per_language", {})
        return dict(per_language)
    if metric in _CORPUS_LEVEL_METRICS:
        return {_CORPUS_ROW: _corpus_level(result).get(metric)}
    return {language: block.get(metric) for language, block in _per_language(result).items()}


def compare(results: Sequence[LoadedResult], metric: str) -> ComparisonTable:
    """Table one metric across several published results (PRD §8.2).

    Args:
        results: at least two documents read by
            :func:`~glotscope.document.load_result`.
        metric: one of :data:`METRICS`, named as §9 publishes it.

    Raises:
        IncomparableError: if the results were computed under parameters that
            make this metric's numbers incomparable. This is the point of the
            function, not an obstacle to it.
        ValueError: for an unknown metric, for fewer than two results, or for a
            document carrying no tier1 block.
    """
    if metric not in METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; compare tables one of "
            f"{', '.join(METRICS)}. Names are the ones §9 publishes."
        )
    if len(results) < 2:
        raise ValueError(
            f"comparing needs at least two results, got {len(results)}. A single "
            f"result is already readable as the document it came from."
        )

    reference = _Key(_metric_key(results[0], metric))
    for other in results[1:]:
        require_comparable(reference, _Key(_metric_key(other, metric)))

    columns = tuple(_label(result) for result in results)
    per_result = [_values(result, metric) for result in results]
    row_labels = sorted({row for values in per_result for row in values})
    rows = {row: tuple(values.get(row) for values in per_result) for row in row_labels}
    return ComparisonTable(metric=metric, columns=columns, rows=rows)
