"""Rendering the board as Markdown (PRD §16.1, §7).

Two rules shape this file, and both are about what a table implies rather than
about formatting.

**Never invent a cell.** A skipped row has no numbers, and it is rendered as a
skipped row carrying its reason — not dropped. A board that dropped its skips
would look complete while being short, and a reader could not tell which model
was missing or why.

**The caveat travels with the table.** §7.2, §7.5 and §7.7 each carry published
evidence *against* the metric predicting downstream quality. A ranked table is
precisely the artifact that invites the causal reading, so the disclaimer is
rendered into the file rather than left on a documentation page the reader of a
pasted table will never see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["render_markdown"]

_CAVEAT = (
    "These are **diagnostics, not quality predictions.** The literature does not "
    "support the claim that any metric here predicts downstream model quality, "
    "and in several places contradicts it. A row ranking above another is not a "
    "better model."
)

_COLUMNS = (
    "Model",
    "Vocab",
    "Ill-formed",
    "CPT",
    "Parity (worst)",
    "Gini",
    "Tier 2",
    "Notes",
)


def _get(block: Any, *path: str) -> Any:
    """Walk a nested mapping, returning ``None`` at the first missing step."""
    current = block
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _number(value: Any, *, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _percentage(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value * 100:.2f}%"


def _mean_cpt(result: Mapping[str, Any]) -> Any:
    """Mean CPT across the languages measured.

    A single number in a per-model row over a many-language corpus, so it is
    labelled as a mean rather than presented as *the* CPT: the per-language
    values are in the JSON beside it, and collapsing them is a rendering choice,
    not a measurement.
    """
    per_language = _get(result, "tier1", "per_language")
    if not isinstance(per_language, Mapping) or not per_language:
        return None
    values = [
        block["cpt"]
        for block in per_language.values()
        if isinstance(_get(block, "cpt"), (int, float))
    ]
    return sum(values) / len(values) if values else None


def _row_cells(row: Mapping[str, Any]) -> list[str]:
    label = str(row.get("label") or row.get("id"))
    notes: list[str] = []
    if row.get("is_mirror"):
        notes.append(f"**mirror** — {row.get('note') or 'source not stated'}")
    elif row.get("note"):
        notes.append(str(row["note"]))

    skipped = row.get("skipped")
    if skipped:
        # Every measurement cell stays empty on purpose: there is nothing to put
        # in them, and a dash is the honest rendering of a row that did not run.
        return [label, "—", "—", "—", "—", "—", "—", f"**skipped** — {skipped}"]

    result = row.get("result")
    if not isinstance(result, Mapping):
        return [label, "—", "—", "—", "—", "—", "—", "; ".join(notes)]

    return [
        label,
        _number(_get(result, "tier0", "vocab_size")),
        _percentage(_get(result, "tier0", "ill_formed_vocab_rate")),
        _number(_mean_cpt(result)),
        _number(_get(result, "tier1", "corpus_level", "parity", "worst_case_parity")),
        _number(_get(result, "tier1", "corpus_level", "gini")),
        str(row.get("tier2") or "—"),
        "; ".join(notes),
    ]


def _cell(text: str) -> str:
    """One cell's text, made safe to sit in a table row.

    A CommonMark table row is one line, and a pipe starts a new column. Neither
    is guaranteed of what reaches a cell: a skipped row carries the exception
    text that skipped it, and a Hub 404 embeds a blank line in that text — which
    ended the table on the first published board and left the rows after it as
    loose paragraphs. The ``note`` field is free text from ``leaderboard.yaml``
    and could carry a pipe just as easily. Newlines become spaces and pipes are
    escaped; nothing is dropped, so the reason is still there to read.
    """
    flattened = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return flattened.replace("|", "\\|")


def _table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "|".join("---" for _ in _COLUMNS) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(cell) for cell in _row_cells(row)) + " |" for row in rows)
    return lines


def render_markdown(document: Mapping[str, Any]) -> str:
    """Render a leaderboard document as a Markdown page.

    Args:
        document: the mapping :meth:`LeaderboardDocument.to_dict` produces.
    """
    corpus = document.get("corpus", {})
    parameters = document.get("parameters", {})
    rows = document.get("rows", [])

    languages = corpus.get("languages") or []
    header = [
        "# glotscope leaderboard",
        "",
        _CAVEAT,
        "",
        "## What this was computed under",
        "",
        f"- **Corpus** — `{corpus.get('id')}` {corpus.get('version')} "
        f"`{corpus.get('split')}`, {len(languages)} languages, "
        f"sha256 `{corpus.get('sha256')}`",
        f"- **Segmenter** — {parameters.get('segmenter') or 'none (segmenter-free metrics only)'}",
        f"- **Parity reference** — {parameters.get('parity_reference') or 'not computed'}",
        f"- **Rényi** — alpha {parameters.get('renyi_alpha')}, "
        f"normalizer {parameters.get('renyi_normalizer')}",
        f"- **Normalization** — {parameters.get('normalization')}, "
        f"leading space {parameters.get('leading_space')}, "
        f"special tokens {parameters.get('add_special_tokens')}",
        f"- **glotscope** — {document.get('glotscope_version')}, backend {document.get('backend')}",
        f"- **Rows** — {document.get('published')} published, {document.get('skipped')} skipped",
        "",
        "Every row carries its full manifest in `leaderboard.json` beside this file.",
        "",
    ]
    footer = [
        "",
        "`Tier 2` reads *n/a (tokenizer-only)* where a row has no open weights to "
        "read. That is a property of the model, not a failed measurement.",
        "",
        "`CPT` is the **mean** characters-per-token across the languages measured; "
        "the per-language values are in `leaderboard.json`.",
    ]
    return "\n".join([*header, *_table(rows), *footer]) + "\n"
