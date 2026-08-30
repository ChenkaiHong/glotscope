"""Re-checking a published board against a fresh run (PRD §12.3, §16.1).

§16.1 requires a nightly re-run against pinned revisions that fails if any
published number moves. **Pinning does not make that redundant**, which is the
reason this file exists rather than a comment saying the revisions are fixed:

* a tiktoken encoding is pinned by the installed library version, not by content
  — the merge ranks come from a URL and the split pattern from
  ``tiktoken_ext.openai_public``;
* a repository can be deleted, or newly gated, under a revision that still
  resolves;
* our own code can change a number without anyone noticing.

What must **not** fail the check is everything that legitimately varies. ``verify``
already drew that line for a single result — environment, backend and
``glotscope_version`` are reported rather than compared, because comparing them
would make every release invalidate every published number. Drawing it
differently here would produce a nightly job that goes red on a release instead
of on a regression, and a job that cries wolf is a job everyone ignores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["ALL_TIERS", "check_board"]

ALL_TIERS = ("tier0", "tier1", "tier2")

_VOLATILE = frozenset({"environment", "backend", "glotscope_version", "warnings"})
"""Keys reported rather than compared.

``warnings`` travels with them because it carries provenance commentary — which
link of the reference-set chain supplied a set, whether a revision was pinned —
and its wording is ours to improve. A reworded warning is not a number moving.
"""


def _rows_by_id(board: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("id")): row for row in board.get("rows", []) if isinstance(row, Mapping)}


def _differences(published: Any, regenerated: Any, path: str) -> list[str]:
    """Every leaf that differs, named by its path.

    Reported exhaustively rather than stopping at the first: a nightly job that
    named one moved number would send someone to fix it and hide the other four.
    """
    if isinstance(published, Mapping) and isinstance(regenerated, Mapping):
        found: list[str] = []
        for key in sorted(set(published) | set(regenerated)):
            if key in _VOLATILE:
                continue
            if key not in published:
                found.append(f"{path}.{key}: absent from the published board")
            elif key not in regenerated:
                found.append(f"{path}.{key}: not regenerated")
            else:
                found.extend(_differences(published[key], regenerated[key], f"{path}.{key}"))
        return found

    if isinstance(published, list) and isinstance(regenerated, list):
        if len(published) != len(regenerated):
            return [f"{path}: {len(published)} entries published, {len(regenerated)} regenerated"]
        return [
            difference
            for index, (was, now) in enumerate(zip(published, regenerated, strict=True))
            for difference in _differences(was, now, f"{path}[{index}]")
        ]

    if published != regenerated:
        return [f"{path}: published {published!r}, regenerated {regenerated!r}"]
    return []


def check_board(
    published: Mapping[str, Any],
    regenerated: Mapping[str, Any],
    *,
    tiers: Sequence[str] = ALL_TIERS,
) -> list[str]:
    """Compare a published board against a fresh run.

    Args:
        published: the board on disk, as ``results/leaderboard.json`` holds it.
        regenerated: a board produced by re-running the same configuration.
        tiers: which tier blocks were actually recomputed. The nightly job runs
            anonymously where FLORES+ is gated and can regenerate Tier 0 alone;
            comparing a tier it never measured would report a column as
            unchanged when nothing looked at it.

    Returns:
        One entry per difference, most specific first. Empty means nothing moved.
    """
    differences: list[str] = []
    was_rows, now_rows = _rows_by_id(published), _rows_by_id(regenerated)

    for row_id in sorted(set(was_rows) | set(now_rows)):
        if row_id not in now_rows:
            differences.append(f"{row_id}: missing from the regenerated board")
            continue
        if row_id not in was_rows:
            differences.append(f"{row_id}: not in the published board")
            continue

        was, now = was_rows[row_id], now_rows[row_id]
        was_result, now_result = was.get("result"), now.get("result")

        if was_result is not None and now_result is None:
            differences.append(f"{row_id}: stopped publishing — {now.get('skipped')}")
            continue
        if was_result is None and now_result is not None:
            differences.append(f"{row_id}: now publishes, and the board does not say so")
            continue
        if was_result is None or now_result is None:
            continue

        for key in ("schema_version", *tiers):
            # A key absent from both is a tier this board never carried. A key
            # absent from the regenerated run alone is a tier that was not
            # recomputed — the caller said so through ``tiers`` — and comparing
            # it would report a column as unchanged when nothing looked at it.
            if key not in was_result or key not in now_result:
                continue
            differences.extend(_differences(was_result[key], now_result[key], f"{row_id}.{key}"))

    return differences
