"""Generate a version-pinned Unicode Script table (PRD §14.3, D14).

D14 makes script attribution the paper's independent variable: ``UTR_l`` is
defined over the vocabulary *partitioned by script*, so every published number
depends on which script each token was assigned to.

**That is why the table is committed rather than read from the runtime.** Script
assignments change between Unicode versions, and the supported matrix bundles
four of them:

    Python 3.10 -> Unicode 13.0.0      Python 3.12 -> Unicode 15.0.0
    Python 3.11 -> Unicode 14.0.0      Python 3.13 -> Unicode 15.1.0

``unicodedata`` exposes no script property at all, and ``regex``'s script
property tracks whichever Unicode version that release vendored. Either way the
attribution would differ between CI cells, so a published ``script`` field — and
the paper's independent variable with it — would depend on which interpreter ran
the analysis. G4 promises the opposite: numbers regenerate bit-identically across
OS and Python. A pinned table is what makes that true.

Scripts are recorded as **ISO 15924 short codes** (``Latn``, ``Deva``, ``Hani``)
rather than long names, because that is what FLORES+ language codes already
carry — ``hin_Deva`` — so §14's grouping by script needs no second mapping.

Regenerating, after fetching the three UCD files for the pinned version:

    uv run --no-sync python scripts/generate_script_table.py \\
        --ucd-dir ./ucd --output python/glotscope/data/unicode-scripts.json

Every input is checked against a recorded SHA-256, so an upstream edit fails the
run rather than moving a number. A new Unicode release means a new table, a new
digest, and a deliberate decision to move — never a silent drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

UNICODE_VERSION = "17.0.0"
"""The pinned release. Bumping it is a decision that moves published numbers."""

SOURCE_URL = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd"

SOURCES: Mapping[str, str] = {
    "PropertyValueAliases.txt": (
        "64e9a5f76f7a1e8b5a47d6a1f9a26522a251208f5276bdfa1559dac7cf2e827a"
    ),
    "ScriptExtensions.txt": "ec2107e58825a1586acee8e0911ce18260394ac8b87e535ca325f1ccbeb06bc6",
    "Scripts.txt": "9f5e50d3abaee7d6ce09480f325c706f485ae3240912527e651954d2d6b035bf",
}
"""Filename to SHA-256. The digests are the pin; the version string alone is not,
because the files for a release can be republished."""

_RANGE = re.compile(r"^([0-9A-F]{4,6})(?:\.\.([0-9A-F]{4,6}))?\s*;\s*([^#]+?)\s*(?:#.*)?$")
_ALIAS = re.compile(r"^sc\s*;\s*(\w+)\s*;\s*(\w+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, value)`` for every data line of a UCD file."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.split("#", 1)[0].strip():
            continue
        match = _RANGE.match(line)
        if match is None:
            continue
        start, end, value = match.groups()
        first = int(start, 16)
        yield first, int(end, 16) if end else first, value.strip()


def script_aliases(path: Path) -> dict[str, str]:
    """Long script name to ISO 15924 short code, e.g. ``Devanagari`` -> ``Deva``.

    Short codes map to themselves, because ``ScriptExtensions.txt`` already uses
    them while ``Scripts.txt`` uses long names, and one lookup should serve both.
    """
    aliases: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ALIAS.match(line)
        if match is not None:
            short, long_name = match.groups()
            aliases[long_name] = short
            aliases[short] = short
    return aliases


def script_ranges(path: Path, aliases: Mapping[str, str]) -> list[tuple[int, int, str]]:
    """Codepoint ranges to short script code, sorted and coalesced.

    Coalesced because adjacent ranges sharing a script are common upstream, and a
    smaller table is a shallower bisect on every lookup.
    """
    rows = sorted((start, end, aliases[value]) for start, end, value in _rows(path))
    merged: list[tuple[int, int, str]] = []
    for start, end, code in rows:
        if merged and merged[-1][2] == code and merged[-1][1] + 1 == start:
            previous_start, _, previous_code = merged[-1]
            merged[-1] = (previous_start, end, previous_code)
        else:
            merged.append((start, end, code))
    return merged


def script_extensions(path: Path, aliases: Mapping[str, str]) -> dict[str, list[str]]:
    """Codepoint ranges to the scripts a ``Common``/``Inherited`` character serves.

    UAX #24: a character whose Script is ``Common`` or ``Inherited`` carries a
    Script_Extensions set naming the scripts it is actually used with. ``U+00B7``
    MIDDLE DOT is ``Common`` and used by a dozen scripts — attributing it to
    "Common" and stopping would give punctuation its own bucket and take those
    tokens out of every script's denominator, which is the denominator ``UTR_l``
    is defined over.
    """
    extensions: dict[str, list[str]] = {}
    for start, end, value in _rows(path):
        extensions[f"{start:X}..{end:X}"] = sorted({aliases[part] for part in value.split()})
    return extensions


def build(ucd_dir: Path) -> dict[str, Any]:
    """Assemble the committed table, verifying every input digest first."""
    for name, expected in SOURCES.items():
        actual = sha256_file(ucd_dir / name)
        if actual != expected:
            raise ValueError(
                f"{name}: SHA-256 mismatch — expected {expected}, got {actual}. "
                f"The digest is the pin: a changed input is a different Unicode "
                f"table, and it would move every script-attributed number"
            )

    aliases = script_aliases(ucd_dir / "PropertyValueAliases.txt")
    return {
        "extensions": dict(
            sorted(script_extensions(ucd_dir / "ScriptExtensions.txt", aliases).items())
        ),
        "ranges": [
            [start, end, code]
            for start, end, code in script_ranges(ucd_dir / "Scripts.txt", aliases)
        ],
        "schema_version": 1,
        "source_sha256": dict(sorted(SOURCES.items())),
        "source_url": SOURCE_URL,
        "unicode_version": UNICODE_VERSION,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucd-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        table = build(args.ucd_dir)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(table, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
