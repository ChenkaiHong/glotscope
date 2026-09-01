"""Check the Stanza language table against the resources file Stanza publishes.

``glotscope.segmenters.stanza_languages`` maps ISO 639-3 codes to what Stanza
calls the language. Stanza's side of that mapping is its ``resources_<version>.json``
— the file its pipeline reads ``lang`` against — so that file is what the table
is audited against, for the resources version the ``segmenters`` extra pins.

Three checks, and the first two fail the run:

* every target in the table is a language Stanza ships a ``tokenize`` model for;
* every ISO 639-3 alias Stanza itself declares agrees with the table;
* every Stanza tokenize language is reachable from some ISO 639-3 code —
  reported, not failed, because a language the table does not reach is a gap
  rather than an error.

Network access is required. Run it when the pinned Stanza version moves::

    uv run --no-sync python scripts/audit_stanza_languages.py --resources-version 1.14.0
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from glotscope.segmenters.stanza_languages import STANZA_LANGUAGE_CODES

RESOURCES_URL = (
    "https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_{version}.json"
)


def _load(version: str, path: Path | None) -> Mapping[str, Any]:
    if path is not None:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    with urllib.request.urlopen(RESOURCES_URL.format(version=version)) as response:
        return dict(json.load(response))


def audit(resources: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(problems, unreached)``: table entries Stanza contradicts, and
    Stanza tokenize languages no ISO 639-3 code reaches."""
    entries = {k: v for k, v in resources.items() if isinstance(v, Mapping)}
    tokenize = {k for k, v in entries.items() if "alias" not in v and "tokenize" in v}
    aliases = {k: v["alias"] for k, v in entries.items() if "alias" in v}

    problems: list[str] = []
    for code, target in STANZA_LANGUAGE_CODES.items():
        if target not in tokenize:
            problems.append(f"{code} -> {target}: Stanza ships no tokenize model under that code")
        declared = aliases.get(code.split("_", 1)[0])
        if declared is not None and declared != target:
            problems.append(f"{code}: Stanza aliases it to {declared!r}, the table says {target!r}")

    # A three-letter Stanza key is an ISO 639-3 code and passes through
    # unchanged; a two-letter one is reachable only through the table.
    reached = set(STANZA_LANGUAGE_CODES.values()) | {k for k in tokenize if len(k) == 3}
    unreached = sorted(tokenize - reached)
    return problems, unreached


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resources-version", default="1.14.0")
    parser.add_argument("--file", type=Path, default=None, help="a downloaded resources file")
    args = parser.parse_args(argv)

    resources = _load(args.resources_version, args.file)
    problems, unreached = audit(resources)

    for problem in problems:
        print(f"problem: {problem}")
    for code in unreached:
        print(f"unreached: Stanza tokenize language {code!r} has no ISO 639-3 entry")
    print(
        f"{len(STANZA_LANGUAGE_CODES)} table entries, {len(problems)} problems, "
        f"{len(unreached)} Stanza languages unreached"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
