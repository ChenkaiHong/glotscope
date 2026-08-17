"""Reading a published §9 result document back (PRD §9, M3).

A :class:`~glotscope.report.Report` is what an *analysis* produces and it holds
full fidelity. A published document is a narrower thing: §9's ``tier0`` block
records ``unreachable_count``, not the id list behind it, so a document can never
rebuild the report that wrote it.

That is why loading returns :class:`LoadedResult` rather than a ``Report``. The
dangerous operation — ``Tier0Report.stage1_exclusions()``, whose empty return
value would look perfectly valid — is not merely disabled here, it never existed
on this type.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glotscope.errors import SchemaVersionError
from glotscope.manifest import SCHEMA_VERSION, Manifest

__all__ = ["LoadedResult", "load_result"]


def _schema_major(version: str) -> str:
    return version.split(".", 1)[0]


@dataclass(frozen=True, slots=True)
class LoadedResult:
    """One published §9 document, read back.

    The manifest is typed because it round-trips exactly. The tier blocks are
    kept as the mappings they were published as: they are already the finished
    numbers, and re-typing them would mean inventing values §9 does not carry.
    """

    manifest: Manifest
    tier0: Mapping[str, Any]
    tier1: Mapping[str, Any] | None = None
    tier2: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Re-assemble the §9 document.

        Built from the typed manifest rather than echoed from what was read, so
        that a document surviving a load/emit cycle unchanged is evidence the
        parse was lossless rather than evidence the bytes were copied.
        """
        document: dict[str, Any] = self.manifest.to_dict()
        document["tier0"] = dict(self.tier0)
        if self.tier1 is not None:
            document["tier1"] = dict(self.tier1)
        if self.tier2 is not None:
            document["tier2"] = dict(self.tier2)
        document["warnings"] = list(self.warnings)
        return document


def load_result(path: str | Path) -> LoadedResult:
    """Read a ``result.json`` written by :meth:`glotscope.report.Report.to_json`.

    Raises:
        SchemaVersionError: if the document's schema major version differs from
            this glotscope's. Minor versions are read, and the document keeps
            the version it was published under.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if "schema_version" not in document:
        raise ValueError(
            f"{path} is not a glotscope result document: it declares no "
            f"schema_version. A tokenizer.json is the usual mix-up — run "
            f"`glotscope analyze` on it first, and read the result.json that "
            f"writes."
        )
    found = document["schema_version"]
    if _schema_major(found) != _schema_major(SCHEMA_VERSION):
        raise SchemaVersionError(str(path), found, _schema_major(SCHEMA_VERSION))
    return LoadedResult(
        manifest=Manifest.from_dict(document),
        tier0=document["tier0"],
        tier1=document.get("tier1"),
        tier2=document.get("tier2"),
        warnings=tuple(document.get("warnings", ())),
    )
