"""Generate a release-pinned Universal Dependencies license audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

LicenseClassification = tuple[str, bool | None]

_LICENSES: dict[str, LicenseClassification] = {
    "CC BY-SA 4.0": ("CC-BY-SA-4.0", True),
    "CC BY-SA 3.0": ("CC-BY-SA-3.0", True),
    "CC BY 4.0": ("CC-BY-4.0", True),
    "CC0 1.0": ("CC0-1.0", True),
    "GNU GPL 3.0": ("GPL-3.0-only", True),
    "PD": ("LicenseRef-Public-Domain", True),
    "CC BY-NC-SA 4.0": ("CC-BY-NC-SA-4.0", False),
    "CC BY-NC-SA 3.0": ("CC-BY-NC-SA-3.0", False),
    "CC BY-NC-SA 2.5": ("CC-BY-NC-SA-2.5", False),
    "CC BY-NC-ND 4.0": ("CC-BY-NC-ND-4.0", False),
    "CC BY-NC 4.0": ("CC-BY-NC-4.0", False),
    "C-UDA 1.0": ("LicenseRef-C-UDA-1.0", None),
    "LGPL-LR": ("LicenseRef-LGPL-LR", None),
}

_LICENSE_MARKERS: dict[str, tuple[str, ...]] = {
    "CC BY-SA 4.0": ("creativecommons.org/licenses/by-sa/4.0",),
    "CC BY-SA 3.0": ("creativecommons.org/licenses/by-sa/3.0",),
    "CC BY 4.0": ("creativecommons.org/licenses/by/4.0",),
    "CC0 1.0": ("cc0 1.0",),
    "GNU GPL 3.0": ("gnu general public license", "version 3"),
    "PD": ("public domain",),
    "CC BY-NC-SA 4.0": ("creativecommons.org/licenses/by-nc-sa/4.0",),
    "CC BY-NC-SA 3.0": ("creativecommons.org/licenses/by-nc-sa/3.0",),
    "CC BY-NC-SA 2.5": ("creativecommons.org/licenses/by-nc-sa/2.5",),
    "CC BY-NC-ND 4.0": ("creativecommons.org/licenses/by-nc-nd/4.0",),
    "CC BY-NC 4.0": ("creativecommons.org/licenses/by-nc/4.0",),
    "C-UDA 1.0": ("computational use of data agreement v1.0",),
    "LGPL-LR": ("lesser general public license for linguistic resources",),
}


def classify_license(raw_license: str) -> LicenseClassification:
    """Return SPDX-like identifier and conservative commercial-use status."""
    try:
        return _LICENSES[raw_license]
    except KeyError as error:
        raise ValueError(f"unsupported license: {raw_license}") from error


def license_metadata_matches(raw_license: str, license_text: str) -> bool:
    """Return whether LICENSE.txt contains evidence for the README declaration."""
    classify_license(raw_license)
    normalized = license_text.casefold()
    return all(marker in normalized for marker in _LICENSE_MARKERS[raw_license])


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_value(readme: str, field: str) -> str:
    prefix = f"{field}:"
    values = [
        line.removeprefix(prefix).strip() for line in readme.splitlines() if line.startswith(prefix)
    ]
    if len(values) != 1 or not values[0]:
        raise ValueError(f"expected one non-empty {field!r} field")
    return values[0]


def _includes_text(raw_value: str) -> bool | None:
    normalized = raw_value.casefold()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    if normalized in {"unknown", "n/a"}:
        return None
    raise ValueError(f"unsupported Includes text value: {raw_value}")


def _treebank_file(member_name: str) -> tuple[str, str] | None:
    parts = PurePosixPath(member_name).parts
    for index, part in enumerate(parts):
        if part.startswith("UD_") and index + 1 < len(parts):
            filename = parts[index + 1]
            if index + 2 == len(parts) and filename in {"README.md", "README.txt", "LICENSE.txt"}:
                return part.removeprefix("UD_"), filename
    return None


def audit_archive(archive_path: Path, *, release: str, source_url: str) -> dict[str, Any]:
    """Audit every treebank README and license in an official UD archive."""
    files: dict[str, dict[str, bytes]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            resolved = _treebank_file(member.name)
            if resolved is None or not member.isfile():
                continue
            treebank_id, filename = resolved
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read {member.name}")
            treebank = files.setdefault(treebank_id, {})
            if filename in treebank:
                raise ValueError(f"duplicate {filename} for {treebank_id}")
            treebank[filename] = extracted.read()

    entries: list[dict[str, Any]] = []
    for treebank_id, treebank_files in sorted(files.items()):
        readme_name = "README.md" if "README.md" in treebank_files else "README.txt"
        if readme_name not in treebank_files or "LICENSE.txt" not in treebank_files:
            raise ValueError(f"missing README or LICENSE.txt for {treebank_id}")
        readme_bytes = treebank_files[readme_name]
        license_bytes = treebank_files["LICENSE.txt"]
        readme = readme_bytes.decode("utf-8")
        raw_license = _metadata_value(readme, "License")
        spdx, declared_commercial_ok = classify_license(raw_license)
        metadata_matches = license_metadata_matches(
            raw_license, license_bytes.decode("utf-8", errors="replace")
        )
        commercial_ok = declared_commercial_ok if metadata_matches else None
        entries.append(
            {
                "commercial_ok": commercial_ok,
                "includes_text": _includes_text(_metadata_value(readme, "Includes text")),
                "license_file": f"UD_{treebank_id}/LICENSE.txt",
                "license_raw": raw_license,
                "license_sha256": _sha256_bytes(license_bytes),
                "license_spdx": spdx,
                "license_metadata_matches": metadata_matches,
                "readme_file": f"UD_{treebank_id}/{readme_name}",
                "readme_sha256": _sha256_bytes(readme_bytes),
                "treebank_id": treebank_id,
            }
        )

    return {
        "archive_sha256": _sha256_file(archive_path),
        "schema_version": 1,
        "source_url": source_url,
        "summary": {
            "commercial_compatible": sum(entry["commercial_ok"] is True for entry in entries),
            "manual_review": sum(entry["commercial_ok"] is None for entry in entries),
            "noncommercial": sum(entry["commercial_ok"] is False for entry in entries),
            "treebanks": len(entries),
        },
        "treebanks": entries,
        "ud_release": release,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    actual_sha256 = _sha256_file(args.archive)
    if actual_sha256 != args.expected_sha256:
        parser.error(
            f"archive SHA-256 mismatch: expected {args.expected_sha256}, got {actual_sha256}"
        )
    result = audit_archive(args.archive, release=args.release, source_url=args.source_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
