from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path

import pytest
from scripts.audit_ud_licenses import (
    audit_archive,
    classify_license,
    license_metadata_matches,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_classify_license_is_conservative_for_noncommercial_and_custom_terms() -> None:
    assert classify_license("CC BY-SA 4.0") == ("CC-BY-SA-4.0", True)
    assert classify_license("CC BY-NC-SA 4.0") == ("CC-BY-NC-SA-4.0", False)
    assert classify_license("C-UDA 1.0") == ("LicenseRef-C-UDA-1.0", None)


def test_audit_archive_records_hashes_metadata_and_summary(tmp_path: Path) -> None:
    archive_path = tmp_path / "ud-treebanks-v2.18.tgz"
    readme = b"""# Example
License: CC BY-SA 4.0
Includes text: yes
"""
    license_text = b"https://creativecommons.org/licenses/by-sa/4.0/legalcode\n"
    _write_archive(
        archive_path,
        {
            "ud-treebanks-v2.18/UD_English-EWT/README.md": readme,
            "ud-treebanks-v2.18/UD_English-EWT/LICENSE.txt": license_text,
        },
    )

    result = audit_archive(
        archive_path,
        release="2.18",
        source_url="https://example.invalid/ud.tgz",
    )

    assert result["summary"] == {
        "commercial_compatible": 1,
        "manual_review": 0,
        "noncommercial": 0,
        "treebanks": 1,
    }
    assert result["treebanks"] == [
        {
            "commercial_ok": True,
            "includes_text": True,
            "license_file": "UD_English-EWT/LICENSE.txt",
            "license_raw": "CC BY-SA 4.0",
            "license_sha256": __import__("hashlib").sha256(license_text).hexdigest(),
            "license_spdx": "CC-BY-SA-4.0",
            "license_metadata_matches": True,
            "readme_file": "UD_English-EWT/README.md",
            "readme_sha256": __import__("hashlib").sha256(readme).hexdigest(),
            "treebank_id": "English-EWT",
        }
    ]


def test_audit_archive_rejects_missing_or_unknown_license(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tgz"
    _write_archive(
        archive_path,
        {
            "root/UD_English-EWT/README.txt": b"License: Mystery\nIncludes text: no\n",
            "root/UD_English-EWT/LICENSE.txt": b"terms\n",
            "root/UD_French-GSD/README.md": b"License: CC BY-SA 4.0\n",
        },
    )

    with pytest.raises(ValueError, match=r"unsupported license.*Mystery"):
        audit_archive(archive_path, release="test", source_url="https://example.invalid")


def test_license_file_mismatch_fails_closed() -> None:
    assert license_metadata_matches(
        "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/legalcode"
    )
    assert not license_metadata_matches(
        "CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/legalcode"
    )


def test_cli_refuses_a_wrong_archive_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_path = tmp_path / "ud.tgz"
    _write_archive(
        archive_path,
        {
            "root/UD_English-EWT/README.md": b"License: CC BY-SA 4.0\nIncludes text: yes\n",
            "root/UD_English-EWT/LICENSE.txt": b"terms\n",
        },
    )
    output = tmp_path / "audit.json"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                str(archive_path),
                "--release",
                "test",
                "--source-url",
                "https://example.invalid/ud.tgz",
                "--expected-sha256",
                "0" * 64,
                "--output",
                str(output),
            ]
        )

    assert exc_info.value.code == 2
    assert "archive SHA-256 mismatch" in capsys.readouterr().err
    assert not output.exists()


def test_committed_ud_218_audit_is_complete_and_fail_closed() -> None:
    audit = json.loads((PROJECT_ROOT / "data/ud-license-audit.json").read_text())

    assert (
        audit["archive_sha256"]
        == "a93fe8520bc4c5ff34670d9a93a5a7689c018c1e59643fa27e03036717841b8a"
    )
    assert audit["summary"] == {
        "commercial_compatible": 268,
        "manual_review": 54,
        "noncommercial": 31,
        "treebanks": 353,
    }
    treebanks = {entry["treebank_id"]: entry for entry in audit["treebanks"]}
    assert len(treebanks) == len(audit["treebanks"]) == 353
    assert treebanks["Korean-GSD"]["commercial_ok"] is True
    assert treebanks["Korean-Kaist"]["commercial_ok"] is True
    assert treebanks["Tamil-TTB"]["commercial_ok"] is False
    assert treebanks["Ancient_Hebrew-PTNK"]["license_metadata_matches"] is False
    assert treebanks["Ancient_Hebrew-PTNK"]["commercial_ok"] is None
    assert audit["summary"] == {
        "commercial_compatible": sum(
            entry["commercial_ok"] is True for entry in audit["treebanks"]
        ),
        "manual_review": sum(entry["commercial_ok"] is None for entry in audit["treebanks"]),
        "noncommercial": sum(entry["commercial_ok"] is False for entry in audit["treebanks"]),
        "treebanks": len(audit["treebanks"]),
    }
    assert all(
        set(entry)
        == {
            "commercial_ok",
            "includes_text",
            "license_file",
            "license_metadata_matches",
            "license_raw",
            "license_sha256",
            "license_spdx",
            "readme_file",
            "readme_sha256",
            "treebank_id",
        }
        for entry in audit["treebanks"]
    )
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", entry[hash_field])
        for entry in audit["treebanks"]
        for hash_field in ("license_sha256", "readme_sha256")
    )
    assert all(
        entry["commercial_ok"] is None
        for entry in audit["treebanks"]
        if not entry["license_metadata_matches"]
    )
    for entry in audit["treebanks"]:
        expected_spdx, expected_commercial_ok = classify_license(entry["license_raw"])
        assert entry["license_spdx"] == expected_spdx
        if entry["license_metadata_matches"]:
            assert entry["commercial_ok"] is expected_commercial_ok
    assert all(
        entry["commercial_ok"] is not True
        for entry in audit["treebanks"]
        if entry["license_raw"].startswith("CC BY-NC")
    )
