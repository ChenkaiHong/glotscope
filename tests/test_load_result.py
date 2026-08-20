"""Reading a published §9 document back (M3's manifest round-trip).

The document records *counts*, not the Tier 0 id lists (§9's ``tier0`` block is
``ill_formed_vocab_rate`` and ``unreachable_count``). A loaded document therefore
cannot reconstruct a :class:`~glotscope.report.Tier0Report`, and pretending
otherwise would hand callers a ``stage1_exclusions()`` that returns an empty set
and looks valid. :func:`~glotscope.document.load_result` returns a separate
read-only type that never had the method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glotscope.document import load_result
from glotscope.enums import Algorithm, Backend, Capability, Normalization, Segmenter
from glotscope.errors import SchemaVersionError
from glotscope.manifest import canonical_json

FIXTURE = Path(__file__).resolve().parents[1] / "verification"
RESULT = FIXTURE / "result.json"

pytestmark = pytest.mark.skipif(
    not RESULT.is_file(),
    reason=(
        "the G4 fixture lives in the repository, not in the distribution: "
        "tests/ ships in the sdist and verification/ deliberately does not, so "
        "these run from a checkout and skip from an unpacked release"
    ),
)


def test_a_committed_result_re_emits_byte_identically() -> None:
    # Arrange
    original = RESULT.read_text(encoding="utf-8")

    # Act
    reloaded = load_result(RESULT)

    # Assert
    assert canonical_json(reloaded.to_dict()) + "\n" == original


def test_the_manifest_block_loads_as_typed_values() -> None:
    # Arrange
    loaded = load_result(RESULT)

    # Act
    manifest = loaded.manifest

    # Assert
    assert manifest.backend is Backend.PYTHON
    assert manifest.tokenizer.algorithm is Algorithm.BYTE_LEVEL_BPE
    assert manifest.parameters.segmenter is Segmenter.WHITESPACE
    assert manifest.parameters.normalization is Normalization.NFC
    assert manifest.corpus is not None
    assert manifest.corpus.capabilities == frozenset({Capability.PARALLEL})


def _fixture_document() -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    return parsed


def test_a_three_tier_document_round_trips(tmp_path: Path) -> None:
    # G2's claim is that one document spans every tier that ran. The committed
    # fixture has no weights, so the tier2 block is only exercised here.
    # Arrange
    document = _fixture_document()
    document["tier2"] = {
        "candidate_count": 3,
        "confidence": "HIGH",
        "indicator": "l2_e_in",
        "tied": False,
        "top_pct": 2.0,
    }
    path = tmp_path / "three-tier.json"
    original = canonical_json(document) + "\n"
    path.write_text(original, encoding="utf-8")

    # Act
    reloaded = load_result(path)

    # Assert
    assert canonical_json(reloaded.to_dict()) + "\n" == original


def test_a_document_from_a_different_schema_major_is_refused(tmp_path: Path) -> None:
    # Arrange
    document = _fixture_document()
    document["schema_version"] = "2.0"
    path = tmp_path / "future.json"
    path.write_text(canonical_json(document), encoding="utf-8")

    # Act / Assert
    with pytest.raises(SchemaVersionError) as excinfo:
        load_result(path)
    assert "2.0" in str(excinfo.value)


def test_an_earlier_minor_schema_still_loads_and_keeps_its_own_version(tmp_path: Path) -> None:
    # A leaderboard that could not read the results it published last release
    # would make every schema bump a silent republication.
    # Arrange
    document = _fixture_document()
    document["schema_version"] = "1.0"
    path = tmp_path / "earlier.json"
    path.write_text(canonical_json(document), encoding="utf-8")

    # Act
    loaded = load_result(path)

    # Assert
    assert loaded.manifest.schema_version == "1.0"
    assert loaded.to_dict()["schema_version"] == "1.0"
