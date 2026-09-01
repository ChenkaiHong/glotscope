"""The §9 manifest round-trips (PRD §9, §15's M3 criterion, G4).

M3's exit criteria include "manifest round-trips", and it does — but until now
nothing asserted it. A property that holds by accident is one refactor away from
being false, and this one is load-bearing: `glotscope verify` reads a published
document back into a `Manifest` before it recomputes anything, so a field that
silently failed to survive the trip would make `verify` compare the wrong thing.

**What round-trips and what deliberately does not.** The manifest block is the
lossless half of the document. The tier blocks are not — §9 publishes
`unreachable_count` and not the ids behind it — which is why loading a document
yields a `LoadedResult` rather than a `Report`. That asymmetry is a design
decision, not an omission, and the test at the bottom pins it so it cannot be
"fixed" into a document that carries a quarter of a million ids per row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glotscope.enums import Algorithm, Backend, Normalization, Segmenter
from glotscope.manifest import (
    SCHEMA_VERSION,
    CorpusManifest,
    EnvironmentManifest,
    Manifest,
    ParameterManifest,
    TokenizerManifest,
    WeightsManifest,
    canonical_json,
    environment,
)

_DOCUMENT_KEYS = ("schema_version", "glotscope_version", "backend", "manifest")
"""What `Manifest.to_dict` emits: the manifest block plus the three top-level
fields that describe the document itself."""

_REPO = Path(__file__).resolve().parents[1]


def _manifest_half(document: dict[str, Any]) -> dict[str, Any]:
    return {key: document[key] for key in _DOCUMENT_KEYS}


def _assert_round_trips(document: dict[str, Any]) -> None:
    published = _manifest_half(document)
    rebuilt = Manifest.from_dict(document).to_dict()
    assert canonical_json(rebuilt) == canonical_json(published)


def _tokenizer(**overrides: Any) -> TokenizerManifest:
    fields: dict[str, Any] = {
        "id": "acme/model",
        "revision": "a" * 40,
        "tokenizer_json_sha256": "b" * 64,
        "vocab_size_tokenizer": 32000,
        "vocab_size_config": 32768,
        "embedding_rows": 32768,
        "algorithm": Algorithm.BYTE_LEVEL_BPE,
        "source": "hub",
        "source_is_mirror": False,
        "license_spdx": "Apache-2.0",
    }
    fields.update(overrides)
    return TokenizerManifest(**fields)


def _manifest(**overrides: Any) -> Manifest:
    fields: dict[str, Any] = {
        "tokenizer": _tokenizer(),
        "parameters": ParameterManifest(
            leading_space=True,
            normalization=Normalization.NFC,
            add_special_tokens=False,
        ),
        "environment": environment(),
        "backend": Backend.PYTHON,
        "glotscope_version": "0.1.0",
    }
    fields.update(overrides)
    return Manifest(**fields)


# -- real published documents ------------------------------------------------


def test_the_committed_verification_result_round_trips() -> None:
    """The document CI already regenerates on all twelve cells.

    Asserting against a *published* document rather than a constructed one is the
    point: a constructed manifest exercises the fields this file thought of, and
    the fixture exercises the ones an actual run emits.
    """
    path = _REPO / "verification" / "result.json"
    if not path.is_file():  # pragma: no cover - the fixture is outside the sdist
        pytest.skip("the G4 fixture lives in the repository, not in the distribution")

    _assert_round_trips(json.loads(path.read_text(encoding="utf-8")))


def test_every_published_leaderboard_row_round_trips() -> None:
    """Thirteen manifests from three different loaders — a local file, Hub
    repositories, and tiktoken encodings, whose `revision` is a library version
    rather than a commit."""
    path = _REPO / "results" / "leaderboard.json"
    if not path.is_file():  # pragma: no cover - results are outside the sdist
        pytest.skip("the published board lives in the repository, not in the distribution")

    board = json.loads(path.read_text(encoding="utf-8"))
    published = [row["result"] for row in board["rows"] if row.get("result")]
    assert published, "a board with no published rows would make this vacuous"
    for document in published:
        _assert_round_trips(document)


# -- the optional blocks -----------------------------------------------------


def test_a_manifest_with_no_corpus_round_trips() -> None:
    """The shape the nightly Tier 0 re-check produces. It reads no corpus, so
    the block is absent rather than empty — and an absent block that came back
    as `{}` would be a document claiming a corpus with no id."""
    manifest = _manifest(corpus=None)

    rebuilt = Manifest.from_dict(manifest.to_dict())

    assert rebuilt.corpus is None
    assert canonical_json(rebuilt.to_dict()) == canonical_json(manifest.to_dict())


def test_a_manifest_carrying_weights_round_trips() -> None:
    """Tier 2's half of the document, which `verify --weights` reads back before
    it recomputes anything."""
    manifest = _manifest(
        weights=WeightsManifest(
            shard_sha256="d" * 64,
            dtype="bfloat16",
            tied_embeddings=True,
            license_spdx="Apache-2.0",
        )
    )

    rebuilt = Manifest.from_dict(manifest.to_dict())

    assert rebuilt.weights is not None
    assert rebuilt.weights.dtype == "bfloat16"
    assert canonical_json(rebuilt.to_dict()) == canonical_json(manifest.to_dict())


def test_the_mirror_flag_survives_the_trip() -> None:
    """§11 requires mirror-sourced rows visibly labelled. A flag that did not
    round-trip would be lost the moment a board was read back and re-rendered."""
    manifest = _manifest(tokenizer=_tokenizer(source_is_mirror=True))

    assert Manifest.from_dict(manifest.to_dict()).tokenizer.source_is_mirror is True


def test_a_null_vocab_size_config_stays_null() -> None:
    """`None` is a claim about what nobody looked at; `vocab_size_tokenizer` is a
    claim that the embedding matrix has no padding rows — which is what §7.9's
    reference chain reads at link 2. Round-tripping one into the other would
    invent that claim."""
    manifest = _manifest(tokenizer=_tokenizer(vocab_size_config=None, embedding_rows=None))

    rebuilt = Manifest.from_dict(manifest.to_dict()).tokenizer

    assert rebuilt.vocab_size_config is None
    assert rebuilt.embedding_rows is None


def test_a_segmenter_and_its_model_version_survive() -> None:
    """§7.1 requires the segmenter model recorded. Losing it on a read would
    make two results computed under different models compare as equal."""
    manifest = _manifest(
        parameters=ParameterManifest(
            leading_space=True,
            normalization=Normalization.NFC,
            add_special_tokens=False,
            segmenter=Segmenter.UDPIPE,
            segmenter_model_versions={"eng_Latn": "udpipe 1.3, model english.udpipe sha256:ff"},
        )
    )

    rebuilt = Manifest.from_dict(manifest.to_dict()).parameters

    assert rebuilt.segmenter is Segmenter.UDPIPE
    assert rebuilt.segmenter_model_versions == {
        "eng_Latn": "udpipe 1.3, model english.udpipe sha256:ff"
    }


# -- what must not happen ----------------------------------------------------


def test_the_schema_version_is_read_and_not_restamped() -> None:
    """Re-emitting under the current schema would silently relabel someone
    else's published result as conforming to a version it was never written
    against — and `verify` compares `schema_version`, so the relabelling would
    make a stale document appear to pass."""
    manifest = _manifest()
    document = manifest.to_dict()
    document["schema_version"] = "0.9"

    rebuilt = Manifest.from_dict(document)

    assert rebuilt.schema_version == "0.9"
    assert rebuilt.to_dict()["schema_version"] == "0.9"
    assert SCHEMA_VERSION != "0.9", "the point of the test is that they differ"


def test_the_tier_blocks_are_deliberately_not_round_tripped() -> None:
    """A design decision, pinned so it is not "fixed" by accident.

    §9 publishes `unreachable_count` and not the ids behind it. Carrying them
    would put a quarter of a million integers in every leaderboard row, and the
    published count is what a reader checks. Loading therefore yields a
    `LoadedResult` — the document plus its parsed manifest — and never a
    `Report`, and this asserts the absence rather than leaving it to be
    rediscovered.
    """
    from glotscope import report as report_module

    assert not hasattr(report_module.Report, "from_json")
    assert not hasattr(report_module.Report, "from_dict")
    assert not hasattr(report_module.Tier0Report, "from_dict")


def test_a_corpus_block_round_trips_with_its_capabilities() -> None:
    """Capabilities are what gate Tier 1 (D5). A set that came back short would
    let a metric run that the original corpus refused."""
    from glotscope.enums import Capability

    manifest = _manifest(
        corpus=CorpusManifest(
            id="flores_plus",
            version="2024.08",
            split="devtest",
            languages=("eng_Latn", "hin_Deva"),
            sha256="e" * 64,
            license="CC-BY-SA-4.0",
            capabilities=frozenset({Capability.PARALLEL}),
        )
    )

    rebuilt = Manifest.from_dict(manifest.to_dict()).corpus

    assert rebuilt is not None
    assert rebuilt.capabilities == frozenset({Capability.PARALLEL})
    assert rebuilt.languages == ("eng_Latn", "hin_Deva")


def test_the_environment_round_trips_although_verify_does_not_compare_it() -> None:
    """Recorded because it varies, and still part of the document — a reader
    comparing two published results needs to see what differed."""
    manifest = _manifest(
        environment=EnvironmentManifest(
            python="3.13.12", platform="darwin-arm64", tokenizers="0.23.1"
        )
    )

    rebuilt = Manifest.from_dict(manifest.to_dict()).environment

    assert rebuilt.python == "3.13.12"
    assert rebuilt.platform == "darwin-arm64"
