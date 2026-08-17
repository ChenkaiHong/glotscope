"""The provenance manifest (PRD §9, G4).

Every result carries one. This is what makes the leaderboard citable and it is
the differentiator §2.2 identifies: no competing tool pins tokenizer revisions or
publishes artifact hashes, and for a tool whose entire output is numbers other
people will cite, that is a real and cheap advantage.

**Nothing here may contain a timestamp, a hostname, a path, or any other value
that varies between runs.** G4 promises that re-running a manifest reproduces the
numbers bit-identically, and ``glotscope verify`` asserts it in CI. A wall-clock
field would make that assertion permanently red. Corpus and model *versions* are
upstream release identifiers, not dates glotscope generates.
"""

from __future__ import annotations

import json
import platform as platform_module
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

from glotscope.enums import (
    Algorithm,
    Backend,
    Capability,
    Normalization,
    RenyiNormalizer,
    Segmenter,
)

__all__ = [
    "SCHEMA_VERSION",
    "UNKNOWN_LICENSE",
    "CorpusManifest",
    "EnvironmentManifest",
    "Manifest",
    "ParameterManifest",
    "TokenizerManifest",
    "WarningLog",
    "WeightsManifest",
    "canonical_json",
    "environment",
]

UNKNOWN_LICENSE = "UNKNOWN"
"""Recorded where no SPDX identifier is knowable from the artifact itself.

A local ``tokenizer.json`` or ``safetensors`` file carries no license metadata
and the repository it was exported from is not recoverable from the bytes.
``--license-filter=commercial`` reads these fields, so an unverifiable guess here
would be worse than an explicit unknown."""

SCHEMA_VERSION = "1.2"
"""Bumped whenever any serialized key or enum value changes.

1.2 added ``p_continued`` to the per-language block: §7.1 defines it beside
fertility and the two answer different questions, so computing it and
publishing only the other left a defined quantity unreadable."""


@dataclass(frozen=True, slots=True)
class TokenizerManifest:
    """Identity and provenance of the tokenizer under test (PRD §9)."""

    id: str
    revision: str
    """Commit SHA, never a branch name. Mirror-sourced rows are pinned by
    revision precisely because a leaderboard that silently follows a moving
    reference is a legitimate line of attack."""

    tokenizer_json_sha256: str
    vocab_size_tokenizer: int
    vocab_size_config: int | None
    """Often differs from :attr:`vocab_size_tokenizer` — Qwen3 reports 151669 and
    151936. Both are recorded because Tier 2's reference-set chain uses the gap.

    ``None`` when the source cannot supply it: a bare ``tokenizer.json`` has no
    ``config.json`` beside it. Serialized as an explicit ``null`` rather than
    back-filled from :attr:`vocab_size_tokenizer`, because equal values are a
    *claim* that the embedding matrix has no padding rows — and that claim is
    exactly what §7.9's reference-set chain reads."""

    embedding_rows: int | None
    """May exceed ``|V|``. Padding rows are a reference-set source (§7.9).
    ``None`` until weights are read; see :attr:`vocab_size_config`."""

    algorithm: Algorithm
    source: str
    source_is_mirror: bool
    """``True`` for third-party re-uploads such as the ungated Llama mirrors.
    Mirror-sourced rows must be **visibly labelled** in the leaderboard (§11)."""

    license_spdx: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "revision": self.revision,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "vocab_size_tokenizer": self.vocab_size_tokenizer,
            "vocab_size_config": self.vocab_size_config,
            "embedding_rows": self.embedding_rows,
            "algorithm": self.algorithm.value,
            "source": self.source,
            "source_is_mirror": self.source_is_mirror,
            "license_spdx": self.license_spdx,
        }

    @classmethod
    def from_dict(cls, block: Mapping[str, Any]) -> TokenizerManifest:
        return cls(
            id=block["id"],
            revision=block["revision"],
            tokenizer_json_sha256=block["tokenizer_json_sha256"],
            vocab_size_tokenizer=block["vocab_size_tokenizer"],
            vocab_size_config=block["vocab_size_config"],
            embedding_rows=block["embedding_rows"],
            algorithm=Algorithm(block["algorithm"]),
            source=block["source"],
            source_is_mirror=block["source_is_mirror"],
            license_spdx=block["license_spdx"],
        )


@dataclass(frozen=True, slots=True)
class WeightsManifest:
    """Provenance of the embedding tensors, when Tier 2 ran (PRD §9, §14.2)."""

    shard_sha256: str
    dtype: str
    """Recorded because a quantized ``E_in`` destroys the L2-norm indicator.
    Community mirrors republish 4-bit, GGUF and merged variants under
    near-identical names, so the dtype is the audit trail for a hard refusal."""

    tied_embeddings: bool
    """Selects which indicators can run at all. Tied checkpoints have only the
    cosine indicator and cannot be fully automated."""

    license_spdx: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_sha256": self.shard_sha256,
            "dtype": self.dtype,
            "tied_embeddings": self.tied_embeddings,
            "license_spdx": self.license_spdx,
        }

    @classmethod
    def from_dict(cls, block: Mapping[str, Any]) -> WeightsManifest:
        return cls(
            shard_sha256=block["shard_sha256"],
            dtype=block["dtype"],
            tied_embeddings=block["tied_embeddings"],
            license_spdx=block["license_spdx"],
        )


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Corpus identity, capabilities and license (PRD §9, §10.4)."""

    id: str
    version: str
    split: str
    languages: tuple[str, ...]
    sha256: str
    capabilities: frozenset[Capability]
    license: str
    """SPDX where available. ``--license-filter=commercial`` excludes
    non-commercial resources; UD requires a per-treebank audit because an unknown
    subset is CC BY-NC-SA 4.0."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "split": self.split,
            "languages": list(self.languages),
            "sha256": self.sha256,
            "capabilities": sorted(c.value for c in self.capabilities),
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, block: Mapping[str, Any]) -> CorpusManifest:
        return cls(
            id=block["id"],
            version=block["version"],
            split=block["split"],
            languages=tuple(block["languages"]),
            sha256=block["sha256"],
            capabilities=frozenset(Capability(c) for c in block["capabilities"]),
            license=block["license"],
        )


@dataclass(frozen=True, slots=True)
class ParameterManifest:
    """Every contested parameter choice, recorded (PRD §9, §7).

    Each field here corresponds to a §7 gotcha. If a parameter is absent from this
    struct, either the metric has no comparability hazard or the struct is
    incomplete — there is no third case.
    """

    leading_space: bool
    normalization: Normalization
    add_special_tokens: bool
    unk_exclusion_threshold: float = 0.10
    segmenter: Segmenter | None = None
    segmenter_model_version: str | None = None
    renyi_alpha: float | None = None
    renyi_normalizer: RenyiNormalizer | None = None
    top_pct: float | None = None
    candidates_pre_exclusion: int | None = None
    candidates_post_exclusion: int | None = None
    """``top_pct`` is applied *after* Stage-1 exclusion, and both counts are
    recorded because that ordering is what makes the denominator reproducible
    (§7.9)."""

    first_pc_removed: bool = False
    """Off by default (D9): the source paper's own Table 2 shows no consistent
    improvement from centering or PCA-1 removal across seven models."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "leading_space": self.leading_space,
            "normalization": self.normalization.value,
            "add_special_tokens": self.add_special_tokens,
            "unk_exclusion_threshold": self.unk_exclusion_threshold,
            "first_pc_removed": self.first_pc_removed,
        }
        if self.segmenter is not None:
            out["segmenter"] = self.segmenter.value
            out["segmenter_model_version"] = self.segmenter_model_version
        if self.renyi_alpha is not None:
            out["renyi_alpha"] = self.renyi_alpha
            out["renyi_normalizer"] = self.renyi_normalizer.value if self.renyi_normalizer else None
        if self.top_pct is not None:
            out["top_pct"] = self.top_pct
            out["candidates_pre_exclusion"] = self.candidates_pre_exclusion
            out["candidates_post_exclusion"] = self.candidates_post_exclusion
        return out

    @classmethod
    def from_dict(cls, block: Mapping[str, Any]) -> ParameterManifest:
        """Invert :meth:`to_dict`, where an absent key means the parameter is unset.

        ``to_dict`` omits whole groups rather than writing nulls — a document
        that never ran Tier 2 has no ``top_pct`` key at all — so absence and
        ``None`` are the same statement and ``get`` is the correct reader.
        """
        normalizer = block.get("renyi_normalizer")
        return cls(
            leading_space=block["leading_space"],
            normalization=Normalization(block["normalization"]),
            add_special_tokens=block["add_special_tokens"],
            unk_exclusion_threshold=block["unk_exclusion_threshold"],
            segmenter=Segmenter(block["segmenter"]) if block.get("segmenter") else None,
            segmenter_model_version=block.get("segmenter_model_version"),
            renyi_alpha=block.get("renyi_alpha"),
            renyi_normalizer=RenyiNormalizer(normalizer) if normalizer else None,
            top_pct=block.get("top_pct"),
            candidates_pre_exclusion=block.get("candidates_pre_exclusion"),
            candidates_post_exclusion=block.get("candidates_post_exclusion"),
            first_pc_removed=block["first_pc_removed"],
        )


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    """Library and platform versions (PRD §9).

    Deliberately coarse. ``platform`` is a normalized family string such as
    ``"linux-x86_64"``, not ``platform.platform()`` — the latter embeds kernel
    build numbers and would differ between two machines that produce identical
    numbers.
    """

    python: str
    tokenizers: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "tokenizers": self.tokenizers,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, block: Mapping[str, Any]) -> EnvironmentManifest:
        return cls(
            python=block["python"],
            tokenizers=block["tokenizers"],
            platform=block["platform"],
        )


def environment() -> EnvironmentManifest:
    """Capture the current environment (PRD §9).

    ``platform`` is assembled from :func:`platform.system` and
    :func:`platform.machine` rather than taken from :func:`platform.platform`,
    which embeds kernel build numbers. Two machines that produce identical
    numbers must produce identical manifests, or G4's bit-identical assertion is
    permanently red for a reason that has nothing to do with the numbers.
    """
    return EnvironmentManifest(
        python=platform_module.python_version(),
        tokenizers=version("tokenizers"),
        platform=f"{platform_module.system()}-{platform_module.machine()}".lower(),
    )


@dataclass(frozen=True, slots=True)
class Manifest:
    """The complete provenance record for one result (PRD §9)."""

    tokenizer: TokenizerManifest
    parameters: ParameterManifest
    environment: EnvironmentManifest
    backend: Backend
    glotscope_version: str
    weights: WeightsManifest | None = None
    corpus: CorpusManifest | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the §9 nesting, omitting absent tiers."""
        inner: dict[str, Any] = {"tokenizer": self.tokenizer.to_dict()}
        if self.weights is not None:
            inner["weights"] = self.weights.to_dict()
        if self.corpus is not None:
            inner["corpus"] = self.corpus.to_dict()
        inner["parameters"] = self.parameters.to_dict()
        inner["environment"] = self.environment.to_dict()
        return {
            "schema_version": self.schema_version,
            "glotscope_version": self.glotscope_version,
            "backend": self.backend.value,
            "manifest": inner,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Manifest:
        """Rebuild a manifest from a §9 document.

        The manifest block is the *lossless* half of the document: every field
        here round-trips. The tier blocks do not — §9 publishes
        ``unreachable_count`` and not the ids behind it — which is why loading a
        document yields :class:`~glotscope.document.LoadedResult` rather than a
        :class:`~glotscope.report.Report`.

        ``schema_version`` is read from the document rather than stamped with
        :data:`SCHEMA_VERSION`. Re-emitting a document under a newer schema
        version than it was written with would silently relabel someone else's
        published result.
        """
        inner = document["manifest"]
        weights = inner.get("weights")
        corpus = inner.get("corpus")
        return cls(
            tokenizer=TokenizerManifest.from_dict(inner["tokenizer"]),
            parameters=ParameterManifest.from_dict(inner["parameters"]),
            environment=EnvironmentManifest.from_dict(inner["environment"]),
            backend=Backend(document["backend"]),
            glotscope_version=document["glotscope_version"],
            weights=WeightsManifest.from_dict(weights) if weights is not None else None,
            corpus=CorpusManifest.from_dict(corpus) if corpus is not None else None,
            schema_version=document["schema_version"],
        )


def canonical_json(document: Mapping[str, Any]) -> str:
    """Serialize deterministically, so G4's bit-identical promise is testable.

    Three choices, all load-bearing:

    * ``sort_keys=True`` — dict iteration order must not leak into the artifact.
    * a fixed separator pair — no platform-dependent whitespace.
    * ``ensure_ascii=False`` — token strings in Tier 0 and Tier 2 output are
      genuinely non-ASCII, and escaping them would make the file unreadable for
      exactly the scripts the tool exists to examine.

    Float repr is left to Python's shortest-round-trip ``repr``, which is stable
    across CPython 3.10-3.13 on all supported platforms. Any metric that needs a
    tighter guarantee should round explicitly before serialization rather than
    relying on the encoder.
    """
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


@dataclass(frozen=True, slots=True)
class WarningLog:
    """Accumulator for the ``warnings`` array (PRD §9).

    The array is load-bearing, not decorative: any parameter choice known to be
    contested emits one. Kept immutable — :meth:`with_warning` returns a new log
    rather than appending in place.
    """

    entries: tuple[str, ...] = field(default_factory=tuple)

    def with_warning(self, message: str) -> WarningLog:
        return WarningLog(entries=(*self.entries, message))

    def extended(self, messages: Iterable[str]) -> WarningLog:
        return WarningLog(entries=(*self.entries, *messages))
