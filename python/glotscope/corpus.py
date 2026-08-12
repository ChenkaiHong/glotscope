"""Corpora, declared by capability rather than identity (PRD §6, D5, §10).

The single decision this module exists to enforce: **Tier 1 gates on what a
corpus can support, not on which corpus it is.** Requesting parity on a
monolingual corpus raises :class:`~glotscope.errors.CapabilityError` rather than
silently computing a meaningless number, and §6 identifies that as preventing the
most common category of wrong result in this space.

Ships no corpora (D12). Download recipes, SHA-256 checksums, and SPDX license
fields are release-pinned. The UD 2.18 per-treebank audit is generated from the
official archive and stored in ``data/ud-license-audit.json``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from glotscope.enums import Capability
from glotscope.errors import CapabilityError, CorpusIntegrityError, LicenseError
from glotscope.manifest import CorpusManifest

__all__ = [
    "FLORES_PLUS_VERSION",
    "REGISTRY",
    "UD_VERSION",
    "Corpus",
    "CorpusSpec",
    "LoadedCorpus",
]

FLORES_PLUS_VERSION = "2024.08"
"""Pinned FLORES+ release. An upstream identifier, never a date glotscope
generates — §9 forbids wall-clock values in a manifest."""

UD_VERSION = "2.18"
"""The release the committed per-treebank license audit covers. Using another
release means regenerating that audit first."""

_LICENSE_FILTERS = ("commercial",)
"""Known values for ``--license-filter`` (§10.4)."""


@dataclass(frozen=True, slots=True)
class LoadedCorpus:
    """A verified corpus and its text.

    Separate from :class:`Corpus` because loading *produces* the digest: the
    corpus that was asked for and the corpus that was read are different
    objects, and conflating them would let a manifest change under its own
    result.
    """

    corpus: Corpus
    lines: Mapping[str, tuple[str, ...]]
    """Language to documents, in file order. Read-only."""


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """Static description of a corpus resource (PRD §10.1)."""

    id: str
    capabilities: frozenset[Capability]
    license: str
    is_commercial_ok: bool
    """``False`` for research-only resources. ``--license-filter=commercial``
    excludes these from a run."""

    gated: bool = False
    """FLORES+ is gated on Hugging Face, so anonymous CI cannot fetch it. Plan for
    ``HF_TOKEN`` or a vendored hash-checked subset."""

    note: str = ""

    recipe: str = ""
    """How to obtain the resource. Non-empty for every registry entry, and tested
    for: the library ships no corpora (D12), so a row without a recipe is a dead
    end rather than a citation."""


REGISTRY: dict[str, CorpusSpec] = {
    "flores_plus": CorpusSpec(
        id="flores_plus",
        capabilities=frozenset({Capability.PARALLEL}),
        license="CC-BY-SA-4.0",
        is_commercial_ok=True,
        gated=True,
        note="229 varieties; dev 997 / devtest 1012. Primary parallel corpus.",
        recipe="Hugging Face dataset 'openlanguagedata/flores_plus', pinned by "
        "revision. Gated: needs HF_TOKEN or accepted terms. Write one file per "
        "language as <root>/flores_plus/<version>/<split>/<lang>.txt.",
    ),
    "europarl": CorpusSpec(
        id="europarl",
        capabilities=frozenset({Capability.PARALLEL}),
        license="Research-only (Koehn 2005)",
        is_commercial_ok=False,
        note="21 European languages, ~200k lines. Used where FLORES is too small "
        "for stable statistics.",
        recipe="Release v7 from statmt.org/europarl. Research use only, so it is "
        "excluded by --license-filter=commercial.",
    ),
    "fineweb2": CorpusSpec(
        id="fineweb2",
        capabilities=frozenset(),
        license="ODC-By-1.0",
        is_commercial_ok=True,
        note="1000+ languages, monolingual. Commercially safe. The only corpus "
        "large enough for the secondary corpus-attribution check in §14.3, which "
        "needs 1e8-1e9 tokens per language plus saturation curves.",
        recipe="Hugging Face dataset 'HuggingFaceFW/fineweb-2', pinned by "
        "revision. Ungated. Sample per language and record the sample's digest, "
        "since the full release is far larger than any run needs.",
    ),
    "universal_dependencies": CorpusSpec(
        id="universal_dependencies",
        capabilities=frozenset({Capability.WORD_SEGMENTATION, Capability.MORPH_GOLD}),
        license="Per-treebank — see data/ud-license-audit.json",
        is_commercial_ok=False,
        note="The UD 2.18 audit covers all 353 treebanks: 268 are commercial-compatible, "
        "31 are noncommercial, and 54 require manual review. Record the exact treebank "
        "because Korean "
        "treebanks, for example, use different segmentation conventions.",
        recipe="UD 2.18 release archive from the LINDAT repository. Check each "
        "treebank against data/ud-license-audit.json before use: 31 are "
        "noncommercial and 54 need manual review.",
    ),
    "morphynet": CorpusSpec(
        id="morphynet",
        capabilities=frozenset({Capability.MORPH_GOLD}),
        license="CC-BY-SA",
        is_commercial_ok=True,
        note="15 languages. Required for the full-alignment measure in §7.7(c).",
        recipe="Derivational and inflectional TSVs from the MorphyNet GitHub "
        "repository, pinned by commit.",
    ),
    "strr_wordlists": CorpusSpec(
        id="strr_wordlists",
        capabilities=frozenset({Capability.WORDLIST}),
        license="See upstream repo",
        is_commercial_ok=False,
        note="7 languages x 1000 words. Upstream provenance is unclear; cite the "
        "repo, not the source site. The lists are English-frequency lists "
        "translated, so they are not per-language frequency lists.",
        recipe="Word lists from the upstream STRR repository, pinned by commit. "
        "Provenance beyond that repository is unclear, so cite the repository "
        "itself rather than the site it credits.",
    ),
}
"""Known corpora (PRD §10.1). Capabilities here are what gate Tier 1."""


@dataclass(frozen=True, slots=True)
class Corpus:
    """A resolved corpus: a spec, a language set, and a pinned version."""

    spec: CorpusSpec
    languages: tuple[str, ...]
    version: str
    split: str
    sha256: str

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.spec.capabilities

    def has(self, capability: Capability) -> bool:
        return capability in self.spec.capabilities

    def require(self, capability: Capability, metric: str) -> None:
        """Assert a capability, or refuse (PRD §6, D5).

        Raises:
            CapabilityError: naming the metric, the missing capability, and what
                the corpus actually declares.
        """
        if not self.has(capability):
            raise CapabilityError(
                metric=metric,
                required=capability.value,
                corpus_id=self.spec.id,
                available=(c.value for c in self.spec.capabilities),
            )

    @classmethod
    def flores_plus(
        cls,
        languages: Sequence[str],
        *,
        split: str = "devtest",
        version: str = FLORES_PLUS_VERSION,
        sha256: str = "",
    ) -> Corpus:
        """The primary parallel corpus (PRD §10.1).

        Note that FLORES+ has **no gold word segmentation**, so
        :attr:`~glotscope.enums.Segmenter.UD_GOLD` is illegal against it and
        raises. That is §7.1 rule 1, and it is the distinction between "UD
        segmentation" as an annotation and as a trained model.

        ``sha256`` is empty on a first run and filled by :meth:`load`. Pass the
        recorded digest to assert that a later run reads the same bytes.
        """
        return cls.from_registry(
            "flores_plus", languages, split=split, version=version, sha256=sha256
        )

    @classmethod
    def fineweb2(
        cls,
        languages: Sequence[str],
        *,
        version: str,
        split: str = "train",
        sha256: str = "",
    ) -> Corpus:
        """Monolingual corpus for compression and Renyi (PRD §10.1).

        Declares no capabilities, so requesting parity against it raises. That
        refusal is the worked example in §8.1.

        ``version`` is required and has no default. FineWeb2 is far larger than
        any run needs, so what gets evaluated is a *sample*, and no upstream
        release tag describes it. Inventing one would put an unverifiable string
        in a manifest field other people are meant to cite.
        """
        return cls.from_registry("fineweb2", languages, split=split, version=version, sha256=sha256)

    @classmethod
    def universal_dependencies(
        cls,
        treebanks: Sequence[str],
        *,
        version: str = UD_VERSION,
        split: str = "train",
        sha256: str = "",
    ) -> Corpus:
        """UD treebanks, which carry both gold segmentation and morphology.

        Takes treebank identifiers rather than language codes: UD Korean
        treebanks disagree among themselves — Kaist segments morphologically, GSD
        by eojeol — so "UD" is not a single convention even within gold data.
        """
        return cls.from_registry(
            "universal_dependencies", treebanks, split=split, version=version, sha256=sha256
        )

    def load(self, root: str | Path, *, license_filter: str | None = None) -> LoadedCorpus:
        """Read this corpus from a local download and verify it (§10.4, D12).

        glotscope ships no corpora; it reads what the recipe told the user to
        fetch, from ``<root>/<id>/<version>/<split>/<language>.txt``, one
        document per line.

        Args:
            root: directory holding downloaded corpora.
            license_filter: ``"commercial"`` excludes research-only resources.
                ``None`` runs without a filter.

        Returns:
            A :class:`LoadedCorpus` whose ``corpus`` carries the computed
            digest. The receiver is not modified — these objects are frozen, and
            a manifest that changed under its own result would be worthless.

        Raises:
            LicenseError: if the filter excludes this resource.
            CorpusIntegrityError: if a language file is missing, if the digest
                does not match one that was pinned, or if a corpus declaring
                itself parallel has unequal line counts across languages.
            ValueError: if ``license_filter`` is not a known filter.
        """
        self._check_license(license_filter)

        directory = Path(root) / self.spec.id / self.version / self.split
        lines: dict[str, tuple[str, ...]] = {}
        digests: list[str] = []
        for language in self.languages:
            path = directory / f"{language}.txt"
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise CorpusIntegrityError(
                    self.spec.id,
                    f"no file for {language!r}: expected {path}. The library ships "
                    f"no corpora — see the download recipe on its registry entry",
                ) from exc
            lines[language] = tuple(raw.decode("utf-8").splitlines())
            digests.append(f"{language}:{hashlib.sha256(raw).hexdigest()}")

        digest = hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest()
        if self.sha256 and self.sha256 != digest:
            raise CorpusIntegrityError(
                self.spec.id,
                f"digest {digest} does not match the pinned {self.sha256}. The "
                f"bytes on disk are not the bytes this result claims to describe",
            )

        if Capability.PARALLEL in self.spec.capabilities:
            counts = {language: len(text) for language, text in lines.items()}
            if len(set(counts.values())) > 1:
                raise CorpusIntegrityError(
                    self.spec.id,
                    f"a parallel corpus with unequal line counts: {counts}. Parity "
                    f"is a ratio of means and only equals the ratio of totals when "
                    f"the counts match (§7.3, D7)",
                )

        return LoadedCorpus(
            corpus=replace(self, sha256=digest),
            lines=MappingProxyType(lines),
        )

    def to_manifest(self) -> CorpusManifest:
        """The §9 corpus block for this resolved corpus.

        Assembled here rather than by the caller so that ``sha256`` is whatever
        :meth:`load` computed: a manifest naming a corpus it did not read would
        pass every check while describing different bytes.
        """
        return CorpusManifest(
            id=self.spec.id,
            version=self.version,
            split=self.split,
            languages=self.languages,
            sha256=self.sha256,
            capabilities=self.spec.capabilities,
            license=self.spec.license,
        )

    def _check_license(self, license_filter: str | None) -> None:
        if license_filter is None:
            return
        if license_filter not in _LICENSE_FILTERS:
            raise ValueError(
                f"license_filter must be one of {sorted(_LICENSE_FILTERS)} or None, "
                f"got {license_filter!r}"
            )
        if not self.spec.is_commercial_ok:
            raise LicenseError(self.spec.id, self.spec.license, license_filter)

    @classmethod
    def from_registry(
        cls,
        corpus_id: str,
        languages: Iterable[str],
        *,
        split: str,
        version: str,
        sha256: str,
    ) -> Corpus:
        """Build a corpus from a registry entry, validating the id."""
        if corpus_id not in REGISTRY:
            raise KeyError(
                f"unknown corpus {corpus_id!r}; known: {sorted(REGISTRY)}. "
                f"Add a CorpusSpec with explicit capabilities and an SPDX license "
                f"rather than passing an unregistered corpus."
            )
        return cls(
            spec=REGISTRY[corpus_id],
            languages=tuple(languages),
            version=version,
            split=split,
            sha256=sha256,
        )
