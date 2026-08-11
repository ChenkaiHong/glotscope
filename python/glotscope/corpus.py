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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from glotscope.enums import Capability
from glotscope.errors import CapabilityError

__all__ = ["REGISTRY", "Corpus", "CorpusSpec"]


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


REGISTRY: dict[str, CorpusSpec] = {
    "flores_plus": CorpusSpec(
        id="flores_plus",
        capabilities=frozenset({Capability.PARALLEL}),
        license="CC-BY-SA-4.0",
        is_commercial_ok=True,
        gated=True,
        note="229 varieties; dev 997 / devtest 1012. Primary parallel corpus.",
    ),
    "europarl": CorpusSpec(
        id="europarl",
        capabilities=frozenset({Capability.PARALLEL}),
        license="Research-only (Koehn 2005)",
        is_commercial_ok=False,
        note="21 European languages, ~200k lines. Used where FLORES is too small "
        "for stable statistics.",
    ),
    "fineweb2": CorpusSpec(
        id="fineweb2",
        capabilities=frozenset(),
        license="ODC-By-1.0",
        is_commercial_ok=True,
        note="1000+ languages, monolingual. Commercially safe. The only corpus "
        "large enough for the secondary corpus-attribution check in §14.3, which "
        "needs 1e8-1e9 tokens per language plus saturation curves.",
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
    ),
    "morphynet": CorpusSpec(
        id="morphynet",
        capabilities=frozenset({Capability.MORPH_GOLD}),
        license="CC-BY-SA",
        is_commercial_ok=True,
        note="15 languages. Required for the full-alignment measure in §7.7(c).",
    ),
    "strr_wordlists": CorpusSpec(
        id="strr_wordlists",
        capabilities=frozenset({Capability.WORDLIST}),
        license="See upstream repo",
        is_commercial_ok=False,
        note="7 languages x 1000 words. Upstream provenance is unclear; cite the "
        "repo, not the source site. The lists are English-frequency lists "
        "translated, so they are not per-language frequency lists.",
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
        version: str | None = None,
    ) -> Corpus:
        """The primary parallel corpus (PRD §10.1).

        Note that FLORES+ has **no gold word segmentation**, so
        :attr:`~glotscope.enums.Segmenter.UD_GOLD` is illegal against it and
        raises. That is §7.1 rule 1, and it is the distinction between "UD
        segmentation" as an annotation and as a trained model.
        """
        raise NotImplementedError

    @classmethod
    def fineweb2(cls, languages: Sequence[str], *, version: str | None = None) -> Corpus:
        """Monolingual corpus for compression and Renyi (PRD §10.1).

        Declares no capabilities, so requesting parity against it raises. That
        refusal is the worked example in §8.1.
        """
        raise NotImplementedError

    @classmethod
    def universal_dependencies(
        cls,
        treebanks: Sequence[str],
        *,
        version: str | None = None,
    ) -> Corpus:
        """UD treebanks, which carry both gold segmentation and morphology.

        Takes treebank identifiers rather than language codes: UD Korean
        treebanks disagree among themselves — Kaist segments morphologically, GSD
        by eojeol — so "UD" is not a single convention even within gold data.
        """
        raise NotImplementedError

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
