"""Running every row of the leaderboard (PRD §16.1, §11).

The distinction this module exists to hold: **a row that cannot be reached is a
skip, and anything else is a failure.** §11 says gated resources skip with a
message and never fail the run — half of §11's roster is gated or manually
approved, so a board that died on the first unreachable repository would never
publish at all. But that rule is about *artifacts*, not about bugs: a board that
swallowed a ``TypeError`` would publish a shorter table and call it a gated skip,
and nobody reading the output could tell the difference.

So the catch list is typed and short, and everything else propagates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glotscope import __version__, backend
from glotscope.corpus import Corpus, LoadedCorpus
from glotscope.embeddings import Embeddings
from glotscope.errors import (
    CapabilityError,
    NoReferenceSetError,
    TokenizerLoadError,
    UnsupportedCheckpointError,
)
from glotscope.leaderboard.check import ALL_TIERS
from glotscope.leaderboard.config import LeaderboardConfig, RosterEntry
from glotscope.loading import load_embeddings, load_tokenizer
from glotscope.manifest import Manifest, ParameterManifest, environment
from glotscope.report import Report
from glotscope.reporting import attach_tier2, build_report
from glotscope.tokenizer import Tokenizer

__all__ = ["LeaderboardDocument", "LeaderboardRow", "run_leaderboard"]

TOKENIZER_ONLY = "n/a (tokenizer-only)"
"""What the Tier 2 column says for a row with no weights.

§16.1 is explicit that roughly half the core roster is tokenizer-only and that
the column must not be left visually empty: a blank cell reads as a measurement
that failed, and the launch positioning — "reads both your corpus and your
checkpoint" — is undercut by a table that looks mostly broken.
"""

_UNREACHABLE: tuple[type[Exception], ...] = (
    TokenizerLoadError,
    UnsupportedCheckpointError,
    NoReferenceSetError,
    CapabilityError,
    ModuleNotFoundError,
    OSError,
)
"""Failures that mean *this artifact is not available here*.

Each is a typed refusal or a missing file — the shapes a gated repository, an
absent extra and a mistyped path arrive in. Nothing broader: a bug must not be
able to enter this list by accident, which is what catching ``Exception`` would
allow.
"""


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    """One roster entry, run or skipped."""

    entry: RosterEntry
    result: Mapping[str, Any] | None = None
    skipped: str | None = None

    @property
    def tier2_status(self) -> str:
        """What the Tier 2 column shows for this row."""
        if self.result is not None and "tier2" in self.result:
            return "measured"
        return TOKENIZER_ONLY

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.entry.id,
            "label": self.entry.display,
            "is_mirror": self.entry.is_mirror,
            "note": self.entry.note,
            "skipped": self.skipped,
            "tier2": self.tier2_status,
            "result": self.result,
        }


@dataclass(frozen=True, slots=True)
class LeaderboardDocument:
    """The whole board: what it was computed under, and every row's manifest."""

    config: LeaderboardConfig
    rows: tuple[LeaderboardRow, ...]
    corpus_sha256: str

    @property
    def published(self) -> int:
        return sum(1 for row in self.rows if row.skipped is None)

    @property
    def skipped(self) -> int:
        return sum(1 for row in self.rows if row.skipped is not None)

    def to_dict(self) -> dict[str, Any]:
        parameters = self.config.parameters
        return {
            "glotscope_version": __version__,
            "backend": backend().value,
            "environment": environment().to_dict(),
            "corpus": {
                "id": self.config.corpus.id,
                "version": self.config.corpus.version,
                "split": self.config.corpus.split,
                "languages": list(self.config.corpus.languages),
                "sha256": self.corpus_sha256,
            },
            "parameters": {
                "leading_space": parameters.leading_space,
                "normalization": parameters.normalization.value,
                "add_special_tokens": parameters.add_special_tokens,
                "segmenter": parameters.segmenter.value if parameters.segmenter else None,
                "parity_reference": parameters.parity_reference,
                "gini": parameters.gini,
                "renyi_alpha": parameters.renyi_alpha,
                "renyi_normalizer": parameters.renyi_normalizer.value,
            },
            "published": self.published,
            "skipped": self.skipped,
            "rows": [row.to_dict() for row in self.rows],
        }


def _embeddings(tokenizer: Tokenizer, entry: RosterEntry) -> Embeddings | None:
    """The row's weights, read — or ``None`` for a tokenizer-only row."""
    if entry.weights is None:
        return None
    return load_embeddings(
        entry.weights,
        vocab_size=tokenizer.manifest.vocab_size_tokenizer,
        revision=entry.weights_revision,
    )


def _tier0_only(tokenizer: Tokenizer, config: LeaderboardConfig) -> Report:
    """A document for a run that read no corpus.

    The nightly re-check runs where FLORES+ is gated, so it can recompute Tier 0
    and nothing else. The manifest's ``corpus`` block is ``None`` rather than
    copied from the published board: this run did not read a corpus, and a
    document claiming one it never opened is the kind of provenance §9 exists to
    prevent.
    """
    parameters = config.parameters
    return Report(
        tier0=tokenizer.lint(),
        manifest=Manifest(
            tokenizer=tokenizer.manifest,
            parameters=ParameterManifest(
                leading_space=parameters.leading_space,
                normalization=parameters.normalization,
                add_special_tokens=parameters.add_special_tokens,
                segmenter=parameters.segmenter,
            ),
            environment=environment(),
            backend=backend(),
            glotscope_version=__version__,
            corpus=None,
        ),
        warnings=tokenizer.warnings,
    )


def _run_row(
    entry: RosterEntry,
    loaded: LoadedCorpus | None,
    config: LeaderboardConfig,
    *,
    top_pct: float,
) -> LeaderboardRow:
    try:
        tokenizer = load_tokenizer(entry.id, entry.revision)
        if loaded is None:
            return LeaderboardRow(entry=entry, result=_tier0_only(tokenizer, config).to_dict())
        report: Report = build_report(
            tokenizer,
            loaded,
            leading_space=config.parameters.leading_space,
            normalization=config.parameters.normalization,
            add_special_tokens=config.parameters.add_special_tokens,
            segmenter=config.parameters.segmenter,
            parity_reference=config.parameters.parity_reference,
            gini=config.parameters.gini,
            renyi_alpha=config.parameters.renyi_alpha,
            renyi_normalizer=config.parameters.renyi_normalizer,
            nominal_vocab_size=config.parameters.nominal_vocab_size,
        )
        embeddings = _embeddings(tokenizer, entry)
        if embeddings is not None:
            # The same assembly `detect` uses, so the row carries the weights
            # manifest and §7.9's parameters beside its tier2 block — not the
            # Tier 1 manifest with a tier2 block bolted on, which is what it
            # carried before and what verify could not read.
            report = attach_tier2(report, tokenizer, embeddings, top_pct=top_pct)
    except _UNREACHABLE as exc:
        # Skipped, with the reason the row is missing. §11 requires the run to
        # continue; it does not permit the output to be silent about the gap.
        return LeaderboardRow(entry=entry, skipped=f"{type(exc).__name__}: {exc}")

    return LeaderboardRow(entry=entry, result=report.to_dict())


def run_leaderboard(
    config: LeaderboardConfig,
    *,
    corpus_root: str | Path,
    top_pct: float = 2.0,
    tiers: Sequence[str] = ALL_TIERS,
) -> LeaderboardDocument:
    """Run every roster row over one corpus read once.

    The corpus is loaded before the loop and shared. That is a cost decision at
    the real shape — 221 varieties by 1012 documents — where a per-row read would
    make an eighteen-model board eighteen corpus reads. It is also a correctness
    one: every row must be measured over the same bytes, and a re-read is a
    second chance for them to differ.

    A corpus that cannot load raises rather than skipping. One unreachable row is
    one row; an unreachable corpus is every number on the board, and skipping it
    would publish an empty table as a success.

    ``tiers`` names what to compute. Without ``tier1`` no corpus is read at all
    and every row is Tier 0 — which is what lets the nightly re-check run on an
    anonymous runner, where FLORES+ is gated and there is no corpus to read.
    """
    loaded: LoadedCorpus | None = None
    if "tier1" in tiers:
        corpus = Corpus.resolve(
            config.corpus.id,
            list(config.corpus.languages),
            split=config.corpus.split,
            version=config.corpus.version,
        )
        loaded = corpus.load(corpus_root, license_filter=config.corpus.license_filter)

    rows: Sequence[LeaderboardRow] = [
        _run_row(entry, loaded, config, top_pct=top_pct) for entry in config.roster
    ]
    return LeaderboardDocument(
        config=config,
        rows=tuple(rows),
        corpus_sha256=loaded.corpus.sha256 if loaded is not None else "",
    )
