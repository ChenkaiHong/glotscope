"""The leaderboard configuration (PRD §16.1, §11).

A leaderboard is a published artifact that other people cite, so this file is
where strictness pays for itself. Every refusal below is something that would
otherwise reach a table: an unpinned row, an unlabelled mirror, a key its author
believed was applied.

**Two formats, one meaning.** §8.2 spells the file ``leaderboard.yaml``, and YAML
is the right shape for a roster of eighteen models — it holds comments, and the
reason a row is a mirror belongs beside the row. But a YAML parser is not in the
core install, and G1's clean-install promise is measured on core. So ``.json`` is
read with the standard library and ``.yaml`` names the extra when it is missing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from glotscope.corpus import REGISTRY
from glotscope.enums import Capability, Normalization, RenyiNormalizer, Segmenter
from glotscope.segmenters import requires_model

__all__ = [
    "ConfigError",
    "CorpusPlan",
    "LeaderboardConfig",
    "ParameterPlan",
    "RosterEntry",
    "load_config",
]

_TIKTOKEN_PREFIX = "tiktoken:"

_COMMIT = re.compile(r"[0-9a-f]{40}")
"""A full commit SHA — what §11 pins every Hub row by, tokenizer and weights."""

_LOCAL_SUFFIXES = (".json", ".safetensors")


def _names_a_hub_repo(identifier: str) -> bool:
    """Whether ``identifier`` is a Hub repository rather than a place on disk."""
    path = Path(identifier)
    return not (
        path.is_absolute()
        or identifier.startswith(("~", "./", "../", ".\\", "..\\"))
        or path.suffix in _LOCAL_SUFFIXES
    )


class ConfigError(ValueError):
    """The configuration cannot be used as written.

    A subclass of :class:`ValueError` rather than of ``GlotscopeError``: this is
    a malformed input file, not a refusal to compute something. The CLI maps it
    to exit 1 alongside the typed refusals, because both are real answers about
    the input rather than missing features.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """One row of the leaderboard: what to load, and how to label it."""

    id: str
    revision: str | None = None
    is_mirror: bool = False
    note: str = ""
    label: str = ""
    weights: str | None = None
    weights_revision: str | None = None

    @property
    def is_encoding(self) -> bool:
        """Whether this row names a tiktoken encoding: a tokenizer, and no checkpoint."""
        return self.id.startswith(_TIKTOKEN_PREFIX)

    @property
    def is_hub(self) -> bool:
        """Whether this row names a Hub repository, which is what must be pinned.

        A tiktoken encoding is defined by the installed library and a local file
        by its own bytes; neither has a commit, so neither can carry one.
        """
        if self.is_encoding:
            return False
        return _names_a_hub_repo(self.id)

    @property
    def weights_is_hub(self) -> bool:
        """Whether this row reads its weights from a Hub repository."""
        return self.weights is not None and _names_a_hub_repo(self.weights)

    @property
    def display(self) -> str:
        return self.label or self.id


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    """Which corpus, at which version, over which languages."""

    id: str
    languages: tuple[str, ...]
    version: str
    split: str
    license_filter: str | None = None


@dataclass(frozen=True, slots=True)
class ParameterPlan:
    """Every contested §7 parameter the whole board is computed under.

    One set for the board rather than one per row: ``compare`` refuses to table
    results computed under different segmenters or alpha values, and a
    leaderboard is that comparison rendered. Allowing per-row parameters would
    build the incomparability the error exists to prevent.
    """

    leading_space: bool = True
    normalization: Normalization = Normalization.NFC
    add_special_tokens: bool = False
    segmenter: Segmenter | None = None
    parity_reference: str | None = None
    gini: bool = False
    renyi_alpha: float | None = None
    renyi_normalizer: RenyiNormalizer = RenyiNormalizer.OBSERVED
    nominal_vocab_size: int | None = None


@dataclass(frozen=True, slots=True)
class LeaderboardConfig:
    """A validated leaderboard run."""

    corpus: CorpusPlan
    parameters: ParameterPlan
    roster: tuple[RosterEntry, ...] = field(default_factory=tuple)
    version: int = 1


def _require_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, not {type(value).__name__}")
    return value


def _reject_unknown(block: Mapping[str, Any], known: Sequence[str], *, where: str) -> None:
    """Refuse a key nobody reads.

    Ignoring one silently is how a board gets published under parameters its
    author believed were applied — the failure is invisible in the output, which
    looks exactly like a board computed the way the file says.
    """
    unknown = sorted(set(block) - set(known))
    if unknown:
        raise ConfigError(
            f"{where}: unknown {'keys' if len(unknown) > 1 else 'key'} "
            f"{', '.join(repr(key) for key in unknown)}. Known: "
            f"{', '.join(sorted(known))}."
        )


def _enum(kind: type[Any], value: Any, *, where: str) -> Any:
    try:
        return kind(value)
    except ValueError as exc:
        members = ", ".join(sorted(member.value for member in kind))
        raise ConfigError(f"{where}: {value!r} is not one of {members}") from exc


def _corpus(block: Mapping[str, Any]) -> CorpusPlan:
    _reject_unknown(
        block, ("id", "languages", "version", "split", "license_filter"), where="corpus"
    )
    corpus_id = block.get("id")
    if not isinstance(corpus_id, str) or corpus_id not in REGISTRY:
        raise ConfigError(
            f"corpus.id: {corpus_id!r} is not in the registry. Known: "
            f"{', '.join(sorted(REGISTRY))}."
        )
    spec = REGISTRY[corpus_id]
    languages = block.get("languages")
    if not isinstance(languages, Sequence) or isinstance(languages, str) or not languages:
        raise ConfigError("corpus.languages must be a non-empty list of language codes")

    version = block.get("version", spec.default_version)
    if not isinstance(version, str) or not version:
        raise ConfigError(
            f"corpus.version: {corpus_id!r} pins no default version, so the run "
            f"must name the release it evaluated. A manifest field other people "
            f"cite cannot be filled from a guess."
        )
    return CorpusPlan(
        id=corpus_id,
        languages=tuple(str(code) for code in languages),
        version=version,
        split=str(block.get("split", spec.default_split)),
        license_filter=block.get("license_filter"),
    )


def _parameters(block: Mapping[str, Any], *, languages: Sequence[str]) -> ParameterPlan:
    known = (
        "leading_space",
        "normalization",
        "add_special_tokens",
        "segmenter",
        "parity_reference",
        "gini",
        "renyi_alpha",
        "renyi_normalizer",
        "nominal_vocab_size",
    )
    _reject_unknown(block, known, where="parameters")

    reference = block.get("parity_reference")
    if reference is not None and reference not in languages:
        raise ConfigError(
            f"parameters.parity_reference: {reference!r} is not in "
            f"corpus.languages. Parity is measured against a language the run "
            f"actually read, and a reference outside the set fails deep inside "
            f"the fold where it reads as a bug rather than a configuration "
            f"mistake."
        )

    segmenter = block.get("segmenter")
    alpha = block.get("renyi_alpha")
    return ParameterPlan(
        leading_space=bool(block.get("leading_space", True)),
        normalization=_enum(
            Normalization,
            block.get("normalization", Normalization.NFC.value),
            where="parameters.normalization",
        ),
        add_special_tokens=bool(block.get("add_special_tokens", False)),
        segmenter=(
            _enum(Segmenter, segmenter, where="parameters.segmenter")
            if segmenter is not None
            else None
        ),
        parity_reference=reference,
        gini=bool(block.get("gini", False)),
        renyi_alpha=float(alpha) if alpha is not None else None,
        renyi_normalizer=_enum(
            RenyiNormalizer,
            block.get("renyi_normalizer", RenyiNormalizer.OBSERVED.value),
            where="parameters.renyi_normalizer",
        ),
        nominal_vocab_size=block.get("nominal_vocab_size"),
    )


def _roster(rows: Any) -> tuple[RosterEntry, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, str) or not rows:
        raise ConfigError("roster must be a non-empty list of rows")

    entries: list[RosterEntry] = []
    for index, row in enumerate(rows):
        where = f"roster[{index}]"
        block = _require_mapping(row, where=where)
        _reject_unknown(
            block,
            ("id", "revision", "is_mirror", "note", "label", "weights", "weights_revision"),
            where=where,
        )
        identifier = block.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ConfigError(f"{where}.id must be a non-empty string")

        entry = RosterEntry(
            id=identifier,
            revision=block.get("revision"),
            is_mirror=bool(block.get("is_mirror", False)),
            note=str(block.get("note", "")),
            label=str(block.get("label", "")),
            weights=block.get("weights"),
            weights_revision=block.get("weights_revision"),
        )
        if entry.is_hub and not entry.revision:
            raise ConfigError(
                f"{where}: {identifier!r} names a Hub repository and carries no "
                f"revision. §11 pins every row by commit. An unpinned run still "
                f"records a resolved SHA — but one nobody chose, so §16.1's "
                f"nightly job would report an upstream branch advancing as a "
                f"number moving."
            )
        if not entry.is_hub and entry.revision:
            raise ConfigError(
                f"{where}: {identifier!r} has no commit to pin — a local file is "
                f"identified by its own bytes and an encoding by the installed "
                f"tiktoken — so a revision here would pin nothing."
            )
        _require_commits(entry, where=where)
        if entry.is_mirror and not entry.note:
            raise ConfigError(
                f"{where}: a mirror row needs a note. §11 requires "
                f"mirror-sourced rows to be visibly labelled, because a "
                f"leaderboard silently using unofficial re-uploads is a "
                f"legitimate line of attack — and a flag with nothing to show a "
                f"reader is that failure with a flag set."
            )
        entries.append(entry)
    return tuple(entries)


def _require_supported(plan: ParameterPlan, *, corpus_id: str) -> ParameterPlan:
    """Refuse a parameter the whole board could never compute.

    Each of these is a property of the corpus and the parameters together, never
    of one artifact, so reaching the run would fail every row the same way — and
    a row-level failure is a skip, which published an empty board as a success.
    """
    capabilities = REGISTRY[corpus_id].capabilities
    segmenter = plan.segmenter
    if segmenter is not None and requires_model(segmenter):
        raise ConfigError(
            f"parameters.segmenter: {segmenter.value} needs a pinned model file, "
            f"and the leaderboard config has no key to name one. Use a rule-based "
            f"segmenter here, or run `glotscope analyze --segmenter-model` per model."
        )
    if (
        segmenter is not None
        and segmenter.requires_gold_segmentation
        and Capability.WORD_SEGMENTATION not in capabilities
    ):
        raise ConfigError(
            f"parameters.segmenter: {segmenter.value} reads gold word boundaries, and "
            f"corpus {corpus_id!r} declares no {Capability.WORD_SEGMENTATION.value} "
            f"(§7.1 rule 1)."
        )
    wanted = (("gini", plan.gini), ("parity_reference", plan.parity_reference is not None))
    parallel_only = [name for name, requested in wanted if requested]
    if parallel_only and Capability.PARALLEL not in capabilities:
        raise ConfigError(
            f"parameters.{parallel_only[0]}: needs a {Capability.PARALLEL.value} "
            f"corpus, and {corpus_id!r} declares none. Gating is on capability (D5)."
        )
    return plan


def _require_commits(entry: RosterEntry, *, where: str) -> None:
    """Refuse a row pinned by anything but a commit, and unpinned Hub weights.

    A branch or tag names something that moves, so a board pinned by one is
    unpinned in all but name. Weights are the same artifact class as the
    tokenizer: a pinned tokenizer beside weights read from a mutable branch is
    half a pin, and a revision beside a local file pins nothing.
    """
    if entry.revision and not _COMMIT.fullmatch(entry.revision):
        raise ConfigError(
            f"{where}: revision {entry.revision!r} is not a commit. §11 pins every "
            f"row by a full 40-character commit SHA; a branch or tag names "
            f"something that moves."
        )
    if entry.weights_is_hub and not entry.weights_revision:
        raise ConfigError(
            f"{where}: weights {entry.weights!r} names a Hub repository and carries "
            f"no weights_revision. A pinned tokenizer beside weights read from a "
            f"mutable branch is half a pin."
        )
    if entry.weights_revision is None:
        return
    if not entry.weights_is_hub:
        raise ConfigError(
            f"{where}: weights_revision pins nothing here — the row's weights are "
            f"not a Hub repository, and a local file is identified by its own bytes."
        )
    if not _COMMIT.fullmatch(entry.weights_revision):
        raise ConfigError(
            f"{where}: weights_revision {entry.weights_revision!r} is not a commit. "
            f"§11 pins artifacts by a full 40-character commit SHA."
        )


def _parse(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the extra
            raise ConfigError(
                "reading a YAML config needs `pyyaml`, which ships in the "
                "`leaderboard` extra: pip install 'glotscope[leaderboard]'. A "
                "JSON config needs nothing."
            ) from exc
        try:
            # safe_load, not load: a config file is input, and the leaderboard is
            # the one command a maintainer runs against files other people wrote.
            document = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path.name} is not valid YAML ({exc})") from exc
    else:
        try:
            document = json.loads(text)
        except ValueError as exc:
            raise ConfigError(f"{path.name} is not valid JSON ({exc})") from exc
    return _require_mapping(document, where=path.name)


def load_config(path: str | Path) -> LeaderboardConfig:
    """Read and validate a leaderboard configuration.

    Raises:
        ConfigError: for anything that would put an unauditable row in a
            published table — an unpinned Hub row, an unlabelled mirror, an
            unreadable key, a parity reference outside the language set.
    """
    source = Path(path)
    try:
        document = _parse(source)
    except OSError as exc:
        raise ConfigError(f"cannot read {source.name}: {exc}") from exc

    _reject_unknown(document, ("version", "corpus", "parameters", "roster"), where=source.name)
    corpus = _corpus(_require_mapping(document.get("corpus"), where="corpus"))
    return LeaderboardConfig(
        corpus=corpus,
        parameters=_require_supported(
            _parameters(
                _require_mapping(document.get("parameters") or {}, where="parameters"),
                languages=corpus.languages,
            ),
            corpus_id=corpus.id,
        ),
        roster=_roster(document.get("roster")),
        version=int(document.get("version", 1)),
    )
