"""Command-line surface (PRD §8.2).

Six subcommands. The one that carries the most weight is ``compare``, which
*refuses* to table metrics computed under different segmenters, alpha values,
normalizers, or language sets — §8.2 is explicit that this is a feature and that
the error message should say so.

Stdlib ``argparse`` rather than a third-party CLI framework: the PRD pins the
toolchain and does not sanction one, and the core install's dependency list is
load-bearing for the clean-environment install promise in G1.

``lint``, ``analyze``, ``detect``, ``compare`` and ``verify`` are implemented.
``leaderboard`` prints a targeted message and exits non-zero rather than raising,
so the release does not ship a console script that tracebacks.

Exit codes are part of the interface, because a script reading the status has to
be able to tell these apart:

* ``0`` — the command produced its document.
* ``1`` — a **typed refusal**. Catching one here is not the fallback the errors
  exist to prevent: nothing is substituted, the message is printed verbatim and
  the process still fails. What it buys is a CLI that reports "this corpus is
  not parallel" instead of a traceback.
* ``2`` — a path that is **scheduled but not built**, which is a different claim
  about the world and must not be reported as a refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from glotscope import __version__, backend
from glotscope.compare import METRICS
from glotscope.compare import compare as compare_results
from glotscope.corpus import REGISTRY, Corpus, LoadedCorpus
from glotscope.document import load_result
from glotscope.embeddings import Embeddings
from glotscope.enums import MorphologicalType, Normalization, RenyiNormalizer, Segmenter
from glotscope.errors import GlotscopeError, TokenizerLoadError
from glotscope.manifest import Manifest, ParameterManifest, canonical_json, environment
from glotscope.report import Report, Tier0Report
from glotscope.results import CorpusMetrics
from glotscope.tokenizer import Tokenizer

__all__ = ["build_parser", "main"]

_MORPHOLOGICAL_TYPES = tuple(member.value for member in MorphologicalType)


def _morphological_types(pairs: Sequence[str] | None) -> Mapping[str, MorphologicalType] | None:
    """Parse repeated ``LANG=TYPE`` arguments (§7.7 rule 2).

    ``None`` when the flag was never passed, which is what tells ``analyze``
    apart from a run that passed an empty mapping and meant it: the first is "no
    morphology run", the second is "these languages, none of them declared", and
    the second is a mistake worth a refusal.
    """
    if pairs is None:
        return None
    types: dict[str, MorphologicalType] = {}
    for pair in pairs:
        language, separator, value = pair.partition("=")
        if not separator or not language:
            raise ValueError(
                f"--morphological-type takes LANG=TYPE, got {pair!r}. Scope is "
                f"derived per language and there is no default (§7.7 rule 2)."
            )
        try:
            types[language] = MorphologicalType(value)
        except ValueError as exc:
            raise ValueError(
                f"unknown morphological type {value!r} for {language!r}; "
                f"one of {list(_MORPHOLOGICAL_TYPES)}"
            ) from exc
    return types


def _tri_state(value: str | None) -> bool | None:
    """``"true"``/``"false"``/unset. Spelled out because §7.7 rule 4 gives these
    parameters no default, and a store_true flag would supply one silently."""
    if value is None:
        return None
    return value == "true"


_REFUSED = 1
"""Exit code for a typed refusal: the library declined to emit a number."""

_NOT_YET = 2
"""Exit code for a subcommand whose implementation is still scheduled."""

_MILESTONES = {
    "leaderboard": "M3",
}


def build_parser() -> argparse.ArgumentParser:
    """Assemble the §8.2 command surface."""
    parser = argparse.ArgumentParser(
        prog="glotscope",
        description=(
            "Multilingual tokenizer diagnostics. Reports descriptive metrics, not "
            "quality predictions — see the docs on what these metrics do and do "
            "not tell you."
        ),
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    lint = sub.add_parser("lint", help="Tier 0: vocabulary and UTF-8 integrity")
    lint.add_argument("tokenizer")
    lint.add_argument("--revision", help="commit SHA; unpinned runs emit a warning")

    analyze = sub.add_parser("analyze", help="Tier 1: corpus-based metrics")
    analyze.add_argument("tokenizer")
    analyze.add_argument("--revision")
    analyze.add_argument(
        "--corpus",
        required=True,
        choices=sorted(REGISTRY),
        help="registry id. These are the strings that land in the manifest, so "
        "the set is closed and a typo fails before anything is read",
    )
    analyze.add_argument(
        "--corpus-root",
        required=True,
        help="directory holding the downloaded corpus; glotscope ships none, so "
        "follow the recipe on the registry entry first",
    )
    analyze.add_argument(
        "--corpus-version",
        default=None,
        help="defaults to the release this library pins, where one exists",
    )
    analyze.add_argument("--split", default=None, help="defaults to the registry entry's split")
    analyze.add_argument("--languages", required=True, help="comma-separated codes")
    analyze.add_argument(
        "--segmenter",
        choices=[member.value for member in Segmenter],
        help=(
            "required for word-level metrics (fertility, STRR, morphology); there "
            "is no default. Omit it to compute only the segmenter-free metrics."
        ),
    )
    analyze.add_argument(
        "--leading-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="§7.1 rule 5. Recorded either way, and it can move STRR by tens of points",
    )
    analyze.add_argument(
        "--normalization",
        choices=[member.value for member in Normalization],
        default=Normalization.NFC.value,
    )
    analyze.add_argument(
        "--add-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="recorded either way; specials inflate token counts and do not decode "
        "back to the source text",
    )
    analyze.add_argument(
        "--morphological-type",
        action="append",
        metavar="LANG=TYPE",
        default=None,
        help=(
            "repeatable, e.g. --morphological-type tur_Latn=agglutinative. Required "
            "for a corpus with gold morphology and rejected without one. Scope is "
            f"derived from it (§7.7 rule 2); one of {_MORPHOLOGICAL_TYPES}"
        ),
    )
    analyze.add_argument(
        "--frequency-weighted",
        choices=["true", "false"],
        default=None,
        help=(
            "whether the gold is a stream of occurrences rather than a list of "
            "types. Recorded, not applied, and spelled out rather than a flag "
            "because §7.7 rule 4 gives it no default"
        ),
    )
    analyze.add_argument(
        "--include-single-token-words",
        choices=["true", "false"],
        default=None,
        help="whether words the tokenizer emitted whole are scored; no default",
    )
    analyze.add_argument("--license-filter", choices=["commercial"], default=None)
    analyze.add_argument("--parity-reference", default=None)
    analyze.add_argument("--gini", action="store_true")
    analyze.add_argument("--renyi-alpha", type=float, default=None)
    analyze.add_argument(
        "--renyi-normalizer",
        choices=[member.value for member in RenyiNormalizer],
        default=RenyiNormalizer.OBSERVED.value,
    )
    analyze.add_argument("--nominal-vocab-size", type=int, default=None)
    analyze.add_argument("--out", default=None, help="write result.json here (default: stdout)")

    detect = sub.add_parser("detect", help="Tier 2: under-trained-token candidates")
    detect.add_argument("tokenizer")
    detect.add_argument("--revision")
    detect.add_argument("--weights", required=True, help="local path or HF repo id")
    detect.add_argument(
        "--top-pct",
        type=float,
        default=2.0,
        help=(
            "applied AFTER Stage-1 exclusion. For analysis, sweep this rather than "
            "pinning it: a result that exists only at 2%% is an artifact."
        ),
    )
    detect.add_argument(
        "--remove-first-pc",
        action="store_true",
        help=(
            "project out the leading principal component before ranking. Off by "
            "default (D9): the source paper's own Table 2 shows no consistent "
            "improvement from it across seven models. Recorded either way."
        ),
    )
    detect.add_argument(
        "--weights-revision",
        default=None,
        help=(
            "commit SHA for a Hub checkpoint. §11 pins artifacts by revision, and "
            "without one the weights come from a mutable `main` while the "
            "tokenizer beside them is pinned. Rejected beside a local path, where "
            "it would pin nothing."
        ),
    )
    detect.add_argument("--out", default=None, help="write result.json here (default: stdout)")

    compare = sub.add_parser(
        "compare",
        help="tabulate a metric across published results; refuses incomparable ones",
    )
    compare.add_argument(
        "results",
        nargs="+",
        help=(
            "result.json documents written by `glotscope analyze`. §8.2 sketches "
            "these as tokenizers, but its own refusal requirement is unreachable "
            "that way: tokenizers analyzed together share one set of flags and "
            "can never disagree. Only a published document records the parameters "
            "its numbers were produced under"
        ),
    )
    compare.add_argument("--metric", required=True, choices=list(METRICS))
    compare.add_argument("--format", choices=["md", "json", "csv"], default="md")

    leaderboard = sub.add_parser("leaderboard", help="regenerate the published leaderboard")
    leaderboard.add_argument("--config", required=True)
    leaderboard.add_argument("--out", required=True)

    verify = sub.add_parser("verify", help="re-check that a manifest reproduces its numbers")
    verify.add_argument("result", help="path to a result.json")
    verify.add_argument(
        "--tokenizer",
        required=True,
        help=(
            "the tokenizer.json the result describes. Required because §9 keeps "
            "filesystem paths out of the manifest, so the document records what "
            "the artifact is (a SHA-256) but not where it lives. The hash is "
            "checked against the manifest before anything is recomputed."
        ),
    )
    verify.add_argument(
        "--weights",
        default=None,
        help=(
            "the embedding tensors the result describes, for a document carrying "
            "a tier2 block. Required for the same reason --tokenizer is: §9 keeps "
            "filesystem paths out of the manifest, so the document records what "
            "the artifact is (a SHA-256) but not where it lives. The hash is "
            "checked against the manifest before anything is recomputed."
        ),
    )
    verify.add_argument(
        "--weights-revision",
        default=None,
        help=(
            "commit SHA, when --weights names a Hub checkpoint. §9 records the "
            "weights by hash and not by revision, so re-fetching the artifact a "
            "result describes needs the pin the original run used."
        ),
    )
    verify.add_argument(
        "--corpus-root",
        default=".",
        help="directory holding the downloaded corpora; the library ships none (D12)",
    )
    verify.add_argument(
        "--license-filter",
        choices=["commercial"],
        default=None,
        help="exclude research-only resources, as the original run may have done",
    )

    return parser


def _looks_like_a_local_path(source: str) -> bool:
    """Tell a path the user meant from a Hub identifier they meant.

    Only reached once the source has been shown not to exist, and the two
    answers differ in what they tell the reader: a path is a wrong argument to
    fix now, an identifier is a feature scheduled for a later release. Guessing
    "identifier" for both is what made a typo look like a missing feature.
    """
    path = Path(source)
    return (
        path.is_absolute()
        or source.startswith(("~", "./", "../", ".\\", "..\\"))
        or path.suffix == ".json"
        # A parent that exists means the user was naming a place on this disk.
        # "." is excluded: a bare name is the shape a Hub identifier takes.
        or (str(path.parent) != "." and path.parent.is_dir())
    )


def _load_tokenizer(source: str, revision: str | None) -> Tokenizer:
    """Load the tokenizer named on the command line.

    Local sources only — a ``tokenizer.json`` or a directory holding one.
    ``from_pretrained`` is not implemented, so a revision or a Hub-style
    identifier is reported as unbuilt rather than resolved to something else: a
    leaderboard row that silently analysed a different artifact than it names is
    the failure §11 exists to prevent.

    Raises:
        TokenizerLoadError: the source names a place on this disk that holds no
            tokenizer. Exit 1 — a wrong argument, not a missing feature.
        NotImplementedError: the source is a Hub identifier or carries a
            revision. Exit 2 — scheduled, not refused.
    """
    if revision is not None:
        raise NotImplementedError(
            "--revision selects a Hugging Face revision, and from_pretrained() is "
            "not implemented in this release. Pass a local tokenizer.json path."
        )
    path = Path(source)
    if path.is_dir():
        candidate = path / "tokenizer.json"
        if not candidate.is_file():
            raise TokenizerLoadError(source, "a directory holding no tokenizer.json")
        return Tokenizer.from_file(candidate)
    if path.is_file():
        return Tokenizer.from_file(path)
    if _looks_like_a_local_path(source):
        raise TokenizerLoadError(source, "no such file or directory")
    raise NotImplementedError(
        f"{source!r} is not a local file. Loading by Hub identifier needs "
        f"from_pretrained(), which is not implemented in this release."
    )


def _load_embeddings(source: str, *, vocab_size: int, revision: str | None = None) -> Embeddings:
    """Load the weights named on the command line.

    A bare name is a Hub identifier and is resolved as one; anything that names
    a place on this disk is read from there. Collapsing the two would report a
    mistyped path as a missing model, or a model as a mistyped path, and send
    the reader after the wrong fix either way.

    Unlike the tokenizer, the Hub path here is implemented:
    ``Embeddings.from_checkpoint`` reads only the shards holding the embedding
    tensors, which is 8.94 GB of Jamba's 96.06 GB rather than all of it.

    Raises:
        FileNotFoundError: the source names a place on this disk holding nothing.
    """
    path = Path(source)
    if path.is_file():
        if revision is not None:
            raise ValueError(
                f"--weights-revision names a commit on the Hub, and {source} is a "
                f"file on this disk. Recording a revision beside a local path "
                f"would pin nothing; that file's identity is its shard_sha256."
            )
        return Embeddings.from_file(path, vocab_size=vocab_size)
    if _looks_like_a_local_path(source):
        raise FileNotFoundError(f"{source}: no such file or directory")
    return Embeddings.from_checkpoint(source, revision=revision)


def _emit(document: str, out: str | None) -> None:
    if out is None:
        sys.stdout.write(document)
        return
    Path(out).write_text(document, encoding="utf-8")


def _echo_warnings(document: Mapping[str, Any]) -> None:
    """Echo a document's own warnings array to stderr.

    Warnings go to stderr so stdout stays a document a pipe can read. It is the
    **assembled** array that gets echoed, never a subset: the entries that matter
    most for a Tier 2 run — which link of the ``t_ref`` chain supplied the
    reference set, and the ``LOW_CONFIDENCE`` verdict when the two indicators
    disagree — are contributed by the tier reports, not by the tokenizer.
    Printing only the tokenizer's is worse than printing none, because a reader
    watching stderr for trouble reads silence as "nothing to report".
    """
    for warning in document.get("warnings", ()):
        print(f"warning: {warning}", file=sys.stderr)


def _lint(args: argparse.Namespace) -> int:
    """Tier 0. Warnings go to stderr so stdout stays a document a pipe can read.

    Tier 0 has no tier-level warnings to aggregate — :meth:`Tier0Report.to_dict`
    emits no ``warnings`` key — so this echoes the tokenizer's directly.
    """
    tokenizer = _load_tokenizer(args.tokenizer, args.revision)
    report = tokenizer.lint()
    for warning in tokenizer.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    _emit(canonical_json(report.to_dict()) + "\n", None)
    return 0


def _build_report(
    tokenizer: Tokenizer,
    loaded: LoadedCorpus,
    *,
    leading_space: bool,
    normalization: Normalization,
    add_special_tokens: bool,
    segmenter: Segmenter | None,
    parity_reference: str | None,
    gini: bool,
    renyi_alpha: float | None,
    renyi_normalizer: RenyiNormalizer,
    nominal_vocab_size: int | None,
    morphological_types: Mapping[str, MorphologicalType] | None = None,
    frequency_weighted: bool | None = None,
    include_single_token_words: bool | None = None,
) -> Report:
    """Assemble the §9 document. One code path, used by ``analyze`` and ``verify``.

    Shared rather than reimplemented because ``verify``'s whole claim is that it
    regenerates what ``analyze`` produced. A second assembly of the same document
    would be a second thing to drift, and that drift would surface as a
    verification failure blamed on the numbers.
    """
    tier1 = tokenizer.analyze(
        loaded,
        leading_space=leading_space,
        normalization=normalization,
        add_special_tokens=add_special_tokens,
        segmenter=segmenter,
        morphological_types=morphological_types,
        frequency_weighted=frequency_weighted,
        include_single_token_words=include_single_token_words,
    )
    tier1 = replace(
        tier1,
        corpus_level=CorpusMetrics(
            gini=tier1.gini() if gini else None,
            renyi=(
                tier1.renyi_efficiency(
                    renyi_alpha,
                    normalizer=renyi_normalizer,
                    nominal_vocab_size=nominal_vocab_size,
                )
                if renyi_alpha is not None
                else None
            ),
            parity=tier1.parity(parity_reference) if parity_reference is not None else None,
        ),
    )
    if tier1.parameters is None or tier1.corpus is None:
        raise RuntimeError("analyze() returned a report without its manifest fragments")

    return Report(
        tier0=tokenizer.lint(),
        tier1=tier1,
        manifest=Manifest(
            tokenizer=tokenizer.manifest,
            parameters=tier1.parameters,
            environment=environment(),
            backend=backend(),
            glotscope_version=__version__,
            corpus=tier1.corpus,
        ),
        warnings=tokenizer.warnings,
    )


def _analyze(args: argparse.Namespace) -> int:
    """Tier 0 + Tier 1 in one §9 document.

    Tier 0 costs milliseconds and needs nothing a Tier 1 run does not already
    have, so it is always included: G2's claim is that a single document spans
    the tiers that were run.
    """
    tokenizer = _load_tokenizer(args.tokenizer, args.revision)
    corpus = Corpus.resolve(
        args.corpus,
        [code.strip() for code in args.languages.split(",") if code.strip()],
        split=args.split,
        version=args.corpus_version,
    )
    loaded = corpus.load(args.corpus_root, license_filter=args.license_filter)
    report = _build_report(
        tokenizer,
        loaded,
        leading_space=args.leading_space,
        normalization=Normalization(args.normalization),
        add_special_tokens=args.add_special_tokens,
        segmenter=Segmenter(args.segmenter) if args.segmenter is not None else None,
        parity_reference=args.parity_reference,
        gini=args.gini,
        renyi_alpha=args.renyi_alpha,
        renyi_normalizer=RenyiNormalizer(args.renyi_normalizer),
        nominal_vocab_size=args.nominal_vocab_size,
        morphological_types=_morphological_types(args.morphological_type),
        frequency_weighted=_tri_state(args.frequency_weighted),
        include_single_token_words=_tri_state(args.include_single_token_words),
    )
    document = report.to_dict()
    _echo_warnings(document)
    _emit(canonical_json(document) + "\n", args.out)
    return 0


_VOLATILE_MANIFEST_FIELDS = ("environment",)
"""Manifest blocks that legitimately differ between machines.

Environment is recorded *because* it varies — Python version, platform, CPU. A
verify that demanded it match would only ever succeed on the machine that
produced the file, so the check would never run in CI, which is the one place
G4 needs it. The numbers are what must reproduce; the environment is reported.
"""

_VOLATILE_TOP_LEVEL_FIELDS = ("glotscope_version", "backend")
"""Provenance that identifies the *producer* rather than the result.

Comparing these would make every release invalidate every result published
before it: upgrading glotscope would fail a verify whose numbers had not moved,
and the message would say the version differed, which is not a reproduction
failure. G4 promises the numbers regenerate.

Reported instead, and the difference is the interesting part — "0.2.0 reproduces
what 0.1.0 published" is the claim a reader wants, and in v2 "the Rust backend
reproduces the Python numbers" is exactly the backend-parity evidence §13 needs.
``schema_version`` is deliberately *not* here: a schema change changes the
document, so it must fail.
"""


def _detect_report(
    tokenizer: Tokenizer,
    tier0: Tier0Report,
    embeddings: Embeddings,
    *,
    top_pct: float,
    remove_first_pc: bool,
) -> Report:
    """Assemble the Tier 0 + Tier 2 document. One code path, used by ``detect``
    and ``verify``.

    The same rule :func:`_build_report` follows for Tier 1, and for the same
    reason: a verify that assembled the document its own way would be comparing
    a published result against a second implementation rather than regenerating
    it, and the two could drift apart without either being wrong on its own.

    No corpus block: Tier 2 reads weights and needs no text, so the manifest
    omits the corpus rather than writing nulls into it — §9's nesting is what a
    reader uses to tell which tiers ran.

    The parameter block records ``leading_space=False`` and
    ``normalization=none`` because that is what happened: nothing here encodes
    corpus text, and Tier 0's reachability check encodes each vocabulary entry
    verbatim with no special tokens. Copying ``analyze``'s defaults in would put
    a claim about text processing into a document that processed none.

    ``tier0`` is passed in rather than re-linted, because the caller has already
    computed it to size the embedding read.
    """
    tier2 = tokenizer.detect_undertrained(
        embeddings,
        top_pct=top_pct,
        remove_first_pc=remove_first_pc,
    )
    # The weights have been read, so `embedding_rows` is known and the load-time
    # warning that called it unknown is no longer true of this document.
    tokenizer_manifest, warnings = tokenizer.with_weights(embedding_rows=embeddings.n_rows)
    return Report(
        tier0=tier0,
        tier2=tier2,
        manifest=Manifest(
            tokenizer=tokenizer_manifest,
            parameters=ParameterManifest(
                leading_space=False,
                normalization=Normalization.NONE,
                add_special_tokens=False,
                top_pct=top_pct,
                candidates_pre_exclusion=tier2.candidates_pre_exclusion,
                candidates_post_exclusion=tier2.candidates_post_exclusion,
                first_pc_removed=tier2.first_pc_removed,
            ),
            environment=environment(),
            backend=backend(),
            glotscope_version=__version__,
            weights=embeddings.manifest,
        ),
        warnings=warnings,
    )


def _detect(args: argparse.Namespace) -> int:
    """Tier 0 + Tier 2 in one §9 document."""
    tokenizer = _load_tokenizer(args.tokenizer, args.revision)
    tier0 = tokenizer.lint()
    embeddings = _load_embeddings(
        args.weights, vocab_size=tier0.vocab_size, revision=args.weights_revision
    )
    report = _detect_report(
        tokenizer,
        tier0,
        embeddings,
        top_pct=args.top_pct,
        remove_first_pc=args.remove_first_pc,
    )
    document = report.to_dict()
    _echo_warnings(document)
    _emit(canonical_json(document) + "\n", args.out)
    return 0


def _comparable(document: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a result that must reproduce bit-identically."""
    stripped = {
        key: value
        for key, value in document.items()
        if key != "manifest" and key not in _VOLATILE_TOP_LEVEL_FIELDS
    }
    manifest = document.get("manifest")
    if isinstance(manifest, Mapping):
        stripped["manifest"] = {
            key: value for key, value in manifest.items() if key not in _VOLATILE_MANIFEST_FIELDS
        }
    return stripped


def _first_difference(committed: Any, regenerated: Any, path: str = "") -> str | None:
    """The path to the first value that differs, in document order.

    Reported rather than a whole diff: the useful thing is *which* number moved,
    and a caller who wants the rest can diff the files. Returns ``None`` when the
    two agree.
    """
    if isinstance(committed, Mapping) and isinstance(regenerated, Mapping):
        for key in sorted(set(committed) | set(regenerated)):
            here = f"{path}.{key}" if path else str(key)
            if key not in committed:
                return f"{here} (only in the regenerated result)"
            if key not in regenerated:
                return f"{here} (only in the committed result)"
            difference = _first_difference(committed[key], regenerated[key], here)
            if difference is not None:
                return difference
        return None
    if isinstance(committed, list) and isinstance(regenerated, list):
        if len(committed) != len(regenerated):
            return f"{path} (length {len(committed)} vs {len(regenerated)})"
        for index, (left, right) in enumerate(zip(committed, regenerated, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if committed != regenerated:
        return f"{path}: committed {committed!r}, regenerated {regenerated!r}"
    return None


def _verify(args: argparse.Namespace) -> int:
    """Regenerate a committed result and compare it (PRD §12.3, G4).

    §12.3 wants regeneration rather than a re-read, so this re-runs the analysis
    from the recorded parameters. §9 forbids filesystem paths in a manifest, so
    the artifact cannot be resolved from the document — the caller supplies it
    and the recorded SHA-256 decides whether it is the right one. That check
    comes first: handing verify the wrong tokenizer must fail on identity, not
    by producing different numbers and blaming the result.
    """
    committed = json.loads(Path(args.result).read_text(encoding="utf-8"))
    manifest = committed.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError(
            f"{args.result!r} is not a glotscope result document: it carries no "
            f"manifest, so there is nothing to regenerate from."
        )

    tokenizer = _load_tokenizer(args.tokenizer, None)
    recorded = manifest["tokenizer"]["tokenizer_json_sha256"]
    if tokenizer.manifest.tokenizer_json_sha256 != recorded:
        raise TokenizerLoadError(
            args.tokenizer,
            f"tokenizer_json_sha256 {tokenizer.manifest.tokenizer_json_sha256} does "
            f"not match the {recorded} this result was produced with. The manifest "
            f"pins the artifact by hash and by nothing else",
        )

    # Dispatch on which tiers the document declares, not on the presence of a
    # corpus. Keying off the corpus alone refused every Tier 2 result — `detect`
    # correctly writes no corpus block, because Tier 2 reads weights and no text
    # — which left the whole tier outside G4's promise that a published number
    # regenerates. §9's nesting is what says which tiers ran; reading it is the
    # generalization, and it is what a third producer will need too.
    report = _regenerate(args, committed, manifest, tokenizer)

    regenerated = json.loads(canonical_json(report.to_dict()))
    difference = _first_difference(_comparable(committed), _comparable(regenerated))
    if difference is not None:
        print(f"glotscope verify: {args.result} did not reproduce: {difference}", file=sys.stderr)
        return _REFUSED

    print(f"{args.result}: reproduced bit-identically.")
    print(
        "environment is excluded from the comparison and reported instead: "
        f"committed {manifest['environment']}, this run {environment().to_dict()}"
    )
    produced_by = committed.get("glotscope_version")
    if produced_by != __version__:
        # Worth saying loudly rather than burying: the numbers a previous
        # release published still regenerate under this one. That is the claim
        # G4 is for, and it is only visible when the versions differ.
        print(
            f"produced by glotscope {produced_by}, reproduced by {__version__} — "
            f"the numbers regenerate across releases."
        )
    produced_backend = committed.get("backend")
    if produced_backend != backend().value:
        print(
            f"produced on the {produced_backend} backend, reproduced on "
            f"{backend().value} — backend parity holds for this result."
        )
    return 0


def _regenerate(
    args: argparse.Namespace,
    committed: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tokenizer: Tokenizer,
) -> Report:
    """Re-run whichever tiers the committed document declares.

    Raises:
        ValueError: for a document no subcommand writes — one carrying both a
            corpus and a tier2 block, or neither. The first cannot be
            regenerated because nothing produces it and there is no single run
            to reproduce; the second is a ``lint`` document, which has no
            recomputed numbers to compare.
    """
    has_corpus = "corpus" in manifest
    has_tier2 = "tier2" in committed
    if has_corpus and has_tier2:
        raise ValueError(
            f"{args.result!r} carries both a corpus and a tier2 block, and no "
            f"subcommand writes that: `analyze` produces Tier 1 and `detect` "
            f"produces Tier 2. Regenerating it would mean inventing a single run "
            f"that never happened."
        )
    if has_tier2:
        return _regenerate_detect(args, manifest, tokenizer)
    if has_corpus:
        return _regenerate_analyze(args, committed, manifest, tokenizer)
    raise ValueError(
        f"{args.result!r} carries neither a corpus nor a tier2 block, so it "
        f"records no recomputed numbers. `glotscope lint` writes that document; "
        f"verify expects one from `analyze` or `detect`."
    )


def _regenerate_detect(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    tokenizer: Tokenizer,
) -> Report:
    """Re-run Tier 0 + Tier 2 from the recorded parameters (§7.9, G4).

    Raises:
        ValueError: if ``--weights`` was not supplied, or if the supplied
            checkpoint is not the one the manifest pins.
    """
    if args.weights is None:
        raise ValueError(
            f"{args.result!r} carries a tier2 block, so regenerating it needs the "
            f"embedding tensors — pass --weights, exactly as --tokenizer is "
            f"passed. §9 keeps filesystem paths out of the manifest: it records "
            f"what the artifact is (a SHA-256), not where it lives."
        )
    weights_block = manifest.get("weights")
    if weights_block is None:
        raise ValueError(
            f"{args.result!r} carries a tier2 block and no weights manifest, so "
            f"the checkpoint it describes cannot be identified. It was not "
            f"written by `glotscope detect`."
        )

    parameters = manifest["parameters"]
    tier0 = tokenizer.lint()
    embeddings = _load_embeddings(
        args.weights, vocab_size=tier0.vocab_size, revision=args.weights_revision
    )
    # The same rule the tokenizer follows two frames up: identity is checked
    # before anything is recomputed, so the wrong checkpoint fails on what it is
    # rather than by producing different numbers and blaming the result.
    if embeddings.shard_sha256 != weights_block["shard_sha256"]:
        raise ValueError(
            f"{args.weights}: shard_sha256 {embeddings.shard_sha256} does not "
            f"match the {weights_block['shard_sha256']} this result was produced "
            f"with. The manifest pins the artifact by hash and by nothing else."
        )
    return _detect_report(
        tokenizer,
        tier0,
        embeddings,
        top_pct=parameters["top_pct"],
        remove_first_pc=parameters["first_pc_removed"],
    )


def _regenerate_analyze(
    args: argparse.Namespace,
    committed: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tokenizer: Tokenizer,
) -> Report:
    """Re-run Tier 0 + Tier 1 from the recorded parameters (§12.3, G4)."""
    corpus_block = manifest["corpus"]
    parameters = manifest["parameters"]
    corpus_level = committed.get("tier1", {}).get("corpus_level", {})
    corpus = Corpus.resolve(
        corpus_block["id"],
        corpus_block["languages"],
        split=corpus_block["split"],
        version=corpus_block["version"],
        sha256=corpus_block["sha256"],
    )
    loaded = corpus.load(args.corpus_root, license_filter=args.license_filter)
    segmenter = parameters.get("segmenter")
    # §7.7's three recorded parameters are published in the per-language
    # morphology block rather than in ParameterManifest, so they are read from
    # where the value actually is — the same rule this function already follows
    # for Renyi's alpha and normalizer. Reproducing a morphology run without them
    # would either refuse or score under a convention the document did not use.
    morphology_blocks = [
        entry["morphology"]
        for entry in committed.get("tier1", {}).get("per_language", {}).values()
        if "morphology" in entry
    ]
    morphological_types = (
        {
            language: MorphologicalType(entry["morphology"]["morphological_type"])
            for language, entry in committed["tier1"]["per_language"].items()
            if "morphology" in entry
        }
        if morphology_blocks
        else None
    )
    return _build_report(
        tokenizer,
        loaded,
        leading_space=parameters["leading_space"],
        normalization=Normalization(parameters["normalization"]),
        add_special_tokens=parameters["add_special_tokens"],
        segmenter=Segmenter(segmenter) if segmenter is not None else None,
        # Which corpus-level metrics ran is read off the document rather than
        # asked for again: a verify that skipped them would pass a file whose
        # gini or renyi no longer reproduces.
        parity_reference=corpus_level.get("parity", {}).get("reference_language"),
        gini="gini" in corpus_level,
        # Renyi's alpha and normalizer are published in the corpus_level block
        # rather than in ParameterManifest, whose matching fields analyze leaves
        # unset. Read from where the value actually is, and fall back to the
        # parameter block so a document that fills it in still verifies.
        renyi_alpha=corpus_level.get("renyi_alpha", parameters.get("renyi_alpha")),
        renyi_normalizer=RenyiNormalizer(
            corpus_level.get("renyi_normalizer") or parameters.get("renyi_normalizer") or "observed"
        ),
        nominal_vocab_size=corpus_level.get("renyi_nominal_vocab_size"),
        morphological_types=morphological_types,
        # One analyze call produced every block, so the flags are the same in all
        # of them; the first is read rather than cross-checked because a document
        # that disagreed with itself could not have been produced by this library.
        frequency_weighted=morphology_blocks[0]["frequency_weighted"]
        if morphology_blocks
        else None,
        include_single_token_words=(
            morphology_blocks[0]["include_single_token_words"] if morphology_blocks else None
        ),
    )


def _compare(args: argparse.Namespace) -> int:
    """Table one metric across published results (PRD §8.2).

    Reads documents rather than tokenizers. The refusal §8.2 asks for is a
    statement about two *runs*, and only a document carries the parameters its
    run was performed under.
    """
    table = compare_results([load_result(path) for path in args.results], metric=args.metric)
    if args.format == "json":
        print(canonical_json(table.to_dict()))
    elif args.format == "csv":
        print(table.to_csv(), end="")
    else:
        print(table.to_markdown())
    return 0


_HANDLERS = {
    "lint": _lint,
    "analyze": _analyze,
    "detect": _detect,
    "compare": _compare,
    "verify": _verify,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``glotscope`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"glotscope {__version__} (backend: {backend().value})")
        return 0

    if args.command is None:
        parser.print_help()
        return 1

    handler = _HANDLERS.get(args.command)
    if handler is None:
        milestone = _MILESTONES[args.command]
        print(
            f"glotscope {args.command}: not implemented in this release. Scheduled for {milestone}.",
            file=sys.stderr,
        )
        return _NOT_YET

    try:
        return handler(args)
    except NotImplementedError as exc:
        print(f"glotscope {args.command}: {exc}", file=sys.stderr)
        return _NOT_YET
    except (GlotscopeError, ValueError, KeyError, OSError) as exc:
        # Surfacing the refusal, not substituting for it: nothing is defaulted,
        # the message is printed as written and the process still fails.
        print(f"glotscope {args.command}: {exc}", file=sys.stderr)
        return _REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
