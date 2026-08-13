"""Command-line surface (PRD §8.2).

Six subcommands. The one that carries the most weight is ``compare``, which
*refuses* to table metrics computed under different segmenters, alpha values,
normalizers, or language sets — §8.2 is explicit that this is a feature and that
the error message should say so.

Stdlib ``argparse`` rather than a third-party CLI framework: the PRD pins the
toolchain and does not sanction one, and the core install's dependency list is
load-bearing for the clean-environment install promise in G1.

``lint`` and ``analyze`` are implemented. The rest print a targeted message and
exit non-zero rather than raising, so the release does not ship a console script
that tracebacks.

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
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from glotscope import __version__, backend
from glotscope.corpus import REGISTRY, Corpus
from glotscope.enums import Normalization, RenyiNormalizer, Segmenter
from glotscope.errors import GlotscopeError, TokenizerLoadError
from glotscope.manifest import Manifest, canonical_json, environment
from glotscope.report import Report
from glotscope.results import CorpusMetrics
from glotscope.tokenizer import Tokenizer

__all__ = ["build_parser", "main"]

_REFUSED = 1
"""Exit code for a typed refusal: the library declined to emit a number."""

_NOT_YET = 2
"""Exit code for a subcommand whose implementation is still scheduled."""

_MILESTONES = {
    "detect": "M2 (Tier 2)",
    "compare": "M1",
    "leaderboard": "M3",
    "verify": "M1 — the CI job that delivers G4",
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

    compare = sub.add_parser(
        "compare",
        help="tabulate a metric across tokenizers; refuses incomparable results",
    )
    compare.add_argument("tokenizers", nargs="+")
    compare.add_argument("--metric", required=True)
    compare.add_argument("--format", choices=["md", "json", "csv"], default="md")

    leaderboard = sub.add_parser("leaderboard", help="regenerate the published leaderboard")
    leaderboard.add_argument("--config", required=True)
    leaderboard.add_argument("--out", required=True)

    verify = sub.add_parser("verify", help="re-check that a manifest reproduces its numbers")
    verify.add_argument("result", help="path to a result.json")

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


def _emit(document: str, out: str | None) -> None:
    if out is None:
        sys.stdout.write(document)
        return
    Path(out).write_text(document, encoding="utf-8")


def _lint(args: argparse.Namespace) -> int:
    """Tier 0. Warnings go to stderr so stdout stays a document a pipe can read."""
    tokenizer = _load_tokenizer(args.tokenizer, args.revision)
    report = tokenizer.lint()
    for warning in tokenizer.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    _emit(canonical_json(report.to_dict()) + "\n", None)
    return 0


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
    tier1 = tokenizer.analyze(
        loaded,
        leading_space=args.leading_space,
        normalization=Normalization(args.normalization),
        add_special_tokens=args.add_special_tokens,
        segmenter=Segmenter(args.segmenter) if args.segmenter is not None else None,
    )
    parity = tier1.parity(args.parity_reference) if args.parity_reference is not None else None
    gini = tier1.gini() if args.gini else None
    renyi = (
        tier1.renyi_efficiency(
            args.renyi_alpha,
            normalizer=RenyiNormalizer(args.renyi_normalizer),
            nominal_vocab_size=args.nominal_vocab_size,
        )
        if args.renyi_alpha is not None
        else None
    )
    tier1 = replace(tier1, corpus_level=CorpusMetrics(gini=gini, renyi=renyi, parity=parity))
    if tier1.parameters is None or tier1.corpus is None:
        raise RuntimeError("analyze() returned a report without its manifest fragments")

    report = Report(
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
    _emit(canonical_json(report.to_dict()) + "\n", args.out)
    return 0


_HANDLERS = {"lint": _lint, "analyze": _analyze}


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
