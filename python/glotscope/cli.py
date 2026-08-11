"""Command-line surface (PRD §8.2).

Six subcommands. The one that carries the most weight is ``compare``, which
*refuses* to table metrics computed under different segmenters, alpha values,
normalizers, or language sets — §8.2 is explicit that this is a feature and that
the error message should say so.

Stdlib ``argparse`` rather than a third-party CLI framework: the PRD pins the
toolchain and does not sanction one, and the core install's dependency list is
load-bearing for the clean-environment install promise in G1.

At version 0.0.0 every handler is a stub. They print a targeted message and exit
non-zero rather than raising, so the placeholder release does not ship a console
script that tracebacks.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

__all__ = ["build_parser", "main"]

_NOT_YET = 2
"""Exit code for a subcommand whose implementation is still scheduled."""

_MILESTONES = {
    "lint": "M1 (Tier 0)",
    "analyze": "M1 (Tier 1)",
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
    analyze.add_argument("--corpus", required=True)
    analyze.add_argument("--languages", required=True, help="comma-separated codes")
    analyze.add_argument(
        "--segmenter",
        help=(
            "required for word-level metrics (fertility, STRR, morphology); there "
            "is no default. Omit it to compute only the segmenter-free metrics."
        ),
    )
    analyze.add_argument("--leading-space", action="store_true", default=True)
    analyze.add_argument("--normalization", default="NFC")
    analyze.add_argument("--license-filter", choices=["commercial"], default=None)

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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``glotscope`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from glotscope import __version__, backend

        print(f"glotscope {__version__} (backend: {backend().value})")
        return 0

    if args.command is None:
        parser.print_help()
        return 1

    milestone = _MILESTONES[args.command]
    print(
        f"glotscope {args.command}: not implemented in this release. "
        f"Scheduled for {milestone}.",
        file=sys.stderr,
    )
    return _NOT_YET


if __name__ == "__main__":
    raise SystemExit(main())
