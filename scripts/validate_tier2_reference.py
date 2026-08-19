"""Validate §7.9's candidate sets against Land & Bartolo's published ones (D18).

Their *confirmed* under-trained counts (3161 / 49 / 6) come from a verification
prompt run through the model, which is Tier 3 and out of scope for v1.0. D18
therefore validates against the quantity glotscope actually computes: the Tier
0 + Tier 2 **candidate set**, published as 999 / 5117 / 1280.

Two things are checked, and they are not the same claim:

1. **Ranking.** Spearman rho between the two implementations' indicator values
   over every token both score. This is the claim that matters — a count can
   agree by accident, an ordering over 250,000 tokens cannot.
2. **Selection.** The candidate sets themselves, with every difference in size
   attributed to a named cause rather than absorbed into a tolerance.

The counts do **not** match exactly, and nothing here is tuned to make them
(§17). ``candidate_delta`` decomposes the difference into three causes, computed
on the reference implementation's own published indicator values so that the
indicator is held fixed while only the selection rule varies:

``ok_special_admitted``
    magikarp derives its threshold from tokens whose category is exactly ``OK``
    and then selects with ``category.startswith("OK")``, so ``OK_SPECIAL`` ids
    enter the candidate set they were excluded from defining. §7.9 Stage 1
    excludes special ids outright, so glotscope never scores them.

``threshold_rule_extra``
    magikarp takes ``np.percentile(..., 2.0)`` — linearly interpolated — and
    admits ties with ``<=``. §7.9 specifies the top ``top_pct`` share, which is
    ``floor(N x 2%)``. The interpolated threshold lands on or just above the
    k-th value, so it admits one more token.

``domain_difference``
    what is left once both rules are applied to the same data: the two
    implementations disagree about which ids Stage 1 removes.

The three sum to the observed delta by construction; the point of writing them
out is that each is a statement someone can check, and a nonzero residue would
have to appear as ``domain_difference`` rather than vanish.

Regenerating this is not a CI job. It reads three checkpoints from the Hub —
about 15 GB — which is why the result is committed and the run is manual:

    curl -sL -O https://raw.githubusercontent.com/cohere-ai/magikarp/764e8cd02e598deb65692184a03f843ce3543ded/results/verifications/openai_community_gpt2_medium.jsonl.gz
    curl -sL -O https://raw.githubusercontent.com/cohere-ai/magikarp/764e8cd02e598deb65692184a03f843ce3543ded/results/verifications/google_gemma_2b.jsonl.gz
    curl -sL -O https://raw.githubusercontent.com/cohere-ai/magikarp/764e8cd02e598deb65692184a03f843ce3543ded/results/verifications/ai21labs_Jamba_v0_1.jsonl.gz

    uv run --no-sync python scripts/validate_tier2_reference.py \
        --reference-dir . --output data/tier2-reference-validation.json

Every input is pinned by digest: the reference files by SHA-256, the checkpoints
by Hub revision. An upstream edit fails the run rather than moving a number.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOP_PCT = 2.0
"""§7.9's share, and magikarp's ``DEFAULT_THRESHOLD_PERCENTILE``. The same 2 is
a top-k share on one side and a percentile on the other, which is the whole of
``threshold_rule_extra``."""

MAIN_INDICATOR = 0
"""magikarp writes its indicators most-preferred first: ``E_in`` L2 norm for an
untied checkpoint, ``E_out`` cosine distance for a tied one. That is §7.9's own
ordering, so index 0 is the indicator glotscope ranks by in both cases."""

DIGITS = 6
"""Decimal places kept for floats. The indicators are float32 reductions over
matrices of 2048 columns and more, so the last bits move with the BLAS kernel;
committing them at full precision would make this file churn between machines
without any number having changed."""

MAGIKARP_REVISION = "764e8cd02e598deb65692184a03f843ce3543ded"
_RESULTS_URL = (
    "https://raw.githubusercontent.com/cohere-ai/magikarp/"
    f"{MAGIKARP_REVISION}/results/verifications"
)


@dataclass(frozen=True)
class ReferenceModel:
    """One checkpoint with everything needed to reproduce its comparison."""

    checkpoint: str
    revision: str
    """Hub revision. A tokenizer that changes silently is the failure mode the
    nightly leaderboard job exists for; here it would move the candidate set
    under a committed number."""

    reference_file: str
    reference_sha256: str
    published_candidate_count: int
    """From D18. Recomputed from the reference file and compared, so a drifted
    pin fails rather than quietly redefining what is being validated against."""


MODELS: tuple[ReferenceModel, ...] = (
    ReferenceModel(
        checkpoint="openai-community/gpt2-medium",
        revision="6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        reference_file="openai_community_gpt2_medium.jsonl.gz",
        reference_sha256="f71b52586210aadd1ec6f13165b6b07b152d2f0d07a4643799cfb5ad0b3207c5",
        published_candidate_count=999,
    ),
    ReferenceModel(
        checkpoint="google/gemma-2b",
        revision="9cf48e52b224239de00d483ec8eb84fb8d0f3a3a",
        reference_file="google_gemma_2b.jsonl.gz",
        reference_sha256="80829dc4b7a3516a071dd25b171249a48e3c509357d4062a88408b2d3a34dfcd",
        published_candidate_count=5117,
    ),
    ReferenceModel(
        checkpoint="ai21labs/Jamba-v0.1",
        revision="9efd11575ba791d9e3d25d4c8b670e78506b2df7",
        reference_file="ai21labs_Jamba_v0_1.jsonl.gz",
        reference_sha256="edc12e51e4ffd0ff7f186f053323cca802c7dd566a12422b85b6b687e96cbbdb",
        published_candidate_count=1280,
    ),
)


@dataclass(frozen=True)
class ReferenceToken:
    """One row of a magikarp verification file."""

    token_id: int
    category: str
    """``OK``, ``OK_SPECIAL``, ``UNDECODEABLE`` and so on. Both the exact value
    and its ``OK`` prefix are load-bearing upstream, and they are used
    inconsistently, which is one of the two differences this script names."""

    indicator: float
    decoded: str


def _round(value: float) -> float:
    return round(float(value), DIGITS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_reference(lines: Iterable[str]) -> tuple[ReferenceToken, ...]:
    """Read a magikarp verification file into rows.

    Only the first record carries ``indicator_names``; the rest inherit it, so
    the index rather than the name is what identifies the main indicator.
    """
    tokens: list[ReferenceToken] = []
    for line in lines:
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        tokens.append(
            ReferenceToken(
                token_id=int(record["i"]),
                category=str(record["category"]),
                indicator=float(record["indicators"][MAIN_INDICATOR]),
                decoded=str(record.get("decoded", "")),
            )
        )
    return tuple(tokens)


def reference_candidates(
    tokens: Sequence[ReferenceToken], *, threshold_pct: float = TOP_PCT
) -> tuple[frozenset[int], float]:
    """magikarp's own selection rule, transcribed from ``candidates_for_verification``.

    The asymmetry is deliberate on their side or it is not — either way it is
    reproduced verbatim here, because the purpose is to reproduce their number,
    not to improve on it. The threshold is drawn over ``category == "OK"``; the
    selection then accepts any ``category`` beginning with ``OK``.
    """
    import numpy as np

    strictly_ok = [token.indicator for token in tokens if token.category == "OK"]
    if not strictly_ok:
        raise ValueError("no token is categorised OK, so no threshold can be drawn")
    threshold = float(np.percentile(strictly_ok, threshold_pct))
    selected = frozenset(
        token.token_id
        for token in tokens
        if token.indicator <= threshold and token.category.startswith("OK")
    )
    return selected, threshold


def top_share(values: Mapping[int, float], *, top_pct: float = TOP_PCT) -> frozenset[int]:
    """§7.9's rule: the lowest ``floor(N x top_pct%)`` ids, ties broken by id.

    ``glotscope.detect`` breaks ties by row order, which is id order over a
    contiguous domain. Sorting on ``(value, id)`` here reproduces that without
    reaching into it.
    """
    keep = max(1, int(len(values) * top_pct / 100.0))
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return frozenset(token_id for token_id, _ in ordered[:keep])


def decompose_delta(
    *,
    ours: frozenset[int],
    theirs: frozenset[int],
    reference_by_id: Mapping[int, ReferenceToken],
) -> dict[str, int]:
    """Attribute every token of the size difference to a named cause.

    Each component is measured on the reference implementation's published
    indicator values, so the indicator is constant and only the rule varies.
    The three sum to the delta by construction — that is the check, not the
    finding: a cause nobody named would have to show up as
    ``domain_difference``.
    """
    strictly_ok = {
        token_id: token.indicator
        for token_id, token in reference_by_id.items()
        if token.category == "OK"
    }
    theirs_ok_only = theirs & frozenset(strictly_ok)
    ours_rule_on_their_data = top_share(strictly_ok)

    ok_special_admitted = len(theirs) - len(theirs_ok_only)
    threshold_rule_extra = len(theirs_ok_only) - len(ours_rule_on_their_data)
    domain_difference = len(ours_rule_on_their_data) - len(ours)
    return {
        "domain_difference": domain_difference,
        "observed": len(theirs) - len(ours),
        "ok_special_admitted": ok_special_admitted,
        "threshold_rule_extra": threshold_rule_extra,
    }


def _token_rows(
    token_ids: Iterable[int], reference_by_id: Mapping[int, ReferenceToken]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token_id in sorted(token_ids):
        token = reference_by_id.get(token_id)
        rows.append(
            {
                "category": token.category if token is not None else None,
                "decoded": token.decoded if token is not None else None,
                "token_id": token_id,
            }
        )
    return rows


def compare_model(model: ReferenceModel, *, reference_dir: Path) -> dict[str, Any]:
    """Run glotscope over one checkpoint and compare it to the published result.

    Reaches the Hub. Imports are local because this file is imported by the test
    suite, which must not need ``numpy`` or a network.
    """
    import numpy as np
    from huggingface_hub import hf_hub_download

    from glotscope.detect import spearman
    from glotscope.embeddings import Embeddings
    from glotscope.tokenizer import Tokenizer

    reference_path = reference_dir / model.reference_file
    actual_sha256 = sha256_file(reference_path)
    if actual_sha256 != model.reference_sha256:
        raise ValueError(
            f"{model.reference_file}: SHA-256 mismatch — expected "
            f"{model.reference_sha256}, got {actual_sha256}. The published result "
            f"is the fixed point of this comparison; a changed one is a different "
            f"claim, not a tolerance"
        )
    with gzip.open(reference_path, "rt", encoding="utf-8") as handle:
        reference_tokens = parse_reference(handle)
    reference_by_id = {token.token_id: token for token in reference_tokens}

    theirs, threshold = reference_candidates(reference_tokens)
    if len(theirs) != model.published_candidate_count:
        raise ValueError(
            f"{model.checkpoint}: the reference file yields {len(theirs)} "
            f"candidates under magikarp's own rule, and D18 records "
            f"{model.published_candidate_count}. One of the two pins has moved"
        )

    tokenizer = Tokenizer.from_file(
        hf_hub_download(model.checkpoint, "tokenizer.json", revision=model.revision)
    )
    tier0 = tokenizer.lint()
    embeddings = Embeddings.from_checkpoint(model.checkpoint, revision=model.revision)

    # top_pct=100 ranks the whole post-exclusion domain, which is what the rank
    # correlation needs; the published set is the leading share of that same
    # ranking, and the assertion below is what says so rather than assuming it.
    full = tokenizer.detect_undertrained(embeddings, top_pct=100.0)
    report = tokenizer.detect_undertrained(embeddings, top_pct=TOP_PCT)
    ours = frozenset(candidate.token_id for candidate in report.candidates)
    if ours != frozenset(
        candidate.token_id for candidate in full.candidates[: report.candidate_count]
    ):
        raise ValueError(
            f"{model.checkpoint}: the top_pct=2 set is not the leading share of "
            f"the full ranking, so the two runs disagree about the ordering"
        )

    our_values = {candidate.token_id: candidate.indicator_value for candidate in full.candidates}
    shared = sorted(set(our_values) & set(reference_by_id))
    rho = spearman(
        np.array([our_values[token_id] for token_id in shared], dtype=np.float64),
        np.array([reference_by_id[token_id].indicator for token_id in shared], dtype=np.float64),
    )

    excluded = tier0.stage1_exclusions()
    their_excluded = frozenset(
        token.token_id for token in reference_tokens if token.category != "OK"
    )
    intersection = ours & theirs
    return {
        "candidate_delta": decompose_delta(
            ours=ours, theirs=theirs, reference_by_id=reference_by_id
        ),
        "candidates": {
            "containment_of_ours_in_theirs": _round(len(intersection) / len(ours)),
            "glotscope": len(ours),
            "intersection": len(intersection),
            "jaccard": _round(len(intersection) / len(ours | theirs)),
            "ours_only": _token_rows(ours - theirs, reference_by_id),
            "reference": len(theirs),
            "theirs_only": _token_rows(theirs - ours, reference_by_id),
        },
        "checkpoint": model.checkpoint,
        "glotscope": {
            "confidence": report.confidence.value,
            "dtype": embeddings.dtype,
            "indicator": report.indicator.value,
            "indicator_agreement": (
                None if report.indicator_agreement is None else _round(report.indicator_agreement)
            ),
            "shard_sha256": embeddings.shard_sha256,
            "tied": embeddings.tied,
            "vocab_size": tier0.vocab_size,
        },
        "published_candidate_count": model.published_candidate_count,
        "rank_agreement": {
            "shared_domain": len(shared),
            "spearman_rho": _round(rho),
        },
        "reference_file": model.reference_file,
        "reference_sha256": model.reference_sha256,
        "revision": model.revision,
        "stage1_exclusion": {
            "glotscope": len(excluded),
            "ours_only": _token_rows(excluded - their_excluded, reference_by_id),
            "partial_utf8": len(tier0.partial_utf8_tokens),
            "reference_non_ok": len(their_excluded),
            "special": len(tier0.special_tokens),
            "theirs_only": _token_rows(their_excluded - excluded, reference_by_id),
            "unreachable": len(tier0.unreachable_tokens),
        },
        "threshold": {
            "reference_percentile_value": _round(threshold),
            "top_pct": TOP_PCT,
        },
    }


def validate(*, reference_dir: Path) -> dict[str, Any]:
    """Compare every pinned model and assemble the committed document."""
    models = [compare_model(model, reference_dir=reference_dir) for model in MODELS]
    return {
        "indicator_index": MAIN_INDICATOR,
        "models": models,
        "reference_implementation": {
            "name": "magikarp",
            "results_url": _RESULTS_URL,
            "revision": MAGIKARP_REVISION,
            "url": "https://github.com/cohere-ai/magikarp",
        },
        "rounding_digits": DIGITS,
        "schema_version": 1,
        "top_pct": TOP_PCT,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        required=True,
        type=Path,
        help="directory holding the downloaded magikarp verification files",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = validate(reference_dir=args.reference_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
