"""Under-trained-token indicators (PRD §7.9, D9, D10).

    u_ref             = (1/|t_ref|) sum_{i in t_ref} E_out,i
    C(A, x)_i         = 1 - (A_i . x) / (||A_i|| . ||x||)
    indicator_tied    = C(E_out, u_ref)      # low implies under-trained
    indicator_untied  = || E_in,i ||         # low implies under-trained

Both point the same way, so both rank ascending and the most under-trained token
is first in either.

Why the two differ at all, which is the interesting part: every row of the output
embedding participates in the softmax at every step, so untrained rows drift
together and share a cosine signature. An *input* row for a token that never
appeared participates in no forward pass — under weight decay its norm decays
toward zero, and without weight decay it simply never leaves initialization.
Either way it separates from trained rows, but how cleanly depends on an
optimizer setting checkpoints rarely document. Hence D10: when the embeddings
are untied, run both and report the Spearman agreement instead of picking one
and hoping.

This module is numeric only. It takes arrays and returns ranked ids — no
tokenizer, no I/O, no decoding. Turning an id into a printable token needs the
tokenizer and is the caller's job; this needs to stay testable on four-row
matrices.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from glotscope.enums import Confidence, Indicator

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray

    FloatMatrix = NDArray[np.floating[Any]]
    FloatVector = NDArray[np.floating[Any]]

__all__ = ["AGREEMENT_THRESHOLD", "Detection", "cosine_distance", "detect", "spearman"]

AGREEMENT_THRESHOLD = 0.7
"""Spearman rho below which the two indicators are reported as disagreeing.

§7.9 requires ``LOW_CONFIDENCE`` "when they disagree beyond threshold" and does
not fix the threshold, which makes this a contested parameter in the §9 sense.
0.7 is the conventional line for strong rank agreement; it is not derived from
the source paper, which never states one. Every result computed under it carries
a warning naming the value, so a reader can see what ``HIGH`` was measured
against rather than inferring it.
"""


@dataclass(frozen=True, slots=True)
class Detection:
    """Ranked under-training candidates and the confidence in them."""

    ranked: tuple[tuple[int, float], ...]
    """``(token_id, indicator value)``, most under-trained first."""

    indicator: Indicator
    """Which indicator produced :attr:`ranked`. For untied checkpoints both were
    computed, but the ranking is ``L2(E_in)`` — the one needing no reference set
    — and the cosine indicator's role is to corroborate it."""

    agreement: float | None
    """Spearman rho between the two indicators; ``None`` when tied, because then
    only one exists. Not a quality score: it says whether two measurements of the
    same thing concur, and nothing about whether either is right."""

    confidence: Confidence
    pre_exclusion: int
    post_exclusion: int
    """Vocabulary size before Stage-1 exclusion, and after it. Both recorded
    because ``top_pct`` applies to the second, and that ordering is what makes a
    published candidate-set size reproducible."""

    first_pc_removed: bool
    warnings: tuple[str, ...]


def _average_ranks(values: FloatVector) -> FloatVector:
    """Ranks with ties averaged over the span they share."""
    import numpy as np

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    unique, inverse = np.unique(values, return_inverse=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    np.add.at(counts, inverse, 1.0)
    averaged: FloatVector = (sums / counts)[inverse]
    return averaged


def spearman(left: FloatVector, right: FloatVector) -> float:
    """Spearman rank correlation, with tied ranks averaged.

    Ties get the mean of the ranks they span. Without that the coefficient is
    not Spearman's — and indicator values tie constantly, since embedding rows
    that never moved from a shared initialization have exactly equal norms.

    Returns ``nan`` when either side is constant, because no rank correlation
    exists there — every value shares one rank, so there is no variation to
    correlate. That is the same answer ``corrcoef`` gives, arrived at without its
    divide-by-zero ``RuntimeWarning``; the caller must decide what an undefined
    agreement means, and :func:`detect` treats it as ``LOW_CONFIDENCE``.
    """
    import numpy as np

    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if float(left_ranks.std()) == 0.0 or float(right_ranks.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def cosine_distance(matrix: FloatMatrix, reference: FloatVector) -> FloatVector:
    """``C(A, x)_i = 1 - (A_i . x) / (||A_i|| . ||x||)`` (PRD §7.9).

    A zero row has no direction, so its cosine is undefined rather than zero.
    Those are reported at distance 1 — maximally unlike the reference — which
    keeps them out of the under-trained candidates they would otherwise top on a
    division that produced ``nan``. A zero row is a real possibility under weight
    decay, and it is the ``L2`` indicator that reads it unambiguously.
    """
    import numpy as np

    row_norms = np.linalg.norm(matrix, axis=1)
    reference_norm = float(np.linalg.norm(reference))
    denominator = row_norms * reference_norm
    similarity = np.divide(
        matrix @ reference,
        denominator,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=denominator > 0,
    )
    distance: FloatVector = 1.0 - similarity
    return distance


def _remove_first_principal_component(matrix: FloatMatrix) -> FloatMatrix:
    """Centre and project out the leading singular direction (D9, default OFF).

    The leading direction comes from an eigendecomposition of the ``d x d`` Gram
    matrix rather than an SVD of the ``n x d`` one. Both cost ``O(n d^2)`` flops,
    but ``svd(..., full_matrices=False)`` also materialises ``U`` at the full
    ``n x d`` — 2 GB on gemma-2b's 256000 x 2048, on top of the 2 GB copy of
    ``centred`` — only to discard it, since ``right[0]`` is the sole output used.
    The Gram matrix is ``d x d`` whatever ``n`` is, so peak memory stops scaling
    with the vocabulary. Its eigenvectors are the right singular vectors.
    """
    import numpy as np

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    gram = centred.T @ centred
    # `eigh` returns eigenvalues ascending, so the leading direction is last.
    leading = np.linalg.eigh(gram)[1][:, -1]
    projected: FloatMatrix = centred - np.outer(centred @ leading, leading)
    return projected


def _detection(
    *,
    rows: FloatVector,
    values: FloatVector,
    indicator: Indicator,
    agreement: float | None,
    confidence: Confidence,
    vocab_size: int,
    candidate_count: int,
    top_pct: float,
    first_pc_removed: bool,
    warnings: list[str],
) -> Detection:
    """Rank ascending and keep the top share, for either indicator.

    Both indicators are low-implies-under-trained, so one ascending sort serves
    both and there is no branch here that could invert one of them.
    """
    import numpy as np

    order = np.argsort(values, kind="stable")
    keep = max(1, int(candidate_count * top_pct / 100.0))
    return Detection(
        ranked=tuple((int(rows[index]), float(values[index])) for index in order[:keep]),
        indicator=indicator,
        agreement=agreement,
        confidence=confidence,
        pre_exclusion=vocab_size,
        post_exclusion=candidate_count,
        first_pc_removed=first_pc_removed,
        warnings=tuple(warnings),
    )


def detect(
    *,
    e_in: FloatMatrix,
    e_out: FloatMatrix,
    tied: bool,
    reference_ids: Iterable[int] | None,
    excluded: frozenset[int],
    vocab_size: int,
    top_pct: float = 2.0,
    first_pc_removed: bool = False,
) -> Detection:
    """Rank under-trained candidates (PRD §7.9 Stage 2).

    Args:
        e_in: input embedding matrix; rows may exceed ``vocab_size``.
        e_out: output embedding matrix; the same array as ``e_in`` when tied.
        tied: whether the checkpoint ties its embeddings. Decides how many
            indicators exist, not merely which one is preferred.
        reference_ids: ``t_ref``, resolved by
            :func:`~glotscope.reference_set.resolve_reference_set`. ``None`` when
            the fallback chain was exhausted, which §7.9's table allows only for
            an untied checkpoint — ``L2(E_in)`` then runs alone, at
            ``LOW_CONFIDENCE``.
        excluded: §7.9 Stage 1 exclusions — partial-UTF-8, unreachable and
            special ids.
        vocab_size: ``|V|``. Rows above it are padding, never candidates.
        top_pct: share of the **post-exclusion** vocabulary to return.
        first_pc_removed: D9. Off by default; the source paper's own Table 2
            shows no consistent improvement from it.

    Raises:
        ValueError: if ``top_pct`` is outside ``(0, 100]``; if Stage 1 excluded
            the whole vocabulary — an empty candidate set drawn from an empty
            domain is not a finding; or if ``reference_ids`` is ``None`` for a
            tied checkpoint, which leaves no indicator that can run.
    """
    import numpy as np

    if not 0.0 < top_pct <= 100.0:
        raise ValueError(f"top_pct must be in (0, 100], got {top_pct}")

    candidate_ids = [token_id for token_id in range(vocab_size) if token_id not in excluded]
    if not candidate_ids:
        raise ValueError(
            "Stage 1 excluded the entire vocabulary, so no domain is left to "
            "rank. An empty candidate set here would read as 'no under-trained "
            "tokens' when it means 'nothing was examined'."
        )

    warnings: list[str] = []
    scored_in = _remove_first_principal_component(e_in) if first_pc_removed else e_in
    scored_out = _remove_first_principal_component(e_out) if first_pc_removed else e_out
    if first_pc_removed:
        warnings.append(
            "first principal component removed from the embedding matrices; this "
            "is off by default (D9) because the source paper's Table 2 shows no "
            "consistent improvement from it across seven models"
        )

    if reference_ids is None and tied:
        raise ValueError(
            "a tied checkpoint has only the cosine indicator, and that indicator "
            "is defined against u_ref. With no reference set there is nothing to "
            "degrade to, so this is a refusal rather than a warning."
        )

    rows = np.asarray(candidate_ids)
    agreement: float | None = None
    confidence = Confidence.HIGH

    if reference_ids is None:
        # §7.9's untied row: L2(E_in) alone needs no reference set. Degrading is
        # what that table prescribes — but not at HIGH confidence, because D10's
        # position is precisely that one indicator is never run alone and trusted.
        indicator = Indicator.L2_E_IN
        values = np.linalg.norm(scored_in[rows], axis=1)
        confidence = Confidence.LOW_CONFIDENCE
        warnings.append(
            "the reference set fallback chain was exhausted, so the cosine "
            "indicator could not run. Ranking is by L2(E_in) alone and no "
            "agreement was measurable; treat the candidate set as provisional"
        )
        return _detection(
            rows=rows,
            values=values,
            indicator=indicator,
            agreement=agreement,
            confidence=confidence,
            vocab_size=vocab_size,
            candidate_count=len(candidate_ids),
            top_pct=top_pct,
            first_pc_removed=first_pc_removed,
            warnings=warnings,
        )

    reference_rows = np.asarray(sorted(set(reference_ids)))
    u_ref = scored_out[reference_rows].mean(axis=0)
    if not float(np.linalg.norm(u_ref)) > 0.0:
        # A zero u_ref has no direction, so every cosine collapses to exactly 1.0
        # and the ranking degenerates to token-id order — a plausible-looking
        # result that measures nothing. It is the normal case rather than an
        # exotic one: chain link 2 is padding rows, and padding rows are usually
        # exactly zero. `cosine_distance` guards a zero *row*; nothing guarded a
        # zero *reference*, and that is the one that destroys the indicator
        # rather than a single entry.
        if tied:
            raise ValueError(
                f"the reference set resolved to {len(reference_rows)} rows whose "
                f"mean is the zero vector, so u_ref has no direction and every "
                f"cosine distance is exactly 1.0. A tied checkpoint has no other "
                f"indicator to fall back to, so this is a refusal: the ranking "
                f"would be token-id order wearing the name of a measurement."
            )
        indicator = Indicator.L2_E_IN
        values = np.linalg.norm(scored_in[rows], axis=1)
        confidence = Confidence.LOW_CONFIDENCE
        warnings.append(
            "the reference set's mean is the zero vector, so the cosine "
            "indicator has no direction to measure against and could not run. "
            "Ranking is by L2(E_in) alone and no agreement was measurable; "
            "treat the candidate set as provisional"
        )
        return _detection(
            rows=rows,
            values=values,
            indicator=indicator,
            agreement=agreement,
            confidence=confidence,
            vocab_size=vocab_size,
            candidate_count=len(candidate_ids),
            top_pct=top_pct,
            first_pc_removed=first_pc_removed,
            warnings=warnings,
        )

    cosine = cosine_distance(scored_out[rows], u_ref)

    if tied:
        indicator = Indicator.COSINE_TO_UNUSED_MEAN
        values = cosine
    else:
        indicator = Indicator.L2_E_IN
        values = np.linalg.norm(scored_in[rows], axis=1)
        measured = spearman(values, cosine)
        if math.isnan(measured):
            # `corrcoef` divides by a standard deviation of zero when either
            # indicator is constant across the domain — rows that never moved
            # from a shared initialization have exactly equal norms, which is the
            # tie case `spearman` documents. `nan < threshold` is False, so the
            # disagreement branch would be skipped and HIGH published; the nan
            # then reaches `canonical_json`, which sets `allow_nan=False` and
            # kills the run after every number has already been computed.
            confidence = Confidence.LOW_CONFIDENCE
            warnings.append(
                "agreement between the two indicators is undefined: one of them "
                "is constant across the whole domain, so no rank correlation "
                "exists. Ranking is by L2(E_in); treat the candidate set as "
                "provisional"
            )
        else:
            agreement = measured
            if agreement < AGREEMENT_THRESHOLD:
                confidence = Confidence.LOW_CONFIDENCE
                warnings.append(
                    f"the two indicators disagree: Spearman rho {agreement:.3f} is "
                    f"below the {AGREEMENT_THRESHOLD} threshold. Ranking is by "
                    f"L2(E_in); treat the candidate set as provisional"
                )

    return _detection(
        rows=rows,
        values=values,
        indicator=indicator,
        agreement=agreement,
        confidence=confidence,
        vocab_size=vocab_size,
        candidate_count=len(candidate_ids),
        top_pct=top_pct,
        first_pc_removed=first_pc_removed,
        warnings=warnings,
    )
