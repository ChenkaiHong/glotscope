"""The Tier 2 reference validation and its committed result (PRD §12.1, D18).

D18 fixes what §7.9 is validated against: Land & Bartolo's *candidate sets*
(999 / 5117 / 1280), not their confirmed counts (3161 / 49 / 6), which come from
a verification prompt run through the model and are therefore Tier 3.

The comparison itself needs three checkpoints and a network, so it is a manual
script whose output is committed. What runs here is everything that does not:
the two selection rules on data small enough to check by hand, and the committed
document's internal consistency — which is what would catch a regenerated file
whose numbers no longer add up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.validate_tier2_reference import (
    MODELS,
    ReferenceToken,
    decompose_delta,
    parse_reference,
    reference_candidates,
    top_share,
)

RESULT = Path(__file__).resolve().parents[1] / "data" / "tier2-reference-validation.json"

M2_RANK_AGREEMENT_GATE = 0.9
"""M2's exit criterion for agreement with the reference implementation."""


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    return json.loads(RESULT.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _token(token_id: int, indicator: float, category: str = "OK") -> ReferenceToken:
    return ReferenceToken(
        token_id=token_id, category=category, indicator=indicator, decoded=str(token_id)
    )


def test_parse_reads_the_main_indicator_and_ignores_blank_lines() -> None:
    lines = [
        json.dumps(
            {
                "i": 0,
                "category": "OK",
                "decoded": "!",
                "indicator_names": ["E_{out} Cosine Distance", "E_{out} L2 Distance"],
                "indicators": [0.62, 2.97],
            }
        ),
        "",
        json.dumps({"i": 1, "category": "UNDECODEABLE", "decoded": "?", "indicators": [0.1, 9.9]}),
    ]

    tokens = parse_reference(lines)

    # Only the first record carries `indicator_names`; the rest inherit it, so
    # the index identifies the indicator and the name cannot.
    assert [(t.token_id, t.category, t.indicator) for t in tokens] == [
        (0, "OK", 0.62),
        (1, "UNDECODEABLE", 0.1),
    ]


def test_the_reference_rule_draws_its_threshold_over_ok_and_selects_over_ok_prefixes() -> None:
    # Ten OK values 1..10: the 2nd percentile interpolates to 1.18, so only the
    # lowest OK token clears it. The OK_SPECIAL token at 0.5 was invisible to
    # the threshold and is still selected, which is the asymmetry being
    # reproduced rather than corrected.
    tokens = [_token(i, float(i + 1)) for i in range(10)]
    tokens.append(_token(99, 0.5, category="OK_SPECIAL"))
    tokens.append(_token(98, 0.4, category="UNDECODEABLE"))

    selected, threshold = reference_candidates(tokens)

    assert selected == {0, 99}
    assert threshold == pytest.approx(1.18)


def test_the_reference_rule_needs_a_strictly_ok_token_to_draw_a_threshold() -> None:
    with pytest.raises(ValueError, match="OK"):
        reference_candidates([_token(0, 1.0, category="UNDECODEABLE")])


def test_the_top_share_rule_floors_and_breaks_ties_by_id() -> None:
    # floor(10 x 2%) is 0, and a candidate set of nothing is not a finding, so
    # the floor is one token.
    assert top_share({i: float(i) for i in range(10)}) == {0}
    # 200 tokens keep 4. Ties resolve by id, matching `detect`'s stable sort
    # over a contiguous domain.
    values = {i: 0.0 if i in {7, 3, 1, 5, 9} else 1.0 for i in range(200)}
    assert top_share(values) == {1, 3, 5, 7}


def test_the_delta_decomposition_accounts_for_every_token() -> None:
    tokens = [_token(i, float(i + 1)) for i in range(100)]
    tokens.append(_token(500, 0.5, category="OK_SPECIAL"))
    reference_by_id = {token.token_id: token for token in tokens}
    theirs, _ = reference_candidates(tokens)

    delta = decompose_delta(ours=frozenset({0}), theirs=theirs, reference_by_id=reference_by_id)

    assert delta["ok_special_admitted"] == 1
    assert (
        delta["ok_special_admitted"] + delta["threshold_rule_extra"] + delta["domain_difference"]
        == delta["observed"]
    )


def test_every_pinned_model_is_recorded(document: dict[str, Any]) -> None:
    recorded = {model["checkpoint"]: model for model in document["models"]}

    assert set(recorded) == {model.checkpoint for model in MODELS}
    for pinned in MODELS:
        entry = recorded[pinned.checkpoint]
        assert entry["revision"] == pinned.revision
        assert entry["reference_sha256"] == pinned.reference_sha256
        assert entry["published_candidate_count"] == pinned.published_candidate_count
        # D18's numbers, restated where a future edit would have to touch them.
        assert entry["candidates"]["reference"] == pinned.published_candidate_count


def test_the_committed_deltas_still_add_up(document: dict[str, Any]) -> None:
    for entry in document["models"]:
        delta = entry["candidate_delta"]
        assert (
            delta["ok_special_admitted"]
            + delta["threshold_rule_extra"]
            + delta["domain_difference"]
            == delta["observed"]
        )
        assert (
            entry["candidates"]["reference"] - entry["candidates"]["glotscope"] == delta["observed"]
        )


def test_the_rankings_agree_past_m2s_gate(document: dict[str, Any]) -> None:
    # The counts differ by a handful of tokens; the ordering is what says the
    # two implementations compute the same quantity.
    for entry in document["models"]:
        agreement = entry["rank_agreement"]
        assert agreement["spearman_rho"] >= M2_RANK_AGREEMENT_GATE, entry["checkpoint"]
        assert agreement["shared_domain"] > 0


def test_the_whole_difference_is_the_two_named_rules(document: dict[str, Any]) -> None:
    # The finding, stated as an assertion: apply §7.9's own rule to the
    # reference implementation's published indicator values and it reproduces
    # glotscope's count exactly. Nothing is left over for the indicator, the
    # exclusion set, or an unnamed cause to explain.
    for entry in document["models"]:
        assert entry["candidate_delta"]["domain_difference"] == 0, entry["checkpoint"]


def test_the_candidate_sets_overlap_almost_entirely(document: dict[str, Any]) -> None:
    # Counts that agree can still describe different tokens. These do not: every
    # candidate glotscope reports is one magikarp reports too, bar a single
    # gemma token that sits either side of the two rules' cut.
    for entry in document["models"]:
        assert entry["candidates"]["containment_of_ours_in_theirs"] >= 0.999, entry["checkpoint"]
