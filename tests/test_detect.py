"""The two Tier 2 under-training indicators (PRD §7.9, D9, D10).

    u_ref             = (1/|t_ref|) sum_{i in t_ref} E_out,i
    C(A, x)_i         = 1 - (A_i . x) / (||A_i|| . ||x||)
    indicator_tied    = C(E_out, u_ref)      # low implies under-trained
    indicator_untied  = || E_in,i ||         # low implies under-trained

Both point the same way, so both rank ascending. When the embeddings are untied
D10 requires running both and reporting their Spearman agreement rather than
assuming which one weight decay favours — applied weight decay is frequently
undocumented, and measuring beats guessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from glotscope.detect import cosine_distance, detect, spearman
from glotscope.enums import Confidence, Indicator


def test_spearman_is_one_for_a_monotone_pair() -> None:
    assert spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0])) == 1.0


def test_spearman_is_minus_one_for_a_reversed_pair() -> None:
    assert spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0])) == -1.0


def test_spearman_averages_tied_ranks() -> None:
    # Ties must share the mean of the ranks they span, or the coefficient is
    # not Spearman's. Ranks become [1.5, 1.5, 3.5, 3.5] against [1, 2, 3, 4],
    # giving 4 / (2 * sqrt(5)).
    value = spearman(np.array([1.0, 1.0, 2.0, 2.0]), np.array([10.0, 20.0, 30.0, 40.0]))
    assert value == pytest.approx(2 / np.sqrt(5), abs=1e-12)


def test_cosine_distance_is_zero_along_the_reference_direction() -> None:
    matrix = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    reference = np.array([1.0, 0.0])

    distances = cosine_distance(matrix, reference)

    # Scale-invariant: row 1 is twice row 0 and just as aligned.
    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    assert distances[1] == pytest.approx(0.0, abs=1e-12)
    assert distances[2] == pytest.approx(1.0, abs=1e-12)


def test_removing_the_first_principal_component_changes_the_values() -> None:
    # Found by mutation: forcing `first_pc_removed` off inside `detect` broke no
    # test. The flag was asserted as *recorded* and never as *effective*, so D9's
    # switch could have been inert and every test would still have passed.
    #
    # Rows share a dominant direction with a smaller orthogonal component;
    # projecting the dominant one out leaves the orthogonal part, so the ranking
    # is computed over different numbers.
    e_in = np.array([[3.0, 0.1], [3.0, 0.9], [3.0, 0.5], [0.0, 0.0]])
    e_out = np.array([[1.0, 0.2], [1.0, 0.8], [1.0, 0.4], [1.0, 0.0]])

    plain = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=(3,),
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
    )
    removed = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=(3,),
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
        first_pc_removed=True,
    )

    assert removed.first_pc_removed is True
    assert [value for _, value in plain.ranked] != [value for _, value in removed.ranked]


def test_a_zero_reference_mean_is_refused_on_a_tied_checkpoint() -> None:
    # Chain link 2 is padding rows and padding rows are usually exactly zero, so
    # u_ref = 0 is the ordinary case rather than a contrived one. Every cosine is
    # then exactly 1.0 and the ranking is token-id order — which looks like a
    # result. A tied checkpoint has no second indicator, so this must refuse.
    e_out = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector"):
        detect(
            e_in=e_out,
            e_out=e_out,
            tied=True,
            reference_ids=(3, 4),
            excluded=frozenset(),
            vocab_size=3,
            top_pct=50.0,
        )


def test_a_zero_reference_mean_degrades_an_untied_checkpoint_to_l2() -> None:
    # Untied has somewhere to fall back to, so this degrades rather than refuses
    # — at LOW_CONFIDENCE, because one indicator run alone is exactly what D10
    # says never to trust.
    e_in = np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    e_out = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    result = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=(3, 4),
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
    )

    assert result.indicator is Indicator.L2_E_IN
    assert result.confidence is Confidence.LOW_CONFIDENCE
    assert result.agreement is None
    assert any("zero vector" in warning for warning in result.warnings)
    # Ascending L2, so the smallest row leads — not token-id order.
    assert [token_id for token_id, _ in result.ranked] == [2, 1, 0]


def test_an_undefined_agreement_is_low_confidence_and_not_a_nan() -> None:
    # Every candidate row of E_in shares a norm, so the L2 indicator is constant
    # and `corrcoef` divides by a zero standard deviation. `nan < threshold` is
    # False, so without an explicit check this published HIGH — and the nan then
    # killed `canonical_json`, which sets allow_nan=False, after every number had
    # already been computed.
    e_in = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    e_out = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [1.0, 0.0]])

    result = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=(3,),
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
    )

    assert result.agreement is None
    assert result.confidence is Confidence.LOW_CONFIDENCE
    assert any("undefined" in warning for warning in result.warnings)


def test_a_tied_checkpoint_runs_the_cosine_indicator_alone() -> None:
    # Arrange — rows 0 and 1 point at the reference row 3; row 2 does not.
    e_out = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.0]])

    # Act
    result = detect(
        e_in=e_out,
        e_out=e_out,
        tied=True,
        reference_ids=(3,),
        excluded=frozenset({3}),
        vocab_size=4,
        top_pct=50.0,
    )

    # Assert
    assert result.indicator is Indicator.COSINE_TO_UNUSED_MEAN
    assert result.agreement is None
    assert result.confidence is Confidence.HIGH
    assert result.ranked[0][0] == 0


def test_an_untied_checkpoint_runs_both_and_reports_agreement() -> None:
    # Arrange — L2 and cosine rank these the same way.
    e_in = np.array([[0.01, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])
    e_out = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [1.0, 0.0]])

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=(3,),
        excluded=frozenset({3}),
        vocab_size=4,
        top_pct=100.0,
    )

    # Assert
    assert result.indicator is Indicator.L2_E_IN
    assert result.agreement is not None
    assert result.confidence is Confidence.HIGH


def test_indicators_that_disagree_are_reported_as_low_confidence() -> None:
    # Arrange — L2 ascending is the exact reverse of cosine ascending.
    e_in = np.array([[0.1, 0.0], [1.0, 0.0], [10.0, 0.0], [1.0, 0.0]])
    e_out = np.array([[0.0, 1.0], [0.7, 0.7], [1.0, 0.0], [1.0, 0.0]])

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=(3,),
        excluded=frozenset({3}),
        vocab_size=4,
        top_pct=100.0,
    )

    # Assert
    assert result.agreement is not None
    assert result.agreement < 0
    assert result.confidence is Confidence.LOW_CONFIDENCE
    assert any("disagree" in warning for warning in result.warnings)


def test_stage_one_exclusions_never_appear_as_candidates() -> None:
    # Arrange
    e_in = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [2.0, 0.0]])

    # Act — row 0 has the smallest norm of all and is excluded.
    result = detect(
        e_in=e_in,
        e_out=e_in,
        tied=True,
        reference_ids=(3,),
        excluded=frozenset({0, 3}),
        vocab_size=4,
        top_pct=100.0,
    )

    # Assert
    assert 0 not in [token_id for token_id, _ in result.ranked]
    assert 3 not in [token_id for token_id, _ in result.ranked]


def test_top_pct_is_applied_after_exclusion_and_both_counts_are_recorded() -> None:
    # §7.9: the ordering is what makes the denominator reproducible.
    # Arrange — 10 rows, 2 excluded, so 8 survive and 50% of 8 is 4.
    e_in = np.arange(1, 21, dtype=np.float64).reshape(10, 2)

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_in,
        tied=True,
        reference_ids=(9,),
        excluded=frozenset({0, 9}),
        vocab_size=10,
        top_pct=50.0,
    )

    # Assert
    assert result.pre_exclusion == 10
    assert result.post_exclusion == 8
    assert len(result.ranked) == 4


def test_padding_rows_are_not_candidates() -> None:
    # Rows above |V| are not tokens; they are the reference set's second link.
    # Arrange — 6 embedding rows, 4 of them vocabulary.
    e_in = np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [4.0, 0.0], [0.0, 0.1], [0.0, 0.1]])

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_in,
        tied=True,
        reference_ids=(4, 5),
        excluded=frozenset(),
        vocab_size=4,
        top_pct=100.0,
    )

    # Assert
    assert max(token_id for token_id, _ in result.ranked) < 4
    assert result.pre_exclusion == 4


def test_first_principal_component_removal_is_off_by_default() -> None:
    # D9: the source paper's own Table 2 shows no consistent improvement, so
    # shipping it on would be cargo-culting.
    # Arrange
    e_in = np.array([[1.0, 0.0], [0.5, 0.0], [0.25, 0.0], [1.0, 1.0]])

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_in,
        tied=True,
        reference_ids=(3,),
        excluded=frozenset({3}),
        vocab_size=4,
        top_pct=100.0,
    )

    # Assert
    assert result.first_pc_removed is False


def test_first_principal_component_removal_is_recorded_when_asked_for() -> None:
    # Arrange
    e_in = np.array([[1.0, 0.0], [0.5, 0.0], [0.25, 0.0], [1.0, 1.0]])

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_in,
        tied=True,
        reference_ids=(3,),
        excluded=frozenset({3}),
        vocab_size=4,
        top_pct=100.0,
        first_pc_removed=True,
    )

    # Assert
    assert result.first_pc_removed is True
    assert any("first principal component" in warning.lower() for warning in result.warnings)


def test_excluding_everything_leaves_no_candidates_rather_than_guessing() -> None:
    # Arrange
    e_in = np.array([[1.0, 0.0], [0.5, 0.0]])

    # Act / Assert
    with pytest.raises(ValueError, match="Stage 1"):
        detect(
            e_in=e_in,
            e_out=e_in,
            tied=True,
            reference_ids=(1,),
            excluded=frozenset({0, 1}),
            vocab_size=2,
            top_pct=100.0,
        )


def test_an_out_of_range_top_pct_is_refused() -> None:
    # Arrange
    e_in = np.array([[1.0, 0.0], [0.5, 0.0]])

    # Act / Assert
    with pytest.raises(ValueError, match="top_pct"):
        detect(
            e_in=e_in,
            e_out=e_in,
            tied=True,
            reference_ids=(1,),
            excluded=frozenset({1}),
            vocab_size=2,
            top_pct=0.0,
        )


def test_an_untied_checkpoint_degrades_to_l2_when_no_reference_set_exists() -> None:
    # §7.9's table: L2(E_in) alone needs no reference set, so an exhausted chain
    # is a degradation for an untied checkpoint, not a refusal. The refusal
    # belongs to the tied case, where nothing else can run.
    # Arrange
    e_in = np.array([[0.1, 0.0], [1.0, 0.0], [2.0, 0.0]])
    e_out = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=None,
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
    )

    # Assert
    assert result.indicator is Indicator.L2_E_IN
    assert result.ranked[0][0] == 0
    assert result.agreement is None


def test_the_degraded_run_is_not_reported_as_high_confidence() -> None:
    # D10 exists because running one indicator alone and trusting it is the
    # failure mode. When the second cannot run, saying so is the whole point.
    # Arrange
    e_in = np.array([[0.1, 0.0], [1.0, 0.0], [2.0, 0.0]])
    e_out = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Act
    result = detect(
        e_in=e_in,
        e_out=e_out,
        tied=False,
        reference_ids=None,
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
    )

    # Assert
    assert result.confidence is Confidence.LOW_CONFIDENCE
    assert any("reference set" in warning for warning in result.warnings)


def test_a_tied_checkpoint_cannot_run_without_a_reference_set() -> None:
    # The cosine indicator is the only one available when tied, and it is
    # defined against u_ref. There is nothing to degrade to.
    # Arrange
    e_out = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Act / Assert
    with pytest.raises(ValueError, match="tied"):
        detect(
            e_in=e_out,
            e_out=e_out,
            tied=True,
            reference_ids=None,
            excluded=frozenset(),
            vocab_size=2,
            top_pct=100.0,
        )


def test_first_pc_removal_also_reaches_the_tied_cosine_indicator() -> None:
    # The untied test above exercises E_in only, so the E_out projection stayed
    # unkilled by mutation. Tied is the common shape — two of the three reference
    # checkpoints tie — and there the cosine against u_ref is the *whole*
    # indicator, so an inert projection would silently do nothing on most models.
    # Three columns: centring plus PC1 removal on two columns leaves the zero
    # matrix, which detect refuses outright — the guard added earlier today.
    e_out = np.array([[3.0, 0.1, 0.2], [3.0, 0.9, 0.1], [3.0, 0.5, 0.9], [3.0, 0.0, 0.5]])

    plain = detect(
        e_in=e_out,
        e_out=e_out,
        tied=True,
        reference_ids=(3,),
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
    )
    removed = detect(
        e_in=e_out,
        e_out=e_out,
        tied=True,
        reference_ids=(3,),
        excluded=frozenset(),
        vocab_size=3,
        top_pct=100.0,
        first_pc_removed=True,
    )

    assert [value for _, value in plain.ranked] != [value for _, value in removed.ranked]
