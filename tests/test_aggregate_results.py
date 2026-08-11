from __future__ import annotations

import pytest

from glotscope.aggregate import (
    BoundaryCounts,
    aggregate_documents,
    aggregate_words,
    align_boundaries,
    attribute_scripts,
)
from glotscope.enums import RenyiNormalizer, TypologicalScope
from glotscope.errors import IncomparableError
from glotscope.results import (
    CompressionResult,
    MorphologyResult,
    RenyiResult,
    require_comparable,
    sorted_costs,
)


def test_boundary_counts_calculates_metrics_and_handles_empty_denominators() -> None:
    gathered = BoundaryCounts(true_positive=1, false_positive=7, false_negative=0)
    empty = BoundaryCounts(true_positive=0, false_positive=0, false_negative=0)

    assert gathered.precision == pytest.approx(1 / 8)
    assert gathered.recall == 1.0
    assert gathered.f1 == pytest.approx(2 / 9)
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)


def test_aggregate_boundaries_are_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError):
        aggregate_documents([[1], []], [1, 0], [1, 0])
    with pytest.raises(NotImplementedError):
        aggregate_words([[1], [], [2, 3]])
    with pytest.raises(NotImplementedError):
        attribute_scripts([1, 2], [215, 220])
    with pytest.raises(NotImplementedError):
        align_boundaries([[1, 2]], [[1]])


def test_comparability_accepts_matching_keys_and_names_first_difference() -> None:
    left = RenyiResult(0.8, 2.5, RenyiNormalizer.OBSERVED, 1.2)
    same = RenyiResult(0.7, 2.5, RenyiNormalizer.OBSERVED, 1.1)
    different = RenyiResult(0.8, 3.0, RenyiNormalizer.NOMINAL, 1.2)

    require_comparable(left, same)
    with pytest.raises(IncomparableError, match="differing alpha") as raised:
        require_comparable(left, different)
    assert raised.value.field == "alpha"


def test_result_contracts_sort_costs_and_enforce_morphological_scope() -> None:
    assert sorted_costs([3.0, 1.0, 2.0]) == (1.0, 2.0, 3.0)
    assert CompressionResult("eng", 1.0, 1.0, 1, 1.0, "bytes").comparability_key() == {
        "compression_rate_unit": "bytes"
    }

    out_of_scope = MorphologyResult(
        "cmn", TypologicalScope.OUT_OF_SCOPE, None, None, None, False, False
    )
    assert out_of_scope.scope is TypologicalScope.OUT_OF_SCOPE
    with pytest.raises(ValueError, match="OUT_OF_SCOPE"):
        MorphologyResult("cmn", TypologicalScope.OUT_OF_SCOPE, 0.5, None, None, False, False)
    with pytest.raises(ValueError, match="carries no measure"):
        MorphologyResult("tur", TypologicalScope.IN_SCOPE, None, None, None, False, False)
