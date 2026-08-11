from __future__ import annotations

import pytest

from glotscope.metrics import parity


def test_reference_language_parity_is_one_and_line_counts_match() -> None:
    result = parity({"eng": (1, 3), "hin": (2, 4)}, reference="eng")

    assert result.per_language["eng"] == 1.0
    assert result.per_language["hin"] == pytest.approx(1.5)
    assert result.n_lines_per_language == {"eng": 2, "hin": 2}


@pytest.mark.parametrize(
    ("counts", "reference", "exception", "message"),
    [
        ({}, "eng", ValueError, "at least one language"),
        ({"eng": (1,)}, "fra", KeyError, "reference language"),
        ({"eng": ()}, "eng", ValueError, "at least one aligned line"),
        ({"eng": (1,), "hin": (1, 2)}, "eng", ValueError, "unequal line counts"),
        ({"eng": (1,), "hin": (-1,)}, "eng", ValueError, "cannot be negative"),
        ({"eng": (0,), "hin": (1,)}, "eng", ValueError, "zero tokens"),
    ],
)
def test_parity_rejects_invalid_parallel_counts(
    counts: dict[str, tuple[int, ...]],
    reference: str,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        parity(counts, reference=reference)
