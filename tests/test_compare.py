"""Tabling published results side by side (PRD §8.2).

The refusal is the feature. §8.2 says ``compare`` must decline to table metrics
computed under different segmenters, alpha values, normalizers or language sets,
and the error message must say that this is deliberate.

Comparability is scoped **per metric**, exactly as :mod:`glotscope.results`
already scopes it: a Renyi alpha that differs makes two Renyi numbers
incomparable and says nothing at all about two compression numbers. A global
check would over-refuse, and an over-refusing tool teaches its users to work
around it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glotscope.compare import compare
from glotscope.document import LoadedResult, load_result
from glotscope.errors import IncomparableError
from glotscope.manifest import canonical_json

FIXTURE = Path(__file__).resolve().parents[1] / "verification"
RESULT = FIXTURE / "result.json"

pytestmark = pytest.mark.skipif(
    not RESULT.is_file(),
    reason=(
        "the G4 fixture lives in the repository, not in the distribution: "
        "tests/ ships in the sdist and verification/ deliberately does not, so "
        "these run from a checkout and skip from an unpacked release"
    ),
)


def _document() -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(RESULT.read_text(encoding="utf-8"))
    return parsed


def _write(path: Path, document: Any) -> LoadedResult:
    path.write_text(canonical_json(document), encoding="utf-8")
    return load_result(path)


def _other_tokenizer(document: dict[str, Any]) -> dict[str, Any]:
    """A second result that differs only in which tokenizer produced it."""
    document["manifest"]["tokenizer"]["tokenizer_json_sha256"] = "b" * 64
    return document


def test_a_comparable_pair_tables_one_row_per_language(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    right = _write(tmp_path / "b.json", _other_tokenizer(_document()))

    # Act
    table = compare([left, right], metric="fertility")

    # Assert
    assert table.columns == ("local@49697ba047fd", "local@bbbbbbbbbbbb")
    assert table.rows["eng_Latn"] == (4.6, 4.6)
    assert tuple(table.rows) == ("eng_Latn", "hin_Deva")


def test_differing_segmenters_refuse_to_share_a_fertility_table(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    other = _other_tokenizer(_document())
    other["manifest"]["parameters"]["segmenter"] = "icu"
    other["tier1"]["segmenter"] = "icu"
    right = _write(tmp_path / "b.json", other)

    # Act / Assert
    with pytest.raises(IncomparableError) as excinfo:
        compare([left, right], metric="fertility")
    assert excinfo.value.field == "segmenter"


def test_a_differing_renyi_alpha_does_not_block_a_compression_comparison(
    tmp_path: Path,
) -> None:
    # The discriminating case for per-metric scoping: alpha is a parameter of
    # the Renyi question and of nothing else.
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    other = _other_tokenizer(_document())
    other["tier1"]["corpus_level"]["renyi_alpha"] = 3.0
    right = _write(tmp_path / "b.json", other)

    # Act
    table = compare([left, right], metric="cpt")

    # Assert
    assert table.rows["hin_Deva"] == (0.5454545454545454, 0.5454545454545454)


def test_a_differing_renyi_alpha_refuses_a_renyi_comparison(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    other = _other_tokenizer(_document())
    other["tier1"]["corpus_level"]["renyi_alpha"] = 3.0
    right = _write(tmp_path / "b.json", other)

    # Act / Assert
    with pytest.raises(IncomparableError) as excinfo:
        compare([left, right], metric="renyi_efficiency")
    assert excinfo.value.field == "alpha"


def test_differing_language_sets_refuse(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    other = _other_tokenizer(_document())
    other["manifest"]["corpus"]["languages"] = ["eng_Latn"]
    del other["tier1"]["per_language"]["hin_Deva"]
    right = _write(tmp_path / "b.json", other)

    # Act / Assert
    with pytest.raises(IncomparableError) as excinfo:
        compare([left, right], metric="cpt")
    assert excinfo.value.field == "languages"


def test_parity_is_read_from_the_corpus_level_block(tmp_path: Path) -> None:
    # §9 publishes per-language parity inside corpus_level.parity, not beside
    # the per-language compression numbers.
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    right = _write(tmp_path / "b.json", _other_tokenizer(_document()))

    # Act
    table = compare([left, right], metric="parity")

    # Assert
    assert table.rows["hin_Deva"] == (1.75, 1.75)


def test_a_corpus_level_metric_tables_a_single_row(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    right = _write(tmp_path / "b.json", _other_tokenizer(_document()))

    # Act
    table = compare([left, right], metric="gini")

    # Assert
    assert tuple(table.rows) == ("corpus",)
    assert table.rows["corpus"] == (0.13636363636363635, 0.13636363636363635)


def test_an_unknown_metric_names_the_ones_that_exist(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _document())
    right = _write(tmp_path / "b.json", _other_tokenizer(_document()))

    # Act / Assert
    with pytest.raises(ValueError, match="fertility") as excinfo:
        compare([left, right], metric="perplexity")
    assert "perplexity" in str(excinfo.value)


def _lint_document() -> dict[str, Any]:
    """What `glotscope lint` writes: tier0, and no corpus at all."""
    document = _document()
    del document["tier1"]
    del document["manifest"]["corpus"]
    return document


def test_two_lint_documents_table_a_tier0_metric(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _lint_document())
    right = _write(tmp_path / "b.json", _other_tokenizer(_lint_document()))

    # Act
    table = compare([left, right], metric="ill_formed_vocab_rate")

    # Assert
    assert table.rows == {"vocab": (0.5, 0.5)}


def test_a_tier0_metric_needs_no_corpus_agreement(tmp_path: Path) -> None:
    # Tier 0 is a property of the tokenizer artifact alone. Refusing to compare
    # two vocabularies because they were later run against different corpora
    # would be over-refusal, and an over-refusing tool gets routed around.
    # Arrange
    left = _write(tmp_path / "a.json", _lint_document())
    other = _other_tokenizer(_document())
    other["manifest"]["corpus"]["languages"] = ["eng_Latn"]
    right = _write(tmp_path / "b.json", other)

    # Act
    table = compare([left, right], metric="unreachable_count")

    # Assert
    assert table.rows == {"vocab": (131, 131)}


def test_a_tier0_only_document_is_refused_by_name(tmp_path: Path) -> None:
    # What `glotscope lint` writes. It has no corpus metric to compare, and the
    # message has to say which command produces one.
    # Arrange
    document = _document()
    del document["tier1"]
    left = _write(tmp_path / "a.json", document)
    right = _write(tmp_path / "b.json", _other_tokenizer(_document()))

    # Act / Assert
    with pytest.raises(ValueError, match="analyze") as excinfo:
        compare([left, right], metric="cpt")
    assert "local@49697ba047fd" in str(excinfo.value)


def test_comparing_fewer_than_two_results_refuses(tmp_path: Path) -> None:
    # Arrange
    left = _write(tmp_path / "a.json", _document())

    # Act / Assert
    with pytest.raises(ValueError, match="two"):
        compare([left], metric="cpt")
