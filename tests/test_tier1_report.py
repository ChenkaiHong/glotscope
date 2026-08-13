"""Corpus-level Tier 1 methods (PRD §7.3-§7.5, §8.1).

The report carries the folded per-language statistics, so alpha is a parameter
of the *call* rather than of the run: §8.1 shows ``renyi_efficiency(alpha=2.5)``
chosen at request time, and D17 requires alpha to be recorded on whatever comes
back.
"""

from __future__ import annotations

import pytest

from glotscope.aggregate import DocumentStats, aggregate_documents
from glotscope.enums import Capability, Normalization, RenyiNormalizer, Segmenter
from glotscope.errors import CapabilityError
from glotscope.manifest import CorpusManifest, ParameterManifest
from glotscope.metrics import renyi_efficiency
from glotscope.report import Tier1Report
from glotscope.results import (
    CompressionResult,
    CorpusMetrics,
    GiniResult,
    LanguageMetrics,
    RenyiResult,
    StrrPair,
)


def _stats(encodings: list[list[int]]) -> DocumentStats:
    lengths = [len(encoding) for encoding in encodings]
    return aggregate_documents(encodings, char_lengths=lengths, byte_lengths=lengths)


def _corpus(
    *,
    capabilities: frozenset[Capability] = frozenset({Capability.PARALLEL}),
    corpus_id: str = "flores_plus",
) -> CorpusManifest:
    """The §9 corpus block, which is also what the capability gate reads.

    One field rather than a loose ``corpus_id`` beside a loose capability set:
    claiming a corpus is parallel now requires naming the corpus that said so.
    """
    return CorpusManifest(
        id=corpus_id,
        version="2024.08",
        split="devtest",
        languages=("eng_Latn", "hin_Deva"),
        sha256="a" * 64,
        capabilities=capabilities,
        license="CC-BY-SA-4.0",
    )


def _report(
    *,
    capabilities: frozenset[Capability] = frozenset({Capability.PARALLEL}),
    corpus_id: str = "flores_plus",
) -> Tier1Report:
    document_stats = {
        "eng_Latn": _stats([[1, 2], [3, 4]]),
        "hin_Deva": _stats([[5, 5, 6], [7, 8, 9]]),
    }
    per_language = {
        language: LanguageMetrics(
            language=language,
            compression=CompressionResult(language, 1.0, 1.0, 1, 1.0, "bytes"),
        )
        for language in document_stats
    }
    return Tier1Report(
        per_language=per_language,
        corpus_level=CorpusMetrics(),
        segmenter=None,
        document_stats=document_stats,
        corpus=_corpus(capabilities=capabilities, corpus_id=corpus_id),
    )


def test_parity_is_computed_from_the_stored_per_line_token_counts() -> None:
    result = _report().parity(reference="eng_Latn")

    # eng_Latn totals 4 tokens over 2 lines, hin_Deva 6 over the same 2 lines.
    assert result.per_language["eng_Latn"] == pytest.approx(1.0)
    assert result.per_language["hin_Deva"] == pytest.approx(1.5)
    assert result.reference_language == "eng_Latn"


def test_parity_refuses_a_corpus_that_does_not_declare_itself_parallel() -> None:
    # Gating is on declared capability, never on corpus identity (D5). Parity
    # against a reference that does not exist is meaningless, so this is a
    # refusal rather than a missing feature.
    report = _report(capabilities=frozenset(), corpus_id="fineweb2")

    with pytest.raises(CapabilityError, match="parallel") as raised:
        report.parity(reference="eng_Latn")
    assert raised.value.corpus_id == "fineweb2"


def test_gini_costs_are_tokens_per_aligned_line() -> None:
    result = _report().gini()

    assert result.cost_unit == "tokens_per_line"
    assert result.languages == frozenset({"eng_Latn", "hin_Deva"})
    assert 0.0 <= result.value <= 1.0


def test_gini_refuses_a_corpus_that_is_not_parallel() -> None:
    with pytest.raises(CapabilityError, match="parallel"):
        _report(capabilities=frozenset(), corpus_id="fineweb2").gini()


def test_gini_refuses_a_single_language_report() -> None:
    report = _report()
    monolingual = Tier1Report(
        per_language={"eng_Latn": report.per_language["eng_Latn"]},
        corpus_level=CorpusMetrics(),
        segmenter=None,
        document_stats={"eng_Latn": report.document_stats["eng_Latn"]},
        corpus=CorpusManifest(
            id="flores_plus",
            version="2024.08",
            split="devtest",
            languages=("eng_Latn",),
            sha256="a" * 64,
            capabilities=frozenset({Capability.PARALLEL}),
            license="CC-BY-SA-4.0",
        ),
    )

    with pytest.raises(ValueError, match="at least two"):
        monolingual.gini()


def test_renyi_efficiency_is_computed_at_the_alpha_requested() -> None:
    report = _report()

    at_two_five = report.renyi_efficiency(alpha=2.5)
    at_three = report.renyi_efficiency(alpha=3.0)

    # alpha has no default anywhere in this library: the reference
    # implementation silently uses 3.0 and reproduces nobody. Two calls, two
    # numbers, each carrying the alpha it was computed at.
    assert at_two_five.alpha == 2.5
    assert at_three.alpha == 3.0
    assert at_two_five.value != at_three.value


def test_renyi_efficiency_matches_the_token_sequence_implementation() -> None:
    report = _report()
    tokens = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]

    from_report = report.renyi_efficiency(alpha=2.5)
    from_tokens = renyi_efficiency(tokens, alpha=2.5)

    assert from_report.value == pytest.approx(from_tokens.value)
    assert from_report.entropy_bits == pytest.approx(from_tokens.entropy_bits)


def test_renyi_efficiency_records_its_normalizer() -> None:
    report = _report()

    observed = report.renyi_efficiency(alpha=2.5)
    nominal = report.renyi_efficiency(
        alpha=2.5,
        normalizer=RenyiNormalizer.NOMINAL,
        nominal_vocab_size=32000,
    )

    assert observed.normalizer is RenyiNormalizer.OBSERVED
    assert nominal.normalizer is RenyiNormalizer.NOMINAL
    assert nominal.value < observed.value


def test_corpus_level_methods_refuse_a_report_with_no_stored_statistics() -> None:
    report = Tier1Report(per_language={}, corpus_level=CorpusMetrics(), segmenter=None)

    with pytest.raises(ValueError, match="no stored document statistics"):
        report.gini()
    with pytest.raises(ValueError, match="no stored document statistics"):
        report.renyi_efficiency(alpha=2.5)


def test_tier1_serialization_carries_the_segmenter_and_the_compression_family() -> None:
    report = Tier1Report(
        per_language={
            "hin_Deva": LanguageMetrics(
                language="hin_Deva",
                compression=CompressionResult("hin_Deva", 2.88, 3.4, 41, 3.4, "bytes"),
                parity_vs_reference=1.94,
            )
        },
        corpus_level=CorpusMetrics(),
        segmenter=Segmenter.STANZA,
        warnings=("hin_Deva: fertility computed under a UD segmenter",),
    )

    document = report.to_dict()

    assert document["segmenter"] == "stanza"
    assert document["per_language"]["hin_Deva"]["cpt"] == 2.88
    assert document["per_language"]["hin_Deva"]["ctc"] == 41
    assert document["per_language"]["hin_Deva"]["parity_vs_reference"] == 1.94
    # Absent metrics are absent, never zero: None means "not computed", and a
    # zero would tabulate as a real measurement.
    assert "fertility" not in document["per_language"]["hin_Deva"]


def test_tier1_serialization_carries_every_optional_metric_that_ran() -> None:
    report = Tier1Report(
        per_language={
            "hin_Deva": LanguageMetrics(
                language="hin_Deva",
                compression=CompressionResult("hin_Deva", 2.88, 3.4, 41, 3.4, "bytes"),
                strr=StrrPair("hin_Deva", 12.4, 31.7, lowercased=False, n_words=5000),
                roundtrip_rate=0.999,
            )
        },
        corpus_level=CorpusMetrics(
            gini=GiniResult(0.081, frozenset({"hin_Deva"}), "tokens_per_line"),
            renyi=RenyiResult(0.412, 2.5, RenyiNormalizer.OBSERVED, 9.1),
            parity=_report().parity(reference="eng_Latn"),
        ),
        segmenter=Segmenter.STANZA,
    )

    document = report.to_dict()

    # Both STRR conventions or neither: §7.6 refuses to emit one unqualified
    # number, and a document carrying only one would be exactly that.
    assert document["per_language"]["hin_Deva"]["strr_bare"] == 12.4
    assert document["per_language"]["hin_Deva"]["strr_leading_space"] == 31.7
    assert document["per_language"]["hin_Deva"]["roundtrip_rate"] == 0.999
    # alpha and the normalizer travel with the Renyi value, never without it.
    assert document["corpus_level"]["renyi_efficiency"] == 0.412
    assert document["corpus_level"]["renyi_alpha"] == 2.5
    assert document["corpus_level"]["renyi_normalizer"] == "observed"
    assert document["corpus_level"]["gini"] == 0.081
    assert document["corpus_level"]["parity"]["reference_language"] == "eng_Latn"
    assert document["corpus_level"]["parity"]["worst_case_parity"] == 1.5


def test_gini_refuses_a_language_set_with_no_aligned_lines() -> None:
    report = Tier1Report(
        per_language={},
        corpus_level=CorpusMetrics(),
        segmenter=None,
        document_stats={"eng_Latn": _stats([]), "hin_Deva": _stats([])},
        corpus=CorpusManifest(
            id="flores_plus",
            version="2024.08",
            split="devtest",
            languages=("eng_Latn", "hin_Deva"),
            sha256="a" * 64,
            capabilities=frozenset({Capability.PARALLEL}),
            license="CC-BY-SA-4.0",
        ),
    )

    with pytest.raises(ValueError, match="at least one line"):
        report.gini()


def test_a_report_refuses_parameters_that_disagree_with_its_segmenter() -> None:
    # The segmenter appears twice by necessity: on the report because it is the
    # enforcement point for the fertility refusal, and in the §9 parameter block
    # because that is what gets published. Two copies of one decision drift, so
    # disagreement is a construction error rather than something a reader has to
    # notice.
    with pytest.raises(ValueError, match="segmenter"):
        Tier1Report(
            per_language={},
            corpus_level=CorpusMetrics(),
            segmenter=None,
            parameters=ParameterManifest(
                leading_space=True,
                normalization=Normalization.NFC,
                add_special_tokens=False,
                segmenter=Segmenter.STANZA,
            ),
        )


def test_tier1_serialization_omits_a_segmenter_that_was_never_chosen() -> None:
    report = Tier1Report(
        per_language={
            "eng_Latn": LanguageMetrics(
                language="eng_Latn",
                compression=CompressionResult("eng_Latn", 4.0, 4.2, 10, 4.2, "bytes"),
            )
        },
        corpus_level=CorpusMetrics(),
        segmenter=None,
    )

    document = report.to_dict()

    assert document["segmenter"] is None
