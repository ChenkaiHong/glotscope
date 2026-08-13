from __future__ import annotations

import json
from pathlib import Path

import pytest

from glotscope.enums import (
    Algorithm,
    Backend,
    Capability,
    Confidence,
    Indicator,
    Normalization,
    RenyiNormalizer,
    Segmenter,
    TokenClass,
    TokenizerFamily,
)
from glotscope.errors import SegmenterRequiredError
from glotscope.manifest import (
    CorpusManifest,
    EnvironmentManifest,
    Manifest,
    ParameterManifest,
    TokenizerManifest,
    WarningLog,
    WeightsManifest,
    canonical_json,
    environment,
)
from glotscope.report import Report, Tier0Report, Tier1Report, Tier2Report, TokenCandidate
from glotscope.results import CompressionResult, CorpusMetrics, FertilityResult, LanguageMetrics


def test_the_environment_is_captured_without_machine_specific_detail() -> None:
    import platform as platform_module

    captured = environment()

    # A family string, not platform.platform(): the latter embeds kernel build
    # numbers, so two machines producing identical numbers would produce
    # different manifests and G4's bit-identical assertion would never be green.
    assert captured.platform == captured.platform.lower()
    assert " " not in captured.platform
    assert captured.python == platform_module.python_version()
    assert captured.tokenizers
    # Nothing here may vary between two calls in one process.
    assert captured == environment()


def _manifest(*, with_optional_tiers: bool = False) -> Manifest:
    tokenizer = TokenizerManifest(
        id="acme/tokenizer",
        revision="a" * 40,
        tokenizer_json_sha256="b" * 64,
        vocab_size_tokenizer=10,
        vocab_size_config=12,
        embedding_rows=14,
        algorithm=Algorithm.BYTE_FALLBACK_BPE,
        source="https://example.test/acme/tokenizer",
        source_is_mirror=True,
        license_spdx="Apache-2.0",
    )
    parameters = ParameterManifest(
        leading_space=True,
        normalization=Normalization.NFC,
        add_special_tokens=False,
        segmenter=Segmenter.STANZA,
        segmenter_model_version="1.10.1",
        renyi_alpha=2.5,
        renyi_normalizer=RenyiNormalizer.NOMINAL,
        top_pct=2.0,
        candidates_pre_exclusion=100,
        candidates_post_exclusion=80,
        first_pc_removed=True,
    )
    weights: WeightsManifest | None = None
    corpus: CorpusManifest | None = None
    if with_optional_tiers:
        weights = WeightsManifest(
            shard_sha256="c" * 64,
            dtype="float16",
            tied_embeddings=False,
            license_spdx="Apache-2.0",
        )
        corpus = CorpusManifest(
            id="flores-plus",
            version="2026.1",
            split="dev",
            languages=("eng", "hin"),
            sha256="d" * 64,
            capabilities=frozenset({Capability.PARALLEL, Capability.WORD_SEGMENTATION}),
            license="CC-BY-4.0",
        )
    return Manifest(
        tokenizer=tokenizer,
        parameters=parameters,
        environment=EnvironmentManifest(
            python="3.13.0", tokenizers="0.23.0", platform="linux-x86_64"
        ),
        backend=Backend.PYTHON,
        glotscope_version="0.1.0",
        weights=weights,
        corpus=corpus,
    )


def _tier0() -> Tier0Report:
    return Tier0Report(
        vocab_size=10,
        family=TokenizerFamily.BYTE_FALLBACK,
        ill_formed_vocab_rate=0.2,
        token_classes={
            TokenClass.WELL_FORMED: 8,
            TokenClass.PARTIAL_UTF8: 1,
            TokenClass.ILL_FORMED_NOT_PARTIAL: 1,
        },
        partial_utf8_tokens=(1, 1),
        unreachable_tokens=(2,),
        special_tokens=(2, 3),
        byte_fallback_coverage=256,
        non_utf8_byte_values=(),
    )


def test_manifest_serializes_optional_tiers_and_contested_parameters() -> None:
    document = _manifest(with_optional_tiers=True).to_dict()

    assert document["schema_version"] == "1.1"
    assert document["backend"] == "python"
    assert document["manifest"]["tokenizer"] == {
        "id": "acme/tokenizer",
        "revision": "a" * 40,
        "tokenizer_json_sha256": "b" * 64,
        "vocab_size_tokenizer": 10,
        "vocab_size_config": 12,
        "embedding_rows": 14,
        "algorithm": "byte_fallback_bpe",
        "source": "https://example.test/acme/tokenizer",
        "source_is_mirror": True,
        "license_spdx": "Apache-2.0",
    }
    assert document["manifest"]["corpus"]["capabilities"] == ["parallel", "word_segmentation"]
    assert document["manifest"]["weights"]["tied_embeddings"] is False
    assert document["manifest"]["parameters"] == {
        "leading_space": True,
        "normalization": "NFC",
        "add_special_tokens": False,
        "unk_exclusion_threshold": 0.1,
        "first_pc_removed": True,
        "segmenter": "stanza",
        "segmenter_model_version": "1.10.1",
        "renyi_alpha": 2.5,
        "renyi_normalizer": "nominal",
        "top_pct": 2.0,
        "candidates_pre_exclusion": 100,
        "candidates_post_exclusion": 80,
    }


def test_manifest_omits_absent_tiers_and_optional_parameters() -> None:
    manifest = _manifest()
    document = manifest.to_dict()

    assert set(document["manifest"]) == {"tokenizer", "parameters", "environment"}
    assert set(document["manifest"]["parameters"]) == {
        "leading_space",
        "normalization",
        "add_special_tokens",
        "unk_exclusion_threshold",
        "first_pc_removed",
        "segmenter",
        "segmenter_model_version",
        "renyi_alpha",
        "renyi_normalizer",
        "top_pct",
        "candidates_pre_exclusion",
        "candidates_post_exclusion",
    }


def test_canonical_json_is_sorted_compact_and_preserves_unicode() -> None:
    document = {"z": "東京", "a": {"y": 2, "b": 1}}

    assert canonical_json(document) == '{"a":{"b":1,"y":2},"z":"東京"}'
    assert json.loads(canonical_json(document)) == document


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_refuses_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range"):
        canonical_json({"metric": value})


def test_warning_log_returns_new_values_without_mutating_prior_log() -> None:
    original = WarningLog()
    one_warning = original.with_warning("first")
    all_warnings = one_warning.extended(message for message in ("second", "third"))

    assert original.entries == ()
    assert one_warning.entries == ("first",)
    assert all_warnings.entries == ("first", "second", "third")


def test_tier0_serialization_and_stage1_exclusions_are_precise() -> None:
    tier0 = _tier0()

    assert tier0.unreachable_count == 1
    assert tier0.stage1_exclusions() == frozenset({1, 2, 3})
    assert tier0.to_dict() == {
        "vocab_size": 10,
        "family": "byte_fallback",
        "ill_formed_vocab_rate": 0.2,
        "token_classes": {
            "ill_formed_not_partial": 1,
            "partial_utf8": 1,
            "well_formed": 8,
        },
        "unreachable_count": 1,
        "byte_fallback_coverage": 256,
        "non_utf8_byte_values": [],
    }


def test_tier1_fertility_requires_a_segmenter_and_omits_uncomputed_languages() -> None:
    no_segmenter = Tier1Report(per_language={}, corpus_level=CorpusMetrics(), segmenter=None)
    with pytest.raises(SegmenterRequiredError, match="requires an explicit segmenter") as error:
        _ = no_segmenter.fertility
    assert error.value.metric == "fertility"

    fertility = FertilityResult(
        language="eng",
        fertility=1.25,
        p_continued=0.2,
        segmenter=Segmenter.WHITESPACE,
        segmenter_model_version=None,
        leading_space=False,
        n_zero_length_words=0,
        unk_char_rate=0.0,
    )
    compression = CompressionResult(
        language="eng",
        cpt=3.0,
        bpt=3.0,
        ctc=10,
        compression_rate=0.5,
        compression_rate_unit="chars",
    )
    with_segmenter = Tier1Report(
        per_language={
            "eng": LanguageMetrics(language="eng", compression=compression, fertility=fertility),
            "hin": LanguageMetrics(language="hin", compression=compression),
        },
        corpus_level=CorpusMetrics(),
        segmenter=Segmenter.WHITESPACE,
    )

    assert with_segmenter.fertility == {"eng": 1.25}


def test_report_reconstruction_is_still_unimplemented() -> None:
    # Tier0Report.to_dict is deliberately lossy: it omits the partial-UTF-8,
    # unreachable and special-token id lists, which run to thousands of entries
    # on a real vocabulary. Rebuilding a Report from a document is therefore not
    # possible without a schema change, and pretending otherwise would hand back
    # an object whose Stage-1 exclusion set is silently empty.
    with pytest.raises(NotImplementedError):
        Report.from_json("unused.json")


def test_a_report_serializes_every_tier_that_ran() -> None:
    tier1 = Tier1Report(per_language={}, corpus_level=CorpusMetrics(), segmenter=Segmenter.STANZA)
    report = Report(tier0=_tier0(), manifest=_manifest(), tier1=tier1)

    document = report.to_dict()

    assert document["tier1"]["segmenter"] == "stanza"
    assert document["tier0"]["vocab_size"] == 10
    assert "tier2" not in document


def test_report_serializes_tier0_tier2_warnings_and_deterministic_json(tmp_path: Path) -> None:
    tier2 = Tier2Report(
        candidates=(
            TokenCandidate(
                token_id=7,
                token_repr="<unused>",
                indicator=Indicator.L2_E_IN,
                indicator_value=0.03,
                rank=1,
                token_class=TokenClass.WELL_FORMED,
                script=None,
            ),
        ),
        indicator=Indicator.L2_E_IN,
        tied=False,
        top_pct=2.0,
        candidates_pre_exclusion=100,
        candidates_post_exclusion=80,
        confidence=Confidence.HIGH,
        indicator_agreement=0.95,
        first_pc_removed=True,
        warnings=("tier two",),
    )
    report = Report(tier0=_tier0(), manifest=_manifest(), tier2=tier2, warnings=("report",))
    output = tmp_path / "result.json"

    assert report.to_dict()["warnings"] == ["report", "tier two"]
    assert report.to_dict()["tier2"] == {
        "indicator": "l2_e_in",
        "tied": False,
        "indicator_agreement": 0.95,
        "confidence": "HIGH",
        "candidate_count": 1,
        "top_pct": 2.0,
        "candidates_pre_exclusion": 100,
        "candidates_post_exclusion": 80,
        "first_pc_removed": True,
    }

    report.to_json(output)

    assert output.read_text(encoding="utf-8") == canonical_json(report.to_dict()) + "\n"
