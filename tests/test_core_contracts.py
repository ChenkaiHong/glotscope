from __future__ import annotations

import pytest

import glotscope
from glotscope.corpus import REGISTRY, Corpus
from glotscope.enums import Backend, Capability, MorphologicalType, Segmenter, TypologicalScope
from glotscope.errors import (
    CapabilityError,
    IncomparableError,
    NoReferenceSetError,
    SegmenterRequiredError,
    UnsupportedCheckpointError,
)
from glotscope.tokenizer import Tokenizer


def test_public_surface_exports_the_primary_contract_types() -> None:
    assert glotscope.Corpus is Corpus
    assert glotscope.Tokenizer is Tokenizer
    assert {"Corpus", "Tokenizer", "backend", "__version__"} <= set(glotscope.__all__)


def test_backend_defaults_to_python_when_core_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLOTSCOPE_IMPLEMENTATION", raising=False)
    monkeypatch.setattr(glotscope, "_core_available", lambda: False)

    assert glotscope.backend() is Backend.PYTHON


def test_backend_explicit_rust_refuses_silent_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLOTSCOPE_IMPLEMENTATION", "rust")
    monkeypatch.setattr(glotscope, "_core_available", lambda: False)

    with pytest.raises(ImportError, match="Refusing to fall back to Python silently"):
        glotscope.backend()


def test_backend_rejects_unknown_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLOTSCOPE_IMPLEMENTATION", "wasm")

    with pytest.raises(ValueError, match="not a valid backend"):
        glotscope.backend()


def test_corpus_registry_preserves_capabilities_and_generator_languages() -> None:
    corpus = Corpus.from_registry(
        "flores_plus",
        (language for language in ("eng_Latn", "hin_Deva")),
        split="devtest",
        version="v1",
        sha256="abc",
    )

    assert corpus.languages == ("eng_Latn", "hin_Deva")
    assert corpus.has(Capability.PARALLEL)
    assert corpus.capabilities == REGISTRY["flores_plus"].capabilities
    corpus.require(Capability.PARALLEL, "parity")


def test_corpus_require_refuses_a_metric_without_its_capability() -> None:
    corpus = Corpus.from_registry(
        "fineweb2", ["eng_Latn"], split="train", version="v1", sha256="def"
    )

    with pytest.raises(CapabilityError) as raised:
        corpus.require(Capability.PARALLEL, "parity")

    assert raised.value.metric == "parity"
    assert raised.value.required == "parallel"
    assert raised.value.corpus_id == "fineweb2"
    assert raised.value.available == ()


def test_corpus_rejects_an_unregistered_identifier() -> None:
    with pytest.raises(KeyError, match="unknown corpus 'unknown'"):
        Corpus.from_registry("unknown", [], split="train", version="v1", sha256="not-a-real-hash")


def test_segmentation_and_morphology_scope_enums_enforce_their_contracts() -> None:
    assert Segmenter.UD_GOLD.requires_gold_segmentation
    assert not Segmenter.STANZA.requires_gold_segmentation
    assert TypologicalScope.for_type(MorphologicalType.FUSIONAL) is TypologicalScope.IN_SCOPE
    assert TypologicalScope.for_type(MorphologicalType.AGGLUTINATIVE) is TypologicalScope.IN_SCOPE
    assert (
        TypologicalScope.for_type(MorphologicalType.NON_CONCATENATIVE)
        is TypologicalScope.OUT_OF_SCOPE
    )
    assert TypologicalScope.for_type(MorphologicalType.ISOLATING) is TypologicalScope.OUT_OF_SCOPE


def test_tokenizer_direct_construction_is_refused() -> None:
    with pytest.raises(
        TypeError, match=r"through from_pretrained\(\), from_tiktoken\(\) or from_file\(\)"
    ):
        Tokenizer()


def test_domain_errors_retain_actionable_context() -> None:
    incomparable = IncomparableError("alpha", 2.5, 3.0)
    no_reference = NoReferenceSetError("model", tied=False)
    segmenter = SegmenterRequiredError("fertility")
    checkpoint = UnsupportedCheckpointError("model", "quantized", dtype="int4")

    assert (incomparable.field, incomparable.left, incomparable.right) == ("alpha", 2.5, 3.0)
    assert "Recompute one side" in str(incomparable)
    assert no_reference.tied is False
    assert "L2(E_in)" in str(no_reference)
    assert segmenter.metric == "fertility"
    assert "no default" in str(segmenter)
    assert checkpoint.dtype == "int4"
    assert "original-dtype embedding tensors" in str(checkpoint)
