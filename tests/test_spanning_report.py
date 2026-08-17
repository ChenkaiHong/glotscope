"""One document spanning Tier 0, Tier 1 and Tier 2 (PRD §6, G2, M2).

This is the claim §2.2 rests on: no competing tool reports vocabulary integrity,
corpus-based compression *and* weight-based under-training indicators about the
same artifact, and the reason nobody does is that Tier 1 and Tier 2 look like
they need different infrastructure. They do not — Tier 2 reads two tensors.

So the test builds the document the only way that proves the claim: by running
Tier 1 over a real corpus and Tier 2 over real embedding rows, and asserting the
two land in one ``result.json`` beside the Tier 0 lint, with a manifest carrying
both a corpus block and a weights block.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from glotscope import __version__, backend
from glotscope.corpus import Corpus, LoadedCorpus
from glotscope.embeddings import Embeddings
from glotscope.lint import byte_to_unicode
from glotscope.manifest import Manifest, canonical_json, environment
from glotscope.report import Report
from glotscope.tokenizer import Tokenizer

_ENGLISH = ("The cat sat.", "It rained.")
_HINDI = ("बिल्ली बैठी।", "बारिश हुई।")


def _tokenizer(tmp_path: Path) -> Tokenizer:
    mapping = byte_to_unicode()
    engine = BackendTokenizer(models.BPE(vocab={mapping[v]: v for v in range(256)}, merges=[]))
    engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    engine.decoder = decoders.ByteLevel()
    path = tmp_path / "tokenizer.json"
    engine.save(str(path))
    return Tokenizer.from_file(path)


def _corpus(root: Path) -> LoadedCorpus:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    directory = root / corpus.spec.id / corpus.version / corpus.split
    directory.mkdir(parents=True, exist_ok=True)
    for language, lines in (("eng_Latn", _ENGLISH), ("hin_Deva", _HINDI)):
        (directory / f"{language}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
    return corpus.load(root)


def _embeddings(tokenizer: Tokenizer) -> Embeddings:
    rows = tokenizer.lint().vocab_size
    scale = np.arange(1, rows + 1, dtype=np.float64).reshape(rows, 1)
    return Embeddings(
        e_in=scale * np.full((rows, 4), 0.5),
        e_out=None,
        tied=True,
        dtype="bfloat16",
        shard_sha256="a" * 64,
        checkpoint="acme/model",
        n_rows=rows,
        vocab_size=rows,
        license_spdx="apache-2.0",
    )


def _spanning(tmp_path: Path) -> Report:
    tokenizer = _tokenizer(tmp_path)
    tier1 = tokenizer.analyze(_corpus(tmp_path / "corpora"))
    embeddings = _embeddings(tokenizer)
    tier2 = tokenizer.detect_undertrained(embeddings, top_pct=5.0)
    assert tier1.parameters is not None
    assert tier1.corpus is not None
    return Report(
        tier0=tokenizer.lint(),
        tier1=tier1,
        tier2=tier2,
        manifest=Manifest(
            tokenizer=tokenizer.manifest,
            parameters=tier1.parameters,
            environment=environment(),
            backend=backend(),
            glotscope_version=__version__,
            corpus=tier1.corpus,
            weights=embeddings.manifest,
        ),
        warnings=tokenizer.warnings,
    )


def test_one_document_carries_all_three_tiers(tmp_path: Path) -> None:
    # Act
    document = _spanning(tmp_path).to_dict()

    # Assert
    assert {"tier0", "tier1", "tier2"} <= set(document)
    assert {"corpus", "weights"} <= set(document["manifest"])


def test_each_tier_answers_a_question_the_others_cannot(tmp_path: Path) -> None:
    # Not three copies of the same number under different names: Tier 0 needs
    # only the tokenizer, Tier 1 needs a corpus, Tier 2 needs weights, and §6
    # distinguishes them by what they require rather than by what they measure.
    # Act
    document = _spanning(tmp_path).to_dict()

    # Assert
    assert document["tier0"]["byte_fallback_coverage"] == 256
    assert "bpt" in document["tier1"]["per_language"]["eng_Latn"]
    assert document["tier2"]["indicator"] == "cosine_to_unused_mean"


def test_the_warnings_array_aggregates_every_tier(tmp_path: Path) -> None:
    # Load-bearing, not decorative: a reader scanning one array has to see the
    # contested choices from all three tiers, or a Tier 2 caveat is invisible in
    # a document whose headline numbers are Tier 1's.
    # Act
    report = _spanning(tmp_path)
    warnings = report.to_dict()["warnings"]

    # Assert
    assert report.tier2 is not None
    assert set(report.tier2.warnings) <= set(warnings)
    assert any("unused_bytes" in warning for warning in warnings)


def test_the_spanning_document_is_written_deterministically(tmp_path: Path) -> None:
    # G4: `glotscope verify` asserts bit-identical regeneration, so adding a
    # third tier must not add a source of nondeterminism.
    # Arrange
    report = _spanning(tmp_path)
    out = tmp_path / "result.json"

    # Act
    report.to_json(out)

    # Assert
    text = out.read_text(encoding="utf-8")
    assert text == canonical_json(report.to_dict()) + "\n"
    assert json.loads(text)["schema_version"] == report.manifest.schema_version
