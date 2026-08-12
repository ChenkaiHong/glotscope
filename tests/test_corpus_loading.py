"""Corpus resolution, integrity and licensing (PRD §10.1, §10.4, D12).

glotscope ships no corpora. It ships recipes and checksums, reads what the user
already downloaded, and refuses anything it cannot verify — so these tests are
about the refusals as much as the loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glotscope.corpus import REGISTRY, Corpus
from glotscope.enums import Capability
from glotscope.errors import CapabilityError, CorpusIntegrityError, LicenseError

_ENGLISH = ("The cat sat.", "It rained.")
_HINDI = ("बिल्ली बैठी।", "बारिश हुई।")

_SAMPLE = "sample-2026-08"
"""FineWeb2 has no upstream release tag that describes what was evaluated, so the
caller names the sample. That is why ``version`` is required rather than
defaulted."""


def _write(
    root: Path,
    corpus_id: str,
    version: str,
    split: str,
    **languages: tuple[str, ...],
) -> None:
    directory = root / corpus_id / version / split
    directory.mkdir(parents=True, exist_ok=True)
    for language, lines in languages.items():
        (directory / f"{language}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_flores_plus_resolves_its_capabilities_and_a_pinned_version() -> None:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])

    assert corpus.has(Capability.PARALLEL)
    # No gold word segmentation, which is what makes UD_GOLD illegal here.
    assert not corpus.has(Capability.WORD_SEGMENTATION)
    assert corpus.languages == ("eng_Latn", "hin_Deva")
    assert corpus.split == "devtest"
    assert corpus.version


def test_fineweb2_declares_no_capabilities_so_parity_is_refused() -> None:
    corpus = Corpus.fineweb2(version=_SAMPLE, languages=["hin_Deva"])

    assert corpus.capabilities == frozenset()
    with pytest.raises(CapabilityError, match="parallel"):
        corpus.require(Capability.PARALLEL, "parity")


def test_fineweb2_refuses_to_invent_a_version() -> None:
    # No upstream release tag describes the sample that was actually evaluated,
    # and a defaulted one would put an unverifiable string in a manifest field
    # other people are meant to cite.
    with pytest.raises(TypeError, match="version"):
        Corpus.fineweb2(["eng_Latn"])  # type: ignore[call-arg]


def test_resolving_a_corpus_that_pins_no_release_refuses_rather_than_defaults() -> None:
    # Same rule as the keyword-only version on fineweb2(), enforced at the one
    # place the defaults live: an entry with no pinned release has none to fall
    # back to, and a manufactured string would be unverifiable in a manifest
    # field other people are meant to cite.
    with pytest.raises(ValueError, match="pins no release"):
        Corpus.resolve("fineweb2", ["eng_Latn"])


def test_resolving_fills_the_split_and_version_from_the_registry() -> None:
    corpus = Corpus.resolve("flores_plus", ["eng_Latn"])

    assert corpus.version == REGISTRY["flores_plus"].default_version
    assert corpus.split == "devtest"


def test_universal_dependencies_records_treebanks_not_language_codes() -> None:
    corpus = Corpus.universal_dependencies(["UD_Korean-Kaist", "UD_Korean-GSD"])

    # UD Korean treebanks disagree among themselves — Kaist segments
    # morphologically, GSD by eojeol — so "UD" is not one convention, and the
    # treebank rather than the language is what gets recorded.
    assert corpus.languages == ("UD_Korean-Kaist", "UD_Korean-GSD")
    assert corpus.has(Capability.WORD_SEGMENTATION)
    assert corpus.has(Capability.MORPH_GOLD)


def test_loading_reads_the_downloaded_files_and_records_a_digest(tmp_path: Path) -> None:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    _write(tmp_path, "flores_plus", corpus.version, "devtest", eng_Latn=_ENGLISH, hin_Deva=_HINDI)

    loaded = corpus.load(tmp_path)

    assert loaded.lines["eng_Latn"] == _ENGLISH
    assert loaded.lines["hin_Deva"] == _HINDI
    assert len(loaded.corpus.sha256) == 64
    # Loading returns a new corpus carrying the digest rather than mutating the
    # one that was asked for.
    assert corpus.sha256 == ""


def test_the_digest_is_stable_across_loads_and_sensitive_to_content(tmp_path: Path) -> None:
    corpus = Corpus.flores_plus(["eng_Latn"])
    _write(tmp_path, "flores_plus", corpus.version, "devtest", eng_Latn=_ENGLISH)
    first = corpus.load(tmp_path).corpus.sha256

    assert corpus.load(tmp_path).corpus.sha256 == first

    _write(tmp_path, "flores_plus", corpus.version, "devtest", eng_Latn=("Changed.", "It rained."))
    assert corpus.load(tmp_path).corpus.sha256 != first


def test_a_digest_mismatch_is_refused_rather_than_reported(tmp_path: Path) -> None:
    # A manifest pins the corpus by hash. Loading different bytes under the same
    # identity would make every number downstream unreproducible while looking
    # exactly like a successful run.
    corpus = Corpus.flores_plus(["eng_Latn"], sha256="0" * 64)
    _write(tmp_path, "flores_plus", corpus.version, "devtest", eng_Latn=_ENGLISH)

    with pytest.raises(CorpusIntegrityError, match="does not match"):
        corpus.load(tmp_path)


def test_a_parallel_corpus_with_unequal_line_counts_is_refused(tmp_path: Path) -> None:
    # Parity is a ratio of means and equals the ratio of totals only when the
    # line counts match. Unequal counts break that identity silently.
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    _write(
        tmp_path,
        "flores_plus",
        corpus.version,
        "devtest",
        eng_Latn=_ENGLISH,
        hin_Deva=("बिल्ली बैठी।",),
    )

    with pytest.raises(CorpusIntegrityError, match="unequal line counts"):
        corpus.load(tmp_path)


def test_a_monolingual_corpus_may_have_unequal_line_counts(tmp_path: Path) -> None:
    corpus = Corpus.fineweb2(version=_SAMPLE, languages=["eng_Latn", "hin_Deva"])
    _write(
        tmp_path,
        "fineweb2",
        corpus.version,
        corpus.split,
        eng_Latn=_ENGLISH,
        hin_Deva=("बारिश हुई।",),
    )

    loaded = corpus.load(tmp_path)

    assert len(loaded.lines["eng_Latn"]) == 2
    assert len(loaded.lines["hin_Deva"]) == 1


def test_a_missing_language_file_names_what_was_expected(tmp_path: Path) -> None:
    corpus = Corpus.flores_plus(["eng_Latn", "hin_Deva"])
    _write(tmp_path, "flores_plus", corpus.version, "devtest", eng_Latn=_ENGLISH)

    with pytest.raises(CorpusIntegrityError, match="hin_Deva"):
        corpus.load(tmp_path)


def test_the_commercial_licence_filter_excludes_research_only_corpora(tmp_path: Path) -> None:
    corpus = Corpus.from_registry(
        "europarl",
        ["eng_Latn"],
        split="train",
        version="v7",
        sha256="",
    )
    _write(tmp_path, "europarl", "v7", "train", eng_Latn=_ENGLISH)

    with pytest.raises(LicenseError, match="commercial"):
        corpus.load(tmp_path, license_filter="commercial")


def test_the_commercial_filter_admits_a_permissively_licensed_corpus(tmp_path: Path) -> None:
    corpus = Corpus.fineweb2(version=_SAMPLE, languages=["eng_Latn"])
    _write(tmp_path, "fineweb2", corpus.version, corpus.split, eng_Latn=_ENGLISH)

    assert corpus.load(tmp_path, license_filter="commercial").lines["eng_Latn"] == _ENGLISH


def test_an_unknown_license_filter_is_refused(tmp_path: Path) -> None:
    corpus = Corpus.fineweb2(version=_SAMPLE, languages=["eng_Latn"])
    _write(tmp_path, "fineweb2", corpus.version, corpus.split, eng_Latn=_ENGLISH)

    with pytest.raises(ValueError, match="license_filter"):
        corpus.load(tmp_path, license_filter="permissive")


def test_every_registry_entry_publishes_a_download_recipe() -> None:
    # D12: the library ships no corpora, so a resource nobody can obtain is a
    # resource nobody can reproduce.
    for corpus_id, spec in REGISTRY.items():
        assert spec.recipe, f"{corpus_id} has no download recipe"
