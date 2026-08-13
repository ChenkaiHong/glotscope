"""The primary entry point (PRD §8.1).

Wraps ``tokenizers`` and ``tiktoken``. glotscope never trains, modifies or
implements a tokenizer — that is the first non-goal in §3.2, and it is what keeps
the library's claims about other people's tokenizers credible.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from tokenizers import Tokenizer as BackendTokenizer

from glotscope.aggregate import DocumentStats, aggregate_documents, aggregate_words
from glotscope.compression import compression
from glotscope.corpus import Corpus, LoadedCorpus
from glotscope.enums import Algorithm, Capability, Normalization, Segmenter
from glotscope.errors import TokenizerLoadError
from glotscope.fertility import fertility
from glotscope.lint import detect_algorithm, lint_backend
from glotscope.manifest import ParameterManifest, TokenizerManifest
from glotscope.report import Tier0Report, Tier1Report, Tier2Report
from glotscope.results import CorpusMetrics, FertilityResult, LanguageMetrics
from glotscope.roundtrip import roundtrip_matches_from_ids
from glotscope.segmenters import get_segmenter

if TYPE_CHECKING:
    from glotscope.embeddings import Embeddings

__all__ = ["Tokenizer"]

_LOCAL_REVISION = "local"
"""Recorded where an upstream revision does not exist. A local file has no commit
to pin, and inventing one would put an unverifiable string in the field
``glotscope verify`` trusts."""

_UNKNOWN_LICENSE = "UNKNOWN"

_NO_REVISION_WARNING = (
    "local tokenizer.json: no upstream revision exists, so revision is recorded "
    "as 'local' and the artifact's identity rests on tokenizer_json_sha256"
)
_NO_CONFIG_WARNING = (
    "local tokenizer.json: vocab_size_config and embedding_rows are unknown and "
    "recorded as null; load the checkpoint to fill them"
)
_UNKNOWN_ALGORITHM_WARNING = (
    "algorithm could not be identified from tokenizer.json and is recorded as "
    "'unknown'; §7.9 is validated on BPE only"
)
_NO_NORMALIZATION_WARNING = (
    "normalization is switched off: CPT, STRR and round-trip losslessness are "
    "all sensitive to Unicode form, so these numbers are comparable only against "
    "another run over identically encoded text"
)

_ANALYSIS_BATCH_SIZE = 8192
"""Maximum documents whose normalized text and encodings are live together."""

_UNICODE_FORMS: dict[Normalization, Literal["NFC", "NFD", "NFKC", "NFKD"]] = {
    Normalization.NFC: "NFC",
    Normalization.NFD: "NFD",
    Normalization.NFKC: "NFKC",
    Normalization.NFKD: "NFKD",
}


def _normalized(text: str, form: Normalization) -> str:
    """Apply the recorded Unicode form; :attr:`Normalization.NONE` passes through."""
    unicode_form = _UNICODE_FORMS.get(form)
    if unicode_form is None:
        return text
    return unicodedata.normalize(unicode_form, text)


def _empty_document_warning(language: str, count: int) -> str:
    return (
        f"{language}: {count} document(s) encoded to zero tokens. They are counted "
        f"in the corpus totals and excluded from the compression rate, never "
        f"silently dropped — normalization alone can strip a document to nothing "
        f"(§12.2)"
    )


def _require_loaded(corpus: LoadedCorpus | Corpus) -> LoadedCorpus:
    """Insist on a corpus that has actually been read (D12).

    glotscope ships no corpora, so a bare :class:`~glotscope.corpus.Corpus` is a
    description of something on disk rather than the thing itself. Refusing here
    also guarantees the digest in the manifest and the numbers in the report
    describe the same bytes.
    """
    if isinstance(corpus, LoadedCorpus):
        return corpus
    raise ValueError(
        f"corpus {corpus.spec.id!r} carries no text. glotscope ships no corpora "
        f"(D12): call corpus.load(root) on a local download — the recipe is on "
        f"its registry entry — and pass what that returns."
    )


def _check_segmenter_capability(segmenter: Segmenter | None, corpus: Corpus) -> None:
    """Refuse ``UD_GOLD`` on a corpus without gold word boundaries (§7.1 rule 1).

    Permanent, and independent of what is implemented: UD supplies gold
    boundaries only for sentences inside the treebanks, and applying "UD
    segmentation" to arbitrary text is a different operation performed by a
    trained model with its own per-language accuracy and its own version.
    """
    if segmenter is not None and segmenter.requires_gold_segmentation:
        corpus.require(Capability.WORD_SEGMENTATION, "fertility")


def _unk_token_id(backend: BackendTokenizer, spec: Mapping[str, Any]) -> int | None:
    """The id of the ``[UNK]`` token, or ``None`` where the model has none.

    Byte-level BPE has no UNK by construction — every byte is representable — so
    ``None`` here means the >10% exclusion rule cannot fire, not that it passed.
    """
    unk_token = spec.get("model", {}).get("unk_token")
    if not isinstance(unk_token, str):
        return None
    return backend.token_to_id(unk_token)


def _unk_char_count(encodings: Sequence[Any], unk_id: int | None) -> int:
    """Characters covered by ``[UNK]`` tokens across a batch (§7.1 rule 6).

    Counted from the encoding's character offsets rather than by counting UNK
    tokens: one UNK can stand for a whole run of unrepresentable text, and the
    rule is stated over *characters* precisely because a token count would
    understate how much of the language was lost.
    """
    if unk_id is None:
        return 0
    return sum(
        end - start
        for encoding in encodings
        for token_id, (start, end) in zip(encoding.ids, encoding.offsets, strict=True)
        if token_id == unk_id
    )


def _word_encodings(
    backend: BackendTokenizer,
    words: Sequence[str],
    *,
    leading_space: bool,
    add_special_tokens: bool,
) -> list[list[int]]:
    """Encode each word on its own, applying the leading-space convention.

    ``tau("the")`` and ``tau(" the")`` differ for byte-level BPE, often by a
    whole token, so the convention is applied here — once, visibly — and
    recorded in the manifest rather than left to whichever call site got there
    first (§7.1 rule 5).
    """
    texts = [f" {word}" if leading_space else word for word in words]
    return [
        encoding.ids
        for encoding in backend.encode_batch(texts, add_special_tokens=add_special_tokens)
    ]


class Tokenizer:
    """A loaded tokenizer plus the provenance needed to cite its numbers.

    Construct through one of the ``from_*`` classmethods rather than directly:
    each one records where the tokenizer came from, which is what the manifest
    needs and what no competing tool captures.
    """

    _backend: Any
    _spec: Mapping[str, Any]
    _manifest: TokenizerManifest
    _warnings: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError(
            "construct a Tokenizer through from_pretrained(), from_tiktoken() or "
            "from_file(); direct construction would skip provenance capture (G4)"
        )

    @classmethod
    def _loaded(
        cls,
        *,
        backend: Any,
        spec: Mapping[str, Any],
        manifest: TokenizerManifest,
        warnings: tuple[str, ...],
    ) -> Tokenizer:
        """Assemble an instance around an already-captured provenance record.

        Bypasses :meth:`__init__` deliberately: the refusal there is what keeps
        construction on the paths that capture provenance.
        """
        tokenizer = object.__new__(cls)
        tokenizer._backend = backend
        tokenizer._spec = spec
        tokenizer._manifest = manifest
        tokenizer._warnings = warnings
        return tokenizer

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        is_mirror: bool = False,
    ) -> Tokenizer:
        """Load from the Hugging Face Hub.

        ``revision`` accepts a commit SHA. Passing ``None`` resolves the default
        branch and records the SHA it resolved to, plus a warning that the caller
        did not pin — the manifest stays honest either way, but an unpinned run is
        not a reproducible one, and silent upstream tokenizer changes are exactly
        what the nightly CI job in §12.4 exists to catch.

        ``is_mirror`` marks third-party re-uploads such as the ungated Llama
        mirrors. Mirror-sourced rows must be visibly labelled in the leaderboard
        (§11): a leaderboard that silently uses unofficial mirrors is a legitimate
        line of attack.
        """
        raise NotImplementedError

    @classmethod
    def from_tiktoken(cls, encoding: str) -> Tokenizer:
        """Load an OpenAI encoding by name.

        The complete registry as of the PRD's verification pass: ``o200k_base``,
        ``o200k_harmony``, ``cl100k_base``, ``r50k_base`` and the older encodings.
        No OpenAI encoding newer than ``o200k_base``/``o200k_harmony`` exists —
        GPT-5 and the o-series all map to ``o200k_base``.

        tiktoken encodings support Tier 0 and Tier 1 only: they are tokenizer-only,
        with no weights, so Tier 2 is unavailable. Roughly half the §11 core tier
        is in this position, which is why the leaderboard must mark the Tier 2
        column ``n/a (tokenizer-only)`` rather than leaving it visually empty.

        ``tiktoken`` is the ``tiktoken`` extra rather than a core dependency.
        Import it inside this method and name the extra when it is missing: an
        unguarded top-level import would make every core install carry a second
        tokenizer library in order to reach ``from_file``.
        """
        raise NotImplementedError

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        tokenizer_id: str | None = None,
        license_spdx: str = _UNKNOWN_LICENSE,
    ) -> Tokenizer:
        """Load a local ``tokenizer.json``.

        The file's bytes are read once and used for all three of the parsed spec,
        the backend tokenizer and ``tokenizer_json_sha256``, so the hash in the
        manifest is provably the hash of what was analysed rather than of whatever
        the path pointed at a moment later.

        ``tokenizer_id`` names the artifact in the manifest and defaults to
        ``"local"`` rather than the filename: §9 manifests carry no filesystem
        paths, because a path is a property of one machine and would break the
        bit-identical regeneration ``glotscope verify`` asserts.

        Raises:
            TokenizerLoadError: if the file cannot be read, or is not a valid
                ``tokenizer.json``.
        """
        source = Path(path)
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise TokenizerLoadError(source.name, str(exc)) from exc

        try:
            text = raw.decode("utf-8")
            spec: Mapping[str, Any] = json.loads(text)
            backend = BackendTokenizer.from_str(text)
        except Exception as exc:
            # The backend raises its own Rust-side exception type for a malformed
            # document, so this catch is broad on purpose; the alternative is
            # letting an untyped foreign exception escape the public API.
            raise TokenizerLoadError(source.name, f"not a valid tokenizer.json ({exc})") from exc

        algorithm = detect_algorithm(spec)
        warnings = [_NO_REVISION_WARNING, _NO_CONFIG_WARNING]
        if algorithm is Algorithm.UNKNOWN:
            warnings.append(_UNKNOWN_ALGORITHM_WARNING)

        manifest = TokenizerManifest(
            id=tokenizer_id or _LOCAL_REVISION,
            revision=_LOCAL_REVISION,
            tokenizer_json_sha256=hashlib.sha256(raw).hexdigest(),
            vocab_size_tokenizer=backend.get_vocab_size(with_added_tokens=True),
            vocab_size_config=None,
            embedding_rows=None,
            algorithm=algorithm,
            source="file",
            source_is_mirror=False,
            license_spdx=license_spdx,
        )
        return cls._loaded(backend=backend, spec=spec, manifest=manifest, warnings=tuple(warnings))

    # -- provenance ---------------------------------------------------------

    @property
    def manifest(self) -> TokenizerManifest:
        """Provenance for this tokenizer, ready to embed in a result (§9)."""
        return self._manifest

    @property
    def warnings(self) -> tuple[str, ...]:
        """Contested or incomplete provenance, for the §9 ``warnings`` array.

        Load-bearing, not decorative: a null ``embedding_rows`` is invisible in a
        table of numbers, and a reader needs the difference between "no padding
        rows" and "nobody looked".
        """
        return self._warnings

    # -- tiers --------------------------------------------------------------

    def lint(self) -> Tier0Report:
        """Tier 0: vocabulary introspection (PRD §7.8).

        Always available and costs milliseconds. Needs no corpus and no weights.
        """
        return lint_backend(self._backend, self._spec)

    def analyze(
        self,
        corpus: LoadedCorpus | Corpus,
        *,
        leading_space: bool = True,
        normalization: Normalization | str = Normalization.NFC,
        add_special_tokens: bool = False,
        segmenter: Segmenter | None = None,
        segmenter_model_version: str | None = None,
    ) -> Tier1Report:
        """Tier 1: corpus-based metrics (PRD §7.1-§7.7, §8.1).

        Encodes each language once, folds the batch across the §13 aggregation
        boundary, and returns the per-language compression family and round-trip
        rate together with the folded statistics themselves. The corpus-level
        metrics are deliberately *not* computed here: §8.1 spells them
        ``t1.parity(reference=...)``, ``t1.gini()`` and
        ``t1.renyi_efficiency(alpha=...)``, and alpha in particular is a
        parameter of the question rather than of the run — a report holding one
        finished Renyi number could not answer a second alpha without
        re-encoding the corpus.

        ``segmenter`` defaults to ``None``, which is legal: parity, Gini, Renyi and
        the compression family are segmenter-free, and §14.3 step 1 runs parity
        over all 229 FLORES+ varieties precisely because it costs nothing.
        Word-level metrics then raise :class:`SegmenterRequiredError` on access
        rather than silently picking a convention (D6). Passing a segmenter
        currently raises :class:`NotImplementedError`, because the adapters ship
        as optional extras that do not exist yet and recording a segmenter that
        never ran would put a false claim in the manifest.

        ``leading_space`` defaults to ``True`` for byte-level BPE per §7.1 rule 5,
        and is recorded either way. It can move STRR by tens of points.

        ``normalization`` is recorded because NFKC versus NFC shifts CPT, STRR and
        round-trip losslessness. It is applied *before* anything is measured, and
        round-trip is scored against the normalized text — comparing a normalized
        encoding against unnormalized source would report a failure caused by the
        pipeline rather than by the tokenizer.

        Args:
            corpus: a :class:`~glotscope.corpus.LoadedCorpus`. A bare
                :class:`~glotscope.corpus.Corpus` is refused: glotscope ships no
                corpora (D12), so an unloaded one carries no text.

        Raises:
            CapabilityError: if ``segmenter`` is
                :attr:`~glotscope.enums.Segmenter.UD_GOLD` and the corpus does not
                declare gold word segmentation. Applying "UD segmentation" to
                arbitrary text requires a trained model with its own per-language
                accuracy and its own version, which is a different operation.
            NotImplementedError: if any segmenter is requested.
            ValueError: if the corpus was never loaded, resolved to no languages,
                or holds no documents for a language it names.
        """
        normalization = Normalization(normalization)
        loaded = _require_loaded(corpus)
        _check_segmenter_capability(segmenter, loaded.corpus)
        if not loaded.lines:
            raise ValueError(
                f"corpus {loaded.corpus.spec.id!r} resolved to no languages, so "
                f"there is nothing to measure"
            )

        warnings: list[str] = []
        if normalization is Normalization.NONE:
            warnings.append(_NO_NORMALIZATION_WARNING)

        per_language: dict[str, LanguageMetrics] = {}
        document_stats: dict[str, DocumentStats] = {}
        unk_id = _unk_token_id(self._backend, self._spec)
        for language, documents in loaded.lines.items():
            if not documents:
                raise ValueError(
                    f"corpus {loaded.corpus.spec.id!r} holds no documents for "
                    f"{language!r}, and every Tier 1 ratio divides by a corpus total"
                )
            chunks: list[DocumentStats] = []
            byte_lengths: list[int] = []
            is_blank: list[bool] = []
            roundtrip_matches = 0
            unk_chars = 0
            for start in range(0, len(documents), _ANALYSIS_BATCH_SIZE):
                texts = [
                    _normalized(document, normalization)
                    for document in documents[start : start + _ANALYSIS_BATCH_SIZE]
                ]
                chunk_bytes = [len(text.encode("utf-8")) for text in texts]
                encodings = self._backend.encode_batch(texts, add_special_tokens=add_special_tokens)
                token_ids = [encoding.ids for encoding in encodings]
                unk_chars += _unk_char_count(encodings, unk_id)
                chunks.append(
                    aggregate_documents(
                        token_ids,
                        char_lengths=[len(text) for text in texts],
                        byte_lengths=chunk_bytes,
                    )
                )
                byte_lengths.extend(chunk_bytes)
                is_blank.extend(not text.strip() for text in texts)
                matched, _ = roundtrip_matches_from_ids(self._backend, texts, token_ids)
                roundtrip_matches += matched
            stats = DocumentStats.combine(chunks)
            if stats.n_empty_documents:
                warnings.append(_empty_document_warning(language, stats.n_empty_documents))
            document_stats[language] = stats
            words: FertilityResult | None = None
            if segmenter is not None:
                adapter = get_segmenter(segmenter, language=language)
                segmented = [
                    word
                    for document in documents
                    for word in adapter.segment(_normalized(document, normalization))
                ]
                words = fertility(
                    aggregate_words(
                        _word_encodings(
                            self._backend,
                            segmented,
                            leading_space=leading_space,
                            # Never on a word: a special token per word would be
                            # counted as part of the word's tokenization and
                            # inflate fertility by one for every word.
                            add_special_tokens=False,
                        )
                    ),
                    language=language,
                    segmenter=segmenter,
                    segmenter_model_version=segmenter_model_version or adapter.model_version,
                    leading_space=leading_space,
                    unk_char_rate=(unk_chars / stats.total_chars if stats.total_chars else 0.0),
                )
            per_language[language] = LanguageMetrics(
                language=language,
                compression=compression(
                    stats, unit_lengths=byte_lengths, is_blank=is_blank, language=language
                ),
                fertility=words,
                roundtrip_rate=roundtrip_matches / stats.n_documents,
            )

        return Tier1Report(
            per_language=MappingProxyType(per_language),
            corpus_level=CorpusMetrics(),
            segmenter=segmenter,
            warnings=tuple(warnings),
            document_stats=MappingProxyType(document_stats),
            corpus=loaded.corpus.to_manifest(),
            parameters=ParameterManifest(
                leading_space=leading_space,
                normalization=normalization,
                add_special_tokens=add_special_tokens,
                segmenter=segmenter,
                segmenter_model_version=segmenter_model_version,
            ),
        )

    def detect_undertrained(
        self,
        embeddings: Embeddings,
        *,
        top_pct: float = 2.0,
        remove_first_pc: bool = False,
    ) -> Tier2Report:
        """Tier 2: under-trained-token indicators (PRD §7.9).

        ``top_pct`` is applied **after** Stage-1 exclusion; both counts are
        recorded, because that ordering is what makes the denominator reproducible.

        For the paper, do not pin to 2%: Land & Bartolo justify that threshold
        solely against their verification budget, which v1.0 does not spend. §14.3
        step 2 requires reporting ``UTR_l`` as a function of threshold, since a
        result that exists only at 2% is an artifact.

        ``remove_first_pc`` is off by default (D9). The source paper's own Table 2
        shows no consistent improvement from centering or PCA-1 removal across
        seven models; their conclusion is that no consistent pattern justifies the
        more complex alternatives.

        Raises:
            NoReferenceSetError: if the reference-set fallback chain is exhausted.
            UnsupportedCheckpointError: if the embeddings are quantized or
                otherwise not in original dtype.
        """
        raise NotImplementedError
