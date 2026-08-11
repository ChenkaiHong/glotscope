"""The primary entry point (PRD §8.1).

Wraps ``tokenizers`` and ``tiktoken``. glotscope never trains, modifies or
implements a tokenizer — that is the first non-goal in §3.2, and it is what keeps
the library's claims about other people's tokenizers credible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tokenizers import Tokenizer as BackendTokenizer

from glotscope.enums import Algorithm, Normalization, Segmenter
from glotscope.errors import TokenizerLoadError
from glotscope.lint import detect_algorithm, lint_backend
from glotscope.manifest import TokenizerManifest
from glotscope.report import Tier0Report, Tier1Report, Tier2Report

if TYPE_CHECKING:
    from glotscope.corpus import Corpus
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
        corpus: Corpus,
        *,
        leading_space: bool = True,
        normalization: Normalization = Normalization.NFC,
        add_special_tokens: bool = False,
        segmenter: Segmenter | None = None,
        segmenter_model_version: str | None = None,
    ) -> Tier1Report:
        """Tier 1: corpus-based metrics (PRD §7.1-§7.7, §8.1).

        ``segmenter`` defaults to ``None``, which is legal: parity, Gini, Renyi and
        the compression family are segmenter-free, and §14.3 step 1 runs parity
        over all 229 FLORES+ varieties precisely because it costs nothing.
        Word-level metrics then raise :class:`SegmenterRequiredError` on access
        rather than silently picking a convention (D6).

        ``leading_space`` defaults to ``True`` for byte-level BPE per §7.1 rule 5,
        and is recorded either way. It can move STRR by tens of points.

        ``normalization`` is recorded because NFKC versus NFC shifts CPT, STRR and
        round-trip losslessness.

        Raises:
            CapabilityError: if ``segmenter`` is
                :attr:`~glotscope.enums.Segmenter.UD_GOLD` and the corpus does not
                declare gold word segmentation. Applying "UD segmentation" to
                arbitrary text requires a trained model with its own per-language
                accuracy and its own version, which is a different operation.
        """
        raise NotImplementedError

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
