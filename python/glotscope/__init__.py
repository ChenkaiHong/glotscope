"""glotscope — multilingual tokenizer diagnostics across four capability tiers.

**What these metrics do and do not tell you.** glotscope reports diagnostics, not
quality predictions. The literature does not support the claim that any metric
here predicts model quality: compression correlates -0.71 to -0.996 with quality
when only training-corpus size varies but +0.241 when the algorithm varies across
54 models; Renyi efficiency can be provably raised while BLEU falls; and
morphological alignment shows no significant correlation with perplexity. Treat
every number here as descriptive.

Public surface (PRD §8.1)::

    import glotscope as gs

    tok  = gs.Tokenizer.from_pretrained("Qwen/Qwen3-8B", revision="<sha>")
    lint = tok.lint()                                    # Tier 0, milliseconds
    t1   = tok.analyze(corpus, segmenter=gs.Segmenter.STANZA)   # Tier 1
    emb  = gs.Embeddings.from_checkpoint("Qwen/Qwen3-8B", revision="<sha>")
    t2   = tok.detect_undertrained(emb, top_pct=2.0)     # Tier 2

    gs.Report(tier0=lint, tier1=t1, tier2=t2).to_json("result.json")
"""

from __future__ import annotations

import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version

from glotscope.aggregate import BoundaryCounts, DocumentStats, WordStats
from glotscope.corpus import Corpus
from glotscope.embeddings import Embeddings
from glotscope.enums import (
    Algorithm,
    Backend,
    Capability,
    Confidence,
    Indicator,
    MorphologicalType,
    Normalization,
    RenyiNormalizer,
    Segmenter,
    TokenClass,
    TokenizerFamily,
    TypologicalScope,
)
from glotscope.errors import (
    CapabilityError,
    GlotscopeError,
    IncomparableError,
    NoReferenceSetError,
    SegmenterRequiredError,
    TokenizerLoadError,
    UnsupportedCheckpointError,
)
from glotscope.manifest import Manifest
from glotscope.report import Report, Tier0Report, Tier1Report, Tier2Report, TokenCandidate
from glotscope.results import (
    CompressionResult,
    FertilityResult,
    GiniResult,
    MorphologyResult,
    ParityResult,
    RenyiResult,
    StrrPair,
)
from glotscope.tokenizer import Tokenizer

__all__ = [
    "Algorithm",
    "Backend",
    "BoundaryCounts",
    "Capability",
    "CapabilityError",
    "CompressionResult",
    "Confidence",
    "Corpus",
    "DocumentStats",
    "Embeddings",
    "FertilityResult",
    "GiniResult",
    "GlotscopeError",
    "IncomparableError",
    "Indicator",
    "Manifest",
    "MorphologicalType",
    "MorphologyResult",
    "NoReferenceSetError",
    "Normalization",
    "ParityResult",
    "RenyiNormalizer",
    "RenyiResult",
    "Report",
    "Segmenter",
    "SegmenterRequiredError",
    "StrrPair",
    "Tier0Report",
    "Tier1Report",
    "Tier2Report",
    "TokenCandidate",
    "TokenClass",
    "Tokenizer",
    "TokenizerFamily",
    "TokenizerLoadError",
    "TypologicalScope",
    "UnsupportedCheckpointError",
    "WordStats",
    "__version__",
    "backend",
]

try:
    __version__ = version("glotscope")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

_IMPLEMENTATION_ENV = "GLOTSCOPE_IMPLEMENTATION"
_CORE_MODULE = "glotscope._core"


def _core_available() -> bool:
    """Whether the compiled core is importable, without importing it.

    ``find_spec`` rather than a ``try: import`` on purpose: importing to test
    availability would execute the module, and a ``_core`` that imports but
    misbehaves would then be indistinguishable from one that is absent.
    """
    try:
        return importlib.util.find_spec(_CORE_MODULE) is not None
    except (ImportError, ValueError):
        return False


def backend() -> Backend:
    """Which implementation is active (PRD §13).

    The environment variable is checked **before** any import is attempted, so the
    escape hatch cannot be shadowed by a failed ``try``/``except``. §13 requires
    this ordering and requires this function to be a public, tested predicate —
    every published result records its backend in the manifest, and a shim that
    silently disagrees with the manifest would be the single failure mode capable
    of destroying the project's credibility.

    An explicit request for a backend that is not present raises rather than
    falling back. A silent downgrade would make the v2 backend-parity CI job
    meaningless: it would compare Python against Python and pass.

    Raises:
        ValueError: if ``GLOTSCOPE_IMPLEMENTATION`` is set to an unknown value.
        ImportError: if ``rust`` is requested but the compiled core is absent.
    """
    requested = os.environ.get(_IMPLEMENTATION_ENV)
    if requested is not None:
        try:
            choice = Backend(requested.strip().lower())
        except ValueError as exc:
            valid = sorted(b.value for b in Backend)
            raise ValueError(
                f"{_IMPLEMENTATION_ENV}={requested!r} is not a valid backend; "
                f"expected one of {valid}"
            ) from exc
        if choice is Backend.RUST and not _core_available():
            raise ImportError(
                f"{_IMPLEMENTATION_ENV}=rust was requested but {_CORE_MODULE} is not "
                f"installed. Refusing to fall back to Python silently: a downgrade "
                f"here would make backend-parity testing vacuous."
            )
        return choice
    return Backend.RUST if _core_available() else Backend.PYTHON
