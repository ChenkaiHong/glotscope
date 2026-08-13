"""Shared plumbing for the segmenter adapters (PRD §7.1, §10.3).

Three things every adapter needs and none should reimplement: resolving a
language code to the part that identifies the language, refusing a third-party
package that is not installed, and reading the version of whatever actually
produced the boundaries.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Iterable
from types import ModuleType

from glotscope.enums import Segmenter
from glotscope.errors import SegmenterScopeError, SegmenterUnavailableError

__all__ = ["import_or_refuse", "language_prefix", "package_version", "require_scope"]


def language_prefix(language: str) -> str:
    """The language part of a corpus code, lowercased.

    FLORES+ codes are ``<iso639-3>_<script>`` — ``jpn_Jpan``, ``zho_Hans`` — and
    other corpora use bare two-letter codes. Both are accepted, which is why the
    scoped adapters list ISO 639-1 and 639-3 spellings side by side rather than
    normalizing to one: mapping between them needs a table this library has no
    reason to carry.
    """
    return language.split("_", 1)[0].lower()


def require_scope(segmenter: Segmenter, language: str, supported: Iterable[str]) -> None:
    """Refuse a language-scoped segmenter used outside its language (§10.3).

    Raises:
        SegmenterScopeError: naming the language asked for and the ones this
            segmenter is built for.
    """
    allowed = frozenset(supported)
    if language_prefix(language) not in allowed:
        raise SegmenterScopeError(segmenter.value, language, allowed)


def import_or_refuse(segmenter: Segmenter, module: str, package: str) -> ModuleType:
    """Import an optional segmenter dependency, or refuse.

    The import happens here rather than at module scope so that a core install
    never pays for an extra it did not ask for: importing
    :mod:`glotscope.segmenters` must not require jieba, MeCab or system ICU.

    Raises:
        SegmenterUnavailableError: naming the package and the pip extra. Never
            falls back to another segmenter (D6).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise SegmenterUnavailableError(segmenter.value, package) from exc


def package_version(distribution: str) -> str:
    """The installed version of a distribution, for the manifest.

    §7.1 requires the *model* version rather than a treebank release. Where a
    segmenter has a model or dictionary separable from its code — MeCab's
    UniDic, ICU's data — the adapter reports that instead; where it has none,
    the adapter says so in the string rather than passing a package version off
    as a model version.
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"
