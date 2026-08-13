"""Word segmentation adapters, one per :class:`~glotscope.enums.Segmenter` (§7.1, §10.3).

``W(D)`` is the load-bearing choice in fertility and the field silently
disagrees on it, so this package exists to make the choice explicit, recorded,
and impossible to make by accident. Three rules follow, enforced here rather
than documented:

* **No fallback, ever.** A missing extra raises
  :class:`~glotscope.errors.SegmenterUnavailableError`. Substituting whitespace
  would produce a number under a manifest naming a segmenter that never ran.
* **No use outside scope.** MeCab is Japanese, jieba is Chinese, PyThaiNLP is
  Thai, khmer-nltk is Khmer. Each still returns a segmentation for other
  languages, so the check refuses rather than warns.
* **No unpinned model.** Everything but ``WHITESPACE`` reports a version — the
  UniDic dictionary rather than fugashi, ICU's data rather than PyICU — and
  where no model exists separately from the package, the string says so.

Importing this package pulls in no third-party segmenter. Every optional import
happens inside the adapter that needs it, so a core install stays a core
install (G1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from glotscope.enums import Segmenter
from glotscope.errors import SegmenterUnavailableError
from glotscope.segmenters.builtin import load_icu, load_whitespace
from glotscope.segmenters.east_asian import load_jieba, load_mecab
from glotscope.segmenters.southeast_asian import load_khmer_nltk, load_pythainlp

__all__ = ["WordSegmenter", "available", "get_segmenter"]


@runtime_checkable
class WordSegmenter(Protocol):
    """What every adapter provides.

    Deliberately narrow. An adapter returns words and says what produced them;
    anything else — counting, encoding, aggregating — belongs downstream, where
    it is shared by every segmenter rather than reimplemented per language.
    """

    @property
    def segmenter(self) -> Segmenter:
        """The enum member this adapter implements, for the manifest."""

    @property
    def model_version(self) -> str | None:
        """The model or dictionary version, or ``None`` where no model applies."""

    def segment(self, text: str) -> tuple[str, ...]:
        """Split ``text`` into words. Blank runs are dropped; punctuation is not."""


_LOADERS: dict[Segmenter, Callable[[str], WordSegmenter]] = {
    Segmenter.WHITESPACE: load_whitespace,
    Segmenter.ICU: load_icu,
    Segmenter.JIEBA: load_jieba,
    Segmenter.MECAB: load_mecab,
    Segmenter.PYTHAINLP: load_pythainlp,
    Segmenter.KHMER_NLTK: load_khmer_nltk,
}

_PROBE_LANGUAGE: dict[Segmenter, str] = {
    Segmenter.WHITESPACE: "eng_Latn",
    Segmenter.ICU: "eng_Latn",
    Segmenter.JIEBA: "zho_Hans",
    Segmenter.MECAB: "jpn_Jpan",
    Segmenter.PYTHAINLP: "tha_Thai",
    Segmenter.KHMER_NLTK: "khm_Khmr",
}
"""A language each adapter accepts, so :func:`available` can build one without
tripping the scope check it is not asking about."""

_UNBUILT: dict[Segmenter, str] = {
    Segmenter.STANZA: (
        "stanza segmentation needs a pinned model, and how a model is provisioned "
        "and recorded is an open decision: a silent download on first use would "
        "put an unrecorded artifact behind a published number"
    ),
    Segmenter.UDPIPE: (
        "UDPipe segmentation needs a pinned model, and how a model is provisioned "
        "and recorded is an open decision: a silent download on first use would "
        "put an unrecorded artifact behind a published number"
    ),
    Segmenter.UD_GOLD: (
        "UD_GOLD reads gold word boundaries out of a treebank, and the corpus "
        "layer reads plain text one document per line. Gold boundaries need a "
        "CoNLL-U path through Corpus.load first"
    ),
}
"""Scheduled, not refused. These raise ``NotImplementedError`` so a caller is
told the feature is unbuilt rather than sent to install an extra that would not
help."""


def get_segmenter(segmenter: Segmenter, *, language: str) -> WordSegmenter:
    """Build the adapter for ``segmenter``, for ``language``.

    Args:
        segmenter: which convention to apply. There is no default (D6).
        language: the corpus language code. Used for the scope check and, for
            ICU, to pick the locale.

    Raises:
        SegmenterUnavailableError: the optional extra is not installed.
        SegmenterScopeError: this segmenter is not built for this language.
        NotImplementedError: the adapter is scheduled but not written.
    """
    if segmenter in _UNBUILT:
        raise NotImplementedError(f"{segmenter.value}: {_UNBUILT[segmenter]}.")
    return _LOADERS[segmenter](language)


def available(segmenter: Segmenter) -> bool:
    """Whether ``segmenter`` can run in this environment.

    For skip-with-message in tests, and for a caller that wants to check before
    committing to a run. It answers only "is the dependency importable" — a
    scope violation is a refusal rather than an availability question, and an
    unbuilt adapter is unavailable in a sense no install fixes.
    """
    if segmenter in _UNBUILT or segmenter not in _LOADERS:
        return False
    try:
        _LOADERS[segmenter](_PROBE_LANGUAGE[segmenter])
    except SegmenterUnavailableError:
        return False
    return True
