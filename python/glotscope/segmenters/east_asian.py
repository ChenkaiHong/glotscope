"""Chinese and Japanese segmentation (PRD §10.3).

Two of the three no-whitespace languages in the §10.2 core set, and the two the
whitespace segmenter fails hardest on. Both adapters are scoped: outside their
language they still return a segmentation, which is precisely what makes the
scope check a refusal rather than a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from glotscope.enums import Segmenter
from glotscope.segmenters._support import import_or_refuse, package_version, require_scope

__all__ = [
    "CHINESE",
    "JAPANESE",
    "JiebaSegmenter",
    "MecabSegmenter",
    "load_jieba",
    "load_mecab",
]

CHINESE = frozenset({"zho", "zh", "cmn", "yue"})
"""Codes jieba is built for. ``cmn`` and ``yue`` are accepted because FLORES+
distinguishes varieties that jieba does not."""

JAPANESE = frozenset({"jpn", "ja"})


@dataclass(frozen=True, slots=True)
class JiebaSegmenter:
    """jieba, the standard Chinese word segmenter.

    jieba ships one prefix dictionary and no versioned model separable from the
    package, so the recorded version says exactly that rather than dressing a
    package version up as a model version.
    """

    package: str
    segmenter: Segmenter = Segmenter.JIEBA

    @property
    def model_version(self) -> str:
        return f"jieba {self.package} (bundled prefix dictionary, no separate model)"

    def segment(self, text: str) -> tuple[str, ...]:
        jieba = import_or_refuse(Segmenter.JIEBA, "jieba", "jieba")
        return tuple(word for word in jieba.lcut(text) if word.strip())


@dataclass(frozen=True, slots=True)
class MecabSegmenter:
    """MeCab through fugashi, with the UniDic Lite dictionary.

    The dictionary decides the boundaries, so the dictionary version is what
    gets recorded — fugashi is the binding, and its version says nothing about
    where a Japanese word ends.
    """

    dictionary_version: str
    tagger: Any
    segmenter: Segmenter = Segmenter.MECAB

    @property
    def model_version(self) -> str:
        return f"unidic-lite {self.dictionary_version}"

    def segment(self, text: str) -> tuple[str, ...]:
        return tuple(word.surface for word in self.tagger(text) if word.surface.strip())


def load_jieba(language: str) -> JiebaSegmenter:
    """Build the jieba segmenter, refusing any language but Chinese.

    Raises:
        SegmenterScopeError: for a non-Chinese language.
        SegmenterUnavailableError: if jieba is not installed.
    """
    require_scope(Segmenter.JIEBA, language, CHINESE)
    import_or_refuse(Segmenter.JIEBA, "jieba", "jieba")
    return JiebaSegmenter(package=package_version("jieba"))


def load_mecab(language: str) -> MecabSegmenter:
    """Build the MeCab segmenter, refusing any language but Japanese.

    The tagger is constructed here rather than per call: it loads the dictionary
    from disk, and rebuilding it for every document would dominate the run.

    Raises:
        SegmenterScopeError: for a non-Japanese language.
        SegmenterUnavailableError: if fugashi or unidic-lite is missing. MeCab
            needs a native build, which is why it is an optional extra.
    """
    require_scope(Segmenter.MECAB, language, JAPANESE)
    fugashi = import_or_refuse(Segmenter.MECAB, "fugashi", "fugashi")
    unidic_lite = import_or_refuse(Segmenter.MECAB, "unidic_lite", "unidic-lite")
    return MecabSegmenter(
        dictionary_version=package_version("unidic-lite"),
        tagger=fugashi.Tagger(f'-d "{unidic_lite.DICDIR}"'),
    )
