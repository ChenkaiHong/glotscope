"""Stanza and UDPipe — the two trained segmenters (PRD §7.1, §10.3, D6).

Every other adapter in this package wraps a rule set or a bundled dictionary that
ships with its package. These two wrap a **trained model**, and that difference
is the whole design:

* **The model is an explicit path the caller pins.** Both libraries will happily
  download one on first use, and that is precisely what must not happen: a
  published fertility number would then rest on an artifact the manifest never
  saw and nobody chose. §7.1 requires the segmenter *model* version recorded,
  not a treebank release, and a download has no version until after it lands.
* **The recorded version is a digest of the file that produced the boundaries.**
  A package version says nothing about where a word ends, and a model file name
  can be anything. The digest is the only identity that cannot be wrong — the
  same argument as ``tokenizer_json_sha256`` in §9.

Neither is scoped to a language: both are multilingual by construction, and the
model the caller pins is what decides which language it is for. Passing an
English model and a Hindi corpus is a mistake this layer cannot see — which is
why the model is recorded rather than described.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glotscope.enums import Segmenter
from glotscope.segmenters._support import import_or_refuse, package_version
from glotscope.segmenters.stanza_languages import stanza_language

__all__ = [
    "StanzaSegmenter",
    "UdpipeSegmenter",
    "load_stanza",
    "load_udpipe",
    "model_digest",
    "recorded_digest",
    "require_model",
]

_PACKAGES = {
    Segmenter.STANZA: "stanza",
    Segmenter.UDPIPE: "ufal.udpipe",
}


def require_model(segmenter: Segmenter, model: str | Path | None) -> Path:
    """Resolve the pinned model file, or refuse.

    Raises:
        ValueError: no model was given. There is no default: both libraries
            would download one, and an artifact nobody chose would then sit
            behind every number the run publishes.
        FileNotFoundError: the path names nothing on this disk.
    """
    if model is None:
        raise ValueError(
            f"{segmenter.value} needs an explicit model: pass "
            f"get_segmenter(..., model=<path>). There is no default, and this "
            f"adapter will not download one — a model fetched on first use is an "
            f"artifact nobody chose, sitting behind every number the run "
            f"publishes, and §7.1 requires the model version recorded."
        )
    path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such segmenter model")
    return path


def model_digest(path: Path) -> str:
    """SHA-256 of the model file, streamed: a Stanza model is hundreds of
    megabytes and a UDPipe one is not small either."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_DIGEST_MARKER = "sha256:"


def _model_version(package: str, version: str, path: Path) -> str:
    return f"{package} {version}, model {path.name} {_DIGEST_MARKER}{model_digest(path)}"


def recorded_digest(model_version: str) -> str | None:
    """The digest a recorded model version carries, or ``None`` if it has none.

    The inverse of the string these adapters write, kept beside it so the two
    cannot drift apart. ``verify`` reads this out of a manifest to check the
    model it was handed *before* recomputing anything — the same rule the
    tokenizer and the weights already follow: identity fails on what the
    artifact is, not by producing different numbers and blaming the result.
    """
    _, marker, digest = model_version.rpartition(_DIGEST_MARKER)
    return digest if marker else None


@dataclass(frozen=True, slots=True)
class UdpipeSegmenter:
    """UDPipe, through the ``ufal.udpipe`` bindings.

    The tokenizer is built once at load time rather than per call: constructing
    one reads the model, and doing that per document would dominate the run.
    """

    tokenizer: Any
    sentence_factory: Any
    error_factory: Any
    model_version_string: str
    segmenter: Segmenter = Segmenter.UDPIPE

    @property
    def model_version(self) -> str:
        return self.model_version_string

    def segment(self, text: str) -> tuple[str, ...]:
        self.tokenizer.setText(text)
        sentence = self.sentence_factory()
        error = self.error_factory()
        words: list[str] = []
        while self.tokenizer.nextSentence(sentence, error):
            # UDPipe puts a root token at index 0 of every sentence; it is
            # structural and has no surface form, so it is dropped rather than
            # counted as an empty word.
            words.extend(word.form for word in sentence.words if word.form.strip())
            sentence = self.sentence_factory()
        return tuple(words)


@dataclass(frozen=True, slots=True)
class StanzaSegmenter:
    """Stanza's tokenize processor, and nothing else.

    Only ``tokenize`` is loaded. The other processors — POS, lemma, depparse —
    are far more expensive and change no boundary, and loading them would make
    a fertility run pay for annotation nothing here reads.
    """

    pipeline: Any
    model_version_string: str
    segmenter: Segmenter = Segmenter.STANZA

    @property
    def model_version(self) -> str:
        return self.model_version_string

    def segment(self, text: str) -> tuple[str, ...]:
        document = self.pipeline(text)
        return tuple(
            token.text
            for sentence in document.sentences
            for token in sentence.tokens
            if token.text.strip()
        )


def load_udpipe(language: str, model: str | Path | None = None) -> UdpipeSegmenter:
    """Build the UDPipe segmenter around a pinned model file.

    Raises:
        ValueError: if no model was given.
        FileNotFoundError: if the path names nothing.
        SegmenterUnavailableError: if ``ufal.udpipe`` is not installed.
    """
    del language  # UDPipe's language is the model's, not this argument's.
    path = require_model(Segmenter.UDPIPE, model)
    udpipe = import_or_refuse(Segmenter.UDPIPE, "ufal.udpipe", "ufal.udpipe")

    loaded = udpipe.Model.load(str(path))
    if loaded is None:
        raise FileNotFoundError(f"{path}: udpipe could not read this as a model")
    return UdpipeSegmenter(
        tokenizer=loaded.newTokenizer(udpipe.Model.DEFAULT),
        sentence_factory=udpipe.Sentence,
        error_factory=udpipe.ProcessingError,
        model_version_string=_model_version("udpipe", package_version("ufal.udpipe"), path),
    )


def load_stanza(language: str, model: str | Path | None = None) -> StanzaSegmenter:
    """Build the Stanza segmenter around a pinned tokenizer model.

    ``download_method=None`` is the load-bearing argument: without it Stanza
    fetches whatever its resources file points at, and the manifest would record
    a model this process never chose.

    ``lang`` is what Stanza calls the language, resolved through
    :func:`stanza_language` rather than truncated from the corpus code: Stanza
    keys its resources by ISO 639-1, and the first two letters of an ISO 639-3
    code are not that — ``spa`` is ``es``, ``jpn`` is ``ja``. Stanza reads the
    code to find the language's resources entry, not to tokenize; the pinned
    model does that. For a language whose default package pairs the tokenizer
    with a multi-word-token expander, Stanza also loads that expander from its
    resources directory; it changes the words a sentence is split into, never
    the surface tokens this adapter reads.

    Raises:
        ValueError: if no model was given.
        FileNotFoundError: if the path names nothing.
        SegmenterUnavailableError: if ``stanza`` is not installed.
    """
    path = require_model(Segmenter.STANZA, model)
    stanza = import_or_refuse(Segmenter.STANZA, "stanza", "stanza")

    pipeline = stanza.Pipeline(
        lang=stanza_language(language),
        processors="tokenize",
        tokenize_model_path=str(path),
        download_method=None,
        logging_level="ERROR",
    )
    return StanzaSegmenter(
        pipeline=pipeline,
        model_version_string=_model_version("stanza", package_version("stanza"), path),
    )
