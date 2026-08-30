"""Turning a command-line source into a loaded artifact (PRD §8.2).

Shared by the CLI and by the leaderboard rather than written twice. The routing
is the part worth keeping in one place: a path that does not exist and an
identifier that does not resolve are different answers, and a board that reported
one as the other would send a reader after the wrong fix on every row.
"""

from __future__ import annotations

from pathlib import Path

from glotscope.embeddings import Embeddings
from glotscope.errors import TokenizerLoadError
from glotscope.tokenizer import Tokenizer

__all__ = ["load_embeddings", "load_tokenizer"]


def looks_like_a_local_path(source: str) -> bool:
    """Tell a path the user meant from a Hub identifier they meant.

    Only reached once the source has been shown not to exist, and the two
    answers differ in what they tell the reader: a path is a wrong argument to
    fix now, an identifier is a feature scheduled for a later release. Guessing
    "identifier" for both is what made a typo look like a missing feature.
    """
    path = Path(source)
    return (
        path.is_absolute()
        or source.startswith(("~", "./", "../", ".\\", "..\\"))
        or path.suffix == ".json"
        # A parent that exists means the user was naming a place on this disk.
        # "." is excluded: a bare name is the shape a Hub identifier takes.
        or (str(path.parent) != "." and path.parent.is_dir())
    )


_TIKTOKEN_PREFIX = "tiktoken:"
"""Marks an OpenAI encoding on the command line, e.g. ``tiktoken:o200k_base``.

An explicit prefix rather than a list of known encoding names: the registry is
whatever the installed ``tiktoken`` knows, so a name-sniffing rule would send a
newly-published encoding to the Hub and report it as a missing repository. A Hub
identifier cannot contain a colon, so the two namespaces cannot collide.
"""


def load_tokenizer(source: str, revision: str | None) -> Tokenizer:
    """Load the tokenizer named on the command line.

    An OpenAI encoding behind ``tiktoken:``, a local ``tokenizer.json``, a
    directory holding one, or a Hub identifier.
    The routing matters more than either branch: a path that does not exist is a
    wrong argument to fix now, while a bare name is an identifier to resolve, and
    reporting one as the other sends the reader after the wrong fix.

    A Hub identifier resolves to a commit SHA and that SHA is what the manifest
    records, so a row can never silently name one artifact and analyse another —
    the failure §11 exists to prevent.

    Raises:
        TokenizerLoadError: the source names a place on this disk holding no
            tokenizer, or a repository publishing no ``tokenizer.json``. Exit 1 —
            a real answer about this input, not a missing feature.
    """
    if source.startswith(_TIKTOKEN_PREFIX):
        if revision is not None:
            raise TokenizerLoadError(
                source,
                "--revision selects a commit on the Hub, and an OpenAI encoding "
                "is defined by the installed tiktoken rather than by a commit. "
                "Its identity is the tokenizer_json_sha256 the manifest records",
            )
        return Tokenizer.from_tiktoken(source[len(_TIKTOKEN_PREFIX) :])

    path = Path(source)
    if revision is None:
        if path.is_dir():
            candidate = path / "tokenizer.json"
            if not candidate.is_file():
                raise TokenizerLoadError(source, "a directory holding no tokenizer.json")
            return Tokenizer.from_file(candidate)
        if path.is_file():
            return Tokenizer.from_file(path)
    if looks_like_a_local_path(source):
        raise TokenizerLoadError(
            source,
            "no such file or directory"
            if revision is None
            else "--revision selects a commit on the Hub, and this names a local path",
        )
    return Tokenizer.from_pretrained(source, revision=revision)


def load_embeddings(source: str, *, vocab_size: int, revision: str | None = None) -> Embeddings:
    """Load the weights named on the command line.

    A bare name is a Hub identifier and is resolved as one; anything that names
    a place on this disk is read from there. Collapsing the two would report a
    mistyped path as a missing model, or a model as a mistyped path, and send
    the reader after the wrong fix either way.

    Unlike the tokenizer, the Hub path here is implemented:
    ``Embeddings.from_checkpoint`` reads only the shards holding the embedding
    tensors, which is 8.94 GB of Jamba's 96.06 GB rather than all of it.

    Raises:
        FileNotFoundError: the source names a place on this disk holding nothing.
    """
    path = Path(source)
    if path.is_file():
        if revision is not None:
            raise ValueError(
                f"--weights-revision names a commit on the Hub, and {source} is a "
                f"file on this disk. Recording a revision beside a local path "
                f"would pin nothing; that file's identity is its shard_sha256."
            )
        return Embeddings.from_file(path, vocab_size=vocab_size)
    if looks_like_a_local_path(source):
        raise FileNotFoundError(f"{source}: no such file or directory")
    return Embeddings.from_checkpoint(source, revision=revision)
