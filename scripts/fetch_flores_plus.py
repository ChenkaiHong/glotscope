"""Fetch FLORES+ into the layout ``Corpus.load`` reads (PRD §10.1, D12).

glotscope ships no corpora. The registry entry carries the recipe as prose; this
is that recipe made executable, so what a published number was computed over is
a command someone else can re-run rather than a paragraph they must interpret.

**The dataset is gated** (``gated: auto``): accept the terms once at
https://huggingface.co/datasets/openlanguagedata/flores_plus and export
``HF_TOKEN``. Anonymous access returns 403, which this script reports as that
rather than as a missing file.

What it guarantees, and refuses rather than papering over:

* The **revision is resolved before anything is fetched**, so the files and the
  recorded commit are the same revision even if the branch moves in between.
* Every language in a split must have the **same number of documents**, and the
  same sentence ids in the same order where the release carries them. FLORES+ is
  the parallel corpus; parity is a ratio of means and equals the ratio of totals
  only when the counts match (§7.3, D7). An off-by-one here is invisible in the
  output and wrong in every downstream number.
* A document containing a newline is refused. One line is one document, so an
  embedded newline would silently split a sentence in two and desynchronise the
  language against every other.
* The text field is **discovered, not assumed**: the first record's keys are
  read, and a release that renames the field is an error naming what it found.

Usage::

    python scripts/fetch_flores_plus.py --root ~/corpora
    python scripts/fetch_flores_plus.py --root ~/corpora --languages eng_Latn hin_Deva
    python scripts/fetch_flores_plus.py --root ~/corpora --split dev --revision <sha>

The manifest it writes records the resolved revision, the field extracted, and
both the source and written digests per language, plus the ``corpus_digest`` to
pass as ``Corpus.flores_plus(..., sha256=...)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from glotscope.corpus import FLORES_PLUS_VERSION, REGISTRY, corpus_digest

__all__ = ["fetch", "main"]

REPO = "openlanguagedata/flores_plus"
"""The dataset the registry recipe names. Not a parameter: a different repository
is a different corpus, and the id in the manifest would be a lie."""

_TERMS_URL = f"https://huggingface.co/datasets/{REPO}"

DEFAULT_TEXT_FIELD = "text"
_ID_FIELD = "id"


def _hub() -> tuple[Any, Any]:
    """``(dataset_info, hf_hub_download)``, or a message naming the extra."""
    try:
        import huggingface_hub
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the extra
        raise ModuleNotFoundError(
            "fetching from the Hub needs `huggingface_hub`, which ships in the "
            "`tier2` extra: pip install 'glotscope[tier2]'"
        ) from exc
    return huggingface_hub.dataset_info, huggingface_hub.hf_hub_download


def _resolve(dataset_info: Any, revision: str | None) -> tuple[str, list[str]]:
    """Resolve the revision and list its files, before fetching anything."""
    try:
        info = dataset_info(REPO, revision=revision)
    except Exception as exc:
        raise SystemExit(
            f"cannot read {REPO} ({exc}).\n"
            f"The dataset is gated: accept the terms at {_TERMS_URL} and export "
            f"HF_TOKEN. Anonymous access is 403, not 404 — the repository exists."
        ) from exc
    resolved = revision or str(getattr(info, "sha", "") or "")
    files = [str(sibling.rfilename) for sibling in info.siblings]
    return resolved, files


def _languages_in(files: Sequence[str], split: str) -> list[str]:
    prefix = f"{split}/"
    return sorted(
        name[len(prefix) : -len(".jsonl")]
        for name in files
        if name.startswith(prefix) and name.endswith(".jsonl")
    )


def _records(raw: bytes, *, language: str) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise SystemExit(f"{language}: line {number} is not JSON ({exc})") from exc
        if not isinstance(record, dict):
            raise SystemExit(f"{language}: line {number} is not a JSON object")
        records.append(record)
    if not records:
        raise SystemExit(f"{language}: the release file holds no records")
    return records


def _texts(records: Sequence[Mapping[str, Any]], *, language: str, field: str) -> list[str]:
    if field not in records[0]:
        raise SystemExit(
            f"{language}: no {field!r} field in the release. The record carries "
            f"{sorted(records[0])}. Pass --text-field to name the right one — "
            f"guessing here would put an unknown column under every number."
        )
    texts: list[str] = []
    for number, record in enumerate(records, start=1):
        value = record.get(field)
        if not isinstance(value, str):
            raise SystemExit(f"{language}: record {number} has no string {field!r}")
        if "\n" in value or "\r" in value:
            raise SystemExit(
                f"{language}: record {number} contains a newline. One line is one "
                f"document, so writing it would split the sentence in two and "
                f"desynchronise this language from every other."
            )
        texts.append(value)
    return texts


def _check_parallel(
    counts: Mapping[str, int],
    ids: Mapping[str, tuple[Any, ...]],
) -> None:
    """Refuse a set of languages that is not actually parallel."""
    if len(set(counts.values())) > 1:
        raise SystemExit(
            f"unequal document counts across languages: {dict(sorted(counts.items()))}. "
            f"FLORES+ is the parallel corpus, and parity is a ratio of means that "
            f"equals the ratio of totals only when the counts match (§7.3, D7)."
        )
    carried = {language: sequence for language, sequence in ids.items() if sequence}
    if len(set(carried.values())) > 1:
        disagreeing = sorted(carried)
        raise SystemExit(
            f"sentence ids differ across {disagreeing}. Equal counts are not "
            f"alignment: two languages can hold the same number of different "
            f"sentences, and every parity number would be computed across them."
        )


def fetch(
    root: Path,
    *,
    split: str,
    languages: Sequence[str] | None = None,
    revision: str | None = None,
    text_field: str = DEFAULT_TEXT_FIELD,
    version: str = FLORES_PLUS_VERSION,
) -> dict[str, Any]:
    """Download one split into ``<root>/flores_plus/<version>/<split>/``.

    Returns the fetch manifest: the resolved revision, the field extracted, per
    language the source and written digests and the document count, and the
    ``corpus_digest`` to pin the result with.
    """
    dataset_info, download = _hub()
    resolved, files = _resolve(dataset_info, revision)

    available = _languages_in(files, split)
    if not available:
        raise SystemExit(f"the release at {resolved} has no {split!r} split")
    wanted = list(languages) if languages else available
    missing = [language for language in wanted if language not in available]
    if missing:
        raise SystemExit(
            f"not in the {split!r} split at {resolved}: {missing}. That split "
            f"carries {len(available)} varieties; the splits do not hold the same "
            f"set, so a variety present in one can be absent from the other."
        )

    directory = root / REGISTRY["flores_plus"].id / version / split
    directory.mkdir(parents=True, exist_ok=True)

    per_language: dict[str, dict[str, Any]] = {}
    written_digests: dict[str, str] = {}
    counts: dict[str, int] = {}
    ids: dict[str, tuple[Any, ...]] = {}
    for language in wanted:
        source = Path(
            download(
                REPO,
                f"{split}/{language}.jsonl",
                repo_type="dataset",
                revision=resolved or revision,
            )
        )
        raw = source.read_bytes()
        records = _records(raw, language=language)
        texts = _texts(records, language=language, field=text_field)

        payload = "".join(f"{text}\n" for text in texts).encode("utf-8")
        (directory / f"{language}.txt").write_bytes(payload)

        written = hashlib.sha256(payload).hexdigest()
        written_digests[language] = written
        counts[language] = len(texts)
        ids[language] = tuple(record[_ID_FIELD] for record in records if _ID_FIELD in record)
        per_language[language] = {
            "documents": len(texts),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "written_sha256": written,
        }

    _check_parallel(counts, ids)

    return {
        "repo": REPO,
        "repo_type": "dataset",
        "revision": resolved,
        "revision_was_pinned": revision is not None,
        "split": split,
        "version": version,
        "text_field": text_field,
        "languages": per_language,
        # What Corpus.flores_plus(..., sha256=...) pins, so the run that uses
        # these files declares the bytes it read rather than trusting the path.
        "corpus_sha256": corpus_digest(written_digests),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, help="directory holding downloaded corpora")
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="FLORES+ codes, e.g. eng_Latn hin_Deva. Default: every variety in the split",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="dataset commit SHA. Unpinned resolves the default branch and records what it resolved to",
    )
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--version", default=FLORES_PLUS_VERSION)
    parser.add_argument("--manifest", default=None, help="where to write the fetch manifest")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    manifest = fetch(
        root,
        split=args.split,
        languages=args.languages,
        revision=args.revision,
        text_field=args.text_field,
        version=args.version,
    )
    if not manifest["revision_was_pinned"]:
        print(
            f"warning: --revision was not given; resolved to {manifest['revision']} "
            f"and recorded. An unpinned fetch is reproducible against a commit "
            f"nobody chose.",
            file=sys.stderr,
        )

    destination = (
        Path(args.manifest).expanduser()
        if args.manifest
        else root / "flores_plus" / args.version / f"{args.split}.fetch.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"{len(manifest['languages'])} varieties -> "
        f"{root / 'flores_plus' / args.version / args.split}\n"
        f"revision {manifest['revision']}\n"
        f"corpus_sha256 {manifest['corpus_sha256']}\n"
        f"manifest {destination}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
