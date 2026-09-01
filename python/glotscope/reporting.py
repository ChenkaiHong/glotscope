"""Assembling the §9 document (PRD §9, G2, G4).

One code path, shared by ``analyze``, ``verify`` and the leaderboard. Shared
rather than reimplemented because ``verify``'s whole claim is that it regenerates
what ``analyze`` produced: a second assembly of the same document would be a
second thing to drift, and that drift would surface as a verification failure
blamed on the numbers rather than on the assembly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from glotscope import __version__, backend
from glotscope.corpus import LoadedCorpus
from glotscope.detect import AGREEMENT_THRESHOLD
from glotscope.embeddings import Embeddings
from glotscope.enums import (
    MorphologicalType,
    Normalization,
    RenyiNormalizer,
    Segmenter,
)
from glotscope.manifest import Manifest, environment
from glotscope.report import Report
from glotscope.results import CorpusMetrics
from glotscope.tokenizer import Tokenizer

__all__ = ["attach_tier2", "build_report"]


def build_report(
    tokenizer: Tokenizer,
    loaded: LoadedCorpus,
    *,
    leading_space: bool,
    normalization: Normalization,
    add_special_tokens: bool,
    segmenter: Segmenter | None,
    segmenter_model: str | None = None,
    parity_reference: str | None,
    gini: bool,
    renyi_alpha: float | None,
    renyi_normalizer: RenyiNormalizer,
    nominal_vocab_size: int | None,
    morphological_types: Mapping[str, MorphologicalType] | None = None,
    frequency_weighted: bool | None = None,
    include_single_token_words: bool | None = None,
) -> Report:
    """Assemble the §9 document. One code path, used by ``analyze`` and ``verify``.

    Shared rather than reimplemented because ``verify``'s whole claim is that it
    regenerates what ``analyze`` produced. A second assembly of the same document
    would be a second thing to drift, and that drift would surface as a
    verification failure blamed on the numbers.
    """
    tier1 = tokenizer.analyze(
        loaded,
        leading_space=leading_space,
        normalization=normalization,
        add_special_tokens=add_special_tokens,
        segmenter=segmenter,
        segmenter_model=segmenter_model,
        morphological_types=morphological_types,
        frequency_weighted=frequency_weighted,
        include_single_token_words=include_single_token_words,
    )
    tier1 = replace(
        tier1,
        corpus_level=CorpusMetrics(
            gini=tier1.gini() if gini else None,
            renyi=(
                tier1.renyi_efficiency(
                    renyi_alpha,
                    normalizer=renyi_normalizer,
                    nominal_vocab_size=nominal_vocab_size,
                )
                if renyi_alpha is not None
                else None
            ),
            parity=tier1.parity(parity_reference) if parity_reference is not None else None,
        ),
    )
    if tier1.parameters is None or tier1.corpus is None:
        raise RuntimeError("analyze() returned a report without its manifest fragments")

    return Report(
        tier0=tokenizer.lint(),
        tier1=tier1,
        manifest=Manifest(
            tokenizer=tokenizer.manifest,
            parameters=tier1.parameters,
            environment=environment(),
            backend=backend(),
            glotscope_version=__version__,
            corpus=tier1.corpus,
        ),
        warnings=tokenizer.warnings,
    )


def attach_tier2(
    report: Report,
    tokenizer: Tokenizer,
    embeddings: Embeddings,
    *,
    top_pct: float,
    remove_first_pc: bool = False,
) -> Report:
    """Run §7.9 over ``embeddings`` and add the result to ``report``, with the
    provenance that makes it a Tier 2 document. One code path, used by
    ``detect``, the leaderboard and ``verify``.

    Three things change beside the ``tier2`` block, and all three are what a
    reader — or ``verify`` — needs to know which measurement this was:

    * ``manifest.weights`` identifies the checkpoint by shard digest and dtype.
      Without it the document carries Tier 2 numbers and no record of what they
      were read from, which §9 forbids.
    * ``manifest.parameters`` gains §7.9's five recorded choices — ``top_pct``,
      the agreement threshold, both candidate counts, whether the first PC was
      removed. ``verify`` regenerates from exactly these.
    * ``manifest.tokenizer.embedding_rows`` is filled in, and the load-time
      warning that called it unknown is replaced: the rows have now been read.

    The leaderboard once rebuilt the report with ``tier2`` set and the Tier 1
    manifest unchanged, which published rows none of that was true of. Whatever
    ``report`` already carries — Tier 1 and its corpus block, for a leaderboard
    row — is kept: those numbers rest on that corpus, and a document that
    dropped the block would say less than the run did.

    ``report.warnings`` is expected to be the tokenizer's own — which is what
    :func:`build_report` and ``detect`` put there; Tier 1 and Tier 2 carry
    theirs on their own blocks and :meth:`Report.to_dict` aggregates.
    """
    tier2 = tokenizer.detect_undertrained(
        embeddings,
        top_pct=top_pct,
        remove_first_pc=remove_first_pc,
    )
    tokenizer_manifest, warnings = tokenizer.with_weights(embedding_rows=embeddings.n_rows)
    return Report(
        tier0=report.tier0,
        tier1=report.tier1,
        tier2=tier2,
        manifest=replace(
            report.manifest,
            tokenizer=tokenizer_manifest,
            weights=embeddings.manifest,
            parameters=replace(
                report.manifest.parameters,
                top_pct=top_pct,
                # §7.9 requires LOW_CONFIDENCE "when they disagree beyond
                # threshold" and fixes no value, so the verdict is unreadable
                # without the line it was measured against: a HIGH from a run at
                # 0.7 says something different from a HIGH at 0.3.
                agreement_threshold=AGREEMENT_THRESHOLD,
                candidates_pre_exclusion=tier2.candidates_pre_exclusion,
                candidates_post_exclusion=tier2.candidates_post_exclusion,
                first_pc_removed=tier2.first_pc_removed,
            ),
        ),
        warnings=warnings,
    )
