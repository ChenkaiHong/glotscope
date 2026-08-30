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

__all__ = ["build_report"]


def build_report(
    tokenizer: Tokenizer,
    loaded: LoadedCorpus,
    *,
    leading_space: bool,
    normalization: Normalization,
    add_special_tokens: bool,
    segmenter: Segmenter | None,
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
