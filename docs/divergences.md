# Known divergences

This file records deliberate and unresolved differences from published implementations. Entries are
not tuned away; each states whether it may be used as a reproduction gate.

## Nayeem et al. fertility ranges

**Status:** not a reproduction gate.

The source does not state the word segmenter used for its published fertility ranges. Segmentation is
a material parameter, so glotscope requires an explicit segmenter and records its model version rather
than selecting a default. We will not tune a segmenter to match an under-specified range.

## Turkish full-alignment value (`arabaları → 0.5`)

**Status:** upstream inconsistency; not a reproduction gate.

The paper's implementation uses micro-aggregated exact-boundary precision, recall, and F1. Against
gold `araba/lar/ı`, both `araba/ları` and `arabalar/ı` have one true boundary, one predicted boundary,
and two reference boundaries, so F1 is `2/3`, not the table's `0.5`. The same implementation correctly
gives `gathered → g/a/t/h/e/r/e/d` F1 `0.25`. Retain `0.25` as a gate, test `2/3` as the derived Turkish
value, and do not tune to the table. See [`m0-source-audit.md`](m0-source-audit.md#u2-full-alignment-f1).

## Foroutan parity-aware BPE Gini rows

**Status:** artifacts unavailable; not a reproduction gate.

The official repository publishes training and conversion code but no trained tokenizer files,
artifact hashes, or immutable model/dataset revisions. Its current paper table also differs from the
older planned values: the base parity-aware BPE row is `0.007`, not `0.011`, and there is no UnigramLM
row. Glotscope validates Gini with hand-computed and property tests until reproducible upstream
artifacts exist; it does not reconstruct and tune a tokenizer to the published numbers. See
[`m0-source-audit.md`](m0-source-audit.md#u3-parity-aware-bpe-artifacts).

## Morphology (§7.7) computes, but no corpus feeds it yet

**Status:** a stated absence, narrower than it was. The measures themselves are gated.

All three §7.7 measures are implemented in `morphology.py` — MorphScore v1 (binary accuracy on the
annotated stem–suffix boundary), MorphScore v2 (P/R/F1 over that same boundary), and full alignment
(P/R/F1 over *all* boundaries, suffix–suffix included). `gathered → g/a/t/h/e/r/e/d` reproduces
boundary-F1 `0.25` exactly and is a gate; the derived Turkish `2/3` is tested as the entry above
prescribes.

What is **not** built is the path from a corpus to those measures: nothing parses MorphyNet's TSVs
into gold segmentations, and `analyze` neither requests nor emits a `MorphologyResult`. The registry
entry `morphynet` exists with its download recipe and `MORPH_GOLD` capability, and glotscope ships no
corpora (D12), so the gap is the loader rather than the metric.

Recorded here because G3's exit condition is that every §7 subsection has either a reference test or
an entry in this file, and a metric that is computable but unreachable from a run is exactly the kind
of half-state that reads as "implemented" in a feature list. A reader comparing glotscope against a
tool that publishes morphology numbers should know that glotscope currently publishes none.

Two properties are enforced by the types rather than by convention: precision is non-optional in the
return type (D11) — recall alone rewards oversegmentation, and a character-level tokenizer earns
perfect recall on `gathered` — and Semitic root-and-pattern and isolating languages return
`TypologicalScope.OUT_OF_SCOPE` carrying no numbers at all, even though the reference implementation
publishes numbers there (see the next entry).

## Morphological alignment outside typological scope

**Status:** deliberate semantic divergence.

For non-concatenative and isolating languages, glotscope returns `OUT_OF_SCOPE` rather than a numeric
morphological-alignment score. Semitic root-and-pattern morphology has no linear boundary to score,
and isolating languages lack affixation. This differs from reference tables that publish values for
Hebrew and Mandarin; the Mandarin row is a single-token artifact rather than morphological alignment.

## Zouhar Rényi README values

**Status:** upstream example discrepancy; the documented numbers are not an α=2.5 reproduction gate.

The `tokenization-scorer` README labels two example calls `power=2.5`, but its documented outputs
(`0.8031528501359657` and `0.9105681923824472`) are exactly the implementation's α=3.0 results. The
formula produces `0.8265064834225245` and `0.9204840242168807` at α=2.5. Glotscope tests both facts and
records α on every result rather than relabelling the published values.
