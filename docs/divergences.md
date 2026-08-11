# Known divergences

This file records deliberate and unresolved differences from published implementations. Entries are
not tuned away; each states whether it may be used as a reproduction gate.

## Nayeem et al. fertility ranges

**Status:** not a reproduction gate.

The source does not state the word segmenter used for its published fertility ranges. Segmentation is
a material parameter, so glotscope requires an explicit segmenter and records its model version rather
than selecting a default. We will not tune a segmenter to match an under-specified range.

## Turkish full-alignment value (`arabaları → 0.5`)

**Status:** unresolved; not a reproduction gate until U2 is resolved.

The reported Turkish value does not reproduce under either standard F1 definition. The exact
full-alignment F1 definition remains U2. Retain the independently reproducible
`gathered → g/a/t/h/e/r/e/d` boundary-F1 0.25 check, and do not tune to the Turkish value.

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
