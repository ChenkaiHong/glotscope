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

## MorphyNet's gold is canonical, so most of it cannot be scored

**Status:** an upstream property, measured and reported rather than worked around.

§7.7(c) scores full alignment against MorphyNet, and boundaries are **character offsets** — the gold
pieces and the tokenizer's pieces have to index into the same string. MorphyNet's segmentation column
does not satisfy that. It is *canonical*: `microtome|ing` against the inflected form `microtoming`,
Mongolian `далай|аар` against `далайгаар`. Concatenating the morphemes spells a different word, so
offsets taken from them describe a string nobody tokenized, and every score computed from them looks
entirely reasonable.

Measured over the published files rather than assumed:

| File | Rows | `-` sentinel | Canonical mismatch | Usable |
|---|---|---|---|---|
| `eng/eng.inflectional.v1.tsv` | 649,593 | 66.05% | 9.75% | **24.20%** |
| `mon/mon.inflectional.v1.tsv` | 30,129 | 0% | 69.59% | **30.41%** |

`glotscope.morphynet` drops both classes plus zero-length morphemes and forms carrying conflicting
segmentations, and **counts every drop**: the counts go into the §9 warnings array beside the number
they constrain, because a reader told an F1 without being told it describes a quarter of the file
will read it as a statement about the language.

Consequence for comparability: a published `full_alignment` is over the usable subset of a pinned
MorphyNet commit. Another implementation that repairs the canonical forms — by applying orthographic
rules, or by aligning approximately — is measuring a larger and different set. That is a divergence
in the *input*, not in the formula, and it is why the coverage warning is not optional.

Two further upstream facts, recorded because they shape what can be run: of MorphyNet's 15 languages
only 12 publish an inflectional file (`hbs`, `pol` and `rus` ship derivational only), and **Turkish is
absent altogether**, although §10.2 keeps it in the core set as the canonical MorphScore test bed.
Upstream file names are irregular (`por/pt.inflectional.v1.tsv`, `hun/hu.inflectional.segmentation.v1.tsv`,
`spa` split across two parts), so the registry recipe asks for a rename into the standard corpus
layout rather than constructing a filename from a language code.

## Sub-character token splits claim no morpheme boundary

**Status:** deliberate divergence, and it moves numbers in the opposite direction to the reference
implementations.

Predicted boundaries come from the tokenizer's **character offsets**, not from decoded token strings.
A byte-level vocabulary spells a space `Ġ` and a byte-fallback vocabulary spells one byte `<0xNN>`,
and neither string's length is a character offset; more importantly, several tokens covering one
multi-byte character all report that character's span, so they collapse to a single piece and no
boundary is claimed inside a character.

§7.7 rule 1 records that Llama tokenizers "split below the character level on non-Latin scripts and
score artificially high". Under offsets that inflation does not happen — a word split into bytes
inside every character scores as a word the tokenizer did not split at all. glotscope therefore
reports *lower* alignment than an implementation counting decoded pieces, and the difference is
largest exactly where the published critique says the artifact is worst.

A word whose offsets do not tile it — a model with no UNK drops what it cannot represent, and returns
zero tokens — is dropped and counted in the warnings rather than scored.

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
