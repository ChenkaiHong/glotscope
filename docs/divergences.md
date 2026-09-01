# Known divergences

This file records deliberate and unresolved differences from published implementations. Entries are
not tuned away; each states whether it may be used as a reproduction gate.

## TokLens' GPT-2 Japanese premium (~56x) is not reachable on FLORES+ devtest

**Status:** does not reproduce; not a reproduction gate. Unresolved pending TokLens' own definition.

§12.1 lists "GPT-2 Japanese ~56x English (TokLens, +/-10%)" with the note that the segmenter must
match TokLens or the difference be logged. This is the log. Measured on FLORES+ devtest at revision
`5fec6c13`, 1012 aligned documents per language, GPT-2's vocabulary (identical counts through
`tokenizers` and through tiktoken `r50k_base`):

| framing | eng_Latn | jpn_Jpan | ratio |
|---|---|---|---|
| tokens per document (ratio of means, D7) | 26.72 | 79.26 | **2.97** |
| tokens per document (mean of per-document ratios) | | | 3.02 |
| tokens per character | 0.2049 | 1.4087 | 6.87 |
| tokens per byte | 0.2047 | 0.4800 | 2.34 |
| per-document ratio, worst case | | | 5.67 |

No framing tried comes near 56. The bound is stronger than a disagreement about Japanese: across
**all 221 devtest varieties**, the largest GPT-2 premium against English is `shn_Mymr` at 18.69x, and
`jpn_Jpan` ranks 93rd. So 56x is outside the range this corpus and this tokenizer can produce for any
language, and no choice of word segmenter changes that — every quantity above is segmenter-free.

What is *not* claimed: that the published figure is wrong. TokLens' definition was not read from
source, and the same exercise for TokEval (U1) showed that a compression "rate" can mean a ratio of
totals in one implementation and a mean of per-document ratios in another. The resolution is to pin
TokLens' revision and read the quantity it computes, exactly as U1 was resolved — not to search for a
framing that lands on 56.

Reproduce with `Corpus.flores_plus(["eng_Latn", "jpn_Jpan"]).load(root)` and
`Tokenizer.from_tiktoken("r50k_base")`.

## FLORES+ publishes 227 dev and 221 devtest varieties, not 229

**Status:** upstream count; changes the denominator of a planned analysis.

The internal specification describes the paper's analysis as running over "all 229 FLORES+
varieties". The release does not carry 229 in either split. Read from the Hub metadata of
`openlanguagedata/flores_plus` at revision `5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06` on
29 August 2026:

| | count |
|---|---|
| `dev/*.jsonl` | 227 |
| `devtest/*.jsonl` | 221 |
| union of both splits | 230 |
| present in both | 218 |

Nine varieties are dev-only — `brx_Deva`, `dar_Cyrl`, `dgo_Deva`, `gom_Deva`, `mni_Mtei`, `snd_Deva`,
`udm_Cyrl`, `uzs_Arab`, `wuu_Hans` — and three are devtest-only: `cat_Latn_vale1252`, `kaa_Latn`,
`khk_Mong`.

FLORES+ also renames one code the internal specification's core set depends on: **§10.2's
`zho_Hans` does not exist in this release**, which carries `cmn_Hans` (and `cmn_Hant`) instead — the
individual Mandarin code rather than the Chinese macrolanguage code FLORES-200 used. The board uses
`cmn_Hans` and says so here; it is the same variety under a code the release actually publishes, not
a substitution of one language for another. The other fourteen core-set codes are present unchanged.

This matters because the analysis is specified over **devtest** — clean translated prose is what makes
corpus attribution return ≈0 and script attribution the primary variable. So the planned n is **221**,
not 229, and 230 is reachable only by mixing splits, which would mix clean devtest prose with dev.
The number is not adjusted anywhere in code to make 229 appear; `fetch_flores_plus.py` refuses a
variety absent from the split being fetched and names it, rather than silently returning fewer files
than were asked for.

Counts were read from the repository listing, which is public; the files themselves are gated and
were not fetched. A count taken from file names cannot be wrong about how many files exist, but it
says nothing about their contents — the per-language document counts (dev 997 / devtest 1012) remain
unverified here.

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

## `compare` takes result documents where §8.2 spells tokenizers

**Status:** deliberate API divergence, confirmed rather than accidental.

§8.2 writes the subcommand as `glotscope compare <tok1> <tok2> ... --metric parity`. glotscope
implements `glotscope compare a.json b.json ... --metric parity`, over published §9 documents.

The reason is the refusal §8.2 itself requires two lines later: *"`compare` refuses to table metrics
computed under different segmenters, α values, normalizers, or language sets. This is a feature and
the error message should say so."* That refusal is **unreachable** from the tokenizer form. Handed two
tokenizers, `compare` would have to analyze them itself, under one shared set of command-line flags —
so the segmenter, α, normalizer and language set would be identical by construction, and
`IncomparableError` could never fire. The check that §8.2 calls the feature would exist and never run.

Over documents it fires, because two files can disagree, and disagreement between published results is
exactly the situation a leaderboard has to refuse to table. Comparability is scoped per metric through
`results.require_comparable`.

Two consequences worth stating rather than discovering:

- STRR is deliberately absent from `compare`'s `METRICS`. §9 publishes neither `lowercased` nor
  `n_words`, so its comparability key cannot be reconstructed from a document, and tabling it would
  mean comparing two numbers whose conventions are unknown. Closing that gap is a schema change, held
  until something needs it.
- Comparing results requires them to have been published first, which is the shape the leaderboard
  wants anyway: `results/` holds documents, and the nightly re-run compares documents to documents.

## Tier 2 candidate sets are smaller than the published ones by two selection rules

**Status:** the *ranking* is a reproduction gate and is committed as one; the *count* is not, and the
shortfall is decomposed rather than tuned away.

D18 validates §7.9 against Land & Bartolo's **candidate sets** — 999 / 5117 / 1280 — rather than their
confirmed counts of 3161 / 49 / 6, which come from a verification prompt run through the model and are
therefore Tier 3. `scripts/validate_tier2_reference.py` regenerates the comparison and
[`data/tier2-reference-validation.json`](../data/tier2-reference-validation.json) is its committed
output; every checkpoint is pinned by Hub revision and every reference file by SHA-256.

The rankings are identical. Spearman ρ between the two implementations' indicator values is
**1.000000** on all three models, over 49,912 / 255,738 / 63,765 tokens respectively — which is the
claim that matters, because a count can agree by coincidence and an ordering over a quarter of a
million tokens cannot. That clears M2's ρ > 0.9 gate with no margin left to spend.

The counts differ by one, three and five tokens, and every one of them is a selection rule rather than
a measurement:

| Model | glotscope | published | `OK_SPECIAL` admitted | threshold rule | unexplained |
|---|---|---|---|---|---|
| `openai-community/gpt2-medium` | 998 | 999 | 0 | +1 | **0** |
| `google/gemma-2b` | 5,114 | 5,117 | +2 | +1 | **0** |
| `ai21labs/Jamba-v0.1` | 1,275 | 1,280 | +4 | +1 | **0** |

**Special ids.** magikarp draws its threshold over tokens whose category is exactly `OK`, then selects
with `category.startswith("OK")` — so `OK_SPECIAL` ids enter the candidate set they were excluded from
defining. §7.9 Stage 1 excludes special ids outright, so glotscope never scores them. Jamba's four are
`<|pad|>`, `<|startoftext|>`, `<|endoftext|>` and `<|unk|>`; gemma's two are `<pad>` and `<unk>`. Their
embedding rows being untrained is a statement about the checkpoint's padding, not about its vocabulary.

**Threshold rule.** §7.9 specifies the top `top_pct` share, which is `floor(N × 2%)`. magikarp takes
`np.percentile(..., 2.0)` — linearly interpolated — and admits ties with `<=`. The interpolated
threshold lands on or just above the k-th value, so it admits exactly one more token in each of the
three. For gpt2-medium that token is id 8421, ` Before`; for Jamba it is id 2220, `ck`.

The last column is what makes this a decomposition rather than an excuse: applying §7.9's own rule to
the reference implementation's published indicator values reproduces glotscope's count **exactly**, on
all three. Nothing is left for the indicator, the reference set, or an unnamed cause to absorb, and
`tests/test_tier2_reference_validation.py` asserts that residue stays zero.

Set membership follows: containment of glotscope's candidates in magikarp's is 1.000 / 0.9998 / 1.000.
The single gemma exception is id 255285, `ↄ`, which sits either side of the two rules' cut.

**Stage 1 also disagrees on gemma, by three ids out of 256,000.** glotscope excludes 262 where magikarp
excludes 261. glotscope drops ids 106 and 107 — `<start_of_turn>` and `<end_of_turn>`, which the
tokenizer declares as added special tokens — where magikarp categorises both `OK`; magikarp drops id 4,
`<mask>`, as `OK_SPECIAL` where glotscope does not. gpt2-medium and Jamba agree exactly (345 and 1,771).

**Not adopted.** A magikarp-compatible selection mode was considered and declined. §7.9 is normative
and its Stage 1 excludes special ids by design; reproducing the upstream asymmetry would mean
publishing under a rule whose own threshold and selection predicates disagree. The differences are
named here instead, which is what this file is for.

## `compare` offers no gini column

**Status:** deliberate omission, on the same rule STRR is held to; reversible by a schema change.

§7.4's Gini is comparable only at a fixed cost unit — `GiniResult.comparability_key()` returns
`languages` **and** `cost_unit`, because a Gini computed per aligned line and one computed per
sentence are different numbers wearing the same name. §9 publishes `corpus_level.gini` as a bare
float and no unit beside it.

So `compare` could not check the half that matters. Its gini branch keyed on the language sets alone
and returned "comparable" whether or not the units agreed — an answer that looks like the refusal
working and is not. The column was offered from the module's first version; it took a review pass to
notice it had the same shape as the STRR case the same file had already excluded, three paragraphs up
in its own docstring.

Removed from `METRICS` and from the corpus-level value path both, so no branch reads a name that can
no longer be requested. `glotscope compare --metric gini` now names the metrics that exist.

The alternative was schema 1.4, publishing `cost_unit` in §9. Kai chose the removal: it ships without
moving every committed `result.json` or the G4 fixture, and it keeps one rule rather than two. Three
candidates are now parked behind that same bump — `cost_unit`, §7.9's agreement threshold, and STRR's
`lowercased`/`n_words`, all of them the same unpublished-comparability problem. If the schema moves,
it should move once and clear all three.

## UD multiword tokens are not morpheme boundaries

**Status:** deliberate scope limit, decided on measurement rather than on argument.

§7.7(c) needs gold morpheme boundaries as character offsets, and MorphyNet — the one resource that
supplies them — ships **no Turkish at all**, while §10.2 keeps Turkish as the canonical MorphScore
test bed. UD has Turkish. UD also has a construct that looks like the missing gold: the multiword
token, where one surface token expands into several syntactic words.

```
3-4	al	_	_	_	_	_	_	_	_
3	a	a	ADP	_	_	5	case	_	_
4	el	el	DET	_	_	5	det	_	_
```

It is not the missing gold. That notation records where a **surface token differs from its syntactic
words**, which is a different linguistic relation from morpheme structure and coincides with it only
sometimes. Measured over three treebanks pinned by commit, using the `train` split of each:

| Treebank | Commit | Surface words | MWTs | Share of words | Concatenate to their token |
|---|---|---|---|---|---|
| `UD_Turkish-IMST` | `0c93911` | 36,415 | 1,082 | 2.97% | **99.2%** |
| `UD_Spanish-AnCora` | `20adddb` | 442,892 | 10,129 | 2.29% | **13.6%** |
| `UD_English-EWT` | `4a4d77f` | 201,963 | 2,614 | 1.29% | 100% |


**How these were counted**, because the first attempt got them wrong. Surface words are
`integer-id rows − rows covered by ranges + range rows`, computed statelessly so no counter can leak
across a sentence boundary. The first version tracked "skip up to id N" without resetting at the blank
line between sentences, which silently dropped the opening words of every sentence following one that
contained a multiword token: Turkish came out at 17,057 surface words against the correct 36,415, and
the share was overstated at 6.34%. Concatenation is checked per range by joining the covered rows'
`FORM` fields and comparing to the range row's own `FORM`; those figures were unaffected. Both
statistics were recomputed by two independent methods that agree exactly.

The direction of the error is worth noting: correcting it *strengthens* the conclusion. Turkish
multiword tokens cover less of the treebank than first reported, not more.

Two independent objections, and the treebanks fail different ones:

**Spanish fails mechanically.** `del → de + el` spells `deel`, `al → a + el` spells `ael`. Boundaries
are character offsets, so 86.4% of that treebank's expansions cannot produce offsets at all — the same
canonical-versus-surface trap MorphyNet sets, arriving through a different door. Nor would they be
morphology if they did concatenate: `a` and `el` are separate grammatical words that orthography
fused, not morphemes of one word.

**Turkish fails linguistically, having passed mechanically.** Its expansions do spell their tokens
(99.2%), so offsets are computable. But they mark **derivational** boundaries on 2.97% of words, where
§7.7(c) scores the inflectional segmentation MorphyNet annotates. Turkish is agglutinative: close to
every word has internal morpheme structure, so a gold covering one word in sixteen — selected by a
different criterion — is a biased sample rather than a test bed. A `full_alignment` computed over it
would be a real number about the wrong population.

English concatenates perfectly and covers 1.29% of words, all of them clitics.

So `universal_dependencies` declares `word_segmentation` and **not** `morph_gold`. That makes
`analyze(..., morphological_types=...)` against UD a clean `CapabilityError` at the gate, rather than
an acceptance that fails later inside the MorphyNet parser with a column-count error naming the wrong
corpus. Gating is on capability, never on corpus identity (D5), so the capability has to be true.

**Consequence, stated rather than hidden:** Turkish morphology is not measurable from either resource
today. MorphyNet has no Turkish file; UD has Turkish but not this annotation. `hbs`, `pol` and `rus`
are in the same position — MorphyNet ships derivational files only for them. That is a coverage gap in
§10.2's core set, and it is recorded here rather than filled by whichever gold happens to be parseable.

Reversible on evidence: a treebank whose expansions are inflectional, surface-preserving and broadly
distributed would qualify, per treebank and never for "UD" as a whole. The three columns above are the
test any candidate has to pass.

## STRR's published values are not reproducible, because the convention is unstated

**Status:** not a reproduction gate. The formula is implemented exactly; the *published numbers* cannot
be regenerated from what the source reports.

§7.6 is Nayeem, Alqahtani, Laskar, Mohiuddin & Bari, *Beyond Fertility* (arXiv:2510.09947):

```
STRR(T; W) = (1/n) Σ 1( |T(w_i)| = 1 ) × 100
```

The formula is unambiguous and glotscope matches it. What cannot be reproduced is any published STRR
*value*, for two independent reasons.

**The leading-space convention is not stated, and it dominates the result.** `τ("the")` and
`τ(" the")` are different tokenizations of the same word, and for a byte-level BPE the second is
frequently one token where the first is two — so the same tokenizer over the same word list scores
tens of points apart depending on a choice the paper does not record. There is no way to tell which
was used from the numbers alone, and picking one to match a table would be tuning a parameter until a
result appeared, which §17 calls misconduct.

So glotscope **returns a pair and offers no single-value counterpart**: `strr_bare` and
`strr_leading_space`, both always, with `lowercased` recorded beside them. `StrrPair` deliberately
exposes no `value` attribute — the refusal is structural rather than advisory, because an unqualified
STRR is the thing that cannot be compared across two runs.

**The word lists carry a confound the source documents but does not correct.** They are English
frequency lists *translated*, not per-language frequency lists, so word rank does not mean the same
thing in any two languages. Hindi scores lowest across every tokenizer tested despite 128k–255k
vocabularies — a property of the list at least as much as of the tokenizers. Reproducing a published
per-language ranking would therefore reproduce the confound rather than validate the implementation.

**A scale bug found while writing this entry, and fixed rather than documented.** `strr()` returned a
fraction while `Tier1Report.to_dict` published the number unchanged and every docstring described the
conventions as differing "by tens of *points*". The same §9 field could carry `0.42` or `42.0`
depending on which code path built the pair — `tests/test_tier1_report.py` had already assumed the
percentage. §7's `× 100` is part of the formula, not presentation, so this was a deviation and not a
convention: the implementation now matches. No published result was affected, because STRR appears in
no committed `result.json` and `compare` does not offer the metric.

**Consequence for comparability:** two STRR numbers may be compared only at the same convention, the
same casing and the same word list. §9 publishes neither `lowercased` nor `n_words`, which is why
`compare` excludes STRR — the same rule that later excluded gini. Publishing those two fields is a
schema change and is parked with the other 1.4 candidates.

## Zouhar Rényi README values

**Status:** upstream example discrepancy; the documented numbers are not an α=2.5 reproduction gate.

The `tokenization-scorer` README labels two example calls `power=2.5`, but its documented outputs
(`0.8031528501359657` and `0.9105681923824472`) are exactly the implementation's α=3.0 results. The
formula produces `0.8265064834225245` and `0.9204840242168807` at α=2.5. Glotscope tests both facts and
records α on every result rather than relabelling the published values.
