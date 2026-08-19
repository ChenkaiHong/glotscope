# glotscope

**Multilingual tokenizer diagnostics with integrated under-trained-token detection.**

`glotscope` computes tokenizer diagnostics across four capability tiers — from pure vocabulary
introspection, through corpus-based fairness metrics, to weight-based under-trained-token detection
read directly from model checkpoints.

It is the first package to span corpus metrics *and* model-weight metrics. Existing corpus-metric
suites stop at the tokenizer; the existing glitch-token detector starts at the weights and is not
installable.

> **Status: v0.1.0 prepared, not yet published.** Tiers 0, 1 and 2 are implemented and tested, and
> `glotscope lint`, `analyze`, `detect`, `compare` and `verify` run. `leaderboard` is scheduled and
> exits 2 rather than guessing. Tier 2's indicators are not yet validated against the published
> candidate sets. Morphology now runs end to end against MorphyNet, over the quarter of that
> corpus whose gold segmentation spells its own surface form — the coverage, and what it costs,
> are recorded in [`divergences.md`](docs/divergences.md). The only release on PyPI is the 0.0.0
> name reservation.
> See the [`build order and open questions`](docs/build-order.md).

## What these metrics do and do not tell you

Read this before using any number this library produces.

`glotscope` reports **diagnostics, not quality predictions.** The literature does not support the
claim that any metric here predicts downstream model quality, and in several cases actively
contradicts it:

- **Compression** correlates −0.71 to −0.996 with quality when only training-corpus size varies, but
  **+0.241** with an inverted U when the algorithm varies across 54 models.
- **Rényi efficiency** can be provably *raised* while BLEU *falls* — two published constructions do
  exactly that — and it correlates −0.891 with corpus token count, making it largely redundant with
  compression.
- **Morphological alignment** shows no significant correlation with perplexity in the work that
  introduced it (F(1,13)=0.323, p=0.580).

The library will never imply causation, and metrics known to be contested emit a warning attached to
the result.

## The tier model

Metrics differ in what they *require*, and conflating those requirements is what makes existing tools
either narrow or fragile.

| Tier | Requires | Cost | Contents |
|---|---|---|---|
| **0** | tokenizer only | milliseconds | vocab size, script composition, UTF-8 vocabulary integrity, unreachable tokens, byte-fallback coverage |
| **1** | tokenizer + corpus | seconds–minutes | fertility, CPT/BPT/CTC, compression, Rényi efficiency, parity/premium, Gini, STRR, morphological alignment, round-trip losslessness |
| **2** | tokenizer + embedding tensors | seconds | under-trained-token indicators, embedding-norm distributions |
| **3** | tokenizer + full inference | hours + GPU | prompt-based glitch verification — *specified, not implemented* |

Tier 2 is cheaper than it looks: it needs two tensors, readable from `safetensors` without
instantiating the model.

## Design commitments

**It refuses rather than guesses.** Requesting parity on a monolingual corpus raises a typed error
instead of returning a meaningless number. Fertility has no default word segmenter, because the
choice of segmenter is the single largest source of silent incomparability in this literature and a
default would manufacture exactly that problem. Comparing results computed under different
segmenters, α values, normalizers, or language sets raises rather than tabling them together.

**Every result carries a manifest.** Tokenizer revision SHA, `tokenizer.json` SHA-256, weight-shard
SHA-256 and dtype, corpus version, segmenter and its model version, and every contested parameter.
Re-running the manifest reproduces the numbers bit-identically, and CI asserts it. No competing tool
pins revisions or publishes artifact hashes.

**Where it disagrees with other implementations, it says so.** `docs/divergences.md` records every
divergence and why. A documented divergence is a contribution; a silently tuned one is misconduct.

**It ships no corpora.** Download recipes, checksums, and an SPDX license field per resource, plus a
`--license-filter=commercial` switch.

## Install

```bash
pip install glotscope                 # core: Tier 0 and Tier 1
pip install "glotscope[tier2]"        # + reading embedding tensors
pip install "glotscope[tiktoken]"     # + OpenAI encodings by name
pip install "glotscope[segmenters]"   # optional word segmenters
```

Python 3.10–3.13, Linux/macOS/Windows. The core install is one dependency, because every tier past
the first two costs a dependency tree the user may not need: Tier 2 reads `safetensors` shards as
arrays, and `tiktoken` is a second tokenizer library. Segmenters are separate again because MeCab
needs a native build and PyICU needs system ICU; the core install has no such requirement.

PyICU builds against system ICU and fails with `KeyError: 'ICU_VERSION'` when `icu-config` is not on
`PATH`. On macOS with Homebrew:

```bash
brew install icu4c
export PATH="$(brew --prefix icu4c)/bin:$PATH"
export PKG_CONFIG_PATH="$(brew --prefix icu4c)/lib/pkgconfig"
pip install "glotscope[segmenters]"
```

**Word segmentation is a required, recorded parameter — there is no default.** `W(D)` is the single
largest source of silent incomparability in this literature, so a fertility number without a
segmenter raises, a missing segmenter extra raises rather than falling back to whitespace, and a
language-scoped segmenter used on another language raises rather than returning a plausible number.

## Development

```bash
pip install -e ".[dev]"

ruff check python/ tests/
mypy --strict
pytest
pytest --cov --cov-report=term-missing   # gate: 85% line coverage
```

Tests are marked by kind: `reference` (reproduces a published value), `property` (Hypothesis),
`segmenter` / `gated` / `network` (skip when the resource is unavailable). Run just the fast,
dependency-free reference tests with:

```bash
pytest -m "reference and not network and not gated"
```

## License

Apache-2.0. The patent grant matters more than brevity here, it matches the license of the method
Tier 2 reimplements, and it is the license enterprise users can adopt without review.
