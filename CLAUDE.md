# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Always: update `HISTORY.md`

`HISTORY.md` (repo root, gitignored) is the narrative log of everything that has happened here. **Append to it before ending any turn in which you created, edited, deleted, or verified something.** A turn that changes nothing in the repo needs no entry.

Newest first, under a `## YYYY-MM-DD` heading, absolute dates only. Record **what changed, why, and how it was verified** — and record decisions *not* to act with the same weight, since those are what get forgotten and re-litigated. Never upgrade "wrote it" to "it works" without having run something. Read the file's header before appending; it states the conventions.

Being gitignored is the point: it holds the reasoning, dead ends and reversals a commit log cannot.

## Repository state

Specified in full, barely implemented. `glotscope-PRD.md` (862 lines) is the spec; `python/glotscope/` holds a typed contract layer — 11 modules, `ruff` clean and `mypy --strict` clean, every metric body still `NotImplementedError`. `pyproject.toml` builds an installable 0.0.0 wheel.

**Not yet:** any metric implementation, any file in `tests/` (so `pytest` collects nothing and the 85% gate would fail), `.github/workflows/`, `docs/divergences.md`, the reserved PyPI name, and `git init` — this is still not a git repo, so `.gitignore` has no effect yet.

**The PRD is normative, not aspirational.** Its own §0: *"This is a build spec, not a proposal."* §7 says: *"Implementations must match these formulas exactly; deviations are bugs."* Appendix A (D1–D18) records decisions that are **made** — do not re-survey them, re-open them, or "improve" on them without the user explicitly reversing the decision.

Appendix B splits claims into **Verified correct**, **Corrected during verification**, and **`UNVERIFIED`**. The "Corrected" list is a list of errors already made once — reintroducing any of them is a regression. Never launder an `UNVERIFIED` number into code, tests, docs, or the paper.

## What glotscope is

A `pip install`-able Python library + CLI computing multilingual tokenizer diagnostics across four **tiers**, distinguished by *what they require*, not by what they measure. The tier model (§6, D4) is the whole architecture:

| Tier | Requires | Cost | Output |
|---|---|---|---|
| 0 | tokenizer only | ms | vocab lint, UTF-8 integrity, unreachable tokens, byte-fallback coverage |
| 1 | tokenizer + corpus | s–min | fertility, CPT/BPT/CTC, parity, Gini, Rényi, STRR, morphology, round-trip |
| 2 | tokenizer + **embedding tensors** | s | under-trained-token indicators |
| 3 | tokenizer + **full inference** | hrs + GPU | **specified, NOT implemented in v1.0** |

Tier 2 needs only `E_in`/`E_out` read from `safetensors` — no model instantiation. Spanning Tier 1 + Tier 2 in one package is the entire differentiator (§2.2) and is what makes §14's research question askable.

Anything requiring a forward pass is Tier 3 and out of scope. When a published reference number turns out to be Tier 3 (e.g. Land & Bartolo's *confirmed* counts 3161/49/6), validate against the Tier 0+2 quantity instead (*candidate sets* 5117/999/1280) — see D18.

## Toolchain (committed by the PRD — do not substitute)

No commands exist yet. When scaffolding, use exactly these; §12.4 and §13 pin them:

- `pytest` — tests; `hypothesis` for the property tests in §12.2
- `mypy --strict`
- `ruff` (not black/flake8/isort)
- Coverage gate **≥85% line coverage** (G1)
- CI matrix: Python **3.10–3.13** × {ubuntu, macos, windows}
- v2 Rust: `maturin` mixed layout, `python-source = "python"`, `module-name = "glotscope._core"`, `abi3-py310`, PyO3 0.29 post-0.26 idioms (`Python::attach`, never `with_gil`), `tokenizers = { version = "0.23", default-features = false }`
- Benchmarks: `criterion` (Rust), `pytest-benchmark` parametrised over both backends, `pytest-codspeed` for CI regression gating

Two CI jobs are load-bearing and easy to forget:
1. **`glotscope verify` against a committed `result.json`** — from v1, not v2 (§12.3). Without it G4 is an untested aspiration.
2. **Nightly leaderboard re-run against pinned revisions that fails if any published number moves** — silent upstream tokenizer changes are otherwise undetectable.

Extras: all segmenters are optional (`pip install glotscope[segmenters]`). MeCab needs a native build, PyICU needs system ICU. G1 promises green Windows CI for the **core install only**; segmenter tests skip-with-message.

## Invariants that a naive implementation will violate

These are the traps §7 exists to prevent. Each one is a silently-plausible wrong answer.

**Refuse rather than guess.** The library's credibility comes from typed refusals, not from always returning a number:
- `CapabilityError` — corpus declares `is_parallel` / `has_word_segmentation` / `has_morph_gold` / `is_wordlist`; parity on a monolingual corpus raises (D5). Gating is on corpus *capabilities*, never corpus identity.
- `IncomparableError` — `compare` refuses to table results computed under different segmenters, α values, Rényi normalizers, or language sets. The error message should explain that this is deliberate.
- `NoReferenceSetError` — Tier 2 `t_ref` fallback chain exhausted. Never fall through to an empty mean.
- `TypologicalScope.OUT_OF_SCOPE` — morphology on Semitic root-and-pattern or isolating languages returns this, not a number, *even though the reference implementation publishes numbers there*. Log as a deliberate divergence.

**Fertility (§7.1)** — `Segmenter` is required with **no default** (D6). `UD_GOLD` is not the same operation as `UDPIPE`/`STANZA` and is legal only on corpora with gold word boundaries — requesting it on FLORES+ raises. Record the segmenter *model* version, not a treebank release. Apply the >10% UNK exclusion rule. Prefer parity for cross-lingual claims; fertility is within-language, cross-tokenizer only.

**Parity (§7.3)** — **ratio of means**, not mean of per-sentence ratios (D7). Only the ratio of means maps to API cost. Keep `premium_{A|B}` (per-pair, definitional only) and `parity_ℓ` (corpus-level, what everything downstream computes) strictly distinct. Report worst-case parity alongside English-relative.

**Gini (§7.4)** — ascending sort (descending is equally order-invariant and just negates — the range check `∈ [0,1]` is what catches it). Cost unit is **tokens per aligned line**. Not comparable across different language sets.

**Rényi (§7.5)** — α required explicit and recorded; the reference implementation defaults to 3.0, the paper says 2.5. `normalizer` defaults to `"observed"` (D17) because that is what the only published implementation does. Special-case α = 1 to Shannon. Ship as **supplementary, not headline** (D8) — two published counterexamples raise efficiency while lowering BLEU.

**STRR (§7.6)** — emit both `strr_bare` and `strr_leading_space`; refuse to emit one unqualified number.

**Morphology (§7.7)** — three measures side by side (v1, v2, full-alignment vs MorphyNet). **Precision is non-optional in the return type** (D11); recall alone rewards oversegmentation and reporting it alone is misinformation. Carry the null result: this is a descriptive linguistic property, never a quality proxy.

**UTF-8 (§7.8)** — three *disjoint* classes: `PARTIAL_UTF8`, `ILL_FORMED_NOT_PARTIAL`, `WELL_FORMED`. `IllFormedVocab` = first two. **§7.9 Stage 1 drops only `PARTIAL_UTF8`** — handing it the full ill-formed set over-excludes and the candidate-set denominators stop reproducing. Ship a family classifier + a rate, never a binary alarm.

**Tier 2 (§7.9)** — run **both** indicators when untied, report Spearman agreement, emit `LOW_CONFIDENCE` on disagreement (D10). First-PC removal implemented but **default OFF** (D9). Hard-refuse quantized or non-original dtypes — a 4-bit `E_in` destroys the L2 indicator. `top_pct` applied *after* Stage-1 exclusion; record pre- and post-exclusion counts.

**Never imply causation.** The library reports diagnostics. §7.2, §7.5 and §7.7 each carry contradicting evidence on downstream quality. The docs' front page must say so.

## Cross-cutting requirements

**Every result carries a manifest** (§9, G4): tokenizer revision SHA, `tokenizer.json` SHA-256, weight-shard SHA-256 + dtype, corpus version + SHA-256, segmenter + model version, every contested parameter, environment, backend. `glotscope verify` must regenerate the numbers bit-identically.

**The `warnings` array is load-bearing, not decorative** — any contested parameter choice from §7 emits one.

**`docs/divergences.md` is a deliverable, not a scratch file.** Where a reference value does not reproduce: document the discrepancy, never tune until it matches. §17 calls a silently-tuned value misconduct. G3's exit condition (§12.1) is that **every §7 subsection has either a reference test or a `divergences.md` entry** — M3 cannot exit otherwise.

**Ship no corpora** (D12). Download recipes + SHA-256 + SPDX license field per resource, plus `--license-filter=commercial`. UD needs a per-treebank license audit (M0 deliverable).

**Mirror-sourced tokenizers** are pinned by commit revision, publish `tokenizer.json` SHA-256, and are **visibly labelled** in the leaderboard. Gated resources skip-with-message; never fail the run.

**The FFI boundary constrains v1 Python code** (§13, D3): aggregation is batch-oriented — `aggregate(list[list[int]]) → Stats`, never per-string. v1 must be written against the boundary v2 will implement.

**Performance claims** (§13): HF `tokenizers` is *already* Rust with rayon, so "I rewrote the hot path in Rust" is false by construction. Claim speedups over the **aggregation layer** only, in the same sentence as the number. Pin and state `TOKENIZERS_PARALLELISM`. Separate load time / encode throughput / aggregation throughput. Report tokens/sec and MB/sec on the multilingual corpus.

## The paper (§14) — one design decision dominates

**Script attribution is primary; corpus attribution is a demoted sanity check** (D14, reversed after verification). Under-trained tokens are by construction tokens the model never saw; FLORES+ devtest is clean translated prose, so corpus attribution would return ≈0 for every language and make the correlation undefined. Use Unicode Script (UAX #24) with `Common`/`Inherited` resolved via Script_Extensions. Define `UTR_ℓ` over the **vocabulary partitioned by script** so its denominator is decoupled from `parity_ℓ`'s numerator.

Analysis runs over **all 229 FLORES+ varieties**, not the 15-language core set (D16) — n=15 has no power once confounds are partialled out, and parity is segmenter-free. Pre-register the mixed model (language and model as crossed random effects), the SESOI, and the equivalence test **before** analysis; the "a null is also publishable" hedge fails without them. Run the label-permutation null. Do not pin the analysis to `top_pct=2.0` — report `UTR_ℓ` as a function of threshold.

The 15-language core set (§10.2) is for metrics needing per-language linguistic resources (fertility, morphology, STRR). Its typological rationale — 11 scripts, 4 morphological types, 3 with no whitespace — belongs in the docs verbatim; every cell of its "why" column is a testable hypothesis.

## PRD section map

Grep the PRD rather than re-reading 77 KB:

`§2.2` competitive landscape · `§3.2` non-goals (refuse without re-litigating) · `§6` tier model · `§7.1–7.9` normative metric formulas · `§8` public Python + CLI API · `§9` manifest JSON schema · `§10` corpora, 15-language core set, segmenters, licensing · `§11` tokenizer roster by access tier · `§12.1` reference-value tests · `§12.2` property tests · `§13` Rust/PyO3 architecture and honest perf framing · `§14` paper design · `§15` milestones M0–M5 and the cut order · `§16` leaderboard/docs/demo · `§17` risk register · `App. A` decisions D1–D18 · `App. B` verification status

## Schedule constraints that change what to build

Milestones have **binary** exit criteria (§15) — nothing is done on judgment. Ship order is front-loaded: **v0.1.0 to PyPI at M1 (18 Sep 2026)**, before the paper and before Rust (D15). Under schedule pressure the cut order is fixed: **M5 (Rust) → extended language set → HF Space → M4 model count (floor 8) →** *never* M1 or M2.

Cheapest high-value work, in order: the two Zouhar Rényi reference values (§12.1, tolerance 1e-9, **no external dependency**), the hand-built UTF-8 vocabulary test, and the `parity_L(L) = 1.0` identity. Reserve the PyPI name with a 0.0.0 placeholder in week 1 — `tokscope` was taken in July.
