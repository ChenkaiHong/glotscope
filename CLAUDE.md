# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Always: update `HISTORY.md`

`HISTORY.md` (repo root, gitignored) is the narrative log of everything that has happened here. **Append to it before ending any turn in which you created, edited, deleted, or verified something.** A turn that changes nothing in the repo needs no entry.

Newest first, under a `## YYYY-MM-DD` heading, absolute dates only. Record **what changed, why, and how it was verified** — and record decisions *not* to act with the same weight, since those are what get forgotten and re-litigated. Never upgrade "wrote it" to "it works" without having run something. Read the file's header before appending; it states the conventions.

Being gitignored is the point: it holds the reasoning, dead ends and reversals a commit log cannot.

## Commands

`uv` is the environment tool; CI runs exactly these steps (`.github/workflows/ci.yml`).

```bash
uv pip install --editable ".[dev]"

uv run --no-sync ruff check python/glotscope tests
uv run --no-sync ruff format --check python/glotscope tests
uv run --no-sync mypy --strict python/glotscope tests
uv run --no-sync pytest --cov=glotscope --cov-report=term-missing -q   # gate: 85%
```

Baseline as of 16 Aug 2026: **280 tests pass, 97.13% coverage** (264 + 16 skipped without the segmenter extras), ruff/format/mypy clean. Coverage below 85% fails the run, so `--cov` is not optional when judging a change.

`.[dev]` installs the extras too. The **core install is `tokenizers` alone** — `numpy`/`safetensors` are the `tier2` extra and `tiktoken` is its own, because G1's clean-install promise is measured on a core install that only claims Tier 0 and Tier 1. Import either inside the function that needs it and name the extra when it is missing.

Single test / subset:

```bash
uv run --no-sync pytest tests/test_renyi.py -q
uv run --no-sync pytest -m "reference and not network and not gated" -q
```

The local venv is 3.13, and **the floor is where the version-dependent failures are**. Before pushing
anything that touches a dataclass default, a stdlib call, or a typing construct, run the floor too:

```bash
uv run --python 3.10 --isolated --with-editable ".[dev]" pytest -q
```

A `MappingProxyType` dataclass default cost a full red matrix this way: `dataclasses` rejects any
default whose `__hash__` is `None` on 3.10/3.11, and 3.12 relaxed the check. Green on 3.13, `ValueError`
at import time on half the cells.

Markers (`--strict-markers` is on, so an unregistered marker is an error): `reference`, `property`, `segmenter`, `gated`, `network`.

Packaging — never build or upload by hand; the `Makefile` chains the guards:

```bash
make build          # uv build into dist/
make leak-check     # build + assert no PRD/HISTORY/.env/key/pyc in the artifacts
make twine-check    # leak-check + twine 7.0.0 check
make upload                     # -> testpypi (default)
make upload REPOSITORY=pypi     # -> PyPI
```

`make leak-check` is mandatory after any change to `pyproject.toml` packaging keys. A published sdist cannot be retracted; `include` is an allowlist and `exclude` is a second guard.

## Repository state (14 Aug 2026)

M0 is frozen and merged. Git repo with public remote `ChenkaiHong/glotscope`; **the default branch is `foundation`, not `main`** — CI push events trigger on `foundation` and PRs target it. `master` is a stale local branch.

Implemented and tested: `metrics.py` (Rényi, parity, Gini), `utf8.py` (three-class classification + `Tier0Report` assembly), `lint.py` (Tier 0 vocabulary lint), the four `aggregate.py` folds, `compression.py` (CPT/BPT/CTC + CR), `strr.py`, `roundtrip.py`, all three `Tokenizer.from_*` loaders and `analyze` end to end, the contract layer (`enums`/`errors`/`results`/`report`/`manifest`/`corpus`/`tokenizer`/`embeddings`), and `scripts/audit_ud_licenses.py` + `data/ud-license-audit.json` (353 UD 2.18 treebanks; fail-closed README/`LICENSE.txt` agreement — 268 commercial, 31 noncommercial, 54 manual review).

`glotscope lint` and `glotscope analyze` are live and produce a §9 document. **Exit codes are part of the interface**: `0` produced a document, `1` is a typed refusal, `2` is scheduled but not built. A mistyped path is `1`; so is `--revision` beside a local path or a `tiktoken:` encoding, because both loaders exist now and "scheduled for a later release" would send the reader after something already built. Only `leaderboard` still exits `2`.

**Three loaders, three provenance stories.** `from_file` records `revision="local"`; `from_pretrained` resolves the repo *before* fetching, so the artifact and the recorded SHA are the same commit; `from_tiktoken` has neither a `tokenizer.json` nor a commit, so `tokenizer_json_sha256` is a digest over the encoding's own definition (merge ranks, special tokens, split pattern, vocabulary size — the name deliberately excluded) and `revision` is the pinned `tiktoken` version. An OpenAI encoding reaches Tier 0/1 through `tiktoken_backend.TiktokenBackend`, an adapter presenting the `tokenizers` surface — **not** a conversion, because building a `tokenizers` BPE from `mergeable_ranks` means implementing a tokenizer (§3.2's first non-goal) and a mistranslated split rule shifts every Tier 1 number with nothing to notice it by.

Still `NotImplementedError`: the `leaderboard` handler, and the `STANZA`/`UDPIPE` segmenter adapters. `Report.from_json` is blocked on `Tier0Report.to_dict` being lossy — it drops the partial-UTF-8/unreachable/special id lists. No `tier0/`/`tier1/`/`tier2/` packages — deliberately not stubbed (`docs/build-order.md`); their contracts are already pinned by the `Tier1Report` methods and the `aggregate` boundary.

**G4 is closed.** `glotscope verify` regenerates a result from its manifest and compares, and the CI job runs it against the committed `verification/result.json` on **all twelve cells** — so the claim is that a published number reproduces on a different OS and Python, not merely where it was made. The artifact is passed as `--tokenizer` because §9 keeps filesystem paths out of the manifest: the document records what the artifact *is* (a SHA-256), not where it lives, and that hash is checked before anything is recomputed. Environment is excluded from the comparison and printed instead — it is recorded *because* it varies, so demanding it match would make a result verifiable only on the machine that produced it.

`verification/` holds the fixture: four invented sentences under a `verification_fixture` registry entry (CC0), plus the tokenizer and the result. It is the one exception to D12 and an exception in name only — a verification job with no inputs cannot run. Its id is its own rather than borrowed from a real corpus, so the manifest tells the truth about what was measured. `.gitattributes` marks `verification/**` as `-text`, because the digest is over the bytes on disk and a CRLF checkout on Windows would fail the job for a reason unrelated to any number. The fixture sits outside the sdist allowlist, so `tests/test_g4_verification.py` runs from a checkout and skips from an unpacked release.

Not yet written: the nightly leaderboard re-run, `leaderboard.yaml`, `results/`. The 12-cell quality matrix (3.10–3.13 × {ubuntu, macos, windows}) is implemented and green.

Blocking unknowns U1–U5 are **resolved** — evidence in `docs/m0-source-audit.md`, sequencing in `docs/build-order.md`, discrepancies in `docs/divergences.md`. Two PRD items remain `UNVERIFIED` and must not be cited: the gated Command-R / Command-A / Aya Expanse / Gemma 2 vocab sizes, and the Phi-3/3.5 and ByT5 vocab sizes.

PyPI name reserved 10 Aug 2026 with a 0.0.0 placeholder. **v0.1.0 is prepared and not yet uploaded** — `make upload REPOSITORY=pypi` needs Kai's token, and a published sdist cannot be retracted.

`verify` compares the **numbers**, not the producer: `glotscope_version` and `backend` are reported rather than compared, because comparing them would make every release invalidate every result published before it. `schema_version` *is* compared — a schema change changes the document. The committed fixture is deliberately left at `0.0.0`, so every CI run asserts that a result published by an earlier release still regenerates.

## Private documents live in a sibling repo

`glotscope-PRD.md` and `HISTORY.md` in the repo root are **symlinks** into `../glotscope-internal/` — a separate private git repo (`ChenkaiHong/glotscope-internal`, branch `docs/internal-record`). Editing either writes into that repo and needs its own commit and push there; nothing in this tree records their content. Both paths are gitignored here and excluded from every distribution — the PRD was once committed and had to be purged from history, which is why the leak check exists.

`PROGRESS.md` (gitignored) is the short live state file: Done / Next / Blocked. Keep it current during long or unattended work — `AGENTS.md` requires it alongside `HISTORY.md`.

## Normative sources

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

§12.4 and §13 pin these; the first five are already wired into `pyproject.toml` and CI:

- `pytest` — tests; `hypothesis` for the property tests in §12.2
- `mypy --strict`
- `ruff` (not black/flake8/isort — this overrides the global Python style rules)
- Coverage gate **≥85% line coverage** (G1)
- CI matrix: Python **3.10–3.13** × {ubuntu, macos, windows}
- v2 Rust: `maturin` mixed layout, `python-source = "python"`, `module-name = "glotscope._core"`, `abi3-py310`, PyO3 0.29 post-0.26 idioms (`Python::attach`, never `with_gil`), `tokenizers = { version = "0.23", default-features = false }`
- Benchmarks: `criterion` (Rust), `pytest-benchmark` parametrised over both backends, `pytest-codspeed` for CI regression gating

Two CI jobs are load-bearing and easy to forget:
1. **`glotscope verify` against a committed `result.json`** — from v1, not v2 (§12.3). **Done**: it runs on every cell of the quality matrix against `verification/result.json`.
2. **Nightly leaderboard re-run against pinned revisions that fails if any published number moves** — silent upstream tokenizer changes are otherwise undetectable.

Extras: all segmenters are optional (`pip install glotscope[segmenters]`). MeCab needs a native build, PyICU needs system ICU. G1 promises green Windows CI for the **core install only**; segmenter tests skip-with-message.

## How the package is wired

`python/glotscope/` (maturin mixed layout — the directory is `python/` now so the v2 switch is a build-backend change, not a directory move):

| Module | Role |
|---|---|
| `enums.py` | closed vocabularies; **the string values ARE the §9 manifest schema** — renaming one breaks committed `result.json` |
| `errors.py` | the typed refusals. Nothing here is a fallback path; catching one and substituting a default is the bug it exists to prevent |
| `aggregate.py` | the v1↔v2 FFI boundary: ints in, frozen struct out, one call per batch, never a `str` |
| `results.py` | metric results carrying their own comparability parameters (what `IncomparableError` checks against) |
| `report.py` | `Tier0/1/2Report` + the spanning `Report` |
| `manifest.py` | §9 provenance + `canonical_json` for bit-identical reproduction |
| `corpus.py` | capability gating + the §10.1 registry |
| `tokenizer.py` / `embeddings.py` | the §8.1 entry points; `Embeddings` refuses quantized dtypes |
| `metrics.py` | pure Tier 1 calculations, no I/O |
| `compression.py` | CPT/BPT/CTC and the TokEval-frozen compression rate |
| `segmenters/` | one adapter per `Segmenter` member; refuses a missing extra and a language-scoped segmenter used elsewhere, and never falls back to whitespace |
| `fertility.py` | fertility, continuation rate, and the >10% UNK exclusion rule |
| `strr.py` / `roundtrip.py` | STRR under both conventions; round-trip losslessness |
| `utf8.py` | Tier 0 UTF-8 classification |
| `lint.py` | Tier 0 vocabulary lint: unreachable ids, special ids, byte-fallback coverage, family/algorithm inference |
| `cli.py` | the six §8.2 subcommands via stdlib `argparse` (no CLI framework — the core dependency list is load-bearing for G1) |

Three invariants are enforced structurally rather than by comment, and are already tested — preserve the shape, not just the behaviour:

- `Tier0Report.stage1_exclusions()` unions exactly partial-UTF-8 + unreachable + special. The wider ill-formed set is not reachable from it, so §7.9 Stage 1 cannot over-exclude by accident.
- `BoundaryCounts` exposes no recall without precision.
- `MorphologyResult` raises if an `OUT_OF_SCOPE` language carries a measure value.

`glotscope.backend()` reads `GLOTSCOPE_IMPLEMENTATION` **before** attempting any import, and raises rather than silently downgrading `rust` → `python`. A silent downgrade would make v2 backend-parity CI vacuous; keep that ordering.

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

**Ship no corpora** (D12). Download recipes + SHA-256 + SPDX license field per resource, plus `--license-filter=commercial`. The UD per-treebank audit is done for 2.18 (`data/ud-license-audit.json`, regenerated deterministically by `scripts/audit_ud_licenses.py` — regenerate rather than hand-edit, and a new UD release means a new audit).

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

Five §12.1 rows are **done** (Zouhar Rényi pair at 1e-9, hand-built UTF-8 vocabulary, `parity_L(L) = 1.0`, `Gini([1,2,3,4,5]) == 4/15`, and §7.2's compression family against TokEval at 1e-6), as is the PyPI reservation. Next work is selected from the frozen order in `docs/build-order.md`: Tier 0 (`lint.py`, `Tokenizer.from_*`) needs no external data and is the head of the queue; Tier 1 segmenter-free work needs FLORES+ (gated), and Tier 1 word-level work needs the segmenter extras.

Branch per task off `origin/foundation`, and do not push to `foundation` directly — the reviewed history goes through PRs.
