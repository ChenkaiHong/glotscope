# Build order and open questions

Sequencing only. The normative internal specification is intentionally not distributed; formulas,
decisions and rationale are not restated here — two copies of a normative spec is the failure mode
§12.1 and `divergences.md` exist to prevent.

## Blocking unknowns

Five things had to be resolved by reading a source, not by reasoning. Each gated work downstream and
was investigated during M0 (window: **Mon 10 – Fri 21 Aug 2026**). The evidence and pinned revisions
are recorded in [`m0-source-audit.md`](m0-source-audit.md).

| # | Question | M0 outcome |
|---|---|---|
| U1 | TokEval compression-rate formula and unit | **Resolved.** Ratio of total measured units to total tokens; default unit UTF-8 bytes. |
| U2 | Poelman et al. full-alignment F1 | **Resolved.** Micro-aggregated exact-boundary precision/recall/F1. The Turkish toy value is an upstream inconsistency. |
| U3 | Published parity-aware BPE tokenizers | **Resolved unavailable.** Training code exists, but no trained artifacts or immutable revisions were published; numerical reference rows are not gates. |
| U4 | TokEval license | **Resolved.** The current pinned repository contains an MIT license covering its source code. |
| U5 | UD per-treebank licenses | **Resolved for UD 2.18.** The generated audit records all 353 treebanks; fail-closed README/`LICENSE.txt` agreement leaves 268 commercial-compatible, 31 noncommercial, and 54 for manual review. |

Two more items are `UNVERIFIED` in the PRD and must not be cited until checked: the gated Command-R /
Command-A / Aya Expanse / Gemma 2 vocabulary sizes, and the Phi-3/3.5 and ByT5 vocabulary sizes.

## Current state

Done — the contract layer, verified `ruff` clean and `mypy --strict` clean:

```
python/glotscope/
  errors.py       typed refusals: Capability, SegmenterRequired, Incomparable,
                  NoReferenceSet, UnsupportedCheckpoint
  enums.py        closed vocabularies; the string values ARE the §9 schema
  aggregate.py    the v1<->v2 FFI boundary (batch-in, frozen-struct-out)
  results.py      metric results carrying their own comparability parameters
  manifest.py     §9 provenance + canonical_json for bit-identical output
  report.py       Tier0/1/2 reports and the spanning Report
  corpus.py       capability gating + the §10.1 registry
  tokenizer.py    public entry point (§8.1)
  embeddings.py   two tensors, no model instantiation; refuses quantized dtypes
  cli.py          the six §8.2 subcommands
```

Three invariants are enforced structurally rather than by comment, and are already verified:

- `Tier0Report.stage1_exclusions()` unions exactly partial-UTF-8 + unreachable + special. The wider
  ill-formed set is not reachable from it, so §7.9 Stage 1 cannot over-exclude by accident.
- `BoundaryCounts` has no recall without precision, and reproduces `gathered → F1 0.25` exactly.
- `MorphologyResult` raises if an `OUT_OF_SCOPE` language carries a measure value.

## Order

Dependencies, not dates. Anything at the same indent level is parallelisable.

```
U1..U5 (blocking unknowns)
  └─ §7 spec freeze
       ├─ Tier 0                      <- no external data; start here
       │    ├─ utf8.py    three disjoint classes; hand-built vocab test        [done]
       │    ├─ lint.py    unreachable, byte-fallback coverage, family classifier [done]
       │    └─ Tokenizer.from_file [done] / from_tiktoken / from_pretrained + manifest
       │
       ├─ Tier 1 segmenter-free       <- needs FLORES+ (gated: HF_TOKEN or vendored subset)
       │    ├─ compression.py   CPT/BPT/CTC + CR (U1 resolved)
       │    ├─ renyi.py         <- two Zouhar reference values, NO external data
       │    ├─ parity.py        ratio of means
       │    └─ gini.py          ascending sort; property tests (U3 rows dropped)
       │
       ├─ Tier 1 word-level           <- needs segmenter extras
       │    ├─ segmenters/       one adapter per Segmenter member, model version pinned
       │    ├─ fertility.py
       │    ├─ strr.py           both conventions
       │    └─ morphology.py     three measures (blocked on U2 for gating)
       │
       ├─ Report.to_json / from_json  -> `glotscope verify` CI job  <- delivers G4
       │
       └─ Tier 2                      <- needs open weights
            ├─ reference_set.py   the three-link fallback chain
            └─ detect.py          both indicators + Spearman agreement
```

## Completed foundation

Completed in the original value-per-unit-of-blocked-ness order:

1. **The two Zouhar Rényi reference values** (§12.1, tolerance 1e-9). The cheapest reference test in
   the document and the only place a published number is reproducible from a literal string. Two
   assertions, and they exercise α handling, the `normalizer="observed"` default, and the α=1 special
   case.
2. **The hand-built UTF-8 vocabulary test** (§12.1). A constructed vocabulary with known ill-formed,
   unreachable and partial-UTF-8 counts. This is what protects the Tier 2 candidate-set reproduction —
   if Stage-1 exclusion is wrong, nothing downstream is real.
3. **`parity_L(L) = 1.0`** exactly, for every L, plus the equal-line-count assertion. Definitional,
   but it catches the dropped-line bug that silently breaks the ratio-of-means identity.
4. **`Gini([1,2,3,4,5]) == 4/15`** as a hard-coded value, plus the `in [0,1]` range check. The range
   check is the one that catches a descending sort; order-invariance does not.

The PyPI name was reserved on 10 August 2026.

## Cut order under schedule pressure

Fixed by §15: **M5 (Rust) → extended language set → HF Space → M4 model count (floor 8).** M1 and M2
are never cut. The package ships at M1 (**Fri 18 Sep 2026**), before the paper and before Rust.

## Not yet written

- The two load-bearing CI jobs are not yet written: `glotscope verify` against a committed
  `result.json`, and the nightly leaderboard re-run that fails if any published number moves. The
  3.10–3.13 × {ubuntu, macos, windows} quality matrix is implemented and green.
- `leaderboard.yaml`, `results/` — M3.
- Tier metric packages (`tier0/`, `tier1/`, `tier2/`) — deliberately not stubbed. The contracts are
  pinned by the `Tier1Report` methods and the `aggregate` boundary; empty files would be churn.
