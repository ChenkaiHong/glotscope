# M0 source audit

This record freezes the five source-dependent decisions that gate implementation. Findings are pinned
to upstream revisions or releases; inference is labelled. No upstream code was copied into glotscope.

## U1: TokEval compression rate

**Decision:** compute the ratio of sums:

```text
CR = sum(measured text units) / sum(tokens)
```

TokEval accumulates a text-unit count and token count for each valid record, then divides the global
totals. It excludes blank text, zero-unit text, and empty tokenizations. Its default text measurement
is UTF-8 bytes, so default CR is numerically the same quantity as BPT. Glotscope records the selected
unit and defaults to bytes; it does not implement the previously planned mean of per-segment ratios.

- Upstream revision: [`7980633`](https://github.com/cimeister/tokenizer-intrinsic-evals/tree/798063302bc1d96a75fb963bcb91c9aab53ee9a1)
- Formula: [`information_theoretic.py` lines 241–324](https://github.com/cimeister/tokenizer-intrinsic-evals/blob/798063302bc1d96a75fb963bcb91c9aab53ee9a1/tokenizer_analysis/metrics/information_theoretic.py#L241-L324)
- Default measurement: [`information_theoretic.py` lines 53–60](https://github.com/cimeister/tokenizer-intrinsic-evals/blob/798063302bc1d96a75fb963bcb91c9aab53ee9a1/tokenizer_analysis/metrics/information_theoretic.py#L53-L60)

## U2: full-alignment F1

**Decision:** use micro-aggregated exact-boundary precision, recall, and F1.

The ConfoundingFactors implementation converts candidate and reference segmentations into boundary
positions, accumulates true/predicted/reference counts across examples, and computes F1 from the
aggregate precision and recall. Therefore:

- `gathered` vs `g/a/t/h/e/r/e/d`: TP=1, predicted=7, reference=1, F1=`0.25`.
- `araba/lar/ı` vs either one-boundary candidate: TP=1, predicted=1, reference=2, F1=`2/3`.

The paper's Turkish toy-table value `0.5` equals recall, conflicts with its `gathered` row, and is not a
gate. This is a source inconsistency, not an alternate formula.

- Paper: [Poelman et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.369/)
- Upstream revision: [`1ac60cb`](https://github.com/LAGoM-NLP/ConfoundingFactors/commit/1ac60cb9b57995186255b307d964779305b75a3d)
- Caller: [`morphological_alignment.py`](https://github.com/LAGoM-NLP/ConfoundingFactors/blob/1ac60cb9b57995186255b307d964779305b75a3d/scripts/morphology/morphological_alignment.py)
- Boundary comparison: [`TkTkT morphological.py`](https://github.com/bauwenst/TkTkT/blob/master/src/tktkt/evaluation/morphological.py)

## U3: parity-aware BPE artifacts

**Decision:** drop the numerical Foroutan Gini rows as reproduction gates.

The official repository publishes code that trains from caller-provided corpora and writes local
merges/vocabulary files. It publishes no trained tokenizer artifacts, hashes, or immutable Hugging
Face revisions. The current paper's 128k, unbalanced-30-language FineWeb2 table reports classical BPE
`0.064` and base PA-BPE `0.007`; it does not contain the older planned UnigramLM `0.094` or PA-BPE
`0.011` rows. Reconstructing and tuning an unpinned tokenizer would not be a reproduction.

- Repository revision: [`c7395d4`](https://github.com/swiss-ai/parity-aware-bpe/tree/c7395d469a653f3e48e80984e37593256ff8f365)
- Training instructions: [repository README](https://github.com/swiss-ai/parity-aware-bpe#usage-instructions)
- Paper: [Foroutan et al., ACL 2026](https://aclanthology.org/2026.acl-long.342/)
- Hugging Face paper record: [no linked models or datasets](https://huggingface.co/papers/2508.04796)

## U4: TokEval license

**Decision:** the pinned source is MIT licensed.

The repository API still classifies the license as `NOASSERTION`, but the pinned tree contains a
top-level MIT license explicitly covering the repository source code. Corpus and tokenizer assets
downloaded at runtime retain their own licenses. This resolves source-code reuse permissions, though
glotscope continues to implement formulas independently.

- License at pinned revision: [`LICENSE`](https://github.com/cimeister/tokenizer-intrinsic-evals/blob/798063302bc1d96a75fb963bcb91c9aab53ee9a1/LICENSE)

## U5: Universal Dependencies licenses

**Decision:** audit every treebank in official UD 2.18 and fail closed.

The official release archive contains 353 treebanks. The generated audit records each README and
license-file hash, raw license, normalized identifier, `Includes text` value, whether the README
declaration is evidenced by `LICENSE.txt`, and a conservative commercial-use classification:

| Classification | Treebanks |
|---|---:|
| Commercial-compatible | 268 |
| Noncommercial | 31 |
| Manual review | 54 |

Fifty README declarations do not match the license evidenced by their `LICENSE.txt`; another four use
`LGPL-LR` or `C-UDA 1.0`. All 54 are excluded by a commercial-only filter until explicitly reviewed.
The 31 matching Creative Commons NonCommercial rows are excluded. This catches real upstream metadata
conflicts rather than treating a hashed-but-unread license file as agreement. A permissive or copyleft
license label does not erase underlying-text caveats; users must retain the per-treebank license and
attribution.

- Official release: [UD 2.18, handle 11234/1-6149](https://hdl.handle.net/11234/1-6149)
- Archive SHA-256: `a93fe8520bc4c5ff34670d9a93a5a7689c018c1e59643fa27e03036717841b8a`
- Generated audit: [`data/ud-license-audit.json`](../data/ud-license-audit.json)
- Reproducer: [`scripts/audit_ud_licenses.py`](../scripts/audit_ud_licenses.py)
- Official rule: [use each treebank's `LICENSE.txt`](https://universaldependencies.org/contributing/licensing.html)

Reproduce after downloading the official archive:

```shell
python scripts/audit_ud_licenses.py ud-treebanks-v2.18.tgz \
  --release 2.18 \
  --source-url https://hdl.handle.net/11234/1-6149 \
  --expected-sha256 a93fe8520bc4c5ff34670d9a93a5a7689c018c1e59643fa27e03036717841b8a \
  --output data/ud-license-audit.json
```

The archive is a verification input, not a repository artifact. Glotscope ships the audit and recipe,
not the corpus.
