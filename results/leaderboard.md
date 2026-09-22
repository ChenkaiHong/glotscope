# glotscope leaderboard

These are **diagnostics, not quality predictions.** The literature does not support the claim that any metric here predicts downstream model quality, and in several places contradicts it. A row ranking above another is not a better model.

## What this was computed under

- **Corpus** — `flores_plus` 2024.08 `devtest`, 15 languages, sha256 `99af3d81c8653af96260b59f9e4c4bef7effe065e96c86f676a6b515f6b61482`
- **Segmenter** — none (segmenter-free metrics only)
- **Parity reference** — eng_Latn
- **Rényi** — alpha 2.5, normalizer observed
- **Normalization** — NFC, leading space True, special tokens False
- **glotscope** — 0.1.0, backend python
- **Rows** — 13 published, 3 skipped

Every row carries its full manifest in `leaderboard.json` beside this file.

| Model | Vocab | Ill-formed | CPT | Parity (worst) | Gini | Tier 2 | Notes |
|---|---|---|---|---|---|---|---|
| o200k_base | 200,000 | 0.78% | 2.540 | 13.700 | 0.459 | n/a (tokenizer-only) |  |
| o200k_harmony | 201,088 | 0.78% | 2.540 | 13.700 | 0.459 | n/a (tokenizer-only) |  |
| cl100k_base | 100,261 | 0.77% | 1.739 | 14.979 | 0.441 | n/a (tokenizer-only) |  |
| r50k_base (GPT-2) | 50,257 | 0.68% | 1.368 | 18.689 | 0.411 | n/a (tokenizer-only) | the deliberately-bad multilingual baseline |
| GPT-2 | 50,257 | 0.68% | 1.368 | 18.689 | 0.411 | not run (no weights configured) | same vocabulary as r50k_base, loaded through a different library |
| Qwen3-8B | 151,669 | 0.95% | 2.105 | 10.432 | 0.414 | not run (no weights configured) |  |
| DeepSeek-V3 | 128,815 | 1.14% | 2.258 | 7.967 | 0.369 | not run (no weights configured) |  |
| phi-4 | 100,352 | 0.77% | 1.739 | 14.979 | 0.441 | not run (no weights configured) |  |
| gpt-oss-20b | 200,019 | 0.78% | 2.540 | 13.700 | 0.459 | not run (no weights configured) |  |
| BLOOM | 250,680 | 0.58% | 2.319 | 12.621 | 0.483 | not run (no weights configured) |  |
| NLLB-200 | 256,204 | 0.00% | 2.999 | 2.479 | 0.138 | not run (no weights configured) |  |
| XLM-R | 250,002 | 0.00% | 3.102 | 4.420 | 0.195 | not run (no weights configured) |  |
| Aya-101 | 250,100 | 0.00% | 2.641 | 3.283 | 0.210 | not run (no weights configured) |  |
| mT5 | — | — | — | — | — | — | **skipped** — TokenizerLoadError: tokenizer 'google/mt5-base' could not be read: no tokenizer.json in this repository (404 Client Error. (Request ID: Root=1-6ab2d8da-15bbb44f151c65f66d9497cc;9966af3c-8dc4-4db0-a817-ca107a2acbea)  Entry Not Found for url: https://huggingface.co/google/mt5-base/resolve/2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f/tokenizer.json.). A SentencePiece-only model would have to be converted, and implementing a tokenizer is §3.2's first non-goal |
| ByT5 | — | — | — | — | — | — | **skipped** — TokenizerLoadError: tokenizer 'google/byt5-small' could not be read: no tokenizer.json in this repository (404 Client Error. (Request ID: Root=1-6ab2d8da-2b09f94f2bac1a176891e009;e2a1aa28-4f3a-4528-82d2-7aa499416a91)  Entry Not Found for url: https://huggingface.co/google/byt5-small/resolve/68377bdc18a2ffec8a0533fef03b1c513a4dd49d/tokenizer.json.). A SentencePiece-only model would have to be converted, and implementing a tokenizer is §3.2's first non-goal |
| Glot500 | — | — | — | — | — | — | **skipped** — TokenizerLoadError: tokenizer 'cis-lmu/glot500-base' could not be read: no tokenizer.json in this repository (404 Client Error. (Request ID: Root=1-6ab2d8da-650e57ea24b072a6696404cd;6ae8902c-067b-4886-9b20-341504ae31e1)  Entry Not Found for url: https://huggingface.co/cis-lmu/glot500-base/resolve/d4d7c1ec01828fdf7452a4ccf7b55177aced175e/tokenizer.json.). A SentencePiece-only model would have to be converted, and implementing a tokenizer is §3.2's first non-goal |

`Tier 2` reads *n/a (tokenizer-only)* for an encoding, which names a tokenizer and no checkpoint. It reads *not run (no weights configured)* where the row names a model and this board reads none of its weights — a gap in the board, not a property of the model, and not a failed measurement.

`CPT` is the **mean** characters-per-token across the languages measured; the per-language values are in `leaderboard.json`.
