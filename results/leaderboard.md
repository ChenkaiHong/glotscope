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
| GPT-2 | 50,257 | 0.68% | 1.368 | 18.689 | 0.411 | n/a (tokenizer-only) | same vocabulary as r50k_base, loaded through a different library |
| Qwen3-8B | 151,669 | 0.95% | 2.105 | 10.432 | 0.414 | n/a (tokenizer-only) |  |
| DeepSeek-V3 | 128,815 | 1.14% | 2.258 | 7.967 | 0.369 | n/a (tokenizer-only) |  |
| phi-4 | 100,352 | 0.77% | 1.739 | 14.979 | 0.441 | n/a (tokenizer-only) |  |
| gpt-oss-20b | 200,019 | 0.78% | 2.540 | 13.700 | 0.459 | n/a (tokenizer-only) |  |
| BLOOM | 250,680 | 0.58% | 2.319 | 12.621 | 0.483 | n/a (tokenizer-only) |  |
| NLLB-200 | 256,204 | 0.00% | 2.999 | 2.479 | 0.138 | n/a (tokenizer-only) |  |
| XLM-R | 250,002 | 0.00% | 3.102 | 4.420 | 0.195 | n/a (tokenizer-only) |  |
| Aya-101 | 250,100 | 0.00% | 2.641 | 3.283 | 0.210 | n/a (tokenizer-only) |  |
| mT5 | — | — | — | — | — | — | **skipped** — TokenizerLoadError: tokenizer 'google/mt5-base' could not be read: no tokenizer.json in this repository (404 Client Error. (Request ID: Root=1-6a939e5a-680ab97c796cf7713dff9136;16a923b6-6be0-48f1-96aa-0d795dac5ba6)

Entry Not Found for url: https://huggingface.co/google/mt5-base/resolve/2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f/tokenizer.json.). A SentencePiece-only model would have to be converted, and implementing a tokenizer is §3.2's first non-goal |
| ByT5 | — | — | — | — | — | — | **skipped** — TokenizerLoadError: tokenizer 'google/byt5-small' could not be read: no tokenizer.json in this repository (404 Client Error. (Request ID: Root=1-6a939e5a-18d007db34219ab007dd7472;8538afe5-95aa-45ab-8894-e0f9cbd94c2b)

Entry Not Found for url: https://huggingface.co/google/byt5-small/resolve/68377bdc18a2ffec8a0533fef03b1c513a4dd49d/tokenizer.json.). A SentencePiece-only model would have to be converted, and implementing a tokenizer is §3.2's first non-goal |
| Glot500 | — | — | — | — | — | — | **skipped** — TokenizerLoadError: tokenizer 'cis-lmu/glot500-base' could not be read: no tokenizer.json in this repository (404 Client Error. (Request ID: Root=1-6a939e5a-6e6bccd37044f23b0535d962;2fd72768-296c-404b-8a0c-cfb796ba2967)

Entry Not Found for url: https://huggingface.co/cis-lmu/glot500-base/resolve/d4d7c1ec01828fdf7452a4ccf7b55177aced175e/tokenizer.json.). A SentencePiece-only model would have to be converted, and implementing a tokenizer is §3.2's first non-goal |

`Tier 2` reads *n/a (tokenizer-only)* where a row has no open weights to read. That is a property of the model, not a failed measurement.

`CPT` is the **mean** characters-per-token across the languages measured; the per-language values are in `leaderboard.json`.
