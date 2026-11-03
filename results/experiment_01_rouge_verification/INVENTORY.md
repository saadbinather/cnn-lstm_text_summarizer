# Experiment 01 inventory

## Required PubMed inference artifacts

| Artifact | State |
|---|---|
| `hierarchical_cnn_lstm_summarizer.pth` | present, 1,786,649 bytes |
| `local_dataset/test/data-00000-of-00001.arrow` | present, 110,637,344 bytes |
| `embeddings/test_labels.pkl` | present, 905,898 bytes |
| `embeddings/test_mask_chunk_0.pt` | present, 6,555,247 bytes |
| `embeddings/test_mask_chunk_1.pt` | present, 5,736,047 bytes |
| `embeddings/test_emb_chunk_0.pt` | **missing** |
| `embeddings/test_emb_chunk_1.pt` | **missing** |

## Dataset artifacts

- test Arrow dataset: present, 5,025 rows;
- validation Arrow dataset: present, 4,676 rows;
- train metadata: present, declares 83,233 rows;
- train Arrow dataset: missing;
- archived label counts: 10,000 train, 3,000 validation, 3,000 test;
- retained embedding tensors: only `train_emb_chunk_6.pt` (400 × 512 × 768); no validation or test embedding tensor remains.

## Checkpoints

All five original checkpoint files remain unchanged at repository root:

- `cnn_lstm_summarizer.pth`
- `cnn_only.pth`
- `hierarchical_cnn_lstm_summarizer.pth`
- `lstm.pth`
- `stacked_cnn_only.pth`

`results/checkpoints/` contains relative symlinks for navigation; it does not duplicate or replace the binaries.

## Code and dependency state

No Python source content was edited. Files were grouped into `src/` and `scripts/`. The existing ignored `venv/`, `local_dataset/`, `embeddings/`, and large clinical CSV were left in place. The current environment includes `rouge-score==0.1.2`, but historical run dependency versions remain unverified.

