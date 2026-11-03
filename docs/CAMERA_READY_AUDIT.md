# Camera-Ready Repository Audit

Audit date: 2026-10-19 (Asia/Karachi)  
Scope: local repository and already-present ignored artifacts  
Paper: [arXiv:2609.13481](https://arxiv.org/abs/2609.13481), v1  
Policy: no experiments rerun; no data downloaded; no remote state modified

## Executive finding

This checkout is **not sufficient to reproduce or substantiate the camera-ready claims**. The most consequential verified issues are:

1. PubMed “sentence embeddings” are actually the 512 token states of one document-level SciBERT encoding; token masks are used as sentence masks.
2. The main training/evaluation checkpoint has three sequential convolutional layers. The paper describes four.
3. `cnn_ablation.py` exposes depths 1–4 but defines only three convolutions; depths 3 and 4 execute identical code.
4. Clinical model evaluation uses a top-30%-with-minimum-3 rule, not the paper's dynamic mean-plus-standard-deviation threshold.
5. Clinical datasets are never split or used for training; each script evaluates a seeded 500-row sample from a training/full CSV.
6. There are no full training/evaluation logs, raw predictions, dependency lockfile, fixed random seeds, validation/checkpoint-selection code, or provenance metadata for reported results.

## 1. Dataset ledger

### PubMed

Evidence:

- `dataset.py:8-10` loads provider-supplied `train.jsonl`, `val.jsonl`, and `test.jsonl` separately. There is no repository-defined random split.
- Saved `dataset_info.json` files record upstream sizes of **83,233 train / 4,676 validation / 5,025 test**.
- `embeddings.py:23-25` takes `range(10000)`, `range(3000)`, and `range(3000)`: exact processed sizes are **10,000 / 3,000 / 3,000**, selected as the first records, with no sampling seed.
- `evaluate_baselines.py:17-20`, `bert.py`, `cnn.py`, and `cnn_ablation.py` use the first **300** test records. `evaluate_model.py:48` uses the first **500**. `ablation.py` uses 500.
- The retained validation and test Arrow files each contain 3,000 unique exact document texts and unique `text_id` tuples. Their cross-split overlap is **0 exact texts and 0 `text_id` tuples**.

Limitations:

- The train metadata claims 83,233 rows, but `local_dataset/train/data-00000-of-00001.arrow` is missing. Train-versus-validation/test document overlap is **NOT VERIFIED**.
- The data has no patient identifier. Patient overlap is **NOT VERIFIED / not measurable from available fields**.
- Immutable Hugging Face dataset revision/commit, source checksums, download date, license snapshot, and rationale for first-N subsampling are **NOT VERIFIED**.

Evidence needed: restore the exact train Arrow/JSONL artifact; record dataset revision and SHA-256 checksums; document what `text_id` represents; provide a split-overlap report over all source and summary identifiers.

### MIMIC-CXR

Evidence:

- The local cached CSV has **64,586 rows and 64,586 distinct `subject_id` values**.
- `mimic.py:65-69` and `mimicCXRbaseline.py` load `mimic_cxr_aug_train.csv` and call `df.sample(n=500, random_state=42)`.
- The exact seeded local sample contains **500 rows / 500 patients**. The parser used by the scripts accepts only the first report in each serialized list and requires literal `Impression:`; **353 rows / 353 patients** pass those conditions. There are 14 duplicated complete `text` list strings in the 500 sampled rows.
- There are no train, validation, or test partitions in repository code. Patient/document overlap across splits is therefore **not applicable**, not verified as zero.

Limitations:

- Dataset version is not pinned in code. The local cache is Kaggle version 2, but checkpoint/results provenance does not prove that version was used.
- The source file name says `train`, yet it is used as a test sample.
- Study-level grouping and the relationship between a row, its multiple reports, and its multiple images are undocumented.

Evidence needed: immutable dataset version/checksum, a patient-grouped split manifest, exact evaluated row IDs, parser acceptance log, and per-example predictions/references.

### MIMIC-IV BHC

Evidence:

- The local cached CSV has **269,516 rows**, **131,400 patients**, and **269,516 admissions**; input and target have zero nulls.
- `mimicIV.py` and `mimic4baseline.py:47-53` drop null input/target rows and sample **500** with `random_state=42`.
- The exact local sample contains **500 unique notes, 500 admissions, and 496 patients**. It contains no duplicated input strings.
- There are no train, validation, or test partitions and no clinical training code. Cross-split patient/document overlap is therefore **not applicable**.

Limitations:

- Dataset version is not pinned in code. The local cache is Kaggle version 1, but run provenance is absent.
- Four patients occur more than once in the evaluation sample, though admissions/notes are unique.

Evidence needed: immutable version/checksum, intended task provenance, patient-grouped split manifest, evaluated note IDs, and per-example predictions/references.

## 2. Preprocessing, labels, padding, and truncation

### PubMed path

| Item | Verified implementation | Verdict |
|---|---|---|
| Sentence splitting | None in repository preprocessing. Provider `text` is already a list of sentences. | Verified as provider-supplied; upstream splitter **NOT VERIFIED**. |
| Encoder input | `" ".join(sentences)` creates one full-document string (`embeddings.py:56`). | Conflicts with paper's per-sentence encoding. |
| Transformer truncation | One document encoding, `max_length=512`, `padding="max_length"` (`embeddings.py:57-63`). Units are subword tokens. | Verified; paper describes 512 sentences. |
| Saved representation | `out.last_hidden_state`, shape `(document, 512, 768)` (`embeddings.py:102-108`). | Token states, not sentence `[CLS]` states. |
| Saved mask | Transformer token attention mask, shape `(document, 512)`. | Later misused as a sentence mask. |
| Oracle labels | Dataset `score` array, defaulting to zeros only if absent (`embeddings.py:54`); saved unchanged. | Label generation algorithm/provenance **NOT VERIFIED**. |
| Label padding | Zeros to length 512; truncation at 512 (`train_model.py`). | Verified. |
| Loss masking | Positions whose token attention mask equals 1. | Misaligned with sentence labels. |
| Evaluation reference | Source sentences with score `>= 0.5` concatenated; dataset `summary` ignored. | Verified. |
| Valid sentence count | `min(len(original_sentences), token_mask.sum())`. | Token count is incorrectly treated as sentence count. |

Saved artifact counts corroborate code: label lists contain 10,000/3,000/3,000 entries; saved masks cover the same counts. Only training embedding chunk 6 remains locally; the other train embeddings and all validation/test embeddings are missing.

### Clinical paths

- `mimic.py` and `mimicIV.py` use `nltk.sent_tokenize` at inference.
- Each sentence is separately tokenized with dynamic batch padding, truncation enabled, and `max_length=128`; `[CLS]` (`last_hidden_state[:, 0, :]`) is retained.
- Documents are zero-padded to `(1, 512, 768)` or truncated to the first 512 sentences.
- MIMIC-CXR parses only the first list member, removes `Findings:`, splits on the first `Impression:`, and uses the latter as the reference.
- MIMIC-IV uses `input` as source and `target` as reference.
- NLTK resource versions and Punkt model data used in historical runs are **NOT VERIFIED**.

## 3. Training and checkpoint audit

### Main script (`train_model.py`)

- Adam learning rate: **1e-3**.
- Batch size: **16**.
- Epochs: **10**.
- Objective: PyTorch `BCELoss` after sigmoid, masked with `mask == 1`.
- Architecture: kernels **1, 2, 3 only**, 128 channels; one bidirectional LSTM with hidden size 128/direction.
- Device preference: MPS → CUDA → CPU.
- Checkpoint: saved once after the final epoch, overwriting `hierarchical_cnn_lstm_summarizer.pth`.
- Validation: none.
- Checkpoint selection: final epoch only; no metric selection.
- Early stopping: none.
- Seeds: none set for Python, NumPy, or PyTorch. DataLoader uses `shuffle=True` without a generator. Chunk enumeration derives from unsorted `os.listdir()` and is then accessed by numeric assumed names.
- Hardware actually used: **NOT VERIFIED**. Device-selection code is not a run log.
- Software versions used to create checkpoints: **NOT VERIFIED**.

The current local virtual environment contains Python 3.14.6, torch 2.13.0, transformers 5.14.1, datasets 5.0.0, rouge-score 0.1.2, NLTK 3.10.0, NumPy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, NetworkX 3.6.1, Sumy 0.12.0, and KaggleHub 1.0.2. These are an audit-time inventory only; nothing ties them to the July 2026 checkpoints.

### Architecture/checkpoint consistency

| Checkpoint | Verified state-dict structure | SHA-256 |
|---|---|---|
| `hierarchical_cnn_lstm_summarizer.pth` | sequential kernels 1–3 + BiLSTM | `c4a01ad6828fbead4042d9f5a01181aa445ae4638e70e3e6dc87c6ea9ed09a0e` |
| `cnn_lstm_summarizer.pth` | four parallel kernels over 768-channel input + BiLSTM | `5a9bd8456e1f609856a0185a7aa6a20276149179ac7d18f2ee0852ad80814c1f` |
| `stacked_cnn_only.pth` | sequential kernels 1–4, no LSTM | `ad3651298710467ae06b1aa92a77768f1abb55f1845abbfbcfaae65b52ed28ee` |
| `cnn_only.pth` | four parallel kernels, no LSTM | `709b0a075baa4c2c41b0026fe69ad5b4d00af93698d86fb4845b6a414d5e3bed` |
| `lstm.pth` | BiLSTM-only | `eae74c0751bef557714a1894bb301c5a033f278d6a3b7e61cd3d426162a394c2` |

The main saved checkpoint loads into the three-layer definitions in `train_model.py`, `evaluate_model.py`, `bert.py`, `mimic.py`, `mimicIV.py`, and `case_study.py`. It cannot contain a fourth convolution. `cnn_ablation.py` similarly defines only conv1–conv3, so `num_kernels=3` and `num_kernels=4` are operationally identical. Its four-depth output cannot validate a fourth layer.

Evidence needed: original immutable training commit, complete embeddings, command/config, stdout/stderr logs, epoch/validation metrics, RNG states/seeds, hardware capture, environment lock/container, and checkpoint manifest linking each file to a run.

## 4. Threshold and ROUGE verification

### Dynamic threshold

In `evaluate_model.py:77-88`, `cnn.py`, `cnn_ablation.py`, `bert.py`, and `case_study.py`:

1. valid length = minimum of source sentence count and the (incorrectly repurposed) token-mask sum;
2. `tau = np.mean(valid_scores) + 0.5 * np.std(valid_scores)`;
3. select indices satisfying strict `>`;
4. if fewer than **2** pass, select `min(3, valid_len)` highest scores;
5. sort selected indices into original order.

`np.std` uses `ddof=0`. There is no NaN/empty-document guard before mean/std, though upstream data generally supplies sentences.

Contradictory paths:

- `mimic.py` and `mimicIV.py`: `max(3, int(valid_len * 0.3))`, capped at valid length; no dynamic threshold.
- `ablation.py`: top 30%, minimum 3; its `max(3, ...)` is not capped and can request more indices than short documents, though slicing simply returns available entries.
- `arxiv.py`: fixed `min(6, num_sentences)`.

Therefore the paper's claim that the clinical results use the dynamic threshold is **NOT VERIFIED and contradicted by the available clinical scripts**.

### ROUGE

All relevant scripts instantiate:

```python
rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
```

Verified semantics:

- library: Google `rouge-score` Python package (import path `rouge_score`);
- historical package version: **NOT VERIFIED**; current environment has 0.1.2;
- metrics: ROUGE-1, ROUGE-2, ROUGE-L;
- statistic: `.fmeasure` (F1), not recall;
- stemming: enabled package stemmer (Porter), not NLTK stemming;
- aggregation: arithmetic mean of per-document F1 values via `numpy.mean` (macro average); no bootstrap confidence intervals;
- skipped PubMed examples: empty oracle reference; baseline additionally skips documents with fewer than 3 source sentences;
- references: PubMed thresholded dataset scores, CXR Impression, BHC target, ArXiv abstract.

No saved per-example scores or predictions exist, so aggregation counts, skipped-example counts for historical runs, confidence intervals, and exact paper-table regeneration are **NOT VERIFIED**.

## 5. Results evidence

`ablation_results.txt` is the only textual saved-results artifact:

| Model | R1 | R2 | RL |
|---|---:|---:|---:|
| CNN-Only | 0.6339 | 0.6191 | 0.6312 |
| LSTM-Only | 0.6766 | 0.6712 | 0.6766 |

Its SHA-256 is `723da0b649b6fff84417429947633a02e43c9ddada38ffb55dcf067cd2284f9e`. It lacks run metadata. The paper reports LSTM-only ROUGE-L 0.6741, a direct mismatch with the saved file's 0.6766.

The PNG figures are rendered artifacts without underlying result tables or provenance. `a.py` hard-codes four ROUGE-1 values and does not read evaluation output; its second and third values differ from the paper's depth table. Paper tables, console-only print statements, comments saying “matching paper,” and checkpoint presence are not proof of results. All non-ablation-file scores are **NOT VERIFIED from repository results**.

## 6. Files inspected

Tracked source/artifacts inspected:

- `a.py`
- `ablation.py`
- `ablation_results.txt`
- `arxiv.py`
- `bert.py`
- `case_study.py`
- `cnn.py`
- `cnn_ablation.py`
- `dataset.py`
- `embeddings.py`
- `evaluate_baselines.py`
- `evaluate_model.py`
- `lstm.py`
- `mimic.py`
- `mimic4baseline.py`
- `mimicCXRbaseline.py`
- `mimicIV.py`
- `train_model.py`
- all five `.pth` state dictionaries (keys/shapes and hashes, without executing model inference)
- all four PNG files (presence, size, and hash; no numeric provenance embedded)
- Git status and the sole commit metadata

Ignored/local evidence inspected:

- `local_dataset/{train,val,test}/dataset_info.json` and `state.json`
- validation/test Arrow datasets; train Arrow was absent
- all retained `embeddings/*.pt` tensor shapes and all label-pickle counts
- cached MIMIC-CXR and MIMIC-IV CSV schemas, counts, and the deterministic samples defined by code
- current `venv` package metadata and `pyvenv.cfg`

Not found anywhere in the inspected tree: README before this audit, requirements/lockfile, YAML/JSON experiment configs, TensorBoard/W&B records, stdout logs, raw predictions, per-example metrics, split manifests, dataset checksums, checkpoint metadata, license file, citation file, or CI tests.

## 7. Evidence required before camera-ready release

1. Freeze exact dataset revisions and publish permitted split manifests/checksums; use patient-grouped clinical splits.
2. Restore or regenerate sentence-level embeddings with one `[CLS]` vector per sentence, then archive dimensionality/label alignment tests.
3. Decide whether the proposed model has three or four sequential convolutions; make paper, code, checkpoint, figure, and ablations agree.
4. Use one explicitly documented extraction rule across the reported evaluations or clearly label protocol differences.
5. Add deterministic seeds and record all stochastic state.
6. Add validation, checkpoint-selection, and early-stopping documentation—or explicitly report their absence.
7. Pin Python, packages, pretrained model revisions, NLTK data, OS, accelerator, and driver/runtime versions.
8. Save commands, configs, logs, checkpoint hashes, evaluated IDs, generated summaries, references, per-example ROUGE, skipped counts, and aggregate scripts.
9. Reconcile every paper table with signed/hashed raw outputs. In particular, resolve the LSTM ROUGE-L mismatch and invalidate or repair the impossible 3-versus-4-layer depth comparison.

This audit intentionally does not repair code or rerun experiments: doing so would create new results rather than verify the historical claims.
