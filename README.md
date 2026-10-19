# Hybrid Hierarchical CNN–BiLSTM Biomedical Summarizer

This repository is the research artifact associated with **“A Hybrid Hierarchical 1D-CNN-BiLSTM Framework for Extractive Summarization of Biomedical and Clinical Text”** ([arXiv:2609.13481](https://arxiv.org/abs/2609.13481)). It explores extractive summarization for PubMed articles, MIMIC-CXR reports, and MIMIC-IV Brief Hospital Course notes.

> **Reproducibility status:** this checkout is an incomplete research snapshot, not a fully reproducible release. The code and saved checkpoints do not consistently implement the four-layer, sentence-level system described in the paper. Do not use the paper's tables as reproduced results from this checkout. See [CAMERA_READY_AUDIT.md](CAMERA_READY_AUDIT.md) before citing or running the code.

## Method overview

The intended pipeline scores source sentences for extraction:

1. split a document into sentences;
2. encode each sentence with the 768-dimensional SciBERT `[CLS]` representation;
3. refine adjacent-sentence features with sequential 1D convolutions;
4. model document context with a one-layer bidirectional LSTM;
5. assign a sigmoid salience score to every valid sentence; and
6. select sentences above a document-specific threshold, falling back to the three highest-scoring sentences when fewer than two pass.

The dynamic rule implemented in the PubMed evaluator is

\[
\tau = \operatorname{mean}(p) + 0.5\operatorname{std}(p),
\]

with strict selection `p > tau`. Selected indices are sorted back into source order. The implementation uses NumPy's population standard deviation (`ddof=0`).

### Architecture actually represented by the main checkpoint

`hierarchical_cnn_lstm_summarizer.pth` contains three sequential convolutional layers, not four:

| Stage | Verified shape/configuration |
|---|---|
| Input | `(batch, 512, 768)` |
| Conv 1 | 768 → 128 channels, kernel 1, same padding, ReLU |
| Conv 2 | 128 → 128 channels, kernel 2, same padding, ReLU |
| Conv 3 | 128 → 128 channels, kernel 3, same padding, ReLU |
| BiLSTM | 128 hidden units per direction, one layer |
| Head | 256 → 1, sigmoid |

`cnn_lstm_summarizer.pth` is a separate parallel four-kernel model: every convolution consumes the original 768-channel input and their outputs are concatenated. It is not the sequential four-layer architecture described in the paper.

## Datasets and observed sampling

Access to MIMIC data is governed by its own credentials, data-use agreements, and licenses. No clinical data should be redistributed with this repository.

| Dataset | Source named in code | Verified behavior in this checkout |
|---|---|---|
| PubMed | `nianlong/long-doc-extractive-summarization-truncated-pubmed` | Upstream split metadata: 83,233 train, 4,676 validation, 5,025 test. `embeddings.py` takes the first 10,000/3,000/3,000 records without shuffling. Main baseline scripts use the first 300 test records; `evaluate_model.py` instead uses 500. |
| MIMIC-CXR | Kaggle `simhadrisadaram/mimic-cxr-dataset`, `mimic_cxr_aug_train.csv` | One 64,586-row file is sampled to 500 rows with pandas `random_state=42`. Only the first report in each serialized list is parsed; 353 of the saved seeded sample have usable Findings/Impression pairs. There is no train/validation/test split. |
| MIMIC-IV BHC | Kaggle `aminexdr/bhc-mimic-iv-summary` | One 269,516-row file is sampled to 500 rows with pandas `random_state=42` after dropping null input/target rows (none are null in the local file). There is no train/validation/test split. |

The PubMed validation and test slices retained locally contain 3,000 unique documents each and have zero exact-document and zero `text_id` overlap. Training overlap is **NOT VERIFIED** because the train Arrow file is absent. Clinical cross-split patient overlap is inapplicable: the scripts define only one evaluation sample per dataset.

## Preprocessing and supervision

PubMed records already contain sentence lists and dataset-provided `score` arrays. `embeddings.py` saves those scores as labels; this repository contains no oracle-label construction algorithm or provenance documentation.

There is an important implementation defect in the saved PubMed pipeline. `embeddings.py` joins each document's sentences into one string, tokenizes the entire string to 512 **tokens**, and stores all token hidden states with a token attention mask. Training and evaluation then interpret those positions as 512 **sentences** and align sentence labels to them. Consequently, the saved PubMed tensors are not sentence-level SciBERT `[CLS]` embeddings.

The on-the-fly clinical evaluators do use NLTK `sent_tokenize`, embed each resulting sentence separately with SciBERT `[CLS]`, truncate individual sentences to 128 subword tokens, and pad/truncate documents to 512 sentences. They do not train on either clinical dataset.

## Training code

The main training script specifies:

| Setting | Verified value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Loss | masked binary cross-entropy |
| Batch size | 16 |
| Epochs | 10 |
| Device preference | MPS, then CUDA, then CPU |
| Checkpoint output | final epoch only, `hierarchical_cnn_lstm_summarizer.pth` |

There is no validation loop, checkpoint selection metric, early stopping, scheduler, gradient clipping, or random-seed initialization. Chunk ordering depends on unsorted `os.listdir()` output, while batches are shuffled without a seeded generator. Exact training hardware, software versions, loss history, and checkpoint lineage are **NOT VERIFIED**.

The CNN-only and LSTM-only training scripts each use Adam at `1e-3`, batch size 16, and 3 epochs. Their saved checkpoint identities and limitations are recorded in the audit.

## Evaluation

All evaluator scripts use `rouge-score`'s `RougeScorer` for ROUGE-1, ROUGE-2, and ROUGE-L with `use_stemmer=True`, read `.fmeasure`, and macro-average per-document F1 scores with `numpy.mean`. This is the Porter stemmer supplied by `rouge-score`, not “NLTK stemming.”

Reference construction differs by dataset:

- PubMed: concatenate source sentences whose saved dataset score is at least 0.5. The dataset's `summary` field is not used.
- MIMIC-CXR: use the parsed text after `Impression:` from the first report in each row.
- MIMIC-IV BHC: use the CSV `target` text.

The claimed dynamic threshold is implemented only by the PubMed main/ablation evaluators and the vanilla-BERT variant. MIMIC-CXR and MIMIC-IV model evaluators instead select the highest-scoring 30% of sentences, with a minimum of three. The ArXiv transfer script selects exactly six. These are different evaluation protocols.

## Saved results

Only one machine-readable text result is present:

| PubMed ablation | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| CNN-only | 0.6339 | 0.6191 | 0.6312 |
| LSTM-only | 0.6766 | 0.6712 | 0.6766 |

Source: `ablation_results.txt` (SHA-256 `723da0b649b6fff84417429947633a02e43c9ddada38ffb55dcf067cd2284f9e`). The file does not record dataset fingerprint, sample count, environment, seed, command, timestamp, or checkpoint hash. Its LSTM ROUGE-L value differs from the paper. All other plotted or paper-reported numbers lack raw saved outputs in this checkout and are **NOT VERIFIED**.

## Installation

No `requirements.txt`, lockfile, container, package metadata, or environment export is included, so an exact historical environment cannot be installed from the repository. Imports show that the project requires Python plus PyTorch, Transformers, Datasets, `rouge-score`, NLTK, NumPy, pandas, scikit-learn, NetworkX, Sumy, KaggleHub, tqdm, and Matplotlib.

Create an isolated environment and install versions appropriate for your hardware only after deciding how to repair the data representation and architecture mismatches. Installing current packages does not reproduce the archived runs. NLTK Punkt resources and pretrained models are downloaded at runtime by several scripts.

## Repository map

| Path | Purpose/status |
|---|---|
| `dataset.py` | downloads and preserves the provider's PubMed splits |
| `embeddings.py` | creates saved PubMed tensors; contains the token/sentence mismatch described above |
| `train_model.py` | trains the three-convolution main checkpoint |
| `evaluate_model.py` | evaluates 500 PubMed records with the dynamic threshold |
| `evaluate_baselines.py` | evaluates Lead-3 and Sumy TextRank on 300 PubMed records |
| `cnn.py`, `lstm.py`, `ablation.py` | component-ablation training/evaluation paths |
| `cnn_ablation.py` | depth evaluation; defines only three convolutions, so requested depths 3 and 4 execute the same graph |
| `mimic.py`, `mimicIV.py` | clinical model evaluation with the 30% rule |
| `mimicCXRbaseline.py`, `mimic4baseline.py` | clinical Lead-3/TextRank evaluation |
| `bert.py`, `arxiv.py` | auxiliary backbone/domain transfer evaluations |
| `CAMERA_READY_AUDIT.md` | evidence ledger, discrepancies, and missing reproducibility evidence |

## Reproducibility checklist

Before treating this as a camera-ready artifact, archive the exact immutable dataset revisions; define patient-safe clinical splits; restore the training data and all embedding chunks; fix sentence-level encoding; settle the three-versus-four-layer architecture; seed Python, NumPy, PyTorch, DataLoaders, and samplers; add validation/checkpoint selection; pin the software environment; and save per-example predictions, references, metrics, commands, logs, and hardware metadata. None of those actions were performed as part of this audit, and no experiments were rerun.

