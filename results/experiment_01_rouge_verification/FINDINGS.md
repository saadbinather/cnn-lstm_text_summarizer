# Experiment 01 — ROUGE verification and leakage audit

Audit date: 2026-11-04 (Asia/Karachi)

## Outcome

ROUGE-1/2/L F1 for actual model-generated PubMed summaries against genuine reference abstracts is **NOT VERIFIED**. The exact test embedding chunks required by the saved checkpoint are missing, so inference cannot be performed under the constraint forbidding regenerated embeddings.

## Verified reference behavior

The genuine dataset reference is the `summary` list in each PubMed test record. The existing evaluators do not use it. `src/evaluate_model.py` and the PubMed ablation scripts instead concatenate source sentences for saved values `>= 0.5`.

`src/embeddings.py` populates those saved values from the dataset field named `score`, not `label`. In the first 3,000 test rows:

- saved arrays approximately equal `score` in 2,809 rows (the remainder indicates artifact/dataset-version drift or another unresolved mismatch);
- saved arrays exactly equal the dataset `label` oracle in 0 rows;
- only 96 saved arrays contain any value at least 0.5;
- 2,904 dataset `label` arrays contain a positive label.

The fields are also structurally misaligned. Only 212/3,000 rows have equal lengths for `text`, `score`, and the saved array. Only 396/3,000 have equal lengths for `text`, `label`, and the saved array. The evaluator enumerates the saved array and indexes `original_sentences[i]`, silently treating candidate scores as sentence labels.

## Evaluation selection leakage

For the first 300 documents used by several paper scripts, only 3 saved arrays have a value `>= 0.5`. Consequently, the current evaluator's empty-reference guard skips 297/300 documents and reports the mean over only 3. For the first 500 documents, it skips 489/500 and averages only 11.

This is the clearest verified explanation for the unusually high historical scores:

1. references are constructed from source sentences using the same misinterpreted saved values that serve as supervision;
2. genuine abstracts are never scored;
3. almost the entire evaluation sample is silently discarded;
4. no evaluated-document count is printed alongside the mean;
5. token-level SciBERT states and token masks are treated as sentence positions, creating an additional label/input misalignment.

The exact numerical contribution of each defect to the paper's reported scores is **NOT VERIFIED**, because historical predictions and per-document scores are absent and current inference is blocked.

## Overlap audit

For the locally retained first 3,000 validation and first 3,000 test documents:

- exact full-document text hashes are unique within each slice;
- validation/test exact-text overlap is 0;
- the earlier audit also found zero overlap in the available `text_id` tuples.

Train/test overlap is **NOT VERIFIED** because `local_dataset/train/data-00000-of-00001.arrow` is absent. PubMed patient overlap is **NOT VERIFIED** because these records expose no patient identifier.

The absence of exact validation/test overlap does not cure the evaluation leakage described above: that leakage occurs through reference construction and selective skipping, not necessarily duplicated documents.

## ROUGE configuration verified from code

- implementation: `rouge_score.rouge_scorer.RougeScorer`;
- metrics: `rouge1`, `rouge2`, `rougeL`;
- stemming: `use_stemmer=True`;
- statistic: `.fmeasure`;
- aggregation: arithmetic mean of per-document F1 using `numpy.mean`;
- reference/hypothesis argument order: reference first, generated summary second.

The historical package version remains **NOT VERIFIED**. The currently installed local environment reports `rouge-score==0.1.2`, but it is not tied to checkpoint provenance.

## Evidence needed to finish Experiment 01

Restore, without recomputing or modifying, the exact historical:

1. `embeddings/test_emb_chunk_0.pt` and `embeddings/test_emb_chunk_1.pt` files;
2. dataset revision/checksum associated with `test_labels.pkl` (2,809 rather than 3,000 arrays match the current `score` values);
3. if available, historical evaluated IDs, predictions, references, and logs.

With the embedding chunks restored, inference should preserve the checkpoint and threshold settings but replace only the evaluation reference with `" ".join(record["summary"])`. Because changing the evaluator was prohibited in this request, no new driver was added.

