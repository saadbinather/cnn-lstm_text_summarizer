# Experiment 01 status

**Status: BLOCKED — model ROUGE was not recalculated.**

The existing local artifacts are insufficient to generate PubMed model summaries. Both files expected by the evaluator are absent:

- `embeddings/test_emb_chunk_0.pt`
- `embeddings/test_emb_chunk_1.pt`

Only the corresponding masks and `test_labels.pkl` remain. A mask cannot reconstruct the 768-dimensional input states. Creating replacement embeddings would be a new preprocessing run and was excluded by the experiment constraints. No prediction or ROUGE value has been fabricated.

The genuine reference abstract is available locally as `local_dataset/test[*]["summary"]`. If the exact historical embedding chunks are restored, the existing checkpoint can be evaluated against that field without retraining, but a separate evaluation driver will be required because `src/evaluate_model.py` constructs references from `test_labels.pkl` instead.

Files that would normally be emitted but are intentionally absent:

- `predictions.csv`
- `per_document_rouge.csv`

Their absence is evidence of the blocked run, not missing reporting.

