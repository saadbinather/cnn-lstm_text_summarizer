# Repository layout

The repository was organized on 2026-11-04 without editing model or experiment source code.

- `src/`: primary PubMed dataset preparation, embedding, training, and evaluation programs.
- `scripts/`: baselines, ablations, clinical evaluations, transfer tests, and figure helper.
- `results/legacy_metrics/`: pre-existing saved metric text.
- `results/legacy_figures/`: pre-existing figures.
- `results/experiment_01_rouge_verification/`: Experiment 01 status, commands, summary, and findings.
- `results/checkpoints/`: relative symlink index to unchanged root checkpoint files.
- `docs/`: audit and layout documentation.
- `local_dataset/`, `embeddings/`, `venv/`: ignored local artifacts left in place.
- repository root: unchanged checkpoint binaries and large ignored clinical CSV, preserving all hard-coded runtime paths.

Run scripts from the repository root, for example `python src/train_model.py` or `python scripts/evaluate_baselines.py`. This preserves their existing relative references to `local_dataset/`, `embeddings/`, and root checkpoint names. No Python source contents or hyperparameters were changed during organization.

