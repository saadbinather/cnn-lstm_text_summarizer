# Hybrid Hierarchical CNN–BiLSTM Biomedical Summarizer

This repository is the research artifact associated with **“A Hybrid Hierarchical 1D-CNN-BiLSTM Framework for Extractive Summarization of Biomedical and Clinical Text”** ([arXiv:2609.13481](https://arxiv.org/abs/2609.13481)). It studies extractive summarization for PubMed articles, MIMIC-CXR reports, and MIMIC-IV Brief Hospital Course notes.

## Scope

The project explores a convolutional–recurrent approach that scores content from biomedical and clinical documents and selects source material for an extractive summary. Because the summaries are extractive, their wording remains grounded in the source. Extraction can nevertheless omit important findings or preserve wording without enough context, so source copying should not be interpreted as clinical correctness.

## Reproducibility status

This checkout is an incomplete research snapshot rather than a fully reproducible release. The repository audit identified differences between the paper description and some saved artifacts, along with incomplete historical provenance. Paper tables should not be treated as independently reproduced from this checkout.

Verified findings and unresolved issues are documented in the camera-ready audit and experiment records. Claims that lack matching datasets, predictions, logs, or saved outputs remain marked **NOT VERIFIED**.

## Datasets

The study covers the following domains:

- PubMed biomedical articles;
- MIMIC-CXR radiology reports; and
- MIMIC-IV Brief Hospital Course notes.

Access to MIMIC data is governed by its own credentials, licenses, and data-use agreements. Clinical data must not be redistributed through this repository.

The audit found that the available clinical material represents evaluation samples rather than independently established training, validation, and test partitions. PubMed partition and overlap findings are recorded in the audit documentation.

## Evaluation

Evaluation includes ROUGE-based summary comparison, threshold-selection ablations, paired statistical analysis, and automated clinical-content proxies. The experiments distinguish human-written reference abstracts from extractive-oracle references because those protocols answer different questions and their scores are not directly interchangeable.

The verified genuine-abstract PubMed evaluation produced substantially lower scores than the original oracle-based evaluation. The threshold study found that fixed Top-5 selection performed best among the four tested strategies on the recorded genuine-abstract sample. Clinical analysis found exact source support for selected text but limited coverage of important findings. These automated clinical proxies are not clinician validation.

## Important limitations

- Historical training provenance is incomplete.
- Some saved representations do not match the sentence-level description in the paper.
- Evaluation subsets and reference protocols differ across legacy scripts.
- Clinical evaluation does not establish patient-safe development and test partitions.
- Exact historical hardware and software environments were not retained.
- ROUGE and automated lexical proxies do not establish clinical factuality or completeness.

## Repository contents

The repository contains source programs, evaluation utilities, saved research artifacts, experiment outputs, and camera-ready audit documentation. Existing checkpoints and recorded results are preserved for historical reference. Consult the audit before interpreting or citing any result.

## Responsible use

This project is a research artifact and is not a clinical decision-support system. Outputs require independent review, particularly when they concern diagnoses, negation, treatment, or other safety-critical findings.
