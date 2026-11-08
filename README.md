# A Hybrid Hierarchical 1D-CNN-BiLSTM Framework for Extractive Summarization of Biomedical and Clinical Text

This repository contains a biomedical extractive-summarization framework.

| Dataset | Domain | Description |
| :--- | :--- | :--- |
| **PubMed** | Scientific Literature | Research articles and abstracts |
| **MIMIC-CXR** | Radiology | Radiology reports |
| **MIMIC-IV** | Clinical | Brief Hospital Course notes |

The pipeline encodes document content with SciBERT, applies sequential 1D-CNN layers for local feature extraction, and uses a bidirectional LSTM to score content for summary selection. The repository includes separate training and testing pipelines, threshold analyses, baseline comparisons, and ROUGE-based evaluation utilities.

Paper: [arXiv:2609.13481](https://arxiv.org/abs/2609.13481)
