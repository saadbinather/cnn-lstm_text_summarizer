# Biomedical and Clinical Text Summarization

This repository contains a biomedical extractive-summarization framework evaluated on PubMed research articles, MIMIC-CXR radiology reports, and MIMIC-IV Brief Hospital Course notes. It supports both in-domain and cross-domain experiments across scientific and clinical text.

The pipeline encodes document content with SciBERT, applies sequential 1D-CNN layers for local feature extraction, and uses a bidirectional LSTM to score content for summary selection. The repository includes separate training and testing pipelines, threshold analyses, baseline comparisons, and ROUGE-based evaluation utilities.

Paper: [arXiv:2609.13481](https://arxiv.org/abs/2609.13481)
