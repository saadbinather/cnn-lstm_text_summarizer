"""Train the local-only MIMIC-IV BHC summarization pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clinical_cross_domain_pipeline import train_domain


if __name__ == "__main__":
    train_domain(Path("configs/mimic_iv_full_pipeline.json"))
