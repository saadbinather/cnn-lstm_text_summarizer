"""Evaluate a selected checkpoint on the patient-held-out MIMIC-CXR split."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clinical_cross_domain_pipeline import test_domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    test_domain(Path("configs/mimic_cxr_full_pipeline.json"), args.checkpoint, args.run_name)
