"""Evaluate a selected checkpoint on the patient-held-out MIMIC-IV BHC split."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clinical_cross_domain_pipeline import MIMIC_IV_BHC_CONFIG, test_domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=Path(MIMIC_IV_BHC_CONFIG["dataset_csv"]))
    parser.add_argument("--output-dir", type=Path, default=Path(MIMIC_IV_BHC_CONFIG["output_dir"]))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = dict(MIMIC_IV_BHC_CONFIG)
    config.update(dataset_csv=str(args.dataset_csv), output_dir=str(args.output_dir))
    test_domain(config, args.checkpoint, args.run_name)
