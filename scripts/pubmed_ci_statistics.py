"""Compute paired bootstrap confidence intervals for PubMed threshold results.

The input must be the per-document CSV created by
``pubmed_threshold_multiseed_evaluation.py``. This program performs no model
inference or training. It verifies pairing, calculates percentile confidence
intervals for every threshold, and calculates paired improvement intervals
relative to the configured reference threshold.

This file is intentionally not executed during repository preparation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path("configs/pubmed_ci_statistics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing CI output files.",
    )
    return parser.parse_args()


def read_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "input",
        "output_dir",
        "confidence_level",
        "bootstrap_resamples",
        "seed",
        "reference_threshold_multiplier",
        "metrics",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
    if not 0.0 < float(config["confidence_level"]) < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if int(config["bootstrap_resamples"]) < 1:
        raise ValueError("bootstrap_resamples must be positive")
    return config


def load_scores(path: Path, metrics: list[str]) -> dict[float, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Per-document threshold results not found: {path}. "
            "Run the threshold evaluator separately before this analysis."
        )

    grouped: dict[float, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"document_index", "threshold_multiplier", "reference", *metrics}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            multiplier = float(row["threshold_multiplier"])
            grouped.setdefault(multiplier, []).append(row)

    if len(grouped) < 2:
        raise ValueError("At least two threshold strategies are required")

    prepared = {}
    expected_ids = None
    expected_references = None
    for multiplier, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["document_index"]))
        ids = tuple(int(row["document_index"]) for row in rows)
        references = tuple(row["reference"] for row in rows)
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate document IDs for threshold {multiplier}")
        if expected_ids is None:
            expected_ids = ids
            expected_references = references
        elif ids != expected_ids:
            raise ValueError(f"Document IDs are not paired for threshold {multiplier}")
        elif references != expected_references:
            raise ValueError(f"References differ for threshold {multiplier}")
        prepared[multiplier] = {
            "document_ids": ids,
            "references": references,
            "scores": np.asarray(
                [[float(row[metric]) for metric in metrics] for row in rows],
                dtype=np.float64,
            ),
        }
    return prepared


def percentile_bounds(samples: np.ndarray, confidence_level: float) -> tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    return lower, upper


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_interval(mean: float, lower: float, upper: float) -> str:
    return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"


def latex_table(method_rows: list[dict], paired_rows: list[dict], resamples: int, seed: int) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{PubMed ROUGE F1 means and 95\% paired-bootstrap confidence intervals ({resamples:,} resamples; seed {seed}).}}",
        r"\label{tab:pubmed_threshold_ci}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Threshold & ROUGE-1 & ROUGE-2 & ROUGE-L \\",
        r"\hline",
    ]
    by_threshold: dict[float, dict[str, dict]] = {}
    for row in method_rows:
        by_threshold.setdefault(float(row["threshold_multiplier"]), {})[row["metric"]] = row
    for multiplier, metric_map in sorted(by_threshold.items()):
        cells = [
            format_interval(
                float(metric_map[metric]["mean"]),
                float(metric_map[metric]["ci_lower"]),
                float(metric_map[metric]["ci_upper"]),
            )
            for metric in ("rouge1_f1", "rouge2_f1", "rougeL_f1")
        ]
        lines.append(rf"$k={multiplier:g}$ & " + " & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\vspace{2mm}"])
    lines.extend(
        [
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Paired difference & $\Delta$ROUGE-1 & $\Delta$ROUGE-2 & $\Delta$ROUGE-L \\",
            r"\hline",
        ]
    )
    by_comparator: dict[float, dict[str, dict]] = {}
    for row in paired_rows:
        by_comparator.setdefault(float(row["comparator_threshold_multiplier"]), {})[
            row["metric"]
        ] = row
    for comparator, metric_map in sorted(by_comparator.items()):
        cells = [
            format_interval(
                float(metric_map[metric]["mean_difference"]),
                float(metric_map[metric]["ci_lower"]),
                float(metric_map[metric]["ci_upper"]),
            )
            for metric in ("rouge1_f1", "rouge2_f1", "rougeL_f1")
        ]
        lines.append(rf"$k=0.5$ minus $k={comparator:g}$ & " + " & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    metrics = list(config["metrics"])
    scores = load_scores(Path(config["input"]), metrics)
    reference = float(config["reference_threshold_multiplier"])
    if reference not in scores:
        raise ValueError(f"Reference threshold {reference} is absent from the input")

    document_count = len(scores[reference]["document_ids"])
    resamples = int(config["bootstrap_resamples"])
    confidence_level = float(config["confidence_level"])
    rng = np.random.default_rng(int(config["seed"]))
    bootstrap_indices = rng.integers(0, document_count, size=(resamples, document_count))

    method_rows = []
    for multiplier, data in sorted(scores.items()):
        values = data["scores"]
        bootstrap_means = values[bootstrap_indices].mean(axis=1)
        lower, upper = percentile_bounds(bootstrap_means, confidence_level)
        means = values.mean(axis=0)
        for index, metric in enumerate(metrics):
            method_rows.append(
                {
                    "threshold_multiplier": multiplier,
                    "metric": metric,
                    "mean": means[index],
                    "ci_lower": lower[index],
                    "ci_upper": upper[index],
                    "confidence_level": confidence_level,
                    "sample_size": document_count,
                }
            )

    paired_rows = []
    reference_scores = scores[reference]["scores"]
    for comparator, data in sorted(scores.items()):
        if comparator == reference:
            continue
        differences = reference_scores - data["scores"]
        bootstrap_differences = differences[bootstrap_indices].mean(axis=1)
        lower, upper = percentile_bounds(bootstrap_differences, confidence_level)
        means = differences.mean(axis=0)
        for index, metric in enumerate(metrics):
            paired_rows.append(
                {
                    "reference_threshold_multiplier": reference,
                    "comparator_threshold_multiplier": comparator,
                    "metric": metric,
                    "mean_difference": means[index],
                    "ci_lower": lower[index],
                    "ci_upper": upper[index],
                    "confidence_level": confidence_level,
                    "sample_size": document_count,
                }
            )

    output_dir = Path(config["output_dir"])
    output_paths = {
        "method_intervals": output_dir / "method_confidence_intervals.csv",
        "paired_intervals": output_dir / "paired_improvement_intervals.csv",
        "summary": output_dir / "ci_summary.json",
        "latex": output_dir / "ieee_pubmed_ci_table.tex",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing outputs:\n  " + "\n  ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_paths["method_intervals"], method_rows)
    write_csv(output_paths["paired_intervals"], paired_rows)
    summary = {
        "status": "computed from recorded per-document scores; no inference or training",
        "input": config["input"],
        "sample_size": document_count,
        "paired_document_ids_verified": True,
        "identical_references_verified": True,
        "confidence_level": confidence_level,
        "bootstrap_resamples": resamples,
        "seed": int(config["seed"]),
        "reference_threshold_multiplier": reference,
        "metrics": metrics,
        "method_confidence_intervals": method_rows,
        "paired_improvement_intervals": paired_rows,
    }
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output_paths["latex"].write_text(
        latex_table(method_rows, paired_rows, resamples, int(config["seed"])),
        encoding="utf-8",
    )

    for path in output_paths.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
