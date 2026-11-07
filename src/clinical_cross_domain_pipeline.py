"""Local-only training and evaluation utilities for MIMIC text domains.

Nothing in this module downloads data, models, or tokenizer resources. The
training path creates patient-grouped manifests before fitting the repository's
three-layer CNN--BiLSTM sentence scorer. The evaluation path requires a saved
manifest and checkpoint and emits per-document predictions plus ROUGE F1.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


@dataclass(frozen=True)
class ClinicalRecord:
    row_id: int
    patient_id: str
    source: str
    reference: str


class HierarchicalSequentialSummarizer(nn.Module):
    def __init__(self, embed_dim: int = 768, cnn_filters: int = 128, lstm_hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=1, padding="same")
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=2, padding="same")
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding="same")
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            cnn_filters,
            lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = embeddings.transpose(1, 2)
        hidden = self.relu(self.conv1(hidden))
        hidden = self.relu(self.conv2(hidden))
        hidden = self.relu(self.conv3(hidden)).transpose(1, 2)
        hidden, _ = self.lstm(hidden)
        return torch.sigmoid(self.classifier(hidden).squeeze(-1))


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "domain", "dataset_csv", "patient_id_column", "source_column", "parser",
        "scibert_model", "output_dir", "seed", "train_fraction", "validation_fraction",
        "test_fraction", "max_sentences", "max_sentence_tokens", "oracle_sentences",
        "cnn_filters", "lstm_hidden", "learning_rate", "batch_size", "epochs",
        "selection_fraction", "minimum_selected_sentences",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
    split_sum = sum(float(config[key]) for key in ("train_fraction", "validation_fraction", "test_fraction"))
    if not np.isclose(split_sum, 1.0):
        raise ValueError("Train, validation, and test fractions must sum to one")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_cxr(value: object) -> tuple[str, str] | None:
    try:
        reports = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(reports, list) or not reports:
        return None
    report = str(reports[0])
    if "Impression:" not in report:
        return None
    findings, impression = report.split("Impression:", maxsplit=1)
    findings = findings.replace("Findings:", "", 1).strip()
    impression = impression.strip()
    return (findings, impression) if findings and impression else None


def load_records(config: dict) -> list[ClinicalRecord]:
    dataset_path = Path(config["dataset_csv"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Local dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    required_columns = {config["patient_id_column"], config["source_column"]}
    if config["parser"] == "direct_columns":
        required_columns.add(config["reference_column"])
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(
            "Dataset lacks required columns: " + ", ".join(missing) +
            ". A patient identifier is mandatory; row-level splitting is not allowed."
        )

    records = []
    for row_id, row in frame.iterrows():
        patient_id = str(row[config["patient_id_column"]]).strip()
        if not patient_id or patient_id.lower() == "nan":
            continue
        if config["parser"] == "mimic_cxr_findings_impression":
            parsed = parse_cxr(row[config["source_column"]])
            if parsed is None:
                continue
            source, reference = parsed
        elif config["parser"] == "direct_columns":
            source = str(row[config["source_column"]]).strip()
            reference = str(row[config["reference_column"]]).strip()
            if not source or not reference or source.lower() == "nan" or reference.lower() == "nan":
                continue
        else:
            raise ValueError(f"Unsupported parser: {config['parser']}")
        records.append(ClinicalRecord(int(row_id), patient_id, source, reference))
    if not records:
        raise ValueError("No usable clinical records were parsed")
    return records


def patient_group_split(records: list[ClinicalRecord], config: dict) -> dict[str, list[ClinicalRecord]]:
    patients = sorted({record.patient_id for record in records})
    rng = np.random.default_rng(int(config["seed"]))
    rng.shuffle(patients)
    train_end = int(len(patients) * float(config["train_fraction"]))
    validation_end = train_end + int(len(patients) * float(config["validation_fraction"]))
    patient_sets = {
        "train": set(patients[:train_end]),
        "validation": set(patients[train_end:validation_end]),
        "test": set(patients[validation_end:]),
    }
    if any(patient_sets[left] & patient_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Patient overlap detected")
    return {
        split: [record for record in records if record.patient_id in split_patients]
        for split, split_patients in patient_sets.items()
    }


def write_manifests(splits: dict[str, list[ClinicalRecord]], output_dir: Path) -> None:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        path = manifest_dir / f"{split}.csv"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite manifest: {path}")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["row_id", "patient_id"])
            writer.writeheader()
            writer.writerows({"row_id": row.row_id, "patient_id": row.patient_id} for row in records)


def load_manifest_records(config: dict, split: str) -> list[ClinicalRecord]:
    records = {record.row_id: record for record in load_records(config)}
    path = Path(config["output_dir"]) / "manifests" / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Saved patient-level manifest not found: {path}")
    selected = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_id = int(row["row_id"])
            if row_id not in records:
                raise ValueError(f"Manifest row {row_id} is absent from the parsed dataset")
            record = records[row_id]
            if record.patient_id != row["patient_id"]:
                raise ValueError(f"Patient mismatch for manifest row {row_id}")
            selected.append(record)
    return selected


def sentence_split(text: str) -> list[str]:
    try:
        return nltk.sent_tokenize(text)
    except LookupError as error:
        raise RuntimeError(
            "Required NLTK sentence tokenizer data is not installed locally; downloads are disabled"
        ) from error


def oracle_labels(sentences: list[str], reference: str, count: int) -> np.ndarray:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    similarities = np.asarray(
        [scorer.score(reference, sentence)["rougeL"].fmeasure for sentence in sentences],
        dtype=np.float32,
    )
    labels = np.zeros(len(sentences), dtype=np.float32)
    top_k = min(count, len(sentences))
    if top_k:
        labels[np.argsort(similarities)[-top_k:]] = 1.0
    return labels


def load_encoder(config: dict, device: torch.device):
    model_name = config["scibert_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return tokenizer, encoder


def encode_document(sentences: list[str], tokenizer, encoder, config: dict, device: torch.device) -> torch.Tensor:
    sentences = sentences[: int(config["max_sentences"])]
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=int(config["max_sentence_tokens"]),
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        embeddings = encoder(**inputs).last_hidden_state[:, 0, :]
    padded = torch.zeros(int(config["max_sentences"]), embeddings.shape[-1], device=device)
    padded[: len(sentences)] = embeddings
    return padded.cpu()


class EncodedClinicalDataset(Dataset):
    def __init__(self, rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def prepare_training_rows(records: Iterable[ClinicalRecord], tokenizer, encoder, config: dict, device: torch.device):
    rows = []
    maximum = int(config["max_sentences"])
    for record in records:
        sentences = sentence_split(record.source)[:maximum]
        if not sentences:
            continue
        embeddings = encode_document(sentences, tokenizer, encoder, config, device)
        labels = torch.zeros(maximum, dtype=torch.float32)
        labels[: len(sentences)] = torch.from_numpy(
            oracle_labels(sentences, record.reference, int(config["oracle_sentences"]))
        )
        mask = torch.zeros(maximum, dtype=torch.bool)
        mask[: len(sentences)] = True
        rows.append((embeddings, labels, mask))
    return rows


def masked_loss(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(predictions[mask], labels[mask])


def evaluate_loss(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for embeddings, labels, mask in loader:
            embeddings, labels, mask = embeddings.to(device), labels.to(device), mask.to(device)
            losses.append(float(masked_loss(model(embeddings), labels, mask).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_domain(config_path: Path) -> None:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_validation_loss.pth"
    metadata_path = output_dir / "training_metadata.json"
    if checkpoint_path.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing cross-domain training run")

    records = load_records(config)
    splits = patient_group_split(records, config)
    write_manifests(splits, output_dir)
    device = select_device()
    tokenizer, encoder = load_encoder(config, device)
    train_rows = prepare_training_rows(splits["train"], tokenizer, encoder, config, device)
    validation_rows = prepare_training_rows(splits["validation"], tokenizer, encoder, config, device)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        EncodedClinicalDataset(train_rows),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        EncodedClinicalDataset(validation_rows), batch_size=int(config["batch_size"])
    )
    model = HierarchicalSequentialSummarizer(
        cnn_filters=int(config["cnn_filters"]), lstm_hidden=int(config["lstm_hidden"])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history = []
    best_loss = float("inf")
    for epoch in range(int(config["epochs"])):
        model.train()
        train_losses = []
        for embeddings, labels, mask in train_loader:
            embeddings, labels, mask = embeddings.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = masked_loss(model(embeddings), labels, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation_loss = evaluate_loss(model, validation_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), checkpoint_path)

    metadata = {
        "domain": config["domain"],
        "config": config,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "patient_counts": {split: len({row.patient_id for row in rows}) for split, rows in splits.items()},
        "zero_patient_overlap_verified": True,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def test_domain(config_path: Path, checkpoint: Path | None = None, run_name: str | None = None) -> None:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    checkpoint = checkpoint or Path(config["output_dir"]) / "checkpoints" / "best_validation_loss.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    source_name = run_name or checkpoint.parent.parent.name
    output_dir = Path(config["output_dir"]) / "evaluations" / source_name
    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    if predictions_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation")

    records = load_manifest_records(config, "test")
    device = select_device()
    tokenizer, encoder = load_encoder(config, device)
    model = HierarchicalSequentialSummarizer(
        cnn_filters=int(config["cnn_filters"]), lstm_hidden=int(config["lstm_hidden"])
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    rows = []
    with torch.no_grad():
        for record in records:
            sentences = sentence_split(record.source)[: int(config["max_sentences"])]
            if not sentences:
                continue
            embeddings = encode_document(sentences, tokenizer, encoder, config, device).unsqueeze(0).to(device)
            scores = model(embeddings).squeeze(0)[: len(sentences)].cpu().numpy()
            top_k = min(
                len(sentences),
                max(int(config["minimum_selected_sentences"]), int(len(sentences) * float(config["selection_fraction"]))),
            )
            indices = sorted(np.argsort(scores)[-top_k:].tolist())
            prediction = " ".join(sentences[index] for index in indices)
            rouge = scorer.score(record.reference, prediction)
            rows.append(
                {
                    "row_id": record.row_id,
                    "patient_id": record.patient_id,
                    "selected_indices": json.dumps(indices),
                    "prediction": prediction,
                    "reference": record.reference,
                    "rouge1_f1": rouge["rouge1"].fmeasure,
                    "rouge2_f1": rouge["rouge2"].fmeasure,
                    "rougeL_f1": rouge["rougeL"].fmeasure,
                }
            )
    if not rows:
        raise ValueError("No test documents were evaluated")
    output_dir.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "training_checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation_domain": config["domain"],
        "evaluated_documents": len(rows),
        "patient_manifest": str(Path(config["output_dir"]) / "manifests" / "test.csv"),
        "rouge1_f1": float(np.mean([row["rouge1_f1"] for row in rows])),
        "rouge2_f1": float(np.mean([row["rouge2_f1"] for row in rows])),
        "rougeL_f1": float(np.mean([row["rougeL_f1"] for row in rows])),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
