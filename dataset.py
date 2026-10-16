import os
from datasets import load_dataset

# Ek folder bana lo jahan dataset save hoga
os.makedirs("local_dataset", exist_ok=True)

print("📥 Downloading datasets from Hugging Face...")
train_ds = load_dataset("nianlong/long-doc-extractive-summarization-truncated-pubmed", data_files={"train": "train.jsonl"}, split="train")
val_ds = load_dataset("nianlong/long-doc-extractive-summarization-truncated-pubmed", data_files={"val": "val.jsonl"}, split="val")
test_ds = load_dataset("nianlong/long-doc-extractive-summarization-truncated-pubmed", data_files={"test": "test.jsonl"}, split="test")

print("💾 Saving datasets locally to disk...")
train_ds.save_to_disk("local_dataset/train")
val_ds.save_to_disk("local_dataset/val")
test_ds.save_to_disk("local_dataset/test")

print("🎉 Dataset successfully saved locally in 'local_dataset' folder!")