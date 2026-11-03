import os
import pickle
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. Hardware Setup (Optimized for Mac M1 Pro)
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon (MPS) for acceleration!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ==============================================================================
# 2. Load Datasets LOCALLY (No Internet Needed)
# ==============================================================================
print("📂 Loading datasets from local disk...")
train_ds = load_from_disk("local_dataset/train").select(range(10000))
val_ds = load_from_disk("local_dataset/val").select(range(3000))
test_ds = load_from_disk("local_dataset/test").select(range(3000))

# ==============================================================================
# 3. Load SciBERT Model
# ==============================================================================
MODEL_NAME = "allenai/scibert_scivocab_uncased"
print(f"🧠 Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

embedding_model.eval()
for param in embedding_model.parameters():
    param.requires_grad = False

# ==============================================================================
# 4. Dataset Class & Collate Function
# ==============================================================================
class PubMedEmbedDataset(Dataset):
    def __init__(self, hf_dataset, max_len=512):
        self.data = hf_dataset
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        sentences = example["text"]
        
        raw_labels = example.get("score", [0] * len(sentences))
        
        full_text = " ".join(sentences)
        enc = tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_len,
            padding="max_length", 
            return_tensors="pt"
        )
        
        num_sentences = len(sentences)
        labels_tensor = torch.tensor(raw_labels[:num_sentences], dtype=torch.float32)
        
        return (
            enc["input_ids"].squeeze(0), 
            enc["attention_mask"].squeeze(0),
            labels_tensor
        )

def medical_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attn_masks = torch.stack([item[1] for item in batch])
    labels = [item[2] for item in batch]
    return input_ids, attn_masks, labels

# ==============================================================================
# 5. Fast & Silent Generation Pipeline
# ==============================================================================
os.makedirs("embeddings", exist_ok=True)

import gc # Yeh RAM saaf karne ke liye hai

os.makedirs("embeddings", exist_ok=True)

def generate_and_save_embeddings_fast(hf_dataset, split_name, batch_size=16, max_len=512):
    ds = PubMedEmbedDataset(hf_dataset, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=medical_collate_fn)

    print(f"--- Starting {split_name.upper()} processing loop (Chunked Memory Safe) ---")
    
    all_labels = []
    chunk_embeddings = []
    chunk_masks = []
    chunk_index = 0

    for step, (input_ids, attn_mask, labels) in enumerate(loader):
        with torch.no_grad():
            out = embedding_model(
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device)
            )
        
        chunk_embeddings.append(out.last_hidden_state.cpu())
        chunk_masks.append(attn_mask.cpu())
        all_labels.extend([l.cpu() for l in labels])
        
        # 1. Free up MPS memory immediately
        del out, input_ids, attn_mask
        if device.type == 'mps':
            torch.mps.empty_cache()

        # 2. Save to disk every 100 batches and flush RAM
        if (step + 1) % 100 == 0:
            emb_tensor = torch.cat(chunk_embeddings, dim=0)
            mask_tensor = torch.cat(chunk_masks, dim=0)
            
            torch.save(emb_tensor, f"embeddings/{split_name}_emb_chunk_{chunk_index}.pt")
            torch.save(mask_tensor, f"embeddings/{split_name}_mask_chunk_{chunk_index}.pt")
            print(f"Saved {split_name} chunk {chunk_index}... RAM Cleared.")
            
            chunk_index += 1
            chunk_embeddings.clear()
            chunk_masks.clear()
            gc.collect() # Force garbage collection

    # Save any remaining data in the last chunk
    if chunk_embeddings:
        emb_tensor = torch.cat(chunk_embeddings, dim=0)
        mask_tensor = torch.cat(chunk_masks, dim=0)
        torch.save(emb_tensor, f"embeddings/{split_name}_emb_chunk_{chunk_index}.pt")
        torch.save(mask_tensor, f"embeddings/{split_name}_mask_chunk_{chunk_index}.pt")
        print(f"Saved final {split_name} chunk {chunk_index}...")

    # Save labels
    with open(f"embeddings/{split_name}_labels.pkl", "wb") as f:
        pickle.dump(all_labels, f)
        
    print(f"[{split_name.upper()} COMPLETED SUCCESSFULLY]\n")
    return None, None, all_labels
# ==============================================================================
# 6. Execution
# ==============================================================================
if __name__ == "__main__":
    train_emb, train_mask, train_labs = generate_and_save_embeddings_fast(train_ds, "train")
    val_emb, val_mask, val_labs       = generate_and_save_embeddings_fast(val_ds, "val")
    test_emb, test_mask, test_labs    = generate_and_save_embeddings_fast(test_ds, "test")
    print("🎉 All embeddings generated and saved successfully!")