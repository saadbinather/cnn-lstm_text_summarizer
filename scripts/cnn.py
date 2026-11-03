import os, pickle, gc, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_from_disk
from rouge_score import rouge_scorer
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Running Stacked CNN-Only Ablation on: {device}")

# ==============================================================================
# 1. Stacked CNN-Only Ablation Architecture (True Hierarchical, No LSTM)
# ==============================================================================
class StackedCNNOnlySummarizer(nn.Module):
    def __init__(self, embed_dim=768, cnn_filters=128):
        super(StackedCNNOnlySummarizer, self).__init__()
        # True Stacked / Hierarchical CNN layers sequentially refining feature maps
        self.conv1 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=2, padding='same')
        self.conv3 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=4, padding='same')
        self.relu = nn.ReLU()
        
        # Direct classifier from the final stacked CNN layer output
        self.classifier = nn.Linear(cnn_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x)) # Shape: [Batch, cnn_filters, Seq_Len]
        
        cnn_out = x.transpose(1, 2)  # Shape: [Batch, Seq_Len, cnn_filters]
        logits = self.classifier(cnn_out).squeeze(-1)
        return self.sigmoid(logits)

model = StackedCNNOnlySummarizer().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==============================================================================
# 2. Training Loop (3 Epochs)
# ==============================================================================
with open("embeddings/train_labels.pkl", "rb") as f:
    train_labels = pickle.load(f)

chunk_files = [f for f in os.listdir("embeddings") if f.startswith("train_emb_chunk_")]
total_chunks = len(chunk_files)

print("🔥 Training Stacked CNN-Only Model...")
for epoch in range(3):
    label_start_idx = 0
    for chunk_idx in range(total_chunks):
        emb = torch.load(f"embeddings/train_emb_chunk_{chunk_idx}.pt", map_location="cpu", weights_only=True)
        mask = torch.load(f"embeddings/train_mask_chunk_{chunk_idx}.pt", map_location="cpu", weights_only=True)
        
        c_size = emb.size(0)
        c_labels = train_labels[label_start_idx : label_start_idx + c_size]
        label_start_idx += c_size
        
        lbl_tensor = torch.stack([torch.cat([torch.tensor(l[:512]), torch.zeros(max(0, 512-len(l)))])[:512] for l in c_labels])
        
        loader = DataLoader(TensorDataset(emb, mask, lbl_tensor), batch_size=16, shuffle=True)
        for b_emb, b_mask, b_lbl in loader:
            b_emb, b_mask, b_lbl = b_emb.to(device), b_mask.to(device), b_lbl.to(device)
            optimizer.zero_grad()
            preds = model(b_emb)
            active = b_mask.view(-1) == 1
            loss = criterion(preds.view(-1)[active], b_lbl.view(-1)[active])
            loss.backward()
            optimizer.step()
        
        del emb, mask, lbl_tensor, loader
        if device.type == 'mps': torch.mps.empty_cache()
        gc.collect()
    print(f"✅ Epoch {epoch+1} done.")

torch.save(model.state_dict(), "stacked_cnn_only.pth")
print("🎉 Saved as stacked_cnn_only.pth")

# ==============================================================================
# 3. Evaluation & ROUGE Scoring
# ==============================================================================
print("\n🔍 Evaluating Stacked CNN-Only Model on Test Set...")
model.eval()

test_ds = load_from_disk("local_dataset/test").select(range(300))
test_emb = torch.load("embeddings/test_emb_chunk_0.pt", map_location="cpu")
test_mask = torch.load("embeddings/test_mask_chunk_0.pt", map_location="cpu")
with open("embeddings/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
r1, r2, rl = [], [], []

with torch.no_grad():
    for idx in range(len(test_ds)):
        original_sentences = test_ds[idx]["text"]
        labels = test_labels[idx]
        
        gt = " ".join([original_sentences[i] for i, l in enumerate(labels) if l >= 0.5])
        if not gt.strip(): continue
            
        doc_emb = test_emb[idx].unsqueeze(0).to(device)
        scores = model(doc_emb).squeeze(0).cpu().numpy()
        
        num_sents = min(len(original_sentences), int(test_mask[idx].sum().item()))
        valid = scores[:num_sents]
        
        thresh = np.mean(valid) + (np.std(valid) * 0.5)
        top_idx = np.where(valid > thresh)[0]
        if len(top_idx) < 2:
            top_idx = valid.argsort()[-min(3, num_sents):][::-1]
        top_idx = sorted(top_idx)
        
        gen = " ".join([original_sentences[i] for i in top_idx])
        rs = scorer.score(gt, gen)
        r1.append(rs['rouge1'].fmeasure)
        r2.append(rs['rouge2'].fmeasure)
        rl.append(rs['rougeL'].fmeasure)

print("\n" + "="*50)
print("🏆 STACKED CNN-ONLY RESULTS 🏆")
print(f"ROUGE-1: {np.mean(r1):.4f}")
print(f"ROUGE-2: {np.mean(r2):.4f}")
print(f"ROUGE-L: {np.mean(rl):.4f}")
print("="*50)