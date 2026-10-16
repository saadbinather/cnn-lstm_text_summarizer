import torch
import numpy as np
from datasets import load_from_disk
from rouge_score import rouge_scorer
import torch.nn as nn
import pickle

# ==============================================================================
# 1. Hardware Setup & Hierarchical Model Definition
# ==============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Running Evaluation on: {device}")

# Same True Hierarchical architecture used during training
class HierarchicalSequentialSummarizer(nn.Module):
    def __init__(self, embed_dim=768, cnn_filters=128, lstm_hidden=128):
        super(HierarchicalSequentialSummarizer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2)  
        
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out).squeeze(-1)
        return self.sigmoid(logits)

# ==============================================================================
# 2. Load Saved Hierarchical Model & Test Data
# ==============================================================================
model = HierarchicalSequentialSummarizer().to(device)
model.load_state_dict(torch.load("hierarchical_cnn_lstm_summarizer.pth", map_location=device))
model.eval()
print("✅ Hierarchical Model loaded successfully!")

print("📂 Loading test datasets and embeddings...")
test_ds = load_from_disk("local_dataset/test").select(range(500)) # Evaluation on 500 samples
test_emb = torch.load("embeddings/test_emb_chunk_0.pt", map_location="cpu")
test_mask = torch.load("embeddings/test_mask_chunk_0.pt", map_location="cpu")
with open("embeddings/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)

# ==============================================================================
# 3. ROUGE Evaluation Loop
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

print("\n🔍 Generating summaries and calculating ROUGE scores...\n")

with torch.no_grad():
    for idx in range(len(test_ds)):
        original_sentences = test_ds[idx]["text"]
        labels = test_labels[idx]
        
        ground_truth_summary = " ".join([original_sentences[i] for i, label in enumerate(labels) if label >= 0.5])
        
        if not ground_truth_summary.strip():
            continue
            
        doc_emb = test_emb[idx].unsqueeze(0).to(device)
        doc_mask = test_mask[idx]
        
        scores = model(doc_emb).squeeze(0).cpu().numpy()
        
        num_sentences = min(len(original_sentences), int(doc_mask.sum().item()))
        valid_scores = scores[:num_sentences]
        
        threshold = np.mean(valid_scores) + (np.std(valid_scores) * 0.5) 
        top_k_indices = np.where(valid_scores > threshold)[0]
        
        if len(top_k_indices) < 2:
            top_k = min(3, num_sentences)
            top_k_indices = valid_scores.argsort()[-top_k:][::-1]
            
        top_k_indices = sorted(top_k_indices)
        generated_summary = " ".join([original_sentences[i] for i in top_k_indices])
        
        rouge_scores = scorer.score(ground_truth_summary, generated_summary)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(test_ds)} documents...")

# ==============================================================================
# 4. Final Results
# ==============================================================================
print("-" * 50)
print("🏆 FINAL HIERARCHICAL MODEL ROUGE SCORES 🏆")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print("-" * 50)