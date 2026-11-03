import torch
import numpy as np
from datasets import load_from_disk
from rouge_score import rouge_scorer
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Running Stacked CNN Kernel Ablation Study on: {device}")

# ==============================================================================
# Model Definition with Stacked Architecture for Kernel Ablation
# ==============================================================================
class HierarchicalSequentialSummarizer(nn.Module):
    def __init__(self, embed_dim=768, cnn_filters=128, lstm_hidden=128):
        super(HierarchicalSequentialSummarizer, self).__init__()
        # True Stacked / Hierarchical CNN layers
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding='same')
        
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, num_kernels=4):
        x = x.transpose(1, 2)  
        
        # Sequentially apply stacked layers based on ablation config (num_kernels)
        x = self.relu(self.conv1(x))
        if num_kernels >= 2:
            x = self.relu(self.conv2(x))
        if num_kernels >= 3:
            x = self.relu(self.conv3(x))
                    
        cnn_out = x.transpose(1, 2)  
        lstm_out, _ = self.lstm(cnn_out)
        logits = self.classifier(lstm_out).squeeze(-1)
        return self.sigmoid(logits)

# ==============================================================================
# Load Model & Test Data
# ==============================================================================
model = HierarchicalSequentialSummarizer().to(device)
try:
    model.load_state_dict(torch.load("hierarchical_cnn_lstm_summarizer.pth", map_location=device))
    print("✅ Stacked Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Warning: 'hierarchical_cnn_lstm_summarizer.pth' nahi mili. Pehle model train karo!")
model.eval()

print("📂 Loading test datasets and embeddings...")
test_ds = load_from_disk("local_dataset/test").select(range(300)) # 300 samples for fast evaluation
test_emb = torch.load("embeddings/test_emb_chunk_0.pt", map_location="cpu")
test_mask = torch.load("embeddings/test_mask_chunk_0.pt", map_location="cpu")
with open("embeddings/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)

# ==============================================================================
# Ablation Loop Across Stacked Kernel Configurations
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
configs = [1, 2, 3, 4]
ablation_results = {}

with torch.no_grad():
    for n_k in configs:
        print(f"\n🔍 Evaluating Stacked Model with depth up to Layer {n_k}...")
        r1_scores, r2_scores, rL_scores = [], [], []
        
        for idx in range(len(test_ds)):
            original_sentences = test_ds[idx]["text"]
            labels = test_labels[idx]
            
            ground_truth_summary = " ".join([original_sentences[i] for i, label in enumerate(labels) if label >= 0.5])
            if not ground_truth_summary.strip():
                continue
                
            doc_emb = test_emb[idx].unsqueeze(0).to(device)
            doc_mask = test_mask[idx]
            
            # Model Inference with dynamic stacked depth
            scores = model(doc_emb, num_kernels=n_k).squeeze(0).cpu().numpy()
            
            num_sentences = min(len(original_sentences), int(doc_mask.sum().item()))
            valid_scores = scores[:num_sentences]
            
            # Dynamic Confidence Thresholding
            threshold = np.mean(valid_scores) + (np.std(valid_scores) * 0.5) 
            top_k_indices = np.where(valid_scores > threshold)[0]
            
            if len(top_k_indices) < 2:
                top_k = min(3, num_sentences)
                top_k_indices = valid_scores.argsort()[-top_k:][::-1]
                
            top_k_indices = sorted(top_k_indices)
            generated_summary = " ".join([original_sentences[i] for i in top_k_indices])
            
            # Calculate ROUGE
            rouge_scores = scorer.score(ground_truth_summary, generated_summary)
            r1_scores.append(rouge_scores['rouge1'].fmeasure)
            r2_scores.append(rouge_scores['rouge2'].fmeasure)
            rL_scores.append(rouge_scores['rougeL'].fmeasure)
            
        ablation_results[n_k] = (np.mean(r1_scores), np.mean(r2_scores), np.mean(rL_scores))

# ==============================================================================
# Final Ablation Table Output
# ==============================================================================
print("\n" + "="*65)
print("🏆 STACKED CNN KERNEL ABLATION STUDY RESULTS 🏆")
print("="*65)
print(f"{'Configuration':<28} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8}")
print("-" * 65)

names = {
    1: "Layer 1 Only (1-gram)", 
    2: "Stacked (1 to 2-gram)", 
    3: "Stacked (1 to 3-gram)", 
    4: "Full Hierarchical (1 to 4-gram) [Proposed]"
}

for n_k in configs:
    s = ablation_results[n_k]
    print(f"{names[n_k]:<28} | {s[0]:.4f}   | {s[1]:.4f}   | {s[2]:.4f}")
print("="*65)

# ==============================================================================
# Plotting Graph for Paper
# ==============================================================================
x_labels = ['Layer 1 (1-gram)', 'Stacked (1-2)', 'Stacked (1-3)', 'Full Stacked (1-4)']
r1_vals = [ablation_results[i][0] for i in configs]

plt.figure(figsize=(8, 5))
plt.plot(x_labels, r1_vals, marker='o', color='#2ca02c', linewidth=2.5, markersize=8, label='ROUGE-1')

plt.title('Ablation Study: Impact of Stacked Hierarchical CNN Depth', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Stacked CNN Configurations', fontsize=11, labelpad=10)
plt.ylabel('ROUGE-1 Score', fontsize=11, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('stacked_cnn_ablation_graph.png', dpi=300, bbox_inches='tight')
print("✅ Publication-ready graph saved as 'stacked_cnn_ablation_graph.png'!")
plt.show()