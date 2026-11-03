import torch
import pickle
import numpy as np
import torch.nn as nn
from datasets import load_from_disk
from rouge_score import rouge_scorer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
# 1. Model Architectures
# ==============================================================================
class CNNOnlySummarizer(nn.Module):
    def __init__(self, embed_dim=768, cnn_filters=128):
        super(CNNOnlySummarizer, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=2, padding='same')
        self.conv3 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=4, padding='same')
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(cnn_filters * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  
        x1, x2, x3, x4 = self.relu(self.conv1(x)), self.relu(self.conv2(x)), self.relu(self.conv3(x)), self.relu(self.conv4(x))
        cnn_out = torch.cat([x1, x2, x3, x4], dim=1).transpose(1, 2)  
        return self.sigmoid(self.classifier(cnn_out).squeeze(-1))

class LSTMOnlySummarizer(nn.Module):
    def __init__(self, embed_dim=768, lstm_hidden=128):
        super(LSTMOnlySummarizer, self).__init__()
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.sigmoid(self.classifier(lstm_out).squeeze(-1))

# ==============================================================================
# 2. Evaluation Function
# ==============================================================================
def run_evaluation(model, model_name, test_ds, test_emb, test_mask, test_labels, scorer):
    print(f"\n🚀 Evaluating {model_name}...")
    rouge1, rouge2, rougeL = [], [], []
    
    with torch.no_grad():
        for idx in range(len(test_ds)):
            original_sentences = test_ds[idx]["text"]
            labels = test_labels[idx]
            
            ground_truth = " ".join([original_sentences[i] for i, lbl in enumerate(labels) if lbl >= 0.5])
            if not ground_truth.strip(): continue
                
            doc_emb = test_emb[idx].unsqueeze(0).to(device)
            doc_mask = test_mask[idx]
            
            scores = model(doc_emb).squeeze(0).cpu().numpy()
            num_sentences = min(len(original_sentences), int(doc_mask.sum().item()))
            valid_scores = scores[:num_sentences]
            
            top_k = max(3, int(num_sentences * 0.3))
            top_k_indices = sorted(valid_scores.argsort()[-top_k:][::-1])
            generated = " ".join([original_sentences[i] for i in top_k_indices])
            
            r_scores = scorer.score(ground_truth, generated)
            rouge1.append(r_scores['rouge1'].fmeasure)
            rouge2.append(r_scores['rouge2'].fmeasure)
            rougeL.append(r_scores['rougeL'].fmeasure)

    r1, r2, rl = np.mean(rouge1), np.mean(rouge2), np.mean(rougeL)
    
    print("-" * 50)
    print(f"🏆 {model_name.upper()} RESULTS 🏆")
    print(f"ROUGE-1: {r1:.4f}\nROUGE-2: {r2:.4f}\nROUGE-L: {rl:.4f}")
    print("-" * 50)
    
    with open("ablation_results.txt", "a", encoding="utf-8") as f:
        f.write(f"{model_name:<16} | R1: {r1:.4f} | R2: {r2:.4f} | RL: {rl:.4f}\n")
    print(f"✅ Saved to 'ablation_results.txt'")

# ==============================================================================
# 3. Main Execution
# ==============================================================================
if __name__ == "__main__":
    print("📂 Loading Test Data for Ablations...")
    test_ds = load_from_disk("local_dataset/test").select(range(500))
    test_emb = torch.load("embeddings/test_emb_chunk_0.pt", map_location="cpu", weights_only=True)
    test_mask = torch.load("embeddings/test_mask_chunk_0.pt", map_location="cpu", weights_only=True)
    
    with open("embeddings/test_labels.pkl", "rb") as f:
        test_labels = pickle.load(f)
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Eval CNN-Only
    cnn_model = CNNOnlySummarizer().to(device)
    cnn_model.load_state_dict(torch.load("cnn_only.pth", map_location=device))
    cnn_model.eval()
    run_evaluation(cnn_model, "CNN-Only", test_ds, test_emb, test_mask, test_labels, scorer)
    
    # Eval LSTM-Only
    lstm_model = LSTMOnlySummarizer().to(device)
    lstm_model.load_state_dict(torch.load("lstm.pth", map_location=device))
    lstm_model.eval()
    run_evaluation(lstm_model, "LSTM-Only", test_ds, test_emb, test_mask, test_labels, scorer)