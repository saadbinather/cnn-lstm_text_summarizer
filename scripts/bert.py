import torch
import torch.nn as nn
import numpy as np
import nltk
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import pickle

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==============================================================================
# 1. Hardware Setup & Hierarchical Model Definition
# ==============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Evaluation with Vanilla BERT on: {device}")

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

# Load Model
model = HierarchicalSequentialSummarizer().to(device)
try:
    model.load_state_dict(torch.load("hierarchical_cnn_lstm_summarizer.pth", map_location=device))
    print("✅ Hierarchical Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Warning: 'hierarchical_cnn_lstm_summarizer.pth' nahi mili. Pehle model train karo!")
model.eval()

# ==============================================================================
# 2. Load Vanilla BERT & Test Dataset
# ==============================================================================
print("🧠 Loading Vanilla BERT (bert-base-uncased)...")
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
bert_model.eval()

print("📂 Loading test dataset and labels...")
test_ds = load_from_disk("local_dataset/test").select(range(300)) # Evaluation on 300 samples
with open("embeddings/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)

# ==============================================================================
# 3. Evaluation Loop with Vanilla BERT Embeddings
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
r1_scores, r2_scores, rL_scores = [], [], []

print("\n🔍 Generating Vanilla BERT embeddings & evaluating summaries...\n")

with torch.no_grad():
    for idx in tqdm(range(len(test_ds)), desc="Evaluating"):
        original_sentences = test_ds[idx]["text"]
        labels = test_labels[idx]
        
        ground_truth_summary = " ".join([original_sentences[i] for i, label in enumerate(labels) if label >= 0.5])
        if not ground_truth_summary.strip() or len(original_sentences) == 0:
            continue
            
        # 1. Generate Vanilla BERT embeddings for the test document on-the-fly
        inputs = tokenizer(original_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = bert_model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :] # [num_sentences, 768]
        
        # 2. Pad to [1, 512, 768] tensor format
        doc_emb = torch.zeros(1, 512, 768).to(device)
        valid_len = min(len(original_sentences), 512)
        doc_emb[0, :valid_len, :] = sentence_embeddings[:valid_len]
        
        # 3. Model Inference
        scores = model(doc_emb).squeeze(0).cpu().numpy()
        valid_scores = scores[:valid_len]
        
        # 4. Extract Top Sentences
        threshold = np.mean(valid_scores) + (np.std(valid_scores) * 0.5) 
        top_k_indices = np.where(valid_scores > threshold)[0]
        
        if len(top_k_indices) < 2:
            top_k = min(3, valid_len)
            top_k_indices = valid_scores.argsort()[-top_k:][::-1]
            
        top_k_indices = sorted(top_k_indices)
        generated_summary = " ".join([original_sentences[i] for i in top_k_indices])
        
        # 5. Calculate ROUGE
        rouge_scores = scorer.score(ground_truth_summary, generated_summary)
        r1_scores.append(rouge_scores['rouge1'].fmeasure)
        r2_scores.append(rouge_scores['rouge2'].fmeasure)
        rL_scores.append(rouge_scores['rougeL'].fmeasure)

# ==============================================================================
# 4. Final Output
# ==============================================================================
print("\n" + "="*50)
print("🏆 FINAL TEST ROUGE SCORES (VANILLA BERT) 🏆")
print(f"ROUGE-1: {np.mean(r1_scores):.4f}")
print(f"ROUGE-2: {np.mean(r2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rL_scores):.4f}")
print("="*50)