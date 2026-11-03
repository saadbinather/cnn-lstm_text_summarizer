import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import nltk
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import kagglehub

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==============================================================================
# 1. Download & Load MIMIC-IV BHC Dataset
# ==============================================================================
print("📂 Downloading MIMIC-IV BHC dataset...")
dataset_path = kagglehub.dataset_download("aminexdr/bhc-mimic-iv-summary")
print(f"✅ Dataset downloaded at: {dataset_path}")

csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("Bhai, is folder mein koi CSV file nahi mili!")

print(f"📄 Loading CSV file: {csv_files[0]}")
df = pd.read_csv(csv_files[0])

text_col = 'input'
summ_col = 'target'
print(f"🔍 Using '{text_col}' for input and '{summ_col}' for ground truth.")

df_test = df.dropna(subset=[text_col, summ_col]).sample(n=500, random_state=42).copy()

# ==============================================================================
# 2. Hardware Setup & True Hierarchical Model Definition
# ==============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Evaluation on: {device}")

class HierarchicalSequentialSummarizer(nn.Module):
    def __init__(self, embed_dim=768, cnn_filters=128, lstm_hidden=128):
        super(HierarchicalSequentialSummarizer, self).__init__()
        # True Stacked / Hierarchical CNN layers matching your training script
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

        
        x = x.transpose(1, 2)  # Shape: [Batch, Seq_Len, cnn_filters]
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out).squeeze(-1)
        return self.sigmoid(logits)

# Load Saved Hierarchical Model
model = HierarchicalSequentialSummarizer().to(device)
try:
    model.load_state_dict(torch.load("hierarchical_cnn_lstm_summarizer.pth", map_location=device))
    print("✅ Hierarchical CNN-LSTM Model weights loaded successfully!")
except FileNotFoundError:
    print("⚠️ Warning: 'hierarchical_cnn_lstm_summarizer.pth' nahi mili. Random weights use ho rahe hain.")
model.eval()

# ==============================================================================
# 3. Load SciBERT
# ==============================================================================
print("🧠 Loading SciBERT for Sentence Embeddings...")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
scibert = AutoModel.from_pretrained(MODEL_NAME).to(device)
scibert.eval()

# ==============================================================================
# 4. ROUGE Evaluation Loop (On-the-fly Embeddings)
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

print("\n🔍 Generating summaries and calculating ROUGE scores...\n")

with torch.no_grad():
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Evaluating"):
        source_text = str(row[text_col])
        ground_truth_summary = str(row[summ_col])
        
        if not source_text.strip() or not ground_truth_summary.strip():
            continue
            
        # 1. Split Text into sentences
        original_sentences = nltk.sent_tokenize(source_text)
        num_sentences = len(original_sentences)
        
        if num_sentences == 0:
            continue
            
        # 2. Get SciBERT embeddings for each sentence (using [CLS] token)
        inputs = tokenizer(original_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = scibert(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :] # Shape: [num_sentences, 768]
        
        # 3. Pad to [1, 512, 768]
        doc_emb = torch.zeros(1, 512, 768).to(device)
        valid_len = min(num_sentences, 512)
        doc_emb[0, :valid_len, :] = sentence_embeddings[:valid_len]
        
        # 4. Model Inference
        scores = model(doc_emb).squeeze(0).cpu().numpy() # [512]
        
        # 5. Extract Top Sentences
        valid_scores = scores[:valid_len]
        
        # Top 30% rule (minimum 3 sentences)
        top_k = max(3, int(valid_len * 0.3))
        top_k = min(top_k, valid_len) 
        
        top_k_indices = valid_scores.argsort()[-top_k:][::-1]
        top_k_indices = sorted(top_k_indices) # Maintain chronological order
        
        generated_summary = " ".join([original_sentences[i] for i in top_k_indices])
        
        # 6. Calculate ROUGE
        r_scores = scorer.score(ground_truth_summary, generated_summary)
        rouge1_scores.append(r_scores['rouge1'].fmeasure)
        rouge2_scores.append(r_scores['rouge2'].fmeasure)
        rougeL_scores.append(r_scores['rougeL'].fmeasure)

# ==============================================================================
# 5. Final Results
# ==============================================================================
print("\n" + "-" * 50)
print("🏆 FINAL AVERAGE ROUGE SCORES (MIMIC-IV BHC - HIERARCHICAL) 🏆")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print("-" * 50)