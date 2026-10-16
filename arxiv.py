import torch
import torch.nn as nn
import numpy as np
import nltk
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from rouge_score import rouge_scorer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==============================================================================
# 1. Hardware & Model
# ==============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Evaluation on: {device}")

class ExtractiveSummarizer(nn.Module):
    def __init__(self, embed_dim=768, cnn_filters=128, lstm_hidden=128):
        super(ExtractiveSummarizer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=3, padding='same')
        self.conv4 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=4, padding='same')
        self.relu = nn.ReLU()
        
        cnn_out_dim = cnn_filters * 4
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        cnn_out = torch.cat([x1, x2, x3, x4], dim=1).transpose(1, 2)  
        lstm_out, _ = self.lstm(cnn_out)
        logits = self.classifier(lstm_out).squeeze(-1)
        return self.sigmoid(logits)

model = ExtractiveSummarizer().to(device)
try:
    model.load_state_dict(torch.load("cnn_lstm_summarizer.pth", map_location=device))
    print("✅ CNN-LSTM Model weights loaded!")
except FileNotFoundError:
    print("⚠️ Warning: Model nahi mila.")
model.eval()

# ==============================================================================
# 2. Load SciBERT & ArXiv (5 Samples)
# ==============================================================================
print("🧠 Loading SciBERT...")
MODEL_NAME = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
scibert = AutoModel.from_pretrained(MODEL_NAME).to(device)
scibert.eval()

print("📂 Loading ArXiv Test Dataset (Full length, 5 samples)...")
dataset = load_dataset("ccdv/arxiv-summarization", split="test[:5]", trust_remote_code=True)

# ==============================================================================
# 3. ROUGE Evaluation Loop 
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

with torch.no_grad():
    for i, item in enumerate(dataset):
        print(f"\nProcessing Paper {i+1}/5...")
        source_text = item["article"]
        ground_truth_summary = item["abstract"]
        
        original_sentences = nltk.sent_tokenize(source_text)
        num_sentences = len(original_sentences)
        print(f"Total Sentences in this paper: {num_sentences}")
        
        if num_sentences < 5: continue
            
        # Batch processing for SciBERT to avoid OOM on full length
        batch_size = 64
        all_embeddings = []
        
        for start_idx in range(0, num_sentences, batch_size):
            batch_sents = original_sentences[start_idx : start_idx + batch_size]
            inputs = tokenizer(batch_sents, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = scibert(**inputs)
            all_embeddings.append(outputs.last_hidden_state[:, 0, :])
            
        # Combine all sentence embeddings [Total_Sentences, 768]
        sentence_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Add batch dimension for CNN-LSTM [1, Total_Sentences, 768]
        doc_emb = sentence_embeddings.unsqueeze(0)
        
        # 4. Model Inference (Handles dynamic length automatically!)
        scores = model(doc_emb).squeeze(0).cpu().numpy() 
        
        # 5. Extract Top Sentences
        top_k = min(6, num_sentences)  
        top_k_indices = sorted(scores.argsort()[-top_k:][::-1])
        generated_summary = " ".join([original_sentences[idx] for idx in top_k_indices])
        
        # 6. Calculate ROUGE
        r_scores = scorer.score(ground_truth_summary, generated_summary)
        rouge1_scores.append(r_scores['rouge1'].fmeasure)
        rouge2_scores.append(r_scores['rouge2'].fmeasure)
        rougeL_scores.append(r_scores['rougeL'].fmeasure)
        
        print(f"Paper {i+1} ROUGE-1: {r_scores['rouge1'].fmeasure:.4f}")

# ==============================================================================
# 4. Final Results
# ==============================================================================
print("\n" + "-" * 50)
print("🏆 FINAL AVERAGE ROUGE SCORES (ArXiv - 5 Full Papers) 🏆")
print(f"ROUGE-1: {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-2: {np.mean(rouge2_scores):.4f}")
print(f"ROUGE-L: {np.mean(rougeL_scores):.4f}")
print("-" * 50)