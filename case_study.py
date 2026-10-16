import torch
import numpy as np
import pickle
import torch.nn as nn
from datasets import load_from_disk
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt', quiet=True)

# ==============================================================================
# 1. Hardware & Hierarchical Model Definition
# ==============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Running Qualitative Comparison on: {device}")

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
    print("✅ Stacked Hierarchical Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Warning: 'hierarchical_cnn_lstm_summarizer.pth' nahi mili. Pehle model train karo!")
model.eval()

# ==============================================================================
# 2. TextRank Baseline Helper
# ==============================================================================
def get_textrank_indices(sentences, top_k=3):
    if len(sentences) <= top_k:
        return list(range(len(sentences)))
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(X)
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph, max_iter=500)
        ranked = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
        return sorted([item[1] for item in ranked[:top_k]])
    except:
        return list(range(min(top_k, len(sentences))))

# ==============================================================================
# 3. Search for a Rich Document (Where model differs from Lead-3)
# ==============================================================================
test_ds = load_from_disk("local_dataset/test")
test_emb = torch.load("embeddings/test_emb_chunk_0.pt", map_location="cpu")
test_mask = torch.load("embeddings/test_mask_chunk_0.pt", map_location="cpu")

with open("embeddings/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)

selected_idx = None
original_sentences = None
labels = None
model_indices = None

with torch.no_grad():
    for i in range(len(test_ds)):
        sents = test_ds[i]["text"]
        lbls = test_labels[i]
        
        gt = [idx for idx, l in enumerate(lbls) if l >= 0.5]
        if len(sents) < 6 or len(gt) == 0:
            continue
            
        doc_emb = test_emb[i].unsqueeze(0).to(device)
        scores = model(doc_emb).squeeze(0).cpu().numpy()
        num_sents = min(len(sents), int(test_mask[i].sum().item()))
        valid_scores = scores[:num_sents]
        
        threshold = np.mean(valid_scores) + (np.std(valid_scores) * 0.5)
        m_indices = np.where(valid_scores > threshold)[0]
        if len(m_indices) < 2:
            m_indices = valid_scores.argsort()[-min(3, num_sents):][::-1]
        m_indices = sorted(m_indices)
        
        lead3 = [0, 1, 2]
        if m_indices != lead3:
            selected_idx = i
            original_sentences = sents
            labels = lbls
            model_indices = m_indices
            break

if selected_idx is None:
    selected_idx = 42
    original_sentences = test_ds[selected_idx]["text"]
    labels = test_labels[selected_idx]
    doc_emb = test_emb[selected_idx].unsqueeze(0).to(device)
    scores = model(doc_emb).squeeze(0).cpu().numpy()
    num_sents = min(len(original_sentences), int(test_mask[selected_idx].sum().item()))
    valid_scores = scores[:num_sents]
    model_indices = sorted(valid_scores.argsort()[-3:][::-1])

gt_indices = [idx for idx, l in enumerate(labels) if l >= 0.5]
lead3_indices = [0, 1, 2] if len(original_sentences) >= 3 else list(range(len(original_sentences)))
textrank_indices = get_textrank_indices(original_sentences, top_k=len(model_indices))

# ==============================================================================
# 4. Print Comparison for Paper
# ==============================================================================
print("="*90)
print(f"📄 QUALITATIVE CASE STUDY: Document #{selected_idx} (Total Sentences: {len(original_sentences)})")
print("="*90)

print("\n🎯 GROUND TRUTH SUMMARY (Target):")
for i in gt_indices:
    print(f"  [{i}] {original_sentences[i]}")

print("\n❌ LEAD-3 BASELINE (Blindly picks prefix):")
for i in lead3_indices:
    print(f"  [{i}] {original_sentences[i]}")

print("\n📊 TEXTRANK BASELINE (TF-IDF Graph Based):")
for i in textrank_indices:
    match_tag = " 🔥" if i in gt_indices else ""
    print(f"  [{i}] {original_sentences[i]}{match_tag}")

print("\n✅ PROPOSED HIERARCHICAL STACKED CNN-LSTM (Contextual Extraction):")
for i in model_indices:
    match_tag = " 🔥 (MATCH!)" if i in gt_indices else ""
    print(f"  [{i}] {original_sentences[i]}{match_tag}")
print("="*90)