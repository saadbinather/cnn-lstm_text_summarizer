import ast
import pandas as pd
import numpy as np
import nltk
import kagglehub
from kagglehub import KaggleDatasetAdapter
import networkx as nx
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)

# ==============================================================================
# 1. TextRank Summarization Function
# ==============================================================================
def textrank_summarize(text, top_k=3):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= top_k:
        return text
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        
        # Cosine Similarity graph
        sim_matrix = cosine_similarity(X)
        
        # PageRank (TextRank)
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph, max_iter=500)
        
        # Extract top_k and sort chronologically
        ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
        top_indices = sorted([item[2] for item in ranked_sentences[:top_k]])
        
        return " ".join([sentences[i] for i in top_indices])
    except:
        # Fallback if matrix is empty/fails
        return " ".join(sentences[:top_k])

# ==============================================================================
# 2. Load MIMIC-CXR Dataset & Parse
# ==============================================================================
print("📂 Loading MIMIC-CXR Dataset...")
file_path = "mimic_cxr_aug_train.csv"
df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "simhadrisadaram/mimic-cxr-dataset", file_path)

# Extract same 500 samples for fair comparison
df_test = df.sample(n=500, random_state=42).copy()

def parse_mimic(text_list_str):
    try:
        text_list = ast.literal_eval(text_list_str)
        if not text_list: return None, None
        full_text = text_list[0]
        if "Impression:" in full_text:
            parts = full_text.split("Impression:")
            findings = parts[0].replace("Findings:", "").strip()
            impression = parts[1].strip()
            return findings, impression
        return None, None
    except:
        return None, None

# ==============================================================================
# 3. Baseline Evaluation Loop
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

lead3_r1, lead3_r2, lead3_rl = [], [], []
textrank_r1, textrank_r2, textrank_rl = [], [], []

print("\n🔍 Running Lead-3 and TextRank Baselines for MIMIC-CXR...\n")

for idx, row_text in enumerate(tqdm(df_test['text'], desc="Evaluating")):
    findings, ground_truth_summary = parse_mimic(row_text)
    
    if not findings or not ground_truth_summary.strip():
        continue
        
    sentences = nltk.sent_tokenize(findings)
    if len(sentences) == 0:
        continue
        
    # --- LEAD-3 ---
    # Top 3 sentences
    lead3_summary = " ".join(sentences[:3])
    scores_lead3 = scorer.score(ground_truth_summary, lead3_summary)
    lead3_r1.append(scores_lead3['rouge1'].fmeasure)
    lead3_r2.append(scores_lead3['rouge2'].fmeasure)
    lead3_rl.append(scores_lead3['rougeL'].fmeasure)
    
    # --- TEXTRANK ---
    # Top 3 sentences for CXR (since impressions are small)
    textrank_summary = textrank_summarize(findings, top_k=3)
    scores_tr = scorer.score(ground_truth_summary, textrank_summary)
    textrank_r1.append(scores_tr['rouge1'].fmeasure)
    textrank_r2.append(scores_tr['rouge2'].fmeasure)
    textrank_rl.append(scores_tr['rougeL'].fmeasure)

# ==============================================================================
# 4. Results
# ==============================================================================
print("\n" + "="*50)
print("🏆 BASELINE RESULTS FOR MIMIC-CXR 🏆")
print("="*50)
print(f"{'Baseline':<12} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8}")
print("-" * 50)
print(f"{'Lead-3':<12} | {np.mean(lead3_r1):.4f}   | {np.mean(lead3_r2):.4f}   | {np.mean(lead3_rl):.4f}")
print(f"{'TextRank':<12} | {np.mean(textrank_r1):.4f}   | {np.mean(textrank_r2):.4f}   | {np.mean(textrank_rl):.4f}")
print("="*50)