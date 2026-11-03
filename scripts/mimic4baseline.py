import os
import glob
import pandas as pd
import numpy as np
import nltk
import kagglehub
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
        # Create TF-IDF vectors for sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        
        # Calculate Cosine Similarity to form a graph
        sim_matrix = cosine_similarity(X)
        
        # Apply PageRank algorithm (TextRank)
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph, max_iter=500) # Increased max_iter for convergence
        
        # Extract top_k sentences and sort them in original chronological order
        ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
        top_indices = sorted([item[2] for item in ranked_sentences[:top_k]])
        
        return " ".join([sentences[i] for i in top_indices])
    except:
        # Fallback if matrix is empty/fails
        return " ".join(sentences[:top_k])

# ==============================================================================
# 2. Load MIMIC-IV BHC Dataset
# ==============================================================================
print("📂 Loading MIMIC-IV BHC Dataset...")
dataset_path = kagglehub.dataset_download("aminexdr/bhc-mimic-iv-summary")
csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
df = pd.read_csv(csv_files[0])

# Using 500 samples with seed 42 to exactly match your previous test!
df_test = df.dropna(subset=['input', 'target']).sample(n=500, random_state=42).copy()

# ==============================================================================
# 3. Evaluation Loop
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

lead3_r1, lead3_r2, lead3_rl = [], [], []
textrank_r1, textrank_r2, textrank_rl = [], [], []

print("\n🔍 Running Lead-3 and TextRank Baselines...\n")

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    source_text = str(row['input'])
    ground_truth = str(row['target'])
    
    if not source_text.strip() or not ground_truth.strip():
        continue
        
    sentences = nltk.sent_tokenize(source_text)
    
    # --- LEAD-3 ---
    # Sirf pehle 3 sentences uthao
    lead3_summary = " ".join(sentences[:3])
    scores_lead3 = scorer.score(ground_truth, lead3_summary)
    lead3_r1.append(scores_lead3['rouge1'].fmeasure)
    lead3_r2.append(scores_lead3['rouge2'].fmeasure)
    lead3_rl.append(scores_lead3['rougeL'].fmeasure)
    
    # --- TEXTRANK ---
    # TF-IDF + PageRank graph extraction (Extract top 5 for fairer comparison with your model)
    textrank_summary = textrank_summarize(source_text, top_k=5)
    scores_tr = scorer.score(ground_truth, textrank_summary)
    textrank_r1.append(scores_tr['rouge1'].fmeasure)
    textrank_r2.append(scores_tr['rouge2'].fmeasure)
    textrank_rl.append(scores_tr['rougeL'].fmeasure)

# ==============================================================================
# 4. Results
# ==============================================================================
print("\n" + "="*50)
print(f"{'Baseline':<12} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8}")
print("-" * 50)
print(f"{'Lead-3':<12} | {np.mean(lead3_r1):.4f}   | {np.mean(lead3_r2):.4f}   | {np.mean(lead3_rl):.4f}")
print(f"{'TextRank':<12} | {np.mean(textrank_r1):.4f}   | {np.mean(textrank_r2):.4f}   | {np.mean(textrank_rl):.4f}")
print("="*50)