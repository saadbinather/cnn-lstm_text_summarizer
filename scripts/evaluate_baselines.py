import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from rouge_score import rouge_scorer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==============================================================================
# 1. Load PubMed Test Data & Oracle Labels
# ==============================================================================
NUM_SAMPLES = 300  # Evaluation sample size matching paper setup
print(f"📂 Loading PubMed Test Dataset ({NUM_SAMPLES} samples)...")

test_ds = load_from_disk("local_dataset/test").select(range(NUM_SAMPLES))

with open("embeddings/test_labels.pkl", "rb") as f:
    test_labels = pickle.load(f)

# ==============================================================================
# 2. Setup Summarizers & ROUGE Scorer
# ==============================================================================
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
tr_summarizer = TextRankSummarizer()

l3_scores = {'r1': [], 'r2': [], 'rL': []}
tr_scores = {'r1': [], 'r2': [], 'rL': []}

print("🚀 Running Baseline Evaluations on PubMed...\n")

# ==============================================================================
# 3. Baseline Evaluation Loop
# ==============================================================================
for idx in tqdm(range(len(test_ds)), desc="Evaluating Baselines"):
    sents = test_ds[idx]["text"]
    labels = test_labels[idx]
    
    # Construct Ground Truth Summary from Oracle Labels
    ground_truth = " ".join([sents[i] for i, lbl in enumerate(labels) if lbl >= 0.5])
    
    if not ground_truth.strip() or len(sents) < 3:
        continue
        
    # 1. Lead-3 Baseline (First 3 sentences)
    lead3_sum = " ".join(sents[:3])
    
    # 2. TextRank Baseline (Top 3 sentences via Sumy)
    try:
        doc_text = " ".join(sents)
        parser = PlaintextParser.from_string(doc_text, Tokenizer("english"))
        tr_sentences = tr_summarizer(parser.document, 3)
        tr_sum = " ".join([str(s) for s in tr_sentences])
        if not tr_sum.strip():
            tr_sum = lead3_sum
    except Exception:
        tr_sum = lead3_sum
    
    # Calculate Lead-3 ROUGE
    r_l3 = scorer.score(ground_truth, lead3_sum)
    for k, v in zip(['r1', 'r2', 'rL'], ['rouge1', 'rouge2', 'rougeL']):
        l3_scores[k].append(r_l3[v].fmeasure)
        
    # Calculate TextRank ROUGE
    r_tr = scorer.score(ground_truth, tr_sum)
    for k, v in zip(['r1', 'r2', 'rL'], ['rouge1', 'rouge2', 'rougeL']):
        tr_scores[k].append(r_tr[v].fmeasure)

# ==============================================================================
# 4. Output Results Table
# ==============================================================================
print("\n" + "="*50)
print("🏆 PUBMED BASELINE EVALUATION RESULTS 🏆")
print("="*50)
print(f"{'Baseline':<12} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8}")
print("-" * 50)
print(f"{'Lead-3':<12} | {np.mean(l3_scores['r1']):.4f}   | {np.mean(l3_scores['r2']):.4f}   | {np.mean(l3_scores['rL']):.4f}")
print(f"{'TextRank':<12} | {np.mean(tr_scores['r1']):.4f}   | {np.mean(tr_scores['r2']):.4f}   | {np.mean(tr_scores['rL']):.4f}")
print("="*50)