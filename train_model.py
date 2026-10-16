import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gc

# ==============================================================================
# 1. Hardware Setup
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon (MPS) for Training!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ==============================================================================
# 2. True Hierarchical (Stacked) CNN-LSTM Model Architecture
# ==============================================================================
class HierarchicalSequentialSummarizer(nn.Module):
    def __init__(self, embed_dim=768, cnn_filters=128, lstm_hidden=128):
        super(HierarchicalSequentialSummarizer, self).__init__()
        
        # True Stacked / Hierarchical CNN layers (Sequential Feature Refinement)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=2, padding='same')
        self.conv3 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding='same')
        
        self.relu = nn.ReLU()
        
        # Bi-directional LSTM takes the refined feature map from the final CNN layer
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True, bidirectional=True)
        
        # Sentence Scoring Head
        self.classifier = nn.Linear(lstm_hidden * 2, 1) # *2 for Bi-directional
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Seq_Len=512, Embed_Dim=768]
        # Transpose for Conv1d -> [Batch, Embed_Dim, Seq_Len]
        x = x.transpose(1, 2)  
        
        # Layer 1: Unigram features (1-gram)
        x = self.relu(self.conv1(x))
        
        # Layer 2: Bigram features built over unigram feature maps (2-gram)
        x = self.relu(self.conv2(x))
        
        # Layer 3: Fact-level features built over bigram feature maps (3-gram)
        x = self.relu(self.conv3(x))  # Shape: [Batch, cnn_filters, Seq_Len]
        
        # Transpose back for LSTM -> [Batch, Seq_Len, cnn_filters]
        x = x.transpose(1, 2)  
        
        # LSTM pass for sequence context
        lstm_out, _ = self.lstm(x)
        
        # Classifier pass
        logits = self.classifier(lstm_out).squeeze(-1) # Shape: [Batch, Seq_Len]
        
        return self.sigmoid(logits)

# ==============================================================================
# 3. Training Setup
# ==============================================================================
model = HierarchicalSequentialSummarizer().to(device)
criterion = nn.BCELoss() # Binary Cross Entropy for 0/1 labels
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
BATCH_SIZE = 16

print("\n📦 Loading Labels...")
with open("embeddings/train_labels.pkl", "rb") as f:
    train_labels = pickle.load(f)

chunk_files = [f for f in os.listdir("embeddings") if f.startswith("train_emb_chunk_")]
total_chunks = len(chunk_files)

# ==============================================================================
# 4. Chunk-by-Chunk Training Loop
# ==============================================================================
print(f"\n🔥 Starting Hierarchical Training for {EPOCHS} Epochs across {total_chunks} chunks...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    label_start_idx = 0
    
    for chunk_idx in range(total_chunks):
        emb_path = f"embeddings/train_emb_chunk_{chunk_idx}.pt"
        mask_path = f"embeddings/train_mask_chunk_{chunk_idx}.pt"
        
        if not os.path.exists(emb_path):
            continue
            
        emb_tensor = torch.load(emb_path, map_location="cpu")
        mask_tensor = torch.load(mask_path, map_location="cpu")
        
        chunk_size = emb_tensor.size(0)
        chunk_labels = train_labels[label_start_idx : label_start_idx + chunk_size]
        label_start_idx += chunk_size
        
        padded_labels = []
        for lbl in chunk_labels:
            lbl_tensor = torch.zeros(512)
            seq_len = min(len(lbl), 512)
            lbl_tensor[:seq_len] = lbl[:seq_len]
            padded_labels.append(lbl_tensor)
            
        labels_tensor = torch.stack(padded_labels)
        
        dataset = TensorDataset(emb_tensor, mask_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for batch_emb, batch_mask, batch_labels in loader:
            batch_emb = batch_emb.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_emb)
            
            active_loss = batch_mask.view(-1) == 1
            active_preds = predictions.view(-1)[active_loss]
            active_targets = batch_labels.view(-1)[active_loss]
            
            loss = criterion(active_preds, active_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        del emb_tensor, mask_tensor, labels_tensor, dataset, loader
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Processed Chunk {chunk_idx}/{total_chunks-1} | Current Loss: {loss.item():.4f}")

    avg_loss = total_loss / total_chunks
    print(f"✅ Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")
    print("-" * 50)

# ==============================================================================
# 5. Save Hierarchical Model
# ==============================================================================
torch.save(model.state_dict(), "hierarchical_cnn_lstm_summarizer.pth")
print("\n🎉 Hierarchical Model successfully trained and saved as 'hierarchical_cnn_lstm_summarizer.pth'!")