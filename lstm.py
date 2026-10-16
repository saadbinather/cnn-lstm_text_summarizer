import os, pickle, gc, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class LSTMOnlySummarizer(nn.Module):
    def __init__(self, embed_dim=768, lstm_hidden=128):
        super(LSTMOnlySummarizer, self).__init__()
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.sigmoid(self.classifier(lstm_out).squeeze(-1))

model = LSTMOnlySummarizer().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

with open("embeddings/train_labels.pkl", "rb") as f:
    train_labels = pickle.load(f)

total_chunks = len([f for f in os.listdir("embeddings") if f.startswith("train_emb_chunk_")])

print("🔥 Training LSTM-Only Model...")
for epoch in range(3):
    label_start_idx = 0
    for chunk_idx in range(total_chunks):
        emb = torch.load(f"embeddings/train_emb_chunk_{chunk_idx}.pt", map_location="cpu", weights_only=True)
        mask = torch.load(f"embeddings/train_mask_chunk_{chunk_idx}.pt", map_location="cpu", weights_only=True)
        
        c_size = emb.size(0)
        c_labels = train_labels[label_start_idx : label_start_idx + c_size]
        label_start_idx += c_size
        
        lbl_tensor = torch.stack([torch.cat([torch.tensor(l[:512]), torch.zeros(max(0, 512-len(l)))])[:512] for l in c_labels])
        
        loader = DataLoader(TensorDataset(emb, mask, lbl_tensor), batch_size=16, shuffle=True)
        for b_emb, b_mask, b_lbl in loader:
            b_emb, b_mask, b_lbl = b_emb.to(device), b_mask.to(device), b_lbl.to(device)
            optimizer.zero_grad()
            preds = model(b_emb)
            active = b_mask.view(-1) == 1
            loss = criterion(preds.view(-1)[active], b_lbl.view(-1)[active])
            loss.backward()
            optimizer.step()
        
        del emb, mask, lbl_tensor, loader
        if device.type == 'mps': torch.mps.empty_cache()
        gc.collect()
    print(f"✅ Epoch {epoch+1} done.")

torch.save(model.state_dict(), "lstm.pth")
print("🎉 Saved as lstm_only.pth")