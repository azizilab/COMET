import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class CellStateEncoder(nn.Module):
    """RNA -> D mixing weights. Warm-started to reproduce the geometric soft targets q
    (not hard cluster labels)."""
    def __init__(self, G, D, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(G, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU())
        self.head = nn.Linear(hidden // 2, D); self.D = D

    def init_from_soft_targets(self, X, q, n_epochs=40, lr=1e-3, batch=256):
        print("  Pre-training encoder to match geometric soft targets...")
        self.train(); opt = torch.optim.Adam(self.parameters(), lr=lr)
        X_t = torch.tensor(X, dtype=torch.float32); q_t = torch.tensor(q, dtype=torch.float32)
        dl  = DataLoader(TensorDataset(X_t, q_t), batch_size=batch, shuffle=True)
        for ep in range(n_epochs):
            tot = 0.0
            for xb, qb in dl:
                logp = F.log_softmax(self.head(self.encoder(xb)), dim=-1)
                loss = (qb * (torch.log(qb + 1e-8) - logp)).sum(1).mean()   # KL(q||p)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if ep % 10 == 0: print(f"    encoder pretrain epoch {ep}: KL={tot/len(dl):.4f}")

    def forward(self, rna, temperature=1.0):
        return F.softmax(self.head(self.encoder(rna)) / temperature, dim=-1)
