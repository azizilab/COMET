import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset


class CellStateEncoder(nn.Module):
    """
    Maps cell expression -> D-dim mixing weights.
    Warm-started so atom d initially activates for cluster d.
    """
    def __init__(self, G: int, D: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(G, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden // 2, D)
        self.D = D
    
    def init_from_cluster_labels(
        self, 
        X: np.ndarray, 
        cluster_labels: np.ndarray,
        n_epochs: int = 30
    ):
        """
        Pre-train the encoder to predict cluster membership.
        This gives each atom a distinct spatial territory before
        GRN training begins.
        """
        print("Pre-training cell state encoder on cluster labels...")
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        X_t  = torch.tensor(X, dtype=torch.float32)
        y_t  = torch.tensor(cluster_labels, dtype=torch.long)
        ds   = TensorDataset(X_t, y_t)
        dl   = DataLoader(ds, batch_size=256, shuffle=True)
        
        for ep in range(n_epochs):
            total = 0
            for xb, yb in dl:
                logits = self.head(self.encoder(xb))
                loss   = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            if ep % 10 == 0:
                print(f"  Encoder pre-train epoch {ep}: CE={total/len(dl):.3f}")
        
        print("Encoder pre-training complete.")
    
    def forward(self, rna: torch.Tensor, 
                temperature: float = 1.0) -> torch.Tensor:
        h      = self.encoder(rna)
        logits = self.head(h)
        return F.softmax(logits / temperature, dim=-1)   # (B, D)
    
    def get_hard_assignments(self, rna: torch.Tensor) -> torch.Tensor:
        """Argmax — use for evaluation/visualization only."""
        with torch.no_grad():
            return self.forward(rna).argmax(dim=1)

    def dirichlet_sharpening_loss(self, weights: torch.Tensor, #not used, as of now -- entropy loss used instead
                                   alpha: float = 0.1) -> torch.Tensor:
        """
        Sparse Dirichlet prior: log p(w | alpha) with alpha < 1.
        
        Math: L = -(alpha - 1) * mean_i sum_d log(w_id + eps)
        
        With alpha=0.1, this strongly favors weights at the corners
        of the simplex (one-hot), pushing cells to specialize in
        exactly one GRN atom. alpha=1.0 is uniform (no effect).
        """
        return -(alpha - 1.0) * torch.log(weights + 1e-8).mean()