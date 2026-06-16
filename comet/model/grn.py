import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans

class GRNDictionary(nn.Module):
    def __init__(self, G: int, T: int, M: np.ndarray, D: int = 4):
        super().__init__()
        self.D = D
        self.G = G
        self.T = T
        
        self.W_raw = nn.Parameter(torch.zeros(D, G, T))
        
        M_t = torch.tensor(M, dtype=torch.float32)
        self.register_buffer("M", M_t)
        self.register_buffer("TF_access", torch.zeros(T))
        # Register mean TF input for prediction-space diversity (set before training)
        self.register_buffer("mean_tf", torch.zeros(T))
    
    def init_from_clusters(self, W_init: np.ndarray):
        """Call this after construction to break symmetry."""
        with torch.no_grad():
            W_clamped = np.clip(W_init, -0.95, 0.95)
            self.W_raw.copy_(torch.tensor(np.arctanh(W_clamped), dtype=torch.float32))
        print("GRN dictionary initialized from cluster LFC profiles.")
    
    @property
    def A_dict(self) -> torch.Tensor:
        return torch.tanh(self.W_raw) * self.M.unsqueeze(0)   # (D, G, T)
    
    def get_cell_grn(self, weights: torch.Tensor) -> torch.Tensor:
        """weights: (B, D) -> (B, G, T) — kept for compatibility but avoid in hot path"""
        return torch.einsum('bd,dgt->bgt', weights, self.A_dict)

    def predict_rna(self, X_TF: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        X_TF: (B, T), weights: (B, D) -> (B, G)

        Fast path: never materialise (B, G, T).
        P_d = X_TF @ A_d.T  shape (B, G)   — one matmul per atom
        out  = sum_d  w_d * P_d             — weighted sum
        Total memory: D * B * G  instead of B * G * T
        """
        # A_dict: (D, G, T)  →  per-atom predictions: (D, B, G)
        # einsum 'bt,dgt->dbg' then weight by w: (B,D) → (B,G)
        P = torch.einsum('bt,dgt->dbg', X_TF, self.A_dict)   # (D, B, G)
        # weights: (B, D) → (D, B, 1) for broadcasting
        out = (P * weights.T.unsqueeze(-1)).sum(dim=0)         # (B, G)
        return out

    def get_effective_A(self, weights: torch.Tensor) -> torch.Tensor:
        """weights: (D,) -> (G, T)  — for single cell or population mean"""
        return torch.einsum('d,dgt->gt', weights, self.A_dict)
    
    def sparsity_loss(self) -> torch.Tensor:
        return torch.abs(self.A_dict).mean()
    
    def diversity_loss(self) -> torch.Tensor:
        """
        TWO-PART diversity loss:
        
        Part 1 — Parameter-space cosine repulsion (original):
          Penalize cosine similarity of the flattened atom weight vectors.
        
        Part 2 — Prediction-space repulsion (NEW):
          Evaluate each atom on mean TF input and penalize similarity
          of *outputs*. This is what matters for interpretability —
          two atoms with different weights but correlated outputs are
          biologically redundant.
        """
        # Part 1: parameter space 
        A_flat = self.A_dict.reshape(self.D, -1)       # (D, G*T)
        A_norm = F.normalize(A_flat, dim=1)
        gram   = A_norm @ A_norm.T                     # (D, D)
        eye    = torch.eye(self.D, device=gram.device)
        param_div = (gram * (1 - eye)).abs().mean()
        
        # Part 2: prediction space 
        # Evaluate each atom on mean TF input -> (D, G)
        tf_mean = self.mean_tf.unsqueeze(0)             # (1, T)
        pred_per_atom = torch.einsum(
            'dgt,t->dg', self.A_dict, self.mean_tf
        )                                               # (D, G)
        pred_norm = F.normalize(pred_per_atom, dim=1)  # (D, G)
        pred_gram = pred_norm @ pred_norm.T             # (D, D)
        pred_div  = (pred_gram * (1 - eye)).abs().mean()
        
        return 0.5 * param_div + 0.5 * pred_div
    
    def tf_access_consistency_loss(self) -> torch.Tensor:
        closed = 1.0 - self.TF_access.clamp(0, 1)
        return (torch.abs(self.A_dict) * 
                closed.unsqueeze(0).unsqueeze(0)).mean()
    


def initialize_grn_dictionary_from_clusters(
    X: np.ndarray,          # (C, G) RNA
    V_dec2: np.ndarray,     # (C, 2) for clustering
    M: np.ndarray,          # (G, T) motif mask
    D: int,                 # number of atoms
    A_tfs: np.ndarray,
    sub_g2i: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      cluster_labels : (C,) int — which cluster each cell belongs to
      W_init         : (D, G, T) float32 — per-atom initial weights
    """
    # 1. Cluster in Decipher space (already a good low-dim representation)
    print(f"Clustering into {D} groups in Decipher space...")
    km = KMeans(n_clusters=D, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(V_dec2)
    
    # 2. For each cluster, find marker genes (high mean, low in others)
    W_init = np.zeros((D, X.shape[1], len(A_tfs)), dtype=np.float32)
    
    for d in range(D):
        in_mask  = cluster_labels == d
        out_mask = ~in_mask
        
        if in_mask.sum() < 3:
            continue
        
        # Mean expression inside vs outside this cluster
        mean_in  = X[in_mask].mean(axis=0)    # (G,)
        mean_out = X[out_mask].mean(axis=0)   # (G,)
        
        # Log fold change as a simple specificity score
        lfc = np.log1p(mean_in) - np.log1p(mean_out)  # (G,)
        
        # Initialize weights for this atom proportional to cluster-specific LFC
        # Only where motif mask allows (M > 0)
        for t_i in range(len(A_tfs)):
            col = M[:, t_i]                   # (G,) binary motif mask
            if col.sum() == 0:
                continue
            # Weight = LFC scaled to [-0.3, 0.3] within motif-allowed genes
            lfc_masked = lfc * col
            lfc_range  = np.abs(lfc_masked).max() + 1e-8
            W_init[d, :, t_i] = lfc_masked / lfc_range * 0.3
    
    print(f"Cluster sizes: {[int((cluster_labels==d).sum()) for d in range(D)]}")
    return cluster_labels, W_init


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7  –  Masked Reconstruction (BERT task)
# ═══════════════════════════════════════════════════════════════════════════
#
# Algorithm:
#   1. For each cell i, sample a binary mask m_i ∈ {0,1}^G  (fraction p_mask)
#   2. Zero out the masked genes:  x_masked = x * (1 - m)
#   3. Reconstruct using GRN:      x_hat    = TF_expr @ A.T
#   4. Loss only on MASKED positions: MSE(x_hat[m], x[m])
#
# Why this works:
#   If the GRN weights are wrong, x_hat will be random and the loss is high.
#   The ONLY way to minimise is to make A[g, :] point in the direction of the
#   true regulatory relationship between TFs and gene g — learned from the
#   *structure* of co-variation across cells, not from a labelled target.
 
def masked_reconstruction_loss(grn, mix_weights, rna_batch, tf_batch,
                                tf_indices_in_G, mask_frac=0.30):
    """
    mix_weights : (B, D) — already computed by caller, do NOT recompute encoder.
    Removing the redundant encoder forward pass saves ~15% of Phase 1 time.
    """
    B, G = rna_batch.shape
    device = rna_batch.device

    mask = torch.rand(B, G, device=device) < mask_frac

    # Mask out TF inputs for masked gene positions
    tf_input_masked = tf_batch.clone()
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    tf_input_masked[batch_idx,
                    torch.arange(len(tf_indices_in_G), device=device)] *= \
        (~mask[batch_idx, tf_indices_in_G]).float()

    rna_pred = grn.predict_rna(tf_input_masked, mix_weights)

    if mask.sum() == 0:
        return rna_pred.mean() * 0.0
    return F.mse_loss(rna_pred[mask], rna_batch[mask])