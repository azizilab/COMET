import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans

class GRNDictionary1(nn.Module):
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
    
    def per_tf_norm_loss(self) -> torch.Tensor:
        col_norms = torch.norm(self.A_dict, dim=1)    # (D, T)
        target    = (self.M.sum(dim=0).clamp(min=1) ** 0.5) * 0.1
        return ((col_norms - target.unsqueeze(0)) ** 2).mean()
    

class GRNDictionary(nn.Module):
    def __init__(self, G, T, M, D=4):
        super().__init__()
        self.D, self.G, self.T = D, G, T
        self.W_raw  = nn.Parameter(torch.zeros(D, G, T))   # residual atoms
        self.W_base = nn.Parameter(torch.zeros(G, T))      # shared base
        self.register_buffer("M", torch.tensor(M, dtype=torch.float32))
        self.register_buffer("TF_access", torch.zeros(T))
        self.register_buffer("mean_tf",  torch.zeros(T))

    def init_from_atoms(self, W_init):
        with torch.no_grad():
            self.W_raw.copy_(torch.tensor(np.arctanh(np.clip(W_init, -0.95, 0.95)),
                                          dtype=torch.float32))
        print("GRN atoms initialized from archetype soft-LFC profiles (base starts at 0).")

    @property
    def A_dict(self):
        return torch.tanh(self.W_raw) * self.M.unsqueeze(0)   # (D, G, T)

    @property
    def A_base(self):
        return torch.tanh(self.W_base) * self.M               # (G, T)

    def _atom_term(self, X_in, weights):
        # sum_d w_d (X_in @ A_d.T) without materializing (B,G,T)
        P = torch.einsum('bt,dgt->dbg', X_in, self.A_dict)    # (D, B, G)
        return (P * weights.T.unsqueeze(-1)).sum(0)           # (B, G)

    def predict_rna(self, X_TF, weights):
        """Steady-state RNA prediction from TF expression (base + residual atoms)."""
        return (X_TF @ self.A_base.T) + self._atom_term(X_TF, weights)

    def production_from_lead(self, tf_lead, weights):
        """Effective per-cell GRN applied to the ATAC->RNA TF lead (for velocity)."""
        return (tf_lead @ self.A_base.T) + self._atom_term(tf_lead, weights)

    def effective_A(self, weights_vec):
        """weights_vec: (D,) -> (G, T) effective GRN including base."""
        return self.A_base + torch.einsum('d,dgt->gt', weights_vec, self.A_dict)

    def sparsity_loss(self):
        return torch.abs(self.A_dict).mean() + torch.abs(self.A_base).mean()

    def diversity_loss(self):
        """Keep residual atoms distinct in parameter AND prediction space."""
        A_flat = self.A_dict.reshape(self.D, -1)
        A_norm = F.normalize(A_flat, dim=1)
        eye = torch.eye(self.D, device=A_flat.device)
        param_div = ((A_norm @ A_norm.T) * (1 - eye)).abs().mean()

        pred = torch.einsum('dgt,t->dg', self.A_dict, self.mean_tf)
        pred = F.normalize(pred, dim=1)
        pred_div = ((pred @ pred.T) * (1 - eye)).abs().mean()
        return 0.5 * param_div + 0.5 * pred_div

    def tf_access_consistency_loss(self):
        closed = 1.0 - self.TF_access.clamp(0, 1)
        return (torch.abs(self.A_dict) * closed.unsqueeze(0).unsqueeze(0)).mean()




def init_atoms_from_soft_targets(X, q, M, A_tfs):
    """Soft-LFC init: each atom's weights come from cells weighted toward its archetype."""
    D = q.shape[1]
    G = X.shape[1]
    W_init = np.zeros((D, G, len(A_tfs)), dtype=np.float32)
    for d in range(D):
        w = q[:, d]                                   # (N,) soft membership
        wsum = w.sum() + 1e-8
        mean_in  = (X * w[:, None]).sum(0) / wsum
        mean_out = (X * (1 - w)[:, None]).sum(0) / ((1 - w).sum() + 1e-8)
        lfc = np.log1p(mean_in) - np.log1p(mean_out)  # (G,)
        for t_i in range(len(A_tfs)):
            col = M[:, t_i]
            if col.sum() == 0:
                continue
            lfc_masked = lfc * col
            rng = np.abs(lfc_masked).max() + 1e-8
            W_init[d, :, t_i] = lfc_masked / rng * 0.3
    return W_init



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
 
def masked_reconstruction_loss1(grn, mix_weights, rna_batch, tf_batch,
                                tf_indices_in_G, mask_frac=0.40):
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



def masked_reconstruction_loss2(
    grn,
    mix_weights,
    rna_batch,
    tf_batch,
    tf_indices_in_G,
    mask_frac=0.40,
):
    B, G = rna_batch.shape
    device = rna_batch.device
    T = len(tf_indices_in_G)

    # --------------------------------------------------
    # Sample mask only over TFs
    # --------------------------------------------------
    tf_mask = torch.rand(B, T, device=device) < mask_frac

    # --------------------------------------------------
    # Zero masked TF inputs
    # --------------------------------------------------
    tf_input_masked = tf_batch.clone()
    tf_input_masked[tf_mask] = 0.0

    # --------------------------------------------------
    # Predict transcriptome
    # --------------------------------------------------
    rna_pred = grn.predict_rna(tf_input_masked, mix_weights)

    # --------------------------------------------------
    # Build loss mask:
    #   - all non-TF genes included
    #   - only masked TF genes included
    # --------------------------------------------------
    loss_mask = torch.ones(B, G, dtype=torch.bool, device=device)

    # Remove all TF genes first
    loss_mask[:, tf_indices_in_G] = False

    # Add back only the masked TFs
    loss_mask[:, tf_indices_in_G] = tf_mask

    # Safety check
    if loss_mask.sum() == 0:
        return rna_pred.mean() * 0.0

    return F.mse_loss(
        rna_pred[loss_mask],
        rna_batch[loss_mask]
    )