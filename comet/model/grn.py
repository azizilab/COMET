import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRNDictionary(nn.Module):
    def __init__(self, G, T, M, D=4):
        super().__init__()
        self.D, self.G, self.T = D, G, T
        self.W_raw  = nn.Parameter(torch.zeros(D, G, T))   # residual atoms
        self.W_base = nn.Parameter(torch.zeros(G, T))      # shared base
        self.register_buffer("M", torch.tensor(M, dtype=torch.float32))
        self.register_buffer("TF_access", torch.zeros(T))
        self.register_buffer("Gene_access", torch.zeros(G))
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
        """Steady-state RNA prediction from TF expression (base + residual atoms).
        Always uses the full, dense A_base/A_dict — GRN inference itself is never
        pruned, only the trajectory-propagation path in production_from_lead is.
        """
        return (X_TF @ self.A_base.T) + self._atom_term(X_TF, weights)

    @staticmethod
    def _prune_top_frac(W, frac):
        """Zero out everything but the top `frac` fraction (by |weight|) of each
        module's entries, independently per module. W: (D, G, T) — D modules
        (e.g. the D atoms, or 1 for A_base treated as a single module).

        Deliberately per-module rather than per-cell: the per-cell effective GRN
        (A_base + sum_d w_d*A_d) is never materialized elsewhere in this class —
        _atom_term uses an einsum specifically to avoid forming a (B, G, T)
        tensor — and thresholding per cell would require building exactly that,
        for every batch. Per-module thresholding instead works on the fixed-size
        (D, G, T) parameter tensor itself, independent of batch size, so it costs
        a handful of topk calls over small matrices rather than a memory blowup.
        A single global threshold across all modules was considered too, but a
        module with systematically larger weights would then dominate the cutoff
        and could zero out another module's real (but smaller-magnitude) signal.
        """
        D = W.shape[0]
        with torch.no_grad():
            flat_abs = W.detach().abs().reshape(D, -1)
            n_nonzero = (flat_abs > 0).sum(dim=1)
            keep = torch.zeros_like(flat_abs, dtype=torch.bool)
            for d in range(D):
                nnz = int(n_nonzero[d].item())
                if nnz == 0:
                    continue
                k = max(1, int(round(nnz * frac)))
                if k >= nnz:
                    keep[d] = flat_abs[d] > 0
                else:
                    thresh = torch.topk(flat_abs[d], k).values.min()
                    keep[d] = flat_abs[d] >= thresh
            keep = keep.reshape(W.shape)
        return W * keep

    def production_from_lead(self, tf_lead, weights, prune_frac=0.1):
        """Effective per-cell GRN applied to the ATAC->RNA TF lead, for trajectory
        inference. Propagates through a per-module top-`prune_frac` sparsified
        copy of A_base/A_dict (keep only the strongest `prune_frac` of edges in
        each module) — GRN inference (predict_rna, and every loss that shapes
        A_base/A_dict) is untouched and keeps training on the full, dense
        weights; only this trajectory-propagation path sees the pruned GRN.
        """
        A_base_pruned = self._prune_top_frac(self.A_base.unsqueeze(0), prune_frac)[0]   # (G, T)
        A_dict_pruned = self._prune_top_frac(self.A_dict, prune_frac)                    # (D, G, T)
        P = torch.einsum('bt,dgt->dbg', tf_lead, A_dict_pruned)   # (D, B, G) — same fast path as _atom_term
        atom_term = (P * weights.T.unsqueeze(-1)).sum(0)          # (B, G)
        return (tf_lead @ A_base_pruned.T) + atom_term

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
        dict_term = (torch.abs(self.A_dict) * closed.unsqueeze(0).unsqueeze(0)).mean()
        base_term = (torch.abs(self.A_base) * closed).mean()
        return dict_term + base_term

    def gene_access_consistency_loss(self):
        """Same idea as tf_access_consistency_loss, but gates by the TARGET
        gene's own chromatin accessibility instead of the regulating TF's --
        an independent term (not multiplied with the TF-side one), scaling
        whole rows rather than whole columns."""
        closed = 1.0 - self.Gene_access.clamp(0, 1)
        dict_term = (torch.abs(self.A_dict) * closed.view(1, -1, 1)).mean()
        base_term = (torch.abs(self.A_base) * closed.unsqueeze(1)).mean()
        return dict_term + base_term


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


# ── Masked reconstruction (BERT-style self-supervision) ────────────────────
#   1. For each cell i, sample a binary mask m_i ∈ {0,1}^G  (fraction mask_frac)
#   2. Zero the masked genes' TF inputs
#   3. Reconstruct using the GRN:  x_hat = TF_expr @ A.T
#   4. Loss only on MASKED positions: MSE(x_hat[m], x[m])
#
# If the GRN weights are wrong, x_hat is random and the loss is high. The only
# way to minimise is to make A[g, :] point along the true regulatory relationship
# between TFs and gene g — learned from the *structure* of co-variation across
# cells, not from a labelled target.

def masked_reconstruction_loss(grn, mix_weights, rna_batch, tf_batch,
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
