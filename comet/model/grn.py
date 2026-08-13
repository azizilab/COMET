import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def select_archetypes_fps(Z, D, seed=0):
    """Furthest-point sampling -> indices of D extreme cells (manifold corners)."""
    c = Z.mean(axis=0)
    first = int(np.argmax(((Z - c) ** 2).sum(1))); idx = [first]
    dmin = ((Z - Z[first]) ** 2).sum(1)
    for _ in range(D - 1):
        nxt = int(np.argmax(dmin)); idx.append(nxt)
        dmin = np.minimum(dmin, ((Z - Z[nxt]) ** 2).sum(1))
    return np.array(idx, dtype=int)

def geometric_soft_targets(Z, arch_idx, tau_scale=0.5):
    """q_i = softmax(-||z_i - z_arch||^2 / tau). Returns (N, D), rows sum to 1."""
    A = Z[arch_idx]
    d2 = ((Z[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    tau = np.median(d2) * tau_scale + 1e-8
    q = np.exp(-d2 / tau); q = q / (q.sum(1, keepdims=True) + 1e-8)
    return q.astype(np.float32)

def init_atoms_from_soft_targets(X, q, M, A_tfs):
    """Soft-LFC init: each atom's weights come from cells weighted toward its archetype."""
    D = q.shape[1]; G = X.shape[1]; W_init = np.zeros((D, G, len(A_tfs)), dtype=np.float32)
    for d in range(D):
        w = q[:, d]; wsum = w.sum() + 1e-8
        mean_in  = (X * w[:, None]).sum(0) / wsum
        mean_out = (X * (1 - w)[:, None]).sum(0) / ((1 - w).sum() + 1e-8)
        lfc = np.log1p(mean_in) - np.log1p(mean_out)
        for t_i in range(len(A_tfs)):
            col = M[:, t_i]
            if col.sum() == 0: continue
            lfc_masked = lfc * col; rng = np.abs(lfc_masked).max() + 1e-8
            W_init[d, :, t_i] = lfc_masked / rng * 0.3
    return W_init


class GRNDictionary(nn.Module):
    """predict_rna = A_base @ tf + Σ_d w_d (A_d @ tf). The shared base carries the common
    regulatory program so the D residual atoms only encode lineage-specific deviations."""
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
            self.W_raw.copy_(torch.tensor(np.arctanh(np.clip(W_init, -0.95, 0.95)), dtype=torch.float32))
        print("  GRN atoms initialized from archetype soft-LFC profiles (base starts at 0).")

    @property
    def A_dict(self): return torch.tanh(self.W_raw) * self.M.unsqueeze(0)   # (D, G, T)
    @property
    def A_base(self): return torch.tanh(self.W_base) * self.M               # (G, T)

    def _atom_term(self, X_in, weights):
        P = torch.einsum('bt,dgt->dbg', X_in, self.A_dict)    # (D, B, G)
        return (P * weights.T.unsqueeze(-1)).sum(0)           # (B, G)

    def predict_rna(self, X_TF, weights):
        return (X_TF @ self.A_base.T) + self._atom_term(X_TF, weights)

    @staticmethod
    def _prune_top_frac(W, frac):
        """Zero out everything but the top `frac` fraction (by |weight|) of each module's
        entries, independently per module. W: (D, G, T). Ported from cleanup; used only by
        production_from_lead below -- predict_rna and all GRN-inference losses always see the
        full, dense A_base/A_dict."""
        D = W.shape[0]
        with torch.no_grad():
            flat_abs = W.detach().abs().reshape(D, -1)
            n_nonzero = (flat_abs > 0).sum(dim=1)
            keep = torch.zeros_like(flat_abs, dtype=torch.bool)
            for d in range(D):
                nnz = int(n_nonzero[d].item())
                if nnz == 0: continue
                k = max(1, int(round(nnz * frac)))
                if k >= nnz:
                    keep[d] = flat_abs[d] > 0
                else:
                    thresh = torch.topk(flat_abs[d], k).values.min()
                    keep[d] = flat_abs[d] >= thresh
            keep = keep.reshape(W.shape)
        return W * keep

    def production_from_lead(self, tf_lead, weights, prune_frac=None):
        """Effective per-cell GRN applied to the ATAC->RNA TF lead, for trajectory inference.
        prune_frac (ported from cleanup): if set, propagates through a per-module top-`prune_frac`
        sparsified copy of A_base/A_dict (keep only the strongest `prune_frac` of edges in each
        module); None (default) propagates through the dense GRN, unchanged from before this port.
        GRN inference itself (predict_rna, and every loss shaping A_base/A_dict) is never pruned."""
        if prune_frac is None or prune_frac >= 1.0:
            return (tf_lead @ self.A_base.T) + self._atom_term(tf_lead, weights)
        A_base_pruned = self._prune_top_frac(self.A_base.unsqueeze(0), prune_frac)[0]
        A_dict_pruned = self._prune_top_frac(self.A_dict, prune_frac)
        P = torch.einsum('bt,dgt->dbg', tf_lead, A_dict_pruned)
        atom_term = (P * weights.T.unsqueeze(-1)).sum(0)
        return (tf_lead @ A_base_pruned.T) + atom_term

    def effective_A(self, weights_vec):
        return self.A_base + torch.einsum('d,dgt->gt', weights_vec, self.A_dict)

    def sparsity_loss(self):
        return torch.abs(self.A_dict).mean() + torch.abs(self.A_base).mean()

    def diversity_loss(self):
        A_flat = self.A_dict.reshape(self.D, -1); A_norm = F.normalize(A_flat, dim=1)
        eye = torch.eye(self.D, device=A_flat.device)
        param_div = ((A_norm @ A_norm.T) * (1 - eye)).abs().mean()
        pred = torch.einsum('dgt,t->dg', self.A_dict, self.mean_tf); pred = F.normalize(pred, dim=1)
        pred_div = ((pred @ pred.T) * (1 - eye)).abs().mean()
        return 0.5 * param_div + 0.5 * pred_div

    def tf_access_consistency_loss(self):
        closed = 1.0 - self.TF_access.clamp(0, 1)
        dict_term = (torch.abs(self.A_dict) * closed.unsqueeze(0).unsqueeze(0)).mean()
        base_term = (torch.abs(self.A_base) * closed).mean()
        return dict_term + base_term

    def gene_access_consistency_loss(self):
        """Same idea as tf_access_consistency_loss, but gates by the TARGET gene's own
        chromatin accessibility instead of the regulating TF's -- an independent term (not
        multiplied with the TF-side one), scaling whole rows rather than whole columns.
        Ported from cleanup."""
        closed = 1.0 - self.Gene_access.clamp(0, 1)
        dict_term = (torch.abs(self.A_dict) * closed.view(1, -1, 1)).mean()
        base_term = (torch.abs(self.A_base) * closed.unsqueeze(1)).mean()
        return dict_term + base_term
