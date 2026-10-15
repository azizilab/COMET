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
        multiplied with the TF-side one), scaling whole rows rather than whole columns."""
        closed = 1.0 - self.Gene_access.clamp(0, 1)
        dict_term = (torch.abs(self.A_dict) * closed.view(1, -1, 1)).mean()
        base_term = (torch.abs(self.A_base) * closed.unsqueeze(1)).mean()
        return dict_term + base_term
