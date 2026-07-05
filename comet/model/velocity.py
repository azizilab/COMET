"""model/velocity.py — velocity head and associated loss functions.

Ground truth: model_5_30_26_rerun.md notebook.
No psi, no Phase 3, no MarkovProgressionField.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


# ── Velocity head ─────────────────────────────────────────────────────────────

class DictVelocityHead(nn.Module):
    """
    Maps disequilibrium (steady_state - rna) -> gene-space velocity.
    Caller pre-computes disequil; this module does NOT call the encoder.
    """
    def __init__(self, G: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, G),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, rna: torch.Tensor, disequil: torch.Tensor) -> torch.Tensor:
        """rna unused directly; kept for API compatibility."""
        return self.net(disequil) * self.scale  # (B, G)


# ── Velocity loss functions ───────────────────────────────────────────────────

def velocity_consistency_loss(velocity: torch.Tensor,
                               rna: torch.Tensor,
                               steady_state: torch.Tensor) -> torch.Tensor:
    """Cosine alignment toward steady state + magnitude guard."""
    disequil = steady_state - rna
    v_norm = F.normalize(velocity, dim=1)
    d_norm = F.normalize(disequil, dim=1)
    cos_loss = (1.0 - (v_norm * d_norm).sum(dim=1)).mean()
    mag_loss = torch.abs(velocity.norm(dim=1) - disequil.norm(dim=1).detach()).mean()
    return cos_loss + 0.1 * mag_loss


def atac_direction_loss(velocity: torch.Tensor,
                         gene_access_batch: torch.Tensor) -> torch.Tensor:
    closed = 1.0 - gene_access_batch.to(velocity.device)
    return (closed * torch.relu(velocity)).mean()


def atac_direction_loss_asymmetric(velocity: torch.Tensor,
                                    gene_access_batch: torch.Tensor) -> torch.Tensor:
    closed = 1.0 - gene_access_batch.to(velocity.device)
    return (torch.relu(velocity) * closed * 2.0 +
            torch.relu(-velocity) * (1.0 - closed) * 0.5).mean()


def velocity_smoothness_loss(velocity: torch.Tensor,
                              batch_indices: torch.Tensor,
                              neighbor_lookup: torch.Tensor) -> torch.Tensor:
    B = velocity.shape[0]
    batch_indices_cpu = batch_indices.cpu()
    max_idx = int(batch_indices_cpu.max().item()) + 1
    pos_in_batch = torch.full((max_idx,), -1, dtype=torch.long)
    for local_i, global_i in enumerate(batch_indices_cpu.tolist()):
        pos_in_batch[global_i] = local_i
    neighbor_ids = neighbor_lookup[batch_indices]
    neighbor_ids_flat = neighbor_ids.reshape(-1).cpu().clamp(0, max_idx - 1)
    local_positions = pos_in_batch[neighbor_ids_flat].reshape(B, -1)
    valid_mask = local_positions >= 0
    has_any = valid_mask.any(dim=1)
    if not has_any.any():
        return velocity.pow(2).mean() * 1e-4
    first_valid = valid_mask.float().argmax(dim=1)
    neighbor_local = local_positions[torch.arange(B), first_valid]
    src = velocity[has_any]
    tgt = velocity[neighbor_local[has_any]]
    return F.mse_loss(src, tgt)


def density_gradient_loss(velocity: torch.Tensor, *args) -> torch.Tensor:
    return torch.relu(0.1 - velocity.norm(dim=1)).mean()


def divergence_loss(velocity: torch.Tensor,
                     batch_idx: torch.Tensor,
                     rna_b: torch.Tensor,
                     neighbor_indices_t: torch.Tensor,
                     X_t: torch.Tensor) -> torch.Tensor:
    neigh_idx    = neighbor_indices_t[batch_idx]
    X_neighbors  = X_t[neigh_idx]
    centroid_disp = (X_neighbors - rna_b.unsqueeze(1)).mean(dim=1)
    v_unit = F.normalize(velocity, dim=1)
    c_unit = F.normalize(centroid_disp, dim=1)
    return torch.relu((v_unit * c_unit).sum(dim=-1)).mean()


# ── 2D projection utility (used post-training for visualization) ──────────────

def project_to_2d(V_pred_full: np.ndarray,
                   X_2d_vis: np.ndarray,
                   X_full: np.ndarray,
                   k: int = 30,
                   scale: float = 10.0,
                   adaptive_scale: bool = True,
                   density_k: int = 15) -> np.ndarray:
    """Project gene-space velocity to 2D embedding space (vectorized)."""
    N = V_pred_full.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_2d_vis)
    _, idx = nbrs.kneighbors(X_2d_vis)

    if adaptive_scale:
        nbrs_d = NearestNeighbors(n_neighbors=density_k).fit(X_2d_vis)
        dist_d, _ = nbrs_d.kneighbors(X_2d_vis)
        mean_nn_dist = dist_d[:, 1:].mean(axis=1)
        density_factor = 1.0 / (mean_nn_dist + 1e-8)
        lo, hi = density_factor.min(), density_factor.max()
        density_factor = 1.0 + 2.0 * (density_factor - lo) / (hi - lo + 1e-8)
    else:
        density_factor = np.ones(N, dtype=np.float32)

    d_high = X_full[idx] - X_full[:, np.newaxis, :]
    d_low  = X_2d_vis[idx] - X_2d_vis[:, np.newaxis, :]
    v_norm = np.linalg.norm(V_pred_full, axis=1, keepdims=True) + 1e-8
    d_norm = np.linalg.norm(d_high, axis=2) + 1e-8
    dots   = np.einsum('ng,nkg->nk', V_pred_full, d_high)
    sims   = dots / (d_norm * v_norm)
    sims_centered = sims - sims.mean(axis=1, keepdims=True)
    eff_scale = (scale * density_factor)[:, np.newaxis]
    weights = np.exp(sims_centered * eff_scale)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights[np.linalg.norm(V_pred_full, axis=1) < 1e-8] = 0.0
    return np.einsum('nk,nkd->nd', weights, d_low).astype(np.float32)
