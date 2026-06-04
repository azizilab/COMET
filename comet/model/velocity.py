import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9  –  Velocity Head (adapted for dictionary GRN)
# ═══════════════════════════════════════════════════════════════════════════
#
# Velocity should be a learned transformation of the DISEQUILIBRIUM
# signal (steady_state - rna).
#
# For the dictionary model, the steady state is per-cell:
#   steady_i = sum_d  w_{i,d} * A_d @ tf_i
# where w_i are the softmax mixing weights for cell i.
# The disequilibrium is thus also per-cell, which makes the velocity
# automatically cell-state-dependent without needing the ODE.
 

 
class DictVelocityHead(nn.Module):
    """
    Velocity head. Hidden reduced 512→256: same capacity for this signal,
    half the matmul cost. Encoder call removed from forward() — caller
    must pass pre-computed mix_weights or pass grn+encoder for steady state.
    """
    def __init__(self, G: int, hidden: int = 256):   # 512 → 256
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, G)
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, rna: torch.Tensor,
                disequil: torch.Tensor) -> torch.Tensor:
        """
        rna      : (B, G) — current expression (unused directly, kept for API compat)
        disequil : (B, G) — steady - rna, pre-computed by caller
        """
        return self.net(disequil) * self.scale   # (B, G)
    


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10  –  Velocity Self-Consistency Loss (restored from working model)
# ═══════════════════════════════════════════════════════════════════════════
#
# Math:
#   disequil = steady - rna
#   cos_loss = mean(1 - cosine_sim(v_pred, disequil))
#   mag_loss = mean(|v_pred| - |disequil|)
#   loss = cos_loss + 0.1 * mag_loss
 
def velocity_consistency_loss(velocity, rna, steady_state):
    disequil = steady_state - rna
    
    # 1. Cosine alignment: velocity should point toward steady state
    v_norm = F.normalize(velocity, dim=1)
    d_norm = F.normalize(disequil, dim=1)
    cosine_sim = (v_norm * d_norm).sum(dim=1)
    
    # 2. Magnitude guard: don't let velocity collapse to zero
    mag_loss = torch.abs(velocity.norm(dim=1) - disequil.norm(dim=1).detach()).mean()
    
    return (1.0 - cosine_sim).mean() + 0.1 * mag_loss
 
 
def atac_direction_loss(velocity, gene_access_batch):
    closed = 1.0 - gene_access_batch.to(velocity.device)
    return (closed * torch.relu(velocity)).mean()
 
 
def atac_direction_loss_asymmetric(velocity, gene_access_batch):
    closed = 1.0 - gene_access_batch.to(velocity.device)
    growth_penalty  = torch.relu(velocity)  * closed * 2.0
    shrink_penalty  = torch.relu(-velocity) * (1.0 - closed) * 0.5
    return (growth_penalty + shrink_penalty).mean()
 
 
def velocity_smoothness_loss(velocity, batch_indices, neighbor_lookup):
    """
    For each cell in the batch, look up its precomputed neighbors.
    If any neighbor is also in the batch, penalise velocity disagreement.

    Avoids the O(B * k * B) boolean mask — uses index lookup instead.
    """
    # Map global cell indices to their position in the current batch
    B = velocity.shape[0]
    device = velocity.device

    # Build reverse lookup: global_idx -> batch_position (or -1 if absent)
    batch_indices_cpu = batch_indices.cpu()
    max_idx = int(batch_indices_cpu.max().item()) + 1
    pos_in_batch = torch.full((max_idx,), -1, dtype=torch.long)
    for local_i, global_i in enumerate(batch_indices_cpu.tolist()):
        pos_in_batch[global_i] = local_i

    # For each cell, grab its first neighbor that is also in the batch
    neighbor_ids = neighbor_lookup[batch_indices]   # (B, k)
    neighbor_ids_flat = neighbor_ids.reshape(-1).cpu().clamp(0, max_idx - 1)
    local_positions = pos_in_batch[neighbor_ids_flat].reshape(B, -1)  # (B, k)

    # Find the first valid (in-batch) neighbor per cell
    valid_mask = local_positions >= 0   # (B, k)
    has_any = valid_mask.any(dim=1)     # (B,)

    if not has_any.any():
        return velocity.pow(2).mean() * 1e-4

    # Take first valid neighbor for cells that have one
    first_valid = valid_mask.float().argmax(dim=1)   # (B,) — index into k
    neighbor_local = local_positions[torch.arange(B), first_valid]  # (B,)

    src = velocity[has_any]
    tgt = velocity[neighbor_local[has_any]]
    return F.mse_loss(src, tgt)
 
 
def density_gradient_loss(velocity, batch_idx, V_dec2):
    v_mag = velocity.norm(dim=1)
    return torch.relu(0.1 - v_mag).mean()
 
 
def divergence_loss(velocity, batch_idx, rna_b, neighbor_indices_t, X_t):
    neigh_idx    = neighbor_indices_t[batch_idx]
    X_neighbors  = X_t[neigh_idx]
    displacements = X_neighbors - rna_b.unsqueeze(1)
    centroid_disp = displacements.mean(dim=1)
    v_unit = F.normalize(velocity,       dim=1)
    c_unit = F.normalize(centroid_disp,  dim=1)
    sink_score = (v_unit * c_unit).sum(dim=-1)
    return torch.relu(sink_score).mean()
 