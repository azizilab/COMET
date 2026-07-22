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
 

 
class DictVelocityHead1(nn.Module):
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
    

class DictVelocityHead(nn.Module):
    """Reverted to the pre-session MLP head. The notebook's diagonal
    alternative (per-gene log_rate + scalar) has almost no free parameters of
    its own, so it can't absorb the velocity/ATAC-direction objectives'
    gradient the way this MLP can — that pressure gets pushed back through
    disequil into the shared GRN weights instead, which is what caused the
    runaway loss on this dataset (confirmed by isolating it: swapping only
    this head back to the MLP, with every other change kept, restores a
    normal decreasing loss trajectory). See git history for the diagonal
    version if revisiting this tradeoff later.
    """
    def __init__(self, G, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, G),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, rna, disequil):
        return self.net(disequil) * self.scale



# ═══════════════════════════════════════════════════════════════════════════
# STEP 10  –  Velocity Self-Consistency Loss (restored from working model)
# ═══════════════════════════════════════════════════════════════════════════
#
# Math:
#   disequil = steady - rna
#   cos_loss = mean(1 - cosine_sim(v_pred, disequil))
#   mag_loss = mean(|v_pred| - |disequil|)
#   loss = cos_loss + 0.1 * mag_loss
 
def velocity_consistency_loss(velocity, disequil):
    """Align displacement with the chromatin-led disequilibrium; guard magnitude."""
    v = F.normalize(velocity, dim=1)
    d = F.normalize(disequil, dim=1)
    cos = (v * d).sum(1)
    mag = torch.abs(velocity.norm(dim=1) - disequil.norm(dim=1).detach()).mean()
    return (1.0 - cos).mean() + 0.1 * mag
 
 
def atac_direction_loss(velocity, gene_access_batch):
    closed = 1.0 - gene_access_batch.to(velocity.device)
    return (closed * torch.relu(velocity)).mean()
 
 
def atac_direction_loss_asymmetric(velocity, gene_access_batch):
    closed = 1.0 - gene_access_batch.to(velocity.device)
    growth_penalty  = torch.relu(velocity)  * closed * 2.0
    shrink_penalty  = torch.relu(-velocity) * (1.0 - closed) * 0.5
    return (growth_penalty + shrink_penalty).mean()
 
 
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
 