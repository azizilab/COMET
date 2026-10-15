import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# LINEAR DECODER (LD-VAE style) — interpretable 2D embedding
# ═══════════════════════════════════════════════════════════════════════════

class LinearDecoder(nn.Module):
    def __init__(self, hidden: int, G: int, ld_dim: int = 2):
        super().__init__()
        self.ld_dim = ld_dim
        self.bottleneck = nn.Linear(hidden, ld_dim)
        self.W_dec = nn.Linear(ld_dim, G, bias=True)
        nn.init.normal_(self.W_dec.weight, std=0.01)
        nn.init.zeros_(self.W_dec.bias)

        # Orthogonal init: make the two projection directions start perpendicular
        nn.init.orthogonal_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_2d = self.bottleneck(h)
        x_hat = self.W_dec(z_2d)
        return z_2d, x_hat

    def reconstruction_loss(self, h: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        _, x_hat = self.forward(h)
        return F.mse_loss(x_hat, x_target)

    def ld_regularization_loss(self, h: torch.Tensor) -> torch.Tensor:
        """
        Two sub-losses that prevent dimension collapse:

        1. Variance loss: each dim should have std > 0 across the batch.
           We penalise low variance with -log(var + eps).
           If dim 2 is just a constant offset, its variance is ~0 → huge penalty.

        2. Orthogonality loss: the two dims should be uncorrelated.
           We compute the off-diagonal of the covariance matrix and penalise it.
           This forces dim 1 and dim 2 to capture *different* signals.
        """
        z_2d, _ = self.forward(h)             # (B, 2)
        z_centered = z_2d - z_2d.mean(dim=0)  # (B, 2) zero-mean per dim

        # Variance loss: penalise low variance per dimension
        var = (z_centered ** 2).mean(dim=0)    # (2,)
        var_loss = -torch.log(var + 1e-4).mean()   # scalar; pushes var up

        # Orthogonality loss: penalise correlation between dims
        # Covariance = z_centered.T @ z_centered / B → shape (2, 2)
        B = z_centered.shape[0]
        cov = (z_centered.T @ z_centered) / B     # (2, 2)
        # Off-diagonal elements = correlation between dims 0 and 1
        orth_loss = cov[0, 1].pow(2) + cov[1, 0].pow(2)   # scalar

        return var_loss + orth_loss
