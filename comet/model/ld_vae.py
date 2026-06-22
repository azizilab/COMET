import numpy as np

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

    def get_gene_loadings(self) -> np.ndarray:
        # W_dec.weight is (G, ld_dim); return as (G, 2)
        return self.W_dec.weight.detach().cpu().numpy()

def plot_ld_embedding(
    z_2d_rna: np.ndarray,         # (C_rna, 2)
    pseudo_norm: np.ndarray,       # (C_rna,) pseudotime 0–1
    gene_loadings: np.ndarray,     # (G, 2)
    A_genes: np.ndarray,           # (G,) gene names
    z_2d_atac: np.ndarray = None,  # (C_atac, 2) — optional
    highlight_genes: list = None,
    n_arrow_genes: int = 15,       # how many gene arrows to show
    save_path: str = None,
):
    """
    3-panel LD-VAE biplot:
      Panel A: RNA cells colored by pseudotime (+ ATAC cells in grey if provided)
      Panel B: Same but colored by modality (RNA vs ATAC)
      Panel C: Gene loading biplot — arrows show which direction each gene 'pulls'
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    has_atac = z_2d_atac is not None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # ── Panel A: pseudotime ───────────────────────────────────────────────
    ax = axes[0]
    if has_atac:
        ax.scatter(z_2d_atac[:, 0], z_2d_atac[:, 1],
                   c='lightgray', s=6, alpha=0.4, edgecolors='none',
                   label='ATAC', rasterized=True, zorder=1)
    sc = ax.scatter(z_2d_rna[:, 0], z_2d_rna[:, 1],
                    c=pseudo_norm, cmap='viridis',
                    s=10, alpha=0.7, edgecolors='none', rasterized=True, zorder=2)
    plt.colorbar(sc, ax=ax, fraction=0.04, label='Pseudotime')
    ax.set_title('LD-VAE Embedding\n(RNA colored by pseudotime)', fontsize=10)
    ax.set_xlabel('LD Dim 1'); ax.set_ylabel('LD Dim 2')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if has_atac:
        ax.legend(fontsize=7, frameon=False)

    # ── Panel B: modality split ───────────────────────────────────────────
    ax = axes[1]
    ax.scatter(z_2d_rna[:, 0], z_2d_rna[:, 1],
               c='#2166ac', s=8, alpha=0.6, edgecolors='none',
               label=f'RNA (n={len(z_2d_rna)})', rasterized=True)
    if has_atac:
        ax.scatter(z_2d_atac[:, 0], z_2d_atac[:, 1],
                   c='#d73027', s=8, alpha=0.6, edgecolors='none',
                   label=f'ATAC (n={len(z_2d_atac)})', rasterized=True)
    ax.set_title('LD-VAE Embedding\n(by modality)', fontsize=10)
    ax.set_xlabel('LD Dim 1'); ax.set_ylabel('LD Dim 2')
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── Panel C: gene biplot ──────────────────────────────────────────────
    ax = axes[2]
    ax.scatter(z_2d_rna[:, 0], z_2d_rna[:, 1],
               c='lightgray', s=4, alpha=0.3, edgecolors='none', rasterized=True)

    # Pick genes to annotate: top n by loading magnitude
    magnitudes = np.linalg.norm(gene_loadings, axis=1)
    if highlight_genes is not None:
        # Show highlighted genes + fill remaining slots by magnitude
        hi_idx = [i for i, g in enumerate(A_genes) if g in highlight_genes]
        other_top = [i for i in np.argsort(-magnitudes) if i not in hi_idx]
        show_idx = hi_idx + other_top[: max(0, n_arrow_genes - len(hi_idx))]
    else:
        show_idx = np.argsort(-magnitudes)[:n_arrow_genes]

    # Scale arrows to data range
    x_range = z_2d_rna[:, 0].max() - z_2d_rna[:, 0].min()
    y_range = z_2d_rna[:, 1].max() - z_2d_rna[:, 1].min()
    scale = 0.35 * min(x_range, y_range) / (magnitudes[show_idx].max() + 1e-8)

    for idx in show_idx:
        dx, dy = gene_loadings[idx, 0] * scale, gene_loadings[idx, 1] * scale
        color = '#d73027' if (highlight_genes and A_genes[idx] in highlight_genes) else '#2166ac'
        ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        ax.text(dx * 1.08, dy * 1.08, A_genes[idx],
                fontsize=7, color=color, ha='center', va='center')

    ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_title('Gene Loadings Biplot\n(arrows = regulatory direction)', fontsize=10)
    ax.set_xlabel('LD Dim 1'); ax.set_ylabel('LD Dim 2')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"LD-VAE biplot saved: {save_path}" if save_path else "")