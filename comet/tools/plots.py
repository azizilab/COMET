"""
Post-training visualization functions for COMET.

All functions are stateless: they accept explicit array arguments rather than
reading global variables. Nothing here modifies the model or training state.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig_or_path, save_path):
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(loss_history, run_name="Run 0", save_path=None):
    """
    Up to 3-panel Phase 1/2 training loss with phase shading.

    loss_history : dict that always has "epoch" and "phase", plus whichever
    loss terms the training loop happened to log (e.g. "total", and
    optionally per-term entries under names like "mask"/"l_mask" or
    "diversity"/"l_div"/"l_diverse"). Panels for terms that aren't present
    are skipped rather than raising a KeyError.
    """
    epochs = loss_history["epoch"]
    phases = np.array(loss_history["phase"])

    p1_end = int(np.where(phases == 2)[0][0]) if 2 in phases else len(epochs)

    def _shade(ax):
        ax.axvspan(0,      p1_end,      alpha=0.05, color='blue',   label='Phase 1')
        ax.axvspan(p1_end, max(epochs), alpha=0.05, color='orange', label='Phase 2')
        ax.axvline(p1_end, color='gray', linestyle='--', linewidth=1)

    # Each panel lists the candidate keys the training loop might have used;
    # the first one found in loss_history is plotted.
    panel_specs = [
        (["total"],                              "black",     f"{run_name} — Total Loss"),
        (["mask", "l_mask"],                      "steelblue", "Masked Reconstruction Loss"),
        (["diversity", "l_div", "l_diverse"],     "green",     "Atom Diversity Loss"),
    ]
    panels = []
    for candidates, color, title in panel_specs:
        key = next((k for k in candidates if k in loss_history), None)
        if key is not None:
            panels.append((key, color, title))

    if not panels:
        print("plot_loss_curves: none of the expected loss keys were found in "
              f"loss_history (available keys: {list(loss_history.keys())})")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), dpi=120)
    axes = np.atleast_1d(axes)

    for ax, (key, color, title) in zip(axes, panels):
        ax.plot(epochs, loss_history[key], color=color, linewidth=2)
        _shade(ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8)
    plt.suptitle(f"Phase 1/2 Convergence — {run_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Model state inspection
# ─────────────────────────────────────────────────────────────────────────────

def plot_atom_weights(V_dec2, all_weights, D_ATOMS, origin_atom,
                      A_tfs=None, A_dict_np=None, save_path=None):
    """
    Per-atom mixing-weight scatter in Decipher space.

    V_dec2      : (N, 2)     — Decipher 2-D coordinates
    all_weights : (N, D)     — encoder mixing weights
    A_dict_np   : (D, G, T)  — optional; used to annotate top TF per atom
    A_tfs       : (T,) names — required if A_dict_np is provided
    """
    print("Atom usage stats (atoms should NOT all be equal):")
    print("  Mean mixing weights:", all_weights.mean(axis=0).round(3))
    print("  Std  mixing weights:", all_weights.std(axis=0).round(3))
    mean_ent = -(all_weights * np.log(all_weights + 1e-8)).sum(axis=1).mean()
    print(f"  Mean entropy: {mean_ent:.3f} / {np.log(D_ATOMS):.2f} max")

    n_rows = max(1, D_ATOMS // 2)
    n_cols = max(1, (D_ATOMS + n_rows - 1) // n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for d in range(D_ATOMS):
        ax = axes[d]
        sc = ax.scatter(V_dec2[:, 0], V_dec2[:, 1], c=all_weights[:, d],
                        cmap='YlOrRd', s=8, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, fraction=0.04)

        tf_label = ""
        if A_dict_np is not None and A_tfs is not None:
            top_tf   = A_tfs[np.abs(A_dict_np[d]).sum(axis=0).argmax()]
            tf_label = f"  (top TF: {top_tf})"

        marker = " ★ ORIGIN" if d == origin_atom else ""
        ax.set_title(f"Atom {d}{tf_label}{marker}", fontsize=9)
        ax.set_xlabel("Decipher 1")
        ax.set_ylabel("Decipher 2")

    for d in range(D_ATOMS, len(axes)):
        axes[d].set_visible(False)

    plt.suptitle("Per-cell GRN mixing weights\n(each atom should cover a distinct region)",
                 fontsize=11)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Velocity
# ─────────────────────────────────────────────────────────────────────────────

def plot_velocity_streamlines(X_2d, V_smooth, all_weights=None,
                               cell_types=None, facs_pseudotime=None,
                               space_label="Decipher", base_path=".", save=True):
    """
    Static streamline plot with Gaussian-weighted grid interpolation.

    Fixes applied relative to an earlier version:
    - integration_direction='forward'  (not 'both') — avoids backward-loop artifact
    - Adaptive sigma from local cell density — prevents over/under-smoothing
    - Speed normalised at 99th percentile — prevents one fast cell collapsing arrows

    Parameters
    ----------
    X_2d            : (N, 2) 2-D cell coordinates
    V_smooth        : (N, 2) diffusion-smoothed velocity vectors
    cell_types      : (N,) optional — if given, adds a second panel with cell-type colours
    facs_pseudotime : (N,) optional — colours first panel; falls back to gray if None
    """
    grid_n = 60
    x_min, x_max = X_2d[:, 0].min() - 0.2, X_2d[:, 0].max() + 0.2
    y_min, y_max = X_2d[:, 1].min() - 0.2, X_2d[:, 1].max() + 0.2
    Xg, Yg      = np.meshgrid(np.linspace(x_min, x_max, grid_n),
                               np.linspace(y_min, y_max, grid_n))
    grid_points  = np.c_[Xg.ravel(), Yg.ravel()]

    # Gaussian-weighted interpolation of V_smooth onto grid
    nbrs = NearestNeighbors(n_neighbors=30, n_jobs=-1).fit(X_2d)
    dists, idx_grid = nbrs.kneighbors(grid_points)
    sigma  = np.percentile(dists, 30, axis=1, keepdims=True) + 1e-8
    w      = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    w     /= w.sum(axis=1, keepdims=True) + 1e-8
    V_grid = (w[:, :, np.newaxis] * V_smooth[idx_grid]).sum(axis=1)

    # Support mask: only show arrows within 3× median nearest-neighbour distance
    nbrs1            = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_2d)
    dist_to_cells, _ = nbrs1.kneighbors(grid_points)
    cell_nn_dist     = (NearestNeighbors(n_neighbors=2).fit(X_2d)
                        .kneighbors(X_2d)[0][:, 1])
    support_radius   = 3.0 * np.median(cell_nn_dist)
    support_mask     = dist_to_cells.ravel() < support_radius

    U      = V_grid[:, 0].reshape(Xg.shape)
    W      = V_grid[:, 1].reshape(Yg.shape)
    active = support_mask.reshape(Xg.shape)
    U[~active] = 0.0
    W[~active] = 0.0
    U = gaussian_filter(U, sigma=1.0)
    W = gaussian_filter(W, sigma=1.0)

    speed     = np.sqrt(U ** 2 + W ** 2)
    speed_max = np.percentile(speed[active], 99) + 1e-8
    if speed_max < 2e-8:
        print("\nWARNING: V_smooth is effectively zero — no streamlines will appear.")
        print("  Likely cause: the velocity head has not learned a meaningful field yet.")
        print("  Check V_pred.npy — if all rows are near zero, inspect Phase 2 training.")
    U_plot    = U / speed_max
    W_plot    = W / speed_max

    n_panels = 2 if cell_types is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), dpi=150)
    if n_panels == 1:
        axes = [axes]

    # Panel 0: pseudotime or gray scatter + streamlines
    if facs_pseudotime is not None:
        pseudo_norm = ((facs_pseudotime - facs_pseudotime.min()) /
                       (facs_pseudotime.max() - facs_pseudotime.min() + 1e-8))
        sc = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=pseudo_norm, cmap='viridis',
                             s=18, alpha=0.7, edgecolors='none', rasterized=True, zorder=2)
        plt.colorbar(sc, ax=axes[0], fraction=0.04, label='Pseudotime')
    else:
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c='#888888',
                        s=12, alpha=0.5, edgecolors='none', rasterized=True, zorder=2)

    axes[0].streamplot(Xg, Yg, U_plot, W_plot, color='black', density=1.2,
                       linewidth=0.9, arrowsize=1.2, arrowstyle='-|>',
                       minlength=0.4, integration_direction='forward', zorder=3)
    axes[0].set_title(f'Streamline Velocity ({space_label} space)', fontweight='bold')
    axes[0].set_xlabel(f'{space_label} 1', fontweight='bold')
    axes[0].set_ylabel(f'{space_label} 2', fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Panel 1 (optional): cell-type colours + streamlines
    if cell_types is not None:
        palette = {'HSC': '#2166AC', 'MPP': '#4393C3', 'MLP': '#92C5DE',
                   'CMP': '#F4A582', 'MEP': '#D6604D', 'GMP': '#B2182B',
                   'CLOUD': '#CCCCCC'}
        for ct in ['CLOUD'] + [c for c in np.unique(cell_types) if c != 'CLOUD']:
            mask  = cell_types == ct
            color = palette.get(ct, '#999999')
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color,
                            s=8 if ct == 'CLOUD' else 22,
                            alpha=0.2 if ct == 'CLOUD' else 0.85,
                            label=ct, edgecolors='none', rasterized=True, zorder=2)
        axes[1].streamplot(Xg, Yg, U_plot, W_plot, color='black', density=1.2,
                           linewidth=0.9, arrowsize=1.2, arrowstyle='-|>',
                           minlength=0.4, integration_direction='forward', zorder=3)
        axes[1].set_title(f'Velocity + Cell Types ({space_label})', fontweight='bold')
        axes[1].set_xlabel(f'{space_label} 1', fontweight='bold')
        axes[1].set_ylabel(f'{space_label} 2', fontweight='bold')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                       frameon=False, fontsize=8, markerscale=1.5)

    plt.tight_layout()
    if save:
        out = os.path.join(base_path, f"velocity_streamlines_{space_label.lower()}.png")
        os.makedirs(base_path, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Pseudotime
# ─────────────────────────────────────────────────────────────────────────────

def plot_pseudotime(V_dec2, pseudotime, cell_types,
                    root_pct=0.05, facs_pseudotime=None, save_path=None):
    """
    Two-panel pseudotime plot.
      Left:  pseudotime gradient with root cells (★)
      Right: branch-coloured scatter with pseudotime overlay (non-CLOUD only)

    Prints Spearman ρ vs facs_pseudotime when provided.

    root_pct: fraction of lowest-pseudotime cells highlighted as roots — should
              match the root_pct passed to compute_comet_pseudotime.
    """
    pt_thresh = np.percentile(pseudotime, root_pct * 100)
    root_idx  = np.where(pseudotime <= pt_thresh)[0]
    n_roots   = root_idx.size

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

    sc_pt = axes[0].scatter(V_dec2[:, 0], V_dec2[:, 1], c=pseudotime, cmap='viridis',
                            s=12, alpha=0.8, edgecolors='none', rasterized=True)
    axes[0].scatter(V_dec2[root_idx, 0], V_dec2[root_idx, 1],
                    c='red', s=60, zorder=5, marker='*',
                    label=f'Root cells (n={n_roots}, lowest {root_pct*100:.0f}% pseudotime)')
    plt.colorbar(sc_pt, ax=axes[0], fraction=0.04, label='Velocity pseudotime (0=earliest)')
    axes[0].set_title('COMET Velocity Pseudotime\n(Dijkstra on velocity-aligned graph)',
                      fontweight='bold')
    axes[0].set_xlabel('Decipher 1', fontweight='bold')
    axes[0].set_ylabel('Decipher 2', fontweight='bold')
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    branch_palette = {
        'HSC': '#2166AC', 'MPP': '#4393C3', 'MEP': '#D6604D',
        'CMP': '#F4A582', 'GMP': '#B2182B', 'MLP': '#92C5DE',
        'CLOUD': '#DDDDDD',
    }
    for ct in ['CLOUD'] + [c for c in np.unique(cell_types) if c != 'CLOUD']:
        mask = cell_types == ct
        axes[1].scatter(V_dec2[mask, 0], V_dec2[mask, 1],
                        c=branch_palette.get(ct, '#999'),
                        s=6 if ct == 'CLOUD' else 18,
                        alpha=0.15 if ct == 'CLOUD' else 0.85,
                        label=ct, edgecolors='none', rasterized=True)
    non_cloud = cell_types != 'CLOUD'
    sc2 = axes[1].scatter(V_dec2[non_cloud, 0], V_dec2[non_cloud, 1],
                          c=pseudotime[non_cloud], cmap='plasma',
                          s=12, alpha=0.6, edgecolors='none', rasterized=True, zorder=3)
    plt.colorbar(sc2, ax=axes[1], fraction=0.04, label='Pseudotime')
    axes[1].set_title('Pseudotime on Annotated Cell Types', fontweight='bold')
    axes[1].set_xlabel('Decipher 1', fontweight='bold')
    axes[1].set_ylabel('Decipher 2', fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, fontsize=7)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

    if facs_pseudotime is not None:
        rho, p = spearmanr(pseudotime, facs_pseudotime)
        print(f"\nCorrelation with FACS pseudotime: Spearman ρ={rho:.3f}, p={p:.4f}")
        print("  ρ > 0.5 → pseudotime strongly recapitulates known ordering")
        print("  ρ > 0.3 → reasonable, especially for branching trajectories")
        print("  ρ < 0.2 → velocity field may have wrong direction or poor resolution")
        print("  ✓ Purple/dark cells should overlap HSC/MPP.")
        print("  ✓ Yellow/bright cells should overlap GMP, MEP, MLP.")


# ─────────────────────────────────────────────────────────────────────────────
# Marker expression
# ─────────────────────────────────────────────────────────────────────────────

def plot_marker_expression(rna_adata, V_dec2,
                            key_genes=None, save_path=None):
    """
    Key hematopoietic marker expression in Decipher space via scanpy.

    Requires scanpy. Sets rna_adata.obsm['X_decipher'] in place.
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("plot_marker_expression requires scanpy: pip install scanpy")

    if key_genes is None:
        key_genes = ['CD34', 'GATA1', 'MPO', 'SPI1', 'GATA2', 'LMO2', 'DNTT', 'IL7R']

    plot_genes = [g for g in key_genes if g in rna_adata.var_names]
    if not plot_genes:
        print("None of the requested marker genes found in rna_adata.var_names.")
        return

    rna_adata.obsm['X_decipher'] = V_dec2
    fig = sc.pl.embedding(rna_adata, basis='X_decipher', color=plot_genes,
                          cmap='Reds', frameon=False, ncols=4,
                          wspace=0.25, hspace=0.25, show=False, return_fig=True)
    plt.suptitle('Key Hematopoietic Marker Expression in Decipher Space',
                 fontsize=13, y=1.01, fontweight='bold')
    _save(fig, save_path)
    plt.show()
