"""
Post-training biological analysis functions for COMET.

Covers:
  - Transition-zone analysis (GRN entropy vs velocity magnitude)
  - Per-lineage trajectory gene analysis with Velten 2017 ground-truth recovery
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# Velten 2017 ground-truth marker sets (used by analyze_trajectory_genes)
# ─────────────────────────────────────────────────────────────────────────────

_GT_MARKERS = {
    'MEP': {
        'up':   {'GATA1', 'GATA2', 'KLF1', 'EPOR', 'GYPA', 'HBB', 'AHSP', 'BLVRB'},
        'down': {'CD34', 'HLF', 'HOXA9', 'MEIS1', 'SPINK2'},
    },
    'GMP': {
        'up':   {'MPO', 'ELANE', 'SPI1', 'LYZ', 'CEBPA', 'PRTN3', 'AZU1', 'CST7'},
        'down': {'CD34', 'HLF', 'HOXA9', 'MEIS1', 'SPINK2'},
    },
    'MLP': {
        'up':   {'IL7R', 'DNTT', 'FLT3', 'RAG1', 'EBF1', 'CD79A', 'VPREB1', 'IGLL1'},
        'down': {'CD34', 'HLF', 'HOXA9', 'MEIS1', 'SPINK2'},
    },
    'CMP': {'up': {'CSF1R', 'CEBPA', 'FLT3', 'CD34', 'MPO'}, 'down': set()},
    'HSC': {'up': {'CD34', 'THY1', 'HOXA9', 'MEIS1', 'GATA2', 'HLF'},  'down': set()},
    'MPP': {'up': {'CD34', 'FLT3', 'GATA2', 'MPL'},                      'down': set()},
}


# ─────────────────────────────────────────────────────────────────────────────
# Transition zone analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_transition_zones(all_weights, V_smooth, V_dec2, D_ATOMS, save_path=None):
    """
    Verify that transition zones (high GRN entropy) drive the velocity field.

    With the ReLU kernel T[i→j] ∝ ReLU(ψ(j)−ψ(i)), committed cells at ψ-maxima
    have V=0 by construction, so Spearman(entropy, |V|) should be strongly positive.

    Parameters
    ----------
    all_weights : (N, D) encoder mixing weights
    V_smooth    : (N, 2) diffusion-smoothed velocity vectors
    V_dec2      : (N, 2) Decipher 2-D coordinates for plotting
    D_ATOMS     : number of GRN atoms (used only for entropy normalisation)
    save_path   : optional output path

    Returns
    -------
    entropy : (N,) per-cell GRN mixture entropy
    """
    eps          = 1e-8
    entropy      = -(all_weights * np.log(all_weights + eps)).sum(axis=1)
    entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + eps)

    v_mag      = np.linalg.norm(V_smooth, axis=1)
    v_mag_norm = (v_mag - v_mag.min()) / (v_mag.max() - v_mag.min() + eps)

    rho, p = spearmanr(entropy, v_mag)
    print(f"Spearman(GRN entropy, |V_smooth|): rho={rho:.3f}, p={p:.4f}")
    print("(Positive rho = transition cells have higher velocity — expected ✓)")
    print("(Note: with ReLU kernel, committed cells have V=0, "
          "so rho should be strongly positive)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=130)

    sc0 = axes[0].scatter(V_dec2[:, 0], V_dec2[:, 1], c=entropy_norm,
                          cmap='hot', s=10, alpha=0.7, rasterized=True)
    plt.colorbar(sc0, ax=axes[0], fraction=0.04, label='GRN mixing entropy')
    axes[0].set_title('GRN Mixture Entropy\n(bright = transition zone)', fontweight='bold')

    sc1 = axes[1].scatter(V_dec2[:, 0], V_dec2[:, 1], c=v_mag_norm,
                          cmap='Blues', s=10, alpha=0.7, rasterized=True)
    plt.colorbar(sc1, ax=axes[1], fraction=0.04, label='Velocity magnitude')
    axes[1].set_title('Velocity Magnitude\n(bright = actively transitioning)',
                      fontweight='bold')

    axes[2].scatter(entropy, v_mag, s=6, alpha=0.4, c='#2166ac', rasterized=True)
    axes[2].set_xlabel('GRN Mixture Entropy')
    axes[2].set_ylabel('|V_smooth|')
    axes[2].set_title(f'Entropy vs. Velocity Magnitude\nSpearman ρ={rho:.3f}',
                      fontweight='bold')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    for ax in axes[:2]:
        ax.set_xlabel('Decipher 1', fontweight='bold')
        ax.set_ylabel('Decipher 2', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Transition Zones: Where Velocity Originates', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return entropy


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory gene analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_trajectory_genes(X, V_dec2, cell_types, A_genes, vel_pseudotime,
                               n_top=25, save_path=None):
    """
    Per-lineage pseudotime-correlated gene analysis with ground-truth recovery.

    For each non-CLOUD lineage, computes Spearman ρ(gene, pseudotime) and
    generates a 4-panel figure:
      Panel 1: Pseudotime heatmap (cells × top-20 increasing + top-5 decreasing genes)
      Panel 2: Smoothed expression curves for top-5 increasing genes
      Panel 3: Spatial expression of the top increasing gene in Decipher space
      Panel 4: Fraction of known Velten 2017 markers recovered at top-N thresholds

    Parameters
    ----------
    X              : (N, G) expression matrix (subset to active genes)
    V_dec2         : (N, 2) Decipher 2-D coordinates
    cell_types     : (N,) string labels
    A_genes        : (G,) gene names aligned to X columns
    vel_pseudotime : (N,) inferred pseudotime (from infer_velocity_pseudotime)
    n_top          : number of top increasing genes to return
    save_path      : base path for saving; each lineage appended as _<lineage>.png

    Returns
    -------
    results : dict {lineage: {'up': [(gene, rho), ...], 'down': [(gene, rho), ...]}}
    """
    lineages = [ct for ct in np.unique(cell_types) if ct != 'CLOUD']
    print(f"Analyzing trajectories: {lineages}")
    print("\nHOW TO INTERPRET:")
    print("  Panel 1 (heatmap): left=early, right=late. Top rows should brighten left→right.")
    print("  Panel 2 (curves):  lineage markers should rise, stem genes fall.")
    print("  Panel 3 (spatial): top gene should be bright in the lineage region.")
    print("  Panel 4 (GT bar):  fraction of top-N genes in known Velten 2017 markers.")

    results = {}
    for lineage in lineages:
        print(f"\n── {lineage} {'─'*45}")
        lin_mask = cell_types == lineage
        if lin_mask.sum() < 10:
            print(f"  Too few cells ({lin_mask.sum()}) — skipping.")
            continue

        X_lin    = X[lin_mask]
        pt_lin   = vel_pseudotime[lin_mask]
        sort_idx = np.argsort(pt_lin)
        X_sorted = X_lin[sort_idx]

        corrs = np.array([
            spearmanr(pt_lin, X_lin[:, g])[0] if X_lin[:, g].std() > 1e-8 else 0.0
            for g in range(X_lin.shape[1])
        ])
        corrs = np.nan_to_num(corrs, nan=0.0)

        up_idx   = np.argsort(-corrs)[:n_top]
        down_idx = np.argsort(corrs)[:5]

        gt_up   = _GT_MARKERS.get(lineage, {}).get('up',   set())
        gt_down = _GT_MARKERS.get(lineage, {}).get('down', set())

        print("  Top genes INCREASING with pseudotime (should include lineage markers):")
        found_up = []
        for gi in up_idx[:10]:
            mark = "✓ KNOWN MARKER" if A_genes[gi] in gt_up else ""
            print(f"    {A_genes[gi]:<15}  ρ={corrs[gi]:.3f}  {mark}")
            if A_genes[gi] in gt_up:
                found_up.append(A_genes[gi])
        print("  Top genes DECREASING (stem genes leaving):")
        for gi in down_idx:
            mark = "✓" if A_genes[gi] in gt_down else ""
            print(f"    {A_genes[gi]:<15}  ρ={corrs[gi]:.3f}  {mark}")
        print(f"  ✓ Known markers recovered in top-10: {found_up}")

        results[lineage] = {
            'up':   [(A_genes[gi], float(corrs[gi])) for gi in up_idx],
            'down': [(A_genes[gi], float(corrs[gi])) for gi in down_idx],
        }

        fig, axes_row = plt.subplots(1, 4, figsize=(20, 5), dpi=120)

        # Panel 1: pseudotime heatmap
        top_genes_hm = np.concatenate([up_idx[:15], down_idx[:5]])
        heat_data    = X_sorted[:, top_genes_hm].T
        # np.ndarray.ptp was removed in NumPy 2.0; use explicit max-min
        heat_range   = (heat_data.max(axis=1, keepdims=True) -
                        heat_data.min(axis=1, keepdims=True))
        heat_norm    = (heat_data - heat_data.min(axis=1, keepdims=True)) / (heat_range + 1e-8)
        im = axes_row[0].imshow(heat_norm, aspect='auto', cmap='RdYlBu_r',
                                 interpolation='nearest', vmin=0, vmax=1)
        axes_row[0].set_yticks(range(len(top_genes_hm)))
        axes_row[0].set_yticklabels([A_genes[gi] for gi in top_genes_hm], fontsize=6)
        axes_row[0].set_xlabel('Pseudotime →', fontsize=8)
        axes_row[0].set_title(f'{lineage}: Pseudotime Heatmap', fontweight='bold', fontsize=9)
        plt.colorbar(im, ax=axes_row[0], fraction=0.04, label='Norm. expr.')

        # Panel 2: smoothed expression curves
        n_bins      = 50
        bins        = np.linspace(pt_lin.min(), pt_lin.max() + 1e-8, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_ids     = np.digitize(pt_lin, bins)
        for gi in up_idx[:5]:
            bin_means = [
                X_lin[bin_ids == b, gi].mean() if (bin_ids == b).any() else np.nan
                for b in range(1, n_bins + 1)
            ]
            axes_row[1].plot(bin_centers, bin_means, lw=1.5, alpha=0.8,
                             label=A_genes[gi])
        axes_row[1].set_xlabel('Pseudotime', fontsize=8)
        axes_row[1].set_ylabel('Mean expression', fontsize=8)
        axes_row[1].set_title(f'{lineage}: Top Increasing Genes', fontweight='bold', fontsize=9)
        axes_row[1].legend(fontsize=6, frameon=False)
        axes_row[1].spines['top'].set_visible(False)
        axes_row[1].spines['right'].set_visible(False)

        # Panel 3: spatial expression of top gene
        top_gene_idx = up_idx[0]
        sc3 = axes_row[2].scatter(V_dec2[:, 0], V_dec2[:, 1],
                                   c=X[:, top_gene_idx], cmap='Reds',
                                   s=8, alpha=0.7, rasterized=True)
        axes_row[2].scatter(V_dec2[lin_mask, 0], V_dec2[lin_mask, 1],
                            edgecolors='blue', facecolors='none', s=20, lw=0.5,
                            label=f'{lineage} cells', rasterized=True)
        plt.colorbar(sc3, ax=axes_row[2], fraction=0.04, label='Expression')
        axes_row[2].set_title(f'{A_genes[top_gene_idx]} spatial expression',
                              fontweight='bold', fontsize=9)
        axes_row[2].set_xlabel('Decipher 1', fontsize=8)
        axes_row[2].set_ylabel('Decipher 2', fontsize=8)
        axes_row[2].legend(fontsize=6, frameon=False)
        axes_row[2].spines['top'].set_visible(False)
        axes_row[2].spines['right'].set_visible(False)

        # Panel 4: ground-truth recovery bar
        top_n_vals = [5, 10, 15, 20, 25]
        hit_rates  = []
        for tn in top_n_vals:
            up_hits   = sum(1 for g, _ in results[lineage]['up'][:tn]  if g in gt_up)
            down_hits = sum(1 for g, _ in results[lineage]['down'][:5]  if g in gt_down)
            total_gt  = len(gt_up) + len(gt_down)
            hit_rates.append((up_hits + down_hits) / (total_gt + 1e-8))
        axes_row[3].bar(top_n_vals, hit_rates, color='#4393C3', edgecolor='white', width=3)
        axes_row[3].axhline(0.2, color='green', linestyle='--', lw=1.5,
                             label='Target (>20% = good)')
        axes_row[3].set_xlabel('Top-N genes considered', fontsize=8)
        axes_row[3].set_ylabel('Fraction known markers\nrecovered', fontsize=8)
        axes_row[3].set_title(f'{lineage}: ground-truth recovery\n(Velten 2017)',
                              fontweight='bold', fontsize=9)
        axes_row[3].set_ylim(0, min(1.0, max(hit_rates) * 1.4 + 0.05))
        axes_row[3].legend(fontsize=7, frameon=False)
        axes_row[3].spines['top'].set_visible(False)
        axes_row[3].spines['right'].set_visible(False)
        print(f"  Ground-truth hit rate (top-25): {hit_rates[-1]:.1%}")
        if found_up:
            axes_row[3].text(0.5, 0.95, f"Found: {', '.join(found_up[:4])}",
                             transform=axes_row[3].transAxes, ha='center', va='top',
                             fontsize=6, color='darkgreen')

        plt.suptitle(f'{lineage} Trajectory Gene Analysis', fontsize=12,
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            out = f"{base}_{lineage}{ext or '.png'}"
            parent = os.path.dirname(out)
            if parent:
                os.makedirs(parent, exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.show()

    print("\n" + "=" * 60)
    print("GROUND TRUTH SUMMARY (Velten 2017)")
    print("=" * 60)
    print("HSC → MEP:  GATA1, GATA2, KLF1, EPOR, GYPA, HBB, AHSP")
    print("HSC → GMP:  MPO, ELANE, SPI1, LYZ, CEBPA, PRTN3")
    print("HSC → MLP:  IL7R, DNTT, FLT3, RAG1, EBF1, CD79A")
    print("HSC stem (decreasing): CD34, HLF, HOXA9, GATA2, MEIS1")
    print("\nFor each lineage, check the heatmap (Panel 1) and curves (Panel 2).")
    print("Known markers should appear near the TOP of the heatmap and show rising curves.")

    return results
