"""
Velten 2017 FACS gate cell-type annotation for hematopoiesis data.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def assign_velten_celltypes(rna_adata, V_dec2, X_full, gene_to_idx_full,
                             save_path=None, plot_histograms=False):
    """
    Assign cell types using Velten 2017 FACS gate logic with rank-normalised
    thresholds, followed by marker-gene rescue for remaining CLOUD cells.

    Gate logic (rank-normalised to [0, 1] within the dataset):
      lin−  cd34+  cd38lo  cd90+  cd45ra−  →  HSC
      lin−  cd34+  cd38lo  cd90−  cd45ra−  →  MPP
      lin−  cd34+  cd38lo            cd45ra+  →  MLP
      lin−  cd34+  cd38hi  cd135+  cd45ra−  →  CMP
      lin−  cd34+  cd38hi  cd135−  cd45ra−  →  MEP
      lin−  cd34+  cd38hi            cd45ra+  →  GMP
      anything else                           →  CLOUD (rescued by marker score)

    Parameters
    ----------
    rna_adata       : AnnData with .obs containing facs_* columns
    V_dec2          : (N, 2) Decipher vis-space coordinates for plotting
    X_full          : (N, all_genes) raw expression matrix
    gene_to_idx_full: {gene_name: column_index} for X_full
    save_path       : optional path for the cell-type scatter figure
    plot_histograms : if True, show per-marker rank distribution histograms

    Returns
    -------
    cell_types : (N,) object array of string labels
    """
    obs = rna_adata.obs.copy()
    N   = len(obs)

    # ── Marker-gene z-score for rescue ────────────────────────────────────────
    MARKER_GENES = {
        'HSC': ['CD34', 'THY1', 'PROM1', 'ITGA6'],
        'MPP': ['CD34', 'FLT3', 'GATA2'],
        'MLP': ['CD34', 'IL7R', 'FLT3', 'DNTT'],
        'CMP': ['CD34', 'FLT3', 'CSF1R', 'CEBPA'],
        'MEP': ['GATA1', 'GATA2', 'KLF1', 'EPOR', 'GYPA'],
        'GMP': ['MPO', 'ELANE', 'SPI1', 'LYZ', 'CEBPA'],
    }
    scores = {}
    for ct, markers in MARKER_GENES.items():
        s = np.zeros(N, dtype=np.float32)
        n_found = 0
        for gene in markers:
            if gene in gene_to_idx_full:
                expr   = X_full[:, gene_to_idx_full[gene]].astype(np.float32)
                expr_z = ((np.log1p(expr) - np.log1p(expr).mean()) /
                          (np.log1p(expr).std() + 1e-8))
                s       += expr_z
                n_found += 1
        scores[ct] = s / n_found if n_found > 0 else s
    score_df = pd.DataFrame(scores)

    # ── Rank-normalised FACS gates ────────────────────────────────────────────
    def _rank(col):
        if col not in obs.columns:
            return np.zeros(N, dtype=np.float32)
        return rankdata(obs[col].values, method="average").astype(np.float32) / N

    lin_neg = (obs['facs_lin'].values < np.quantile(obs['facs_lin'], 0.2)
               if 'facs_lin' in obs.columns else np.ones(N, dtype=bool))

    cd34_r   = _rank('facs_cd34')
    cd38_r   = _rank('facs_cd38')
    cd90_r   = _rank('facs_cd90')
    cd135_r  = _rank('facs_cd135')
    cd45ra_r = _rank('facs_cd45ra')

    cd34_pos   = cd34_r   > 0.80
    cd38_lo    = cd38_r   < 0.30
    cd38_hi    = cd38_r   > 0.70
    cd90_pos   = cd90_r   > 0.80
    cd135_pos  = cd135_r  > 0.70
    cd135_neg  = cd135_r  < 0.30
    cd45ra_pos = cd45ra_r > 0.70
    cd45ra_neg = cd45ra_r < 0.30

    cell_types = np.full(N, 'CLOUD', dtype=object)
    base = lin_neg & cd34_pos
    cell_types[base & cd38_hi & cd45ra_pos]             = 'GMP'
    cell_types[base & cd38_hi & cd135_neg & cd45ra_neg] = 'MEP'
    cell_types[base & cd38_hi & cd135_pos & cd45ra_neg] = 'CMP'
    cell_types[base & cd38_lo & cd45ra_pos]             = 'MLP'
    cell_types[base & cd38_lo & ~cd90_pos & cd45ra_neg] = 'MPP'
    cell_types[base & cd38_lo &  cd90_pos & cd45ra_neg] = 'HSC'

    # ── Marker rescue for CLOUD cells ─────────────────────────────────────────
    SCORE_THRESHOLD = 0.5
    cloud_idx  = np.where(cell_types == 'CLOUD')[0]
    best_ct    = score_df.values[cloud_idx].argmax(axis=1)
    best_score = score_df.values[cloud_idx].max(axis=1)
    ct_names   = list(score_df.columns)
    for ci, bct, bsc in zip(cloud_idx, best_ct, best_score):
        if bsc > SCORE_THRESHOLD:
            cell_types[ci] = ct_names[bct]

    print("Cell type counts (FACS gates + marker rescue):")
    unique, counts = np.unique(cell_types, return_counts=True)
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {u:<8}: {c}")

    # ── Plot cell type scatter (+ pseudotime if available) ────────────────────
    palette   = {'HSC': '#2166AC', 'MPP': '#4393C3', 'MLP': '#92C5DE',
                 'CMP': '#F4A582', 'MEP': '#D6604D', 'GMP': '#B2182B',
                 'CLOUD': '#666565'}
    has_pt    = 'facs_pseudotime' in obs.columns
    n_panels  = 2 if has_pt else 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(13 if n_panels == 2 else 6, 5.5), dpi=130)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    for ct in ['CLOUD', 'MPP', 'MLP', 'CMP', 'MEP', 'GMP', 'HSC']:
        mask = cell_types == ct
        if not mask.any():
            continue
        ax.scatter(V_dec2[mask, 0], V_dec2[mask, 1],
                   c=palette.get(ct, '#999'), s=8 if ct == 'CLOUD' else 25,
                   alpha=0.2 if ct == 'CLOUD' else 0.9,
                   label=f'{ct} (n={mask.sum()})', edgecolors='none', rasterized=True)
    ax.set_xlabel('Decipher 1', fontweight='bold')
    ax.set_ylabel('Decipher 2', fontweight='bold')
    ax.set_title('Velten 2017 FACS Gate Cell Types', fontweight='bold', pad=10)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if has_pt:
        sc_p = axes[1].scatter(V_dec2[:, 0], V_dec2[:, 1],
                               c=obs['facs_pseudotime'].values,
                               cmap='viridis', s=12, alpha=0.7,
                               edgecolors='none', rasterized=True)
        plt.colorbar(sc_p, ax=axes[1], label='FACS pseudotime', fraction=0.04)
        axes[1].set_xlabel('Decipher 1', fontweight='bold')
        axes[1].set_ylabel('Decipher 2', fontweight='bold')
        axes[1].set_title('FACS Pseudotime (Velten 2017)', fontweight='bold', pad=10)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # ── Optional per-marker rank histograms ───────────────────────────────────
    if plot_histograms:
        for marker in ['facs_cd34', 'facs_cd38', 'facs_cd90', 'facs_cd45ra', 'facs_cd135']:
            if marker in obs.columns:
                plt.figure()
                plt.hist(obs[marker], bins=100)
                plt.title(marker)
                plt.show()

    return cell_types
