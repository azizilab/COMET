"""
GRN interpretation, comparison, and reproducibility tools for COMET.

Covers:
  - Top gene targets per TF
  - Sparsity report
  - CellOracle GRN comparison (Spearman, Jaccard)
  - Inter-run GRN reproducibility
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# GRN inspection
# ─────────────────────────────────────────────────────────────────────────────

def top_tf_targets(A_learned, A_tfs, A_genes, tf_name, n=20):
    """
    Return the top-n target genes for a given TF from the learned GRN,
    ranked by absolute weight magnitude.

    Parameters
    ----------
    A_learned : (G, T) learned GRN weight matrix
    A_tfs     : (T,) TF name array
    A_genes   : (G,) gene name array
    tf_name   : str — must be in A_tfs
    n         : number of targets to return

    Returns
    -------
    list of (gene_name, weight) tuples, sorted descending by |weight|
    """
    if tf_name not in A_tfs:
        raise ValueError(f"{tf_name!r} not in A_tfs")
    t_idx = np.where(A_tfs == tf_name)[0][0]
    col   = A_learned[:, t_idx]
    top   = np.argsort(-np.abs(col))[:n]
    return [(A_genes[i], float(col[i])) for i in top]


def grn_sparsity_report(A_learned, A_genes, A_tfs, M,
                         thresholds=(0.001, 0.005, 0.01, 0.05)):
    """
    Print GRN edge-count statistics at multiple |w| thresholds.

    Parameters
    ----------
    A_learned  : (G, T) learned GRN weights
    A_genes    : (G,) gene names
    A_tfs      : (T,) TF names
    M          : (G, T) binary motif mask
    thresholds : weight thresholds for reporting

    Returns
    -------
    per_tf_active : (T,) int array — active targets per TF at |w| > 0.01
    """
    G, T            = A_learned.shape
    motif_supported = int(M.sum())
    total_possible  = G * T

    print("=" * 60)
    print("GRN SPARSITY REPORT")
    print("=" * 60)
    print(f"Matrix: {G} genes × {T} TFs = {total_possible:,} entries")
    print(f"Motif-supported edges: {motif_supported:,}  "
          f"({100 * motif_supported / total_possible:.1f}%)")

    rows = []
    for thr in thresholds:
        n_active = int((np.abs(A_learned) > thr).sum())
        rows.append({
            "|w| >":              thr,
            "# edges":            n_active,
            "% motif-supported":  f"{100 * n_active / motif_supported:.1f}%",
            "% all possible":     f"{100 * n_active / total_possible:.2f}%",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    per_tf_active = (np.abs(A_learned) > 0.01).sum(axis=0)
    print(f"\nPer-TF active targets (|w|>0.01): "
          f"median={np.median(per_tf_active):.0f}, "
          f"mean={np.mean(per_tf_active):.1f}")
    return per_tf_active


# ─────────────────────────────────────────────────────────────────────────────
# CellOracle comparison
# ─────────────────────────────────────────────────────────────────────────────

def _col_rank_normalize(A):
    """Column-wise rank normalization: each column mapped to [0, 1] by rank."""
    G, T  = A.shape
    A_out = np.zeros_like(A, dtype=np.float32)
    for t in range(T):
        r = A[:, t].argsort().argsort().astype(np.float32)
        A_out[:, t] = r / (G - 1 + 1e-8)
    return A_out


def compare_to_celloracle(A_learned, A_genes, A_tfs,
                           co_A, co_genes, co_tfs,
                           save_path=None):
    """
    Compare the learned GRN against a CellOracle reference network.

    Metrics:
      - Global Spearman ρ on nonzero-union entries (rank-normalised per TF column)
      - Per-TF Spearman ρ (KDE + top-10 bar plot)
      - Top-k Jaccard at 5%, 10%, 20% of genes per TF

    Parameters
    ----------
    A_learned : (G, T)  learned GRN weights
    A_genes   : (G,)    gene names aligned to A_learned
    A_tfs     : (T,)    TF names aligned to A_learned
    co_A      : (G_co, T_co) or (T_co, G_co) CellOracle GRN
    co_genes  : (G_co,) gene names aligned to co_A
    co_tfs    : (T_co,) TF names aligned to co_A
    save_path : optional path for the comparison figure

    Returns
    -------
    tf_df        : DataFrame with per-TF Spearman ρ, sorted descending
    A_co_shared  : (G_shared, T_shared) CellOracle submatrix on common genes/TFs
    common_genes : list of shared gene names
    common_tfs   : list of shared TF names
    """
    # Orient co_A as (genes, TFs)
    if co_A.shape == (len(co_tfs), len(co_genes)):
        co_A = co_A.T
    elif co_A.shape != (len(co_genes), len(co_tfs)):
        raise ValueError(
            f"Unexpected CellOracle shape {co_A.shape}; "
            f"expected ({len(co_genes)}, {len(co_tfs)}) or ({len(co_tfs)}, {len(co_genes)})"
        )

    common_genes = sorted(set(A_genes) & set(co_genes))
    common_tfs   = sorted(set(A_tfs)   & set(co_tfs))
    print(f"Shared: {len(common_genes)} genes, {len(common_tfs)} TFs")

    pred_g2i = {g: i for i, g in enumerate(A_genes)}
    pred_t2i = {t: i for i, t in enumerate(A_tfs)}
    co_g2i   = {g: i for i, g in enumerate(co_genes)}
    co_t2i   = {t: i for i, t in enumerate(co_tfs)}

    A_pred_shared = A_learned[np.ix_(
        [pred_g2i[g] for g in common_genes],
        [pred_t2i[t] for t in common_tfs],
    )]
    A_co_shared = co_A[np.ix_(
        [co_g2i[g] for g in common_genes],
        [co_t2i[t] for t in common_tfs],
    )]

    A_pred_ranked = _col_rank_normalize(np.abs(A_pred_shared))
    A_co_ranked   = _col_rank_normalize(np.abs(A_co_shared))

    # Global Spearman on the nonzero-union mask
    nz_mask  = (np.abs(A_pred_shared) > 0) | (np.abs(A_co_shared) > 0)
    rho_g, _ = spearmanr(A_pred_ranked[nz_mask], A_co_ranked[nz_mask])
    print(f"Global Spearman (nonzero support): ρ={rho_g:.4f}")

    # Per-TF Spearman
    per_tf_rho   = []
    per_tf_names = []
    for ti, tf_name in enumerate(common_tfs):
        cp, cc = A_pred_ranked[:, ti], A_co_ranked[:, ti]
        if cp.std() < 1e-8 or cc.std() < 1e-8:
            continue
        rho, _ = spearmanr(cp, cc)
        per_tf_rho.append(rho)
        per_tf_names.append(tf_name)
    per_tf_rho = np.array(per_tf_rho)
    print(f"Per-TF Spearman: mean={per_tf_rho.mean():.4f} ± {per_tf_rho.std():.4f}, "
          f"frac positive={np.mean(per_tf_rho > 0):.3f}")

    # Top-k Jaccard at multiple fractions
    for frac in [0.05, 0.10, 0.20]:
        jacs = []
        for ti in range(len(common_tfs)):
            k  = max(1, int(np.ceil(frac * len(common_genes))))
            tp = set(np.argpartition(np.abs(A_pred_shared[:, ti]), -k)[-k:].tolist())
            tc = set(np.argpartition(np.abs(A_co_shared[:, ti]),   -k)[-k:].tolist())
            jacs.append(len(tp & tc) / (len(tp | tc) + 1e-8))
        print(f"Top {int(frac*100)}% Jaccard: {np.mean(jacs):.4f}")

    tf_df = (pd.DataFrame({"tf": per_tf_names, "per_tf_spearman": per_tf_rho})
               .sort_values("per_tf_spearman", ascending=False)
               .reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    sns.kdeplot(tf_df['per_tf_spearman'], fill=True, color='skyblue', ax=axes[0], lw=2)
    axes[0].axvline(tf_df['per_tf_spearman'].mean(), color='red', linestyle='--',
                    label=f'Mean: {tf_df["per_tf_spearman"].mean():.2f}')
    axes[0].set_xlabel("Per-TF Spearman ρ vs CellOracle")
    axes[0].set_title("Conservation of Regulatory Rankings", fontweight='bold')
    axes[0].legend(frameon=False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    top10 = tf_df.head(10)
    sns.barplot(data=top10, x='per_tf_spearman', y='tf', palette='viridis', ax=axes[1])
    axes[1].set_title("Top Recapitulated TFs", fontweight='bold')
    axes[1].set_xlabel("ρ")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return tf_df, A_co_shared, common_genes, common_tfs


# ─────────────────────────────────────────────────────────────────────────────
# Inter-run reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _grn_signature(A, A_genes_local, A_tfs_local, common_genes, common_tfs):
    """
    Extract a rank-normalised signature vector from one run's GRN,
    restricted to the shared gene/TF subset.
    """
    g2i   = {g: i for i, g in enumerate(A_genes_local)}
    t2i   = {t: i for i, t in enumerate(A_tfs_local)}
    g_idx = [g2i[g] for g in common_genes if g in g2i]
    t_idx = [t2i[t] for t in common_tfs   if t in t2i]
    sub   = np.abs(A[np.ix_(g_idx, t_idx)]).ravel().astype(np.float64)
    ranks = sub.argsort().argsort().astype(np.float64)
    return ranks / (len(ranks) - 1 + 1e-8)


def plot_grn_consistency(run_matrices, A_co_shared, common_genes, common_tfs,
                          save_path=None):
    """
    Inter-run GRN consistency heatmap + each-run vs CellOracle scatter.

    Parameters
    ----------
    run_matrices : list of (A_learned, A_genes, A_tfs) tuples, one per run
    A_co_shared  : (G_shared, T_shared) CellOracle matrix on the shared gene/TF subset
                   (the second return value of compare_to_celloracle)
    common_genes : list of shared gene names (third return value of compare_to_celloracle)
    common_tfs   : list of shared TF names  (fourth return value of compare_to_celloracle)
    save_path    : optional output path
    """
    N    = len(run_matrices)
    sigs = [_grn_signature(A, ag, at, common_genes, common_tfs)
            for A, ag, at in run_matrices]

    co_sig = np.abs(A_co_shared).ravel().astype(np.float64)
    co_sig = co_sig.argsort().argsort() / (len(co_sig) - 1 + 1e-8)

    corr_mat = np.array([[spearmanr(sigs[i], sigs[j])[0] for j in range(N)]
                          for i in range(N)])
    co_corrs = [float(spearmanr(s, co_sig)[0]) for s in sigs]

    fig = plt.figure(figsize=(12, 5), dpi=150)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.35)

    ax0    = fig.add_subplot(gs[0])
    labels = [f"Run {i+1}" for i in range(N)]
    sns.heatmap(corr_mat, annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", vmin=0, vmax=1, linewidths=0.5, ax=ax0,
                mask=np.eye(N, dtype=bool), annot_kws={"size": 9})
    ax0.set_title("Inter-run GRN Consistency (Spearman)", fontsize=11)

    ax1    = fig.add_subplot(gs[1])
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=N)
    ax1.scatter(np.ones(N) + jitter, co_corrs, color='#1f77b4', s=50)
    ax1.axhline(float(np.mean(co_corrs)), color='#d62728', ls='--', lw=1.5,
                label=f'mean={float(np.mean(co_corrs)):.3f}')
    ax1.set_xlim(0.5, 1.8)
    ax1.set_xticks([])
    ax1.set_ylabel("Spearman ρ vs CellOracle")
    ax1.set_title("Each run vs CellOracle")
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.suptitle("GRN Reproducibility & CellOracle Alignment", fontsize=13, y=1.02)
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
