"""data/load.py — single entrypoint for all data loading."""

import numpy as np
import scipy.io
import pandas as pd
from scipy.sparse import issparse, lil_matrix
from sklearn.preprocessing import normalize as sk_normalize


def as_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def load_all_data(config: dict) -> dict:
    """Load, normalize, select HVGs, build masks, return everything training needs."""
    import anndata as ad
    import scanpy as sc

    # ── A. Load RNA ───────────────────────────────────────────────────────────
    rna_adata = ad.read_h5ad(config["RNA_H5AD"])
    gene_names_all   = np.array(rna_adata.var_names.astype(str))
    gene_to_idx_full = {g: i for i, g in enumerate(gene_names_all)}

    # ── B. Normalize (total-count + log1p) ────────────────────────────────────
    adata_norm = rna_adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    X_full_norm = as_dense(adata_norm.X).astype(np.float32)  # (N, all_genes)

    # ── C. HVG selection on normalized data ───────────────────────────────────
    sc.pp.highly_variable_genes(
        adata_norm, n_top_genes=config["N_HVG"],
        flavor=config["HVG_FLAVOR"], subset=False,
    )
    hvg_set = set(gene_names_all[adata_norm.var["highly_variable"].values])
    print(f"HVGs selected: {len(hvg_set)}")

    # ── D. Build gene/TF universe ──────────────────────────────────────────────
    peak_tf        = scipy.io.mmread(config["PEAK_TF_MTX"]).tocsr()
    motif_tf_names = open(config["TF_NAMES_TXT"]).read().strip().split()
    motif_tf_to_idx = {t: i for i, t in enumerate(motif_tf_names)}

    peak_gene = pd.read_csv(config["PEAK_GENE_CSV"], sep="\t")
    peaks     = pd.read_csv(config["PEAKS_TXT"], header=None)[0].astype(str).tolist()
    peak_to_idx = {p: i for i, p in enumerate(peaks)}

    genes_in_peaks = set(peak_gene["gene"].astype(str).unique())

    # A_genes = HVG ∩ genes_in_peaks  (TFs excluded from HVG filter)
    A_genes = np.array(
        [g for g in gene_names_all if g in hvg_set and g in genes_in_peaks], dtype=str
    )
    # A_tfs = TFs present in RNA (no HVG filter — they are inputs, not targets)
    A_tfs = np.array([t for t in motif_tf_names if t in gene_to_idx_full], dtype=str)

    gene_idx = np.array([gene_to_idx_full[g] for g in A_genes], dtype=int)
    tf_idx   = np.array([gene_to_idx_full[t] for t in A_tfs],   dtype=int)
    sub_g2i  = {g: i for i, g in enumerate(A_genes)}

    X    = X_full_norm[:, gene_idx]  # (N, G) normalized
    X_TF = X_full_norm[:, tf_idx]    # (N, T) normalized

    # ── E. Build motif mask M (G x T) ─────────────────────────────────────────
    P = len(peaks)
    G = len(A_genes)
    pgm = lil_matrix((P, G), dtype=np.float32)
    for _, row in peak_gene.iterrows():
        p, g = str(row["peak"]), str(row["gene"])
        if p in peak_to_idx and g in sub_g2i:
            pgm[peak_to_idx[p], sub_g2i[g]] = 1.0
    pgm = pgm.tocsr()

    M_full = (pgm.T @ peak_tf)
    M_full = (M_full > 0).astype(np.float32)

    # Re-align TFs to those with motif entries
    A_tfs   = np.array([t for t in A_tfs if t in motif_tf_to_idx], dtype=str)
    tf_idx  = np.array([gene_to_idx_full[t] for t in A_tfs], dtype=int)
    tf_col  = np.array([motif_tf_to_idx[t]  for t in A_tfs], dtype=int)
    X_TF    = X_full_norm[:, tf_idx]
    M       = M_full[:, tf_col].toarray().astype(np.float32)
    assert M.shape == (len(A_genes), len(A_tfs))
    print(f"A_genes={len(A_genes)}, A_tfs={len(A_tfs)}, M={M.shape}")

    # ── F. ATAC gene activity ──────────────────────────────────────────────────
    atac_raw = scipy.io.mmread(config["ATAC_MTX"]).tocsr()
    atac_X   = atac_raw.T.astype(np.float32)  # (N_atac, peaks)

    peaks_atac_df  = pd.read_csv(config["ATAC_PEAKS_TXT"], sep="\t")
    peak_names_atac = (peaks_atac_df["Feature"]
                       .str.replace("_", ":", n=1)
                       .str.replace("_", "-", n=1).tolist())
    peak_to_idx_atac = {p: i for i, p in enumerate(peak_names_atac)}

    def fix_peak(p): return str(p).replace("-", ":", 1)

    P2  = len(peak_names_atac)
    pgm2 = lil_matrix((P2, G), dtype=np.float32)
    for _, row in peak_gene.iterrows():
        p, g = fix_peak(row["peak"]), str(row["gene"])
        if p in peak_to_idx_atac and g in sub_g2i:
            pgm2[peak_to_idx_atac[p], sub_g2i[g]] = 1.0
    pgm2 = pgm2.tocsr()

    gene_activity = as_dense(atac_X @ pgm2).astype(np.float32)
    gene_activity = sk_normalize(gene_activity, norm="l1", axis=1)
    gene_activity = np.log1p(gene_activity)

    gene_access = gene_activity.mean(axis=0)
    gene_access = (gene_access - gene_access.min()) / (gene_access.max() - gene_access.min() + 1e-8)

    # ── G. TF access prior ────────────────────────────────────────────────────
    T = len(A_tfs)
    tf_access = np.zeros(T, dtype=np.float32)
    for i, tf in enumerate(A_tfs):
        if tf in sub_g2i:
            tf_access[i] = gene_activity[:, sub_g2i[tf]].mean()
    lo, hi = tf_access.min(), tf_access.max()
    TF_access_prior = (tf_access - lo) / (hi - lo + 1e-8)

    # ── H. Decipher embeddings ────────────────────────────────────────────────
    dec = np.load(config["DECIPHER_NPZ"])
    Z_dec  = dec[config["DECIPHER_Z_KEY"]].astype(np.float32)  # (N, 10)
    V_dec2 = dec[config["DECIPHER_V_KEY"]].astype(np.float32)  # (N, 2)
    assert Z_dec.shape[0] == X.shape[0] == V_dec2.shape[0]

    return {
        "X": X, "X_TF": X_TF, "M": M,
        "gene_activity": gene_activity, "gene_access": gene_access,
        "TF_access_prior": TF_access_prior,
        "A_genes": A_genes, "A_tfs": A_tfs,
        "gene_idx": gene_idx, "tf_idx": tf_idx, "sub_g2i": sub_g2i,
        "Z_dec": Z_dec, "V_dec2": V_dec2,
        "rna_adata": rna_adata,
    }
