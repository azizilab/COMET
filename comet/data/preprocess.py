import numpy as np
import scipy
from scipy.sparse import lil_matrix


def as_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# ─────────────────────────────────────────────────────────────
# 1. Gene + TF universe
# ─────────────────────────────────────────────────────────────

def build_gene_universe(rna_adata, tf_names):
    gene_names = np.array(rna_adata.var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    A_tfs = np.array([t for t in tf_names if t in gene_to_idx], dtype=str)

    return gene_names, gene_to_idx, A_tfs


def subset_rna_and_tfs(rna_adata, gene_to_idx, A_genes, A_tfs):
    X_full = as_dense(rna_adata.X).astype(np.float32)

    gene_idx = np.array([gene_to_idx[g] for g in A_genes], dtype=int)
    tf_idx = np.array([gene_to_idx[t] for t in A_tfs], dtype=int)

    X = X_full[:, gene_idx]
    X_TF = X_full[:, tf_idx]

    sub_g2i = {g: i for i, g in enumerate(A_genes)}

    return X, X_TF, gene_idx, tf_idx, sub_g2i


# ─────────────────────────────────────────────────────────────
# 2. Motif mask M (gene x TF)
# ─────────────────────────────────────────────────────────────

def build_motif_mask(peak_gene_df, peaks, peak_tf, A_genes, A_tfs):
    peak_to_idx = {p: i for i, p in enumerate(peaks)}
    sub_g2i = {g: i for i, g in enumerate(A_genes)}
    tf_to_idx = {t: i for i, t in enumerate(A_tfs)}

    P = len(peaks)
    G = len(A_genes)

    peak_gene_mat = lil_matrix((P, G), dtype=np.float32)

    for _, row in peak_gene_df.iterrows():
        p = str(row["peak"])
        g = str(row["gene"])
        if p in peak_to_idx and g in sub_g2i:
            peak_gene_mat[peak_to_idx[p], sub_g2i[g]] = 1.0

    peak_gene_mat = peak_gene_mat.tocsr()
    M_full = (peak_gene_mat.T @ peak_tf).toarray()
    M_full = (M_full > 0).astype(np.float32)

    # align TF columns
    A_tfs = np.array([t for t in A_tfs if t in tf_to_idx], dtype=str)
    tf_col = np.array([tf_to_idx[t] for t in A_tfs], dtype=int)

    M = M_full[:, tf_col]

    assert M.shape == (len(A_genes), len(A_tfs))

    return M, A_tfs