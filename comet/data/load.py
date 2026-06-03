import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import torch

from scipy.sparse import lil_matrix, issparse
from sklearn.preprocessing import normalize


def load_datasets(config):

    return {
        "rna_adata": ad.read_h5ad(config["rna"]),
        "peak_tf": scipy.io.mmread(config["peak_tf"]).tocsr(),
        "motif_tf_names": open(config["tf_names"]).read().strip().split(),
        "peak_gene": pd.read_csv(config["peak_gene"], sep="\t"),
        "peaks": pd.read_csv(config["peaks"], header=None)[0].astype(str).tolist(),
        "atac_X": scipy.io.mmread(config["atac"]).tocsr().T.astype(np.float32),  # cells x peaks
        "peaks_atac": pd.read_csv(config["atac_peaks"], sep="\t"),
    }



def build_universe(rna_adata, motif_tf_names, peak_gene, peaks, peaks_atac):

    gene_names_all = np.array(rna_adata.var_names.astype(str))
    gene_to_idx_full = {g: i for i, g in enumerate(gene_names_all)}
    X_full = as_dense(rna_adata.X).astype(np.float32)

    motif_tf_to_idx = {t: i for i, t in enumerate(motif_tf_names)}

    peak_to_idx = {p: i for i, p in enumerate(peaks)}

    genes_in_peaks = set(peak_gene["gene"].astype(str).unique())
    A_genes = np.array([g for g in gene_names_all if g in genes_in_peaks], dtype=str)

    # IMPORTANT: TFs must exist as genes in the RNA matrix for X_TF
    A_tfs = np.array([t for t in motif_tf_names if t in gene_to_idx_full], dtype=str)

    print(f"A_genes: {len(A_genes)}, A_tfs: {len(A_tfs)}")

    gene_idx = np.array([gene_to_idx_full[g] for g in A_genes], dtype=int)
    tf_idx = np.array([gene_to_idx_full[t] for t in A_tfs], dtype=int)

    X = X_full[:, gene_idx]      # cells x genes
    X_TF = X_full[:, tf_idx]     # cells x TFs
    sub_g2i = {g: i for i, g in enumerate(A_genes)}

    peak_names = peaks_atac["Feature"].str.replace("_", ":", n=1).str.replace("_", "-", n=1).tolist()
    peak_to_idx_atac = {p: i for i, p in enumerate(peak_names)}

    return {
        # --- core identity sets ---
        "A_genes": A_genes,
        "A_tfs": A_tfs,

        # --- full reference space ---
        "gene_names_all": gene_names_all,
        "gene_to_idx_full": gene_to_idx_full,
        "motif_tf_to_idx": motif_tf_to_idx,
        "peak_to_idx": peak_to_idx,
        "peak_names": peak_names,
        "peak_to_idx_atac": peak_to_idx_atac,

        # --- local indexing ---
        "sub_g2i": sub_g2i,

        # -- original X ---
        "X_full": X_full,

        # --- aligned data ---
        "X": X,
        "X_TF": X_TF,

        # --- index arrays (VERY useful later) ---
        "gene_idx": gene_idx,
        "tf_idx": tf_idx
    }

def build_M(peaks, peak_gene, peak_tf, X_full, A_genes, A_tfs, peak_to_idx, sub_g2i, gene_to_idx_full, motif_tf_to_idx):
    P = len(peaks)
    G = len(A_genes)

    peak_gene_mat = lil_matrix((P, G), dtype=np.float32)
    
    for _, row in peak_gene.iterrows():
        p = str(row["peak"])
        g = str(row["gene"])
        if p in peak_to_idx and g in sub_g2i:
            peak_gene_mat[peak_to_idx[p], sub_g2i[g]] = 1.0
    
    peak_gene_mat = peak_gene_mat.tocsr()

    M_full = (peak_gene_mat.T @ peak_tf)
    M_full = (M_full > 0).astype(np.float32)

    # align M columns to the TFs actually present in RNA
    A_tfs = np.array([t for t in A_tfs if t in motif_tf_to_idx], dtype=str)
    tf_idx = np.array([gene_to_idx_full[t] for t in A_tfs], dtype=int)
    tf_col = np.array([motif_tf_to_idx[t] for t in A_tfs], dtype=int)

    X_TF = X_full[:, tf_idx]  # refresh after final TF filtering
    M = M_full[:, tf_col].toarray().astype(np.float32)

    assert M.shape == (len(A_genes), len(A_tfs))

    return {
        "M": M,
        "X_TF": X_TF,
        "A_tfs": A_tfs,
        "tf_idx": tf_idx,
        "tf_col": tf_col
    }


def build_gene_priors(peak_names, A_genes, peak_gene, peak_to_idx_atac, X, atac_X, sub_g2i):

    P2 = len(peak_names)
    G2 = len(A_genes)

    peak_gene_mat_atac = lil_matrix((P2, G2), dtype=np.float32)
    for _, row in peak_gene.iterrows():
        p = str(row["peak"]).replace("-", ":", 1)
        g = str(row["gene"])
        if p in peak_to_idx_atac and g in sub_g2i:
            peak_gene_mat_atac[peak_to_idx_atac[p], sub_g2i[g]] = 1.0
    peak_gene_mat_atac = peak_gene_mat_atac.tocsr()

    gene_activity = atac_X @ peak_gene_mat_atac   # cells x genes
    gene_activity = normalize(gene_activity, norm="l1", axis=1)
    gene_activity = as_dense(gene_activity).astype(np.float32)
    gene_activity = np.log1p(gene_activity)

    cell_idx = np.arange(X.shape[0])
    #atac_X = as_dense(atac_X)[cell_idx]
    gene_activity = gene_activity[cell_idx]

    gene_access = gene_activity.mean(axis=0)
    gene_access = (gene_access - gene_access.min()) / (gene_access.max() - gene_access.min() + 1e-8)
    gene_access_t = torch.tensor(gene_access, dtype=torch.float32)

    return{
        "gene_activity": gene_activity,
        "gene_access": gene_access,
        "gene_access_t": gene_access_t,
    }

def build_tf_access_prior(gene_activity: np.ndarray,
                           A_tfs: np.ndarray,
                           gene_to_idx_full: dict,
                           sub_g2i) -> np.ndarray:
    """
    Returns TF_access: (T,) float32 — population-mean chromatin accessibility
    for each TF locus, derived from the ATAC gene-activity matrix.
    """
    T = len(A_tfs)
    tf_access = np.zeros(T, dtype=np.float32)
    for i, tf in enumerate(A_tfs):
        if tf in gene_to_idx_full:
            g = gene_to_idx_full[tf]          # index in full gene list
            # find column in gene_activity that corresponds to this gene
            # gene_activity columns align with A_genes (from Step 2/3)
            # We need the sub-index; if tf is in A_genes use sub_g2i
            # otherwise fall back to 0
            if tf in sub_g2i:
                col = sub_g2i[tf]
                tf_access[i] = float(gene_activity[:, col].mean())
    # min-max normalise to [0, 1]
    lo, hi = tf_access.min(), tf_access.max()
    tf_access = (tf_access - lo) / (hi - lo + 1e-8)
    return tf_access



def as_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)    
