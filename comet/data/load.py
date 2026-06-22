import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import torch

from scipy.sparse import lil_matrix, issparse
from sklearn.preprocessing import normalize


def load_datasets(config):

    return {
        "rna_adata": ad.read_h5ad(config["rna"]), # cells x genes, w/ metadata
        "peak_tf": scipy.io.mmread(config["peak_tf"]).tocsr(), # peaks x TFs
        "motif_tf_names": open(config["tf_names"]).read().strip().split(), # ordered list of TF names corresponding to columns of peak_tf
        "peak_gene": pd.read_csv(config["peak_gene"], sep="\t"), # table with columns: peak, gene
        "peaks": pd.read_csv(config["peaks"], header=None)[0].astype(str).tolist(), # master list of peak identifiers
        "atac_X": scipy.io.mmread(config["atac"]).tocsr().T.astype(np.float32),  # cells x peaks 
        "peaks_atac": pd.read_csv(config["atac_peaks"], sep="\t"),
        "decipher_outputs": np.load(config["decipher_outputs"]),
    }


def build_universe(
    rna_adata: ad.AnnData,       
    motif_tf_names: list[str],    
    peak_gene: pd.DataFrame,      
    peaks: list[str],             
    peaks_atac: pd.DataFrame     
) -> dict:
    
    """
    Build active gene/TF universe from scRNA-seq and scATAC-seq inputs.

    A_genes: genes with at least one peak-gene regulatory link
    A_tfs:   TFs present in both the motif database and the RNA matrix
            (TFs without RNA expression are excluded — X_TF requires this)

    peaks_atac["Feature"] is expected in chr_start_end format and is
    converted to chr:start-end to match peak_gene coordinates.
    """

    # RNA reference space
    gene_names_all = np.array(rna_adata.var_names.astype(str))
    gene_to_idx_full = {g: i for i, g in enumerate(gene_names_all)}
    X_full = as_dense(rna_adata.X).astype(np.float32)

    # TF and peak lookup dicts
    motif_tf_to_idx = {t: i for i, t in enumerate(motif_tf_names)}
    peak_to_idx = {p: i for i, p in enumerate(peaks)}

    # filter genes/TFs to get active "universe"
    genes_in_peaks = set(peak_gene["gene"].astype(str).unique())
    A_genes = np.array([g for g in gene_names_all if g in genes_in_peaks], dtype=str)
    A_tfs = np.array([t for t in motif_tf_names if t in gene_to_idx_full], dtype=str)
    #print(f"A_genes: {len(A_genes)}, A_tfs: {len(A_tfs)}")

    # index arrays
    gene_idx = np.array([gene_to_idx_full[g] for g in A_genes], dtype=int)
    tf_idx = np.array([gene_to_idx_full[t] for t in A_tfs], dtype=int)

    X = X_full[:, gene_idx]      # cells x genes
    X_TF = X_full[:, tf_idx]     # cells x TFs
    sub_g2i = {g: i for i, g in enumerate(A_genes)}

    # coordinate format conversion for ATAC peaks (chr_start_end -> chr:start-end)
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

        # --- index arrays ---
        "gene_idx": gene_idx,
        "tf_idx": tf_idx
    }


def build_M(
    peaks: list[str],
    peak_gene: pd.DataFrame,
    peak_tf,                   
    X_full: np.ndarray,       
    A_genes: np.ndarray,       
    A_tfs: np.ndarray,         
    peak_to_idx: dict,         
    sub_g2i: dict,             
    gene_to_idx_full: dict,    
    motif_tf_to_idx: dict,     
) -> dict:
    
    """
    Build M, the binary gene x TF regulatory prior matrix.

    M[g, t] = 1 if TF t has at least one motif hit in any peak linked to gene g.

    Returns:
        M        (G x T) ndarray   — binary gene-TF regulatory prior
        X_TF     (C x T) ndarray   — TF expression, aligned to M columns
        A_tfs    (T,) ndarray      — final TF names after filtering to motif database
        tf_idx   (T,) ndarray      — TF positions in X_full / RNA space
        tf_col   (T,) ndarray      — TF positions in peak_tf / motif space
    """
    
    P = len(peaks)
    G = len(A_genes)

    peak_gene_mat = lil_matrix((P, G), dtype=np.float32)
    for _, row in peak_gene.iterrows():
        p = str(row["peak"])
        g = str(row["gene"])
        if p in peak_to_idx and g in sub_g2i:
            peak_gene_mat[peak_to_idx[p], sub_g2i[g]] = 1.0
    peak_gene_mat = peak_gene_mat.tocsr()

    M_full = (peak_gene_mat.T @ peak_tf) # (G × P) @ (P × T) -> (G × T)
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


def build_gene_priors(
    peak_names: list[str],         
    A_genes: np.ndarray,           
    peak_gene: pd.DataFrame,      
    peak_to_idx_atac: dict,        
    X: np.ndarray,                 
    atac_X,                       
    sub_g2i: dict, 
) -> dict:
    
    """
    Compute per-gene chromatin accessibility priors from scATAC-seq data.

    Constructs a peak x gene map from the peak_gene table and multiplies it with
    the ATAC matrix to get per-cell gene activity scores. These are L1-normalized
    per cell, log1p stabilized, then subsetted to the first C_rna cells (assumes
    RNA and ATAC are pre-aligned in the same cell order).

    Population-level prior gene_access is derived by averaging gene_activity across
    cells and min-max normalizing to [0, 1].

    Returns:
        gene_activity  (C_rna x G) ndarray   — per-cell gene accessibility scores
        gene_access    (G,) ndarray          — population-mean accessibility, normalized
        gene_access_t  (G,) tensor           — gene_access as a torch tensor
    """

    P2 = len(peak_names)
    G2 = len(A_genes)

    peak_gene_mat_atac = lil_matrix((P2, G2), dtype=np.float32)
    for _, row in peak_gene.iterrows():
        p = str(row["peak"]).replace("-", ":", 1)
        g = str(row["gene"])
        if p in peak_to_idx_atac and g in sub_g2i:
            peak_gene_mat_atac[peak_to_idx_atac[p], sub_g2i[g]] = 1.0
    peak_gene_mat_atac = peak_gene_mat_atac.tocsr()

    gene_activity = atac_X @ peak_gene_mat_atac   # (C × P) @ (P × G) -> (C × G)
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


def build_tf_access_prior(
    gene_activity: np.ndarray,
    A_tfs: np.ndarray,
    gene_to_idx_full: dict,
    sub_g2i: dict,
) -> np.ndarray:
    
    """
    Compute population-mean chromatin accessibility for each TF locus.

    Returns:
        tf_access (T,) float32 — population-mean chromatin accessibility for each TF locus, 
        derived from the ATAC gene-activity matrix.
    """

    T = len(A_tfs)
    tf_access = np.zeros(T, dtype=np.float32)
    for i, tf in enumerate(A_tfs):
        if tf in gene_to_idx_full:
            # index in full gene list
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
