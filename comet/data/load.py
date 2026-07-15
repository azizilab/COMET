from logging import config

import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import torch
import re
import scanpy as sc

from config import TRAIN

from scipy.sparse import lil_matrix, issparse
from sklearn.preprocessing import normalize


def load_datasets(config):

    print("Loading ATAC matrix...", flush=True)
    atac_X = scipy.io.mmread(config["atac"])

    # print("Keeping first 3000 cells...", flush=True)
    # mask = atac_X.col < 3000
    # atac_X = scipy.sparse.coo_matrix(
    #     (atac_X.data[mask], (atac_X.row[mask], atac_X.col[mask])),
    #     shape=(atac_X.shape[0], 3000)
    # )

    print("Converting to CSR...", flush=True)
    atac_X = atac_X.tocsr(copy=False)

    print("Transposing...", flush=True)
    atac_X = atac_X.transpose()

    print("Converting data to float32...", flush=True)
    atac_X.data = atac_X.data.astype(np.float32, copy=False)

    return {
        "rna_adata": ad.read_h5ad(config["rna"]), # cells x genes, w/ metadata
        "peak_tf": scipy.io.mmread(config["peak_tf"]).tocsr(), # peaks x TFs
        "motif_tf_names": open(config["tf_names"]).read().strip().split(), # ordered list of TF names corresponding to columns of peak_tf
        "peak_gene": pd.read_csv(config["peak_gene"], sep="\t"), # table with columns: peak, gene
        "peaks": pd.read_csv(config["peaks"], header=None)[0].astype(str).tolist(), # master list of peak identifiers
        #"atac_X": scipy.io.mmread(config["atac"]).tocsr().T.astype(np.float32),  # cells x peaks 
        "atac_X": atac_X,  # cells x peaks
        "peaks_atac": pd.read_csv(config["atac_peaks"], sep="\t"),
        "decipher_outputs": np.load(config["decipher_outputs"]),
    }


def build_universe(
    rna_adata: ad.AnnData,
    motif_tf_names: list[str],
    peak_gene: pd.DataFrame,
    peaks: list[str],
    peaks_atac: pd.DataFrame,
    rna_layer: str | None = None,
) -> dict:

    """
    Build active gene/TF universe from scRNA-seq and scATAC-seq inputs.

    A_genes: genes with at least one peak-gene regulatory link
    A_tfs:   TFs present in both the motif database and the RNA matrix
            (TFs without RNA expression are excluded — X_TF requires this)

    peaks_atac["Feature"] is expected in chr_start_end format and is
    converted to chr:start-end to match peak_gene coordinates.

    rna_layer: name of an `rna_adata.layers[...]` entry to use instead of
        `.X` (e.g. "raw_count_int"). `.X` is not assumed to be raw/non-negative
        expression — some preprocessing pipelines leave it log1p+scaled
        (mean ~0, includes negative values), which silently breaks any
        downstream computation that assumes non-negative expression
        (e.g. np.log1p on a cluster mean). Leave as None for datasets whose
        `.X` is already the intended representation (e.g. raw counts).
    """

    # standard scanpy normalization (operates in place on the given layer, or .X if None)
    sc.pp.normalize_total(rna_adata, target_sum=1e4, layer=rna_layer)
    sc.pp.log1p(rna_adata, layer=rna_layer)

    # HVG selection (kept, but TFs are always protected below)
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=TRAIN["N_TOP_HVG"], subset=False, layer=rna_layer)
    hvg_genes = set(rna_adata.var_names[rna_adata.var['highly_variable']])

    # RNA reference space
    gene_names_all = np.array(rna_adata.var_names.astype(str))
    gene_to_idx_full = {g: i for i, g in enumerate(gene_names_all)}
    X_source = rna_adata.layers[rna_layer] if rna_layer else rna_adata.X
    X_full = as_dense(X_source).astype(np.float32)
    
    # Keep original motif names for peak_tf column alignment
    motif_tf_names_original = motif_tf_names.copy()

    # Clean names only for matching to RNA genes
    cleaned = []
    for tf in motif_tf_names_original:
        cleaned.extend(clean_tf_name(tf))

    # preserve order, remove duplicates
    motif_gene_names = list(dict.fromkeys([x.upper() for x in cleaned]))

    gene_upper_to_original = {
    g.upper(): g for g in gene_names_all
    }

    A_tfs = np.array(
        [
            gene_upper_to_original[t]
            for t in motif_gene_names
            if t in gene_upper_to_original
        ],
        dtype=str
    )

    # TF and peak lookup dicts
    #motif_tf_to_idx = {t: i for i, t in enumerate(motif_tf_names)}
    peak_to_idx = {p: i for i, p in enumerate(peaks)}

    motif_tf_to_idx = {}

    for idx, motif in enumerate(motif_tf_names_original):
        genes = clean_tf_name(motif)

        for gene in genes:
            gene = gene.upper()
            if gene not in motif_tf_to_idx:
                motif_tf_to_idx[gene] = idx

    # filter genes/TFs to get active "universe"
    genes_in_peaks = set(peak_gene["gene"].astype(str).unique())
    #A_genes = np.array([g for g in gene_names_all if g in genes_in_peaks], dtype=str)
    A_tfs = np.array(
        [t for t in A_tfs if t.upper() in motif_tf_to_idx],
        dtype=str
    )

    A_genes = np.array([
        g for g in gene_names_all
        if (g in genes_in_peaks and g in hvg_genes) or g in A_tfs
    ], dtype=str)

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
    A_tfs = np.array(
        [t for t in A_tfs if t.upper() in motif_tf_to_idx],
        dtype=str
    )

    tf_idx = np.array([gene_to_idx_full[t] for t in A_tfs], dtype=int)
    tf_col = np.array([motif_tf_to_idx[t.upper()] for t in A_tfs], dtype=int)


    X_TF = X_full[:, tf_idx]  # refresh after final TF filtering
    M = M_full[:, tf_col].toarray().astype(np.float32)

    assert M.shape == (len(A_genes), len(A_tfs))

    print("TFs:", len(A_tfs))
    print("Edges:", M.sum())
    print("Density:", M.mean())

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

    print("peak_gene_mat_atac nnz:", peak_gene_mat_atac.nnz, "of", peak_gene.shape[0], "rows")

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

def clean_tf_name(tf):
    tf = re.sub(r"\(var\..*\)", "", tf)

    if "::" in tf:
        return tf.split("::")

    if "-" in tf:
        return tf.split("-")

    return [tf]



def build_gene_priors1(
    peak_names: list[str],         
    A_genes: np.ndarray,           
    peak_gene: pd.DataFrame,      
    peak_to_idx_atac: dict,        
    X: np.ndarray,                 
    atac_X,                       
    sub_g2i: dict, 
    min_overlap_frac: float = 0.5,
) -> dict:
    
    """
    Compute per-gene chromatin accessibility priors from scATAC-seq data.

    Constructs a peak x gene map from the peak_gene table and multiplies it with
    the ATAC matrix to get per-cell gene activity scores. These are L1-normalized
    per cell, log1p stabilized, then subsetted to the first C_rna cells (assumes
    RNA and ATAC are pre-aligned in the same cell order).

    peak_gene peaks and ATAC peaks come from different peak sets with different
    coordinate boundaries, so linkage is done via genomic interval overlap
    (min_overlap_frac of the smaller peak's length) rather than exact string match.

    Population-level prior gene_access is derived by averaging gene_activity across
    cells and min-max normalizing to [0, 1].

    Returns:
        gene_activity  (C_rna x G) ndarray   — per-cell gene accessibility scores
        gene_access    (G,) ndarray          — population-mean accessibility, normalized
        gene_access_t  (G,) tensor           — gene_access as a torch tensor
    """

    P2 = len(peak_names)
    G2 = len(A_genes)

    # Only keep peak_gene rows whose gene is in our universe
    peak_gene_valid = peak_gene[peak_gene["gene"].astype(str).isin(sub_g2i)].reset_index(drop=True)
    pg_peak_strs = peak_gene_valid["peak"].astype(str).tolist()

    overlap_df = build_peak_overlap_map(pg_peak_strs, peak_names, min_overlap_frac=min_overlap_frac)
    print("peak overlap matches:", len(overlap_df), "of", len(pg_peak_strs), "peak_gene rows")

    peak_gene_mat_atac = lil_matrix((P2, G2), dtype=np.float32)
    for pg_idx, atac_idx in zip(overlap_df["pg_idx"].values, overlap_df["atac_idx"].values):
        gene = str(peak_gene_valid["gene"].iloc[pg_idx])
        peak_gene_mat_atac[atac_idx, sub_g2i[gene]] = 1.0
    peak_gene_mat_atac = peak_gene_mat_atac.tocsr()

    gene_activity = atac_X @ peak_gene_mat_atac   # (C × P) @ (P × G) -> (C × G)
    gene_activity = normalize(gene_activity, norm="l1", axis=1)
    gene_activity = as_dense(gene_activity).astype(np.float32)
    gene_activity = np.log1p(gene_activity)

    cell_idx = np.arange(X.shape[0])
    gene_activity = gene_activity[cell_idx]

    gene_access = gene_activity.mean(axis=0)
    gene_access = (gene_access - gene_access.min()) / (gene_access.max() - gene_access.min() + 1e-8)
    gene_access_t = torch.tensor(gene_access, dtype=torch.float32)

    return {
        "gene_activity": gene_activity,
        "gene_access": gene_access,
        "gene_access_t": gene_access_t,
    }