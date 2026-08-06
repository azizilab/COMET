import re

import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import scanpy as sc

from config import TRAIN

from scipy.sparse import lil_matrix, issparse
from sklearn.preprocessing import normalize


def as_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def clean_tf_name(tf):
    """Motif name -> list of gene symbols (dimers "A::B" / "A-B" split, variants dropped)."""
    tf = re.sub(r"\(var\..*\)", "", tf)

    if "::" in tf:
        return tf.split("::")

    if "-" in tf:
        return tf.split("-")

    return [tf]


def build_peak_gene_matrix(peak_gene, peak_to_idx, sub_g2i, n_peaks, peak_key=str):
    """Binary (peaks x genes) linkage matrix from the peak_gene table.

    peak_key maps a peak_gene "peak" entry into the key format of peak_to_idx.
    """
    mat = lil_matrix((n_peaks, len(sub_g2i)), dtype=np.float32)
    for _, row in peak_gene.iterrows():
        p = peak_key(row["peak"])
        g = str(row["gene"])
        if p in peak_to_idx and g in sub_g2i:
            mat[peak_to_idx[p], sub_g2i[g]] = 1.0
    return mat.tocsr()


def load_datasets(config):

    print("Loading ATAC matrix...", flush=True)
    atac_X = scipy.io.mmread(config["atac"]).tocsr(copy=False)

    print("Transposing to cells x peaks...", flush=True)
    atac_X = atac_X.transpose()
    atac_X.data = atac_X.data.astype(np.float32, copy=False)

    return {
        "rna_adata": ad.read_h5ad(config["rna"]), # cells x genes, w/ metadata
        "peak_tf": scipy.io.mmread(config["peak_tf"]).tocsr(), # peaks x TFs
        "motif_tf_names": open(config["tf_names"]).read().strip().split(), # ordered list of TF names corresponding to columns of peak_tf
        "peak_gene": pd.read_csv(config["peak_gene"], sep="\t"), # table with columns: peak, gene
        "peaks": pd.read_csv(config["peaks"], header=None)[0].astype(str).tolist(), # master list of peak identifiers
        "atac_X": atac_X, # cells x peaks
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

    A_genes: HVGs with at least one peak-gene regulatory link, plus all TFs
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

    # Motif names are kept as-is for peak_tf column alignment; the cleaned gene
    # symbols are only used to match motifs against RNA genes.
    motif_tf_to_idx = {}
    for idx, motif in enumerate(motif_tf_names):
        for gene in clean_tf_name(motif):
            motif_tf_to_idx.setdefault(gene.upper(), idx)

    peak_to_idx = {p: i for i, p in enumerate(peaks)}

    # filter genes/TFs to get active "universe"
    gene_upper_to_original = {g.upper(): g for g in gene_names_all}
    A_tfs = np.array(
        [gene_upper_to_original[t] for t in motif_tf_to_idx if t in gene_upper_to_original],
        dtype=str,
    )

    genes_in_peaks = set(peak_gene["gene"].astype(str).unique())
    tf_set = set(A_tfs)
    A_genes = np.array([
        g for g in gene_names_all
        if (g in genes_in_peaks and g in hvg_genes) or g in tf_set
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
    A_genes: np.ndarray,
    A_tfs: np.ndarray,
    peak_to_idx: dict,
    sub_g2i: dict,
    motif_tf_to_idx: dict,
) -> dict:

    """
    Build M, the binary gene x TF regulatory prior matrix.

    M[g, t] = 1 if TF t has at least one motif hit in any peak linked to gene g.

    Returns:
        M        (G x T) ndarray   — binary gene-TF regulatory prior
        tf_col   (T,) ndarray      — TF positions in peak_tf / motif space
    """

    peak_gene_mat = build_peak_gene_matrix(peak_gene, peak_to_idx, sub_g2i, len(peaks))

    M_full = (peak_gene_mat.T @ peak_tf) # (G × P) @ (P × T) -> (G × T)
    M_full = (M_full > 0).astype(np.float32)

    # align M columns to A_tfs (already filtered to the motif database in build_universe)
    tf_col = np.array([motif_tf_to_idx[t.upper()] for t in A_tfs], dtype=int)
    M = M_full[:, tf_col].toarray().astype(np.float32)

    assert M.shape == (len(A_genes), len(A_tfs))

    print("TFs:", len(A_tfs))
    print("Edges:", M.sum())
    print("Density:", M.mean())

    return {
        "M": M,
        "tf_col": tf_col
    }


def build_gene_priors(
    peak_names: list[str],
    peak_gene: pd.DataFrame,
    peak_to_idx_atac: dict,
    atac_X,
    sub_g2i: dict,
) -> dict:

    """
    Compute per-gene chromatin accessibility priors from scATAC-seq data.

    Constructs a peak x gene map from the peak_gene table and multiplies it with
    the ATAC matrix to get per-cell gene activity scores, which are L1-normalized
    per cell and log1p stabilized.

    ATAC is unpaired with RNA, so there is no positional cell correspondence to
    truncate to; all ATAC cells are kept. build_local_atac remaps to (C_rna, G)
    via cluster matching, and gene_access averages over the full ATAC population.

    Returns:
        gene_activity  (C_atac x G) ndarray  — per-cell gene accessibility scores
        gene_access    (G,) ndarray          — population-mean accessibility, normalized
    """

    # peak_gene peaks are chr-start-end; ATAC peak keys are chr:start-end
    peak_gene_mat_atac = build_peak_gene_matrix(
        peak_gene, peak_to_idx_atac, sub_g2i, len(peak_names),
        peak_key=lambda p: str(p).replace("-", ":", 1),
    )

    print("peak_gene_mat_atac nnz:", peak_gene_mat_atac.nnz, "of", peak_gene.shape[0], "rows")

    gene_activity = atac_X @ peak_gene_mat_atac   # (C × P) @ (P × G) -> (C × G)
    gene_activity = normalize(gene_activity, norm="l1", axis=1)
    gene_activity = as_dense(gene_activity).astype(np.float32)
    gene_activity = np.log1p(gene_activity)

    gene_access = gene_activity.mean(axis=0)
    gene_access = (gene_access - gene_access.min()) / (gene_access.max() - gene_access.min() + 1e-8)

    return {
        "gene_activity": gene_activity,
        "gene_access": gene_access,
    }


def build_tf_access_prior(
    gene_activity: np.ndarray,
    A_tfs: np.ndarray,
    sub_g2i: dict,
) -> np.ndarray:

    """
    Compute population-mean chromatin accessibility for each TF locus.

    gene_activity columns align with A_genes, so TFs outside A_genes (none, given
    build_universe always keeps TFs) are left at 0.

    Returns:
        tf_access (T,) float32 — population-mean chromatin accessibility for each TF locus,
        derived from the ATAC gene-activity matrix.
    """

    tf_access = np.zeros(len(A_tfs), dtype=np.float32)
    for i, tf in enumerate(A_tfs):
        if tf in sub_g2i:
            tf_access[i] = float(gene_activity[:, sub_g2i[tf]].mean())

    # min-max normalise to [0, 1]
    lo, hi = tf_access.min(), tf_access.max()
    tf_access = (tf_access - lo) / (hi - lo + 1e-8)
    return tf_access
