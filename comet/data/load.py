import glob, os, re
import numpy as np
import pandas as pd
import scipy.io
import anndata as ad
import scanpy as sc

from dataclasses import dataclass, field
from typing import Optional

from scipy.sparse import lil_matrix, csr_matrix, issparse
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from config import N_TOP_HVG, K_MOMENT, RANDOM_STATE, MA_TARGET_CELLS, AML_TARGET_CELLS, MA_KEEP, MA_PSEUDOBULK_SMOOTH
from utils.common import as_dense, norm_name, knn_smooth
from utils.geometry import build_local_atac


@dataclass
class Bundle:
    name: str; paired: bool
    X: np.ndarray; X_TF: np.ndarray; ga_matched: np.ndarray
    M: np.ndarray; A_genes: np.ndarray; A_tfs: np.ndarray
    tf_indices_in_G: np.ndarray; Z_dec: np.ndarray; V_dec2: np.ndarray
    cell_types: np.ndarray; pseudo_norm: np.ndarray; gene_access: np.ndarray
    adata: Optional[ad.AnnData] = None
    ga_true: Optional[np.ndarray] = None
    sub_g2i: dict = field(default_factory=dict)
    # raw per-cell gene activity that ga_matched was pseudo-bulked from (per ATAC cell if unpaired, per RNA cell if
    # paired); utils/trajectory.py rebuilds the pseudo-bulk from it at other resolutions.
    ga_atac_cells: Optional[np.ndarray] = None
    @property
    def N(self): return self.X.shape[0]
    @property
    def G(self): return self.X.shape[1]
    @property
    def T(self): return self.X_TF.shape[1]


def finish_bundle(name, paired, rna_adata, X_full, gene_names_all, gene_to_idx_full,
                  A_genes, A_tfs, M, gene_activity_per_rna_cell, Z_dec, V_dec2,
                  cell_types, pseudo_norm, adata=None, raw_ga=None, ga_atac_cells=None):
    sub_g2i = {g: i for i, g in enumerate(A_genes)}
    gene_idx = np.array([gene_to_idx_full[g] for g in A_genes], int)
    tf_idx   = np.array([gene_to_idx_full[t] for t in A_tfs], int)
    X    = X_full[:, gene_idx].astype(np.float32)
    X_TF = X_full[:, tf_idx].astype(np.float32)
    tf_indices_in_G = np.array([sub_g2i[t] for t in A_tfs], dtype=np.int64)
    ga = gene_activity_per_rna_cell.astype(np.float32)
    nbrs = NearestNeighbors(n_neighbors=K_MOMENT).fit(Z_dec); _, ni = nbrs.kneighbors(Z_dec)
    ga = knn_smooth(ga, ni, k=K_MOMENT)                    # single upstream denoise step
    gacc = ga.mean(0); gacc = (gacc - gacc.min()) / (gacc.max() - gacc.min() + 1e-8)
    ga_true = (raw_ga.astype(np.float32) if raw_ga is not None else ga) if paired else None
    return Bundle(name=name, paired=paired, X=X, X_TF=X_TF, ga_matched=ga,
                  M=M.astype(np.float32), A_genes=A_genes, A_tfs=A_tfs,
                  tf_indices_in_G=tf_indices_in_G, Z_dec=Z_dec.astype(np.float32),
                  V_dec2=V_dec2.astype(np.float32), cell_types=np.asarray(cell_types).astype(str),
                  pseudo_norm=pseudo_norm.astype(np.float32), gene_access=gacc.astype(np.float32),
                  adata=adata, ga_true=ga_true, sub_g2i=sub_g2i,
                  ga_atac_cells=None if ga_atac_cells is None else np.asarray(ga_atac_cells, np.float32))


def load_decipher(decipher_dir):
    cands = sorted(glob.glob(os.path.join(decipher_dir, "*.npz")))
    assert cands, f"No .npz in {decipher_dir}"
    for path in cands:
        d = np.load(path, allow_pickle=True); keys = list(d.keys())
        z = next((d[k] for k in ('z','decipher_z','Z','z_latent','X_decipher_z') if k in keys), None)
        v = next((d[k] for k in ('v','decipher_v','V','z_2d','X_decipher') if k in keys), None)
        if z is None or v is None:
            arrs = [(k, d[k]) for k in keys if getattr(d[k],'ndim',0)==2]
            v = v if v is not None else next((a for _,a in arrs if a.shape[1]==2), None)
            z = z if z is not None else next((a for _,a in arrs if a.shape[1]>2), None)
        if z is not None and v is not None:
            return np.asarray(z,np.float32), np.asarray(v,np.float32)
    raise KeyError("no z/v in decipher dir")


def build_ma2020_pseudotime(rna, Z_dec, V_dec2=None, cell_types=None, n_neighbors=15, n_dcs=10):
    """Diffusion pseudotime on the Decipher latent, anchored at the TAC-1 centroid."""
    a = rna.copy(); a.obsm['X_dec'] = np.asarray(Z_dec)
    if cell_types is None:
        cell_types = a.obs['cell_type'].values if 'cell_type' in a.obs else a.obs['cell_types'].values
    cell_types = np.asarray(cell_types.values if isinstance(cell_types, (pd.Series, pd.Index)) else cell_types)
    sc.pp.neighbors(a, use_rep='X_dec', n_neighbors=n_neighbors)
    early_mask = (cell_types == "TAC-1")
    if not np.any(early_mask):
        early_mask = np.isin(cell_types, ["TAC-1", "TAC-2"])
        if not np.any(early_mask):
            raise ValueError("Could not find TAC-1/TAC-2 to set pseudotime root.")
    early_idx = np.where(early_mask)[0]; ec = a.obsm['X_dec'][early_idx]
    root_idx = int(early_idx[np.argmin(np.linalg.norm(ec - ec.mean(0), axis=1))])
    a.uns["iroot"] = root_idx
    sc.tl.diffmap(a, n_comps=n_dcs); sc.tl.dpt(a)
    pt = a.obs["dpt_pseudotime"].values.copy()
    fin = np.isfinite(pt)
    if not fin.any(): raise RuntimeError("DPT produced no finite values.")
    pt[~fin] = np.nanmax(pt[fin]); lo, hi = np.nanmin(pt), np.nanmax(pt)
    return ((pt - lo) / (hi - lo + 1e-8)).astype(np.float32)


def assign_velten_celltypes(rna, X_full, g2i):
    obs = rna.obs; N = len(obs)
    MARK = {'HSC':['CD34','THY1','PROM1','ITGA6'], 'MPP':['CD34','FLT3','GATA2'],
            'MLP':['CD34','IL7R','FLT3','DNTT'],   'CMP':['CD34','FLT3','CSF1R','CEBPA'],
            'MEP':['GATA1','GATA2','KLF1','EPOR','GYPA'], 'GMP':['MPO','ELANE','SPI1','LYZ','CEBPA']}
    scores = {}
    for ct, ms in MARK.items():
        s = np.zeros(N, np.float32); n = 0
        for gg in ms:
            if gg in g2i:
                e = np.log1p(X_full[:, g2i[gg]].astype(np.float32))
                s += (e - e.mean()) / (e.std() + 1e-8); n += 1
        scores[ct] = s / n if n else s
    sdf = pd.DataFrame(scores)
    def gate(col, pct, above=True):
        if col not in obs.columns: return np.zeros(N, bool)
        t = obs[col].quantile(pct/100); return (obs[col] > t) if above else (obs[col] <= t)
    base = gate('facs_lin',30,False) & gate('facs_cd34',40,True)
    lo, hi = gate('facs_cd38',33,False), gate('facs_cd38',50,True)
    ct = np.full(N,'CLOUD',object)
    ct[base & hi & gate('facs_cd45ra',50,True)] = 'GMP'
    ct[base & hi & gate('facs_cd135',40,False) & gate('facs_cd45ra',50,False)] = 'MEP'
    ct[base & hi & gate('facs_cd135',40,True)  & gate('facs_cd45ra',50,False)] = 'CMP'
    ct[base & lo & gate('facs_cd45ra',50,True)] = 'MLP'
    ct[base & lo & ~gate('facs_cd90',50,True) & gate('facs_cd45ra',50,False)] = 'MPP'
    ct[base & lo & gate('facs_cd90',50,True)  & gate('facs_cd45ra',50,False)] = 'HSC'
    cloud = np.where(ct=='CLOUD')[0]
    best, bs = sdf.values[cloud].argmax(1), sdf.values[cloud].max(1)
    for ci, b, sc_ in zip(cloud, best, bs):
        if sc_ > 0.5: ct[ci] = sdf.columns[b]
    return ct


def clean_tf_name(tf):
    """Motif name -> list of gene symbols (dimers "A::B" / "A-B" split, variants dropped). Used by load_generic."""
    tf = re.sub(r"\(var\..*\)", "", tf)
    if "::" in tf: return tf.split("::")
    if "-" in tf: return tf.split("-")
    return [tf]


def load_generic(base, name, rna_file=None, atac_mtx=None, atac_peaks_file=None,
                 decipher_npz=None, cell_type_col="cell_type", pseudotime_col="pseudotime",
                 rna_layer=None):
    """Loader for every dataset without a bespoke loader (all but hema/aml89/ma2020): unpaired RNA + ATAC +
    peak_tf/peak_gene motif prior + decipher_embeddings.npz. File contract: README_DATASETS.md.

    rna_file/atac_mtx/atac_peaks_file/decipher_npz default to the canonical filenames inside
    `base` (see README_DATASETS.md) when omitted; pass them explicitly to override. rna_layer
    selects an `rna_adata.layers[...]` entry instead of `.X` for datasets whose `.X` isn't raw
    counts.
    """
    rna_file        = rna_file        or f"{base}/rna.h5ad"
    atac_mtx        = atac_mtx        or f"{base}/atac_matrix.mtx.gz"
    atac_peaks_file = atac_peaks_file or f"{base}/atac_peaks.txt"
    decipher_npz    = decipher_npz    or f"{base}/decipher_embeddings.npz"
    rna = ad.read_h5ad(rna_file)
    sc.pp.normalize_total(rna, target_sum=1e4, layer=rna_layer)
    sc.pp.log1p(rna, layer=rna_layer)
    sc.pp.highly_variable_genes(rna, n_top_genes=N_TOP_HVG, subset=False, layer=rna_layer)
    hvg = set(rna.var_names[rna.var['highly_variable']])
    gene_names_all = np.array(rna.var_names.astype(str)); g2i = {g: i for i, g in enumerate(gene_names_all)}
    X_source = rna.layers[rna_layer] if rna_layer else rna.X
    X_full = as_dense(X_source).astype(np.float32)

    peak_tf = scipy.io.mmread(f"{base}/peak_tf.mtx").tocsr()
    motif_tf = open(f"{base}/tf_names.txt").read().split()
    motif2col = {}
    for i, motif in enumerate(motif_tf):
        for gene in clean_tf_name(motif):
            motif2col.setdefault(gene.upper(), i)
    peak_gene = pd.read_csv(f"{base}/peak_gene.csv", sep="\t")
    peaks = pd.read_csv(f"{base}/peaks.txt", header=None)[0].astype(str).tolist()
    p2i = {p: i for i, p in enumerate(peaks)}
    genes_in_peaks = set(peak_gene["gene"].astype(str).unique())
    gene_upper_to_original = {g.upper(): g for g in gene_names_all}
    A_tfs = np.array([gene_upper_to_original[t] for t in motif2col if t in gene_upper_to_original], str)
    A_genes = np.array([g for g in gene_names_all if (g in genes_in_peaks and g in hvg) or g in set(A_tfs)], str)
    sub_g2i = {g: i for i, g in enumerate(A_genes)}; P, G = len(peaks), len(A_genes)
    pg = lil_matrix((P, G), dtype=np.float32)
    for _, r in peak_gene.iterrows():
        p, g = str(r["peak"]), str(r["gene"])
        if p in p2i and g in sub_g2i: pg[p2i[p], sub_g2i[g]] = 1.0
    M_full = ((pg.tocsr().T @ peak_tf) > 0).astype(np.float32)
    tf_col = np.array([motif2col[t.upper()] for t in A_tfs], int)
    M = M_full[:, tf_col].toarray().astype(np.float32)

    atac = scipy.io.mmread(atac_mtx).tocsr().T.astype(np.float32)
    pa = pd.read_csv(atac_peaks_file, sep="\t")
    pnames = pa["Feature"].str.replace("_", ":", n=1).str.replace("_", "-", n=1).tolist()
    p2i_a = {p: i for i, p in enumerate(pnames)}
    fix = lambda p: str(p).replace("-", ":", 1); P2 = len(pnames); pga = lil_matrix((P2, G), dtype=np.float32)
    for _, r in peak_gene.iterrows():
        p, g = fix(r["peak"]), str(r["gene"])
        if p in p2i_a and g in sub_g2i: pga[p2i_a[p], sub_g2i[g]] = 1.0
    ga_atac = np.log1p(as_dense(normalize(atac @ pga.tocsr(), norm="l1", axis=1))).astype(np.float32)

    dec = np.load(decipher_npz); Z_dec, V_dec2 = dec["z"].astype(np.float32), dec["v"].astype(np.float32)
    ga_matched, _ = build_local_atac(X_full[:, [g2i[g] for g in A_genes]], ga_atac, Z_dec)

    if cell_type_col not in rna.obs.columns:
        raise KeyError(f"{name}: expected obs['{cell_type_col}'] for cell types; "
                       f"available columns: {list(rna.obs.columns)}")
    cell_types = rna.obs[cell_type_col].astype(str).values
    if pseudotime_col not in rna.obs.columns:
        raise KeyError(f"{name}: expected obs['{pseudotime_col}'] for pseudotime; "
                       f"available columns: {list(rna.obs.columns)}")
    pt = rna.obs[pseudotime_col].astype(np.float32).values
    finite = np.isfinite(pt)
    pt[~finite] = np.nanmax(pt[finite]) if finite.any() else 0.0
    pt = (pt - np.nanmin(pt)) / (np.ptp(pt[finite]) + 1e-8) if finite.any() else pt
    rna.obsm["X_decipher"] = V_dec2
    return finish_bundle(name, False, rna, X_full, gene_names_all, g2i, A_genes, A_tfs, M,
                         ga_matched, Z_dec, V_dec2, cell_types, pt, adata=rna, ga_atac_cells=ga_atac)


def load_nestorowa(base):
    """Nestorowa et al. 2016 mouse HSPCs (1654 cells), built by comet/scripts/build_nestorowa.py: rna.h5ad with
    layers['raw_count_int'], obs['cell_type'] (sort gates) and obs['pseudotime'] (mean BEELINE Slingshot pseudotime;
    validation only), plus paul15's mouse sciATAC atlas and GRN prior, symlinked."""
    return load_generic(
        base, "nestorowa",
        atac_mtx=f"{base}/atac_matrix.trimmed.mtx.gz",
        atac_peaks_file=f"{base}/atac_matrix.trimmed.peaks.txt",
        cell_type_col="cell_type", pseudotime_col="pseudotime",
        rna_layer="raw_count_int",
    )


def load_paul15(base):
    """Paul et al. 2015 mouse hematopoiesis. comet/scripts/build_paul15_universe.py (cleanup branch) builds
    the peak_tf/peak_gene/peaks prior from CellOracle's mouse sciATAC atlas base GRN. Expects
    Paul_etal_15_2.h5ad with layers['raw_count_int'] (`.X` there is log1p+scaled, not raw),
    obs['paul15_clusters'] for cell type, and obs['dpt_pseudotime'] for pseudotime."""
    return load_generic(
        base, "paul15",
        rna_file=f"{base}/Paul_etal_15_2.h5ad",
        atac_mtx=f"{base}/atac_matrix.trimmed.mtx.gz",
        atac_peaks_file=f"{base}/atac_matrix.trimmed.peaks.txt",
        decipher_npz=f"{base}/decipher_embeddings.npz",
        cell_type_col="paul15_clusters", pseudotime_col="dpt_pseudotime",
        rna_layer="raw_count_int",
    )


# ---- bespoke loaders ----
def load_hema(base):
    RNA_H5AD=f"{base}/velten_rna.h5ad"; PEAK_TF_MTX=f"{base}/peak_tf.mtx"
    TF_NAMES_TXT=f"{base}/tf_names.txt"; PEAK_GENE_CSV=f"{base}/peak_gene.csv"
    PEAKS_TXT=f"{base}/peaks.txt"; ATAC_MTX=f"{base}/GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz"
    ATAC_PEAKS=f"{base}/GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz"
    DEC_NPZ=f"{base}/models/decipher_outputs2/decipher_embeddings.npz"
    rna = ad.read_h5ad(RNA_H5AD)
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=N_TOP_HVG, subset=False)
    hvg = set(rna.var_names[rna.var['highly_variable']])
    gene_names_all = np.array(rna.var_names.astype(str)); g2i = {g:i for i,g in enumerate(gene_names_all)}
    X_full = as_dense(rna.X).astype(np.float32)
    peak_tf = scipy.io.mmread(PEAK_TF_MTX).tocsr(); motif_tf = open(TF_NAMES_TXT).read().split()
    motif2col = {t:i for i,t in enumerate(motif_tf)}
    peak_gene = pd.read_csv(PEAK_GENE_CSV, sep="\t"); peaks = pd.read_csv(PEAKS_TXT, header=None)[0].astype(str).tolist()
    p2i = {p:i for i,p in enumerate(peaks)}; genes_in_peaks = set(peak_gene["gene"].astype(str).unique())
    A_tfs = np.array([t for t in motif_tf if t in g2i], str)
    A_genes = np.array([g for g in gene_names_all if (g in genes_in_peaks and g in hvg) or g in set(A_tfs)], str)
    sub_g2i = {g:i for i,g in enumerate(A_genes)}; P,G = len(peaks), len(A_genes)
    pg = lil_matrix((P,G), dtype=np.float32)
    for _, r in peak_gene.iterrows():
        p,g = str(r["peak"]), str(r["gene"])
        if p in p2i and g in sub_g2i: pg[p2i[p], sub_g2i[g]] = 1.0
    M_full = ((pg.tocsr().T @ peak_tf) > 0).astype(np.float32)
    A_tfs = np.array([t for t in A_tfs if t in motif2col], str); tf_col = np.array([motif2col[t] for t in A_tfs], int)
    M = M_full[:, tf_col].toarray().astype(np.float32)
    atac = scipy.io.mmread(ATAC_MTX).tocsr().T.astype(np.float32)
    pa = pd.read_csv(ATAC_PEAKS, sep="\t")
    pnames = pa["Feature"].str.replace("_",":",n=1).str.replace("_","-",n=1).tolist(); p2i_a = {p:i for i,p in enumerate(pnames)}
    fix = lambda p: str(p).replace("-",":",1); P2 = len(pnames); pga = lil_matrix((P2,G), dtype=np.float32)
    for _, r in peak_gene.iterrows():
        p,g = fix(r["peak"]), str(r["gene"])
        if p in p2i_a and g in sub_g2i: pga[p2i_a[p], sub_g2i[g]] = 1.0
    ga_atac = np.log1p(as_dense(normalize(atac @ pga.tocsr(), norm="l1", axis=1))).astype(np.float32)
    dec = np.load(DEC_NPZ); Z_dec, V_dec2 = dec["z"].astype(np.float32), dec["v"].astype(np.float32)
    ga_matched, _ = build_local_atac(X_full[:, [g2i[g] for g in A_genes]], ga_atac, Z_dec)
    cell_types = assign_velten_celltypes(rna, X_full, g2i)
    pt = rna.obs["facs_pseudotime"].astype(np.float32).values; pt = (pt-pt.min())/(np.ptp(pt)+1e-8)
    rna.obsm["X_decipher"] = V_dec2
    return finish_bundle("hema", False, rna, X_full, gene_names_all, g2i, A_genes, A_tfs, M, ga_matched, Z_dec, V_dec2, cell_types, pt, adata=rna, ga_atac_cells=ga_atac)

def overlap_intervals(q_chr, q_start, q_end, t_chr, t_start, t_end):
    """All (query i, target j) pairs whose intervals overlap (start < other end on both sides). Targets must be
    non-overlapping and sorted by start within each chromosome (true for MERGE_peaks.csv: fixed 500 bp, sorted)."""
    pairs = []
    t_start, t_end = np.asarray(t_start, np.int64), np.asarray(t_end, np.int64)
    q_start, q_end = np.asarray(q_start, np.int64), np.asarray(q_end, np.int64)
    for c in np.unique(q_chr):
        tj = np.where(t_chr == c)[0]
        if not len(tj): continue
        ts, te = t_start[tj], t_end[tj]
        assert np.all(np.diff(ts) >= 0) and np.all(np.diff(te) >= 0), f"target peaks on {c} not sorted/non-overlapping"
        qi = np.where(q_chr == c)[0]
        lo = np.searchsorted(te, q_start[qi], side="right")   # first target with end > q_start
        hi = np.searchsorted(ts, q_end[qi], side="left")      # first target with start >= q_end
        for i, a, b_ in zip(qi, lo, hi):
            pairs.extend((i, j) for j in tj[a:b_])
    return pairs

def load_aml89(base, atac_sample_prefix="08H089"):
    RNA_H5=f"{base}/08H089_allsorted_cleaned.h5"; PEAK_TF=f"{base}/peak_tf.mtx"; TF_NAMES=f"{base}/tf_names.txt"
    PG_CSV=f"{base}/peak_gene.csv"; PEAKS_BED=f"{base}/peaks.bed"; M2G_CSV=f"{base}/motif_to_genes.csv"
    ATAC_MTX=f"{base}/MERGE_counts.mtx"; ATAC_PK=f"{base}/MERGE_peaks.csv"; DEC_NPZ=f"{base}/decipher_outputs_aml89/decipher_embeddings.npz"
    ATAC_META=f"{base}/MERGE_cell_metadata.csv"
    try: rna = ad.read_h5ad(RNA_H5)
    except Exception: rna = sc.read_10x_h5(RNA_H5)
    rna.var_names_make_unique(); chosen = np.arange(rna.n_obs)
    if AML_TARGET_CELLS and rna.n_obs > AML_TARGET_CELLS:
        strat = rna.obs['blast_cell_type_08H089'].values
        _, keep = train_test_split(chosen, test_size=AML_TARGET_CELLS, stratify=strat, random_state=RANDOM_STATE)
        chosen = np.sort(keep); rna = rna[chosen].copy()
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=N_TOP_HVG, subset=False)
    hvg = set(rna.var_names[rna.var['highly_variable']])
    gene_names_all = np.array(rna.var_names.astype(str)); g2i = {g:i for i,g in enumerate(gene_names_all)}
    X_full = as_dense(rna.X).astype(np.float32)
    peak_tf = scipy.io.mmread(PEAK_TF).tocsr(); motif_tf = open(TF_NAMES).read().split(); motif2col = {t:i for i,t in enumerate(motif_tf)}
    peak_gene = pd.read_csv(PG_CSV, sep="\t", header=None); bed = pd.read_csv(PEAKS_BED, sep="\t", header=None)
    peaks = (bed[0].astype(str)+":"+bed[1].astype(str)+"-"+bed[2].astype(str)).tolist(); p2i = {p:i for i,p in enumerate(peaks)}
    mp = pd.read_csv(M2G_CSV); gene2motifcols = {}
    for _, r in mp.iterrows():
        m, gs = str(r["motif_id"]), str(r["gene_symbol"]).upper()
        if m in motif2col: gene2motifcols.setdefault(gs, []).append(motif2col[m])
    genes_in_peaks = set(peak_gene[6].astype(str).unique())
    A_tfs = np.array([t for t in gene2motifcols if t in g2i], str)
    A_genes = np.array([g for g in gene_names_all if (g in genes_in_peaks and g in hvg) or g in set(A_tfs)], str)
    sub_g2i = {g:i for i,g in enumerate(A_genes)}; P,G = len(peaks), len(A_genes)
    pg = lil_matrix((P,G), dtype=np.float32)
    for _, r in peak_gene.iterrows():
        p,g = f"{r[0]}:{r[1]}-{r[2]}", str(r[6])
        if p in p2i and g in sub_g2i: pg[p2i[p], sub_g2i[g]] = 1.0
    M_full = ((pg.tocsr().T @ peak_tf) > 0).astype(np.float32)
    A_tfs = np.array([t for t in A_tfs if t in gene2motifcols and t in g2i], str); tf_col = np.array([gene2motifcols[t][0] for t in A_tfs], int)
    M = M_full[:, tf_col].toarray().astype(np.float32)
    atac = scipy.io.mmread(ATAC_MTX).tocsr().T.astype(np.float32); pa = pd.read_csv(ATAC_PK); P2 = atac.shape[1]
    assert len(pa) == P2, f"MERGE_peaks.csv has {len(pa)} rows but MERGE_counts.mtx has {P2} peaks"
    # The ArchR project (gs://aml-scatac/ARCHR_p08H089) also merges a separate bone-marrow run, `BM2019`
    # (= AMLATAC/BM_2019_01_10, not this patient). Keep only the patient's own ATAC cells.
    meta = pd.read_csv(ATAC_META, index_col=0)
    assert len(meta) == atac.shape[0], f"MERGE_cell_metadata.csv has {len(meta)} rows but MERGE_counts.mtx has {atac.shape[0]} cells"
    patient = meta["Sample"].astype(str).str.startswith(atac_sample_prefix).values
    assert patient.any(), f"no ATAC cells with Sample starting {atac_sample_prefix!r}"
    atac = atac[patient]
    # peak_gene.csv peaks are the prior's peak set (peaks.bed), NOT the MERGE ATAC peaks -> map by interval overlap
    # (row i of peak_gene.csv is not ATAC peak i; the old row-index mapping scrambled ga_matched).
    pga = lil_matrix((P2,G), dtype=np.float32)
    keep = peak_gene[6].astype(str).isin(sub_g2i).values
    for pr_i, at_j in overlap_intervals(peak_gene[0].astype(str).values[keep], peak_gene[1].values[keep], peak_gene[2].values[keep],
                                        pa["seqnames"].astype(str).values, pa["start"].values, pa["end"].values):
        pga[at_j, sub_g2i[str(peak_gene[6].values[keep][pr_i])]] = 1.0
    ga_atac = np.log1p(as_dense(normalize(atac @ pga.tocsr(), norm="l1", axis=1))).astype(np.float32)
    dec = np.load(DEC_NPZ); Z_dec = dec["z"][chosen].astype(np.float32); V_dec2 = dec["v"][chosen].astype(np.float32)
    ga_matched, _ = build_local_atac(X_full[:, [g2i[g] for g in A_genes]], ga_atac, Z_dec)
    cell_types = rna.obs['blast_cell_type_08H089'].astype(str).values
    pt = rna.obs['dm_0_rank'].astype(np.float32).values; pt = (pt-pt.min())/(np.ptp(pt)+1e-8)
    rna.obsm["X_decipher"] = V_dec2
    return finish_bundle("aml89", False, rna, X_full, gene_names_all, g2i, A_genes, A_tfs, M, ga_matched, Z_dec, V_dec2, cell_types, pt, adata=rna, ga_atac_cells=ga_atac)

def load_ma2020(base, target_cells=MA_TARGET_CELLS):
    RNA_H5AD=f"{base}/Ma-2020-RNA.h5ad"; ATAC_H5AD=f"{base}/Ma-2020-ATAC.h5ad"; PEAK_TF=f"{base}/ma2020_peak_tf.mtx"
    TF_NAMES=f"{base}/ma2020_tf_names.txt"; PG_CSV=f"{base}/ma2020_peak_gene.csv"; DEC_DIR=f"{base}/paired_decipher_outputs3"
    rna = ad.read_h5ad(RNA_H5AD); atac = ad.read_h5ad(ATAC_H5AD); assert rna.n_obs == atac.n_obs
    if not (rna.obs_names == atac.obs_names).all(): atac = atac[rna.obs_names].copy()
    ct_raw = rna.obs["cell_type"].astype(str); keep = ct_raw.isin(MA_KEEP).values
    print(f"Ma-2020: dropping {(~keep).sum()} cells; keeping {keep.sum()}")
    rna = rna[keep].copy(); atac = atac[keep].copy()
    Z_full, V_full = load_decipher(DEC_DIR)
    if Z_full.shape[0] == keep.size: Z_dec, V_dec2 = Z_full[keep], V_full[keep]
    else: Z_dec, V_dec2 = Z_full, V_full
    order = np.arange(rna.n_obs)
    if target_cells and rna.n_obs > target_cells:
        _, keep2 = train_test_split(order, test_size=target_cells, stratify=rna.obs["cell_type"].values, random_state=RANDOM_STATE)
        keep2 = np.sort(keep2); rna = rna[keep2].copy(); atac = atac[keep2].copy(); Z_dec, V_dec2 = Z_dec[keep2], V_dec2[keep2]
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=N_TOP_HVG, subset=False)
    hvg = set(rna.var_names[rna.var['highly_variable']])
    gene_names_all = np.array(rna.var_names.astype(str)); g2i = {g:i for i,g in enumerate(gene_names_all)}
    rna_norm2canon = {}
    for g in gene_names_all: rna_norm2canon.setdefault(norm_name(g), g)
    X_full = as_dense(rna.X).astype(np.float32)
    peak_tf = scipy.io.mmread(PEAK_TF).tocsr(); motif_tf = open(TF_NAMES).read().split()
    motif_norm2col = {}
    for i,t in enumerate(motif_tf): motif_norm2col.setdefault(norm_name(t), i)
    peak_gene = pd.read_csv(PG_CSV, sep="\t"); peak_gene.columns = [c.strip().lower() for c in peak_gene.columns]
    genes_in_peaks_norm = set(peak_gene["gene"].astype(str).map(norm_name).unique())
    A_tfs = list(dict.fromkeys(rna_norm2canon[norm_name(t)] for t in motif_tf if norm_name(t) in rna_norm2canon)); A_tfs = np.array(A_tfs, str)
    A_genes = np.array([g for g in gene_names_all if (norm_name(g) in genes_in_peaks_norm and g in hvg) or g in set(A_tfs)], str)
    sub_g2i = {g:i for i,g in enumerate(A_genes)}
    av = atac.var
    if {'chrom','chromStart','chromEnd'}.issubset(av.columns):
        peak_names = (av['chrom'].astype(str)+"_"+av['chromStart'].astype(int).astype(str)+"_"+av['chromEnd'].astype(int).astype(str)).tolist()
    else: peak_names = list(atac.var_names.astype(str))
    pgp = set(peak_gene["peak"].astype(str))
    if len(set(peak_names) & pgp) < 0.2*len(pgp): peak_names = list(atac.var_names.astype(str))
    p2i = {p:i for i,p in enumerate(peak_names)}; assert peak_tf.shape[0] == len(peak_names)
    P,G = len(peak_names), len(A_genes); pg = lil_matrix((P,G), dtype=np.float32)
    for _, r in peak_gene.iterrows():
        p, gk = str(r["peak"]), norm_name(r["gene"])
        if p in p2i and gk in rna_norm2canon:
            g = rna_norm2canon[gk]
            if g in sub_g2i: pg[p2i[p], sub_g2i[g]] = 1.0
    pg = pg.tocsr(); M_full = ((pg.T @ peak_tf) > 0).astype(np.float32)
    tf_col = np.array([motif_norm2col[norm_name(t)] for t in A_tfs], int); M = M_full[:, tf_col].toarray().astype(np.float32)
    aX = atac.X.astype(np.float32)
    if not issparse(aX): aX = csr_matrix(aX)
    ga = np.log1p(as_dense(normalize(aX @ pg, norm="l1", axis=1))).astype(np.float32)
    # per-cell SHARE-seq ATAC is very sparse: pseudo-bulk it over the same RNA clusters as the unpaired loaders, but
    # with each cluster's own paired ATAC (no matching)
    if MA_PSEUDOBULK_SMOOTH:
        ga_in, _ = build_local_atac(X_full[:, [g2i[g] for g in A_genes]], ga, Z_dec, paired=True)
    else:
        ga_in = ga
    cell_types = rna.obs["cell_type"].astype(str).values
    pt = build_ma2020_pseudotime(rna, Z_dec, V_dec2); rna.obsm["X_decipher"] = V_dec2
    return finish_bundle("ma2020", True, rna, X_full, gene_names_all, g2i, A_genes, A_tfs, M, ga_in, Z_dec, V_dec2, cell_types, pt,
                         adata=rna, raw_ga=ga, ga_atac_cells=ga if MA_PSEUDOBULK_SMOOTH else None)

LOADERS = {"hema": load_hema, "aml89": load_aml89, "ma2020": load_ma2020,
          "nestorowa": load_nestorowa, "paul15": load_paul15}
