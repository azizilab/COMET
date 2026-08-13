# COMET — GRN dictionary + ATAC-led velocity, committor-anchored (multi-dataset)

Unpaired/paired scRNA + scATAC → per-cell GRN mixing + differentiation velocity.

**What changed vs. the previous version**
- **Multi-dataset.** All dataset paths are hardcoded in **Config**; every dataset is loaded up
  front (same loaders as `experiments7`). `RUN_DATASETS` selects which ones are actually trained
  when the notebook runs — flip a name in/out without touching anything else.
- **Committor now ORIENTS the velocity (the fix).** Previously the committor was a *competing
  direction loss*: the velocity head saw only the raw ATAC/GRN disequilibrium, so it structurally
  could not represent the committor field, and the committor loss just fought the reconstruction-
  aligned velocity loss to a chaotic draw — raising the committor weight only injected noise (this
  is why ma2020 stayed chaotic at high weight and aml89's sign flip never visibly took hold). Now
  the committor **orients the disequilibrium by construction** (`orient_disequilibrium`): the arrow
  is projected into the committor's forward cone — its sign along the committor axis is locked (no
  backflow / no reversal), while the GRN/ATAC term keeps the per-cell **magnitude** and the
  committor-**orthogonal** tilt (lineage/branch structure). That oriented vector is what the head
  sees, so the committor actually steers the flow and the GRN still supplies the mechanism. The
  committor *loss* remains only as a light reinforcement, not the steering mechanism.
- **First-moment velocity denoising (the ma2020 fix).** Single-cell ATAC is sparse, so the per-cell
  ATAC→RNA lead is noisy — worst for the *paired* ma2020, whose gene-activity is real per-cell ATAC
  (hema/aml89 use 40-cluster pseudo-bulk ATAC, already smooth). That noise rode through into the
  GRN-orthogonal tilt and per-cell magnitude, giving a correct committor backbone under
  high-frequency jitter. The lead signals and the final gene-space velocity are now **kNN-pooled over
  the state-space neighborhood** (`VEL_MOMENT_KNN`, larger for ma2020) — the standard first-moment
  denoising. Because the arrow of time is committor-locked, pooling can only remove jitter, never
  reverse the flow.
- **State space vs. display embedding.** The gene-space velocity is embedding-independent — the
  same (D=G)-dim displacement is projected onto every 2D embedding; only the projection changes.
  The one embedding that changes the trajectory is the latent **state space** the committor/
  archetypes/graph are built in: `STATE_SPACE` picks it per dataset, `DISPLAY_EMBEDDINGS` picks the
  2D projections to plot (a list per dataset).
- **Entropy as a hyperparameter.** The committor root is found from chromatin openness
  (`z(chrom_entropy)+z(breadth)`, root = high). That assumption holds for the differentiation
  datasets but is *inverted* for AML (leukemic blasts gain accessibility). `ENTROPY_ROOT_TRUST`
  is a per-dataset knob in `[-1, 1]`: its **sign** flips the root assumption (set it negative for
  aml89), and its **magnitude** scales how strongly the committor anchors the velocity.
- **One clean arrow plot** per dataset using the `clean_quiver` (grid-aggregated) style ported from
  `experiments7` — no streamplot.
- Velocity direction still comes from the **ATAC→RNA lead** propagated through the learned GRN, not
  from `steady(RNA) − RNA`. No pseudotime is used to build the velocity.

---

## Config — set everything here



```python
import os

# ── which datasets to actually run this pass ──────────────────────────────────
# every dataset below is LOADED; only those named here are TRAINED + plotted.
RUN_DATASETS = ["hema", "ma2020", "aml89"]                 # e.g. ["hema", "ma2020", "aml89"]

# ── file system roots (one per dataset) ───────────────────────────────────────
BASES = {
    "hema":   "/home/sc5625/cll_project/decipher_atac_project/m_atac_pipeline",
    "aml89":  "/home/sc5625/cll_project/decipher_atac_project/aml89",
    "ma2020": "/home/sc5625/cll_project/decipher_atac_project/decipher-atac/paired",
}
OUT_DIR = "comet_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ── data / subsampling ────────────────────────────────────────────────────────
N_TOP_HVG        = 3000
K_MOMENT         = 15
RANDOM_STATE     = 42
MA_TARGET_CELLS  = 8000
AML_TARGET_CELLS = 5000
MA_KEEP = ["TAC-1", "TAC-2", "IRS", "Hair Shaft-Cuticle/Cortex", "Medulla"]

# ── model / training hyperparameters ──────────────────────────────────────────
D_ATOMS       = 4        # number of GRN dictionary atoms (archetypes)
N_EPOCHS      = 80
PHASE1_EPOCHS = 40
BATCH_SIZE    = 512
GEO_TAU_SCALE = 0.5      # smaller -> sharper geometric mixing targets (less mixing)
LAMBDA_GRN    = 1.0      # weight of GRN-propagated lead inside the velocity disequilibrium

# ── training-cost control (speedup w.r.t. #cells) ─────────────────────────────
# The dominant per-batch cost is the O(D·B·G·T) atom einsum, so wall-time scales with the
# NUMBER OF BATCHES per epoch = N/BATCH_SIZE — i.e. linearly in #cells. TARGET_CELLS_PER_EPOCH
# caps how many random minibatches each epoch draws, so per-epoch cost stops growing with N.
# This is ordinary minibatch SGD: every step is still an unbiased 512-cell gradient estimate,
# so the model and losses are mathematically unchanged — only the count of stochastic updates
# per epoch is fixed instead of growing. Small datasets (hema) already have fewer batches than
# the cap and are untouched. Lower it for more speed, raise it (or set to None) to see all cells.
TARGET_CELLS_PER_EPOCH = 2048
FORCE_RERUN = False      # True = retrain even if a saved run exists; False = load & skip finished runs

# ── committor field / "entropy assumption" hyperparameter ─────────────────────
# ENTROPY_ROOT_TRUST[name] in [-1, 1]:
#   sign  = root orientation for the chromatin-entropy assumption
#           (+ = root is HIGH chromatin entropy, the normal-differentiation case;
#            - = FLIP it, for datasets where the assumption is reversed, e.g. aml89)
#   |val| = how strongly the committor field anchors the velocity (0 = ignore committor,
#           1 = full anchor). Tune down when you trust the chromatin axis less.
ENTROPY_ROOT_TRUST = {"hema": 1.0, "ma2020": 1.0, "aml89": -0.6}
# The committor now ORIENTS the velocity by construction (a forward-cone projection), so its
# weight is no longer the lever that decides whether the flow follows it — it always does. The
# consistency loss below is just a light reinforcement, so these weights stay small and uniform.
COMMIT_LOSS_W      = 1.0    # base weight on the (now light) committor reinforcement loss
COMMIT_W_SCALE     = {"hema": 1.0, "ma2020": 1.0, "aml89": 1.0}
COMMIT_ROOT_Q      = 0.90   # top-quantile of phi taken as the root set
COMMIT_SINK_FRAC   = 0.06   # fraction of eccentric low-phi cells taken as sinks
COMMIT_GRAPH_K     = 15     # kNN for the committor graph (in the state space)

# ── velocity denoising (first-moment kNN pooling in the state space) ──────────
# Single-cell ATAC is very sparse, so the raw ATAC->RNA lead is noisy PER CELL. This bites the
# PAIRED ma2020 dataset hardest: its gene-activity is real per-cell ATAC (smoothed once with k=15),
# whereas hema/aml89 use 40-cluster pseudo-bulk ATAC (already very smooth) before the same step.
# That extra per-cell noise rides through `disequil` into the GRN-orthogonal tilt and the per-cell
# magnitude, giving a correct committor backbone buried under high-frequency jitter ("right
# direction, just messy"). We pool the velocity lead — and the final gene-space velocity — over each
# cell's state-space neighborhood (the standard "first moment" denoising) before building/plotting
# the arrows. The committor backbone is already smooth, so this only cleans the noisy part. Use a
# larger neighborhood for the noisy paired dataset.
VEL_MOMENT_KNN  = {"hema": 15, "ma2020": 45, "aml89": 15}
SMOOTH_VELOCITY = True     # also kNN-pool the final gene-space velocity before projecting

# ── how much GRN structure is allowed to tilt the committor-oriented arrow ─────
# The committor fixes the arrow of TIME (forward direction, no backflow). BETA_PERP sets how far
# the GRN/ATAC-derived, committor-ORTHOGONAL component may tilt the arrow inside that forward cone
# (0 = pure committor orientation with GRN magnitude; larger = more GRN-flavoured branching). This
# is what keeps the method from collapsing to "just plot the committor": direction is committor-
# locked, but the gene program, per-cell speed, and lateral branch structure are all GRN.
BETA_PERP = {"hema": 0.5, "ma2020": 0.4, "aml89": 0.5}   # ma2020 ATAC is noisier -> lean on committor a touch more

# ── trajectory STATE SPACE vs. DISPLAY embeddings ─────────────────────────────
# The gene-space velocity is embedding-independent: it is the SAME (D=G)-dim displacement no matter
# which 2D you plot it on — only the final projection changes with the display embedding. The ONE
# embedding choice that actually changes the trajectory is the LATENT STATE SPACE the model is built
# in (committor graph, archetypes, geometric mixing targets, projection search space all live here).
# STATE_SPACE picks that latent per dataset; DISPLAY_EMBEDDINGS picks which 2D projections to plot.
STATE_SPACE = {"hema": "decipher", "ma2020": "decipher", "aml89": "decipher"}   # "decipher" | "pca"
DISPLAY_EMBEDDINGS = {
    "hema":   ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
    "ma2020": ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
    "aml89":  ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
}

MARKER_GENES = {
    "hema":   ["CD34", "MPO", "LYZ", "GATA1", "SPI1", "CEBPA"],
    "aml89":  ["CD34", "KIT", "MPO", "ELANE", "ITGAM", "CD14"],
    "ma2020": ["Krt14", "Krt15", "Sox9", "Krt1", "Krt10", "Lef1"],
}
```

## Imports



```python
import glob, warnings
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from dataclasses import dataclass, field
from typing import Optional
from numpy.random import default_rng

from scipy.sparse import lil_matrix, csr_matrix, issparse
from scipy.sparse.csgraph import shortest_path, dijkstra
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore"); sc.settings.verbosity = 0
np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
```

## Helper functions



```python
def as_dense(x):  return x.toarray() if issparse(x) else np.asarray(x)
def norm_name(s): return str(s).strip().casefold()

def zscore_cols(A, eps=1e-8):
    """Per-gene (per-column) z-score across cells."""
    A = np.asarray(A, dtype=np.float32)
    return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + eps)

def zsc(v):
    v = np.asarray(v, np.float64)
    return ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)

def l2n(v, axis=-1, eps=1e-8):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)

def knn_smooth(signal, neighbor_idx, k=None):
    nb = neighbor_idx if k is None else neighbor_idx[:, :k]
    return signal[nb].mean(axis=1)
```

## Step 1: Bundle + loaders (all datasets, hardcoded paths)

`Bundle` is the single per-dataset container the rest of the notebook reads. The three loaders are
the verbatim data contracts from the pipeline (identical to `experiments7`). Only `RUN_DATASETS`
decides which bundles get trained downstream — everything is loaded here.



```python
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
    @property
    def N(self): return self.X.shape[0]
    @property
    def G(self): return self.X.shape[1]
    @property
    def T(self): return self.X_TF.shape[1]


def build_local_atac(X_rna, gene_activity_atac_cells, cluster_space,
                     n_clusters=40, k_atac=8, seed=RANDOM_STATE):
    """Per-cell ATAC accessibility via pseudo-bulk cluster matching (unpaired-safe).
    Cluster in a space that resolves cell state (the Decipher LATENT), not 2D viz."""
    G = X_rna.shape[1]
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
    labels = km.fit_predict(cluster_space)
    cluster_rna = np.zeros((n_clusters, G), np.float32)
    for c in range(n_clusters):
        m = labels == c
        if m.sum(): cluster_rna[c] = X_rna[m].mean(0)
    cr = normalize(cluster_rna, norm='l2'); ga = normalize(gene_activity_atac_cells, norm='l2')
    _, aidx = NearestNeighbors(n_neighbors=k_atac).fit(ga).kneighbors(cr)
    cluster_access = np.stack([gene_activity_atac_cells[aidx[c]].mean(0) for c in range(n_clusters)])
    local = cluster_access[labels]
    lo, hi = local.min(0, keepdims=True), local.max(0, keepdims=True)
    return ((local - lo) / (hi - lo + 1e-8)).astype(np.float32), labels


def finish_bundle(name, paired, rna_adata, X_full, gene_names_all, gene_to_idx_full,
                  A_genes, A_tfs, M, gene_activity_per_rna_cell, Z_dec, V_dec2,
                  cell_types, pseudo_norm, adata=None):
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
    return Bundle(name=name, paired=paired, X=X, X_TF=X_TF, ga_matched=ga,
                  M=M.astype(np.float32), A_genes=A_genes, A_tfs=A_tfs,
                  tf_indices_in_G=tf_indices_in_G, Z_dec=Z_dec.astype(np.float32),
                  V_dec2=V_dec2.astype(np.float32), cell_types=np.asarray(cell_types).astype(str),
                  pseudo_norm=pseudo_norm.astype(np.float32), gene_access=gacc.astype(np.float32),
                  adata=adata, ga_true=(ga if paired else None), sub_g2i=sub_g2i)


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
```


```python
# ---- the three loaders (verbatim data contracts) ----
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
    return finish_bundle("hema", False, rna, X_full, gene_names_all, g2i, A_genes, A_tfs, M, ga_matched, Z_dec, V_dec2, cell_types, pt, adata=rna)

def load_aml89(base):
    RNA_H5=f"{base}/08H089_allsorted_cleaned.h5"; PEAK_TF=f"{base}/peak_tf.mtx"; TF_NAMES=f"{base}/tf_names.txt"
    PG_CSV=f"{base}/peak_gene.csv"; PEAKS_BED=f"{base}/peaks.bed"; M2G_CSV=f"{base}/motif_to_genes.csv"
    ATAC_MTX=f"{base}/MERGE_counts.mtx"; ATAC_PK=f"{base}/MERGE_peaks.csv"; DEC_NPZ=f"{base}/decipher_outputs_aml89/decipher_embeddings.npz"
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
    pga = lil_matrix((P2,G), dtype=np.float32)
    for idx, r in peak_gene.iterrows():
        g = str(r[6])
        if g in sub_g2i and idx < P2: pga[idx, sub_g2i[g]] = 1.0
    ga_atac = np.log1p(as_dense(normalize(atac @ pga.tocsr(), norm="l1", axis=1))).astype(np.float32)
    dec = np.load(DEC_NPZ); Z_dec = dec["z"][chosen].astype(np.float32); V_dec2 = dec["v"][chosen].astype(np.float32)
    ga_matched, _ = build_local_atac(X_full[:, [g2i[g] for g in A_genes]], ga_atac, Z_dec)
    cell_types = rna.obs['blast_cell_type_08H089'].astype(str).values
    pt = rna.obs['dm_0_rank'].astype(np.float32).values; pt = (pt-pt.min())/(np.ptp(pt)+1e-8)
    rna.obsm["X_decipher"] = V_dec2
    return finish_bundle("aml89", False, rna, X_full, gene_names_all, g2i, A_genes, A_tfs, M, ga_matched, Z_dec, V_dec2, cell_types, pt, adata=rna)

def load_ma2020(base, target_cells=MA_TARGET_CELLS):
    RNA_H5AD=f"{base}/Ma-2020-RNA.h5ad"; ATAC_H5AD=f"{base}/Ma-2020-ATAC.h5ad"; PEAK_TF=f"{base}/ma2020_peak_tf.mtx"
    TF_NAMES=f"{base}/ma2020_tf_names.txt"; PG_CSV=f"{base}/ma2020_peak_gene.csv"; DEC_DIR=f"{base}/paired_decipher_outputs2"
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
    cell_types = rna.obs["cell_type"].astype(str).values
    pt = build_ma2020_pseudotime(rna, Z_dec, V_dec2); rna.obsm["X_decipher"] = V_dec2
    return finish_bundle("ma2020", True, rna, X_full, gene_names_all, g2i, A_genes, A_tfs, M, ga, Z_dec, V_dec2, cell_types, pt, adata=rna)

LOADERS = {"hema": load_hema, "aml89": load_aml89, "ma2020": load_ma2020}

# load ONLY the datasets requested this run (loaders are heavy; no need to load the rest)
DATASETS = {}
for name in RUN_DATASETS:
    DATASETS[name] = LOADERS[name](BASES[name])
for k, b in DATASETS.items():
    print(f"{k:8s} | paired={b.paired} | N={b.N} G={b.G} T={b.T} | types={sorted(set(b.cell_types))}")
```

    Ma-2020: dropping 25034 cells; keeping 7197
    hema     | paired=False | N=1035 G=3016 T=277 | types=[np.str_('CLOUD'), np.str_('CMP'), np.str_('GMP'), np.str_('HSC'), np.str_('MEP'), np.str_('MLP'), np.str_('MPP')]
    ma2020   | paired=True | N=7197 G=2647 T=788 | types=[np.str_('Hair Shaft-Cuticle/Cortex'), np.str_('IRS'), np.str_('Medulla'), np.str_('TAC-1'), np.str_('TAC-2')]
    aml89    | paired=False | N=5000 G=3198 T=785 | types=[np.str_('blast0'), np.str_('blast1'), np.str_('blast2'), np.str_('blast3'), np.str_('ery'), np.str_('immature')]


## Step 2: Committor field (root-anchored, backflow-free) + entropy hyperparameter

The committor `q` (transition-path theory) is harmonic — `q = 0` on the root, `q = 1` on the sinks,
and by the maximum principle it has **no interior extrema**, so its gradient never reverses at the
root. The one thing it needs is the **root**, which we take from chromatin openness
(`z(chrom_entropy) + z(breadth)`, root = high). `ENTROPY_ROOT_TRUST[name]` controls that assumption:
its sign flips root ↔ terminal (negative for aml89, whose blasts *gain* accessibility), its
magnitude scales the anchor strength downstream. The field is built **marker-free** (root = top-φ
cells, sinks = eccentric low-φ leaves), so it never sees the cell-type labels.



```python
def latent_knn(src, k):
    _, idx = NearestNeighbors(n_neighbors=k+1).fit(src).kneighbors(src)
    return idx[:, 1:]

def moment_operator(Z, k):
    """Row-normalized kNN averaging operator (includes self) over the state space Z:
    (S @ x)[i] = mean of x over {i} ∪ kNN_Z(i). Used for first-moment denoising of the noisy
    per-cell ATAC->RNA velocity lead and of the final gene-space velocity — the committor backbone
    is already smooth, so this only cleans the high-frequency GRN/ATAC jitter."""
    Z = np.asarray(Z, np.float32); N = len(Z)
    idx  = latent_knn(Z, k)
    cols = np.concatenate([np.arange(N)[:, None], idx], axis=1)          # (N, k+1), self first
    rows = np.repeat(np.arange(N), k + 1)
    data = np.full(rows.shape, 1.0 / (k + 1), np.float32)
    return sp.csr_matrix((data, (rows, cols.ravel())), shape=(N, N))

def palette_for(cts):
    u = np.unique(cts); cmap = plt.cm.tab20 if len(u) > 10 else plt.cm.tab10
    return {c: cmap(i / max(1, len(u)-1)) for i, c in enumerate(u)}

def scatter_types(ax, V, cts, pal, s=10, alpha=0.35, legend=False):
    for c in np.unique(cts):
        m = cts == c
        ax.scatter(V[m,0], V[m,1], c=[pal[c]], s=s, alpha=alpha, edgecolors='none', label=c, rasterized=True)
    if legend: ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

def chromatin_potential(b):
    """Load-bearing orientation scalar: z(chrom_entropy) + z(breadth). Count-based, root-HIGH."""
    ga = normalize(np.clip(b.ga_matched, 0, None), 'l1', axis=1)
    chrom_entropy = -(ga * np.log(ga + 1e-12)).sum(1)
    thr = np.median(b.ga_matched, axis=0, keepdims=True)
    breadth = (b.ga_matched > thr).mean(1)
    return (zsc(chrom_entropy) + zsc(breadth)).astype(np.float32)

def build_graph(src, k=15, mode="gauss"):
    """Symmetric kNN graph in the latent src. mode='gauss' -> affinity; 'dist' -> edge lengths."""
    idx = latent_knn(src, k); N = len(src)
    rows = np.repeat(np.arange(N), k); cols = idx.ravel()
    d = np.linalg.norm(src[rows] - src[cols], axis=1)
    if mode == "dist":
        W = sp.csr_matrix((d, (rows, cols)), shape=(N, N))
    else:
        sigma = np.median(d) + 1e-8; w = np.exp(-(d**2) / (2*sigma**2))
        W = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
    return W.maximum(W.T), idx

def committor(W, root_mask, sink_mask):
    """Harmonic committor: L q = 0 on interior, q=0 on root, q=1 on sinks."""
    N = W.shape[0]; d = np.asarray(W.sum(1)).ravel(); L = (sp.diags(d) - W).tocsr()
    q = np.zeros(N); q[sink_mask] = 1.0; bnd = root_mask | sink_mask; interior = ~bnd
    if interior.sum() == 0: return q
    L_II = L[interior][:, interior].tocsc(); L_IB = L[interior][:, bnd]
    try:    q[interior] = spsolve(L_II, -L_IB.dot(q[bnd]))
    except Exception: q[interior] = 0.5
    return np.clip(np.nan_to_num(q, nan=0.5), 0, 1)


def committor_field(b, entropy_trust=1.0, Z=None, k_graph=COMMIT_GRAPH_K,
                    root_q=COMMIT_ROOT_Q, sink_frac=COMMIT_SINK_FRAC, sink_phi_pct=40, seed=0):
    """Marker-free committor scalar q in [0,1] (0=root, 1=terminal), plus root/sink masks.

    entropy_trust: sign flips the chromatin-entropy root assumption (use < 0 for aml89);
                   magnitude is applied later, as the velocity-loss weight.
    Z:             latent STATE SPACE the committor graph is built in (defaults to Decipher z).
    """
    Z    = b.Z_dec if Z is None else np.asarray(Z, np.float32)
    sign = 1.0 if entropy_trust >= 0 else -1.0
    phi  = sign * chromatin_potential(b)                     # root = high phi
    N    = b.N
    root_mask = phi >= np.quantile(phi, root_q)
    # sinks: graph-eccentric AND chromatin-closed (low phi), not in the root set
    Wd, _ = build_graph(Z, k=k_graph, mode="dist")
    land  = np.random.default_rng(seed).choice(N, size=min(200, N), replace=False)
    D     = shortest_path(Wd, method="D", indices=land)
    ecc   = np.nan_to_num(np.nanmax(np.where(np.isfinite(D), D, np.nan), 0), nan=0.0)
    low_phi = phi < np.percentile(phi, sink_phi_pct)
    order = np.argsort(-ecc)
    picked = [i for i in order if low_phi[i] and not root_mask[i]][:int(sink_frac*N)]
    if len(picked) < 10:
        picked = [i for i in order if not root_mask[i]][:int(sink_frac*N)]
    sink_mask = np.zeros(N, bool); sink_mask[picked] = True
    Wg, _ = build_graph(Z, k=k_graph, mode="gauss")
    q = committor(Wg, root_mask, sink_mask).astype(np.float32)
    return q, root_mask, sink_mask, phi.astype(np.float32)


def committor_gene_direction(b, q, Z=None, k=COMMIT_GRAPH_K, chunk=1024):
    """Per-cell gene-space direction of increasing committor (the 'forward' displacement the
    velocity should follow). d_i = Σ_j (q_j - q_i)(rna_z_j - rna_z_i) over latent neighbors,
    L2-normalized. Cells on a flat plateau (root/terminal) get ~0 direction and are marked invalid.

    Vectorised: using Σ_j w_ij (rz_j - rz_i) = (Σ_j w_ij rz_j) - (Σ_j w_ij) rz_i, each block of cells
    is one einsum instead of a Python per-cell loop — identical output, far faster for large N.
    Chunked so the (chunk, k, G) neighbor gather never blows up memory.
    Z: latent STATE SPACE whose kNN defines 'forward' (defaults to Decipher z)."""
    Z = b.Z_dec if Z is None else np.asarray(Z, np.float32)
    idx = latent_knn(Z, k); rz = zscore_cols(b.X)
    N, G = rz.shape; q = q.astype(np.float32); D = np.zeros((N, G), np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N); nb = idx[s:e]                        # (c, k)
        w = (q[nb] - q[s:e, None]).astype(np.float32)               # (c, k)
        D[s:e] = np.einsum('ck,ckg->cg', w, rz[nb]) - w.sum(1)[:, None] * rz[s:e]
    nrm = np.linalg.norm(D, axis=1); valid = nrm > 1e-6
    D[valid] /= nrm[valid][:, None]
    return D.astype(np.float32), valid
```

## Step 3: clean_quiver — grid-aggregated arrow plot (the required figure)

Ported from `experiments7`. This replaces the streamplot: it averages the per-cell velocity onto a
coarse grid and draws one arrow per populated cell, coloured by local speed, with the committor root
starred. It reads cleaner than streamlines and is the display used for the final figure.



```python
def clean_quiver(ax, V2D, emb, cts, title, pal=None, n_grid=22, root_mask=None):
    pal = pal or palette_for(cts); scatter_types(ax, emb, cts, pal, s=12, alpha=0.30)
    xmin, xmax = emb[:,0].min(), emb[:,0].max(); ymin, ymax = emb[:,1].min(), emb[:,1].max()
    xs, ys = np.linspace(xmin, xmax, n_grid), np.linspace(ymin, ymax, n_grid); r = (xmax-xmin)/n_grid*1.2
    gx, gy, gu, gv = [], [], [], []
    for xi in xs:
        for yi in ys:
            near = (emb[:,0]-xi)**2 + (emb[:,1]-yi)**2 < r**2
            if near.sum() > 3:
                gx.append(xi); gy.append(yi); gu.append(V2D[near,0].mean()); gv.append(V2D[near,1].mean())
    u, v = np.array(gu), np.array(gv); sp_ = np.sqrt(u**2 + v**2) + 1e-8
    ax.quiver(gx, gy, u/sp_, v/sp_, sp_, cmap='magma', scale=28, width=0.006, headwidth=4, zorder=4)
    if root_mask is not None and root_mask.any():
        rc = emb[root_mask].mean(0); ax.scatter(*rc, marker="*", s=420, c="lime", edgecolors="k", zorder=6)
    ax.set_title(title, fontsize=10, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])


def make_embeddings(b, seed=RANDOM_STATE, wanted=None):
    """Display embeddings: {name: (emb2d, kNN_source_space)}. These are DISPLAY projections only —
    the same gene-space velocity is projected onto each; they do not change the trajectory.
    `wanted` (a set/list of names) restricts which are built so UMAP isn't computed if unused.
    decipher2d + pca2d are always available; the UMAP variants require `umap` (skipped if absent)."""
    want = None if wanted is None else set(wanted)
    need = lambda n: (want is None) or (n in want)
    E = {}
    E["decipher2d"] = (b.V_dec2, b.Z_dec)
    Xz = zscore_cols(b.X)
    Xp = PCA(n_components=min(30, b.G - 1, b.N - 1), random_state=seed).fit_transform(Xz)
    E["pca2d"] = (Xp[:, :2].astype(np.float32), Xp.astype(np.float32))
    if need("umap_decipher") or need("umap_pca"):
        try:
            import umap
            if need("umap_decipher"):
                E["umap_decipher"] = (umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.3)
                                      .fit_transform(b.Z_dec).astype(np.float32), b.Z_dec)
            if need("umap_pca"):
                E["umap_pca"]      = (umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.3)
                                      .fit_transform(Xp).astype(np.float32), Xp.astype(np.float32))
        except Exception as e:
            print(f"  [umap unavailable, plotting decipher2d + pca2d only: {e}]")
    return E
```

## Step 4: GRN dictionary model (shared base + residual atoms)

`predict_rna = A_base @ tf + Σ_d w_d (A_d @ tf)`. The shared base carries the common regulatory
program so the atoms only encode lineage-specific *residuals*, which removes most of the
reconstruction incentive to hard-assign. The same effective per-cell GRN is reused by the velocity.



```python
def select_archetypes_fps(Z, D, seed=0):
    """Furthest-point sampling -> indices of D extreme cells (manifold corners)."""
    c = Z.mean(axis=0)
    first = int(np.argmax(((Z - c) ** 2).sum(1))); idx = [first]
    dmin = ((Z - Z[first]) ** 2).sum(1)
    for _ in range(D - 1):
        nxt = int(np.argmax(dmin)); idx.append(nxt)
        dmin = np.minimum(dmin, ((Z - Z[nxt]) ** 2).sum(1))
    return np.array(idx, dtype=int)

def geometric_soft_targets(Z, arch_idx, tau_scale=0.5):
    """q_i = softmax(-||z_i - z_arch||^2 / tau). Returns (N, D), rows sum to 1."""
    A = Z[arch_idx]
    d2 = ((Z[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    tau = np.median(d2) * tau_scale + 1e-8
    q = np.exp(-d2 / tau); q = q / (q.sum(1, keepdims=True) + 1e-8)
    return q.astype(np.float32)

def init_atoms_from_soft_targets(X, q, M, A_tfs):
    """Soft-LFC init: each atom's weights come from cells weighted toward its archetype."""
    D = q.shape[1]; G = X.shape[1]; W_init = np.zeros((D, G, len(A_tfs)), dtype=np.float32)
    for d in range(D):
        w = q[:, d]; wsum = w.sum() + 1e-8
        mean_in  = (X * w[:, None]).sum(0) / wsum
        mean_out = (X * (1 - w)[:, None]).sum(0) / ((1 - w).sum() + 1e-8)
        lfc = np.log1p(mean_in) - np.log1p(mean_out)
        for t_i in range(len(A_tfs)):
            col = M[:, t_i]
            if col.sum() == 0: continue
            lfc_masked = lfc * col; rng = np.abs(lfc_masked).max() + 1e-8
            W_init[d, :, t_i] = lfc_masked / rng * 0.3
    return W_init


class GRNDictionary(nn.Module):
    def __init__(self, G, T, M, D=4):
        super().__init__()
        self.D, self.G, self.T = D, G, T
        self.W_raw  = nn.Parameter(torch.zeros(D, G, T))   # residual atoms
        self.W_base = nn.Parameter(torch.zeros(G, T))      # shared base
        self.register_buffer("M", torch.tensor(M, dtype=torch.float32))
        self.register_buffer("TF_access", torch.zeros(T))
        self.register_buffer("mean_tf",  torch.zeros(T))

    def init_from_atoms(self, W_init):
        with torch.no_grad():
            self.W_raw.copy_(torch.tensor(np.arctanh(np.clip(W_init, -0.95, 0.95)), dtype=torch.float32))
        print("  GRN atoms initialized from archetype soft-LFC profiles (base starts at 0).")

    @property
    def A_dict(self): return torch.tanh(self.W_raw) * self.M.unsqueeze(0)   # (D, G, T)
    @property
    def A_base(self): return torch.tanh(self.W_base) * self.M               # (G, T)

    def _atom_term(self, X_in, weights):
        P = torch.einsum('bt,dgt->dbg', X_in, self.A_dict)    # (D, B, G)
        return (P * weights.T.unsqueeze(-1)).sum(0)           # (B, G)

    def predict_rna(self, X_TF, weights):
        return (X_TF @ self.A_base.T) + self._atom_term(X_TF, weights)
    def production_from_lead(self, tf_lead, weights):
        return (tf_lead @ self.A_base.T) + self._atom_term(tf_lead, weights)
    def effective_A(self, weights_vec):
        return self.A_base + torch.einsum('d,dgt->gt', weights_vec, self.A_dict)

    def sparsity_loss(self):
        return torch.abs(self.A_dict).mean() + torch.abs(self.A_base).mean()

    def diversity_loss(self):
        A_flat = self.A_dict.reshape(self.D, -1); A_norm = F.normalize(A_flat, dim=1)
        eye = torch.eye(self.D, device=A_flat.device)
        param_div = ((A_norm @ A_norm.T) * (1 - eye)).abs().mean()
        pred = torch.einsum('dgt,t->dg', self.A_dict, self.mean_tf); pred = F.normalize(pred, dim=1)
        pred_div = ((pred @ pred.T) * (1 - eye)).abs().mean()
        return 0.5 * param_div + 0.5 * pred_div

    def tf_access_consistency_loss(self):
        closed = 1.0 - self.TF_access.clamp(0, 1)
        return (torch.abs(self.A_dict) * closed.unsqueeze(0).unsqueeze(0)).mean()


class CellStateEncoder(nn.Module):
    """RNA -> D mixing weights. Warm-started to reproduce the soft targets q (not hard labels)."""
    def __init__(self, G, D, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(G, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU())
        self.head = nn.Linear(hidden // 2, D); self.D = D

    def init_from_soft_targets(self, X, q, n_epochs=40, lr=1e-3, batch=256):
        print("  Pre-training encoder to match geometric soft targets...")
        self.train(); opt = torch.optim.Adam(self.parameters(), lr=lr)
        X_t = torch.tensor(X, dtype=torch.float32); q_t = torch.tensor(q, dtype=torch.float32)
        dl  = DataLoader(TensorDataset(X_t, q_t), batch_size=batch, shuffle=True)
        for ep in range(n_epochs):
            tot = 0.0
            for xb, qb in dl:
                logp = F.log_softmax(self.head(self.encoder(xb)), dim=-1)
                loss = (qb * (torch.log(qb + 1e-8) - logp)).sum(1).mean()   # KL(q||p)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if ep % 10 == 0: print(f"    encoder pretrain epoch {ep}: KL={tot/len(dl):.4f}")

    def forward(self, rna, temperature=1.0):
        return F.softmax(self.head(self.encoder(rna)) / temperature, dim=-1)


class LinearDecoder(nn.Module):
    """Interpretable 2D bottleneck used only to regularize the encoder latent during training."""
    def __init__(self, hidden, G, ld_dim=2):
        super().__init__()
        self.bottleneck = nn.Linear(hidden, ld_dim); self.W_dec = nn.Linear(ld_dim, G, bias=True)
        nn.init.normal_(self.W_dec.weight, std=0.01); nn.init.zeros_(self.W_dec.bias)
        nn.init.orthogonal_(self.bottleneck.weight);  nn.init.zeros_(self.bottleneck.bias)
    def forward(self, h):
        z = self.bottleneck(h); return z, self.W_dec(z)
    def reconstruction_loss(self, h, x):
        _, xh = self.forward(h); return F.mse_loss(xh, x)
    def ld_regularization_loss(self, h):
        z, _ = self.forward(h); zc = z - z.mean(0)
        var_loss = -torch.log((zc ** 2).mean(0) + 1e-4).mean()
        cov = (zc.T @ zc) / zc.shape[0]
        return var_loss + cov[0, 1].pow(2) + cov[1, 0].pow(2)


class DictVelocityHead(nn.Module):
    """Calibrated (degradation/kinetics) transform of the committor-ORIENTED, chromatin-led
    disequilibrium. Because the committor orientation is baked into this input (see
    orient_disequilibrium), the head can represent a committor-consistent field — the fix for the
    earlier failure where it saw only the raw disequilibrium and could not follow the committor."""
    def __init__(self, G, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, G))
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    def forward(self, rna, disequil):
        return self.net(disequil) * self.scale
```

## Step 5: Losses — reconstruction, velocity, and the new committor anchor

**The fix lives here.** The earlier version applied the committor as a *separate direction loss*
that competed with `velocity_consistency_loss` for the same output vector. Because the velocity head
only ever saw the raw disequilibrium as input, it could not represent a committor-derived target, so
the two losses cancelled into a chaotic field and the committor weight was an impotent knob. The new
`orient_disequilibrium` instead **projects the disequilibrium into the committor's forward cone**
before it reaches the head: the component along the committor is sign-locked forward (killing the
"randomly reversed, meet-in-the-middle" artifact), the GRN/ATAC magnitude is preserved, and the
committor-orthogonal GRN structure is kept as an in-cone tilt (`BETA_PERP`). `velocity_consistency_loss`
now aligns the velocity with *that* oriented target, so the committor and the GRN cooperate instead
of fighting. `committor_consistency_loss` is retained only as a light reinforcement. Cells on a flat
committor (root/terminal plateaus) fall back to the raw disequilibrium and are skipped by the
reinforcement term.



```python
def masked_reconstruction_loss(grn, mix_weights, rna_batch, tf_batch, tf_indices_in_G, mask_frac=0.40):
    B, Gg = rna_batch.shape; device = rna_batch.device
    mask = torch.rand(B, Gg, device=device) < mask_frac
    tf_input_masked = tf_batch.clone(); batch_idx = torch.arange(B, device=device).unsqueeze(1)
    tf_input_masked[batch_idx, torch.arange(len(tf_indices_in_G), device=device)] *= \
        (~mask[batch_idx, tf_indices_in_G]).float()
    rna_pred = grn.predict_rna(tf_input_masked, mix_weights)
    if mask.sum() == 0: return rna_pred.mean() * 0.0
    return F.mse_loss(rna_pred[mask], rna_batch[mask])

def velocity_consistency_loss(velocity, disequil):
    """Align displacement with the chromatin-led disequilibrium; guard magnitude."""
    v = F.normalize(velocity, dim=1); d = F.normalize(disequil, dim=1)
    cos = (v * d).sum(1)
    mag = torch.abs(velocity.norm(dim=1) - disequil.norm(dim=1).detach()).mean()
    return (1.0 - cos).mean() + 0.1 * mag

def atac_direction_loss(velocity, gene_access_batch):
    """A closed locus should not be pushed up (cheap consistency guard)."""
    closed = 1.0 - gene_access_batch.to(velocity.device)
    return (closed * torch.relu(velocity)).mean()

def committor_consistency_loss(velocity, commit_dir, valid):
    """Direction-only: velocity should follow the committor's forward (increasing-q) direction.
    Invalid (flat-committor) cells are skipped so root/terminal plateaus don't inject noise.
    NOTE: with the committor now baked into the velocity's INPUT (orient_disequilibrium below),
    this term is only a light reinforcement — not the mechanism that steers the flow."""
    if valid.sum() == 0: return velocity.sum() * 0.0
    v = F.normalize(velocity[valid], dim=1); d = F.normalize(commit_dir[valid], dim=1)
    return (1.0 - (v * d).sum(1)).mean()


def orient_disequilibrium(disequil, commit_dir, valid, beta=0.5):
    """Committor-oriented arrow of time — the fix that makes the committor actually steer.

    The raw ATAC→RNA / GRN disequilibrium carries the gene program and per-cell magnitude, but its
    SIGN is locally ambiguous (az−rz can point forward or backward per cell), which is what produces
    the "arrows randomly reversed to meet in the middle" failure. Here we DECOMPOSE the disequilibrium
    into its component along the committor's forward direction u and its orthogonal remainder, then
    rebuild a strictly-forward vector:

        oriented_i = ||d_i|| · normalize( u_i + beta · ĥ(d_i^perp) )

    so the arrow is guaranteed to have a positive committor component (no backflow / no reversal),
    keeps the GRN/ATAC speed ||d_i||, and is tilted inside the forward cone by the committor-
    orthogonal, GRN-derived structure (branch selection the scalar committor cannot express).
    Cells with a flat/invalid committor keep the raw disequilibrium.

    Feeding THIS as the velocity head's input (instead of the raw disequilibrium) is the key change:
    previously the head saw only `disequil`, so it literally could not represent a committor-derived
    target, and the committor loss just fought the reconstruction-aligned velocity loss to a chaotic
    draw — raising the committor weight only added noise. Now the committor orientation is inside the
    head's input, so the two objectives agree and the field is clean."""
    u    = F.normalize(commit_dir, dim=1)                     # committor forward unit (per cell)
    dot  = (disequil * u).sum(1, keepdim=True)                # projection onto committor axis
    dperp = disequil - dot * u                                # GRN-orthogonal refinement
    dperp_hat = F.normalize(dperp, dim=1)
    mag  = disequil.norm(dim=1, keepdim=True)                 # keep GRN/ATAC per-cell speed
    oriented = F.normalize(u + beta * dperp_hat, dim=1) * mag # strictly-forward cone, GRN-tilted
    return torch.where(valid.view(-1, 1), oriented, disequil)
```

## Step 6: 2D velocity projection (drift-corrected, magnitude-preserving)

Correlation-kernel projection (scVelo logic) with the density/drift correction: subtract the mean
neighbor transition so an ambiguous velocity projects to ~0 instead of drifting toward the densest
neighbor. `search_space` should be a state-resolving space (pass `Z_dec`), while the plotted 2D
displacement always comes from the viz coordinates.



```python
def project_to_2d(V_pred_full, X_2d_vis, X_full, search_space=None, k=30, scale=1.0,
                  drift_correct=True, density_k=15, adaptive_scale=True):
    if search_space is None: search_space = X_2d_vis
    N = V_pred_full.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k).fit(search_space); _, idx = nbrs.kneighbors(search_space)
    d_high = X_full[idx] - X_full[:, None, :]
    d_low  = X_2d_vis[idx] - X_2d_vis[:, None, :]
    v_norm = np.linalg.norm(V_pred_full, axis=1, keepdims=True) + 1e-8
    d_norm = np.linalg.norm(d_high, axis=2) + 1e-8
    sims   = np.einsum('ng,nkg->nk', V_pred_full, d_high) / (d_norm * v_norm)
    sims   = sims - sims.mean(axis=1, keepdims=True)
    if adaptive_scale:
        dk = NearestNeighbors(n_neighbors=density_k).fit(X_2d_vis); dd, _ = dk.kneighbors(X_2d_vis)
        dens = 1.0 / (dd[:, 1:].mean(1) + 1e-8); lo, hi = dens.min(), dens.max()
        dens = 1.0 + 2.0 * (dens - lo) / (hi - lo + 1e-8)
    else:
        dens = np.ones(N, dtype=np.float32)
    w = np.exp(sims * (scale * dens)[:, None]); w = w / (w.sum(1, keepdims=True) + 1e-8)
    d_low_unit = d_low / (np.linalg.norm(d_low, axis=2, keepdims=True) + 1e-8)
    if drift_correct: w = w - 1.0 / k
    V_emb2 = np.einsum('nk,nkd->nd', w, d_low_unit).astype(np.float32)
    speed = np.linalg.norm(V_pred_full, axis=1); speed = speed / (np.percentile(speed, 95) + 1e-8)
    V_emb2 *= speed[:, None]
    V_emb2[np.linalg.norm(V_pred_full, axis=1) < 1e-8] = 0.0
    return V_emb2
```

## Step 7: Training a single dataset (COMET + committor loss)

Phase 1 = GRN + encoder warmup (masked reconstruction + geometric anchor). Phase 2 builds the
committor-**oriented** ATAC-led velocity: the disequilibrium is passed through `orient_disequilibrium`
(forward-cone projection onto the committor) before the velocity head sees it, so the flow direction
is committor-locked while its gene program and magnitude stay GRN-derived. Temperature is fixed at
1.0. The geometric anchor `KL(q‖w)` is annealed down (never to zero). The committor *loss*
(`COMMIT_LOSS_W · |trust| · COMMIT_W_SCALE[name]`) is now only a light reinforcement — the steering
is done by the orientation, not by cranking this weight, which is precisely why the old high-weight
approach could not fix ma2020/aml89.

Two efficiency behaviours:
- **Resume/skip.** If a finished run (`V_pred.npy` …) already exists in the dataset's output dir and
  `FORCE_RERUN` is off, the saved arrays are loaded and training is skipped entirely — reruns of the
  notebook cost nothing for datasets already done.
- **Per-epoch cell budget.** Each epoch draws at most `TARGET_CELLS_PER_EPOCH / BATCH_SIZE` random
  minibatches, so per-epoch cost no longer grows with #cells (the dominant `O(D·B·G·T)` atom einsum
  fires a fixed number of times per epoch). It is still plain unbiased minibatch SGD — the model and
  losses are untouched, only the count of stochastic updates per epoch is capped.



```python
W1 = dict(MASK=2.0, SPARSE=5e-3, ACCESS=0.1, DIVERSE=10.0, ANCHOR=1.0, LD=0.5, LD_REG=0.1)
W2 = dict(MASK=1.0, SPARSE=5e-3, ACCESS=0.1, DIVERSE=5.0, ANCHOR=0.5,
          VEL=3.0, ATAC=1.0, LD=0.3, LD_REG=0.05)

def _load_cached_run(out_dir):
    """Rebuild the results dict from a finished run's saved arrays (no model objects needed
    downstream). Returns None if the run is incomplete."""
    need = ["V_pred.npy", "mix_weights.npy", "A_learned.npy", "A_dict.npy",
            "committor_q.npy", "root_mask.npy", "sink_mask.npy", "phi.npy"]
    if not all(os.path.exists(os.path.join(out_dir, f)) for f in need):
        return None
    L = lambda f: np.load(os.path.join(out_dir, f))
    return dict(V_pred=L("V_pred.npy"), mix=L("mix_weights.npy"), A_learned=L("A_learned.npy"),
                A_dict=L("A_dict.npy"), q_commit=L("committor_q.npy"),
                root_mask=L("root_mask.npy").astype(bool), sink_mask=L("sink_mask.npy").astype(bool),
                phi=L("phi.npy"), grn=None, cell_encoder=None, vel_head=None)


def run_comet(b, out_dir):
    """Train the full COMET model on one bundle; save arrays; return a results dict."""
    os.makedirs(out_dir, exist_ok=True)

    # ---- resume: skip datasets that already finished ----
    if not FORCE_RERUN:
        cached = _load_cached_run(out_dir)
        if cached is not None:
            print(f"\n::: {b.name} ::: found finished run in {out_dir} -> loading & skipping training "
                  f"(set FORCE_RERUN=True to retrain).")
            return cached

    trust    = float(ENTROPY_ROOT_TRUST.get(b.name, 1.0))
    commit_w = COMMIT_LOSS_W * abs(trust) * float(COMMIT_W_SCALE.get(b.name, 1.0))
    beta     = float(BETA_PERP.get(b.name, 0.5))

    # ---- pick the LATENT STATE SPACE the trajectory is built in ----------------
    # This is the only embedding choice that changes the gene-space velocity: committor graph,
    # archetypes, geometric targets and the projection search space all live here.
    space = STATE_SPACE.get(b.name, "decipher")
    if space == "pca":
        Zst = PCA(n_components=min(30, b.G - 1, b.N - 1),
                  random_state=RANDOM_STATE).fit_transform(zscore_cols(b.X)).astype(np.float32)
    else:
        Zst = b.Z_dec
    # first-moment denoiser in the state space (cleans the noisy per-cell ATAC lead; k larger for
    # the paired ma2020 dataset). Built once and reused for the lead and the final velocity.
    mknn  = int(VEL_MOMENT_KNN.get(b.name, K_MOMENT))
    S_mom = moment_operator(Zst, mknn)
    print(f"\n::: {b.name} :::  (entropy_trust={trust:+.2f}, beta_perp={beta:.2f}, "
          f"state_space={space}, moment_knn={mknn}, committor_reinf_w={commit_w:.2f})")

    # ---- shared lead signals + committor field (built once, in the state space) ----
    # Pool the z-scored ATAC/RNA lead over state-space neighborhoods before differencing — this is
    # the first-moment denoising that removes the per-cell jitter (the ma2020 messiness).
    atac_z = np.asarray(S_mom @ zscore_cols(b.ga_matched), dtype=np.float32)
    rna_z  = np.asarray(S_mom @ zscore_cols(b.X),          dtype=np.float32)
    atac_z_t = torch.tensor(atac_z, dtype=torch.float32)
    rna_z_t  = torch.tensor(rna_z,  dtype=torch.float32)

    q_commit, root_mask, sink_mask, phi = committor_field(b, entropy_trust=trust, Z=Zst)
    commit_dir, commit_valid = committor_gene_direction(b, q_commit, Z=Zst)
    commit_dir_t = torch.tensor(commit_dir, dtype=torch.float32)
    commit_valid_t = torch.tensor(commit_valid, dtype=torch.bool)
    print(f"  committor: root={int(root_mask.sum())} cells, sinks={int(sink_mask.sum())}, "
          f"valid-direction cells={int(commit_valid.sum())}/{b.N}")

    # ---- archetypes + geometric soft mixing targets (in the state space) ----
    arch_idx = select_archetypes_fps(Zst, D_ATOMS, seed=0)
    q_target = geometric_soft_targets(Zst, arch_idx, tau_scale=GEO_TAU_SCALE)

    # ---- ATAC accessibility priors ----
    G, T = b.G, b.T
    TF_access = np.zeros(T, np.float32)
    for i, tf in enumerate(b.A_tfs):
        if tf in b.sub_g2i: TF_access[i] = float(b.ga_matched[:, b.sub_g2i[tf]].mean())
    lo, hi = TF_access.min(), TF_access.max(); TF_access = (TF_access - lo) / (hi - lo + 1e-8)
    TF_access_t   = torch.tensor(TF_access, dtype=torch.float32)
    gene_access_t = torch.tensor(b.gene_access, dtype=torch.float32)

    # ---- tensors + loader ----
    X_t    = torch.tensor(b.X,    dtype=torch.float32)
    X_TF_t = torch.tensor(b.X_TF, dtype=torch.float32)
    q_t    = torch.tensor(q_target, dtype=torch.float32)
    cell_idx_t = torch.arange(b.N)
    tf_indices_in_G = torch.tensor(b.tf_indices_in_G, dtype=torch.long)
    dataset = TensorDataset(X_t, X_TF_t, cell_idx_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=RandomSampler(dataset, replacement=False), drop_last=True)

    # ---- models ----
    grn = GRNDictionary(G=G, T=T, M=b.M, D=D_ATOMS)
    grn.TF_access = TF_access_t.clone(); grn.mean_tf = torch.tensor(b.X_TF.mean(0), dtype=torch.float32)
    grn.init_from_atoms(init_atoms_from_soft_targets(b.X, q_target, b.M, b.A_tfs))
    cell_encoder = CellStateEncoder(G=G, D=D_ATOMS); cell_encoder.init_from_soft_targets(b.X, q_target, n_epochs=40)
    vel_head = DictVelocityHead(G=G); ld_decoder = LinearDecoder(hidden=64, G=G, ld_dim=2)

    opt_grn = torch.optim.AdamW(
        list(grn.parameters()) + list(cell_encoder.parameters()) + list(ld_decoder.parameters()), lr=5e-4)
    opt_vel = torch.optim.AdamW(vel_head.parameters(), lr=2e-4)
    sch_grn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_grn, T_max=N_EPOCHS, eta_min=1e-5)
    sch_vel = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vel, T_max=N_EPOCHS - PHASE1_EPOCHS, eta_min=1e-5)

    # cap minibatches/epoch so per-epoch cost is independent of #cells (unbiased SGD subsampling)
    n_batches = len(loader)
    steps_per_epoch = n_batches if TARGET_CELLS_PER_EPOCH is None \
        else max(1, min(n_batches, int(np.ceil(TARGET_CELLS_PER_EPOCH / BATCH_SIZE))))
    print(f"  {n_batches} batches/epoch available; using {steps_per_epoch}/epoch "
          f"({steps_per_epoch*BATCH_SIZE} cells/epoch of {b.N}).")

    loss_history = []
    for epoch in range(N_EPOCHS):
        phase = 1 if epoch < PHASE1_EPOCHS else 2
        if epoch == PHASE1_EPOCHS: print("  -- Phase 2 (joint velocity + committor) --")
        anchor_scale = max(0.2, 1.0 - 0.8 * epoch / N_EPOCHS)
        grn.train(); cell_encoder.train(); vel_head.train(phase == 2)
        for p in vel_head.parameters(): p.requires_grad_(phase == 2)

        epoch_losses = []
        for step, (rna_b, tf_b, bidx) in enumerate(loader):
            if step >= steps_per_epoch: break                  # fresh random subset each epoch
            mix = cell_encoder(rna_b, temperature=1.0)
            l_mask    = masked_reconstruction_loss(grn, mix, rna_b, tf_b, tf_indices_in_G, 0.40)
            l_sparse  = grn.sparsity_loss(); l_access = grn.tf_access_consistency_loss()
            l_diverse = grn.diversity_loss()
            q_b       = q_t[bidx]
            l_anchor  = (q_b * (torch.log(q_b + 1e-8) - torch.log(mix + 1e-8))).sum(1).mean()
            h_b       = cell_encoder.encoder(rna_b)
            l_ld      = ld_decoder.reconstruction_loss(h_b, rna_b)
            l_ld_reg  = ld_decoder.ld_regularization_loss(h_b)

            if phase == 1:
                loss_grn = (W1['MASK']*l_mask + W1['SPARSE']*l_sparse + W1['ACCESS']*l_access +
                            W1['DIVERSE']*l_diverse + W1['ANCHOR']*anchor_scale*l_anchor +
                            W1['LD']*l_ld + W1['LD_REG']*l_ld_reg)
                opt_grn.zero_grad(); loss_grn.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(grn.parameters()) + list(cell_encoder.parameters()) + list(ld_decoder.parameters()), 1.0)
                opt_grn.step(); epoch_losses.append(float(loss_grn.item()))
            else:
                # ATAC->RNA led disequilibrium (the gene program + per-cell magnitude)
                az, rz     = atac_z_t[bidx], rna_z_t[bidx]
                chrom_lead = az - rz
                tf_lead    = az[:, tf_indices_in_G] - rz[:, tf_indices_in_G]
                prod       = grn.production_from_lead(tf_lead, mix.detach())
                disequil   = chrom_lead + LAMBDA_GRN * prod
                # committor-orient it: lock the forward sign, keep GRN magnitude + orthogonal tilt.
                # This is what the head sees, so the committor can actually steer the flow.
                disequil_or = orient_disequilibrium(
                    disequil, commit_dir_t[bidx], commit_valid_t[bidx], beta=beta)
                velocity   = vel_head(rna_b, disequil_or)

                l_vel    = velocity_consistency_loss(velocity, disequil_or.detach())
                ga_b     = gene_access_t if gene_access_t.dim() == 1 else gene_access_t[bidx]
                l_atac   = atac_direction_loss(velocity, ga_b)
                l_commit = committor_consistency_loss(velocity, commit_dir_t[bidx], commit_valid_t[bidx])

                loss_vel = W2['VEL']*l_vel + W2['ATAC']*l_atac + commit_w*l_commit
                loss_grn = (W2['MASK']*l_mask + W2['SPARSE']*l_sparse + W2['ACCESS']*l_access +
                            W2['DIVERSE']*l_diverse + W2['ANCHOR']*anchor_scale*l_anchor +
                            W2['LD']*l_ld + W2['LD_REG']*l_ld_reg)

                opt_grn.zero_grad(); opt_vel.zero_grad()
                loss_grn.backward(retain_graph=True); loss_vel.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(grn.parameters()) + list(cell_encoder.parameters()) + list(ld_decoder.parameters()), 1.0)
                torch.nn.utils.clip_grad_norm_(vel_head.parameters(), 1.0)
                opt_grn.step(); opt_vel.step()
                epoch_losses.append(float((loss_grn + loss_vel).item()))

        sch_grn.step()
        if phase == 2: sch_vel.step()
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            avg = float(np.mean(epoch_losses)); loss_history.append(avg)
            print(f"  [P{phase}] epoch {epoch:03d}  loss={avg:.4f}  anchor_scale={anchor_scale:.2f}")

    # ---- inference + save ----
    grn.eval(); vel_head.eval(); cell_encoder.eval()
    with torch.no_grad():
        mix_full      = cell_encoder(X_t, temperature=1.0)
        az_all        = atac_z_t - rna_z_t
        tf_lead_all   = az_all[:, tf_indices_in_G]
        disequil_full = az_all + LAMBDA_GRN * grn.production_from_lead(tf_lead_all, mix_full)
        disequil_full = orient_disequilibrium(disequil_full, commit_dir_t, commit_valid_t, beta=beta)
        V_pred        = vel_head(X_t, disequil_full).cpu().numpy()
        steady_full   = grn.predict_rna(X_TF_t, mix_full).cpu().numpy()
        A_dict_np     = grn.A_dict.cpu().numpy(); A_base_np = grn.A_base.cpu().numpy()
        A_learned     = grn.effective_A(mix_full.mean(0)).cpu().numpy()
        all_mix       = mix_full.cpu().numpy()

    # first-moment pooling of the final gene-space velocity (denoise before projection). The arrow
    # of time is committor-locked, so this cannot reverse the flow — it only removes per-cell jitter.
    if SMOOTH_VELOCITY:
        V_pred = np.asarray(S_mom @ V_pred, dtype=np.float32)

    np.save(f"{out_dir}/V_pred.npy", V_pred);          np.save(f"{out_dir}/A_dict.npy", A_dict_np)
    np.save(f"{out_dir}/A_base.npy", A_base_np);        np.save(f"{out_dir}/A_learned.npy", A_learned)
    np.save(f"{out_dir}/mix_weights.npy", all_mix);     np.save(f"{out_dir}/steady_state.npy", steady_full)
    np.save(f"{out_dir}/A_genes.npy", b.A_genes);       np.save(f"{out_dir}/A_tfs.npy", b.A_tfs)
    np.save(f"{out_dir}/committor_q.npy", q_commit);    np.save(f"{out_dir}/loss_history.npy", np.array(loss_history))
    np.save(f"{out_dir}/root_mask.npy", root_mask);     np.save(f"{out_dir}/sink_mask.npy", sink_mask)
    np.save(f"{out_dir}/phi.npy", phi)
    print(f"  saved -> {out_dir}")

    return dict(V_pred=V_pred, mix=all_mix, A_learned=A_learned, A_dict=A_dict_np,
                q_commit=q_commit, root_mask=root_mask, sink_mask=sink_mask, phi=phi,
                grn=grn, cell_encoder=cell_encoder, vel_head=vel_head)
```

## Step 8: Run selected datasets



```python
RESULTS = {}
for name in RUN_DATASETS:
    b = DATASETS[name]
    RESULTS[name] = run_comet(b, out_dir=os.path.join(OUT_DIR, name))
print("\nTraining complete for:", list(RESULTS))
```

    
    ::: hema :::  (entropy_trust=+1.00, w_scale=1.0 -> committor weight=2.00)
      committor: root=104 cells, sinks=62, valid-direction cells=1031/1035
      GRN atoms initialized from archetype soft-LFC profiles (base starts at 0).
      Pre-training encoder to match geometric soft targets...
        encoder pretrain epoch 0: KL=0.2489
        encoder pretrain epoch 10: KL=0.0131
        encoder pretrain epoch 20: KL=0.0043
        encoder pretrain epoch 30: KL=0.0015
      2 batches/epoch available; using 2/epoch (1024 cells/epoch of 1035).
      [P1] epoch 000  loss=4.6687  anchor_scale=1.00
      [P1] epoch 020  loss=2.2627  anchor_scale=0.80
      -- Phase 2 (joint velocity + committor) --
      [P2] epoch 040  loss=28.2053  anchor_scale=0.60
      [P2] epoch 060  loss=26.8172  anchor_scale=0.40
      [P2] epoch 079  loss=26.5388  anchor_scale=0.21
      saved -> comet_out/hema
    
    ::: ma2020 :::  (entropy_trust=+1.00, w_scale=2.0 -> committor weight=4.00)
      committor: root=720 cells, sinks=431, valid-direction cells=7166/7197
      GRN atoms initialized from archetype soft-LFC profiles (base starts at 0).
      Pre-training encoder to match geometric soft targets...
        encoder pretrain epoch 0: KL=0.1085
        encoder pretrain epoch 10: KL=0.0018
        encoder pretrain epoch 20: KL=0.0008
        encoder pretrain epoch 30: KL=0.0003
      14 batches/epoch available; using 4/epoch (2048 cells/epoch of 7197).
      [P1] epoch 000  loss=4.9986  anchor_scale=1.00
      [P1] epoch 020  loss=1.0592  anchor_scale=0.80
      -- Phase 2 (joint velocity + committor) --
      [P2] epoch 040  loss=28.9625  anchor_scale=0.60
      [P2] epoch 060  loss=27.1425  anchor_scale=0.40
      [P2] epoch 079  loss=26.6107  anchor_scale=0.21
      saved -> comet_out/ma2020
    
    ::: aml89 :::  (entropy_trust=-0.60, w_scale=2.0 -> committor weight=2.40)
      committor: root=504 cells, sinks=300, valid-direction cells=4985/5000
      GRN atoms initialized from archetype soft-LFC profiles (base starts at 0).
      Pre-training encoder to match geometric soft targets...
        encoder pretrain epoch 0: KL=0.0591
        encoder pretrain epoch 10: KL=0.0009
        encoder pretrain epoch 20: KL=0.0004
        encoder pretrain epoch 30: KL=0.0002
      9 batches/epoch available; using 4/epoch (2048 cells/epoch of 5000).
      [P1] epoch 000  loss=6.9506  anchor_scale=1.00
      [P1] epoch 020  loss=2.3179  anchor_scale=0.80
      -- Phase 2 (joint velocity + committor) --
      [P2] epoch 040  loss=29.4228  anchor_scale=0.60
      [P2] epoch 060  loss=28.0647  anchor_scale=0.40
      [P2] epoch 079  loss=27.7249  anchor_scale=0.21
      saved -> comet_out/aml89
    
    Training complete for: ['hema', 'ma2020', 'aml89']


## Step 9: The velocity figure — clean_quiver on all four embeddings

The same learned per-cell velocity `V_pred`, projected onto each of the four `experiments7`
embeddings (`decipher2d`, `pca2d`, `umap_decipher`, `umap_pca`) and drawn with `clean_quiver`. For
each embedding the velocity is projected using that embedding's own kNN source space; the committor
root (cell-space mask, so it maps onto any embedding) is starred. One row per dataset, one column per
embedding (UMAP columns are dropped automatically if `umap` isn't installed).



```python
# Each dataset chooses its OWN list of display embeddings (DISPLAY_EMBEDDINGS in Config). The same
# gene-space velocity R["V_pred"] is projected onto each requested embedding — the projection is the
# only thing that changes between columns.
emb_order = ["decipher2d", "pca2d", "umap_decipher", "umap_pca"]

EMBEDDINGS, WANTED = {}, {}
for name in RUN_DATASETS:
    print(f"building embeddings: {name}")
    req = [e for e in DISPLAY_EMBEDDINGS.get(name, emb_order) if e in emb_order]
    EMBEDDINGS[name] = make_embeddings(DATASETS[name], wanted=req)
    # keep only those actually built (UMAP may be unavailable), in canonical order
    WANTED[name] = [e for e in req if e in EMBEDDINGS[name]]

nrow = len(RUN_DATASETS); ncol = max(len(WANTED[n]) for n in RUN_DATASETS)
fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow), squeeze=False)
for r, name in enumerate(RUN_DATASETS):
    b = DATASETS[name]; R = RESULTS[name]; cols = WANTED[name]
    for c in range(ncol):
        if c < len(cols):
            en = cols[c]; emb, src = EMBEDDINGS[name][en]
            V_proj2d = project_to_2d(R["V_pred"], emb, b.X, search_space=src, k=45, scale=0.4,
                                     drift_correct=True)
            clean_quiver(axes[r, c], V_proj2d, emb, b.cell_types,
                         title=f"{name} — {en}", root_mask=R["root_mask"])
        else:
            axes[r, c].axis("off")     # ragged rows: hide unused cells
plt.suptitle("COMET velocity (clean_quiver): committor-oriented, GRN-propagated ATAC lead — per-dataset embeddings",
             fontweight="bold", y=1.002)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/velocity_clean_quiver_all_embeddings.png", dpi=140, bbox_inches="tight"); plt.show()
```

    building embeddings: hema
    building embeddings: ma2020
    building embeddings: aml89



    
![png](output_24_1.png)
    


## Step 10: Diagnostics — committor field + GRN mixing weights

Left: the committor scalar `q` on the embedding (0 = root, starred; 1 = terminal). Right: per-atom
GRN mixing weights at the true inference temperature (1.0) — committed tips should be dark for one
atom, transition cells intermediate across ≥2 atoms.



```python
for name in RUN_DATASETS:
    b = DATASETS[name]; R = RESULTS[name]; W_soft = R["mix"]
    ent = -(W_soft * np.log(W_soft + 1e-8)).sum(1)
    print(f"\n[{name}] mean mixing weight/atom: {W_soft.mean(0).round(3)}  "
          f"| mean entropy {ent.mean():.3f}/{np.log(D_ATOMS):.3f}  "
          f"| committed (max>0.8): {(W_soft.max(1) > 0.8).mean():.1%}")

    ncol = D_ATOMS + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.4)); axes = np.atleast_1d(axes)
    sc0 = axes[0].scatter(b.V_dec2[:, 0], b.V_dec2[:, 1], c=R["q_commit"], cmap="viridis", s=8, rasterized=True)
    axes[0].scatter(*b.V_dec2[R["root_mask"]].mean(0), marker="*", s=380, c="lime", edgecolors="k")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046); axes[0].set_title(f"{name}: committor q (0=root)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for d in range(D_ATOMS):
        ax = axes[d + 1]
        scv = ax.scatter(b.V_dec2[:, 0], b.V_dec2[:, 1], c=W_soft[:, d], cmap='YlOrRd', s=8, vmin=0, vmax=1)
        plt.colorbar(scv, ax=ax, fraction=0.046)
        top_tf = b.A_tfs[np.abs(R["A_dict"][d]).sum(0).argmax()]
        ax.set_title(f"atom {d} (top TF: {top_tf})", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"{name}: committor field + per-cell GRN mixing (T=1.0)", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{name}/diagnostics.png", dpi=150, bbox_inches="tight"); plt.show()
```

## Step 11: Top TF targets (sanity check on the learned population GRN)



```python
def top_tf_targets(A_learned, A_tfs, A_genes, tf_name, n=10):
    if tf_name not in A_tfs: raise ValueError(f"{tf_name} not in A_tfs")
    col = A_learned[:, np.where(A_tfs == tf_name)[0][0]]
    top = np.argsort(-np.abs(col))[:n]
    return [(A_genes[i], float(col[i])) for i in top]

for name in RUN_DATASETS:
    b = DATASETS[name]; A_learned = RESULTS[name]["A_learned"]
    tf0 = b.A_tfs[0]
    print(f"\n[{name}] top 10 targets of {tf0}:")
    for g, w in top_tf_targets(A_learned, b.A_tfs, b.A_genes, tf0):
        print(f"  {g:20s} weight={w:+.4f}")
```
