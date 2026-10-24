"""All metric computation for benchmark_figures_new_metrics.ipynb (COMET vs scKINETICS vs CellRank).
The notebook imports this module, calls the functions below per dataset and does all plotting itself.
Self-contained copy of recomb_bench/pipelines/new_metrics.py (code depends on nothing but standard packages;
reads data and caches from the absolute paths below).

COMET's field is the FULL COMET field: committor direction + chromatin lead (utils/trajectory.final_trajectory;
results_final/<ds>/V_pred.npy; the lead's ATAC-RNA gate is off on aml89 and ma2020, as in COMET itself) + the
regulatory-consistency GRN tilt (utils/trajectory.regulatory_consistency_tilt, λ = 0.5, from the saved GRN; computed
by grn_displacement/scripts/run_final.py as V_A in grn_displacement/tables_final/cache_<ds>.npz; nothing retrained).

Metrics
  forward transport   every method's gene-space velocity -> P_ij ∝ exp(β cos(v_i, x_j − x_i)) on a shared kNN graph,
                      T = 0.8P + 0.2 uniform; after k steps, mass on descendant clusters minus mass on ancestor
                      clusters of the known cell-type tree; directional part ½[f(v) − f(−v)]; per cell, then mean over
                      clusters. Graph spaces: Decipher latent and gene PCA (independent of COMET).
  pairwise AUC        P(pt_descendant > pt_ancestor) over every ancestor -> descendant pair of the tree, mean over pairs.
Pseudotimes
  COMET pseudotime            the full COMET field converted to a pseudotime the scVelo way: scVelo velocity graph
                              (cosine of v_i with x_j − x_i, Decipher-latent kNN) -> scVelo transition matrix -> its
                              scalar potential (graph-flow / HodgeRank; the same conversion as "scKINETICS (graph flow)").
                              how="scvelo_vpt" gives scVelo's own velocity_pseudotime instead (eigenvector-picked roots;
                              unstable on COMET's field: pair AUC .52 on aml89, .68 on paul15).
  scKINETICS, CellRank        best of each family's variants by pair AUC (results/per_cell/<ds>.csv; pipelines/score_all.py)
Uncertainty: 95% percentile bootstrap over cells (stratified within clusters for transport); random-field null =
N_NULL i.i.d. Gaussian gene-space fields through the identical transport pipeline.
"""
import os, sys, glob, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

RB = "/home/sc5625/cll_project/decipher_atac_project/COMET/trajectory_benchmarking/recomb_bench"
TB = os.path.dirname(RB); BD = "/home/sc5625/cll_project/decipher_atac_project/benchmark_data"
RF = "/home/sc5625/cll_project/decipher_atac_project/COMET/comet/results_final"
TILT = f"{TB}/grn_displacement/tables_final"
VEL_CACHE = f"{RB}/results_new_metrics/per_cell"          # each method's gene-space velocity (built once)

K_NN, BETA, W_CONN, K_STEPS = 30, 10.0, 0.2, 30
N_BOOT, N_BOOT_AUC, SEED, N_NULL = 1000, 300, 0, 20
METHODS = ["COMET", "scKINETICS", "CellRank"]
SPACES = ["Decipher latent", "gene PCA"]
PT_VARIANTS = {"scKINETICS": ["scKINETICS (graph flow)", "scKINETICS (2D arrows)"],
               "CellRank": ["CellRank", "scVelo (stochastic)", "scVelo (dynamical)"]}
TREES = {  # parent -> children (benchmark cell-type labels); see metric_study/README.md for the choices
    "hema": {"HSC": ["MPP"], "MPP": ["CMP", "MLP"], "CMP": ["MEP", "GMP"], "MLP": ["GMP"]},
    "aml89": {"immature": ["blast0"], "blast0": ["blast1"], "blast1": ["blast2"], "blast2": ["blast3"]},
    "paul15": {"MEP": ["Erythroids", "Megakaryocytes"], "GMP": ["late_GMP"], "late_GMP": ["Granulocytes", "Monocytes"]},
    "ma2020": {"TAC-1": ["TAC-2"], "TAC-2": ["IRS", "Hair Shaft-Cuticle/Cortex", "Medulla"]},
    "nestorowa": {"LT-HSC": ["MPP"], "MPP": ["CMP", "LMPP"], "CMP": ["MEP", "GMP"]},
}


def load_obs(ds):
    """ground-truth cell annotations in COMET results_final cell order (copied from recomb_bench/feasibility/gt.py):
    obs DataFrame indexed by cell id with cell_type, stage and, where available, facs_pseudotime / beeline_pt_*"""
    import anndata as ad
    from sklearn.model_selection import train_test_split
    B = "/home/sc5625/cll_project/decipher_atac_project/"
    if ds == "hema":
        a = ad.read_h5ad(B + "COMET/trajectory_benchmarking/DATA/velten_rna.h5ad", backed="r")
        obs = a.obs[["facs_pseudotime"]].copy()
        ct = pd.read_csv(B + "COMET/trajectory_benchmarking/RESULTS/bench/celltypes.csv", index_col=0)["celltype"]
        obs["cell_type"] = ct.reindex(obs.index).astype(str).values
        smap = {"HSC": 0, "MPP": 1, "CMP": 2, "MLP": 2, "MEP": 3, "GMP": 3}
    elif ds == "aml89":
        a = ad.read_h5ad(B + "aml89/08H089_allsorted_cleaned.h5", backed="r")
        strat = a.obs["blast_cell_type_08H089"].values
        _, keep = train_test_split(np.arange(a.n_obs), test_size=5000, stratify=strat, random_state=42)
        obs = a.obs.iloc[np.sort(keep)][["blast_cell_type_08H089", "sample", "dm_0_rank"]].copy()
        obs["cell_type"] = obs.pop("blast_cell_type_08H089").astype(str)
        smap = {"immature": 0, "blast0": 1, "blast1": 2, "blast2": 3, "blast3": 4}
    elif ds == "paul15":
        a = ad.read_h5ad(B + "COMET/paul2015data_clean/Paul_etal_15_2.h5ad", backed="r")
        obs = a.obs[["cell_type"]].astype(str).copy()
        smap = {"MEP": 0, "GMP": 0, "late_GMP": 1, "Erythroids": 2, "Megakaryocytes": 2,
                "Granulocytes": 2, "Monocytes": 2}
    elif ds == "nestorowa":
        a = ad.read_h5ad(B + "nestorowa_mhsc_all/rna.h5ad", backed="r")
        obs = a.obs[["cell_type", "beeline_pt_E", "beeline_pt_GM", "beeline_pt_L"]].copy()
        obs["cell_type"] = obs.cell_type.astype(str)
        smap = {"LT-HSC": 0, "MPP": 1, "CMP": 2, "LMPP": 2, "GMP": 3, "MEP": 3}
    elif ds == "ma2020":
        a = ad.read_h5ad(B + "decipher-atac/paired/Ma-2020-RNA.h5ad", backed="r")
        KEEP = ["TAC-1", "TAC-2", "IRS", "Hair Shaft-Cuticle/Cortex", "Medulla"]
        obs = a.obs[a.obs.cell_type.astype(str).isin(KEEP).values][["cell_type"]].astype(str).copy()
        smap = {"TAC-1": 0, "TAC-2": 1, "IRS": 2, "Hair Shaft-Cuticle/Cortex": 2, "Medulla": 2}
    obs["stage"] = obs.cell_type.map(smap).astype(float)
    assert len(obs) == len(np.load(f"{RF}/{ds}/committor_q.npy")), ds
    return obs


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def zs(A): A = np.asarray(A, np.float32); return (A - A.mean(0)) / (A.std(0) + 1e-8)
def knn(Z, k): return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]


def tree_info(tree):
    nodes = sorted({p for p in tree} | {c for cs in tree.values() for c in cs})
    desc = {n: set() for n in nodes}
    def walk(n):
        for c in tree.get(n, []): desc[n] |= {c} | walk(c)
        return desc[n]
    for n in nodes: walk(n)
    anc = {n: {m for m in nodes if n in desc[m]} for n in nodes}
    leaves = [n for n in nodes if not tree.get(n)]
    pairs = [(a, d) for a in nodes for d in sorted(desc[a])]
    return dict(nodes=nodes, desc=desc, anc=anc, leaves=leaves, pairs=pairs)


# ------------------------------------------------------------------------------------------------ kernels
def velocity_kernel(idx, Xz, Vz, beta=BETA):
    """P_ij ∝ exp(β cos(v_i, x_j − x_i)) over the k neighbours idx[i]; sparse row-stochastic"""
    N, k = idx.shape; P = np.zeros((N, k), np.float32); chunk = max(16, int(2e7 / (k * Xz.shape[1])))
    for s in range(0, N, chunk):
        e = min(s + chunk, N); Dx = Xz[idx[s:e]] - Xz[s:e, None, :]
        c = np.einsum("ng,nkg->nk", Vz[s:e], Dx) / (np.linalg.norm(Vz[s:e], axis=1)[:, None] * np.linalg.norm(Dx, axis=2) + 1e-8)
        p = np.exp(beta * (c - c.max(1, keepdims=True))); P[s:e] = p / p.sum(1, keepdims=True)
    return sp.csr_matrix((P.ravel(), (np.repeat(np.arange(N), k), idx.ravel())), shape=(N, N))


def uniform_kernel(idx):
    N, k = idx.shape
    return sp.csr_matrix((np.full(N * k, 1.0 / k), (np.repeat(np.arange(N), k), idx.ravel())), shape=(N, N))


def full_T(idx, Xz, Vz):
    return sp.csr_matrix((1 - W_CONN) * velocity_kernel(idx, Xz, Vz) + W_CONN * uniform_kernel(idx))


def project2d(T, idx, emb):
    """CellRank/scVelo embedding projection v_i = Σ_j T_ij u_ij − mean_{j∈kNN(i)} u_ij, u_ij = unit(e_j − e_i)"""
    T = sp.csr_matrix(T); V = np.zeros((T.shape[0], 2))
    for i in range(T.shape[0]):
        j = T.indices[T.indptr[i]:T.indptr[i + 1]]; p = T.data[T.indptr[i]:T.indptr[i + 1]]
        u = emb[j] - emb[i]; u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        un = emb[idx[i]] - emb[i]; un /= np.linalg.norm(un, axis=1, keepdims=True) + 1e-12
        V[i] = ((p / p.sum())[:, None] * u).sum(0) - un.mean(0)
    return V


# ------------------------------------------------------------------------------------------------ pseudotime
def graph_flow_potential(T):
    """scalar potential of the net flow of a transition matrix (HodgeRank): least squares over MUTUAL kNN edges of
    (φ_j − φ_i) ≈ (T_ij − T_ji)/(T_ij + T_ji); min-max scaled. Identical to pipelines/score_all.sckinetics_pt."""
    T = sp.csr_matrix(T); A = (T > 0).astype(np.int8); S = sp.triu(A.multiply(A.T).tocsr(), k=1).tocoo()
    tij = np.asarray(T[S.row, S.col]).ravel(); tji = np.asarray(T[S.col, S.row]).ravel()
    g = (tij - tji) / (tij + tji + 1e-12); m = len(g)
    B = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[S.row, S.col])),
                      shape=(m, T.shape[0]))
    phi = lsqr(B, g, atol=1e-9, btol=1e-9, iter_lim=4000)[0]
    return (phi - phi.min()) / (np.ptp(phi) + 1e-12)


def velocity_pseudotime(Xz, V, Z, how="potential", n_neighbors=K_NN):
    """gene-space field -> pseudotime with scVelo: velocity_graph on Ms = Xz, neighbours on Z;
    how = 'potential' (graph-flow potential of scVelo's transition matrix) or 'scvelo_vpt' (scv.tl.velocity_pseudotime)"""
    import anndata as ad, scanpy as sc, scvelo as scv
    scv.settings.verbosity = 0
    a = ad.AnnData(np.asarray(Xz, np.float32)); a.layers["Ms"] = a.X.copy(); a.layers["velocity"] = np.asarray(V, np.float32)
    a.obsm["X_rep"] = np.asarray(Z, np.float32)
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X_rep", random_state=0)
    scv.tl.velocity_graph(a, vkey="velocity", xkey="Ms", n_jobs=8)
    if how == "scvelo_vpt":
        scv.tl.velocity_pseudotime(a, vkey="velocity"); return a.obs["velocity_pseudotime"].values.astype(float)
    return graph_flow_potential(scv.utils.get_transition_matrix(a, vkey="velocity"))


# ------------------------------------------------------------------------------------------------ metrics
def transport_curves(T, lab, ti, kmax=K_STEPS):
    """per-cell f_i(k) = m_i^k(Desc c(i)) − m_i^k(Anc c(i)), k = 1..kmax; NaN outside non-leaf tree clusters"""
    nodes = [c for c in ti["nodes"] if c not in ti["leaves"]]; N = T.shape[0]
    ind = {c: (lab == c).astype(np.float64) for c in ti["nodes"]}
    M = np.stack([sum((ind[d] for d in ti["desc"][c]), np.zeros(N)) - sum((ind[a] for a in ti["anc"][c]), np.zeros(N))
                  for c in nodes], 1)
    out = np.full((kmax, N), np.nan); cur = M
    col = {c: j for j, c in enumerate(nodes)}; cell_col = np.array([col.get(l, -1) for l in lab]); has = cell_col >= 0
    for k in range(kmax):
        cur = T @ cur; out[k, has] = cur[np.where(has)[0], cell_col[has]]
    return out


def transport(idx, Xz, Vz, lab, ti):
    """directional forward transport per cell and step, ½[f(T(v)) − f(T(−v))]; (K_STEPS, N)"""
    return 0.5 * (transport_curves(full_T(idx, Xz, Vz), lab, ti) - transport_curves(full_T(idx, Xz, -Vz), lab, ti))


def cluster_balanced(v, lab, rng=None):
    """mean over clusters of the mean over cells (rng: resample cells within each cluster)"""
    ms = []
    for c in np.unique(lab[np.isfinite(v)]):
        x = v[(lab == c) & np.isfinite(v)]
        if rng is not None: x = x[rng.integers(0, len(x), len(x))]
        ms.append(x.mean())
    return float(np.mean(ms))


def boot(fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); b = np.array([fn(rng) for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def transport_score(f, lab):
    """(value, lo, hi) of the cluster-balanced transport at the last step"""
    return (cluster_balanced(f[-1], lab), *boot(lambda r: cluster_balanced(f[-1], lab, r)))


def auc(lo, hi):
    r = rankdata(np.r_[lo, hi]); n0, n1 = len(lo), len(hi)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def pair_aucs(pt, lab, ti, rng=None):
    idx = {c: np.where(lab == c)[0] for c in ti["nodes"]}
    if rng is not None: idx = {c: i[rng.integers(0, len(i), len(i))] for c, i in idx.items()}
    return {(a, d): auc(pt[idx[a]], pt[idx[d]]) for a, d in ti["pairs"]}


def pair_auc_score(pt, lab, ti):
    """(value, lo, hi, per-pair dict)"""
    per = pair_aucs(pt, lab, ti)
    return (float(np.mean(list(per.values()))),
            *boot(lambda r: np.mean(list(pair_aucs(pt, lab, ti, r).values())), n=N_BOOT_AUC), per)


def transport_curve(f, lab):
    """cluster-balanced transport at every step; (K_STEPS,)"""
    return np.array([cluster_balanced(fk, lab) for fk in f])


def null_transport(D, space, n=N_NULL):
    """cluster-balanced transport curves of n i.i.d. Gaussian gene-space fields (COMET's genes and space); (n, K_STEPS)"""
    Xz, V = D["fields"]["COMET"]
    return np.stack([transport_curve(transport(D["knn"][space], Xz, np.random.default_rng(1000 + r).standard_normal(V.shape)
                                               .astype(np.float32), D["lab"], D["ti"]), D["lab"]) for r in range(n)])


def arrows_2d(D, method):
    """the 2D arrows drawn for a method: its Decipher-latent transition matrix projected onto the Decipher 2D embedding"""
    Xz, V = D["fields"][method]
    return project2d(full_T(D["knn"]["Decipher latent"], Xz, V), D["knn"]["Decipher latent"], D["emb"])


# ------------------------------------------------------------------------------------------------ inputs
def scvelo_velocity(ds, obs_index):
    """scVelo stochastic velocity exactly as pipelines/native_viz.velocity_adata; -> (cells, genes, Ms, velocity)"""
    import anndata as ad, scanpy as sc, scvelo as scv
    if ds == "ma2020":
        a = ad.read_h5ad(BD + "/ma2020_hf_figshare/hf_shareseq_rna.h5ad"); a.X = a.layers["spliced"].copy()
    else:
        paths = sorted(glob.glob(BD + "/aml89_velocity/loom_masked/*.loom")) if ds == "aml89" else [BD + "/paul15_velocity/loom_masked/paul15.loom"]
        parts = []
        for p in paths:
            x = ad.read_loom(p, sparse=True, obs_names="CellID", var_names="Gene")
            x.obs_names = [n.split(":", 1)[1].rstrip("x") for n in x.obs_names]; x.var_names_make_unique(); parts.append(x)
        a = ad.concat(parts, join="inner") if len(parts) > 1 else parts[0]
    cells = obs_index[obs_index.isin(a.obs_names)]; a = a[cells].copy()
    scv.pp.filter_genes(a, min_shared_counts=20); scv.pp.normalize_per_cell(a); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
    sc.pp.pca(a, n_comps=30, random_state=0); sc.pp.neighbors(a, n_neighbors=30, random_state=0)
    scv.pp.moments(a, n_pcs=30, n_neighbors=30); scv.tl.velocity(a, mode="stochastic")
    return (np.asarray(a.obs_names, str), np.asarray(a.var_names, str), np.asarray(a.layers["Ms"], np.float32),
            np.nan_to_num(np.asarray(a.layers["velocity"], np.float32)))


def comet_field(ds, base):
    """full COMET field (committor + chromatin lead + GRN tilt), unit per cell, in z-scored gene units"""
    c = np.load(f"{TILT}/cache_{ds}.npz")["V_A"].astype(np.float32)
    assert c.shape == base["V_pred"].shape
    return c


def load_methods(ds, obs, base):
    """{method: (cells, genes, X, V)} in each method's own expression space"""
    cache = f"{VEL_CACHE}/{ds}_velocities.npz"
    if os.path.exists(cache):
        m = np.load(cache, allow_pickle=True)["m"].item()
    else:
        bun = np.load(f"{TB}/bio_figures/cache/{ds}_bundle.npz", allow_pickle=True)
        m = {"COMET": (obs.index.values.astype(str), bun["genes"].astype(str), base["X"].astype(np.float32), None)}
        d = f"{TB}/RESULTS/sckinetics/{ds}"
        names = np.load(f"{d}/sckinetics_obs_names.npy", allow_pickle=True).astype(str)
        genes = np.load(f"{d}/sckinetics_genes.npy", allow_pickle=True).astype(str)
        V = np.load(f"{d}/sckinetics_velocity_genes.npy").astype(np.float32)
        X = pd.read_pickle(f"{d}/sckinetics_X_transformed.pkl").reindex(index=names, columns=genes).values.astype(np.float32)
        keep = np.isfinite(V).mean(0) >= 0.5
        m["scKINETICS"] = (names, genes[keep], np.nan_to_num(X[:, keep]), np.nan_to_num(V[:, keep]))
        if ds in ("aml89", "paul15", "ma2020"):
            m["CellRank"] = scvelo_velocity(ds, obs.index.astype(str))
        os.makedirs(VEL_CACHE, exist_ok=True); np.savez(cache, m=np.array(m, dtype=object))
    c, g, X, _ = m["COMET"]; m["COMET"] = (c, g, X, comet_field(ds, base))
    return m


def load_dataset(ds, comet_pt_how="potential"):
    """everything the metrics need, on the cells every method covers. Returns a dict D with
    rows, cells, lab (benchmark cell type), ti (tree), emb (Decipher 2D), knn {space: idx},
    fields {method: (Xz, Vz)} (each in its own z-scored expression space), pt {name: pseudotime}, pt_variant, root_mask"""
    base = np.load(f"{RB}/results/fields/{ds}_base.npz", allow_pickle=True); obs = load_obs(ds)
    pc = pd.read_csv(f"{RB}/results/per_cell/{ds}.csv", index_col=0)
    assert (pc.index.astype(str) == obs.index.astype(str)).all()
    meths = load_methods(ds, obs, base)
    cells_all = obs.index.values.astype(str); have = np.ones(len(cells_all), bool)
    for k, (c, *_r) in meths.items(): have &= np.isin(cells_all, c)
    ptcols = [v for k in meths for v in PT_VARIANTS.get(k, []) if v in pc]
    have &= pc[ptcols].notna().all(1).values
    rows = np.where(have)[0]; cells = cells_all[rows]
    lab = pc.cell_type.astype(str).values[rows]; ti = tree_info(TREES[ds])

    Zd = base["Z_dec"].astype(np.float32); Xc = zs(base["X"])
    D = dict(ds=ds, rows=rows, cells=cells, lab=lab, ti=ti, emb=base["decipher2d"][rows].astype(np.float64),
             root_mask=np.load(f"{RF}/{ds}/root_mask.npy").astype(bool)[rows], pt={}, pt_variant={}, fields={})
    D["knn"] = {"Decipher latent": knn(Zd[rows], K_NN), "gene PCA": knn(PCA(30, random_state=0).fit_transform(Xc[rows]), K_NN)}
    for k, (c, g, X, V) in meths.items():
        pos = pd.Index(c).get_indexer(cells); assert (pos >= 0).all(), k
        X, V = X[pos], V[pos]
        D["fields"][k] = (zs(X), V if k == "COMET" else V / (X.std(0) + 1e-8))   # velocity in the same z units

    # pseudotimes. COMET's are computed on all COMET cells (its own manifold), then restricted to the common cells
    Vc = comet_field(ds, base)
    D["pt"]["COMET"] = velocity_pseudotime(Xc, Vc, Zd, how=comet_pt_how)[rows]
    D["pt_variant"]["COMET"] = f"full field -> scVelo velocity graph -> {comet_pt_how}"
    for k in [m for m in PT_VARIANTS if m in meths]:
        best = None
        for var in PT_VARIANTS[k]:
            if var not in pc: continue
            pt = pc[var].values[rows].astype(float); a = np.mean(list(pair_aucs(pt, lab, ti).values()))
            if best is None or a > best[1]: best = (var, a, pt)
        D["pt"][k], D["pt_variant"][k] = best[2], best[0]
    return D


if __name__ == "__main__":     # smoke test: headline numbers for one dataset, no bootstrap
    ds = sys.argv[1] if len(sys.argv) > 1 else "hema"; D = load_dataset(ds)
    for name, pt in D["pt"].items():
        print(name, D["pt_variant"][name], round(np.mean(list(pair_aucs(pt, D["lab"], D["ti"]).values())), 3))
    for m, (Xz, V) in D["fields"].items():
        print(m, {s: round(cluster_balanced(transport(D["knn"][s], Xz, V, D["lab"], D["ti"])[-1], D["lab"]), 3) for s in SPACES})
