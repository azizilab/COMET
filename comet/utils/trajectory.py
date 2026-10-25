"""Final COMET trajectory formulation (consensus chromatin origin -> committor -> chromatin-lead fill-in).

Developed and validated in trajectory_benchmarking/new_trajectory_experiments.ipynb (§5e-§7) and
trajectory_benchmarking/final_trajectory_testing.ipynb; the math is written up in
trajectory_benchmarking/final_method_writeup.md. This module reproduces final_trajectory_testing exactly.

    origin score  phi = s · [z(chromatin entropy) + z(breadth)] of a finer (120-cluster) pseudo-bulk ATAC
    root          consensus over K in [3, 1/(2·ROOT_FRAC)] × seeds of the cluster-median rule (KMeans on Z,
                  cluster with the highest MEDIAN phi, ROOT_FRAC·N cells geodesically closest to its medoid)
    sinks         consensus over the same K range × seeds of: low-phi cells far from the root, KMeans'd into K groups,
                  the farthest of each group (no number of branches assumed)
    pseudotime    harmonic committor q (0 at the root, 1 at the sinks)
    direction     u = committor_gene_direction(q), smoothed over k_local neighbours -> u_bar
    lead          offset-free residual rho·(A~ − rho·R~) at matched pseudo-bulk resolution -> tangent projection
                  -> forward cone (no component back toward the root) -> k_local average -> × coherence²
    velocity      V = unit[(1 − w) u_bar + λ w c_hat],  w = g·(1 − min(1, Δq / P_plateau(Δq))),
                  g = 1[median ATAC↔RNA coupling ≥ coupling_min]

Nothing in final_trajectory is trained and no labels are used. After the GRN is trained, run_comet applies the
regulatory-consistency tilt (regulatory_consistency_tilt below):
    regulons      L_i = (A_base + Σ_k w_ik A_k)[:, expressed TFs] / σ   (self-edges removed; each module pruned to its
                  top GRN_PRUNE_FRAC by |weight|; σ = gene SD, median-floored)
    activity      a_i = (L_iᵀL_i + α_i I)⁻¹ L_iᵀ V_i                    TF activity velocity
    explained     g_i = L_i a_i,  R²_i = <g_i, V_i> / ||V_i||²
    tilt          ĝ, κ = tangent projection → forward cone along V → k_local average → (unit, coherence²)
                  V' = unit[V + λ κ ĝ]
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path, dijkstra
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize

from config import (
    K_MOMENT, COMMIT_GRAPH_K, COMMIT_SINK_FRAC, TRAJ_ATAC_RES, TRAJ_ROOT_FRAC,
    TRAJ_N_SEEDS, TRAJ_K_LOCAL, TRAJ_PLATEAU_Q, TRAJ_LEAD_LAMBDA, TRAJ_COUPLING_MIN,
    GRN_TILT_LAMBDA, GRN_TILT_ALPHA, GRN_TILT_MIN_DETECT, GRN_PRUNE_FRAC,
)
from utils.common import zsc, zscore_cols, knn_smooth
from utils.geometry import build_local_atac
from utils.committor import build_graph, committor, committor_gene_direction


def _unit(V):
    return (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _knn(Z, k):
    """k nearest neighbours in Z, the cell itself included (column 0)"""
    return NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)[1]


# ---------------------------------------------------------------------------------------------------------------------
# origin
# ---------------------------------------------------------------------------------------------------------------------
def pseudobulk_atac(b, n_clusters, k_atac, k_smooth):
    """ga_matched rebuilt at another resolution from the raw per-cell gene activity (b.ga_atac_cells; paired data
    averages each cluster's own cells instead of matching). (40, 8, 15) reproduces the loader's b.ga_matched.
    No raw ATAC -> b.ga_matched."""
    if getattr(b, "ga_atac_cells", None) is None:
        return np.asarray(b.ga_matched, np.float32)
    ga, _ = build_local_atac(b.X, b.ga_atac_cells, b.Z_dec, n_clusters=n_clusters, k_atac=k_atac, paired=b.paired)
    if k_smooth:
        ga = knn_smooth(ga, _knn(b.Z_dec, k_smooth))
    return ga.astype(np.float32)


def entropy_breadth(mat):
    """z(entropy of the L1-normalised row) + z(fraction of features above their median): root-high openness"""
    mat = np.asarray(mat, np.float32)
    p = sk_normalize(np.clip(mat, 0, None), "l1", axis=1)
    return (zsc(-(p * np.log(p + 1e-12)).sum(1)) + zsc((mat > np.median(mat, 0, keepdims=True)).mean(1))).astype(np.float32)


def origin_score(b, sign):
    return (float(sign) * entropy_breadth(pseudobulk_atac(b, *TRAJ_ATAC_RES))).astype(np.float32)


def root_cluster_medoid(Z, phi, Wd, K, seed, frac):
    """KMeans(K) on Z; origin cluster = highest MEDIAN phi (robust to scattered extreme cells); root = the frac·N
    cells of that cluster geodesically closest to its medoid"""
    cl = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Z)
    inC = cl == np.argmax([np.median(phi[cl == c]) for c in range(K)])
    idx = np.where(inC)[0]; Zc = Z[idx]
    medoid = idx[np.argmin(((Zc - Zc.mean(0)) ** 2).sum(1))]
    d = np.where(inC, shortest_path(Wd, method="D", indices=[int(medoid)])[0], np.inf)
    root = np.zeros(len(phi), bool)
    root[np.argsort(d)[:min(int(round(frac * len(phi))), int(inC.sum()))]] = True
    return root


def consensus_root(Z, phi, Wd, frac=TRAJ_ROOT_FRAC, n_seeds=TRAJ_N_SEEDS, k_range=None):
    """root = the frac·N cells most often selected by root_cluster_medoid over K in k_range × n_seeds seeds.
    Default k_range = [3, 1/(2·frac)]: an average cluster must hold at least two roots' worth of cells.
    Ties are broken by the 30-NN-smoothed phi. Returns (root mask, selection frequency, k_range)."""
    if k_range is None:
        k_range = range(3, int(round(1.0 / (2.0 * frac))) + 1)
    f = np.mean([root_cluster_medoid(Z, phi, Wd, K, s, frac) for K in k_range for s in range(n_seeds)], 0)
    s_ = phi[_knn(Z, 30)].mean(1)
    key = f + 1e-6 * (s_ - s_.min()) / (np.ptp(s_) + 1e-8)
    root = np.zeros(len(f), bool); root[np.argsort(-key)[:int(round(frac * len(f)))]] = True
    return root, f.astype(np.float32), list(k_range)


def sink_tips(Zc, cand, d, K, seed, n_per):
    """KMeans(K) on the candidate cells; the n_per candidates farthest from the root (largest d) in each group"""
    lab = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Zc)
    sink = np.zeros(len(d), bool)
    for k in range(K):
        ck = cand[lab == k]; sink[ck[np.argsort(-d[ck])[:n_per]]] = True
    return sink


def consensus_sinks(Z, phi, Wd, root, k_range, sink_frac=COMMIT_SINK_FRAC, n_seeds=TRAJ_N_SEEDS, phi_pct=40, far_pct=75):
    """sinks = the sink_frac·N cells most often selected as branch tips over K in k_range × n_seeds seeds, where one
    selection = the candidates (low phi, farthest quarter from the root) KMeans'd into K groups and the farthest
    sink_frac·N/K of each group taken. No number of branches is assumed: too many groups only re-split one branch,
    too few would miss one, and the consensus keeps the cells chosen at every K (same K range as the root).
    Returns (sink mask, selection frequency, geodesic distance from the root)."""
    N = len(phi)
    d = dijkstra(Wd, directed=False, indices=np.where(root)[0], min_only=True)
    d = np.where(np.isfinite(d), d, np.nanmax(d[np.isfinite(d)]))
    cand = np.where((phi < np.percentile(phi, phi_pct)) & (d >= np.percentile(d, far_pct)) & ~root)[0]
    n_sink = int(sink_frac * N); sink = np.zeros(N, bool)
    if len(cand) < 4 * max(k_range):               # degenerate geometry: fall back to the farthest low-phi cells
        far = np.where(~root)[0]; sink[far[np.argsort(-d[far])[:n_sink]]] = True
        return sink, sink.astype(np.float32), d.astype(np.float32)
    f = np.mean([sink_tips(Z[cand], cand, d, K, s, max(1, int(sink_frac * N / K)))
                 for K in k_range for s in range(n_seeds)], 0)
    sink[np.argsort(-(f + 1e-6 * d / d.max()))[:n_sink]] = True
    return sink, f.astype(np.float32), d.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# chromatin lead
# ---------------------------------------------------------------------------------------------------------------------
def matched_pseudobulk(b, S_mom, n_clusters=40, k_atac=8):
    """ATAC and RNA at the SAME resolution: RNA goes through ga_matched's pipeline (same KMeans pseudo-bulk clusters,
    per-gene min-max, K_MOMENT-NN smooth in Decipher z); both are then smoothed by S_mom.
    No raw ATAC: per-cell ATAC (ga_matched) vs per-cell RNA. Returns (A, R, cluster labels | None)."""
    if getattr(b, "ga_atac_cells", None) is None:
        return (np.asarray(S_mom @ b.ga_matched, np.float32), np.asarray(S_mom @ b.X, np.float32), None)
    ga, lab = build_local_atac(b.X, b.ga_atac_cells, b.Z_dec, n_clusters=n_clusters, k_atac=k_atac, paired=b.paired)
    r = np.stack([b.X[lab == c].mean(0) for c in range(lab.max() + 1)])[lab]
    r = (r - r.min(0)) / (r.max(0) - r.min(0) + 1e-8)
    ni = _knn(b.Z_dec, K_MOMENT); ga, r = knn_smooth(ga, ni), knn_smooth(r, ni)
    return np.asarray(S_mom @ ga, np.float32), np.asarray(S_mom @ r, np.float32), lab


def residual_lead(A, R):
    """offset-free lead rho_g·(A~_g − rho_g·R~_g), rho_g = corr(A~_g, R~_g) clipped at 0 (A~, R~ standardised after
    smoothing): what ATAC predicts for RNA beyond what the current RNA already explains. Invariant to any per-gene
    centring, so it carries no pull toward an anchor (the regression-to-the-mean artifact of A~ − R~).
    Returns (lead, per-gene coupling rho unclipped)."""
    At, Rt = zscore_cols(A), zscore_cols(R); corr = (At * Rt).mean(0)
    rho = np.clip(corr, 0, None).astype(np.float32)
    return (rho * (At - rho * Rt)).astype(np.float32), corr.astype(np.float32)


def tangent_project(V, Rt, nbr_idx, ridge=1e-2, chunk=500):
    """component of V in the span of each cell's neighbour displacements in Rt (ridge least squares);
    nbr_idx excludes the cell itself"""
    k = nbr_idx.shape[1]; out = np.zeros_like(V)
    for s in range(0, len(V), chunk):
        D = Rt[nbr_idx[s:s + chunk]] - Rt[s:s + chunk, None, :]
        Gm = np.einsum("bkg,blg->bkl", D, D); tr = np.trace(Gm, axis1=1, axis2=2)[:, None, None] / k + 1e-6
        beta = np.linalg.solve(Gm + ridge * tr * np.eye(k)[None],
                               np.einsum("bkg,bg->bk", D, V[s:s + chunk])[..., None])[..., 0]
        out[s:s + chunk] = np.einsum("bk,bkg->bg", beta, D)
    return out


def forward_cone(V, D):
    """remove the component of V pointing backward along the direction field D"""
    D = _unit(D)
    return (V - np.minimum((V * D).sum(1, keepdims=True), 0.0) * D).astype(np.float32)


def coherence(V, nbr_idx):
    """|mean of unit vectors| over the neighbourhood, in [0, 1]"""
    return np.linalg.norm(_unit(V)[nbr_idx].mean(1), axis=1).astype(np.float32)


def chromatin_lead(b, S_mom, Z, outward=None, k_local=TRAJ_K_LOCAL, A=None, R=None):
    """residual -> tangent projection (per-cell RNA~ manifold) -> [forward cone along `outward`] -> k_local average
    -> × coherence². Pass A/R to run the identical pipeline on other inputs (e.g. a shuffled-ATAC null)."""
    if A is None or R is None:
        A, R, _ = matched_pseudobulk(b, S_mom)
    c, corr = residual_lead(A, R)
    Rt_cell = zscore_cols(np.asarray(S_mom @ b.X, np.float32))
    idx = _knn(Z, k_local + 1)
    c = tangent_project(c, Rt_cell, idx[:, 1:])
    if outward is not None:
        c = forward_cone(c, outward)
    nb = _knn(Z, k_local)
    c = c[nb].mean(1).astype(np.float32)
    c = (c * coherence(c, nb)[:, None] ** 2).astype(np.float32)
    return c, corr


# ---------------------------------------------------------------------------------------------------------------------
# full trajectory
# ---------------------------------------------------------------------------------------------------------------------
def final_trajectory(b, Z, S_mom, sign=1.0, k_graph=COMMIT_GRAPH_K, k_local=TRAJ_K_LOCAL,
                     plateau_q=TRAJ_PLATEAU_Q, lam=TRAJ_LEAD_LAMBDA, coupling_min=TRAJ_COUPLING_MIN, verbose=True):
    """Everything above, end to end. Returns a dict:
        q (pseudotime), root_mask, sink_mask, phi, root_freq, sink_freq, k_range,
        u (raw committor direction, unit), u_valid, u_bar (smoothed), lead (chromatin lead), coupling (median rho),
        gate (0/1), w (per-cell lead weight), V (final velocity field, unit per cell)"""
    Z = np.asarray(Z, np.float32)
    phi = origin_score(b, sign)
    Wd, _ = build_graph(Z, k=k_graph, mode="dist"); Wg, _ = build_graph(Z, k=k_graph, mode="gauss")
    root, freq, k_range = consensus_root(Z, phi, Wd)
    sink, sink_freq, d_root = consensus_sinks(Z, phi, Wd, root, k_range)
    q = committor(Wg, root, sink).astype(np.float32)
    u, u_valid = committor_gene_direction(b, q, Z=Z)
    outward = committor_gene_direction(b, (d_root / d_root.max()).astype(np.float32), Z=Z)[0]

    lead, corr = chromatin_lead(b, S_mom, Z, outward=outward, k_local=k_local)
    coupling = float(np.median(corr)); gate = float(coupling >= coupling_min)

    nb = _knn(Z, k_local)
    u_bar = _unit(_unit(u)[nb].mean(1))
    idx16 = _knn(Z, 16); dq = q[idx16].max(1) - q[idx16].min(1)
    a = np.clip(dq / (np.quantile(dq, plateau_q) + 1e-12), 0, 1)
    w = (gate * (1.0 - a)).astype(np.float32)
    V = _unit((1.0 - w)[:, None] * u_bar + lam * w[:, None] * _unit(lead))

    if verbose:
        print(f"  trajectory: consensus root over K={k_range[0]}-{k_range[-1]} × {TRAJ_N_SEEDS} seeds -> {int(root.sum())} cells, "
              f"sinks={int(sink.sum())} | ATAC↔RNA coupling {coupling:.2f} -> chromatin lead "
              f"{'ON' if gate else 'OFF'} | plateau cells (w>0) {float((w > 0).mean()):.0%}")
    return dict(q=q, root_mask=root, sink_mask=sink, phi=phi, root_freq=freq, sink_freq=sink_freq, k_range=k_range,
                u=u, u_valid=u_valid, u_bar=u_bar, lead=lead, coupling=coupling, gate=gate, w=w, V=V)


# ---------------------------------------------------------------------------------------------------------------------
# regulatory-consistency tilt (applied after GRN training)
# ---------------------------------------------------------------------------------------------------------------------
def prune_top_frac(W, frac):
    """W with all but its strongest `frac` of nonzero entries (by |weight|) zeroed; ties at the cutoff are kept.
    None or frac >= 1 returns W unchanged."""
    if frac is None or frac >= 1.0:
        return W
    a = np.abs(W); nz = a[a > 0]
    if nz.size == 0:
        return W
    k = max(1, int(round(nz.size * frac)))
    return W * (a >= np.partition(nz, nz.size - k)[nz.size - k])


def regulon_activity(V, X, X_TF, tf_idx, A_base, A_dict, mix, alpha=GRN_TILT_ALPHA, min_detect=GRN_TILT_MIN_DETECT,
                     prune_frac=GRN_PRUNE_FRAC, chunk=250):
    """per-cell ridge fit of the displacement V_i on the cell's signed regulons L_i (expressed TFs, z units).
    A_base and each atom are first pruned to their own top prune_frac of edges (prune_top_frac).
    The per-cell Gram matrix is assembled from precomputed blocks of (A_base, A_1..A_D) weighted by (1, w_i).
    Returns (a (N, T) TF activity velocity, 0 for unexpressed TFs; g (N, G) GRN-explained displacement;
    R² (N,) GRN-explained fraction)."""
    A_base, A_dict = A_base.copy(), A_dict.copy()
    t = np.arange(A_base.shape[1]); A_base[tf_idx, t] = 0.0; A_dict[:, tf_idx, t] = 0.0     # no self-edges
    A_base = prune_top_frac(A_base, prune_frac)
    A_dict = np.stack([prune_top_frac(A_k, prune_frac) for A_k in A_dict])
    sd = X.std(0); sg = np.maximum(sd, np.median(sd)) + 1e-8                                 # median-floored gene SD
    sel = np.where((X_TF > 0).mean(0) >= min_detect)[0]
    L = [A_base[:, sel] / sg[:, None]] + [A_dict[k][:, sel] / sg[:, None] for k in range(A_dict.shape[0])]
    Wm = np.concatenate([np.ones((len(V), 1), np.float32), mix], 1)                          # L_i = Σ_j Wm[i, j] L[j]
    Gab = np.stack([np.stack([La.T @ Lb for Lb in L]) for La in L])
    rhs = np.stack([V @ La for La in L], 1)
    a = np.zeros((len(V), len(sel)), np.float32); T = len(sel)
    for s in range(0, len(V), chunk):
        w = Wm[s:s + chunk]; Gi = np.einsum("na,nb,abst->nst", w, w, Gab, optimize=True)
        lam = alpha * np.trace(Gi, axis1=1, axis2=2)[:, None, None] / T
        a[s:s + chunk] = np.linalg.solve(Gi + lam * np.eye(T)[None], np.einsum("na,nat->nt", w, rhs[s:s + chunk])[..., None])[..., 0]
    g = sum(Wm[:, j:j + 1] * (a @ L[j].T) for j in range(len(L))).astype(np.float32)
    a_full = np.zeros((len(V), A_base.shape[1]), np.float32); a_full[:, sel] = a
    return a_full, g, ((g * V).sum(1) / ((V * V).sum(1) + 1e-12)).astype(np.float32)


def regulatory_consistency_tilt(b, V, A_base, A_dict, mix, S_mom, Z, lam=GRN_TILT_LAMBDA, k_local=TRAJ_K_LOCAL,
                                verbose=True):
    """V' = unit[V + λ κ ĝ]: lean each cell's displacement toward the part its own regulons explain. ĝ, κ come from the
    chromatin lead's post-processing (tangent projection on the per-cell RNA~ manifold → forward cone along V →
    k_local average → unit, coherence²), so the tilt never points backward and its angle to V is ≤ arctan λ.
    Returns dict(V (tilted field), activity (N, T TF activity velocity), explained (N, GRN-explained fraction), kappa)."""
    Z = np.asarray(Z, np.float32); V = _unit(V)
    a, g, r2 = regulon_activity(V, b.X, b.X_TF, np.asarray(b.tf_indices_in_G), A_base, A_dict, mix)
    Rt_cell = zscore_cols(np.asarray(S_mom @ b.X, np.float32))
    c = tangent_project(g, Rt_cell, _knn(Z, k_local + 1)[:, 1:])
    c = forward_cone(c, V)
    nb = _knn(Z, k_local); c = c[nb].mean(1).astype(np.float32)
    kappa = (coherence(c, nb) ** 2).astype(np.float32)
    Vt = _unit(V + lam * kappa[:, None] * _unit(c))
    if verbose:
        ang = np.degrees(np.arccos(np.clip((Vt * V).sum(1), -1, 1)))
        print(f"  regulatory-consistency tilt (λ={lam}, prune_frac={GRN_PRUNE_FRAC}): "
              f"median GRN-explained fraction {np.median(r2):.3f}, "
              f"median κ {np.median(kappa):.2f}, median gene-space turn {np.median(ang):.1f}°")
    return dict(V=Vt, activity=a, explained=r2, kappa=kappa)
