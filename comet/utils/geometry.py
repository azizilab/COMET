import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from config import RANDOM_STATE


def project_to_2d(V_pred_full, X_2d_vis, X_full, search_space=None, k=30, scale=1.0,
                  drift_correct=True, density_k=15, adaptive_scale=True):
    """Correlation-kernel projection (scVelo-style) with density/drift correction.
    `search_space` should be a state-resolving space (e.g. the Decipher latent Z_dec); the
    plotted 2D displacement always comes from the viz coordinates X_2d_vis."""
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
