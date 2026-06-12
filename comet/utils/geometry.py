import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize


def project_to_2d(V_pred_full, X_2d_vis, X_full,
                  k=30, scale=10.0, adaptive_scale=True, density_k=15):
    """
    Vectorised version — no Python loop over cells.
    Same algorithm, ~50x faster for N~1000.
    """
    N = V_pred_full.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_2d_vis)
    _, idx = nbrs.kneighbors(X_2d_vis)   # (N, k)

    if adaptive_scale:
        nbrs_d = NearestNeighbors(n_neighbors=density_k).fit(X_2d_vis)
        dist_d, _ = nbrs_d.kneighbors(X_2d_vis)
        mean_nn_dist = dist_d[:, 1:].mean(axis=1)
        density_factor = 1.0 / (mean_nn_dist + 1e-8)
        lo, hi = density_factor.min(), density_factor.max()
        density_factor = 1.0 + 2.0 * (density_factor - lo) / (hi - lo + 1e-8)
    else:
        density_factor = np.ones(N, dtype=np.float32)

    # Gene-space displacements: (C, k, G)
    d_high = X_full[idx] - X_full[:, np.newaxis, :]        # (C, k, G)

    # 2D displacements: (C, k, 2)
    d_low  = X_2d_vis[idx] - X_2d_vis[:, np.newaxis, :]   # (C, k, 2)

    # Cosine similarity — vectorised dot product
    # v: (C, G), d_high: (C, k, G)
    v_norm = np.linalg.norm(V_pred_full, axis=1, keepdims=True) + 1e-8   # (C, 1)
    d_norm = np.linalg.norm(d_high, axis=2) + 1e-8                        # (C, k)

    # dot(d_high[i,j], v[i]) = einsum over G
    dots = np.einsum('ng,nkg->nk', V_pred_full, d_high)   # (C, k)
    sims = dots / (d_norm * v_norm)                         # (C, k)

    # Center per cell
    sims_centered = sims - sims.mean(axis=1, keepdims=True)   # (C, k)

    # Adaptive softmax
    eff_scale = (scale * density_factor)[:, np.newaxis]        # (C, 1)
    weights = np.exp(sims_centered * eff_scale)                # (C, k)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8       # (C, k)

    # Zero out cells with near-zero velocity
    zero_vel = np.linalg.norm(V_pred_full, axis=1) < 1e-8     # (C,)
    weights[zero_vel] = 0.0

    # Weighted sum of 2D displacements: (C, k) x (C, k, 2) -> (C, 2)
    V_emb2 = np.einsum('nk,nkd->nd', weights, d_low).astype(np.float32)

    return V_emb2


def build_local_atac(
    X_rna,            # (C_rna, G)  RNA expression (subset genes)
    gene_activity,    # (C_atac, G) ATAC gene-activity matrix
    rna_2d,           # (C_rna, 2)  e.g. V_dec2 or X_pca2d
    n_clusters=15,
    k_atac=5,         # ATAC cells per RNA cluster to average
    random_state=42,
):
    """
    Returns local_gene_access: (C_rna, G) — per-cell ATAC accessibility
    estimated from the nearest ATAC cluster.
 
    Strategy
    ────────
    We don't have paired cells, so we can't do direct cell-cell matching.
    Instead we use a "pseudo-bulk by cluster" approach:
      • Cluster RNA cells in 2-D space (k-means for speed).
      • Represent each cluster by its mean RNA expression.
      • Represent each ATAC cell by its gene-activity vector.
      • Match each RNA cluster to its k nearest ATAC cells
        (in gene-activity space — both are in G-dimensional gene space).
      • Average those ATAC cells → cluster-level accessibility.
      • Broadcast back to RNA cells.
 
    This avoids noise of single-cell ATAC while still giving spatial
    resolution proportional to the number of clusters.
    """
 
    G = X_rna.shape[1]
 
    # 1. Cluster RNA cells in 2-D space
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                         n_init=5)
    cluster_labels = km.fit_predict(rna_2d)
 
    # 2. Mean RNA per cluster (in gene space)
    cluster_rna_mean = np.zeros((n_clusters, G), dtype=np.float32)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_rna_mean[c] = X_rna[mask].mean(axis=0)
 
    # 3. Normalize both spaces (L2) before matching
    cr_norm  = sk_normalize(cluster_rna_mean, norm='l2')
    ga_norm  = sk_normalize(gene_activity,    norm='l2')
 
    # 4. Match each RNA cluster to nearest ATAC cells
    nbrs_atac = NearestNeighbors(n_neighbors=k_atac).fit(ga_norm)
    _, atac_idx = nbrs_atac.kneighbors(cr_norm)   # (n_clusters, k_atac)
 
    # 5. Average ATAC cells per cluster
    cluster_access = np.zeros((n_clusters, G), dtype=np.float32)
    for c in range(n_clusters):
        cluster_access[c] = gene_activity[atac_idx[c]].mean(axis=0)
 
    # 6. Assign back to RNA cells
    local_gene_access = cluster_access[cluster_labels]   # (C_rna, G)
 
    # 7. Normalize to [0,1] per gene across clusters (for use in atac_direction_loss)
    lo = local_gene_access.min(axis=0, keepdims=True)
    hi = local_gene_access.max(axis=0, keepdims=True)
    local_gene_access = (local_gene_access - lo) / (hi - lo + 1e-8)
 
    print(f"Local ATAC: {n_clusters} clusters, k={k_atac} ATAC cells each")
    print(f"  Mean per-cluster nonzero fraction: "
          f"{np.mean(local_gene_access > 0.05):.1%}")
    return local_gene_access, cluster_labels