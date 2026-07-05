import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize


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