import numpy as np
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────
# TF accessibility prior
# ─────────────────────────────────────────────────────────────

def build_tf_access_prior(gene_activity, A_tfs, sub_g2i):
    """
    gene_activity: (N, G)
    returns: (T,)
    """
    T = len(A_tfs)
    tf_access = np.zeros(T, dtype=np.float32)

    for i, tf in enumerate(A_tfs):
        if tf in sub_g2i:
            tf_access[i] = float(gene_activity[:, sub_g2i[tf]].mean())

    lo, hi = tf_access.min(), tf_access.max()
    tf_access = (tf_access - lo) / (hi - lo + 1e-8)

    return tf_access


# ─────────────────────────────────────────────────────────────
# Global gene accessibility
# ─────────────────────────────────────────────────────────────

def build_gene_access(gene_activity):
    gene_access = gene_activity.mean(axis=0)

    gene_access = (gene_access - gene_access.min()) / (
        gene_access.max() - gene_access.min() + 1e-8
    )

    return gene_access


# ─────────────────────────────────────────────────────────────
# Local ATAC prior (cluster-based pseudo-bulk)
# ─────────────────────────────────────────────────────────────

def build_local_atac(X_rna, gene_activity, rna_2d, n_clusters=20, k_atac=5):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize as sk_normalize

    G = X_rna.shape[1]

    km = MiniBatchKMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    clusters = km.fit_predict(rna_2d)

    cluster_rna = np.zeros((n_clusters, G), dtype=np.float32)

    for c in range(n_clusters):
        mask = clusters == c
        cluster_rna[c] = X_rna[mask].mean(axis=0)

    cr = sk_normalize(cluster_rna)
    ga = sk_normalize(gene_activity)

    nbrs = NearestNeighbors(n_neighbors=k_atac).fit(ga)
    _, idx = nbrs.kneighbors(cr)

    cluster_access = np.zeros((n_clusters, G), dtype=np.float32)

    for c in range(n_clusters):
        cluster_access[c] = gene_activity[idx[c]].mean(axis=0)

    local_access = cluster_access[clusters]

    # normalize per gene
    lo = local_access.min(axis=0, keepdims=True)
    hi = local_access.max(axis=0, keepdims=True)

    local_access = (local_access - lo) / (hi - lo + 1e-8)

    return local_access, clusters