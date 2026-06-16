import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def graph_smooth(values: torch.Tensor, neighbor_idx: torch.Tensor,
                  n_hops: int = 2) -> torch.Tensor:
    """
    Mean-aggregation graph smoothing over n_hops rounds.
    Each hop replaces v[i] with the average of v[i] and its neighbours.
    Used to denoise psi_raw before it becomes a velocity field: latent-space
    kNN (higher fidelity) should be used as neighbor_idx here, not 2D kNN.
    """
    v = values
    for _ in range(n_hops):
        v_nbr = v[neighbor_idx]                  # (N, k, ...)
        v = 0.5 * v + 0.5 * v_nbr.mean(dim=1)
    return v


def build_neighbor_index(X, V_dec2, Z_dec, mode="decipher", k=15):
    """
    mode = "decipher"  → neighbors in 2-D Decipher vis-space (default)
    mode = "pca"       → neighbors in top-50 PCA of gene expression
    mode = "latent"    → neighbors in Decipher latent z (more stable)
    """
    if mode == "decipher":
        ref = V_dec2
    elif mode == "pca":
        pca50 = PCA(n_components=min(50, X.shape[1]-1), random_state=42)
        ref   = pca50.fit_transform(X)
    elif mode == "latent":
        ref = Z_dec
    else:
        raise ValueError(f"Unknown NEIGHBOR_SPACE={mode}")
 
    nbrs = NearestNeighbors(n_neighbors=k).fit(ref)
    _, neighbor_indices = nbrs.kneighbors(ref)
    return torch.tensor(neighbor_indices, dtype=torch.long)