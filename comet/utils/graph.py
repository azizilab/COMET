import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


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