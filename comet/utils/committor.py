import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors

from config import COMMIT_GRAPH_K
from utils.common import zscore_cols


def latent_knn(src, k):
    _, idx = NearestNeighbors(n_neighbors=k+1).fit(src).kneighbors(src)
    return idx[:, 1:]

def moment_operator(Z, k):
    """Row-normalized kNN averaging operator over the state space Z (self included):
    (S @ x)[i] = mean of x over {i} ∪ kNN_Z(i)."""
    Z = np.asarray(Z, np.float32); N = len(Z)
    idx  = latent_knn(Z, k)
    cols = np.concatenate([np.arange(N)[:, None], idx], axis=1)          # (N, k+1), self first
    rows = np.repeat(np.arange(N), k + 1)
    data = np.full(rows.shape, 1.0 / (k + 1), np.float32)
    return sp.csr_matrix((data, (rows, cols.ravel())), shape=(N, N))

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


def committor_gene_direction(b, q, Z=None, k=COMMIT_GRAPH_K, chunk=1024):
    """Per-cell gene-space direction of increasing committor (the 'forward' displacement).
    d_i = Σ_j (q_j - q_i)(rna_z_j - rna_z_i) over latent neighbors, L2-normalized; cells on a flat plateau are filled from their neighbours (see below).
    Returns (direction (N, G), valid mask). Z: state space whose kNN defines 'forward' (default Decipher z)."""
    Z = b.Z_dec if Z is None else np.asarray(Z, np.float32)
    idx = latent_knn(Z, k); rz = zscore_cols(b.X)
    N, G = rz.shape; q = q.astype(np.float32); D = np.zeros((N, G), np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N); nb = idx[s:e]                        # (c, k)
        w = (q[nb] - q[s:e, None]).astype(np.float32)               # (c, k)
        D[s:e] = np.einsum('ck,ckg->cg', w, rz[nb]) - w.sum(1)[:, None] * rz[s:e]
    nrm = np.linalg.norm(D, axis=1); valid = nrm > 1e-6
    D[valid] /= nrm[valid][:, None]

    # plateau cells (q locally flat -> ~0 raw direction) take the mean direction of their valid
    # neighbours, propagated until every cell is filled (at most 50 hops)
    idxk = idx; Df = D.copy(); vf = valid.copy()
    for _ in range(50):
        nbv = vf[idxk]                                    # (N, k) bool
        s   = (Df[idxk] * nbv[..., None]).sum(1)          # sum of valid-neighbor dirs
        sn  = np.linalg.norm(s, axis=1)
        take = (~vf) & (nbv.sum(1) > 0) & (sn > 1e-6)
        Df[take] = s[take] / sn[take][:, None]; vf[take] = True
        if not (~vf).any(): break
    return Df.astype(np.float32), vf
