import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from config import COMMIT_GRAPH_K, COMMIT_ROOT_Q, COMMIT_SINK_FRAC
from utils.common import zscore_cols, zsc


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
