"""
comet_arrow_pseudotime.py — pseudotime as the scalar potential of the plotted 2-D arrow field.

Replaces scVelo's velocity_pseudotime (a diffusion *distance* from a root cell, which wraps
around the root: cells upstream AND downstream both get high values) with the discrete
Hodge/Helmholtz gradient of the exact 2-D field clean_quiver draws (velocity_umap). The result
increases along the arrows and decreases against them, so cells *behind* the root — whose arrows
point toward the root — come out EARLIER than the root, matching what you see in the quiver.

Same family of algorithm as CellRank/scVelo (a velocity-derived directed field on the kNN graph);
the robustness gain is in how pseudotime is read off that field: a least-squares scalar potential
instead of a symmetric graph distance.

Depends only on numpy + scipy (+ sklearn for the standalone-graph helper).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.sparse.csgraph import connected_components


def row_normalize(T):
    T = sp.csr_matrix(T, dtype=np.float64)
    rs = np.asarray(T.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ T


def knn_connectivities(X, k=30):
    """Symmetric binary kNN graph on X — only needed for the PCA/embedding branches, which have
    no scanpy/scVelo `connectivities` of their own. In the gene-space branch pass the existing
    adata.obsp['connectivities'] instead."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()  # drop self
    W = sp.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(n, n))
    return W.maximum(W.T)


def pseudotime_from_arrows(emb, V2D, W, root_mask=None, sink_mask=None,
                           smooth_steps=1, log=print):
    """Scalar potential phi whose gradient best matches the 2-D arrow field V2D on graph W.

    We solve, in weighted least squares over the undirected edges (i, j) of W,

            min_phi  sum_ij  w_ij * ( (phi_j - phi_i) - V_ij . (emb_j - emb_i) )^2 ,

    where V_ij = 0.5*(V2D_i + V2D_j) is the arrow averaged over the edge's endpoints. The target
    V_ij . (emb_j - emb_i) is the rise in phi you'd expect between i and j if the arrows really were
    the gradient of phi. The normal equations of this problem are the graph Poisson equation
    L phi = div(V); lsqr on the (signed) incidence matrix solves it directly and returns the
    minimum-norm solution (phi is defined only up to an additive constant), so no root cell is
    needed to *define* the coordinate — the arrows do that. root_mask/sink_mask are used only to
    fix the overall sign and, optionally, to report orientation.

    Because it fits a *gradient*, any rotational (curl) component of the field is discarded rather
    than being integrated into spurious progression — this is the Helmholtz decomposition, and it
    is what makes the result robust where a diffusion distance is not.

    Parameters
    ----------
    emb          : (n, 2) coordinates V2D lives in (e.g. adata.obsm['X_umap']).
    V2D          : (n, 2) per-cell arrows — the SAME array clean_quiver plots (velocity_umap).
    W            : (n, n) sparse neighbor weights. Gene-space: adata_sub.obsp['connectivities'].
                   PCA/embedding: knn_connectivities(X_pca, k).
    root_mask,
    sink_mask    : optional bool (n,); used only to orient the sign so root <= sink on average.
    smooth_steps : kNN-average V2D this many times first, mirroring clean_quiver's per-bin
                   averaging (0 to disable). Reduces single-cell arrow noise before integration.

    Returns
    -------
    phi : (n,) float32 pseudotime, min-max scaled to [0, 1].
    """
    emb = np.asarray(emb, dtype=np.float64)
    n = emb.shape[0]
    W = sp.csr_matrix(W).astype(np.float64)
    W = W.maximum(W.T)  # ensure symmetric

    ncomp, labels = connected_components(W, directed=False)
    if ncomp > 1:
        sizes = np.bincount(labels)
        log(f"pseudotime_from_arrows: graph has {ncomp} components (sizes {sorted(sizes, reverse=True)[:5]}...); "
            "offsets between components are not identifiable — solving jointly, interpret cross-component "
            "comparisons with care.")

    # optional smoothing of the arrow field along the graph (row-stochastic average)
    Vf = np.asarray(V2D, dtype=np.float64).copy()
    if smooth_steps and smooth_steps > 0:
        P = row_normalize(W)
        for _ in range(int(smooth_steps)):
            Vf = np.asarray(P @ Vf)

    # one row per undirected edge: (phi_j - phi_i) ~= V_ij . (emb_j - emb_i)
    E = sp.triu(W, k=1).tocoo()
    i, j, w = E.row, E.col, E.data
    disp = emb[j] - emb[i]                                   # edge displacement in the embedding
    g = 0.5 * ((Vf[i] + Vf[j]) * disp).sum(axis=1)          # antisymmetric: g_ji = -g_ij
    sw = np.sqrt(np.maximum(w, 0.0))

    m = len(i)
    B = sp.csr_matrix(
        (np.concatenate([-sw, sw]),
         (np.concatenate([np.arange(m), np.arange(m)]),
          np.concatenate([i, j]))),
        shape=(m, n),
    )
    phi = lsqr(B, sw * g, atol=1e-9, btol=1e-9, iter_lim=4000)[0].astype(np.float64)

    # orient: root earlier than sink on average (the arrows already set orientation; this is a guard)
    if root_mask is not None and sink_mask is not None:
        root_mask = np.asarray(root_mask, bool)
        sink_mask = np.asarray(sink_mask, bool)
        if root_mask.any() and sink_mask.any() and np.nanmean(phi[root_mask]) > np.nanmean(phi[sink_mask]):
            log("pseudotime_from_arrows: flipping sign so root_mask <= sink_mask on average.")
            phi = np.nanmax(phi) - phi

    lo, hi = np.nanmin(phi), np.nanmax(phi)
    if hi > lo:
        phi = (phi - lo) / (hi - lo)
    return phi.astype(np.float32)
