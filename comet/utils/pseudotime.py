"""COMET pseudotime: the scalar potential of COMET's final velocity field (the one used in all figures).

Read off the finished field V_pred.npy (committor direction + chromatin lead + GRN tilt), not the committor q:
    state        X~ = z-scored expression (per gene), V = V_pred (unit per cell, z-scored gene units)
    graph        30-NN graph on the Decipher latent Z
    transitions  scVelo velocity_graph (cosine of V_i with x~_j − x~_i over neighbours) -> transition matrix T
    potential    least squares over MUTUAL kNN edges of (phi_j − phi_i) ≈ (T_ij − T_ji)/(T_ij + T_ji)  (HodgeRank)
    pseudotime   phi, min-max scaled to [0, 1]

No root cell is needed: the direction comes entirely from the field. Identical to figures/new_metrics.py
(velocity_pseudotime(..., how="potential")).

    from utils.pseudotime import comet_pseudotime
    pt = comet_pseudotime(b, np.load(f"{out_dir}/V_pred.npy"))       # b = LOADERS[ds](BASES[ds])
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

PT_K_NN = 30


def zs(A):
    A = np.asarray(A, np.float32); return (A - A.mean(0)) / (A.std(0) + 1e-8)


def graph_flow_potential(T):
    """scalar potential of the net flow of a transition matrix (HodgeRank): least squares over MUTUAL kNN edges of
    (φ_j − φ_i) ≈ (T_ij − T_ji)/(T_ij + T_ji); min-max scaled."""
    T = sp.csr_matrix(T); A = (T > 0).astype(np.int8); S = sp.triu(A.multiply(A.T).tocsr(), k=1).tocoo()
    tij = np.asarray(T[S.row, S.col]).ravel(); tji = np.asarray(T[S.col, S.row]).ravel()
    g = (tij - tji) / (tij + tji + 1e-12); m = len(g)
    B = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[S.row, S.col])),
                      shape=(m, T.shape[0]))
    phi = lsqr(B, g, atol=1e-9, btol=1e-9, iter_lim=4000)[0]
    return (phi - phi.min()) / (np.ptp(phi) + 1e-12)


def velocity_pseudotime(Xz, V, Z, n_neighbors=PT_K_NN):
    """gene-space field -> pseudotime: scVelo velocity_graph on Ms = Xz with neighbours on Z, then the graph-flow
    potential of its transition matrix"""
    import anndata as ad, scanpy as sc, scvelo as scv
    scv.settings.verbosity = 0
    a = ad.AnnData(np.asarray(Xz, np.float32)); a.layers["Ms"] = a.X.copy(); a.layers["velocity"] = np.asarray(V, np.float32)
    a.obsm["X_rep"] = np.asarray(Z, np.float32)
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X_rep", random_state=0)
    scv.tl.velocity_graph(a, vkey="velocity", xkey="Ms", n_jobs=8)
    return graph_flow_potential(scv.utils.get_transition_matrix(a, vkey="velocity"))


def comet_pseudotime(b, V, n_neighbors=PT_K_NN):
    """COMET pseudotime of a loaded dataset bundle b (b.X expression, b.Z_dec Decipher latent) and its field V
    (the run's V_pred.npy)"""
    return velocity_pseudotime(zs(b.X), V, b.Z_dec, n_neighbors)
