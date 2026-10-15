import numpy as np
from scipy.sparse import issparse


def as_dense(x):  return x.toarray() if issparse(x) else np.asarray(x)
def norm_name(s): return str(s).strip().casefold()

def zscore_cols(A, eps=1e-8):
    """Per-gene (per-column) z-score across cells."""
    A = np.asarray(A, dtype=np.float32)
    return (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + eps)

def zsc(v):
    v = np.asarray(v, np.float64)
    return ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)

def knn_smooth(signal, neighbor_idx, k=None):
    nb = neighbor_idx if k is None else neighbor_idx[:, :k]
    return signal[nb].mean(axis=1)
