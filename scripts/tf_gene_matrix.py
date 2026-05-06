import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite
from scipy.sparse import lil_matrix
import anndata as ad

print("Loading files...")
peak_tf  = mmread("peak_tf.mtx").tocsr()
peak_gene = pd.read_csv("peak_gene.csv")
peaks = pd.read_csv("peaks.txt", header=None)[0].tolist()
tfs   = pd.read_csv("tfs.txt",   header=None)[0].tolist()

# --------------------------------------------------
# Distance filter (50kb)
# --------------------------------------------------
DISTANCE_CUTOFF = 50_000

if "distance" in peak_gene.columns:
    before = len(peak_gene)
    peak_gene = peak_gene[peak_gene["distance"].abs() <= DISTANCE_CUTOFF].copy()
    print(f"Distance filter: {before} -> {len(peak_gene)}")
else:
    print("WARNING: no distance column")

# OPTIONAL: restrict to protein coding (recommended)
if "gene_biotype" in peak_gene.columns:
    before = len(peak_gene)
    peak_gene = peak_gene[peak_gene["gene_biotype"] == "protein_coding"].copy()
    print(f"Protein-coding filter: {before} -> {len(peak_gene)}")

# --------------------------------------------------
# Load RNA
# --------------------------------------------------
print("Loading Velten RNA...")
velten = ad.read_h5ad("velten_rna.h5ad")
rna_genes = list(velten.var_names)

# --------------------------------------------------
# Normalization (CRITICAL: do NOT destroy symbols)
# --------------------------------------------------
def norm(x):
    return str(x).split(".")[0].upper()

# Build fast lookup
norm_to_orig = {}
for g in rna_genes:
    k = norm(g)
    if k not in norm_to_orig:
        norm_to_orig[k] = g

# --------------------------------------------------
# Peak indexing (fix dash vs colon mismatch)
# --------------------------------------------------
def fix_peak(p):
    return p.replace("-", ":", 1)

peak_to_idx = {fix_peak(p): i for i, p in enumerate(peaks)}
gene_to_idx = {g: i for i, g in enumerate(rna_genes)}

# --------------------------------------------------
# Detect peak column
# --------------------------------------------------
print("peak_gene columns:", peak_gene.columns.tolist())

peak_col = None
for c in ["query_region", "query.region", "peak", "query"]:
    if c in peak_gene.columns:
        peak_col = c
        break

if peak_col is None:
    raise ValueError("Could not find peak column")

print(f"Using peak column: {peak_col}")

# --------------------------------------------------
# Build peak-gene matrix
# --------------------------------------------------
print("Building peak-gene matrix...")

P = len(peaks)
G = len(rna_genes)

peak_gene_mat = lil_matrix((P, G))
n_matched = 0

for _, row in peak_gene.iterrows():
    p = fix_peak(row[peak_col])

    if p not in peak_to_idx:
        continue

    # Try gene symbol first
    g_name = norm(row["gene_name"])
    g_id   = norm(row["gene_id"])

    if g_name in norm_to_orig:
        g_orig = norm_to_orig[g_name]
    elif g_id in norm_to_orig:
        g_orig = norm_to_orig[g_id]
    else:
        continue

    peak_gene_mat[peak_to_idx[p], gene_to_idx[g_orig]] = 1
    n_matched += 1

peak_gene_mat = peak_gene_mat.tocsr()

print(f"Matched {n_matched} peak-gene pairs")
print(f"Matrix shape: {peak_gene_mat.shape}")
print(f"Nonzero entries: {peak_gene_mat.nnz}")

# --------------------------------------------------
# Build TF-gene matrix
# --------------------------------------------------
print("Building TF-gene matrix M (genes x TFs)...")

M = (peak_gene_mat.T @ peak_tf)
M = (M > 0).astype(np.float32)

print("Shape:", M.shape)
print("Density:", M.mean())

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
np.savetxt("tf_list.txt", np.array(tfs), fmt="%s")
np.savetxt("gene_list.txt", np.array(rna_genes), fmt="%s")
mmwrite("M_gene_tf.mtx", M)

print("Saved: M_gene_tf.mtx, gene_list.txt, tf_list.txt")
print("DONE")
