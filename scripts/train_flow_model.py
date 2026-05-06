import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import mmread
import anndata as ad
from sklearn.neighbors import NearestNeighbors


# =========================
# LOAD DATA
# =========================
print("Loading data...")

adata = ad.read_h5ad("velten_rna.h5ad")

# Ensure cells × genes
X_full = adata.X
if hasattr(X_full, "toarray"):
    X_full = X_full.toarray()

gene_names = list(adata.var_names)
cell_names = list(adata.obs_names)


# TF list
with open("tf_names.txt") as f:
    raw = f.read()

# split on ANY whitespace (handles your current single-line file)
tfs = raw.split()

def clean_tf(tf):
    tf = tf.split("::")[0]   # handle complexes like MAX::MYC
    tf = tf.split("(")[0]    # remove (var.2)
    return tf

tfs = [clean_tf(tf) for tf in tfs]

print("Num TFs:", len(tfs))
print("First 10 TFs:", tfs[:10])

print("Matching TFs in RNA:", sum(tf in gene_names for tf in tfs))

# ATAC mask
M = mmread("M_gene_tf.mtx").tocsr()
M = M.toarray()


# =========================
# ALIGN GENE ORDER SAFETY CHECK
# =========================
M_genes = pd.read_csv("gene_list.txt", header=None)[0].tolist()

gene_index = {g: i for i, g in enumerate(gene_names)}

keep_idx = []
for g in M_genes:
    if g in gene_index:
        keep_idx.append(gene_index[g])

X = X_full[:, keep_idx]
M = M[range(len(keep_idx)), :]


# =========================
# BUILD X_TF FROM RNA + ALIGN M
# =========================
tf_index = {g: i for i, g in enumerate(gene_names)}

tf_cols = []
tf_keep_idx = []   # indices for M (ATAC space)
rna_tf_idx = []    # indices for X_full (RNA space)

for i, tf in enumerate(tfs):
    if tf in tf_index:
        tf_cols.append(tf)
        tf_keep_idx.append(i)              # index into M columns
        rna_tf_idx.append(tf_index[tf])    # index into RNA genes

# Build X_TF from RNA
X_TF = X_full[:, rna_tf_idx]

# IMPORTANT: subset M to same TFs
M = M[:, tf_keep_idx]


print("Final TF count:", len(tf_cols))
print("X shape:", X.shape)
print("X_TF shape:", X_TF.shape)
print("M shape:", M.shape)

# =========================
# MODEL
# =========================
class RegulatoryFlowModel(nn.Module):
    def __init__(self, G, T, K, M):
        super().__init__()
        self.U = nn.Parameter(torch.randn(G, K) * 0.01)
        self.V = nn.Parameter(torch.randn(T, K) * 0.01)
        self.M = torch.tensor(M, dtype=torch.float32)

    def forward(self, X_TF):
        A = (self.U @ self.V.T) * self.M
        V_pred = X_TF @ A.T
        return V_pred, A


# =========================
# TRAINING
# =========================
def train(X, X_TF, M, K=20, epochs=200, lr=1e-3, eps=0.1):

    X = torch.tensor(X, dtype=torch.float32)
    X_TF = torch.tensor(X_TF, dtype=torch.float32)

    N, G = X.shape
    T = X_TF.shape[1]

    model = RegulatoryFlowModel(G, T, K, M)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    nn_model = NearestNeighbors(n_neighbors=1).fit(X.numpy())

    for epoch in range(epochs):
        opt.zero_grad()

        V_pred, A = model(X_TF)
        X_next = X + eps * V_pred

        nn_idx = nn_model.kneighbors(X_next.detach().numpy(), return_distance=False)
        X_nn = X[nn_idx[:, 0]]

        loss_flow = ((X_next - X_nn) ** 2).mean()
        loss_reg = (model.U ** 2).mean() + (model.V ** 2).mean()

        loss = loss_flow + 1e-3 * loss_reg

        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    return model


# =========================
# RUN
# =========================
model = train(X, X_TF, M)


# =========================
# SAVE OUTPUTS
# =========================
V_pred, A = model(torch.tensor(X_TF, dtype=torch.float32))

np.save("A.npy", A.detach().numpy())
np.save("V_pred.npy", V_pred.detach().numpy())

print("DONE")
