# COMET — Run Guide

Unpaired scRNA + scATAC trajectory inference for hematopoiesis. Directionality emerges from ATAC–RNA discordance without pseudotime, root cells, or splicing.

All paths are relative to `comet/` unless noted. Run everything from that directory.

---

## 1. Environment Setup

### Conda / mamba (recommended)

```bash
mamba create -n comet python=3.11
mamba activate comet
```

### Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r ../requirements.txt
```

`requirements.txt` covers: `numpy pandas scipy anndata torch scikit-learn matplotlib pillow seaborn scanpy`

### Verify

```bash
python -c "import torch, anndata, scanpy, seaborn; print('OK')"
```

---

## 2. Required Data Files

Place all data files in `comet/data/`. The paths are resolved automatically by `config.py` via `pathlib.Path(__file__).parent / "data"` — no hardcoded absolute paths.

### Primary data files

| Filename | Format | Contents |
|---|---|---|
| `velten_rna.h5ad` | AnnData H5AD | scRNA-seq cells × genes; `.obs` should contain FACS columns (`facs_cd34`, `facs_cd38`, `facs_cd90`, `facs_cd135`, `facs_cd45ra`, `facs_lin`) and optionally `facs_pseudotime` |
| `GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz` | Market Matrix (gzipped) | scATAC peaks × cells (transposed to cells × peaks in code) |
| `GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz` | TSV (gzipped) | ATAC peak metadata; must have a `Feature` column in `chr_start_end` format |
| `peak_tf.mtx` | Market Matrix | peaks × TFs motif binary matrix |
| `tf_names.txt` | Plain text | TF names, one per line, ordered to match `peak_tf.mtx` columns |
| `peak_gene.csv` | TSV | Two columns: `peak` (chr:start-end format), `gene` |
| `peaks.txt` | Plain text | Master peak list (chr:start-end), one per line, no header |
| `decipher_embeddings.npz` | NumPy NPZ | Pre-computed Decipher outputs; **must contain exactly the keys `"z"` (latent, shape N×10) and `"v"` (vis-space, shape N×2)**; cell order must match `velten_rna.h5ad` |

The code validates the `.npz` keys at load time and raises a clear `KeyError` with instructions if either key is absent.

### CellOracle comparison files (Step 17)

| Filename | Format | Contents |
|---|---|---|
| `cell_oracle_run/cell_oracle_global_A/A_celloracle.npy` | NumPy | GRN weight matrix, shape `(G_co, T_co)` or `(T_co, G_co)` |
| `cell_oracle_run/cell_oracle_global_A/A_genes.npy` | NumPy | Gene names |
| `cell_oracle_run/cell_oracle_global_A/A_tfs.npy` | NumPy | TF names |

These three files are not needed for training — only for `compare_to_celloracle`.

### Files not in this repository

All raw data files listed above must be obtained separately:
- GEO accession GSE129785 (scATAC matrix and peaks)
- Velten 2017 scRNA-seq (hematopoiesis)
- Motif-to-peak and peak-to-gene preprocessing outputs
- `decipher_embeddings.npz` requires running the Decipher model first; save its output with `np.savez("decipher_embeddings.npz", z=Z_latent, v=V_2d)`
- CellOracle GRN outputs

No trained model weights (`.pt`) are saved by the current pipeline.

---

## 3. Configuration

All hyperparameters are in `config.py`. Data paths resolve relative to `comet/data/` automatically.

```python
# config.py — current values (matches updated_logic_displacement.md)

MODEL = {
    "D_ATOMS": 4,
    "LD_HIDDEN": 64,
}

TRAIN = {
    "NUM_RUNS": 5,
    "N_EPOCHS": 120,
    "PHASE1_EPOCHS": 50,
    "BATCH_SIZE": 512,
    "LR_GRN": 5e-4,
    "CLIP_NORM": 5.0,
    "USE_LOCAL_ATAC": True,
    "N_CLUSTERS": 20,
    "NEIGHBOR_SPACE": "latent",

    "WEIGHTS_PHASE1": { "MASK":2.0, "SPARSE":5e-3, "ACCESS":0.1,
                        "DIVERSE":10.0, "ENTROPY":1.0, "DIRICH":2.0,
                        "LD":0.5, "LD_REG":0.1 },
    "WEIGHTS_PHASE2": { "MASK":1.0, "SPARSE":5e-3, "ACCESS":0.1,
                        "DIVERSE":5.0, "DIV":0.5, "DIRICH":1.0,
                        "LD":0.3, "LD_REG":0.05 },

    "PHASE3_EPOCHS": 500,
    "PHASE3_LR": 1e-1,
    "K_PSI_SMOOTH": 30,
    "N_PSI_SMOOTH_HOPS": 3,
    "WEIGHTS_PHASE3": { "POLARITY":1.0, "FINE_ORDER":0.0, "EPS_REG":0.1 },

    "K_FLOW_2D": 20,
    "N_DIFFUSION_HOPS": 4,
    "DIFFUSION_BETA": 0.4,
    "PSI_EPS": 1e-6,
    "LR_SCHED": { "ETA_MIN": 1e-3 },

    "OUTPUT_DIR": "bert_grn_out_615",
}
```

---

## 4. Data Loading & Preprocessing

### Notebook API (recommended)

Use `load_all_comet_data()` to run all four preprocessing steps in one call and
receive the three primary data objects as named return values.
`rna_adata` is returned explicitly here and **must be defined before any downstream
tool that accepts it** (cell annotation, velocity streamlines, pseudotime plots, etc.).

```python
import sys
sys.path.insert(0, "/path/to/comet")   # adjust to your checkout

from data.load import load_all_comet_data

# All four preprocessing steps run automatically.
# rna_adata      — AnnData (N cells × all_genes); carries .obs FACS columns
# data           — fully assembled dict ready for run_training()
# decipher_data  — NpzFile with keys 'z' (N×10 latent) and 'v' (N×2 vis-space)
rna_adata, data, decipher_data = load_all_comet_data()

# Pull frequently-used objects into named variables for downstream cells.
V_dec2           = decipher_data["v"].astype("float32")  # (N, 2)
X_full           = data["X_full"]                        # (N, all_genes)
gene_to_idx_full = data["gene_to_idx_full"]              # {gene: col_index}
A_genes          = data["A_genes"]                       # (G,) active gene names
A_tfs            = data["A_tfs"]                         # (T,) TF names
M                = data["M"]                             # (G, T) binary motif mask
gene_idx         = data["gene_idx"]                      # (G,) positions in X_full
```

### Script API (CLI)

`train.py` calls the same four functions in sequence. No manual preprocessing step
is needed beyond placing the raw files in `comet/data/`.

### Preprocessing step details

**Step 1 — RNA universe (`build_universe`)**
- Reads `velten_rna.h5ad` → `X_full` (N × all_genes)
- Filters to `A_genes` (genes with ≥1 peak–gene link) and `A_tfs` (TFs present in RNA)
- Expected result: ~11,483 genes, ~277 TFs

**Step 2 — Motif mask (`build_M`)**
- Binary `M` (G × T): `M[g, t] = 1` if TF t has a motif in any peak linked to gene g
- Expected: `M.shape == (11483, 277)`, sparsity ≈ 0.30

**Step 3 — ATAC gene activity (`build_gene_priors`)**
- Reads ATAC matrix (cells × peaks), multiplies by peak–gene map, L1-normalizes, log1p
- Clips to first N_rna cells (RNA and ATAC must share cell order)
- Returns `gene_activity` (N × G), `gene_access` (G,)

**Step 4 — TF accessibility prior (`build_tf_access_prior`)**
- Averages `gene_activity` columns per TF locus → min-max normalizes to [0, 1]

**If `USE_LOCAL_ATAC = True` (default):** `build_local_atac` produces per-cell accessibility (N × G) via 20-cluster pseudo-bulk matching inside `run_training`.

---

## 5. Training

### From a notebook (recommended after Section 4 notebook setup)

```python
from training.training_loop import run_training
from config import MODEL, TRAIN

# data is the dict returned by load_all_comet_data() in Section 4.
outputs = run_training(data, MODEL, TRAIN)

# outputs keys: 'psi', 'A_learned', 'mix_weights',
#               'loss_histories', 'flow_histories'
```

### From the command line

```bash
cd comet/
python train.py
```

No CLI arguments. All configuration is in `config.py`.

### Resume / checkpoint behavior

Each run checks for `{OUTPUT_DIR}/run{i}/psi.npy`.

**If the checkpoint exists**, `run_training()` loads `psi.npy`, `mix_weights.npy`, and `A_learned.npy` from disk and skips all training for that run. The loaded values are returned in the `outputs` dict exactly as if training had just finished.

**Degenerate-checkpoint recovery.** A checkpoint where `psi.max() − psi.min() < 0.1` indicates a flat field (no gradient → no velocity arrows, broken pseudotime). This can happen when the origin atom collapses to zero mixing-weight usage during Phases 1/2. `run_training()` detects this automatically and re-runs Phase 3 in-place using only the saved `mix_weights.npy` — Phases 1/2 are not re-run. If the saved `origin_atom` itself has no dominant cells, a new origin atom is selected (lowest mean weight among atoms with ≥5% of cells) before Phase 3 is re-run. The corrected `psi.npy` and `flow_field_params.json` are overwritten with the new values.

**To force full retraining** from scratch, delete or rename the run directory (e.g. `rm -rf bert_grn_out_615/run0/`) before calling `run_training()`.

### Training pipeline (per run)

**Phase 1 — GRN + encoder warmup (epochs 0–49)**
- AdamW, lr = 5e-4, weight_decay = 1e-4
- Temperature anneals 0.5 → 0.1 over 50 epochs
- Loss: masked reconstruction (2.0) + sparsity (5e-3) + TF access (0.1) + diversity (10.0) + normalized entropy (1.0) + Dirichlet (2.0) + LD reconstruction (0.5) + LD regularization (0.1)
- BERT masking: 30% of genes per cell per batch

**Phase 2 — Refinement (epochs 50–119)**
- AdamW, lr = 2e-4, weight_decay = 1e-4; temperature fixed at 0.2
- Same objective as Phase 1 with reduced weights; no velocity head

**Phase 3 — Markov Progression Field (500 full-batch steps)**
- GRN and encoder frozen
- `compute_global_disc_order` identifies `origin_atom` (highest ATAC-RNA discordance)
- `MarkovProgressionField`: D−1 = 3 `eps_raw` scalars; `P[origin] = −1`, `P[d] = 1 + 0.15 × tanh(eps_raw[d])`
- Adam, lr = 0.1, cosine annealing to eta_min = 1e-3
- Loss: polarity hinge (1.0) + eps_raw regularization (0.1); fine_order disabled (0.0)
- ψ smoothed via 3-hop latent-space kNN (K = 30) before each loss step

**Per-run initialization (before Phase 1)**
- D-cluster KMeans in Decipher 2D space → `W_init` (D × G × T) via LFC profiles
- Encoder pre-trained 30 epochs on cluster cross-entropy

---

## 6. Checkpoint Outputs

Every run saves to its own `{OUTPUT_DIR}/run{i}/` subdirectory. Run 0 saves to `run0/`, not directly to `OUTPUT_DIR/`.

| File | Shape | Description |
|---|---|---|
| `psi.npy` | (N,) | Final smoothed ψ field; also the resume sentinel |
| `A_learned.npy` | (G, T) | Population-weighted mean GRN |
| `mix_weights.npy` | (N, D) | Per-cell atom mixing weights at temperature = 0.1 |
| `gene_idx.npy` | (G,) | Indices of `A_genes` into the full RNA gene list |
| `tf_idx.npy` | (T,) | Indices of `A_tfs` into the full RNA gene list |
| `A_genes.npy` | (G,) | Gene names aligned to `A_learned` columns |
| `A_tfs.npy` | (T,) | TF names aligned to `A_learned` rows |
| `z_ld_2d.npy` | (N, 2) | Linear decoder 2D embedding of RNA cells |
| `gene_loadings.npy` | (G, 2) | LD-VAE decoder weight matrix (`W_dec.weight`) |
| `loss_history.npy` | (N_EPOCHS,) | Total loss per epoch (Phases 1+2) |
| `flow_field_params.json` | — | `origin_atom`, `P` (progression vector), `disc`, `direction_matrix` |

**Conditionally generated** (only when `facs_pseudotime` is in `rna_adata.obs`):

| File | Description |
|---|---|
| `ld_embedding_biplot.png` | 3-panel LD-VAE biplot (pseudotime, modality split, gene loading arrows) |

---

## 7. Velocity Inference

```python
import numpy as np
from model.velocity import build_2d_neighbor_index, compute_markov_velocity_2d

OUTPUT_DIR = "bert_grn_out_615"
psi        = np.load(f"{OUTPUT_DIR}/run0/psi.npy")       # (N,)
V_dec2     = np.load("data/decipher_embeddings.npz")["v"] # (N, 2)

neighbor_idx_2d = build_2d_neighbor_index(V_dec2, k=20)   # K_FLOW_2D

V_raw, V_smooth, T_markov = compute_markov_velocity_2d(
    psi             = psi,
    X_2d            = V_dec2,
    neighbor_idx_2d = neighbor_idx_2d,
    n_hops          = 4,    # N_DIFFUSION_HOPS
    beta            = 0.4,  # DIFFUSION_BETA
    eps             = 1e-6, # PSI_EPS
)
```

Key design:
- 2D kNN only for arrow geometry — latent kNN is used for ψ smoothing (during training)
- ReLU kernel: `T[i→j] ∝ ReLU(ψ(j) − ψ(i))` — zero for downhill neighbors
- Row-stochastic T so diffusion `V_{h+1} = (1−β)V_h + β T V_h` is a proper convex combination

---

## 8. Pseudotime Computation

```python
from model.velocity import infer_velocity_pseudotime

pseudotime = infer_velocity_pseudotime(
    V_dec2          = V_dec2,
    psi_arr         = psi,
    neighbor_idx_2d = neighbor_idx_2d,
    cell_types      = cell_types,   # optional, for logging
    root_pct        = 0.05,
)
# Returns (N,) float32, normalized to [0, 1]
```

Algorithm: multi-source Dijkstra on ψ-uphill directed graph. Roots = lowest 5% of ψ (guaranteed to be origin-atom-dominant cells). Edge cost = Δψ / Euclidean 2D distance. Unreachable cells assigned max pseudotime × 1.05.

---

## 9. Downstream Analyses

All downstream functions are in `tools/`. Import from within `comet/`:

```python
import sys; sys.path.insert(0, ".")
from tools import (
    assign_velten_celltypes,
    plot_loss_curves, plot_phase3_loss, plot_atom_weights,
    plot_psi_field, plot_velocity_streamlines, plot_pseudotime,
    plot_marker_expression,
    analyze_transition_zones, analyze_trajectory_genes,
    top_tf_targets, grn_sparsity_report,
    compare_to_celloracle, plot_grn_consistency,
)
```

### 9.1 Cell type annotation

`rna_adata`, `V_dec2`, `X_full`, and `gene_to_idx_full` are all returned by
`load_all_comet_data()` (Section 4) and must be defined before this call.

```python
# These four variables come from Section 4:
#   rna_adata, data, decipher_data = load_all_comet_data()
#   V_dec2           = decipher_data["v"]
#   X_full           = data["X_full"]
#   gene_to_idx_full = data["gene_to_idx_full"]

cell_types = assign_velten_celltypes(
    rna_adata        = rna_adata,
    V_dec2           = V_dec2,
    X_full           = X_full,
    gene_to_idx_full = gene_to_idx_full,
    save_path        = f"{OUTPUT_DIR}/cell_types.png",
)
# Returns (N,) object array: 'HSC','MPP','MLP','CMP','MEP','GMP','CLOUD'
```

Gate logic (rank-normalized to [0, 1] within dataset):

| Cell type | Gate |
|---|---|
| HSC | lin− cd34+ cd38lo cd90+ cd45ra− |
| MPP | lin− cd34+ cd38lo cd90− cd45ra− |
| MLP | lin− cd34+ cd38lo cd45ra+ |
| CMP | lin− cd34+ cd38hi cd135+ cd45ra− |
| MEP | lin− cd34+ cd38hi cd135− cd45ra− |
| GMP | lin− cd34+ cd38hi cd45ra+ |
| CLOUD | everything else → rescued by marker z-score (threshold 0.5) |

### 9.2 Training curve plots

```python
plot_loss_curves(
    loss_history = outputs["loss_histories"]["run0"],
    run_name     = "Run 0",
    save_path    = f"{OUTPUT_DIR}/run0/loss_curves.png",
)
plot_phase3_loss(
    flow_history = outputs["flow_histories"]["run0"],
    run_name     = "Run 0",
    save_path    = f"{OUTPUT_DIR}/run0/phase3_loss.png",
)
```

### 9.3 Atom mixing weights

```python
import json
mix_weights = np.load(f"{OUTPUT_DIR}/run0/mix_weights.npy")
with open(f"{OUTPUT_DIR}/run0/flow_field_params.json") as f:
    origin_atom = json.load(f)["origin_atom"]

plot_atom_weights(
    V_dec2      = V_dec2,
    all_weights = mix_weights,
    D_ATOMS     = 4,
    origin_atom = origin_atom,
    save_path   = f"{OUTPUT_DIR}/run0/atom_weights.png",
)
```

### 9.4 ψ field

```python
plot_psi_field(
    V_dec2      = V_dec2,
    psi_np      = psi,
    all_weights = mix_weights,
    D_ATOMS     = 4,
    origin_atom = origin_atom,
    save_path   = f"{OUTPUT_DIR}/run0/psi_field.png",
)
```

### 9.5 Velocity streamlines

```python
# rna_adata from load_all_comet_data(); cell_types from assign_velten_celltypes()
plot_velocity_streamlines(
    X_2d            = V_dec2,
    V_smooth        = V_smooth,
    cell_types      = cell_types,
    facs_pseudotime = rna_adata.obs.get("facs_pseudotime"),
    space_label     = "Decipher",
    base_path       = f"{OUTPUT_DIR}/run0",
    save            = True,
)
# Auto-saves to: {OUTPUT_DIR}/run0/velocity_streamlines_decipher.png
```

Uses `integration_direction='forward'` to prevent backward-loop artifact.

### 9.6 Pseudotime

```python
# rna_adata from load_all_comet_data(); pseudotime from infer_velocity_pseudotime()
plot_pseudotime(
    V_dec2          = V_dec2,
    pseudotime      = pseudotime,
    cell_types      = cell_types,
    psi_arr         = psi,
    root_pct        = 0.05,
    facs_pseudotime = rna_adata.obs.get("facs_pseudotime"),
    save_path       = f"{OUTPUT_DIR}/run0/pseudotime.png",
)
```

If `facs_pseudotime` is provided, prints Spearman ρ as a validation metric.

### 9.7 Marker expression

```python
# rna_adata from load_all_comet_data()
plot_marker_expression(
    rna_adata = rna_adata,
    V_dec2    = V_dec2,
    key_genes = ['CD34', 'GATA1', 'MPO', 'SPI1', 'GATA2', 'LMO2', 'DNTT', 'IL7R'],
    save_path = f"{OUTPUT_DIR}/run0/marker_expression.png",
)
# Requires scanpy. Sets rna_adata.obsm['X_decipher'] = V_dec2 in place.
```

### 9.8 Transition zone analysis

```python
entropy = analyze_transition_zones(
    all_weights = mix_weights,
    V_smooth    = V_smooth,
    V_dec2      = V_dec2,
    D_ATOMS     = 4,
    save_path   = f"{OUTPUT_DIR}/run0/transition_zones.png",
)
# Prints Spearman(entropy, |V_smooth|) — should be strongly positive with ReLU kernel
```

### 9.9 Trajectory gene analysis

```python
# A_genes and gene_idx can be loaded from disk or taken from data["A_genes"] / data["gene_idx"]
A_genes  = np.load(f"{OUTPUT_DIR}/run0/A_genes.npy", allow_pickle=True)
gene_idx = np.load(f"{OUTPUT_DIR}/run0/gene_idx.npy")
# X_full from load_all_comet_data() (data["X_full"]); subset to active genes
X_active = X_full[:, gene_idx]   # (N, G)

results = analyze_trajectory_genes(
    X              = X_active,
    V_dec2         = V_dec2,
    cell_types     = cell_types,
    A_genes        = A_genes,
    vel_pseudotime = pseudotime,
    n_top          = 25,
    save_path      = f"{OUTPUT_DIR}/run0/trajectory_genes.png",
)
# One 4-panel figure per lineage, saved as:
#   .../trajectory_genes_MEP.png
#   .../trajectory_genes_GMP.png
#   .../trajectory_genes_MLP.png
#   .../trajectory_genes_CMP.png
#   .../trajectory_genes_HSC.png
#   .../trajectory_genes_MPP.png
```

Ground-truth recovery uses Velten 2017 markers hardcoded in `tools/analysis.py`. Target: >20% recovery at top-25 genes.

### 9.10 GRN inspection

```python
A_learned = np.load(f"{OUTPUT_DIR}/run0/A_learned.npy")
A_tfs     = np.load(f"{OUTPUT_DIR}/run0/A_tfs.npy", allow_pickle=True)

pairs = top_tf_targets(A_learned, A_tfs, A_genes, tf_name='GATA1', n=20)
# Returns [(gene_name, weight), ...] sorted by |weight|

per_tf_active = grn_sparsity_report(A_learned, A_genes, A_tfs, M)
# Prints edge counts / motif support at multiple |w| thresholds
```

### 9.11 CellOracle comparison

```python
CO = "cell_oracle_run/cell_oracle_global_A"
co_A     = np.load(f"{CO}/A_celloracle.npy")
co_genes = np.load(f"{CO}/A_genes.npy", allow_pickle=True).astype(str)
co_tfs   = np.load(f"{CO}/A_tfs.npy",   allow_pickle=True).astype(str)

tf_df, A_co_shared, common_genes, common_tfs = compare_to_celloracle(
    A_learned = A_learned,
    A_genes   = A_genes,
    A_tfs     = A_tfs,
    co_A      = co_A,
    co_genes  = co_genes,
    co_tfs    = co_tfs,
    save_path = f"{OUTPUT_DIR}/celloracle_comparison.png",
)
# Metrics: global Spearman ρ, per-TF Spearman, Jaccard at 5/10/20%
```

### 9.12 Multi-run reproducibility (requires NUM_RUNS > 1)

```python
OUTPUT_DIR = "bert_grn_out_615"
run_matrices = []
for i in range(5):    # NUM_RUNS
    d = f"{OUTPUT_DIR}/run{i}"
    run_matrices.append((
        np.load(f"{d}/A_learned.npy"),
        np.load(f"{d}/A_genes.npy", allow_pickle=True),
        np.load(f"{d}/A_tfs.npy",   allow_pickle=True),
    ))

plot_grn_consistency(
    run_matrices = run_matrices,
    A_co_shared  = A_co_shared,
    common_genes = common_genes,
    common_tfs   = common_tfs,
    save_path    = f"{OUTPUT_DIR}/grn_consistency.png",
)
```

---

## 10. All Figures Generated

| Figure | Function | Saved to |
|---|---|---|
| LD-VAE biplot (3 panels) | `plot_ld_embedding` (auto, in training) | `{OUTPUT_DIR}/run{i}/ld_embedding_biplot.png` ¹ |
| Phase 1/2 loss curves (3 panels) | `plot_loss_curves` | caller-specified |
| Phase 3 loss curves (1 panel) | `plot_phase3_loss` | caller-specified |
| Per-atom mixing weights (D panels) | `plot_atom_weights` | caller-specified |
| ψ field dual-panel | `plot_psi_field` | caller-specified |
| Velocity streamlines (1–2 panels) | `plot_velocity_streamlines` | `{base_path}/velocity_streamlines_decipher.png` |
| Pseudotime (2 panels) | `plot_pseudotime` | caller-specified |
| Marker expression (scanpy) | `plot_marker_expression` | caller-specified |
| Transition zones (3 panels) | `analyze_transition_zones` | caller-specified |
| Trajectory genes — per lineage (4 panels × ≤6 lineages) | `analyze_trajectory_genes` | `{base}_{lineage}.png` |
| Cell type scatter (1–2 panels) | `assign_velten_celltypes` | caller-specified |
| CellOracle comparison (2 panels) | `compare_to_celloracle` | caller-specified |
| GRN consistency heatmap + scatter | `plot_grn_consistency` | caller-specified |

¹ Only produced if `facs_pseudotime` is in `rna_adata.obs`.

---

## 11. Expected Output Locations

```
bert_grn_out_615/
├── run0/                             ← all runs use run{i}/ — no asymmetry
│   ├── psi.npy
│   ├── A_learned.npy
│   ├── mix_weights.npy
│   ├── gene_idx.npy
│   ├── tf_idx.npy
│   ├── A_genes.npy
│   ├── A_tfs.npy
│   ├── z_ld_2d.npy
│   ├── gene_loadings.npy
│   ├── loss_history.npy
│   ├── flow_field_params.json
│   └── ld_embedding_biplot.png       ← only if facs_pseudotime in obs
├── run1/
│   └── ...  (same structure)
├── run2/
├── run3/
├── run4/
└── celloracle_comparison.png         ← downstream, caller-controlled
```

---

## 12. Missing Data Files

| File | Expected at | Source |
|---|---|---|
| `velten_rna.h5ad` | `comet/data/` | Velten 2017 hematopoiesis scRNA-seq |
| `GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz` | `comet/data/` | GEO accession GSE129785 |
| `GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz` | `comet/data/` | GEO accession GSE129785 |
| `peak_tf.mtx` | `comet/data/` | HOMER/JASPAR motif scan output |
| `tf_names.txt` | `comet/data/` | From motif scan |
| `peak_gene.csv` | `comet/data/` | Peak-to-gene linking output |
| `peaks.txt` | `comet/data/` | Master peak list |
| `decipher_embeddings.npz` | `comet/data/` | Run Decipher; save with keys `"z"` and `"v"` |
| `A_celloracle.npy` | `comet/cell_oracle_run/...` | CellOracle GRN output |
| `A_genes.npy` (CellOracle) | `comet/cell_oracle_run/...` | CellOracle output |
| `A_tfs.npy` (CellOracle) | `comet/cell_oracle_run/...` | CellOracle output |

No trained model weights are saved by the current pipeline.

---

## 13. Known Behavioral Notes

These items reflect intentional design decisions, not bugs.

**LD biplot conditioned on `facs_pseudotime`**
`plot_ld_embedding` is only called during training when `facs_pseudotime` is present in `rna_adata.obs`. The embedding arrays (`z_ld_2d.npy`, `gene_loadings.npy`) are always saved regardless, so the biplot can be generated manually post-training.

**`VELOCITY_SPACE` config option**
The MD spec defines `VELOCITY_SPACE = "decipher" | "umap"` to switch between Decipher and UMAP coordinates for velocity display. This switching logic is not yet wired into the visualization functions — only Decipher 2D coordinates are currently supported. The variable can be added to `config.py` as a future extension point.
