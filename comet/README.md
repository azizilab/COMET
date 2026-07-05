# COMET

COMET learns a gene-regulatory trajectory model from **unpaired** scRNA-seq +
scATAC-seq hematopoiesis data. Given a Decipher 2-D/latent embedding of the
RNA cells, it learns:

- **A small dictionary of GRN "atoms"** (`D_ATOMS`, default 4) — each atom is
  a gene × TF regulatory matrix. Every cell is a soft mixture over these
  atoms (`mix_weights`, from `CellStateEncoder`).
- **A steady-state expression predictor** (`GRNDictionary.predict_rna`) — TF
  expression × a cell's mixed GRN → predicted target-gene expression.
- **A velocity head** (`DictVelocityHead`) — maps the disequilibrium between
  observed and predicted steady-state expression to a gene-space velocity
  vector, nudged by chromatin accessibility (open chromatin = can turn on;
  closed = shouldn't).
- **A 2-D linear decoder** (`LinearDecoder`) for an interpretable biplot of
  the latent cell state.

Training is two phases, run back-to-back inside a single `run_training()`
call:

1. **Phase 1** (`PHASE1_EPOCHS` epochs): trains the GRN dictionary, the cell
   encoder, and the LD decoder on masked reconstruction + sparsity/diversity/
   entropy/Dirichlet regularizers.
2. **Phase 2** (remaining epochs): unfreezes `DictVelocityHead` and adds
   velocity-consistency, ATAC-direction, smoothness, density, and divergence
   losses on top of the Phase 1 losses.

Each run saves its checkpoint (`V_pred.npy`, `A_dict.npy`, `mix_weights.npy`,
`steady_state.npy`, gene/TF index arrays, the 2-D LD embedding, a training
GIF, etc.) to its own `outputs/run<i>/` directory and is skipped on re-run if
`V_pred.npy` already exists.

Downstream, `tools.compute_comet_pseudotime` turns the learned velocity field
into a pseudotime ordering (root = cells dominated by the least-used GRN
atom, direction = Dijkstra shortest path along the velocity-aligned 2-D
graph) — no separate psi/Markov training step required.

## Setup

```bash
pip install torch numpy pandas scipy scikit-learn scanpy anndata matplotlib Pillow scikit-misc
```

`scikit-misc` is required by scanpy's `seurat_v3` HVG flavor (`HVG_FLAVOR` in
`config.py`) — without it, `data/load.py` fails at the
`highly_variable_genes` step with `ModuleNotFoundError: No module named
'skmisc'`.

## Running on the default (Velten hematopoiesis) dataset

The default dataset ships in `data/` and `config.py` already points at it.
From the repo root:

```bash
python comet.py
```

This loads and preprocesses the data (`data/load.py`), trains `NUM_RUNS`
independent models (default 5, `N_EPOCHS=120` each), and writes each run's
outputs to `outputs/run0/`, `outputs/run1/`, etc. A run that already has
`V_pred.npy` is skipped, so re-running `python comet.py` resumes rather than
retrains from scratch.

To iterate faster (e.g. to sanity-check the pipeline), edit `config.py` and
lower `NUM_RUNS`, `N_EPOCHS`/`PHASE1_EPOCHS`, or `BATCH_SIZE` before running.

## Running on a new dataset

1. **Prepare inputs** in the same shapes/formats as the default dataset:
   - `RNA_H5AD` — AnnData `.h5ad`, cells × all genes, raw or log-normalizable
     counts in `.X`, gene symbols in `.var_names`.
   - `PEAK_TF_MTX` / `TF_NAMES_TXT` — sparse peak × TF motif-hit matrix
     (MatrixMarket `.mtx`) and the corresponding TF name list.
   - `PEAK_GENE_CSV` / `PEAKS_TXT` — tab-separated `peak, gene` links, and
     the ordered peak list matching `PEAK_TF_MTX`'s rows.
   - `ATAC_MTX` / `ATAC_PEAKS_TXT` — scATAC peak × cell matrix (`.mtx.gz`)
     and its peak table (needs a `Feature` column in `chr_start_end` form).
   - `DECIPHER_NPZ` — a `.npz` with a `z` array (N × latent_dim, e.g. 10) and
     a `v` array (N × 2) — the Decipher latent and 2-D visualization
     coordinates for the RNA cells, in the same cell order as `RNA_H5AD`.
2. **Point `config.py` at the new files** — update the paths under
   `# Data paths` to your new files (they're resolved relative to the repo
   root). If your AnnData doesn't have a pseudotime or FACS-gate `.obs`
   column, set `OBS_PSEUDOTIME_KEY` to `None` and adjust/clear
   `OBS_FACS_KEYS` — these are only used for optional diagnostics/plots, not
   training.
3. **Run it**:
   ```bash
   python comet.py
   ```
   Model architecture (`D_ATOMS`, hidden sizes), training schedule, and loss
   weights all come from `config.py` and don't need to change for a new
   dataset unless cell/gene counts are very different from the default (in
   which case `BATCH_SIZE`, `N_CLUSTERS`, or the `WEIGHTS_PHASE1`/
   `WEIGHTS_PHASE2` dicts may need retuning).

## Post-training analysis

Everything in `tools/` operates on the saved `outputs/run<i>/*.npy`
checkpoint files plus the `data` dict from `data.load.load_all_data`:

```python
from config import CONFIG
from data.load import load_all_data
from tools import (
    compute_comet_pseudotime, assign_velten_celltypes,
    plot_atom_weights, plot_pseudotime, plot_velocity_streamlines,
    plot_marker_expression, analyze_transition_zones,
    analyze_trajectory_genes, grn_sparsity_report,
    compare_to_celloracle, plot_grn_consistency,
)
import numpy as np

data   = load_all_data(CONFIG)
run    = f"{CONFIG['OUTPUT_DIR']}/run0"
V_pred = np.load(f"{run}/V_pred.npy")
V_dec2 = np.load(f"{run}/V_dec2.npy")
mix_w  = np.load(f"{run}/mix_weights.npy")

pt = compute_comet_pseudotime(V_pred, data["X"], V_dec2, mix_w)
```

## Reproducing Plots

This walks through regenerating every plot in `model_5_30_26_rerun.md` (the
notebook the current `training/`, `model/`, `tools/`, and `viz/` modules were
refactored from) using only the checkpoint files `python comet.py` writes to
`outputs/run<i>/`.

Two of the notebook's plots need no extra command — they're written
automatically as training runs (`PLOT_EVERY` in `config.py` controls the GIF
frame rate) and land directly in the run's output directory:

- `outputs/run0/velocity_training.gif` — per-epoch velocity field snapshots
- `outputs/run0/velocity_slide_final.png` — the GIF's final frame, upscaled

Everything else is a `tools`/`model`/`viz` call over the saved `.npy`
checkpoints. Assuming `python comet.py` has already completed and produced
`outputs/run0/` (repeat with `run1`, `run2`, ... for other runs), run the
following from the repo root:

```python
import os
import numpy as np

from config import CONFIG
from data.load import load_all_data
from model.velocity import project_to_2d
from model.ld_vae import plot_ld_embedding
from tools import (
    compute_comet_pseudotime, assign_velten_celltypes,
    plot_loss_curves, plot_atom_weights, plot_pseudotime,
    plot_velocity_streamlines, plot_marker_expression,
    analyze_transition_zones, analyze_trajectory_genes,
    grn_sparsity_report, compare_to_celloracle, plot_grn_consistency,
)

run    = f"{CONFIG['OUTPUT_DIR']}/run0"     # -> outputs/run0
plots  = f"{run}/plots"                     # all figures below land here
os.makedirs(plots, exist_ok=True)

data   = load_all_data(CONFIG)
X      = data["X"]
A_genes, A_tfs = data["A_genes"], data["A_tfs"]
rna_adata      = data["rna_adata"]

V_pred    = np.load(f"{run}/V_pred.npy")
V_dec2    = np.load(f"{run}/V_dec2.npy")
mix_w     = np.load(f"{run}/mix_weights.npy")
A_dict_np = np.load(f"{run}/A_dict.npy")        # (D, G, T)
z_ld_2d   = np.load(f"{run}/z_ld_2d.npy")
gene_loadings = np.load(f"{run}/gene_loadings.npy")

# ── 1. Training loss curves (3-panel Phase 1/2 convergence) ────────────────
import json
with open(f"{run}/loss_history.json") as f:
    loss_history = json.load(f)
plot_loss_curves(loss_history, run_name="Run 0",
                  save_path=f"{plots}/loss_curves.png")

# ── 2. LD-VAE biplot (RNA/ATAC embedding + gene-loading arrows) ────────────
pt_key = CONFIG["OBS_PSEUDOTIME_KEY"]
pseudo = rna_adata.obs[pt_key].astype(np.float32).values
pseudo_norm = (pseudo - pseudo.min()) / (pseudo.max() - pseudo.min() + 1e-8)
plot_ld_embedding(
    z_2d_rna=z_ld_2d, pseudo_norm=pseudo_norm,
    gene_loadings=gene_loadings, A_genes=A_genes,
    highlight_genes=CONFIG["LD_HIGHLIGHT_GENES"],
    save_path=f"{plots}/ld_embedding_biplot.png",
)

# ── 3. Per-atom GRN mixing-weight spatial plots ────────────────────────────
origin_atom = int(np.argmin(mix_w.mean(axis=0)))
plot_atom_weights(V_dec2, mix_w, CONFIG["D_ATOMS"], origin_atom,
                   A_tfs=A_tfs, A_dict_np=A_dict_np,
                   save_path=f"{plots}/atom_weights_spatial.png")

# ── 4. Cell-type assignment (Velten 2017 FACS gates) ───────────────────────
X_full = np.asarray(rna_adata.X.toarray() if hasattr(rna_adata.X, "toarray")
                     else rna_adata.X, dtype=np.float32)
gene_to_idx_full = {g: i for i, g in enumerate(rna_adata.var_names)}
cell_types = assign_velten_celltypes(rna_adata, V_dec2, X_full, gene_to_idx_full,
                                      save_path=f"{plots}/celltype_assignment.png")

# ── 5. Velocity pseudotime + streamlines ───────────────────────────────────
pt = compute_comet_pseudotime(V_pred, X, V_dec2, mix_w, cell_types=cell_types)
plot_pseudotime(V_dec2, pt, cell_types, facs_pseudotime=pseudo,
                 save_path=f"{plots}/pseudotime.png")

V_smooth = project_to_2d(V_pred, V_dec2, X, k=12, scale=12.0)
plot_velocity_streamlines(V_dec2, V_smooth, all_weights=mix_w,
                           cell_types=cell_types, facs_pseudotime=pseudo,
                           space_label="Decipher", base_path=plots)

# ── 6. Marker gene expression in Decipher space ────────────────────────────
plot_marker_expression(rna_adata, V_dec2,
                        save_path=f"{plots}/marker_expression.png")

# ── 7. Transition-zone analysis (GRN entropy vs. velocity magnitude) ───────
analyze_transition_zones(mix_w, V_smooth, V_dec2, CONFIG["D_ATOMS"],
                          save_path=f"{plots}/transition_zones.png")

# ── 8. Per-lineage trajectory genes (one figure per lineage) ───────────────
analyze_trajectory_genes(X, V_dec2, cell_types, A_genes, pt,
                          save_path=f"{plots}/trajectory_genes.png")

# ── 9. GRN sparsity report (printed, no figure) ────────────────────────────
A_learned = np.einsum('d,dgt->gt', mix_w.mean(axis=0), A_dict_np)  # consensus GRN
grn_sparsity_report(A_learned, A_genes, A_tfs, data["M"])

# ── 10. CellOracle comparison + inter-run GRN consistency ──────────────────
co_dir    = CONFIG["CELLORACLE_A_DIR"]     # data/cell_oracle_run/cell_oracle_global_A
co_A      = np.load(f"{co_dir}/A_celloracle.npy")
co_genes  = np.load(f"{co_dir}/A_genes.npy", allow_pickle=True)
co_tfs    = np.load(f"{co_dir}/A_tfs.npy", allow_pickle=True)
tf_df, A_co_shared, common_genes, common_tfs = compare_to_celloracle(
    A_learned, A_genes, A_tfs, co_A, co_genes, co_tfs,
    save_path=f"{plots}/celloracle_comparison.png",
)

# Needs NUM_RUNS > 1 in config.py (one outputs/run<i>/ per run already trained)
run_matrices = []
for i in range(CONFIG["NUM_RUNS"]):
    r  = f"{CONFIG['OUTPUT_DIR']}/run{i}"
    Ad = np.load(f"{r}/A_dict.npy")
    w  = np.load(f"{r}/mix_weights.npy")
    run_matrices.append((
        np.einsum('d,dgt->gt', w.mean(axis=0), Ad),
        np.load(f"{r}/A_genes.npy"), np.load(f"{r}/A_tfs.npy"),
    ))
plot_grn_consistency(run_matrices, A_co_shared, common_genes, common_tfs,
                      save_path=f"{plots}/grn_consistency.png")
```

All figures land in `outputs/run0/plots/` (created above); `analyze_trajectory_genes`
additionally writes one `trajectory_genes_<lineage>.png` per non-CLOUD lineage
(e.g. `trajectory_genes_HSC.png`, `trajectory_genes_GMP.png`).
