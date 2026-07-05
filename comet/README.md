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
