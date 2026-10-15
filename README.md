# COMET

COMET infers differentiation trajectories from single-cell RNA + ATAC (unpaired or paired). It needs no
cell-type labels or reference pseudotime. Separately, it can learn a per-cell gene regulatory network (GRN
dictionary); the GRN does not feed into the trajectory.

The trajectory has three parts:

1. **Origin.** The origin is found from chromatin openness. An origin score φ = s·[z(entropy) + z(breadth)]
   is computed on a 120-cluster pseudo-bulk ATAC profile. The root is a consensus over K ∈ [3, 10] × 10
   seeds: KMeans the state space, take the cluster with the highest *median* φ, then take the 5 % of cells
   geodesically closest to its medoid. Sinks are low-φ cells far from the root,
   chosen by the same kind of consensus: over the same K range × seeds, KMeans the candidates into K groups and take
   the farthest cells of each, so no number of branches is assumed.
2. **Pseudotime.** The harmonic committor q on a kNN graph: 0 at the root, 1 at the sinks.
3. **Velocity field.** V = unit[(1 − w)·ū + λ·w·ĉ].
   - ū is the smoothed gene-space committor direction.
   - ĉ is an offset-free chromatin lead (what ATAC predicts for RNA beyond the current RNA).
   - w > 0 only on committor plateaus, and only when ATAC and RNA are coupled (median per-gene correlation ≥ 0.1).

Nothing in the trajectory is trained. The only per-dataset choice is the sign s of the origin assumption
(`config.ORIGIN_SIGN`): +1 means the origin is the most open chromatin (normal differentiation); −1 is used for
aml89, where the least open chromatin is taken as the origin.

ATAC is always used as a pseudo-bulk over clusters of RNA cells. For unpaired data each cluster gets the mean of the
most similar ATAC cells; for paired data (ma2020) it gets the mean of its own cells' ATAC.

A step-by-step description of the whole pipeline (inputs, preprocessing, trajectory, GRN, plots), with equations,
is in [`method.md`](method.md). The development history and validation results are in
[`trajectory_benchmarking/final_method_writeup.md`](trajectory_benchmarking/final_method_writeup.md).

## Install

```bash
pip install -r requirements.txt
```

`generate_priors.py` (needed only to build a GRN prior for a new dataset) also needs `celloracle` and `genomepy`.

## Running

All commands run from `comet/`:

```bash
cd comet

# trajectory only (pseudotime + velocity field), a few minutes per dataset
python train.py --datasets hema aml89 paul15 --trajectory-only

# trajectory + GRN dictionary training
python train.py --datasets hema

# no --datasets: runs config.RUN_DATASETS
python train.py

# a dataset that is not built in (file layout: comet/README_DATASETS.md)
python train.py --new-dataset name=my_ds base=/path/to/my_ds cell_type_col=cell_type pseudotime_col=pseudotime
```

Built-in datasets: `hema`, `aml89`, `paul15`, `ma2020` and `nestorowa`. Their paths are in `config.BASES`
(`comet/scripts/build_nestorowa.py` builds the nestorowa directory from public data). `--out-dir DIR` writes to
`DIR/<name>/` instead of `config.OUT_DIR`.
Cell types and the reference pseudotime are loaded only for plots and validation; the method never uses them.

A finished run is loaded from disk instead of recomputed. Set `FORCE_RERUN = True` in `config.py`, or delete
the dataset's output folder, to rerun.

### From Python

```python
import train
bundles, results = train.run(["hema"], train=False)       # same as --trajectory-only
q, V = results["hema"]["q_commit"], results["hema"]["V_pred"]
```

To run the trajectory directly on a loaded `Bundle`:

```python
from data.load import LOADERS
from config import BASES
from utils.committor import moment_operator
from utils.trajectory import final_trajectory

b = LOADERS["hema"](BASES["hema"])
traj = final_trajectory(b, b.Z_dec, moment_operator(b.Z_dec, 15), sign=+1)
traj["q"], traj["root_mask"], traj["sink_mask"], traj["V"]
```

## Outputs

Each dataset writes to `comet/<OUT_DIR>/<name>/` (`OUT_DIR = "comet_out_trajectory_final"`):

| file | content |
|---|---|
| `committor_q.npy` | pseudotime q ∈ [0, 1] |
| `root_mask.npy`, `sink_mask.npy` | origin and terminal cells |
| `phi.npy` | origin score φ (root-high) |
| `root_freq.npy`, `sink_freq.npy` | how often each cell was selected by the consensus root / sinks |
| `V_pred.npy` | gene-space velocity field V (N × G, unit rows) |
| `traj_u_bar.npy`, `traj_lead.npy`, `traj_w.npy` | smoothed committor direction ū, chromatin lead ĉ, per-cell lead weight w |
| `trajectory_mode.txt` | ATAC↔RNA coupling, lead gate, K range |
| `A_learned.npy`, `A_dict.npy`, `A_base.npy`, `mix_weights.npy`, `A_genes.npy`, `A_tfs.npy` | learned GRN: mean effective GRN, dictionary atoms, base, per-cell atom weights, gene / TF names (training runs only) |
| `steady_state.npy`, `loss_history.npy` | GRN-predicted RNA, training loss (training runs only) |

Figures:

| file | content |
|---|---|
| `<name>/<name>_clean_quiver.png` | V projected onto each display embedding; star = root centroid |
| `<name>/<name>_celltypes_pseudotime.png` | cell types (labelled) next to the pseudotime q, with root and sink cells outlined |
| `velocity_clean_quiver_all_embeddings.png` | the clean_quiver rows of every dataset in the run, in one figure |

`comet/results_final/` holds a full run (trajectory + GRN) of all five datasets with this code:
`python train.py --datasets hema aml89 paul15 ma2020 nestorowa --out-dir results_final`.

## Configuration (`comet/config.py`)

| setting | default | meaning |
|---|---|---|
| `TRAIN_GRN` | `True` | `False` = trajectory only (same as `--trajectory-only`) |
| `ORIGIN_SIGN` | +1; −1 aml89 | origin assumption (most / least open chromatin) |
| `TRAJ_ATAC_RES` | `(120, 8, 15)` | origin-score pseudo-bulk: clusters, ATAC cells per cluster, kNN smooth |
| `TRAJ_ROOT_FRAC` | `0.05` | root size; the consensus K range is [3, 1/(2·frac)] |
| `TRAJ_N_SEEDS` | `10` | KMeans seeds per K (root and sinks) |
| `COMMIT_SINK_FRAC` | `0.06` | total sink fraction (sinks use the root's K range and seeds) |
| `COMMIT_GRAPH_K` | `15` | committor graph kNN |
| `TRAJ_K_LOCAL` | `30` | neighbourhood for the lead and for smoothing the committor direction |
| `TRAJ_PLATEAU_Q`, `TRAJ_LEAD_LAMBDA` | `0.25`, `2.0` | plateau quantile of the local q-range; lead weight on plateaus |
| `TRAJ_COUPLING_MIN` | `0.10` | minimum median ATAC↔RNA coupling for the chromatin lead to be used |
| `STATE_SPACE` | `"decipher"` | latent the trajectory is built in (`"decipher"` or `"pca"`) |
| `VEL_MOMENT_KNN` | 15; 45 ma2020 | kNN smoothing of the ATAC / RNA pseudo-bulks behind the chromatin lead |

The remaining settings (`D_ATOMS`, epochs, learning rates, loss weights `W1` / `W2`, …) only affect GRN training.

## Repository layout

```
comet/
  train.py                  CLI / run(): load -> run_comet -> plots
  config.py                 dataset paths and all hyperparameters
  utils/trajectory.py       the trajectory method (origin, sinks, committor, chromatin lead, fusion)
  utils/committor.py        kNN graphs, moment operator, harmonic committor, committor direction
  utils/geometry.py         pseudo-bulk ATAC, 2D velocity projection
  training/training_loop.py run_comet (trajectory, then optional GRN training; saving) and train_grn
  model/, losses/           GRN dictionary, cell encoder (per-cell atom weights), linear decoder; GRN loss
  data/load.py              dataset loaders -> Bundle
  viz/visualization.py      display embeddings, clean_quiver and cell-type / pseudotime plots
  comet_arrow_pseudotime.py pseudotime as the scalar potential of a 2D arrow field (evaluation)
  generate_priors.py        build the GRN prior (peak_tf / peak_gene / peaks / tf_names) for a new dataset
  scripts/                  GRN evaluation against BEELINE references; paul15 ATAC trimming; nestorowa dataset build
  README_DATASETS.md        file format for new datasets
trajectory_benchmarking/
  final_method_writeup.md          method, equations, results, next steps
  final_trajectory_testing.ipynb   final pipeline: K stability, lead nulls, fusion ablation, pseudotime check
  new_trajectory_experiments.ipynb development: origin sweeps, lead formulations, combination sweeps
  root_finding_method.md           write-up of the cluster-median root selector
```

The notebooks add `comet/` to `sys.path` through the `COMET_REPO` path in their first cell (edit it if the repo
lives elsewhere), and they save figures under `trajectory_benchmarking/comet_out_beta_sweep/`.
`new_trajectory_experiments.ipynb` and `comet_beta_pseudotime_three_datasets.ipynb` are kept as a record with their
outputs; they use code that has since been removed (the legacy committor root, the trained velocity head), so
rerunning them needs commit `7dda8bf` or earlier.
