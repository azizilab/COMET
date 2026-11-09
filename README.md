# COMET

COMET infers differentiation trajectories from single-cell RNA and ATAC data, paired or unpaired, without cell-type
labels or a reference pseudotime.

1. The origin is found from chromatin accessibility: the cells with the most open chromatin (or the least open, for
   aml89; see `ORIGIN_SIGN`).
2. Pseudotime is the committor on a kNN graph, from 0 at the origin to 1 at the terminal cells.
3. A per-cell gene regulatory network is learned on top of a motif prior from the ATAC peaks.
4. The velocity field follows the pseudotime gradient, uses the chromatin lead where pseudotime is flat, and is tilted
   toward the direction each cell's GRN explains.

The full method is in [`method.md`](method.md).

## Install

```bash
pip install -r requirements.txt
```

`comet/generate_priors.py` also needs `celloracle` and `genomepy`.

## Data

`paul15` is included in `paul2015data_clean/` via [Git LFS](https://git-lfs.com) (run `git lfs install` before
cloning). For the other datasets (`hema`, `aml89`, `ma2020`, `nestorowa`), put each in `datasets/<name>/` or set
`COMET_DATA_ROOT`. The expected files are listed in `comet/data/load.py`; for your own data see
[`comet/README_DATASETS.md`](comet/README_DATASETS.md).

## Usage

From `comet/`:

```bash
python train.py --datasets paul15                    # trajectory + GRN
python train.py --datasets paul15 --trajectory-only  # skip the GRN (and the GRN tilt)
python train.py --new-dataset name=my_ds base=/path/to/my_ds
```

Or from Python:

```python
import train
bundles, results = train.run(["paul15"])
q, V = results["paul15"]["q_commit"], results["paul15"]["V_pred"]
```

Results go to `comet/comet_out_trajectory_final/<name>/` (change with `--out-dir`). Finished runs are reloaded
rather than recomputed unless `FORCE_RERUN = True`. All settings are in `comet/config.py`.

## Outputs

| file | content |
|---|---|
| `committor_q.npy` | pseudotime |
| `root_mask.npy`, `sink_mask.npy` | origin and terminal cells |
| `V_pred.npy` | velocity field (cells × genes) |
| `V_pred_grn_independent.npy` | velocity field before the GRN tilt |
| `A_learned.npy`, `A_genes.npy`, `A_tfs.npy` | learned GRN (genes × TFs) and its labels |
| `<name>_clean_quiver.png`, `<name>_celltypes_pseudotime.png` | velocity and pseudotime plots |

`comet/results_final/` has the plots for all five datasets.

## Repository

```
comet/                    the method (train.py, config.py, utils/, model/, training/, data/, viz/)
comet/scripts/            GRN evaluation on BEELINE, dataset preparation
paul2015data_clean/       paul15 data
figures/                  paper figures (from the precomputed metrics in figures/data/)
decipher_run/             Decipher embeddings for hema
notebooks/, scripts/, data/   hema ATAC and GRN prior preprocessing
trajectory_benchmarking/  method development notes
```
