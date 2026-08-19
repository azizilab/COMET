# Running COMET on a dataset

There are two ways to run COMET, and they're mutually exclusive per dataset:

1. **A built-in dataset** (`hema`, `aml89`, `ma2020`, `nestorowa`, `paul15`) — already registered
   in `config.py`, already has a loader in `data/load.py`. Just name it.
2. **Any other dataset** — always goes through the *generic loader*
   (`data/load.py:load_generic`). There is no per-dataset code to write. You are responsible for
   getting your files into the canonical format described below; COMET does no dataset-specific
   preprocessing beyond what `load_generic` already does (normalization, HVG selection, TF/gene
   matching, kNN smoothing).

## Running a built-in dataset

```bash
python train.py --datasets hema aml89
```
No args at all runs `config.RUN_DATASETS` (currently `hema`, `ma2020`, `aml89`).

## Running a new dataset

```bash
python train.py --new-dataset name=my_ds base=/path/to/my_ds \
    cell_type_col=cell_type pseudotime_col=pseudotime
```

This registers `my_ds` (generic loader, pointed at `/path/to/my_ds`) and runs it — nothing else
runs unless you also pass `--datasets`:

```bash
# run my_ds alongside the built-in defaults / a specific built-in dataset
python train.py --datasets hema --new-dataset name=my_ds base=/path/to/my_ds \
    cell_type_col=cell_type pseudotime_col=pseudotime
```

`--new-dataset` takes `KEY=VALUE` pairs (space-separated, no spaces around `=`):

| Key | Required? | Meaning |
|---|---|---|
| `name` | **yes** | Dataset name used everywhere downstream (output dir, plots, etc). Must not collide with an existing name. |
| `base` | **yes** | Directory containing your files (see format below). |
| `cell_type_col` | recommended | `obs` column name for cell type. Default: `cell_type`. |
| `pseudotime_col` | recommended | `obs` column name for pseudotime. Default: `pseudotime`. |
| `rna_layer` | only if needed | Name of an `adata.layers[...]` entry to use instead of `.X` for raw counts. |
| `rna_file`, `atac_mtx`, `atac_peaks_file`, `decipher_npz` | only if needed | Override individual file paths if you're not using the canonical filenames below. |
| `entropy_root_trust` | tune if needed | See tuning guide below. Default: `1.0`. |
| `commit_w_scale` | tune if needed | Default: `1.0`. |
| `vel_moment_knn` | tune if needed | Default: `15`. |
| `beta_perp` | tune if needed | Default: `0.5`. |
| `state_space` | tune if needed | `decipher` or `pca`. Default: `decipher`. |
| `marker_genes` | optional | Comma-separated gene list for plot annotation, e.g. `marker_genes=CD34,MPO,GATA1`. |
| `display_embeddings` | optional | Comma-separated, e.g. `display_embeddings=decipher2d,pca2d`. |

This same registration is available from Python (e.g. a notebook) via
`train.register_new_dataset({"name": "my_ds", "base": "/path/to/my_ds", ...})` followed by
`train.run(["my_ds"])`.

---

## Canonical file format (required for any dataset not in the built-in 5)

Put these files in one directory (`base`). Default filenames are used if you don't override them
via `rna_file=`/`atac_mtx=`/`atac_peaks_file=`/`decipher_npz=`.

| File | Default filename | Format |
|---|---|---|
| RNA anndata | `rna.h5ad` | AnnData. `obs_names` = cells, `var_names` = gene symbols. Raw (unnormalized) counts in `.X`, or in `layers[rna_layer]` if you pass `rna_layer=`. **Don't pre-normalize** — COMET does its own `normalize_total` + `log1p` + HVG selection. |
| ↳ cell type | column in `rna.obs` | String/categorical. Column name given by `cell_type_col=` (default `cell_type`). |
| ↳ pseudotime | column in `rna.obs` | Float, any range (rescaled to [0,1] internally; NaNs filled with the max). Column name given by `pseudotime_col=` (default `pseudotime`). **You must compute this upstream** (e.g. `scanpy.tl.dpt`) — the generic loader does not compute pseudotime for you. |
| GRN prior: peak×motif | `peak_tf.mtx` | MatrixMarket, shape `(P peaks) x (M motifs)`, nonzero = motif hit in peak. Row order must exactly match `peaks.txt`. **Generated per-dataset — see below.** |
| GRN prior: motif names | `tf_names.txt` | Whitespace-separated motif names, length `M`, one-to-one with `peak_tf.mtx` columns. A motif name should equal (or contain, for dimers like `GATA1::TAL1` — auto-split on `::`/`-`) a gene symbol present in the RNA `var_names`. Matching is exact/uppercase (not fuzzy). **Generated per-dataset — see below.** |
| GRN prior: peak→gene links | `peak_gene.csv` | **Tab-separated**, header row, columns literally named `peak` and `gene`. `peak` values must match `peaks.txt` entries, in `chr-start-end` (dash-joined) format — `load_generic`'s peak-matching step assumes this exact format. **Generated per-dataset — see below.** |
| GRN prior: peak list | `peaks.txt` | Single column, no header, one peak string per row, length `P`, same order as `peak_tf.mtx` rows, `chr-start-end` (dash-joined) format. **Generated per-dataset — see below.** |
| ATAC matrix | `atac_matrix.mtx.gz` | MatrixMarket, shape `(P2 peaks) x (cells)` (transposed on load). Can be pseudobulk/cluster-level or per-cell — used only to build smoothed gene activity, doesn't need to align 1:1 with `peaks.txt` (re-matched via the peaks file below). |
| ATAC peaks (for the matrix above) | `atac_peaks.txt` | Tab-separated, header row with a `Feature` column, one row per ATAC-matrix column. `Feature` values are transformed `chr_start_end` → `chr:start-end` on load — if your peak IDs already use `:`/`-`, pass `atac_peaks_file=` pointing at a file shaped like this or adjust upstream. |
| Decipher embeddings | `decipher_embeddings.npz` | `.npz` with a latent array under key `z` and a 2D array under key `v`, both with length = N cells in the RNA h5ad, **same cell order**. Generated by running the separate Decipher model on this dataset (see `decipher_run/`) — COMET does not compute this. **Required even if `state_space=pca`** — `z` is also used for gene-activity pseudobulk clustering regardless of that setting. |

Everything else (HVG selection, normalization, TF/gene intersection with the prior, kNN
smoothing) is handled identically for every dataset by `load_generic` — you don't need to
replicate any of that.

If your raw files don't match this contract (e.g. no cell-type column and you need
marker-gene-based typing like `hema`, or paired per-cell RNA+ATAC like `ma2020`), preprocess them
into this format first — there is intentionally no per-dataset code path for new datasets.

---

## Generating the GRN prior (`peak_tf.mtx` / `peak_gene.csv` / `peaks.txt` / `tf_names.txt`)

These four files are **not reusable across datasets** — they're a peak×motif scan and a
peak→gene TSS-proximity map built from *that dataset's own ATAC peak set*, so a different peak
set needs its own files. (The one exception in this repo, `nestorowa`, symlinks `paul15`'s prior
only because both datasets were deliberately built on the exact same mouse sciATAC peak set —
that's not the general case.) Regenerate them for every new dataset.

`comet/generate_priors.py` builds all four from your ATAC peaks using CellOracle
(`get_tss_info` for peak→gene TSS mapping, `TFinfo.scan` for the motif scan), so it works for
**either mouse or human** — just pass the matching genome build. It writes canonical filenames
directly into an output directory, so you can point it straight at your dataset's `base=` dir.

It reads peaks straight from a **text file** — no `.h5ad` needed. Concretely, it takes the
`atac_peaks.txt` you already have to prepare for COMET anyway (tab-separated, header row, a
`Feature` column — see the canonical file table above), so there's nothing extra to build just
for this step. (A plain headerless single-column peak list also works, if that's what you have.)

**Prerequisites:** `pip install celloracle genomepy` (not part of COMET's regular dependencies —
only needed for this one-time step).

**Recommended order** for a brand-new dataset: prepare `atac_peaks.txt` first (you need it for
COMET regardless), then feed that same file into `generate_priors.py`:

```bash
pip install celloracle genomepy

python generate_priors.py \
    --input /path/to/my_ds/atac_peaks.txt \
    --genome mm10 \
    --outdir /path/to/my_ds
    # human example: --genome hg38 (or hg19, to match your RNA gene annotation build)
```

This downloads the reference genome via `genomepy` on first run (cached under
`~/.local/share/genomes/`), then runs two checkpointed steps (TSS mapping, then motif scanning —
the slow one, several minutes to hours depending on peak count) with intermediate results cached
as `.cache_*.pkl` files in `--outdir` so a rerun/failure resumes instead of restarting from
scratch. It writes `peak_gene.csv`, `peak_tf.mtx`, `tf_names.txt`, `peaks.txt` directly into
`--outdir` under the exact filenames `load_generic` expects — no renaming needed, and no
`--new-dataset` overrides exist for these four (unlike `rna_file=`/`atac_mtx=`/etc.), so they must
live at exactly `<base>/peak_gene.csv` etc.

**Genome build must match your RNA gene annotation** (e.g. don't mix an `hg19`-based ATAC prior
with `hg38` RNA gene symbols/coordinates) — a mismatch doesn't error, it just silently shrinks the
peak→gene overlap. Species mismatch (mouse ATAC prior against human RNA, or vice versa) is the
same failure mode and is not checked for you.

### What's left after this step

Given a fresh ATAC dataset, `generate_priors.py` covers the 4 GRN-prior files, and you already
had to prepare `atac_peaks.txt` for it (which doubles as COMET's own `atac_peaks_file=` input).
The only files left to supply yourself before running COMET:
- `rna.h5ad` — your raw (unnormalized) RNA counts + cell type/pseudotime columns.
- `atac_matrix.mtx.gz` — your dataset's own ATAC counts, columns matching `atac_peaks.txt` row
  order.
- `decipher_embeddings.npz` — from running Decipher separately on this dataset.

So: `generate_priors.py` handles the GRN prior; **`decipher_embeddings.npz` is the only genuinely
extra file you generate yourself** (`atac_peaks.txt` was never extra — it's shared between
`generate_priors.py` and COMET's own dataset contract).

---

## Hyperparameter tuning guide

**Required — no sane default, must be set per new dataset:**
- `base`, and any file path overrides needed to match your layout
- `cell_type_col`, `pseudotime_col` (or accept the `cell_type`/`pseudotime` defaults if you name your columns that way)
- `rna_layer`, only if raw counts aren't in `.X`

**Recommended to review before trusting results** (wrong value doesn't error, it silently
degrades the trajectory):
- `entropy_root_trust` — **sign** encodes whether this is normal differentiation (root = high
  chromatin entropy, `+1.0`) or reversed (e.g. malignant/dedifferentiating, like `aml89`'s
  `-0.6`). **Magnitude** controls how strongly the committor field anchors the velocity. Wrong
  sign flips your trajectory's direction.
- `marker_genes` — only affects plots, but you'll want genes relevant to your biology to sanity-check the run.
- `state_space` — `decipher` is normally right; fall back to `pca` if you don't trust/have a
  well-trained Decipher embedding for this dataset.
- `vel_moment_knn` — denoising neighborhood size. Existing datasets use `15` for pseudobulk/already-smooth
  ATAC and `45` for noisy per-cell paired ATAC (`ma2020`). Set based on how sparse/noisy your ATAC is.

**Safe to leave at defaults for a first run** (already tuned across 5 diverse datasets — these
aren't exposed via `--new-dataset`; edit `config.py` directly if you need to revisit them):
`beta_perp`/`commit_w_scale` fine-tuning beyond the table above, `COMMIT_LOSS_W`,
`COMMIT_ROOT_Q`, `COMMIT_SINK_FRAC`, `COMMIT_GRAPH_K`, `D_ATOMS`, `N_EPOCHS`, `PHASE1_EPOCHS`,
`BATCH_SIZE`, `GEO_TAU_SCALE`, `LAMBDA_GRN`, `LD_HIDDEN`, `LR_GRN`, `LR_VEL`, `GRN_PRUNE_FRAC`,
`TARGET_CELLS_PER_EPOCH`, `W1`, `W2`, `N_TOP_HVG`, `K_MOMENT`.

---

## Output

Each dataset writes to `comet_out/<name>/` (learned GRN, velocity arrays, plots), independent of
every other dataset. `FORCE_RERUN=False` in `config.py` means a finished run for a given name is
loaded from disk and skipped rather than retrained — delete `comet_out/<name>/` (or set
`FORCE_RERUN=True`) to force a retrain.

---

## TL;DR: new dataset, start to finish

**1. Files you need in hand before running anything:**

| File | What it is |
|---|---|
| `rna.h5ad` | Raw (unnormalized) RNA counts, + a cell-type `obs` column and a pseudotime `obs` column. |
| `atac_matrix.mtx.gz` | Your ATAC counts (MatrixMarket), per-cell or pseudobulk. |
| `atac_peaks.txt` | Tab-separated, header row, `Feature` column — one row per `atac_matrix.mtx.gz` column, same order. |
| `decipher_embeddings.npz` | `z` (latent) + `v` (2D) arrays, from running Decipher separately on this dataset — same cell order as `rna.h5ad`. |

Everything else (`peak_gene.csv`, `peak_tf.mtx`, `tf_names.txt`, `peaks.txt`) gets generated in
step 2, from `atac_peaks.txt` above — nothing extra to prepare for those.

**2. Generate the GRN prior** (mouse: `mm10`/`mm39`; human: `hg19`/`hg38` — match your RNA's gene
annotation build):

```bash
python generate_priors.py \
    --input /path/to/my_ds/atac_peaks.txt \
    --genome mm10 \
    --outdir /path/to/my_ds
```

Put all 4 files above (`rna.h5ad`, `atac_matrix.mtx.gz`, `atac_peaks.txt`,
`decipher_embeddings.npz`) in that same `/path/to/my_ds` directory — `generate_priors.py` writes
`peak_gene.csv`/`peak_tf.mtx`/`tf_names.txt`/`peaks.txt` alongside them, so the directory is now a
complete, self-contained `base=` for COMET.

**3. Run COMET:**

```bash
python train.py --new-dataset name=my_ds base=/path/to/my_ds \
    cell_type_col=cell_type pseudotime_col=pseudotime
```

Output lands in `comet_out/my_ds/`.
