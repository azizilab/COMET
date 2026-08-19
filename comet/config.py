# --- datasets ---------------------------------------------------------------
# Each block describes one dataset; select the active one with DATA at the end.
# rna_layer names the `rna_adata.layers[...]` entry to use instead of `.X`
# (None = use `.X`); see build_universe for why this matters.

_HEMATOPOIESIS = "/home/elicanter/datasets/hematopoiesis"
_NESTOROWA = "/home/elicanter/datasets/nestorowa_mhsc_e"
_PAUL15 = "/home/elicanter/datasets/paul2015data"

DATA_HEMATOPOIESIS = {
    "rna": f"{_HEMATOPOIESIS}/velten_rna.h5ad",
    "rna_layer": None,
    "atac": f"{_HEMATOPOIESIS}/GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz",
    "atac_peaks": f"{_HEMATOPOIESIS}/GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz",
    "peak_tf": f"{_HEMATOPOIESIS}/peak_tf.mtx",
    "tf_names": f"{_HEMATOPOIESIS}/tf_names.txt",
    "peak_gene": f"{_HEMATOPOIESIS}/peak_gene.csv",
    "peaks": f"{_HEMATOPOIESIS}/peaks.txt",
    "decipher_outputs": f"{_HEMATOPOIESIS}/decipher_embeddings.npz",
}

# BEELINE mHSC-E benchmark (Nestorowa 2016 RNA + reused mouse ATAC prior).
# Built by scripts/build_nestorowa_mhsc.py; decipher embeddings from
# comet/nestorowa_decipher.ipynb. Score GRN inference against the BEELINE
# gold-standard networks with scripts/eval_grn_beeline.py on the run's
# OUTPUT_DIR (point OUTPUT_DIR somewhere new first).
# The ATAC prior files are symlinks to paul2015data, so peak_tf / tf_names /
# peak_gene / peaks / atac_matrix are identical to DATA_PAUL15.
DATA_NESTOROWA_MHSC_E = {
    "rna": f"{_NESTOROWA}/Nestorowa_mHSC_E.h5ad",
    "rna_layer": "raw_count_int",
    "atac": f"{_NESTOROWA}/atac_matrix.trimmed.mtx.gz",
    "atac_peaks": f"{_NESTOROWA}/atac_matrix.trimmed.peaks.txt",
    "peak_tf": f"{_NESTOROWA}/peak_tf.mtx",
    "tf_names": f"{_NESTOROWA}/tf_names.txt",
    "peak_gene": f"{_NESTOROWA}/peak_gene.csv",
    "peaks": f"{_NESTOROWA}/peaks.txt",
    "decipher_outputs": f"{_NESTOROWA}/decipher_embeddings.npz",
}

DATA_PAUL15 = {
    "rna": f"{_PAUL15}/Paul_etal_15_2.h5ad",
    "rna_layer": "raw_count_int",  # .X here is log1p+scaled (mean~0, 78% negative) --
                                   # use raw counts instead, matching the Velten run
    "atac": f"{_PAUL15}/atac_matrix.trimmed.mtx.gz",
    "atac_peaks": f"{_PAUL15}/atac_matrix.trimmed.peaks.txt",
    "peak_tf": f"{_PAUL15}/peak_tf.mtx",
    "tf_names": f"{_PAUL15}/tf_names.txt",
    "peak_gene": f"{_PAUL15}/peak_gene.csv",
    "peaks": f"{_PAUL15}/peaks.txt",
    "decipher_outputs": f"{_PAUL15}/decipher_embeddings.npz",
}

DATA = DATA_PAUL15


# --- model / training -------------------------------------------------------

MODEL = {
    "D_ATOMS": 6,
    "LD_HIDDEN": 64,
}

TRAIN = {
    "NUM_RUNS": 1,
    "N_EPOCHS": 120, #120
    "PHASE1_EPOCHS": 120, #40
    "BATCH_SIZE": 128, #512
    "N_TOP_HVG": 3000,
    "LR_GRN": 5e-4,
    "LR_VEL": 2e-4,
    "GEO_TAU_SCALE": 0.5,  # smaller -> sharper geometric targets q (less mixing)
    "LAMBDA_GRN": 1.0, # weight of GRN-propagated lead inside the velocity disequilibrium
    # trajectory inference propagates the TF lead through only the top fraction
    # (by |weight|, per module) of each GRN's edges — GRN inference itself
    # (reconstruction, sparsity/diversity/access losses) is never pruned.
    "GRN_PRUNE_FRAC": 0.1,
    "CLIP_NORM": 1.0,
    "PLOT_EVERY": 10,
    "N_CLUSTERS": 40,
    # rna_adata.obs column used to colour plots (None = flat colour).
    # e.g. "dpt_pseudotime" (Paul15) or "facs_pseudotime" (Velten).
    "PSEUDOTIME_COL": None,
    # kNN smoothing strength for the ATAC/RNA lead (applied to BOTH modalities
    # equally before z-scoring, plus once more on the resulting lead). Smaller
    # for small branching datasets so fan-out survives; larger for linear/large
    # data. This is the one velocity knob worth tuning per dataset.
    "K_MOMENT": 10,
    "NEIGHBOR_SPACE": "latent", # change to "pca" to optimize for PCA appearance

    "WEIGHTS_PHASE1": {
        "MASK": 2.0,
        "SPARSE": 5e-3,
        "ACCESS": 1.0, #.1
        "ACCESS_GENE": 1.0, #.1
        "DIVERSE": 0, #10.0, 5
        "ANCHOR": 1.0,
        "LD": 0, #.5
        "LD_REG": 0, #.1
    },

    "WEIGHTS_PHASE2": {
        "MASK": 1.0,
        "VEL": 3.0,
        "ATAC": 1.0,
        "SPARSE": 5e-3,
        "ACCESS": .1,
        "ACCESS_GENE": .1,
        "DIVERSE": 5.0,
        "ANCHOR": 0.5,
        "LD": 0.3,
        "LD_REG": 0.05,
    },

    "LR_SCHED": {
        "ETA_MIN": 1e-5
    },

    "OUTPUT_DIR": "/home/elicanter/paul_newtest",
}
