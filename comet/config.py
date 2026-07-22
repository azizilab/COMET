#BASE = "/home/elicanter/datasets/hematopoiesis/"

BASE = "/home/elicanter/datasets/paul2015data/"


# DATA = {
#     "rna": f"{BASE}/velten_rna.h5ad",
#     "atac": f"{BASE}/GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz", 
#     "atac_peaks": f"{BASE}/GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz", 
#     "peak_tf": f"{BASE}/peak_tf.mtx",
#     "tf_names": f"{BASE}/tf_names.txt",
#     "peak_gene": f"{BASE}/peak_gene.csv",
#     "peaks": f"{BASE}/peaks.txt",
#     "decipher_outputs": f"{BASE}/decipher_embeddings.npz",
# }


DATA = {
    "rna": f"{BASE}/Paul_etal_15_2.h5ad",
    "rna_layer": "raw_count_int",  # .X here is log1p+scaled (mean~0, 78% negative) --
                                    # use raw counts instead, matching the Velten run
    "atac": f"{BASE}/atac_matrix.trimmed.mtx.gz",
    "atac_peaks": f"{BASE}/atac_matrix.trimmed.peaks.txt",
    "peak_tf": f"{BASE}/peak_tf.mtx",
    "tf_names": f"{BASE}/tf_names.txt",
    "peak_gene": f"{BASE}/peak_gene.csv",
    "peaks": f"{BASE}/peaks.txt",
    "decipher_outputs": f"{BASE}/decipher_embeddings.npz",
}

MODEL = {
    "D_ATOMS": 6,
    "LD_HIDDEN": 64,
}

TRAIN = {
    "NUM_RUNS": 1,
    "N_EPOCHS": 120,
    "PHASE1_EPOCHS": 40,
    "BATCH_SIZE": 512,
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
    "USE_LOCAL_ATAC": True,
    "N_CLUSTERS": 40,
    # kNN smoothing strength for the ATAC/RNA lead (applied to BOTH modalities
    # equally before z-scoring, plus once more on the resulting lead). Smaller
    # for small branching datasets so fan-out survives; larger for linear/large
    # data. This is the one velocity knob worth tuning per dataset.
    "K_MOMENT": 10,
    "NEIGHBOR_SPACE": "latent", # change to "pca" to optimize for PCA appearance
    "WEIGHTS_PHASE1": {
        "MASK": 2.0,
        "SPARSE": 5e-3,
        "ACCESS": 0.1,
        "DIVERSE": 10.0,
        "ANCHOR": 1.0,
        "LD": 0.5,
        "LD_REG": 0.1,
    },

    "WEIGHTS_PHASE2": {
        "MASK": 1.0,
        "VEL": 3.0,
        "ATAC": 1.0,
        "SPARSE": 5e-3,
        "ACCESS": 0.1,
        "DIVERSE": 5.0,
        "ANCHOR": 0.5,
        "LD": 0.3,
        "LD_REG": 0.05,

    },

    "LR_SCHED": {
        "ETA_MIN": 1e-5
    },

    "OUTPUT_DIR": "/home/elicanter/paul2015_outputs_newmethod3_d6",

}