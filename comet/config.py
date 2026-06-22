BASE = "/home/elicanter/datasets/hematopoiesis/"

DATA = {
    "rna": f"{BASE}/velten_rna.h5ad",
    "atac": f"{BASE}/GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz",
    "atac_peaks": f"{BASE}/GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz",
    "peak_tf": f"{BASE}/peak_tf.mtx",
    "tf_names": f"{BASE}/tf_names.txt",
    "peak_gene": f"{BASE}/peak_gene.csv",
    "peaks": f"{BASE}/peaks.txt",
    "decipher_outputs": f"{BASE}/decipher_embeddings.npz",
}

MODEL = {
    "D_ATOMS": 4,
    "LD_HIDDEN": 64,
}

TRAIN = {
    "NUM_RUNS": 5,
    "N_EPOCHS": 120,
    "PHASE1_EPOCHS": 40,
    "BATCH_SIZE": 512,
    "LR_GRN": 5e-4,
    "LR_VEL": 2e-4,
    "CLIP_NORM": 1.0,
    "PLOT_EVERY": 10,
    "USE_LOCAL_ATAC": True,
    "N_CLUSTERS": 20,
    "NEIGHBOR_SPACE": "decipher", # change to "pca" to optimize for PCA appearance
    "WEIGHTS_PHASE1": {
        "MASK": 2.0,
        "SPARSE": 5e-3,
        "ACCESS": 0.1,
        "DIVERSE": 10.0,
        "ENTROPY": 1.0,
        "LD": 1.0,
        "LD_REG": 0.1,
        "DIRICH": 2.0,

    },

    "WEIGHTS_PHASE2": {
        "MASK": 1.0,
        "VEL": 3.0,
        "ATAC": 2.0,
        "SPARSE": 5e-3,
        "ACCESS": 0.1,
        "SMOOTH": 0.01,
        "DIVERSE": 5.0,
        "ENTROPY": 1.0,
        "DENSE": 0.05,
        "DIV": 0.5,
        "SOURCE": 1.0,
        "DIRICH": 1.0,
        "LD": 0.3,
        "LD_REG": 0.05,

    },

    "LR_SCHED": {
        "ETA_MIN": 1e-5
    },

    "OUTPUT_DIR": "bert_grn_out_61",

}