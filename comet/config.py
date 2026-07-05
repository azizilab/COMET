CONFIG = {
    # ── Data paths (all relative to repo root) ────────────────────────────────
    "RNA_H5AD":         "data/velten_rna.h5ad",
    "PEAK_TF_MTX":      "data/peak_tf.mtx",
    "TF_NAMES_TXT":     "data/tf_names.txt",
    "PEAK_GENE_CSV":    "data/peak_gene.csv",
    "PEAKS_TXT":        "data/peaks.txt",
    "ATAC_MTX":         "data/GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz",
    "ATAC_PEAKS_TXT":   "data/GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz",
    "DECIPHER_NPZ":     "data/decipher_embeddings.npz",
    "CELLORACLE_A_DIR": "data/cell_oracle_run/cell_oracle_global_A",
    "OUTPUT_DIR":       "outputs",

    # ── Dataset-specific obs keys (set to None if column absent) ─────────────
    "OBS_PSEUDOTIME_KEY": "facs_pseudotime",
    "OBS_FACS_KEYS": ["facs_lin","facs_cd34","facs_cd38","facs_cd90","facs_cd45ra","facs_cd135"],

    # ── Decipher .npz array keys ──────────────────────────────────────────────
    "DECIPHER_Z_KEY": "z",   # -> Z_dec (N, 10)
    "DECIPHER_V_KEY": "v",   # -> V_dec2 (N, 2)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    "N_HVG":        3000,
    "HVG_FLAVOR":   "seurat_v3",

    # ── Model architecture ────────────────────────────────────────────────────
    "D_ATOMS":        4,
    "ENCODER_HIDDEN": 128,
    "LD_HIDDEN":      64,
    "LD_DIM":         2,
    "VEL_HIDDEN":     256,

    # ── Training ──────────────────────────────────────────────────────────────
    "NUM_RUNS":       1,
    "N_EPOCHS":       120,
    "PHASE1_EPOCHS":  40,
    "BATCH_SIZE":     512,
    "CLIP_NORM":      1.0,
    "PLOT_EVERY":     10,
    "USE_LOCAL_ATAC": True,
    "N_CLUSTERS":     20,
    "NEIGHBOR_SPACE": "latent",
    "MAX_DATALOADER_WORKERS": 8,

    # ── Learning rates ────────────────────────────────────────────────────────
    "LR_GRN":   5e-4,
    "LR_VEL":   2e-4,
    "ETA_MIN":  1e-5,

    # ── Loss weights Phase 1 (GRN + encoder + LD-decoder only) ───────────────
    "WEIGHTS_PHASE1": {
        "MASK": 2.0, "SPARSE": 5e-3, "ACCESS": 0.1, "DIVERSE": 10.0,
        "ENTROPY": 1.0, "DIRICH": 2.0, "LD": 0.5, "LD_REG": 0.1,
    },

    # ── Loss weights Phase 2 (adds velocity head) ─────────────────────────────
    "WEIGHTS_PHASE2": {
        "MASK": 1.0, "SPARSE": 5e-3, "ACCESS": 0.1, "DIVERSE": 5.0,
        "ENTROPY": 0.5, "DIRICH": 1.0, "LD": 0.3, "LD_REG": 0.05,
        "VEL": 3.0, "ATAC": 2.0, "SMOOTH": 0.01, "DENSE": 0.05, "DIV": 0.5,
    },

    # ── Pseudotime (post-training analysis only) ──────────────────────────────
    "PSEUDOTIME_ROOT_PCT":      0.05,
    "PSEUDOTIME_K_NEIGHBORS":   30,

    # ── LD biplot highlight genes ─────────────────────────────────────────────
    "LD_HIGHLIGHT_GENES": ["CD34","MPO","LYZ","GATA1","SPI1","CEBPA"],
}
