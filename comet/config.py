from pathlib import Path

_HERE = Path(__file__).parent    # comet/
_DATA = _HERE / "data"           # comet/data/ — place all raw data files here

DATA = {
    "rna":             str(_DATA / "velten_rna.h5ad"),
    "atac":            str(_DATA / "GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz"),
    "atac_peaks":      str(_DATA / "GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz"),
    "peak_tf":         str(_DATA / "peak_tf.mtx"),
    "tf_names":        str(_DATA / "tf_names.txt"),
    "peak_gene":       str(_DATA / "peak_gene.csv"),
    "peaks":           str(_DATA / "peaks.txt"),
    "decipher_outputs":str(_DATA / "decipher_embeddings.npz"),
}

MODEL = {
    "D_ATOMS": 4,
    "LD_HIDDEN": 64,
}

TRAIN = {
    "NUM_RUNS": 5,
    "N_EPOCHS": 120,
    "PHASE1_EPOCHS": 50,    # longer warmup: atoms must be fully separated before
                             # velocity activates; 40 epochs was not enough
    "BATCH_SIZE": 512,
    "LR_GRN": 5e-4,
    "CLIP_NORM": 5.0,        # raised from 1.0; Phase 3 uses full-batch gradients
    "PLOT_EVERY": 10,
    "USE_LOCAL_ATAC": True,
    "N_CLUSTERS": 20,
    "NEIGHBOR_SPACE": "latent",  # latent-space kNN for ψ smoothing (higher fidelity)

    # ── Loss weights — Phase 1 (GRN + encoder warmup) ────────────────────────
    "WEIGHTS_PHASE1": {
        "MASK":    2.0,
        "SPARSE":  5e-3,
        "ACCESS":  0.1,
        "DIVERSE": 10.0,
        "ENTROPY": 1.0,   # penalize uniform mixing weights
        "DIRICH":  2.0,
        "LD":      0.5,
        "LD_REG":  0.1,
    },

    # ── Loss weights — Phase 2 (GRN + encoder + LD-decoder refinement) ───────
    "WEIGHTS_PHASE2": {
        "MASK":    1.0,
        "SPARSE":  5e-3,
        "ACCESS":  0.1,
        "DIVERSE": 5.0,
        "DIV":     0.5,   # normalized entropy weight
        "DIRICH":  1.0,
        "LD":      0.3,
        "LD_REG":  0.05,
    },

    # ── Phase 3 — Markov Velocity Field (full-batch, separate optimizer) ─────
    "PHASE3_EPOCHS":     500,
    "PHASE3_LR":         1e-1,   # larger LR acceptable: parameter space is tiny
    "K_PSI_SMOOTH":      30,     # latent-space kNN for ψ smoothing
    "N_PSI_SMOOTH_HOPS": 3,      # hops of mean aggregation
    "WEIGHTS_PHASE3": {
        "POLARITY":   1.0,   # main signal: origin cells must have lower ψ
        "FINE_ORDER": 0.0,   # DISABLED — noisy disc differences among branches
                              # create asymmetric ψ → cross-branch arrows
        "EPS_REG":    0.1,   # keep branches symmetric (small eps_raw)
    },

    # ── Visualization-stage Markov field (2D, applied after Phase 3) ─────────
    "K_FLOW_2D":        20,    # 2D-space kNN for arrow geometry
    "N_DIFFUSION_HOPS": 4,    # more hops → smoother field in sparse CLOUD area
    "DIFFUSION_BETA":   0.4,  # gentle blend preserves local direction
    "PSI_EPS":          1e-6, # numerical floor for ReLU kernel normalisation

    "LR_SCHED": {
        "ETA_MIN": 1e-3    # cosine-anneal floor for Phase 3 scheduler
    },

    "OUTPUT_DIR": "bert_grn_out_615",
}
