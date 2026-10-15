import os

# ── datasets ──────────────────────────────────────────────────────────────────
# Every dataset in BASES is registered; `python train.py` with no --datasets runs RUN_DATASETS.
RUN_DATASETS = ["hema", "ma2020", "aml89"]

BASES = {
    "hema":      "/home/sc5625/cll_project/decipher_atac_project/m_atac_pipeline",
    "aml89":     "/home/sc5625/cll_project/decipher_atac_project/aml89",
    "ma2020":    "/home/sc5625/cll_project/decipher_atac_project/decipher-atac/paired",
    "nestorowa": "/home/sc5625/cll_project/decipher_atac_project/nestorowa_mhsc_all",
    "paul15":    "/home/sc5625/cll_project/decipher_atac_project/COMET/paul2015data_clean",
}
OUT_DIR = "comet_out_trajectory_final"
os.makedirs(OUT_DIR, exist_ok=True)

# ── data / subsampling ────────────────────────────────────────────────────────
N_TOP_HVG        = 3000
K_MOMENT         = 15       # kNN smoothing of the gene activity (in Decipher z)
RANDOM_STATE     = 42
MA_TARGET_CELLS  = 8000
AML_TARGET_CELLS = 5000
MA_KEEP = ["TAC-1", "TAC-2", "IRS", "Hair Shaft-Cuticle/Cortex", "Medulla"]
MA_PSEUDOBULK_SMOOTH = True  # pseudo-bulk ma2020's paired ATAC over RNA clusters (each cluster's own cells); False = per cell

# ── trajectory (utils/trajectory.py; trajectory_benchmarking/final_method_writeup.md) ──
# consensus chromatin origin -> committor -> chromatin-lead plateau fill-in. Nothing here is trained.
TRAIN_GRN          = True      # False = trajectory only (no GRN training); same as --trajectory-only
TRAJ_ATAC_RES      = (120, 8, 15)  # origin score: pseudo-bulk clusters, matched ATAC cells per cluster, kNN smooth
TRAJ_ROOT_FRAC     = 0.05      # root size; the consensus K range is [3, 1/(2·TRAJ_ROOT_FRAC)]
TRAJ_N_SEEDS       = 10        # KMeans seeds per K in the consensus root and sinks
TRAJ_K_LOCAL       = 30        # neighbourhood for the lead's tangent / average / coherence and for smoothing u
TRAJ_PLATEAU_Q     = 0.25      # committor plateau = cells whose local q-range is below this quantile
TRAJ_LEAD_LAMBDA   = 2.0       # chromatin-lead weight on full plateaus
TRAJ_COUPLING_MIN  = 0.10      # use the lead only if the median ATAC↔RNA per-gene coupling reaches this

# origin assumption: +1 = the origin is the most open / highest-entropy chromatin (normal differentiation),
#                    -1 = flipped (aml89: the least open chromatin is the origin)
ORIGIN_SIGN = {"hema": 1, "ma2020": 1, "aml89": -1, "nestorowa": 1, "paul15": 1}

# committor graph / sinks
COMMIT_GRAPH_K   = 15       # kNN of the committor graph (in the state space)
COMMIT_SINK_FRAC = 0.06     # fraction of cells taken as sinks

# ── state space vs. display embeddings ────────────────────────────────────────
# STATE_SPACE is the latent the trajectory is built in (committor graph, kNN, GRN archetypes); it changes the
# result. DISPLAY_EMBEDDINGS are 2D projections for plotting only.
STATE_SPACE = {"hema": "decipher", "ma2020": "decipher", "aml89": "decipher",
               "nestorowa": "decipher", "paul15": "decipher"}   # "decipher" | "pca"
_ALL_EMB = ["decipher2d", "pca2d", "umap_decipher", "umap_pca"]
DISPLAY_EMBEDDINGS = {name: list(_ALL_EMB) for name in BASES}
FORCE_EMB = False           # recompute cached display embeddings

# kNN smoothing (moment operator) of the ATAC / RNA pseudo-bulks behind the chromatin lead (larger for noisy ATAC)
VEL_MOMENT_KNN  = {"hema": 15, "ma2020": 45, "aml89": 15, "nestorowa": 15, "paul15": 15}

# ── GRN dictionary training (training/training_loop.py:train_grn) ────────────
D_ATOMS       = 4           # GRN dictionary atoms (archetypes)
N_EPOCHS      = 80
PHASE1_EPOCHS = 40          # phase 1 / phase 2 use the loss weights W1 / W2
BATCH_SIZE    = 512
GEO_TAU_SCALE = 0.5         # smaller -> sharper geometric mixing targets
LD_HIDDEN     = 64          # must equal CellStateEncoder.h_out
LR_GRN        = 5e-4        # grn + cell_encoder + ld_decoder
# minibatches per epoch are capped so epoch cost does not grow with #cells (None = all cells)
TARGET_CELLS_PER_EPOCH        = 2048
TARGET_CELLS_PER_EPOCH_PHASE2 = 16384
FORCE_RERUN = False         # rerun even if a finished run exists in the output dir

W1 = dict(MASK=2.0, SPARSE=5e-3, ACCESS=0.1, ACCESS_GENE=0.1, DIVERSE=10.0, ANCHOR=1.0, LD=0.5, LD_REG=0.1)
W2 = dict(MASK=1.0, SPARSE=5e-3, ACCESS=0.1, ACCESS_GENE=0.1, DIVERSE=5.0, ANCHOR=0.5, LD=0.3, LD_REG=0.05)
