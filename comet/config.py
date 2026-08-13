import os

# ── which datasets to actually run this pass ──────────────────────────────────
# every dataset below is registered; only those passed to run(...) / RUN_DATASETS
# are TRAINED + plotted.
RUN_DATASETS = ["hema", "ma2020", "aml89"]

# ── file system roots (one per dataset) ───────────────────────────────────────
BASES = {
    "hema":   "/home/sc5625/cll_project/decipher_atac_project/m_atac_pipeline",
    "aml89":  "/home/sc5625/cll_project/decipher_atac_project/aml89",
    "ma2020": "/home/sc5625/cll_project/decipher_atac_project/decipher-atac/paired",
    # ported from cleanup (DATA_NESTOROWA_MHSC_E / DATA_PAUL15); registered but not run by
    # default (see RUN_DATASETS above) -- update these paths before adding either to
    # RUN_DATASETS. Not present on this machine yet.
    "nestorowa": "/home/sc5625/cll_project/decipher_atac_project/nestorowa_mhsc_e",
    "paul15":    "/home/sc5625/cll_project/decipher_atac_project/paul2015data",
}
OUT_DIR = "comet_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ── data / subsampling ────────────────────────────────────────────────────────
N_TOP_HVG        = 3000
K_MOMENT         = 15
RANDOM_STATE     = 42
MA_TARGET_CELLS  = 8000
AML_TARGET_CELLS = 5000
MA_KEEP = ["TAC-1", "TAC-2", "IRS", "Hair Shaft-Cuticle/Cortex", "Medulla"]

# ── model / training hyperparameters ──────────────────────────────────────────
D_ATOMS       = 4        # number of GRN dictionary atoms (archetypes)
N_EPOCHS      = 80
PHASE1_EPOCHS = 40
BATCH_SIZE    = 512
GEO_TAU_SCALE = 0.5      # smaller -> sharper geometric mixing targets (less mixing)
LAMBDA_GRN    = 1.0      # weight of GRN-propagated lead inside the velocity disequilibrium

# trajectory inference (production_from_lead) propagates the TF lead through only the top
# GRN_PRUNE_FRAC fraction (by |weight|, per module) of each GRN's edges -- ported from
# cleanup. GRN inference itself (predict_rna, and every loss shaping A_base/A_dict) is never
# pruned; set to None (or 1.0) to disable and propagate through the dense GRN instead.
GRN_PRUNE_FRAC = 0.1

# ── training-cost control (speedup w.r.t. #cells) ─────────────────────────────
# The dominant per-batch cost is the O(D·B·G·T) atom einsum, so wall-time scales with the
# NUMBER OF BATCHES per epoch = N/BATCH_SIZE — i.e. linearly in #cells. TARGET_CELLS_PER_EPOCH
# caps how many random minibatches each epoch draws, so per-epoch cost stops growing with N.
# This is ordinary minibatch SGD: every step is still an unbiased 512-cell gradient estimate,
# so the model and losses are mathematically unchanged — only the count of stochastic updates
# per epoch is fixed instead of growing. Small datasets (hema) already have fewer batches than
# the cap and are untouched. Lower it for more speed, raise it (or set to None) to see all cells.
TARGET_CELLS_PER_EPOCH = 2048
FORCE_RERUN = False      # True = retrain even if a saved run exists; False = load & skip finished runs

# ── committor field / "entropy assumption" hyperparameter ─────────────────────
# ENTROPY_ROOT_TRUST[name] in [-1, 1]:
#   sign  = root orientation for the chromatin-entropy assumption
#           (+ = root is HIGH chromatin entropy, the normal-differentiation case;
#            - = FLIP it, for datasets where the assumption is reversed, e.g. aml89)
#   |val| = how strongly the committor field anchors the velocity (0 = ignore committor,
#           1 = full anchor). Tune down when you trust the chromatin axis less.
ENTROPY_ROOT_TRUST = {"hema": 1.0, "ma2020": 1.0, "aml89": -0.6,
                      "nestorowa": 1.0, "paul15": 1.0}   # ported datasets: same unpaired/normal-differentiation topology as hema
# The committor now ORIENTS the velocity by construction (a forward-cone projection), so its
# weight is no longer the lever that decides whether the flow follows it — it always does. The
# consistency loss below is just a light reinforcement, so these weights stay small and uniform.
COMMIT_LOSS_W      = 1.0    # base weight on the (now light) committor reinforcement loss
COMMIT_W_SCALE     = {"hema": 1.0, "ma2020": 1.0, "aml89": 1.0, "nestorowa": 1.0, "paul15": 1.0}
COMMIT_ROOT_Q      = 0.90   # top-quantile of phi taken as the root set
COMMIT_SINK_FRAC   = 0.06   # fraction of eccentric low-phi cells taken as sinks
COMMIT_GRAPH_K     = 15     # kNN for the committor graph (in the state space)

# ── velocity denoising (first-moment kNN pooling in the state space) ──────────
# Single-cell ATAC is very sparse, so the raw ATAC->RNA lead is noisy PER CELL. This bites the
# PAIRED ma2020 dataset hardest: its gene-activity is real per-cell ATAC (smoothed once with k=15),
# whereas hema/aml89 use 40-cluster pseudo-bulk ATAC (already very smooth) before the same step.
# That extra per-cell noise rides through `disequil` into the GRN-orthogonal tilt and the per-cell
# magnitude, giving a correct committor backbone buried under high-frequency jitter ("right
# direction, just messy"). We pool the velocity lead — and the final gene-space velocity — over each
# cell's state-space neighborhood (the standard "first moment" denoising) before building/plotting
# the arrows. The committor backbone is already smooth, so this only cleans the noisy part. Use a
# larger neighborhood for the noisy paired dataset.
VEL_MOMENT_KNN  = {"hema": 15, "ma2020": 45, "aml89": 15, "nestorowa": 15, "paul15": 15}   # pseudobulk ATAC like hema -> same k
SMOOTH_VELOCITY = True     # also kNN-pool the final gene-space velocity before projecting

# ── how much GRN structure is allowed to tilt the committor-oriented arrow ─────
# The committor fixes the arrow of TIME (forward direction, no backflow). BETA_PERP sets how far
# the GRN/ATAC-derived, committor-ORTHOGONAL component may tilt the arrow inside that forward cone
# (0 = pure committor orientation with GRN magnitude; larger = more GRN-flavoured branching). This
# is what keeps the method from collapsing to "just plot the committor": direction is committor-
# locked, but the gene program, per-cell speed, and lateral branch structure are all GRN.
BETA_PERP = {"hema": 0.5, "ma2020": 0.4, "aml89": 0.5, "nestorowa": 0.5, "paul15": 0.5}   # ma2020 ATAC is noisier -> lean on committor a touch more

# ── trajectory STATE SPACE vs. DISPLAY embeddings ─────────────────────────────
# The gene-space velocity is embedding-independent: it is the SAME (D=G)-dim displacement no matter
# which 2D you plot it on — only the final projection changes with the display embedding. The ONE
# embedding choice that actually changes the trajectory is the LATENT STATE SPACE the model is built
# in (committor graph, archetypes, geometric mixing targets, projection search space all live here).
# STATE_SPACE picks that latent per dataset; DISPLAY_EMBEDDINGS picks which 2D projections to plot.
STATE_SPACE = {"hema": "decipher", "ma2020": "decipher", "aml89": "decipher",
               "nestorowa": "decipher", "paul15": "decipher"}   # "decipher" | "pca"
DISPLAY_EMBEDDINGS = {
    "hema":      ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
    "ma2020":    ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
    "aml89":     ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
    "nestorowa": ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
    "paul15":    ["decipher2d", "pca2d", "umap_decipher", "umap_pca"],
}

MARKER_GENES = {
    "hema":      ["CD34", "MPO", "LYZ", "GATA1", "SPI1", "CEBPA"],
    "aml89":     ["CD34", "KIT", "MPO", "ELANE", "ITGAM", "CD14"],
    "ma2020":    ["Krt14", "Krt15", "Sox9", "Krt1", "Krt10", "Lef1"],
    # mouse HSPC lineage genes (both are mouse hematopoiesis datasets, cleanup-ported)
    "nestorowa": ["Cd34", "Mpo", "Elane", "Gata1", "Gata2", "Cebpa"],
    "paul15":    ["Cd34", "Mpo", "Elane", "Gata1", "Gata2", "Cebpa"],
}

# ── Phase 1 / Phase 2 loss weights (GRN + velocity training) ──────────────────
# ACCESS_GENE (ported from cleanup): gene-side sibling of ACCESS -- gates GRN edges by the
# TARGET gene's own chromatin accessibility instead of the regulating TF's. Set to match ACCESS
# here rather than cleanup's own (differently-tuned) value; drop to 0.0 to disable.
W1 = dict(MASK=2.0, SPARSE=5e-3, ACCESS=0.1, ACCESS_GENE=0.1, DIVERSE=10.0, ANCHOR=1.0, LD=0.5, LD_REG=0.1)
W2 = dict(MASK=1.0, SPARSE=5e-3, ACCESS=0.1, ACCESS_GENE=0.1, DIVERSE=5.0, ANCHOR=0.5,
          VEL=3.0, ATAC=1.0, LD=0.3, LD_REG=0.05)

MODEL = {
    "D_ATOMS": D_ATOMS,
    "LD_HIDDEN": 64,
}
