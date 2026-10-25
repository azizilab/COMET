# COMET: method, step by step

This document describes exactly what the code in `comet/` computes, from raw files to figures. Every
number below is the default in `comet/config.py`; the function that implements each step is given in brackets.
Results and validation are in [`trajectory_benchmarking/final_method_writeup.md`](trajectory_benchmarking/final_method_writeup.md).

COMET has two independent outputs computed from the same preprocessed data:

* **Trajectory** (§4): a root, terminal states, a pseudotime q ∈ [0, 1] and a gene-space velocity field V. Nothing
  is trained and no labels are used.
* **GRN dictionary** (§5): a TF → gene regulatory network made of a shared base plus D = 4 lineage atoms, with
  per-cell mixing weights. It is trained, and it does not feed into the trajectory.

```
raw RNA counts ─┐                          ┌─ §4 trajectory: origin score → root → sinks → committor q → V
raw ATAC counts ┼─ §1 inputs ─ §2 preprocessing ─ §3 shared objects ─┤
ATAC peak list ─┘   (+ Decipher, GRN prior)                         └─ §5 GRN: archetypes → init → training → A
                                                                       §6 visualization
```

## Notation

| symbol | meaning |
|---|---|
| N, G, T | RNA cells, genes kept for modelling, TFs kept for modelling |
| X ∈ ℝ^{N×G} | log-normalized RNA of the modelled genes; x_i its row for cell i |
| X_TF ∈ ℝ^{N×T} | log-normalized RNA of the TFs (TFs are also genes, i.e. columns of X) |
| M ∈ {0,1}^{G×T} | motif prior: M_gt = 1 if a peak linked to gene g contains a motif of TF t |
| Z ∈ ℝ^{N×d} | state space (Decipher latent z by default) |
| zscore_cols(A) | per-column standardization across cells: (A − mean) / (std + 1e−8) |
| z(v) | standardization of a vector across cells |
| unit(v) | v / (‖v‖ + 1e−8), row-wise for matrices |
| N_k(i) | the k nearest neighbours of cell i in Z (Euclidean), **including i itself** unless stated |
| s | origin sign: +1, or −1 for aml89 (`config.ORIGIN_SIGN`) |

All randomness is seeded with `RANDOM_STATE = 42` (`utils/seed.set_seed`, which also forces deterministic torch kernels).

---

## 1. Inputs

Per dataset, one directory (`base`) holds the following files. The exact file contract is in `comet/README_DATASETS.md`.

1. **RNA counts**: an AnnData with raw (unnormalized) counts, gene symbols as `var_names`, and two `obs` columns used
   **only for plots and validation**: cell type and a reference pseudotime.
2. **ATAC counts**: a peaks × cells count matrix plus its peak list. It can be unpaired (different cells from the RNA:
   hema, aml89, paul15, nestorowa) or paired (the same cells: ma2020).
3. **GRN prior** built from the ATAC peak set by `comet/generate_priors.py` (CellOracle):
   * peak → gene links B_pg ∈ {0,1}^{P×G_all}: a peak is linked to a gene if it overlaps the gene's TSS region
     (`celloracle.motif_analysis.get_tss_info`);
   * peak → motif hits B_pm ∈ {0,1}^{P×Mo}: a motif scan of each linked peak at false-positive rate 0.02
     (`TFinfo.scan(fpr=0.02)`), with motifs mapped to TF gene symbols (`dic_motif2TFs`).
4. **Decipher embeddings**: a latent z (N × d) and a 2D projection v (N × 2), same cell order as the RNA. These are
   computed outside COMET with Decipher (`decipher.tl.decipher_train`, default `DecipherConfig`) on the raw counts,
   after keeping genes detected in ≥ 10 cells and the top 5000 HVGs (computed on normalize_total + log1p data).

## 2. Preprocessing [`data/load.py`]

### 2.1 RNA

1. (aml89, ma2020 only) Stratified subsample by cell type to 5000 / 8000 cells; ma2020 also keeps only the hair-follicle
   types in `MA_KEEP`.
2. `normalize_total` to 10⁴ counts per cell, then `log1p`. Call the result X_full (all genes).
3. Highly variable genes: the top `N_TOP_HVG = 3000` (scanpy `highly_variable_genes`, default flavour).

### 2.2 GRN prior → mask M

1. Motif names are converted to gene symbols (dimers `A::B` / `A-B` are split; `(var.x)` suffixes dropped). A TF is
   kept if its symbol is an RNA gene. If a TF has several motifs, its first motif column is used.
2. Modelled genes: A_genes = (HVGs that are linked to at least one peak) ∪ TFs. G = |A_genes|, T = |TFs|.
3. Mask: M = 1[(B_pgᵀ B_pm)_{g,t} > 0], restricted to A_genes × TFs. So M_gt = 1 exactly when some peak linked to
   gene g carries a motif of TF t. Only these entries of the GRN can ever be non-zero.
4. X = X_full[:, A_genes], X_TF = X_full[:, TFs].

### 2.3 Per-cell ATAC gene activity

Let C ∈ ℝ^{N_A×P₂} be the ATAC counts (N_A ATAC cells, P₂ ATAC peaks) and B' ∈ {0,1}^{P₂×G} link each ATAC peak to
modelled genes. B' comes from the prior's peak → gene table by exact peak-name match; for aml89 it uses interval overlap,
because its prior peak set differs from the ATAC peak set. Then

  ga = log1p( rownorm_L1(C B') ) ∈ ℝ^{N_A×G}

so each ATAC cell gets a per-gene activity: the fraction of its gene-linked reads at gene g, log1p-transformed. For
aml89 only the patient's own ATAC cells are used.

### 2.4 Pseudo-bulk ATAC on RNA cells [`utils/geometry.build_local_atac`]

The RNA cells are clustered, and each cluster is given one ATAC profile. Per-cell ATAC is too sparse to use directly,
and unpaired ATAC has no cell correspondence. With resolution (n_clusters, k_atac):

1. `MiniBatchKMeans(n_clusters, n_init=5, random_state=42)` on Z gives labels ℓ_i ∈ {1..n_clusters}.
2. The cluster ATAC profile a_c ∈ ℝ^G is computed in one of two ways:
   * **unpaired**: r̄_c = mean of x_i over the cells in cluster c. The k_atac ATAC cells whose ga rows have the highest
     cosine similarity to r̄_c (nearest neighbours of the L2-normalized vectors) are found, and a_c = the mean of their
     ga rows.
   * **paired**: a_c = the mean of ga over the cluster's own cells (no matching).
3. Each RNA cell takes its cluster's profile, a_{ℓ_i}, and every gene is min-max scaled to [0, 1] across cells.

The loader runs this at (40, 8) and then applies **kNN smoothing**: each row is replaced by its mean over N_15(i) in the
Decipher latent (`K_MOMENT = 15`). The result is `ga_matched` ∈ [0,1]^{N×G}. The raw `ga` is kept as
`Bundle.ga_atac_cells` so the trajectory can rebuild the pseudo-bulk at other resolutions.

The GRN's per-module target accessibility (§5.2) is derived from ga_matched.

## 3. Shared objects

* **State space** Z [`training_loop.state_space`]: the Decipher latent z (`STATE_SPACE = "decipher"`), or the first 30
  PCs of zscore_cols(X) (`"pca"`).
* **Moment operator** S ∈ ℝ^{N×N} [`utils/committor.moment_operator`]: S_ij = 1/(k+1) for j ∈ N_k(i) (self included)
  and 0 otherwise, with k = `VEL_MOMENT_KNN` (15; 45 for ma2020, whose per-cell paired ATAC is noisier). So Sx averages
  x over each cell's neighbourhood.
* **Graphs** [`utils/committor.build_graph`]: a kNN graph on Z with k = `COMMIT_GRAPH_K = 15` (self excluded),
  symmetrized by W = max(W, Wᵀ). There are two edge weightings:
  * W^d_ij = ‖z_i − z_j‖, edge lengths used for geodesic (shortest-path) distances;
  * W^g_ij = exp(−‖z_i − z_j‖² / 2σ²), σ = the median edge length; affinities used for the committor.

---

## 4. Trajectory inference [`utils/trajectory.final_trajectory`]

### 4.1 Origin score φ [`origin_score`]

The assumption is that the origin of differentiation has the most open and most uniformly open chromatin (or the least
open, s = −1). This is measured on a finer pseudo-bulk:

1. P = build_local_atac at resolution (120 clusters, 8 ATAC cells), then kNN-smoothed over N_15 in z
   (`TRAJ_ATAC_RES = (120, 8, 15)`).
2. Let p_i = rownorm_L1(max(P_i, 0)). Two quantities are computed per cell:
   * the entropy H_i = −Σ_g p_ig log(p_ig + 1e−12);
   * the breadth B_i = (1/G) Σ_g 1[P_ig > median_j P_jg].
3. **φ_i = s · [z(H)_i + z(B)_i]**. A high φ means origin-like.

### 4.2 Root: consensus cluster-median rule [`consensus_root`, `root_cluster_medoid`]

Taking the top φ cells directly is unstable because scattered high-φ cells exist in several populations. Instead, the
root is the *cluster* with the highest *median* φ, made robust over many clusterings. With root fraction
f = `TRAJ_ROOT_FRAC = 0.05`, for each K ∈ {3, …, round(1/(2f))} = {3, …, 10} and each seed σ ∈ {0, …, 9}:

1. Run KMeans(K, n_init = 10, random_state = σ) on Z.
2. Take the cluster c* = argmax_c median{φ_i : i ∈ c}.
3. Take the medoid m: the cell of c* closest (Euclidean) to the centroid of c*.
4. Compute the geodesic distance from m over W^d (Dijkstra).
5. root_{K,σ} = the min(round(fN), |c*|) cells of c* geodesically closest to m.

Then:

* freq_i = the fraction of the 8 × 10 = 80 runs in which cell i was in root_{K,σ};
* the tie-break key is freq_i + 10⁻⁶ · minmax(φ̄)_i, where φ̄_i = the mean of φ over N_30(i);
* **R** (the root, q = 0) = the round(fN) cells with the largest key.

### 4.3 Sinks: consensus over K [`consensus_sinks`]

1. d_i = the geodesic distance (W^d, undirected, Dijkstra) from cell i to the nearest root cell. Unreachable cells get
   the maximum finite distance.
2. The candidates are the cells with φ_i < 40th percentile of φ, d_i ≥ 75th percentile of d, and i ∉ R: closed
   chromatin, far from the root.
3. One tip selection for a given K and seed σ: KMeans(K, n_init = 10, random_state = σ) on Z restricted to the
   candidates, then the max(1, ⌊0.06N/K⌋) farthest cells (largest d) of each group (`COMMIT_SINK_FRAC = 0.06`).
4. The selection is repeated over the root's K range, K ∈ {3, …, 10}, and seeds σ ∈ {0, …, 9}. sink_freq_i is the
   fraction of the 80 runs that selected cell i. **T** (q = 1) = the ⌊0.06N⌋ cells with the largest
   sink_freq_i + 10⁻⁶ · d_i / max(d).
5. Fallback, if there are fewer than 4 · 10 = 40 candidates: T = the ⌊0.06N⌋ non-root cells with the largest d.

No number of branches is assumed. Too many groups only re-split a branch, while too few can miss one, so the
consensus keeps the cells chosen as far-end tips at every K. On the five datasets the pseudotime barely changes over
K ∈ [1, 10] for hema, aml89 and Nestorowa, but ma2020 needs K ≥ 3 to give each of its three fates (Medulla, Hair Shaft,
IRS) some sinks.

### 4.4 Pseudotime: the harmonic committor q [`utils/committor.committor`]

With L = D − W^g (D the diagonal degree matrix), q solves the graph Laplace equation with boundary conditions:

  (L q)_i = 0 for i ∉ R ∪ T,  q_R = 0,  q_T = 1,

that is, L_II q_I = −L_IB q_B (sparse direct solve). q is then clipped to [0, 1]. q_i is the probability that a random
walk on W^g started at i reaches T before R; it is used as the pseudotime.

### 4.5 Committor direction ū [`utils/committor.committor_gene_direction`]

This is the gene-space direction in which q increases. With R̂ = zscore_cols(X) and the 15 nearest neighbours of i in Z
(self excluded):

  u_i = unit( Σ_j (q_j − q_i)(R̂_j − R̂_i) ).

Cells where the raw sum has norm ≤ 10⁻⁶ (flat q) are filled in iteratively, for at most 50 rounds: each such cell takes
the unit-normalized sum of its already-valid neighbours' directions. Then u is smoothed:

  **ū_i = unit( mean_{j∈N_30(i)} unit(u_j) )**   (`TRAJ_K_LOCAL = 30`).

The same function applied to d / max(d) instead of q gives the **outward** direction o_i (away from the root), which is
used in §4.6.

### 4.6 Chromatin lead ĉ [`chromatin_lead`]

The chromatin lead is what ATAC predicts for RNA *beyond* what the current RNA already explains. It is used only where
q is flat.

1. **Matched pseudo-bulks** [`matched_pseudobulk`]. ATAC and RNA are brought to the same resolution with the same 40
   clusters as ga_matched:
   * Ă = build_local_atac at (40, 8), the same as ga_matched before smoothing;
   * Ř_i = the mean of x over cell i's cluster, min-max scaled per gene;
   * both are kNN-smoothed over N_15 in z, and then A = S Ă and R = S Ř.

   (If no raw ATAC is available: A = S·ga_matched, R = S·X.)
2. **Offset-free residual** [`residual_lead`]. Let Ã = zscore_cols(A) and R̃ = zscore_cols(R). Per gene:
   * the coupling is corr_g = (1/N) Σ_i Ã_ig R̃_ig, and ρ_g = max(corr_g, 0);
   * c_ig = ρ_g (Ã_ig − ρ_g R̃_ig).

   This is ρ times the part of standardized ATAC not explained by standardized RNA. It is invariant to per-gene
   centring, so it has no pull toward the mean, unlike Ã − R̃.
3. **Tangent projection** [`tangent_project`]. The lead is restricted to directions in which RNA actually varies
   locally. Let R̂^S = zscore_cols(S X), and let Δ_i ∈ ℝ^{30×G} hold the rows R̂^S_j − R̂^S_i for the 30 nearest
   neighbours j of i (self excluded). With the Gram matrix Γ = Δ_i Δ_iᵀ and τ = tr(Γ)/30 + 10⁻⁶:

   β_i = (Γ + 0.01 τ I)⁻¹ Δ_i c_i,  c_i ← Δ_iᵀ β_i.
4. **Forward cone** [`forward_cone`]. Any component pointing back toward the root is removed:
   c_i ← c_i − min(c_i · ô_i, 0) ô_i, with ô = unit(o).
5. **Neighbourhood average**: c_i ← mean_{j∈N_30(i)} c_j.
6. **Coherence weighting** [`coherence`]: κ_i = ‖ mean_{j∈N_30(i)} unit(c_j) ‖ ∈ [0, 1], and **ĉ_i = κ_i² c_i**.
   Where neighbours disagree, the lead shrinks toward 0.
7. **Coupling gate**: coupling = median_g corr_g (unclipped), and g = 1[coupling ≥ `TRAJ_COUPLING_MIN = 0.10`]. If ATAC
   and RNA are not coupled across genes, the lead is not used at all. This happens for ma2020, where V reduces to ū.

### 4.7 Velocity field V

1. **Plateau weight**. Δq_i = max − min of q over N_16(i), and P₂₅ = the 25th percentile of Δq (`TRAJ_PLATEAU_Q`).
   Then

   a_i = clip(Δq_i / P₂₅, 0, 1),  **w_i = g · (1 − a_i)**.

   So w = 0 wherever q varies locally (the committor direction is informative), and w → 1 on flat committor plateaus.
2. **Fusion**, with λ = `TRAJ_LEAD_LAMBDA = 2`:

   **V_i = unit( (1 − w_i) ū_i + λ w_i unit(ĉ_i) )**.

V ∈ ℝ^{N×G} is a unit vector per cell in z-scored gene space. It is saved as `V_pred.npy`.

### 4.8 Trajectory outputs

`committor_q.npy` (q), `root_mask.npy` (R), `sink_mask.npy` (T), `phi.npy` (φ), `root_freq.npy` (freq), `sink_freq.npy`,
`V_pred.npy` (V), `traj_u_bar.npy` (ū), `traj_lead.npy` (ĉ), `traj_w.npy` (w), and `trajectory_mode.txt` (coupling,
gate, K range).

---

## 5. GRN inference [`training/training_loop.train_grn`, `model/`, `losses/`]

### 5.1 Model [`model/grn.GRNDictionary`]

Parameters W_base ∈ ℝ^{G×T} and W_raw ∈ ℝ^{D×G×T} (D = `D_ATOMS = 4`). The effective, masked and bounded weights are

  A_base = tanh(W_base) ⊙ M,  A_d = tanh(W_raw,d) ⊙ M  (d = 1..D),

so a TF can regulate a gene only if the motif prior allows it, and every weight lies in (−1, 1). TF inputs are
standardized per TF, x̃_TF = zscore_cols(X_TF), so a weight means "log-expression per SD of the TF". The predicted RNA of
cell i is

  **x̂_i = A_base x̃_TF,i + Σ_d π_id A_d x̃_TF,i**,

where π_i ∈ Δ^{D} are the cell's atom mixing weights from the cell encoder. The base carries the shared regulatory
program, and the atoms carry lineage-specific deviations.

**Cell encoder** [`model/encoder.CellStateEncoder`]: x_i → Linear(G, 128) → LayerNorm → GELU → Linear(128, 64) → GELU,
giving h_i ∈ ℝ⁶⁴; then a head Linear(64, D) and π_i = softmax(head(h_i)).

**Linear decoder** [`model/ld_vae.LinearDecoder`]: h_i → Linear(64, 2) (orthogonal init) = e_i → Linear(2, G)
(N(0, 0.01²) init). This is an auxiliary 2D reconstruction that regularizes h.

### 5.2 Initialization

1. **Archetypes** [`select_archetypes_fps`]: furthest-point sampling of D cells in Z. The first is the cell farthest from
   the mean of Z. Each next one maximizes the minimum squared distance to those already chosen.
2. **Soft targets** [`geometric_soft_targets`]: with δ_id = ‖z_i − z_{a_d}‖² and τ = 0.5 · median_{i,d}(δ_id) + 10⁻⁸
   (`GEO_TAU_SCALE = 0.5`), Q_id = softmax_d(−δ_id / τ).
3. **Atom init** [`init_atoms_from_soft_targets`]. For each atom d, with weights Q_{·d}:
   * μ^in_d = Σ_i Q_id x_i / Σ_i Q_id, and μ^out_d is the same with weights 1 − Q_id;
   * the soft log-fold-change is lfc_d = log1p(μ^in_d) − log1p(μ^out_d) ∈ ℝ^G;
   * for every TF t with a non-empty mask column, W⁰_{d,:,t} = 0.3 · (lfc_d ⊙ M_{:,t}) / max_g |lfc_dg M_gt|.

   Then W_raw = artanh(clip(W⁰, −0.95, 0.95)) and W_base = 0.
4. **Encoder warm start**: 40 epochs of Adam (lr 10⁻³, batch 256, shuffled) minimizing KL(Q_i ‖ π_i), so the encoder
   starts out reproducing the geometric soft targets.
5. **Per-module target accessibility**: for atom d, the soft-target-weighted mean gene activity
   acc_dg = Σ_i Q_id ga_matched_ig / Σ_i Q_id, min-max scaled jointly over all (d, g) to [0, 1]. The fixed soft targets Q
   are used rather than the learned π, so the encoder cannot move cells between atoms to avoid the penalty.

### 5.3 Losses (per minibatch B)

* **Masked reconstruction** [`losses.masked_reconstruction_loss`]. Draw a mask m_ig ~ Bernoulli(0.4) over the batch's
  cell × gene entries. Each TF's input is zeroed wherever that TF's own gene is masked, so the model cannot copy it:
  x̃'_TF,it = x̃_TF,it · (1 − m_{i,gene(t)}). Then

  L_mask = mean over masked entries (i,g) of (x̂_ig(x̃'_TF) − x_ig)².
* **Sparsity**: L_sparse = mean |A_d| + mean |A_base|.
* **Target accessibility**: L_access_gene = mean_{d,g,t} |A_dgt|(1 − acc_dg) + mean_{g,t} |A_base,gt|(1 − max_d acc_dg),
  so a gene whose locus is closed in a module is discouraged from being regulated by that module's atom, and by the
  shared base only if it is closed in every module. This is the only accessibility term.
* **Anchor**: L_anchor = mean_i KL(Q_i ‖ π_i), multiplied by α(e) = max(0.2, 1 − 0.8 e / N_EPOCHS) at epoch e, so the
  geometric targets are relaxed over training.
* **Linear decoder**: L_LD = mean ‖x̂^LD_i − x_i‖², and L_LDreg = −mean_k log(Var_B(e_k) + 10⁻⁴) + 2 Cov_B(e₁, e₂)².

Total loss:

  L = w_MASK L_mask + w_SPARSE L_sparse + w_ACCESS_GENE L_access_gene
      + w_ANCHOR α(e) L_anchor + w_LD L_LD + w_LD_REG L_LDreg

| weight | MASK | SPARSE | ACCESS_GENE | ANCHOR | LD | LD_REG |
|---|---|---|---|---|---|---|
| phase 1 (`W1`, epochs 0–39) | 2.0 | 5e−3 | 0.1 | 1.0 | 0.5 | 0.1 |
| phase 2 (`W2`, epochs 40–79) | 1.0 | 5e−3 | 0.1 | 0.5 | 0.3 | 0.05 |

### 5.4 Optimization

* **Parameters**: the GRN (W_base, W_raw), the cell encoder and the linear decoder, jointly.
* **Optimizer**: AdamW with lr = `LR_GRN = 5·10⁻⁴` (default weight decay 0.01), cosine-annealed over 80 epochs to
  10⁻⁵. Gradients are clipped to norm 1.
* **Batches**: batch size 512, sampled without replacement by a sampler seeded with 42.
* **Epochs**: 80 (`N_EPOCHS`). An epoch is capped at ⌈2048/512⌉ = 4 minibatches in phase 1
  (`TARGET_CELLS_PER_EPOCH`) and ⌈16384/512⌉ = 32 in phase 2 (`TARGET_CELLS_PER_EPOCH_PHASE2`), or all minibatches if
  there are fewer.

### 5.5 GRN outputs

After training (eval mode, no gradients), with π̄ = mean_i π_i:

* `mix_weights.npy`: π for every cell;
* `A_dict.npy` (A_d) and `A_base.npy` (A_base);
* `A_learned.npy`: A_base + Σ_d π̄_d A_d, the population-average GRN (rows = genes `A_genes.npy`, columns = TFs
  `A_tfs.npy`). The GRN of cell i is A_base + Σ_d π_id A_d.
* `steady_state.npy`: x̂ for every cell, and `loss_history.npy`.

`comet/scripts/eval_grn_beeline.py` and `run_bleval*.py` score these against BEELINE reference networks.

---

## 6. Visualization [`viz/visualization.py`, `utils/geometry.project_to_2d`, `train.run`]

1. **Display embeddings** [`make_embeddings`]. These are for plotting only; they do not affect any result:

   | embedding | 2D coordinates | neighbour (search) space for the projection |
   |---|---|---|
   | `decipher2d` | Decipher v | z |
   | `pca2d` | first 2 PCs of PCA₃₀(zscore_cols(X)) | PCA₃₀ |
   | `umap_decipher` | UMAP(n_neighbors = 15, min_dist = 0.3, seed 42) of z | z |
   | `umap_pca` | the same UMAP of PCA₃₀ | PCA₃₀ |

   `decipher2d` is shown exactly as Decipher outputs it. Every other embedding is **rotated / reflected onto it** by
   orthogonal Procrustes: with centred coordinates E (embedding) and Y (Decipher v), USVᵀ = svd(EᵀY) and
   E ← E·UVᵀ (no scaling or shear; re-centred on Y's mean). So all panels share the Decipher orientation, and reruns
   look the same.
2. **Projecting V onto 2D** [`project_to_2d`, k = 45, scale = 0.4]. This is a scVelo-style correlation kernel. For cell i
   and its 45 nearest neighbours j in the search space (self included):
   * the gene-space displacement is δ_ij = x_j − x_i;
   * the cosine similarity is c_ij = V_i·δ_ij / (‖V_i‖ ‖δ_ij‖), centred over j;
   * the local density is ρ_i = 1 / (mean distance to the 14 nearest neighbours in 2D), rescaled to [1, 3];
   * the transition weights are p_ij = softmax_j(0.4 ρ_i c_ij) − 1/45 (drift correction);
   * the 2D arrow is **v²ᴰ_i = s_i Σ_j p_ij unit(e_j − e_i)**, where e is the 2D embedding and s_i = ‖V_i‖ / its 95th
     percentile (≈ 1, since V has unit rows).
3. **clean_quiver plot** [`clean_quiver`]. Cells are scattered and coloured by cell type. A 22 × 22 grid is laid over
   the embedding; at each grid point with more than 3 cells within radius r = 1.2 · (x-range)/22, the mean v²ᴰ of those
   cells is drawn as a unit-length arrow coloured by its magnitude. The root centroid is marked with a star. `train.run`
   writes one row per dataset and one column per display embedding to `<name>/<name>_clean_quiver.png`, and all
   datasets of the run together to `<OUT_DIR>/velocity_clean_quiver_all_embeddings.png`.
4. **Cell types and pseudotime** [`plot_celltypes_pseudotime`]. For the `decipher2d` and `umap_decipher` embeddings,
   cells are coloured by reference cell type (each labelled at its median position) next to the same cells coloured by
   q, with the root (green, star = centroid) and sink (red) cells outlined. The plot is saved to
   `<name>/<name>_celltypes_pseudotime.png`.

---

## 7. Running it

```bash
cd comet
python train.py --datasets hema --trajectory-only   # §2–§4 (+ §6 figure)
python train.py --datasets hema                     # §2–§5 (+ §6 figure)
```

* The **trajectory-only** path depends only on the RNA, the ATAC, the GRN prior's peak → gene links (for the gene
  activity) and the Decipher latent. The motif part of the prior is used only by the GRN.
* The **reference cell types and pseudotime** in the RNA `obs` are never read by §4 or §5.
