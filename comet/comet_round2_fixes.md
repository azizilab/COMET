# COMET — Round 2 Diagnosis & Fixes
**Date:** 2026-06-23  
**Situation:** Round 1 fixes improved the ψ scalar field (gap widened, gradient smoother) but made pseudotime *worse* (collapsed to near-binary) and the velocity field is still broken (now shows circular loops rather than inward convergence). Gene trajectories remain below the 20% GT recovery threshold with STMN1 as the top CMP gene — a cell-cycle marker with no myeloid relevance.

This document starts from scratch on the diagnosis, using both the original and new run outputs together.

---

## The Master Diagnosis: Three Independent Failure Modes

After examining both runs side by side, the problems are not where Round 1 assumed. The three root causes are:

1. **Atom 0 dominance creates a ψ field that is flat over 80% of the embedding** — the velocity signal is nearly zero everywhere except the tiny Atom 1 boundary.
2. **The Dijkstra graph has near-zero edge costs in the Atom 0 interior**, collapsing pseudotime to a binary step function.
3. **Gene trajectories correlate with cell cycle, not differentiation**, because the stem-cell inclusion (Fix G1) added proliferating HSC/MPP cells whose dominant transcriptional variation is S/G2M phase, not lineage commitment.

These three problems are **independent** — fixing one does not fix the others. They must be addressed in sequence, starting with the atom structure.

---

## Problem 1 — Atom 0 Dominates: The Diversity Loss Failed

### What the data shows

Image 3 (right panel, Dominant GRN Atom) shows:
- **Atom 0 (blue):** covers approximately 80% of all cells, including the entire left half and lower-center of the embedding
- **Atom 1 (red, ORIGIN):** a tiny cluster of ~50 cells at center-bottom (Decipher 1≈0.5–1.5, Decipher 2≈−2 to −0.5)
- **Atom 2 (pink):** a thin streak at upper-right
- **Atom 3 (cyan):** upper-left quadrant

This is a degenerate solution. A four-atom model should partition the dataset into four roughly comparable regions corresponding to HSC/MPP (origin), CMP→GMP, CMP→MEP, and MLP branches. Instead, Atom 0 swallowed everything except a tiny origin pocket and two marginal branches.

### Why this causes everything downstream to fail

ψ for cell `i` is `mix_weights[i] @ P`, where `P[1] = −1` and `P[0,2,3] ≈ +1`. For any cell with `mix_weights[i, 0] ≈ 0.85` (the typical Atom 0-dominant cell), ψ ≈ `0.85 × 1 + 0.10 × (−1) + 0.05 × 1 = +0.80`. For a neighbor `j` with `mix_weights[j, 0] ≈ 0.87`, ψ ≈ `+0.82`. The ψ difference is **0.02** — pure noise from mix_weight estimation variance. The ReLU kernel sees `max(0.02, 0) = 0.02` and produces a near-zero weight — a coin-flip arrow direction.

The only place ψ has real gradient is the transition from Atom 1 → Atom 0 territory, a zone of ~50–100 cells. Outside this zone, `V_raw` is essentially random noise. After 4 hops of diffusion, random noise accumulates into the circular-loop artifact visible in Image 2.

### Root cause of the atom collapse

The training log shows diversity loss decreasing from 0.232 → 0.119 over 120 epochs. But 0.119 is not zero — it means the atoms are still somewhat correlated in parameter space. The real failure is in the **CellStateEncoder**: after pre-training on KMeans labels from the 2D Decipher space, and given that the Decipher space has an imbalanced cluster structure (the stem is a small, dense cluster; the committed branches are diffuse), the encoder learns to put most cells in the large KMeans cluster — which becomes Atom 0.

The encoder's Dirichlet sharpening loss (`−sum(log(mix_w))`) and the diversity loss both push toward equal atom usage, but the mask reconstruction loss dominates (weight 2.0 vs 10.0 × 0.119 ≈ 1.2 at the end) and the encoder finds that putting most cells in the "average" atom minimizes reconstruction error. The entropy regularization (`W1_ENTROPY × H/log(D)`) which should push toward uniform atom usage has weight only 1.0 and is applied to the *entropy of the softmax output* — but a softmax of a well-trained linear head can have high entropy while still being dominated by one class if the logits are balanced at a moderate temperature.

### Fix A1 — Replace KMeans initialization with ATAC-discordance-guided cluster seeding

The current initialization clusters cells in 2D Decipher space using KMeans — this reproduces the spatial imbalance of the embedding. Instead, initialize the encoder's cluster assignments using the ATAC discordance signal directly:

```python
def initialize_from_atac_discordance(X, gene_access, grn, D):
    """
    Seed the D cluster centers using ATAC-RNA discordance per cell.
    Each cell gets a discordance score = mean(ATAC - RNA/max_RNA) over its top expressed genes.
    Then rank cells by discordance, divide into D equal quantile bins, and use those
    bin memberships as initial cluster labels.
    
    This guarantees each atom gets N/D cells and is seeded on a biologically
    meaningful axis (ATAC-ahead-of-RNA = upstream), not arbitrary spatial clusters.
    """
    N = X.shape[0]
    # Simple per-cell discordance: ATAC open but RNA silent
    gene_max = gene_access.max(axis=0) + 1e-8
    rna_norm = X / (X.max(axis=0) + 1e-8)
    per_cell_disc = (gene_access - rna_norm).mean(axis=1)   # (N,)
    
    # Rank and bin into D equal groups
    ranks = np.argsort(np.argsort(per_cell_disc))   # ordinal ranks
    cluster_labels = (ranks / (N / D)).astype(int).clip(0, D - 1)
    return cluster_labels
```

Then use these labels for `cell_encoder.init_from_cluster_labels(X, cluster_labels)`. This seeds Atom 0 in the lowest-discordance (most committed) cells and Atom D−1 in the highest-discordance (stem) cells — guaranteeing equal-sized initial populations and a biologically motivated starting point.

### Fix A2 — Add a Cell-Entropy Balancing Loss

Add a **coverage loss** that penalizes any single atom from capturing more than `2/D` fraction of the total mixing weight:

```python
def atom_coverage_loss(mix_weights: torch.Tensor, D: int, max_frac: float = 0.5) -> torch.Tensor:
    """
    Penalize atoms that monopolize more than max_frac of the total weight.
    mean_weight_per_atom should ideally be 1/D; penalize if any atom exceeds max_frac.
    """
    mean_w = mix_weights.mean(dim=0)           # (D,): mean weight per atom across batch
    excess = torch.relu(mean_w - max_frac)     # positive when an atom dominates
    return excess.sum()
```

Add this to Phase 1 and Phase 2 with weight `W_COVERAGE = 5.0`. This directly prevents Atom 0 from absorbing 80% of cells. Set `max_frac = 0.4` (allowing up to 40% per atom, slightly above the 25% expected for D=4) — lenient enough not to fight the data but strict enough to prevent collapse.

### Fix A3 — Increase temperature schedule persistence in Phase 2

The training code cools temperature to `temp = 0.2` in Phase 2. A temperature of 0.2 in `softmax(logits / 0.2)` makes the softmax nearly deterministic — each cell commits fully to its single highest-logit atom early in training. Once the encoder finds a locally good assignment (most cells → Atom 0), the temperature is too low to allow exploration.

Change Phase 2 temperature to `temp = 0.5` and only drop to 0.2 in the final 20 epochs. This keeps the mixing weights softer for longer, allowing the diversity and coverage losses more time to reshape the assignment.

```python
# In the Phase 2 training loop:
if phase == 2:
    frac_through_p2 = (epoch - PHASE1_EPOCHS) / (N_EPOCHS - PHASE1_EPOCHS)
    temp = 0.5 - 0.3 * frac_through_p2   # 0.5 → 0.2 over Phase 2
```

---

## Problem 2 — Pseudotime Collapsed: The Dijkstra Cost Function Was Broken by Round 1 Fix

### What went wrong with Fix 1c

The original Dijkstra used edge cost = `δψ / dist_2d`. This gave a meaningful range of costs because ψ varies continuously across the embedding. The Round 1 fix replaced ψ differences with `vel_pseudotime_old[j] - vel_pseudotime_old[i]` as the flow potential. But `vel_pseudotime_old` was itself computed from the same broken ψ field — and it had already compressed most cells toward pseudotime ≈ 1.0 (the "yellow everywhere" observation in the new run). When you then use that compressed pseudotime as the Dijkstra edge cost, the cost of every edge is `≈1.0 − 1.0 = 0`. Every cell is reachable from every other cell at near-zero cost, so Dijkstra assigns all cells a distance of 0 from the roots — and after normalization, everyone gets pseudotime = 0... or if rescaling by max distance which is also near-zero, it amplifies noise to produce the near-binary output.

**Do not use vel_pseudotime as the Dijkstra cost.** The correct Dijkstra cost is a *structural property of ψ*, not a derived output. The original formulation was right; the issue was the ψ field itself being too flat, not the cost function.

### The actual pseudotime fix: repair ψ flatness, not the Dijkstra cost

Once Problem 1 (atom collapse) is fixed and ψ has real dynamic range across the full embedding, the original Dijkstra formulation (`cost = δψ / dist_2d` where `δψ = ψ(j) − ψ(i) > 0`) will produce a good pseudotime again. The evidence for this: in the **original run** (before any fixes), Dijkstra with the old ψ gave ρ=0.633. The ψ ordering was correct even then. The problem was never Dijkstra — it was the ψ field being compressed into a narrow range with no gradient in the Atom 0 interior.

Revert the Dijkstra cost function to the original:
```python
delta = psi_arr[j] - psi_arr[i]
if delta > 0:
    dist_ij = np.linalg.norm(V_dec2[j] - V_dec2[i]) + 1e-8
    cost = delta / dist_ij   # ORIGINAL — do not change this
```

**What will improve pseudotime after fixing atom collapse:**
- With balanced atoms (25% each), most cells will have meaningful mix_weight variation across all four atoms.
- ψ will vary continuously from −1 (Atom 1 dominant) to +1 (other atoms dominant) across the full embedding.
- The Dijkstra DAG will have non-trivial edge costs everywhere, producing a smooth gradient pseudotime.

### Supplementary pseudotime fix: multi-source Dijkstra with all ψ-minimum cells, not just bottom 5%

The current code uses `root_pct = 0.05` — bottom 5% of ψ. With 1035 cells, this gives 52 roots. After fixing atom collapse, Atom 1 territory should contain ~250 cells (balanced 4-atom model). Using bottom 5% picks only cells with ψ very close to −1 — cells with nearly pure Atom 1 membership. These are fine as roots. But if ψ has a real gradient and the bottom 5% is 52 cells, they may not span the full extent of the stem region, leaving some stem cells only reachable via long paths → compressed pseudotime at the stem end.

After fixing atom collapse, consider using `root_pct = 0.10` (10%, ~100 cells) to give wider stem coverage. This is a minor point — fix the atoms first.

---

## Problem 3 — Velocity Arrows Are Still Wrong: The Displacement Formula Is Geometrically Incoherent

### The diagnosis from Round 1 was partially right but missed the deeper issue

Round 1 correctly identified that "ψ gradient points inward in 2D" — but the proposed fix (use vel_pseudotime as the flow potential) made things worse because vel_pseudotime was already corrupted. The correct fix is more fundamental.

The core issue with `compute_markov_velocity_2d` is this: the arrow for cell `i` is:
```
V[i] = sum_j w[j] * (X_2d[j] − X_2d[i])
```
where `w[j] = ReLU(ψ(j) − ψ(i))`. This computes a **weighted centroid displacement** toward neighbors with higher ψ. The problem is that "neighbors with higher ψ" in 2D Decipher space are:
- In the Atom 0 interior: whichever of the 20 nearest neighbors happens to have slightly higher ψ due to noise. The resulting arrow points in a random direction.
- At the Atom 1 → Atom 0 boundary: the Atom 0 neighbors genuinely have higher ψ, so the stem cells correctly point into Atom 0. But the direction is "toward the Atom 0 interior centroid" — i.e., *inward* to the dense cluster, not outward toward the branches.

### The replacement: ψ-gradient estimation via local linear regression

Stop computing arrows as weighted-centroid-displacement. Instead, estimate the **2D spatial gradient of ψ** directly using local linear regression:

```python
def compute_psi_gradient_velocity(psi: np.ndarray,
                                   X_2d: np.ndarray,
                                   k: int = 30,
                                   regularization: float = 1e-3) -> np.ndarray:
    """
    For each cell i, fit a local linear model:
        ψ(x, y) ≈ α + β_x * x + β_y * y
    using the k nearest 2D neighbors of i (including i itself).
    The gradient [β_x, β_y] is the velocity arrow for cell i.
    
    This is geometrically correct: it gives the direction of steepest ψ-increase
    in 2D space. If ψ increases toward the upper-right in local coordinates,
    the arrow points upper-right — regardless of whether neighbors are in the
    same or different atom territories.
    
    Regularization prevents singular fits in locally flat ψ regions.
    """
    from sklearn.neighbors import NearestNeighbors
    
    N = len(psi)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_2d)
    dists, idx = nbrs.kneighbors(X_2d)
    idx = idx[:, 1:]    # exclude self
    dists = dists[:, 1:]
    
    V = np.zeros((N, 2), dtype=np.float32)
    
    for i in range(N):
        nbr_i = idx[i]                    # (k,) neighbor indices
        X_nbr = X_2d[nbr_i]              # (k, 2) neighbor 2D coordinates
        ψ_nbr = psi[nbr_i]               # (k,) neighbor ψ values
        
        # Center coordinates for numerical stability
        X_c = X_nbr - X_2d[i]            # (k, 2) centered displacements
        ψ_c = ψ_nbr - psi[i]             # (k,) centered ψ values
        
        # Gaussian weights: closer neighbors contribute more
        σ = np.median(dists[i]) + 1e-8
        w = np.exp(-0.5 * (dists[i] / σ)**2)
        W = np.diag(w)
        
        # Weighted least squares: min ||W(X_c @ β − ψ_c)||²
        # Normal equations: (X_c.T W X_c + λI) β = X_c.T W ψ_c
        A = X_c.T @ W @ X_c + regularization * np.eye(2)
        b = X_c.T @ (w * ψ_c)
        
        try:
            beta = np.linalg.solve(A, b)   # (2,): [dψ/dx, dψ/dy] in 2D
        except np.linalg.LinAlgError:
            beta = np.zeros(2)
        
        V[i] = beta   # the gradient = direction of steepest ψ increase
    
    return V
```

This gives geometrically correct arrows by construction. The estimated gradient `[β_x, β_y]` points in the direction of steepest ψ-ascent in 2D space. No centroid-displacement ambiguity, no kNN-direction cancellation.

**Why this is better than the weighted-centroid approach:**

The weighted centroid approach confounds the *magnitude* of ψ differences (which should determine arrow strength) with the *geometry* of displacement (which should determine arrow direction). Two neighbors at different angles from cell `i` but with equal ψ-differences will both pull the centroid toward their midpoint — which may not be the direction of steepest ψ-ascent. The local linear regression disentangles these: it fits the ψ surface and reads off the gradient, which is purely geometric.

### After computing V via gradient estimation, apply a single-hop diffusion for smoothing

The gradient estimate is noisy for individual cells (especially in flat ψ regions). Apply one round of nearest-neighbor averaging:

```python
# Single-hop diffusion: each cell's arrow is the mean of its neighbors' arrows + its own
V_smooth = 0.5 * V + 0.5 * V[neighbor_idx_2d].mean(axis=1)
```

Use only 1 hop (not 4). Multiple hops washes out the gradient structure and produces the circular-loop artifact. The gradient estimate from local linear regression is already a spatially smoothed quantity (it uses k=30 neighbors); one additional diffusion hop is sufficient.

### Keep integration_direction='forward' and add minlength=0.8

These settings were correct in Round 1. Keep them. Also increase `minlength` from 0.4 to 0.8 to filter out the short, incoherent streamlines in flat-ψ regions.

---

## Problem 4 — Gene Trajectories: Cell Cycle Contamination

### Why STMN1 appears at the top

STMN1 (Stathmin-1) is a microtubule-destabilizing protein transcriptionally induced in S and G2/M phases of the cell cycle. It has the highest expression in actively dividing progenitors. When the Round 1 Fix G1 added HSC and MPP cells to each lineage's trajectory (to increase sample size), it added a large number of cycling progenitors. These cells cluster at the "early" pseudotime end (roots). As cells commit and exit cell cycle, STMN1 drops. But when pseudotime is near-binary (root cluster = 0, everything else = 1), any gene with high expression in the root cluster gets a high Spearman correlation with pseudotime — including all cell-cycle genes. This is a pseudotime artifact, not a biology artifact.

Similarly, IGLL1 (immunoglobulin lambda-like polypeptide) appearing in the CMP top genes is a B-cell gene that shows up spuriously when HSC/MLP cells (which have more B-cell potential than CMP cells) are included in the "CMP trajectory." Ground truth recovery for CMP (CSF1R, CEBPA, FLT3) suffers because these genes are less variable in pseudotime than the noise introduced by the mixed trajectory.

### Fix G1-Revised: Use a Directed Trajectory, Not a Pool

Do not add all HSC/MPP cells to every lineage trajectory. Instead, build a **lineage-specific trajectory** by tracing the Dijkstra shortest path backward from each lineage's centroid to the root, collecting cells within a corridor around that path:

```python
def build_lineage_trajectory(vel_pseudotime, V_dec2, cell_types, target_lineage,
                              origin_cell_types=('HSC', 'MPP'),
                              corridor_width=0.8):
    """
    Build a trajectory mask for `target_lineage` that includes:
    1. All cells of target_lineage
    2. Origin cells (HSC/MPP) that are spatially adjacent to the lineage 
       trajectory — not ALL HSC/MPP, just those on the path to this lineage
    3. Cells with 'CLOUD' label that fall within corridor_width of the 
       target lineage centroid direction
    
    Excludes cells from OTHER terminal lineages (GMP cells don't belong 
    in the MEP trajectory).
    """
    lin_mask = cell_types == target_lineage
    origin_mask = np.isin(cell_types, list(origin_cell_types))
    
    if lin_mask.sum() < 5:
        return lin_mask
    
    # Direction vector: from origin centroid to lineage centroid in 2D
    origin_centroid = V_dec2[origin_mask].mean(axis=0)
    lin_centroid = V_dec2[lin_mask].mean(axis=0)
    direction = lin_centroid - origin_centroid
    direction /= np.linalg.norm(direction) + 1e-8
    
    # For origin cells: include only those whose projection onto this direction
    # is positive (i.e., geometrically between origin and target lineage)
    origin_proj = (V_dec2[origin_mask] - origin_centroid) @ direction
    origin_in_corridor = origin_proj > -0.3    # allow slight negative projection
    
    # Perpendicular distance from the trajectory line
    origin_vec = V_dec2[origin_mask] - origin_centroid
    perp = origin_vec - origin_proj[:, None] * direction
    origin_dist_from_line = np.linalg.norm(perp, axis=1)
    origin_close_to_path = origin_dist_from_line < corridor_width
    
    origin_idx = np.where(origin_mask)[0]
    selected_origin = origin_idx[origin_in_corridor & origin_close_to_path]
    
    traj_mask = lin_mask.copy()
    traj_mask[selected_origin] = True
    return traj_mask
```

This adds only the HSC/MPP cells that are spatially on the path toward the target lineage — excluding, for example, the HSC cells that are adjacent to the MEP arm when building the GMP trajectory.

### Fix G2 — Regress Out Cell Cycle Before Computing Gene-Pseudotime Correlations

Add a cell cycle scoring step using standard S-phase and G2M-phase gene sets, then regress the cell cycle score out of each gene's expression before computing Spearman correlation with pseudotime:

```python
def regress_out_cell_cycle(X_lin, gene_names, pt_lin):
    """
    Estimate per-cell cell cycle phase using S-phase and G2M gene signatures,
    then residualize expression against the cell cycle score.
    Returns residualized X_lin where cell-cycle-correlated variation is removed.
    """
    # Standard hematopoietic cell cycle gene sets (G1/S and G2/M)
    S_GENES = {'MCM5','PCNA','TYMS','FEN1','MCM2','MCM4','RRM1','UNG',
               'GINS2','MCM6','CDCA7','DTL','PRIM1','UHRF1','MLF1IP',
               'HELLS','RFC2','RPA2','NASP','RAD51AP1','GMNN','WDR76',
               'SLBP','CCNE2','UBR7','POLD3','MSH2','ATAD2','RAD51',
               'RRM2','CDC45','CDC6','EXO1','TIPIN','DSCC1','BLM','CASP8AP2',
               'USP1','CLSPN','POLA1','CHAF1B','BRIP1','E2F8'}
    G2M_GENES = {'HMGB2','CDK1','NUSAP1','UBE2C','BIRC5','TPX2','TOP2A',
                 'NDC80','CKS2','NUF2','CKS1B','MKI67','TMPO','CENPF',
                 'TACC3','FAM64A','SMC4','CCNB2','CKAP2L','CKAP2',
                 'AURKB','BUB1','KIF11','ANP32E','TUBB4B','GTSE1','KIF20B',
                 'HJURP','CDCA3','HN1','CDC20','TTK','CDC25C','KIF2C',
                 'RANGAP1','NCAPD2','DLGAP5','CDCA2','CDCA8','ECT2','KIF23',
                 'HMMR','AURKA','PSRC1','ANLN','LBR','CKAP5','CENPE',
                 'CTCF','NEK2','G2E3','GAS2L3','CBX5','CENPA'}
    
    gene_list = list(gene_names)
    
    # Score each cell: mean expression of S-phase genes
    s_idx = [i for i, g in enumerate(gene_list) if g in S_GENES]
    g2m_idx = [i for i, g in enumerate(gene_list) if g in G2M_GENES]
    
    if not s_idx and not g2m_idx:
        return X_lin   # no cell cycle genes in this gene set
    
    s_score = X_lin[:, s_idx].mean(axis=1) if s_idx else np.zeros(len(X_lin))
    g2m_score = X_lin[:, g2m_idx].mean(axis=1) if g2m_idx else np.zeros(len(X_lin))
    
    # Regress out: fit X_gene ~ s_score + g2m_score, keep residuals
    from numpy.linalg import lstsq
    cc_matrix = np.column_stack([s_score, g2m_score, np.ones(len(X_lin))])
    coefs, _, _, _ = lstsq(cc_matrix, X_lin, rcond=None)
    X_residual = X_lin - cc_matrix @ coefs
    
    return X_residual.astype(np.float32)

# In analyze_trajectory_genes, after building X_lin:
X_lin_cc = regress_out_cell_cycle(X_lin, A_genes, pt_lin)
corrs = np.array([spearmanr(pt_lin, X_lin_cc[:, g])[0] ...])
```

After this, STMN1, MCM2, PCNA, and other cell-cycle genes will have their expression variation decoupled from pseudotime, and genuine lineage markers (CSF1R for CMP, MPO for GMP, GATA1 for MEP) will rise to the top.

### Fix G3 — Use Lineage-Specific Pseudotime Rescaling

The current pseudotime assigns each cell a score in [0,1] relative to the global root-to-terminal distance. For gene trajectory analysis *within a lineage*, what matters is the relative pseudotime order *within that lineage's trajectory*, not the global ordering. Rescale:

```python
pt_lin = vel_pseudotime[traj_mask]
# Rescale to [0, 1] within this trajectory's range
pt_lin = (pt_lin - pt_lin.min()) / (pt_lin.max() - pt_lin.min() + 1e-8)
```

This prevents the global pseudotime compression from flattening within-lineage differences. Even if two lineages overlap in global pseudotime (e.g., MEP and GMP branches diverge at similar pseudotime values), their within-lineage orderings should be clear.

---

## Implementation Plan: Correct Order of Operations

The critical insight is that **these problems are hierarchical** — Problem 1 (atom collapse) causes Problem 2 (flat ψ → collapsed pseudotime) which causes Problem 4 (cell-cycle artifact in genes). Problem 3 (velocity arrows) is partially independent but benefits from the atom fix. Do them in this order:

### Step 1 (run first, inspect before proceeding): Fix atom initialization and coverage

Add `atom_coverage_loss`, change Phase 2 temperature schedule, and replace KMeans initialization with ATAC-discordance-based seeding. After training, plot the four atom weight distributions: each atom should have mean weight ~0.25 across the dataset, and no atom should have >50% of cells as its dominant atom.

**Success criterion:** `dominant atom cell counts ≈ [250, 260, 260, 265]` rather than `[~800, ~50, ~100, ~85]`.

**If atom counts are still unbalanced after this:** increase `W_COVERAGE` to 10.0 and/or add a **minimum atom usage penalty**: for any atom `d` where `mean(mix_weights[:, d]) < 0.1`, add a loss `−100 × mean(mix_weights[:, d])` to force the encoder to use all atoms. Dead atoms (like the atoms 4/5 in the 6-atom run) are a known failure mode for mixture models; aggressive regularization is appropriate.

### Step 2 (after Step 1 succeeds): Revert Dijkstra to original cost function

With balanced atoms and a real ψ gradient, revert the pseudotime to the original `cost = δψ / dist_2d`. The Round 1 Dijkstra revert is critical — do not use vel_pseudotime as the flow potential.

Check: does pseudotime now show a gradient (not binary)? Expect ρ ≥ 0.55 vs FACS pseudotime with balanced atoms.

### Step 3 (after Step 2 succeeds): Replace weighted-centroid arrows with ψ-gradient arrows

Implement `compute_psi_gradient_velocity` using local linear regression. Apply 1-hop diffusion (not 4-hop). The streamlines should now show outward divergence from the stem cluster.

**Success criterion:** Entropy-velocity Spearman ρ flips positive (ρ ≥ +0.15). Streamlines show arrows leaving the HSC/MPP region and fanning into GMP (upper-right), MEP (lower-right of the committed region), and MLP (upper-left).

### Step 4 (after Step 3, or in parallel): Fix gene trajectories

Apply `build_lineage_trajectory` (directed corridor), `regress_out_cell_cycle`, and within-lineage pseudotime rescaling. 

**Success criterion:** CSF1R, CEBPA, or MPO appear in the top-5 increasing genes for CMP/GMP. Ground-truth recovery exceeds 20% for at least two lineages.

---

## Why Round 1's Diagnosis Was Wrong on the Key Point

Round 1 correctly identified that "the ψ gradient points inward in 2D" (Problem 3) but proposed fixing it by substituting vel_pseudotime as the flow potential. This was wrong for two reasons:

1. **It is circular**: vel_pseudotime is derived from ψ (via Dijkstra). Using it as input to the same computation that creates velocity arrows, which are then used to compute vel_pseudotime... creates a feedback loop that amplifies whatever the existing pseudotime looked like, rather than correcting it.

2. **It fixed the symptom (wrong arrow direction) rather than the cause** (flat ψ in the Atom 0 interior). Even if vel_pseudotime had been perfectly correct from a previous run, using it as the flow potential in 2D would still have the cross-lineage kNN cancellation problem in the stem region.

The correct approach is: (a) fix the atom structure so ψ has real gradient everywhere, (b) switch from weighted-centroid arrows to ψ-gradient arrows (which are inherently directional without needing a flow potential at all).

The ψ-gradient approach (local linear regression) is not affected by the inward-pointing geometry problem because it does not compute centroid displacements — it fits the ψ surface locally and reads the gradient direction. If ψ increases from lower-left to upper-right in a local neighborhood, the arrow points upper-right, regardless of where the terminal cells happen to be in the global 2D layout.

---

## Summary Table

| Problem | Root Cause | Fix | Expected Outcome |
|---|---|---|---|
| Atom 0 dominates (80% of cells) | KMeans init on imbalanced 2D space + no coverage constraint | ATAC-discordance init + atom_coverage_loss (W=5) + Phase 2 temp = 0.5→0.2 | Atom counts ≈ equal; ψ has real gradient everywhere |
| Pseudotime collapsed to binary | vel_pseudotime used as Dijkstra cost (Round 1 fix) | Revert Dijkstra to original `cost = δψ/dist_2d` | Pseudotime gradient resumes; ρ ≥ 0.55 |
| Velocity arrows circular/wrong | Weighted-centroid displacement is geometrically wrong for inward-pointing ψ gradients | Replace with ψ-gradient via local linear regression + 1-hop diffusion | Streamlines diverge from stem; entropy-velocity ρ flips positive |
| STMN1 dominates gene trajectories | Cell cycle contamination from mixing HSC/MPP into all lineage trajectories | Directed corridor trajectory + cell cycle regression | Lineage markers (CSF1R, GATA1, MPO) rise to top; GT recovery >20% |
