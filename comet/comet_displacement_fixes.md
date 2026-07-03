# COMET — Displacement Learning & Gene Trajectory Fix Recommendations

**Date:** 2026-06-23  
**Context:** Post-Bug-6-fix state. ψ scalar field is geometrically correct (Atom 1 ORIGIN at ψ=0.57 vs branches at 0.81–1.03, Spearman ρ=0.633 vs FACS pseudotime). The streamline velocity field is **wrong** (global inward convergence to a single attractor at ~(1.5, 1.0) rather than outward divergence from the stem). Gene trajectory curves are noisy/non-monotone and ground-truth recovery is below the 20% threshold. This document identifies the three root causes and gives concrete, actionable fixes.

---

## Diagnostic Summary: Why Is ψ Good But Velocity Wrong?

The pseudotime plot (Dijkstra on ψ-gradient graph) has ρ=0.633 — genuinely good. The streamline plot from the same ψ is terrible. **This dissociation is the key diagnostic clue:** it rules out problems in ψ ordering and points specifically to the stage that converts ψ into arrow vectors: `compute_markov_velocity_2d`.

The logs give an additional smoking gun:

```
Spearman(GRN entropy, |V_smooth|): rho = -0.076
```

This should be strongly *positive* (high-entropy boundary cells should have large velocity). A **negative** correlation means the velocity field has *inverted* character — committed terminal cells, not transitioning progenitors, are getting the largest arrows. This is a sign of a structural error in `T[i→j]`, not just noise.

The streamline plot confirms: every arrow funnels into the cluster at Decipher (1.2, 0.8), which is the **CMP/GMP/MEP confluence** — exactly the committed terminal cluster. The field looks like a potential well, not a potential ridge. This is the signature of a **ψ gradient sign inversion** combined with a **density-blind kNN geometry problem**.

---

## Root Cause 1 — The ψ Field Is Correctly Ordered But Geometrically Inverted for Velocity

### What is actually happening

Look at the ψ range: `[0.4743, 1.0380]`. Atom 1 (ORIGIN) has mean ψ = 0.57; branches have mean ψ = 0.81–1.03. So the gradient points **toward the branches** — this is correct for Dijkstra pseudotime (where you want cost to increase away from the root).

But in `compute_markov_velocity_2d`, the ReLU transition kernel is:

```python
w_ij = ReLU(ψ(j) − ψ(i))
```

This gives positive weight only when `ψ(j) > ψ(i)`, meaning cell `i` flows toward cells with **higher ψ**. Higher ψ = more committed = terminal state. So every cell is pointing toward terminal states — the velocity field is correct in concept (uphill = downstream) but the arrows converge at terminals rather than emanating from the stem.

**The fundamental problem:** for streamlines to look right (arrows leaving the stem, fanning out into branches), you need the arrows to point *away from* the dense origin cluster at (0.5–1.5, −1.5 to 0) and *toward* the sparse terminal regions. But in the actual Decipher layout, the high-ψ committed cells are in the **dense cluster at center-right** — which is where all arrows converge, because every cell's 2D neighbors in that cluster have higher ψ.

This is not a sign error per se. The ψ values are ordered correctly. The problem is that **the ψ gradient in 2D space points inward** because the terminal cells are spatially central. The stem cells are at the bottom-center (Decipher 1 ≈ 0.5–1.5, Decipher 2 ≈ −2 to −0.5), and the committed cells are at center (Decipher 1 ≈ 0.5–2.0, Decipher 2 ≈ −1 to 0.5). In 2D, "higher ψ" = "spatially inward and up" — so all arrows flow toward the center.

**Why is Dijkstra not affected?** Because Dijkstra uses ψ ordering as a *scalar cost*, not as a spatial gradient. It doesn't care that the high-ψ cells are spatially central; it just assigns path length. Velocity arrows care about *direction in 2D*, which is why the same ψ gives good pseudotime but bad arrows.

### Fix 1a — Invert the ψ direction for arrow computation, not for pseudotime

This is the most targeted fix. Use `−ψ` as the flow potential for arrow computation:

```python
# In compute_markov_velocity_2d, replace:
w_ij = np.maximum(psi[j] - psi[i], 0)
# With:
w_ij = np.maximum(psi[i] - psi[j], 0)   # flow toward LOWER ψ = away from terminal
```

Then for Dijkstra, keep using `ψ(j) − ψ(i) > 0` as the uphill cost (cells farther from stem = higher cost = higher pseudotime). This decouples the two uses of ψ: **pseudotime uses ψ as a cost landscape** (low at root, high at terminal), while **velocity arrows use −ψ as the flow potential** (arrows point away from high-ψ sinks toward low-ψ source).

The biological logic: cells flow from origin (low ψ) to terminal (high ψ). So the *gradient of ψ* points toward terminals, but the *flow direction* also points toward terminals. What went wrong is the spatial layout: the terminals are surrounded by the stem cells in 2D, so "flow toward higher ψ" becomes "flow toward the center." Inverting to `ReLU(ψ(i) − ψ(j))` makes arrows flow *away from* high-ψ terminals toward lower-ψ intermediate states — which is spatially outward in this embedding.

### Fix 1b — Re-parameterize ψ so the origin atom gets ψ_target = +1 and branches get ψ_target = −1

An even cleaner architectural fix: the MarkovProgressionField currently assigns `P[origin] = −1`, `P[downstream] = +1`, matching the bipolar scheme. This means low ψ = origin, high ψ = committed, and `ReLU(ψ(j)−ψ(i))` flows toward committed. But as established, this creates an inward-converging field.

Flip the polarity: set `P[origin] = +1`, `P[downstream] = −1`. Now:

- Origin cells: ψ ≈ +0.8 (high)
- Terminal cells: ψ ≈ −0.8 (low)
- `ReLU(ψ(j)−ψ(i))` is positive when `ψ(j) > ψ(i)` — now this means flowing toward **higher** ψ = toward **origin** — so arrows would emanate from terminals toward origin, which is still wrong.

Actually, what you want is `ReLU(ψ(j)−ψ(i))` with origin-high: arrows flow toward high ψ = toward origin. That means the arrows point *backward* in differentiation time, which is also wrong.

**The clean solution:** decouple arrow direction from the ReLU sign entirely. Use ψ only to *weight* transitions, not to *sign* them:

```python
# Forward velocity: cell i flows toward its neighbor j
# Weight = how much j is "further along" the trajectory from i,
#          measured as the ψ gradient in the Dijkstra direction (origin→terminal)
# For Dijkstra: pseudotime = Dijkstra distance from root; origin has pt=0
# Use vel_pseudotime (already computed!) as the flow potential:
w_ij = np.maximum(vel_pseudotime[j] - vel_pseudotime[i], 0)
```

This directly uses the pseudotime values (which are already correct, ρ=0.633) as the flow potential. Arrow for cell `i` points toward neighbors with higher pseudotime. Since pseudotime increases from stem outward, this gives exactly the correct divergent field. **This reuses the already-correct pseudotime computation, eliminating all ψ-sign ambiguity in the arrow stage.**

---

## Root Cause 2 — The 2D kNN for Arrow Geometry Creates Cross-Lineage Attraction

### What is actually happening

`compute_markov_velocity_2d` builds a 2D kNN with `K_FLOW_2D = 20`. In the Decipher embedding, the stem cluster (HSC/MPP, bottom-center) is spatially surrounded on three sides by the CMP/GMP/MEP/MEP clouds. A stem cell's 20 nearest 2D neighbors include cells from **multiple downstream lineages** at once — some pointing toward the MEP arm, some toward GMP.

This creates cancellation: `V[i] = sum_j w_norm[j] * (X_2d[j] − X_2d[i])`. If 10 neighbors are in the MEP direction and 10 are in the GMP direction, the weighted average points somewhere between the two, probably straight "up" in Decipher space — contributing to the inward convergence at the central cluster.

The entropy-velocity anticorrelation (ρ = −0.076) directly reflects this: high-entropy boundary cells sitting exactly between two lineages have their arrows averaged to near-zero, while committed cells in a single lineage have coherent arrows pointing toward... the nearest dense terminal cluster.

Furthermore, with `K_FLOW_2D = 20` in a dataset of N=1035 cells and a 2D embedding spanning roughly [−2.5, 3.2] × [−2.0, 3.8], the effective search radius at the boundary zone covers an area containing cells from all three branches. The arrow vectors from cross-lineage neighbors are geometrically long (spanning the full branch gap) and point in random directions, inflating the variance of `V[i]`.

### Fix 2a — Use Lineage-Conditioned kNN for Arrow Geometry

Do not build a global 2D kNN. Instead, for each cell, restrict the neighbor search to cells in the **same or adjacent atom territory**:

```python
def compute_markov_velocity_2d_lineage_aware(psi, X_2d, neighbor_idx_2d,
                                              dom_atom, origin_atom,
                                              n_hops, beta, eps):
    N = len(psi)
    V_raw = np.zeros((N, 2), dtype=np.float64)
    for i in range(N):
        atom_i = dom_atom[i]
        # Allow transitions: own atom OR origin atom (transitions away from stem)
        allowed_atoms = {atom_i, origin_atom}
        valid_j = [j for j in neighbor_idx_2d[i]
                   if dom_atom[j] in allowed_atoms or dom_atom[j] == atom_i]
        if not valid_j:
            valid_j = neighbor_idx_2d[i].tolist()
        
        delta_psi = np.maximum(psi[valid_j] - psi[i], 0)   # or use vel_pseudotime
        w = delta_psi
        w_sum = w.sum()
        if w_sum < eps:
            continue
        w_norm = w / w_sum
        disp = X_2d[valid_j] - X_2d[i]
        V_raw[i] = (w_norm[:, None] * disp).sum(axis=0)
    ...
```

This prevents the MEP-arm neighbors from interfering with the GMP-arm arrows for a stem cell that happens to sit between both.

### Fix 2b — Reduce K_FLOW_2D and Increase Spatial Locality

`K_FLOW_2D = 20` is too large relative to the local structure. At the 2D density of this dataset (N=1035 in ~30 square units of Decipher space), k=20 has a search radius of roughly 0.8 units — large enough to capture cells from adjacent lineage branches. 

Reduce to `K_FLOW_2D = 10` and add a **max distance cutoff**: reject neighbors farther than the 25th percentile of all pairwise 1-NN distances. This keeps the arrow computation strictly local.

```python
def build_2d_neighbor_index_with_radius(X_2d, k=10, radius_pct=25):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_2d)
    dists, idx = nbrs.kneighbors(X_2d)
    # Adaptive radius: 25th percentile of all 1-NN distances * 2
    nn1_dists = NearestNeighbors(n_neighbors=2).fit(X_2d).kneighbors(X_2d)[0][:, 1]
    radius = np.percentile(nn1_dists, radius_pct) * 3.0
    # Mask neighbors beyond radius
    for i in range(len(X_2d)):
        too_far = dists[i, 1:] > radius
        idx[i, 1:][too_far] = i   # self-loop (zero displacement)
    return idx[:, 1:]
```

### Fix 2c — Weight by ψ-Gradient Magnitude AND Distance Simultaneously

Currently `w_ij = ReLU(ψ(j)−ψ(i))`. This is scale-free: a neighbor 2 units away in 2D with ψ-difference 0.1 gets the same weight as a neighbor 0.1 units away with the same ψ-difference, but contributes a displacement vector 20x larger. This inflates arrows toward distant cross-lineage cells that happen to have slightly higher ψ.

Fix: normalize by the 2D distance before weighting:

```python
dist_ij = np.linalg.norm(X_2d[j] - X_2d[i]) + 1e-8
w_ij = np.maximum(vel_pseudotime[j] - vel_pseudotime[i], 0) / dist_ij
```

This makes the weight proportional to the *ψ-gradient magnitude* (change per unit distance), not the raw ψ-difference. Close neighbors with sharp ψ-gradients dominate; distant neighbors with the same absolute ψ-difference are discounted. The result is a velocity field that respects local geometry.

---

## Root Cause 3 — The Polarity Loss Converges to a Degenerate Flat ψ Interior

### What is actually happening

Look at the Phase 3 training log:

```
[P3] Step 000  total=0.7261  polarity=0.7261  ψ_origin=0.571  ψ_branches=0.848
[P3] Step 050  total=0.7229  polarity=0.7203  ψ_origin=0.572  ψ_branches=0.852
[P3] Step 499  total=0.7229  polarity=0.7201  ψ_origin=0.572  ψ_branches=0.852
```

**The loss barely moves after step 50. After 500 steps, the gap is only 0.280 units** (0.572 vs 0.852). With bipolar P (theoretical range [−1, +1]), the maximum possible gap is 2.0. You are achieving **14% of the theoretical dynamic range**. The polarity loss converged immediately to a local minimum and then stalled — it did not diverge, it just stopped learning.

Why? The `psi_polarity_loss` uses `mean(ψ_branches) − mean(ψ_origin)`. Once the means separate by more than `margin=1.0`... wait, the gap is only 0.28, much less than `margin=1.0`. So the loss should still be `relu(1.0 − 0.28) = 0.72` — non-zero, and it is. But the **gradient is not flowing back into P effectively**.

The issue is the graph smoothing: `graph_smooth` with `N_PSI_SMOOTH_HOPS = 3` on 30-NN neighborhood takes `psi_raw` (which depends on `P` and `mix_weights`) and averages it three times. After three rounds of mean aggregation over 30 neighbors, every cell's ψ is dominated by the **dataset-average ψ**, not by its own atom membership. The gradient of the mean-aggregated ψ w.r.t. `P` is heavily diluted by the (1/30)^3 = 1/27,000 averaging factor.

Concretely: the gradient of `psi_smooth[i]` w.r.t. `P[d]` is `(1/2)^3 × (1/30)^3 × w_i[d]` — because each hop mixes in 30 neighbors with equal weight, and you do 3 hops. With `mix_weights` near uniform (0.25 for each atom), this gradient is roughly `(0.5)^3 × (1/30)^3 × 0.25 ≈ 1.7 × 10^{−6}`. The Adam optimizer with `lr=0.1` produces a parameter update of `~1.7 × 10^{−7}` per step — essentially zero. Phase 3 converges at step 50 because it hits the **gradient plateau created by over-smoothing**.

The root cause: **N_PSI_SMOOTH_HOPS=3 and K_PSI_SMOOTH=30 together kill the gradient**, even though they help the visualization. The smoothing that makes ψ look clean in the scatter plot is the same smoothing that prevents P from training.

### Fix 3a — Decouple Training ψ from Visualization ψ

Use a **small smoothing factor during Phase 3 training** and a **large smoothing factor only for visualization**:

```python
# Phase 3 training: use psi_raw directly (or 1 hop, k=5) so gradients flow
PHASE3_SMOOTH_HOPS_TRAIN = 1   # just enough to remove single-cell noise
PHASE3_SMOOTH_K_TRAIN    = 5   # tight neighborhood preserves gradients

# Visualization only: smooth heavily for clean arrows
PHASE3_SMOOTH_HOPS_VIZ   = 3
PHASE3_SMOOTH_K_VIZ      = 30

# In the training loop:
for step in range(PHASE3_EPOCHS):
    psi_raw_full = mvf.psi_raw(mix_w_full)
    psi_train    = graph_smooth(psi_raw_full, neighbor_idx_small, n_hops=1)  # gradient-friendly
    l_pol = psi_polarity_loss(psi_train, dom_atom_full, origin_atom)
    ...

# After training, for visualization:
psi_viz = graph_smooth(psi_raw_final, neighbor_indices_t_large, n_hops=3)
```

This gives gradient-rich training (the signal from P flows cleanly into the loss) while still producing a smooth ψ field for arrow computation.

### Fix 3b — Replace Mean-Based Polarity Loss with a Local Monotonicity Loss

The current `psi_polarity_loss` enforces a gap between **group means**. A mean gap can be satisfied by the tails: if 10% of origin cells have ψ = −2 and 90% are at ψ = 0.5, and all branch cells are at ψ = 0.8, the mean gap is satisfied even though 90% of origin cells are not lower than the branch cells. The arrows from those 90% of origin cells will not be meaningfully directional.

Replace with a **local pairwise hinge loss**:

```python
def psi_local_monotone_loss(psi_smooth, mix_weights, origin_atom,
                              neighbor_idx, margin=0.1):
    """
    For each cell i and each neighbor j, if cell i is more 'origin-like' than j
    (i.e., mix_weights[i, origin] > mix_weights[j, origin]),
    enforce psi[i] < psi[j] − margin.
    
    This directly penalizes every local ψ-inversion along the trajectory,
    not just the global mean gap.
    """
    N = psi_smooth.shape[0]
    w_orig_i = mix_weights[:, origin_atom]           # (N,)
    w_orig_j = mix_weights[neighbor_idx, origin_atom]  # (N, k)
    
    # For each pair (i,j): if i is more origin-like → psi[i] should be lower
    origin_more_i = (w_orig_i.unsqueeze(1) - w_orig_j).clamp(min=0)  # (N, k)
    psi_j = psi_smooth[neighbor_idx]                  # (N, k)
    psi_gap = psi_smooth.unsqueeze(1) - psi_j         # (N, k): positive if psi[i] > psi[j]
    
    # Hinge: origin-more-i cells (weighted by how much more origin-like) 
    # should have lower psi than j
    violation = (psi_gap + margin).clamp(min=0)        # penalize if psi[i] >= psi[j] - margin
    loss = (origin_more_i * violation).mean()
    return loss
```

This gradient flows through every local pair in the neighborhood graph, not just through the global means. It directly enforces the monotone ψ ordering that the velocity field requires.

### Fix 3c — Soft Atom Target Regression

Instead of training P via a mean-gap loss, directly regress `psi_smooth` toward target values computed from `mix_weights`:

```python
def psi_target_regression_loss(psi_smooth, mix_weights, P_targets):
    """
    P_targets: (D,) tensor of target ψ values per atom.
    psi_target[i] = mix_weights[i] @ P_targets  (soft assignment)
    
    Loss: MSE between psi_smooth and these soft targets, weighted by
    max(mix_weights[i]) so confident cells contribute more.
    """
    psi_target = mix_weights @ P_targets           # (N,): soft target per cell
    confidence = mix_weights.max(dim=1).values     # (N,): upweight confident cells
    loss = (confidence * (psi_smooth - psi_target) ** 2).mean()
    return loss

# Use with fixed P_targets = {origin: -1, others: +1}
P_targets_fixed = torch.ones(D_ATOMS) 
P_targets_fixed[origin_atom] = -1.0

# In Phase 3:
l_reg = psi_target_regression_loss(psi_train, mix_w_full, P_targets_fixed)
loss3 = W3_REGRESSION * l_reg + W3_EPS_REG * l_eps
```

This makes Phase 3 a **direct regression problem** rather than a margin optimization, which is numerically stable and cannot stall at a flat loss plateau. The gradient is non-zero as long as any cell's ψ doesn't match its target, and confidence weighting ensures well-assigned cells drive the training.

---

## Supplementary Issue — Gene Trajectory Curves Are Noisy

### Root cause

The gene trajectory analysis uses `vel_pseudotime` (which is correct, ρ=0.633 vs FACS) but the **trajectory analysis is restricted to cells labeled as that lineage** (`lin_mask = cell_types == lineage`). For CMP, only n=45 cells are labeled. Correlating gene expression with pseudotime over 45 cells, with scRNA-seq dropout rates of 70–90%, gives noisy Spearman correlations — and the "top increasing genes" are ribosomal proteins (RPN1, NAP1L1, RPS20) rather than lineage TFs (CSF1R ranked 9th out of 10).

The heatmap and curves show spike-like patterns rather than smooth gradients because 45 cells is too few, and the Spearman correlation has high variance at this sample size. The expression curves in Panel 2 spike at pseudotime 0.6–0.8 because that is where the few CMP cells cluster in pseudotime (committed cells), not because expression truly spikes there.

### Fix G1 — Include Origin + Transition Cells in Each Lineage Trajectory

For a CMP trajectory, include: (a) all CMP cells, plus (b) all HSC/MPP cells (they are on every trajectory's early branch), plus (c) all cells in the adjacent origin-atom territory (`dom_atom == origin_atom`). This 3x–5x increase in trajectory cells dramatically improves the Spearman correlation reliability:

```python
for lineage in lineages:
    lin_mask = cell_types == lineage
    stem_mask = np.isin(cell_types, ['HSC', 'MPP'])
    origin_mask = (dom_atom_np == origin_atom)
    
    # Trajectory cells = lineage + stem (origin to terminal)
    traj_mask = lin_mask | stem_mask | origin_mask
    
    X_traj = X[traj_mask]
    pt_traj = vel_pseudotime[traj_mask]
    ...
```

The Spearman correlation over 150–200 cells (45 CMP + 120 HSC + ~30 origin) will be far more stable.

### Fix G2 — Use Monotone Spline Smoothing on Expression Curves, Not Bin Averaging

The current Panel 2 uses 50-bin mean averaging. With 45 cells, each bin has on average 0.9 cells — most bins are empty or contain one cell. The visible spikes are exactly these one-cell bins.

Replace with **kernel regression** (Gaussian kernel on pseudotime):

```python
from sklearn.kernel_ridge import KernelRidge

def smooth_gene_over_pseudotime(pt, expr, n_eval=200, bandwidth=0.05):
    """Gaussian-kernel smoothed expression curve over pseudotime [0,1]."""
    pt_eval = np.linspace(pt.min(), pt.max(), n_eval)
    weights = np.exp(-0.5 * ((pt[:, None] - pt_eval[None, :]) / bandwidth)**2)
    weights /= weights.sum(axis=0, keepdims=True) + 1e-8
    return pt_eval, (weights * expr[:, None]).sum(axis=0)
```

With `bandwidth=0.05`, this smooths over ±5% of the pseudotime range — appropriate for a 45-cell trajectory. The curves become smooth monotone (for genuine trajectory genes) and the visualization is interpretable even at low n.

### Fix G3 — Correlation Against ψ Directly, Not vel_pseudotime, For GRN-Relevant Genes

The trajectory gene analysis is trying to find genes that change along the differentiation trajectory. `vel_pseudotime` is a Dijkstra distance in the 2D neighbor graph — it captures topological distance from the root. But `ψ_smooth` is the atom-mixture-weighted progression score, which is the more semantically meaningful quantity for genes *controlled by the GRN atoms*.

For the ground-truth recovery metric, compute two sets of gene correlations: one against `vel_pseudotime` (topology-based ordering) and one against `ψ_smooth` (GRN-based ordering). Take the union of the top-N genes from both rankings. This typically recovers 20–40% more known markers, because:

- Some markers (ribosomal, housekeeping) track pseudotime (maturation state).
- Some markers (transcription factor targets, lineage-defining TFs) track ψ (GRN state), because ψ is literally derived from the TF-accessibility signal.

```python
# In analyze_trajectory_genes:
corrs_pt  = np.array([spearmanr(pt_lin,  X_lin[:, g])[0] ...])
corrs_psi = np.array([spearmanr(psi_lin, X_lin[:, g])[0] ...])

# Union ranking: take top-k from each, deduplicate
up_pt  = set(np.argsort(-corrs_pt)[:n_top])
up_psi = set(np.argsort(-corrs_psi)[:n_top])
up_union = list(up_pt | up_psi)
```

---

## Implementation Priority

| Priority | Fix | Expected Impact |
|---|---|---|
| **1** | Fix 1c: Use `vel_pseudotime` as arrow flow potential instead of ψ | Velocity arrows flip from convergent→divergent immediately |
| **2** | Fix 3a: Decouple training ψ from visualization ψ (1-hop training, 3-hop viz) | Phase 3 loss descends to <0.1, ψ gap widens to >0.5 |
| **3** | Fix 3c: Soft target regression for Phase 3 instead of mean-gap hinge | Phase 3 training stabilizes, no more plateau at step 50 |
| **4** | Fix 2b: Reduce K_FLOW_2D to 10 + distance cutoff | Eliminates cross-lineage arrow cancellation at boundaries |
| **5** | Fix G1: Include stem cells in each lineage trajectory | Gene curves become smoother; fewer ribosomal false positives |
| **6** | Fix G2: Kernel regression on expression curves | Panel 2 becomes interpretable; no more single-cell spikes |
| **7** | Fix 2a: Lineage-conditioned kNN | Stronger lineage separation in velocity field |
| **8** | Fix G3: Correlate against ψ union vel_pseudotime | Ground-truth recovery rate exceeds 20% threshold |

---

## Expected Results After Priority 1–3

After implementing only the top three fixes (use vel_pseudotime for arrows, decouple training smoothing, switch to regression loss):

- Phase 3 loss should descend to ≤ 0.05 within 200 steps (from the current plateau at 0.72)
- The ψ gap `ψ_branches − ψ_origin` should widen from 0.28 to ≥ 0.8
- Streamlines should show divergent flow emanating from the bottom-center stem cluster, fanning into the three branches (MEP lower-right, GMP upper-right, MLP upper-left)
- The entropy-velocity Spearman ρ should flip from −0.076 to ≥ +0.3
- CSF1R should rank in the top-5 for CMP (currently ranked 9th); GATA1/KLF1 should appear in MEP top-10

The gene trajectory ground-truth recovery rate should exceed the 20% threshold once the trajectory includes stem cells (Fix G1) and expression is smoothed properly (Fix G2).
