# COMET — Controlled Pivot Action Plan

**Audit of `test_fixes_2.md` vs `test_fixes.md`, and a targeted recovery plan**
**Date:** 2026-06-23
**Reference blueprint:** `test_fixes.md` (flawed but *stable*: ψ ordering correct, ρ=0.633 vs FACS)
**Regressed build:** `test_fixes_2.md` (round‑1 + round‑2 specs applied; catastrophic cascade)

---

## 0. Executive summary

`test_fixes_2.md` did not fail because of many independent bugs. It failed because of **one** change that destroyed the model's representational basis, and then a chain of *correct‑looking* downstream code faithfully propagated that collapse into every visible artifact.

The single root cause is **Fix A1 — `initialize_from_atac_discordance`** (the ATAC‑discordance quantile seeding that replaced KMeans). Everything else in the cascade is a *symptom* of the flat ψ field that A1 produces:

| Observed symptom | Real cause | Innocent code blamed by the specs |
|---|---|---|
| Atom 0 swallows ~80% of cells | A1 collapses the 4‑lineage atom basis onto a 1‑D axis (and seeds on a technical/depth covariate) | KMeans init, "missing coverage loss" |
| Pseudotime collapses to binary {0,1} | ψ is flat over 80% of cells ⇒ Dijkstra edge costs `δψ/dist ≈ 0` everywhere off the origin island | Dijkstra cost function (it is **unchanged** and correct) |
| Velocity becomes circular loops | `∇ψ ≈ noise` on the flat plateau, interpolated by `streamplot` ⇒ closed orbits; `N_DIFFUSION_HOPS=1` removes the smoothing that hid it | Weighted‑centroid arrows |
| STMN1 / cell‑cycle dominate gene curves | Any gene high in the root island correlates ~perfectly with a binary pseudotime | "needs cell‑cycle regression" |

**The pivot is therefore narrow:** revert the atom‑initialization change, restore the working Phase‑3, and re‑apply *only* the genuinely sound ideas (ψ‑gradient arrow **direction**, cell‑cycle regression, kernel‑smoothed gene curves) on top of a **healthy ψ field**. Do not balance atoms by force — the working baseline had a 7%‑weight origin atom and still scored ρ=0.633. Atom *distinctness*, not atom *balance*, is what matters.

---

## 1. Failure‑mode analysis — the exact math steps that broke

### 1.1 The master variable: ψ dynamic range is set by the *encoder*, not by Phase 3

This is the load‑bearing fact for the entire audit. In both files:

```python
# MarkovProgressionField
P[origin]      = -1.0                       # fixed
P[downstream]  =  1.0 + 0.15*tanh(eps_raw)  # eps_raw is the ONLY trainable tensor
psi_raw[i]     = mix_weights[i] @ P         # (N,)
```

`P` is four near‑constant scalars pinned to `[-1, +1, +1, +1]` (ALPHA=0.15 bounds each downstream atom to `[+0.85, +1.15]`). Consequently the **per‑cell ψ contrast is governed entirely by `mix_weights` contrast**, i.e. by the encoder. Phase 3 cannot manufacture dynamic range it isn't handed.

A direct corollary, which the specs missed:

> If the encoder collapses (most cells → one downstream atom), then for those cells `ψ ≈ +0.8 ± noise`, and **no Phase‑3 loss can rescue it** — not the polarity hinge, not the new regression loss. Phase 3 is the wrong place to look.

In `test_fixes.md` this was fine because the **KMeans‑on‑Decipher** init gave four spatially distinct atoms; ψ had a genuine gradient from the origin pocket (ψ≈−1) out to the branches (ψ≈+0.8–1.0). The reported per‑atom means (Atom 1 = 0.57 origin, others 0.81–1.03) confirm a *spread* ψ field.

### 1.2 Fix A1 collapses the atom basis — two compounding errors

```python
def initialize_from_atac_discordance(X, gene_access, D):
    rna_norm      = X / (X.max(axis=0) + 1e-8)            # (N,G) in [0,1]
    per_cell_disc = (gene_access - rna_norm).mean(axis=1) # (N,)  <-- SCALAR per cell
    ranks         = np.argsort(np.argsort(per_cell_disc))
    cluster_labels = (ranks / (N / D)).astype(int).clip(0, D-1)  # D equal quantile bins
    return cluster_labels
```

**Error 1 — dimensionality collapse (the fatal one).** The atoms are defined as **equal quantile bins of a single scalar** `per_cell_disc`. A 4‑atom model is supposed to represent four *biologically distinct programs* (origin / GMP / MEP / MLP). Quantile‑slicing one scalar produces four atoms that differ only along **one** axis (committed ↔ stem) and are **mutually collinear in expression space**. Adjacent bins have near‑identical mean profiles, so the GRN dictionary seeded from them,

```python
lfc = np.log1p(X[bin_d].mean(0)) - np.log1p(X[~bin_d].mean(0))
W_init[d] = (lfc * M[:,t]) / range * 0.3
```

starts with low inter‑atom diversity. The encoder is then **pre‑trained by cross‑entropy to reproduce this 1‑D ordering**, baking the collapse in before Phase‑1 even starts. KMeans‑on‑`V_dec2` (the original) instead returns four *spatially separated* clusters that align with the four lineage arms — the correct basis.

**Error 2 — scale mismatch makes the "discordance" a depth covariate.** `gene_access` is passed the **raw** `gene_activity` matrix (line ~1186, `gene_access=gene_activity`), while `rna_norm ∈ [0,1]`. The subtraction `gene_activity − rna_norm` is dominated by the unnormalized magnitude of `gene_activity`, so

```
per_cell_disc[i] ≈ mean_g gene_activity[i, g]  ≈  (total ATAC signal / library depth of cell i)
```

The ranking is therefore approximately **per‑cell ATAC sequencing depth**, a technical covariate — not "ATAC‑ahead‑of‑RNA" biology. (`gene_max` is even computed and then never used, confirming the normalization was intended but dropped.) So the quantile bins are depth quartiles. This is the worst possible seeding axis.

### 1.3 Why the coverage loss (A2) cannot stop the 80% dominance

```python
def atom_coverage_loss(mix_weights, D, max_frac=0.4):
    mean_w = mix_weights.mean(dim=0)        # (D,) soft share per atom
    return torch.relu(mean_w - max_frac).sum()
```

This penalizes **mean soft weight** above 0.4. It does **not** penalize the **dominant‑atom (argmax) share**, which is what "Atom 0 = 80% of cells" measures. The two diverge sharply:

> A configuration with `mean_w = [0.40, 0.20, 0.20, 0.20]` satisfies the coverage loss exactly (excess = 0), yet because atom 0's soft weight sits above all others for the entire dense committed cluster, atom 0 is the **argmax for ~80% of cells**. Coverage is satisfied and the collapse is fully intact.

So A2 is a no‑op against the actual failure metric. Worse, **A3 (softening Phase‑2 temperature 0.2 → 0.5)** keeps the softmax diffuse for longer, which *raises* the dominant atom's mean soft weight (more cells share weight with it), nudging the system *toward* the degenerate corner, not away from it.

### 1.4 Phase‑3 regression loss (Fix 3c) is a near‑identity — no lever

```python
psi_target  = mix_weights @ P_targets          # P_targets = [-1,+1,+1,+1]
psi_train   = graph_smooth(mix_weights @ P, k=5, hops=1)   # P ≈ [-1,+1,+1,+1]
loss        = (confidence * (psi_train - psi_target)**2).mean()
```

Because `P ≈ P_targets` (they differ only by the `0.15*tanh(eps)` term), the regression target is essentially **the unsmoothed version of the loss's own input**. The only residual the optimizer can act on is the *smoothing shrinkage* introduced by `graph_smooth`, and the only knob is `eps_raw` capped at ±0.15. Net effect: the loss sits near zero from step 0 and `eps_raw` barely moves. It neither helps nor hurts dynamic range — it simply **cannot widen ψ**, which is exactly what a collapsed encoder needs. (The Round‑1 diagnosis that "over‑smoothing kills the P gradient" was aimed at the wrong parameter: the relevant gradient path is encoder → `mix_weights`, which Phase 3 freezes.)

### 1.5 The collapse propagates — three innocent stages

**(a) Binary pseudotime.** Dijkstra cost is **unchanged** from `test_fixes.md`:

```python
delta = psi[j] - psi[i]
if delta > 0:
    cost = delta / (||V_dec2[j]-V_dec2[i]|| + 1e-8)
```

On the flat ψ plateau (80% of cells at ψ≈+0.8 ± 0.02), `delta` is mix‑weight noise ≈ 0.02 ⇒ `cost ≈ 0` for every intra‑plateau edge. Multi‑source Dijkstra from the origin island therefore reaches almost every cell at near‑zero cumulative cost; after min–max normalization the distance field is two‑valued: 0 at/near roots, ≈constant elsewhere → the observed binary {0,1}. **The Dijkstra code is correct; the flatness upstream is the disease.** (Round‑2 §Problem 2 reached this conclusion and correctly *kept* the original cost — but then failed to deliver the non‑flat ψ that would make it work, because A1 made flatness worse.)

**(b) Circular‑loop velocity.** The new arrow is the local WLS gradient `β = [∂ψ/∂x, ∂ψ/∂y]`:

```python
A = X_c.T @ W @ X_c + 1e-3*I
beta = solve(A, X_c.T @ (w * psi_c))   # arrow = beta
```

A *true* gradient field is irrotational (`∇×∇ψ = 0`) and **cannot** form closed loops — so the loops are diagnostic of **noise, not signal**. Over the flat plateau the true gradient is ~0 and `β` is dominated by mix‑weight noise: independent, inconsistent per‑cell vectors. `plot_velocity_streamlines` then **interpolates these onto a regular grid** (`griddata`) and integrates — a random vector field under grid interpolation generically yields closed orbits. Reducing `N_DIFFUSION_HOPS` 4 → 1 (intended to stop the old *inward convergence*) removed the only smoothing that previously masked this turbulence. The old field was *coherent‑but‑wrong* (inward); the new field over flat ψ is *incoherent* (loops). The regression is not the problem — the substrate it reads is.

**(c) Cell‑cycle gene takeover.** With a binary pseudotime, Spearman(`pt`, gene) reduces to "is this gene higher in the root island than outside it?" Cycling progenitors live in the root island, so **every** S/G2M gene (STMN1, MCM2, PCNA…) scores near the top. The Round‑2 cell‑cycle regression (`regress_out_cell_cycle`) removes *some* of this, but it is fighting a pseudotime that is structurally degenerate; it cannot recover lineage TFs while `pt` is binary.

---

## 2. Why the design specs clashed with *this* data

The two design docs are internally reasonable but rest on premises this dataset violates.

**Premise 1 (false here): "ψ range is trained in Phase 3."** Both specs invest heavily in Phase‑3 loss redesign (decoupled smoothing, regression loss, local monotonicity). But §1.1: ψ range = `mix_weights` contrast × `P` (≈fixed). Phase 3 freezes the encoder, so **no Phase‑3 edit can change the ψ field's range.** Effort spent there was structurally incapable of moving the needle.

**Premise 2 (false here): "the atoms need balancing."** Round‑2 framed "Atom 0 = 80%" as a *coverage* failure and prescribed forced balance (`[~260,~260,~260,~260]`). But the working baseline (`test_fixes.md`) had a **tiny** origin atom (mean weight ≈ 0.07) and still produced a correctly‑ordered ψ and ρ=0.633. The biology *is* imbalanced (small dense stem, large diffuse committed compartment). The pathology in `_2` is not imbalance — it is **degeneracy** (one atom is no longer a distinct program). Balancing a degenerate basis does not restore distinctness; it just spreads the same collapsed representation across four labels. The fix that was actually needed (distinct lineage atoms) was *replaced* by A1, which is why the "fix" deepened the wound.

**Premise 3 (false here): "ATAC discordance is a clean 1‑D ordering axis we can quantile‑bin."** Single‑cell ATAC gene‑activity is sparse, depth‑confounded, and — critically — **multi‑lineage**. Compressing it to one scalar per cell and slicing into bins throws away the branch structure that the 4 atoms exist to capture, and (via the scale bug) leaks library depth. The spec's own goal ("seed on a biologically meaningful axis") is unreachable from a single mean‑over‑genes scalar.

**Premise 4 (partially true): "the original velocity field is wrong."** This one is *correct* — the inward convergence in `test_fixes.md` is a real defect. The ψ‑gradient arrow idea is the right cure **in principle** (it makes direction = steepest‑ascent of ψ, which points outward where ψ rises away from the origin). The spec's error was *sequencing*: it deployed gradient arrows and slashed diffusion **before** securing a non‑flat ψ, so the cure had nothing valid to read.

**Net:** Rounds 1–2 optimized the two stages that were *downstream* of the real fault (Phase‑3 ψ‑training and the arrow formula) while *actively breaking* the one stage that was upstream of everything (atom initialization). The cascade is the inevitable result of fixing symptoms while damaging the cause.

---

## 3. Controlled pivot — action plan

**Operating principle:** keep the `test_fixes.md` layout. Change the **minimum** needed to (i) undo the basis collapse, then (ii) layer back *only* the sound ideas, each behind an explicit gate. Run the steps **in order** and verify the gate before proceeding — most steps are no‑ops or harmful until the previous gate passes.

### Step 0 — Revert the atom‑initialization change (the one fix that matters)

Restore KMeans‑on‑Decipher seeding. This alone should lift the cascade.

```python
# In the training loop, revert the init call to the test_fixes.md form:
cluster_labels, W_init = initialize_grn_dictionary_from_clusters(
    X=X, V_dec2=V_dec2, M=M, D=D_ATOMS, A_tfs=A_tfs, sub_g2i=sub_g2i,
    # gene_access=...   <-- REMOVE. Do NOT pass gene_access (kills the quantile path).
)
```

If you want ATAC to *inform* (not *define*) the seed, do it **without** collapsing to 1‑D and **with** matched scales — as an optional enhancement only:

```python
def hybrid_seed(X, V_dec2, gene_activity, D):
    # Scale-matched discordance: normalize ATAC the SAME way as RNA (per-gene max → [0,1])
    rna_norm  = X            / (X.max(0)            + 1e-8)
    atac_norm = gene_activity/ (gene_activity.max(0)+ 1e-8)
    disc      = (atac_norm - rna_norm).mean(1, keepdims=True)   # (N,1) now genuinely "ATAC ahead of RNA"
    disc      = (disc - disc.mean()) / (disc.std() + 1e-8)
    # Cluster in [Decipher_2D ⊕ small disc feature] so geometry (4 lineages) dominates,
    # ATAC only breaks ties along the stem↔committed axis. KEEP 2D structure.
    feat = np.concatenate([V_dec2, 0.5 * disc], axis=1)         # (N,3)
    return KMeans(D, n_init=10, random_state=42).fit_predict(feat)
```

**Gate 0 (must pass before continuing):** dominant‑atom counts are *lineage‑shaped*, not a single blob. You are **not** aiming for `[260,260,260,260]`; the baseline's small origin atom is expected and healthy. Pass condition:
- no atom is the argmax for > ~55% of cells, **and**
- the origin atom (later `disc.argmax`) is a distinct, spatially‑coherent pocket, **and**
- ψ per dominant‑atom group recovers a spread comparable to baseline (origin lowest, branches ≥ 0.6 above it).

### Step 1 — Keep collapse‑prevention *gentle* and correctly targeted

Do **not** reinstate A2/A3 as written. If, after Step 0, any atom is dead (`mean_w[d] < 0.05`) or one still exceeds ~55% argmax share, add a **two‑sided, correctly‑scoped** regularizer — and prefer constraining the *soft count distribution's entropy* over a one‑sided mean cap:

```python
def atom_balance_loss(mix_weights, D, floor=0.05, cap=0.45, w_entropy=0.1):
    mean_w = mix_weights.mean(0)                     # (D,) soft share
    dead   = torch.relu(floor - mean_w).sum()        # punish atoms nobody uses
    hog    = torch.relu(mean_w - cap).sum()          # punish monopoly (loose cap)
    # Entropy of the mean-share distribution: pushes toward usage, not toward equality-by-force
    H_mean = -(mean_w * (mean_w + 1e-8).log()).sum() # maximize → add as -H
    return dead + hog - w_entropy * H_mean
```

Rationale vs the specs: the `cap` is deliberately *loose* (0.45) and paired with a `floor` so the loss fights **degeneracy** (dead/monopoly atoms), not the dataset's genuine imbalance. Use a **modest** weight (`W_BALANCE ≈ 1.0`, not 5.0) so it cannot override real signal. **Revert A3**: keep Phase‑2 temperature at the baseline `0.2` (or a mild `0.3→0.2` anneal). Softer temperatures favor blending into the dominant atom.

**Gate 1:** no dead atoms, no atom > 55% argmax share, mask reconstruction loss not materially worse than baseline.

### Step 2 — Restore the working Phase‑3; treat the regression loss as optional

Phase‑3 in `test_fixes.md` (polarity hinge + eps reg) was never the problem. Restore it:

```python
l_pol  = psi_polarity_loss(psi_full, dom_atom_full, origin_atom, margin=1.0)
l_eps  = (mvf.eps_raw ** 2).mean()
loss3  = W3_POLARITY * l_pol + W3_EPS_REG * l_eps
```

The decoupled train/viz smoothing (Fix 3a) is *harmless* and may keep a slightly sharper field — you may keep it, but understand it is **not** the lever. The regression loss (3c) may also be kept *in addition* to the hinge, but never as a *replacement*, because alone it is a near‑identity (§1.4).

Add a hard **ψ‑range assertion** so a future collapse is caught immediately instead of cascading:

```python
psi_spread = psi_full[dom_atom_full != origin_atom].mean() \
           - psi_full[dom_atom_full == origin_atom].mean()
assert psi_spread > 0.5, f"ψ collapsed (spread={psi_spread:.3f}); fix atoms before velocity/pseudotime."
```

**Gate 2:** `psi_spread > 0.5` (baseline was ≈ 0.28 *post‑smooth* but with a correct sign and good Dijkstra ρ; aim higher than baseline now that atoms are healthy).

### Step 3 — Fix the *original* velocity defect, on a healthy ψ, with flat‑region masking

Now (and only now) address the real `test_fixes.md` defect: inward convergence. Use the ψ‑**gradient direction** (correct in principle), but make it robust to residual flat regions and restore enough smoothing for coherence.

```python
def compute_psi_gradient_velocity_masked(psi, X_2d, k=30, reg=1e-3,
                                         grad_floor_pct=25):
    V = compute_psi_gradient_velocity(psi, X_2d, k=k, regularization=reg)  # WLS β per cell
    mag = np.linalg.norm(V, axis=1)
    # MASK flat/ambiguous cells: where |∇ψ| is in the noise floor, zero the arrow
    # (do NOT let noise become a streamline). Threshold = local noise scale.
    floor = np.percentile(mag, grad_floor_pct)
    V[mag < floor] = 0.0
    return V

V_emb = compute_psi_gradient_velocity_masked(psi_np, X_2d_target, k=30)

# Restore MILD diffusion for spatial coherence (2–3 hops, not 1, not 4).
# Build a row-stochastic T from the SAME 2D kNN and smooth the gradient field.
neighbor_idx_2d = build_2d_neighbor_index(X_2d_target, k=15)   # k=15: between baseline 20 and _2's 10
_, _, T_norm = compute_markov_velocity_2d(psi_np, X_2d_target, neighbor_idx_2d,
                                          n_hops=1, beta=DIFFUSION_BETA, eps=PSI_EPS)
V_smooth = V_emb.copy()
for _ in range(3):                                            # 3 hops ≈ coherence without washout
    V_smooth = 0.6 * V_smooth + 0.4 * (T_norm @ V_smooth)
```

Why this is geometrically correct *here*: where ψ rises from the origin pocket outward into the branches, `∇ψ` points **outward** by construction — the exact direction the inward‑convergent centroid field got wrong. The mask prevents flat‑plateau noise from becoming loops; the 3‑hop diffusion restores the spatial coherence that `N_DIFFUSION_HOPS=1` destroyed, without the over‑smoothing that produced the original convergence at 4 hops.

Keep `integration_direction='forward'` and `minlength≈0.6–0.8` in the streamplot. Optionally drop the `griddata` interpolation in favor of plotting quiver arrows at cell positions to avoid grid‑induced spurious circulation entirely.

**Gate 3:** Spearman(GRN entropy, `|V_smooth|`) flips from −0.076 to **positive** (target ≥ +0.15), and streamlines visibly **diverge** from the origin pocket into the branch arms (no closed orbits).

### Step 4 — Pseudotime: change nothing

The Dijkstra code is correct and is *self‑healing* once ψ is non‑flat (Steps 0–2). Re‑run it unchanged. Optionally widen roots `root_pct = 0.05 → 0.08` only if the stem pocket is under‑covered.

**Gate 4:** pseudotime is continuous (not binary); Spearman vs FACS ρ ≥ 0.55. (If ρ regresses below baseline 0.633, the atom fix in Step 0 is incomplete — return to Gate 0, do not patch Dijkstra.)

### Step 5 — Gene trajectories: keep the two genuinely good ideas, recompute on the healed pseudotime

The Round‑2 gene fixes are sound *as long as pseudotime is continuous*. Keep:
- **Corridor trajectory** (`build_lineage_trajectory`) instead of pooling all HSC/MPP — prevents cross‑lineage contamination.
- **Cell‑cycle regression** (`regress_out_cell_cycle`) — removes STMN1/MCM/PCNA confounds.
- **Kernel‑smoothed curves** (`smooth_gene_over_pseudotime`, bandwidth ≈ 0.05) instead of 50‑bin means — removes single‑cell spikes.
- **Within‑lineage pseudotime rescaling** to `[0,1]`.

But add the union‑ranking guard so ranking does not hinge on a single ordering:

```python
corrs_pt  = np.array([spearmanr(pt_lin,  Xcc[:,g])[0] for g in range(G)])
corrs_psi = np.array([spearmanr(psi_lin, Xcc[:,g])[0] for g in range(G)])  # ψ = GRN-state ordering
top = list(set(np.argsort(-corrs_pt)[:n_top]) | set(np.argsort(-corrs_psi)[:n_top]))
```

**Gate 5:** for ≥2 lineages, a lineage TF (CSF1R/CEBPA for CMP/GMP, GATA1/KLF1 for MEP, MPO for GMP) appears in the top‑5; ground‑truth recovery > 20%.

---

## 4. Sequencing, gates, and a "do‑not" list

**Order of operations (do not parallelize across gates):**

```
Step 0  Revert A1 init  ───────────────► Gate 0  (atoms lineage-shaped, ψ spread restored)
Step 1  Gentle 2-sided balance (opt.) ─► Gate 1  (no dead/monopoly atoms; mask loss OK)
Step 2  Restore Phase-3 hinge + assert ► Gate 2  (psi_spread > 0.5)
Step 3  ψ-grad arrows + mask + 3-hop ──► Gate 3  (entropy–|V| ρ > 0; streamlines diverge)
Step 4  Dijkstra UNCHANGED ────────────► Gate 4  (continuous pt; FACS ρ ≥ 0.55)
Step 5  Corridor + CC-regress + kernel ► Gate 5  (lineage TFs top-5; GT recovery > 20%)
```

**Do NOT (these are the moves that caused or masked the cascade):**
- ❌ Quantile‑bin a single scalar (discordance or anything else) to define atoms. Atoms must be multi‑dimensional, lineage‑distinct programs.
- ❌ Pass raw `gene_activity` against `[0,1]`‑normalized RNA — always scale‑match before subtracting.
- ❌ Force atom balance to `[N/D,…]`. Target *distinctness and non‑degeneracy*, not equal counts. The origin atom is legitimately small.
- ❌ Replace the polarity hinge with the regression loss alone (it is a near‑identity; §1.4).
- ❌ "Fix" pseudotime by editing the Dijkstra cost — the cost is correct; fix ψ upstream.
- ❌ Reduce diffusion to a single hop while ψ may still be locally flat — mask the flat cells and keep 2–3 hops.
- ❌ Use `vel_pseudotime` as a flow potential or Dijkstra cost (Round‑1 Fix 1c) — it is circular and was already (correctly) reverted; keep it reverted.

**Minimal‑change recovery (if you want the smallest possible diff):** Steps 0 + 2 + 4 alone should clear all three cascade symptoms (atom collapse → spread ψ → continuous pt → coherent‑enough arrows), because they simply *undo* the basis damage and *restore* the working Phase‑3. Steps 1, 3, 5 are quality improvements on top of a recovered baseline.

---

## 5. Expected outcome after the pivot

- Dominant‑atom distribution returns to a lineage‑shaped, non‑degenerate split (small origin pocket + three branch atoms), ψ spread ≥ 0.5–0.8.
- Pseudotime continuous again; FACS Spearman ρ back to ≈ 0.6 (≥ 0.55 gate).
- Velocity streamlines **diverge** from the origin pocket into GMP/MEP/MLP arms; entropy–|V| correlation flips positive. The original inward‑convergence defect is cured by using `∇ψ` *direction* (Step 3), which was the one genuinely correct idea in the specs — now applied on a substrate that can support it.
- Gene curves smooth and monotone; cell‑cycle confounds regressed out; ≥2 lineages recover canonical TFs in the top‑5 and exceed the 20% ground‑truth threshold.

The throughline: **`_2` failed by repairing the roof while removing the foundation.** Restore the foundation (atom basis) first; the roof repairs (gradient arrows, cell‑cycle regression, kernel curves) then work as the specs intended.
