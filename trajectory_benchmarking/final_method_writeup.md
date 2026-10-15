# COMET trajectory inference: current state, method, and next steps

*Status as of 2026-10-12. Source of truth: `trajectory_benchmarking/final_trajectory_testing.ipynb` (clean and self-contained; a fresh run takes about 20 minutes). Development history: `new_trajectory_experiments.ipynb` (§5–§7). Earlier write-up of the root rule: `root_finding_method.md`.*

---

## 1. Summary

The method is unsupervised trajectory inference for **unpaired scRNA + scATAC**:

1. **Origin.** The root is the most open-chromatin region of the state space. Picking it uses a *cluster-median* rule, aggregated over many clusterings (consensus over K).
2. **Pseudotime.** The harmonic **committor** q between that root and sinks at the far ends of the branches (a consensus over K, like the root).
3. **Velocity field.** The committor's gene-space direction, smoothed. On committor *plateaus* (where q is locally flat and its direction is undetermined), it is filled in by an **offset-free chromatin lead**. This lead is "what ATAC predicts for RNA beyond what RNA already shows", and it is constrained so that it cannot point back toward the origin.

How to describe it honestly:

- **Where the accuracy comes from.** The trajectory's accuracy comes from the chromatin-defined origin plus the committor.
- **What the chromatin lead does.** It is an **independent cross-modal corroboration**: its high-confidence arrows agree with the committor, and it beats shuffled ATAC. It is also a **local refinement** on plateaus.
- **What it does not do.** It does not change integrated pseudotime.

No COMET training is needed for the trajectory. The trained velocity head and GRN did not improve the trajectory: the best `LAMBDA_GRN` on hema was 0, and cos(trained, untrained `vel_head`) = 0.10–0.24. The velocity head has therefore been removed from the code; the GRN is fit separately and does not feed into the trajectory.

---

## 2. Data and notation

| symbol | meaning |
|---|---|
| $X \in \mathbb R^{N\times G}$ | log-normalised RNA (`b.X`) |
| $Z \in \mathbb R^{N\times 10}$ | Decipher latent (`b.Z_dec`): the state space for all graphs |
| $a_i \in \mathbb R^G$ | pseudo-bulk ATAC gene activity per RNA cell (unpaired; `utils/geometry.build_local_atac`) |
| $S$ | moment operator: row-normalised 15-NN average in $Z$, self included |
| $N_k(i)$ | the $k$ nearest neighbours of cell $i$ in $Z$ |
| $W^d, W$ | symmetric 15-NN graph in $Z$: edge lengths (geodesics) / Gaussian affinities $w_{ij} = \exp(-d_{ij}^2/2\sigma^2)$, $\sigma$ = median edge |
| $\tilde x$ | per-gene z-scored $X$ |
| $\mathrm{unit}(v) = v/\lVert v\rVert$ | per-cell normalisation |

**Pseudo-bulk ATAC (unpaired).**
1. KMeans the RNA cells into C clusters in $Z$.
2. For each cluster, average the 8 ATAC cells whose gene activity is most cosine-similar to the cluster's mean RNA.
3. Every cell in the cluster inherits that profile.
4. Apply per-gene min–max scaling, then a 15-NN smooth.

Paired data (ma2020) uses the same clusters, but step 2 is replaced by the mean of the cluster's own cells' paired ATAC (no matching).

The loader uses C = 40. The origin score uses C = 120 (finer resolution).

Datasets (validation references are used **only** for scoring):

| dataset | N | genes | validation reference | expected origin |
|---|---|---|---|---|
| hema | 1035 | 3016 | FACS pseudotime | HSC / MPP |
| aml89 | 5000 | 3198 | blast stage (immature 0 → blast3 4) | immature |
| paul15 | 2671 | 1850 | per-branch stage rank | 7MEP (9/10GMP) |

---

## 3. Method

### 3.1 Origin score

Let $a$ be the 120-cluster pseudo-bulk ATAC. Define

$$
H_i = -\sum_g p_{ig}\log p_{ig},\quad p_{ig} = \frac{\max(a_{ig},0)}{\sum_{g'}\max(a_{ig'},0)},
\qquad
\beta_i = \frac1G\sum_g \mathbb 1\!\left[a_{ig} > \operatorname{median}_j a_{jg}\right]
$$

$$
\phi_i = s_d\,\big[z(H)_i + z(\beta)_i\big]
$$

- $H_i$ is the chromatin entropy.
- $\beta_i$ is the breadth: the fraction of genes above their dataset median.
- $\phi_i$ is root-high.

$s_d = +1$ for hema and paul15 (origin = most open chromatin). **$s_d = -1$ for aml89** (origin = least open). This is an explicit modelling assumption, and it is the only per-dataset choice in the method.

### 3.2 Consensus cluster-median root

For each $K \in \mathcal K$ and seed $r \in \{0,\dots,9\}$:

$$
\ell^{(K,r)} = \mathrm{KMeans}_K(Z;\ r),\qquad
c^\star = \arg\max_c \operatorname{median}\{\phi_i : \ell_i = c\}
$$

$$
m = \arg\min_{i\in c^\star}\lVert Z_i - \bar Z_{c^\star}\rVert,\qquad
R^{(K,r)} = \text{the } 0.05N \text{ cells of } c^\star \text{ with the smallest geodesic distance } d_{W^d}(m,\cdot)
$$

The **consensus root** is the $0.05N$ cells with the largest selection frequency

$$
f_i = \frac{1}{|\mathcal K|\cdot 10}\sum_{K,r}\mathbb 1[i\in R^{(K,r)}],
$$

with ties broken by the 30-NN-smoothed $\phi$.

**The K range is derived, not tuned.** $\mathcal K = [3,\ 1/(2\rho_{root})] = [3, 10]$ with $\rho_{root} = 0.05$. The upper end means an average cluster holds at least two roots' worth of cells. The lower end separates root / intermediate / terminal.

*Why the median.* In aml89 (flipped), `immature` has the best median score, but the single most extreme cells are in `blast0`, and `immature` is a side tip off `blast0`. Argmax, quantile and compact rules therefore land on `blast0`. The progenitors are central in hema and paul15, so "prefer tips" fails there.

### 3.3 Sinks (consensus over K)

$$
\delta_i = \min_{r\in R} d_{W^d}(r,i),\qquad
\mathcal C = \{i\notin R:\ \phi_i < P_{40}(\phi),\ \delta_i \ge P_{75}(\delta)\}
$$

For each K in the root's range $[3,10]$ and 10 seeds: KMeans $\mathcal C$ into K groups in $Z$ and take the $\lfloor 0.06N/K\rfloor$ cells with the largest $\delta$ in each group. $T$ = the $\lfloor 0.06N\rfloor$ cells selected most often (ties by $\delta$).

*Earlier version:* a fixed 4 groups. A fixed K assumes a number of branches. aml89 has one (all 4 tips fell in blast3), and on Nestorowa the 4th tip became a spurious LT-HSC group. With the consensus, ρ(q, reference) is hema .729 (.728 with 4 tips), aml89 .956 (.962), ma2020 .908 (.906), Nestorowa .820 / stage .631 (.797 / .602). Data-driven alternatives that pick a single K did worse: persistence of δ peaks and graph components miss ma2020's IRS branch (.863), and eigengap / gap statistic / silhouette gave unstable K.

### 3.4 Pseudotime: harmonic committor

With $L = D - W$:

$$
(Lq)_i = 0\ \ (i\notin R\cup T),\qquad q|_R = 0,\qquad q|_T = 1 .
$$

$q\in[0,1]$ is the pseudotime: the probability that a random walk from cell $i$ reaches a terminal state before returning to the origin.

### 3.5 Committor direction

$$
u_i \propto \sum_{j\in N_{15}(i)} (q_j - q_i)\,(\tilde x_j - \tilde x_i),
\qquad
\bar u_i = \mathrm{unit}\Big(\tfrac1{30}\sum_{j\in N_{30}(i)} \hat u_j\Big).
$$

Plateau cells (with $u_i \approx 0$) are filled from valid neighbours (`committor_gene_direction`).

### 3.6 Chromatin lead (offset-free, fixed)

**Matched resolution.** RNA goes through the same pseudo-bulk pipeline as ATAC: the same 40 clusters, min–max, and 15-NN smooth. Then:

$$
A = S\,a,\quad R = S\,r,\quad \tilde A_g = z(A_g),\ \tilde R_g = z(R_g)\ \text{(standardised after smoothing)},\quad
\rho_g = \operatorname{corr}(\tilde A_g,\tilde R_g).
$$

**Offset-free residual.** This is ATAC's prediction of RNA beyond what the current RNA already explains:

$$
c_{ig} = \rho_g^+\,\big(\tilde A_{ig} - \rho_g^+ \tilde R_{ig}\big),\qquad \rho^+ = \max(\rho, 0).
$$

**Why not $\tilde A - \tilde R$.**
- With imperfect coupling, $\tilde A \approx \rho\tilde R + \varepsilon$, so $\tilde A - \tilde R \approx -(1-\rho)(\tilde R - \text{anchor}) + \varepsilon$. That is a *pull toward whatever the offsets are centred on*.
  - Global z-scoring pulls toward the dataset mean.
  - Root-centring reverses every arrow.
  - Sink-centring looks great, but a constant "toward the sinks" vector with no ATAC at all does even better.
- The repo's current `chrom_lead = S·z(ATAC) − S·z(RNA)` also z-scores *before* smoothing. RNA shrinks to SD ≈ 0.3 while ATAC stays at ≈ 0.95, so the "lead" is mostly the ATAC state (median cos 0.9).
- The residual is uncorrelated with $\tilde R$ for every gene, so it is invariant to any centring.

**Post-processing** (one neighbourhood size, $k=30$, throughout):

1. Tangent projection. With $\Delta_i$ the $k\times G$ matrix of neighbour displacements in per-cell $\tilde x$ (ridge $\lambda_r$ = 0.01·mean squared displacement):
   $$c_i \leftarrow \Delta_i^\top(\Delta_i\Delta_i^\top + \lambda_r I)^{-1}\Delta_i c_i$$
2. Forward cone (fixed lead). With $\hat o_i$ the gene-space direction of increasing geodesic distance from $R$ (built like $u$ from $\delta$; no committor):
   $$c_i \leftarrow c_i - \min(0,\ c_i\cdot\hat o_i)\,\hat o_i$$
   The lead cannot point back at the origin.
3. Local average: $c_i \leftarrow \frac1k\sum_{j\in N_k(i)} c_j$.
4. Coherence weight: $c_i \leftarrow \kappa_i^2 c_i$, where $\kappa_i = \big\lVert\frac1k\sum_{j\in N_k(i)}\hat c_j\big\rVert \in[0,1]$.

### 3.7 Fusion (final velocity field)

$$
\boxed{\ V_i = \mathrm{unit}\big[(1-w_i)\,\bar u_i + \lambda\, w_i\,\hat c_i\big],\qquad
w_i = g\,(1 - \alpha_i),\qquad
\alpha_i = \min\!\Big(1,\ \frac{\Delta q_i}{P_{25}(\Delta q)}\Big)\ }
$$

- $\Delta q_i = \max_{j\in N_{15}(i)} q_j - \min_{j\in N_{15}(i)} q_j$ is the committor confidence. $\alpha_i < 1$ only on the flattest quarter of cells (the plateaus).
- $\lambda = 2$.
- $g = \mathbb 1\big[\operatorname{median}_g \rho_g \ge 0.1\big]$ is the dataset QC gate: does ATAC predict RNA at all?

Outside plateaus, $V = \bar u$ exactly. Where $g = 0$ (aml89), $V = \bar u$ everywhere.

### 3.8 Hyperparameters

| parameter | value | status |
|---|---|---|
| $s_d$ (origin sign) | +1; −1 for aml89 | **modelling assumption, per dataset** |
| pseudo-bulk resolution for $\phi$ | 120 clusters, 8 ATAC cells, 15-NN | generic default |
| root size $\rho_{root}$ | 5 % | generic default |
| K range | $[3, 1/(2\rho_{root})] = [3,10]$ | derived from $\rho_{root}$ |
| sinks | 6 %, consensus over the root's K range | repo default; no branch count |
| graph k | 15 | repo default (`COMMIT_GRAPH_K`) |
| local neighbourhood | 30 | one value everywhere |
| plateau | bottom quartile of $\Delta q$ | generic; ablated |
| $\lambda$ | 2 | flat over 2–8 on hema; ablated |
| coupling gate | median $\rho \ge 0.1$ | QC rule |

---

## 4. Results (validation only)

### 4.1 Origin and pseudotime

| | hema | aml89 | paul15 |
|---|---|---|---|
| root composition | CLOUD 46 %, HSC 29 %, MPP 17 % | immature 100 % | 7MEP 72 %, 6Ery 22 % |
| root purity | .46 (mean FACS pt .31; HSC/MPP mean .35) | 1.00 | .75 |
| sinks | MEP / GMP / CLOUD | blast3 96 % | 1–2Ery, 14Mo, … |
| **ρ(q, reference)** | **.73** | **.93** | **.52** |

**K stability** (mean root purity over 10 seeds):
- With a single K, aml89 fails for K ≥ 11 and paul15 fails at K = 4–5.
- The consensus root is identical for every K range whose upper end is ≤ 15. It degrades only when K = 20 is included.
- Versus the previous K = 6 rule: aml89 and paul15 are unchanged; hema purity drops from .63 to .46 at the same FACS pseudotime and ρ.

### 4.2 Chromatin lead (`new_trajectory_experiments` §5f–5g)

| | median 2D cos(lead, committor) hema / aml89 / paul15 | top-10 % most colourful arrows agreeing with the committor | shuffled-ATAC null |
|---|---|---|---|
| repo `chrom_lead` | .55 / −.18 / .61 | 57 % / 18 % / 83 % | — |
| offset-free residual (tangent + kNN + coh²) | .73 / −.30 / .64 | **100 % / 54 % / 100 %** | .19 / −.32 / −.07 |
| + forward cone (fixed) | .77 / −.25 / .66 | 100 % / 56 % / 100 % | .31 / −.21 / .01 |

aml89's lead is indistinguishable from its null (median ρ = 0.04), hence the gate.

### 4.3 Fusion ablation (plateau cells = flattest quarter of q)

| setting | plateau: agree with committor's general direction (hema / aml89 / paul15) | plateau: cos with reference | all cells: cos(ref) |
|---|---|---|---|
| committor $u$ | .76 / .84 / .83 | .52 / .77 / −.56 | .94 / .87 / .61 |
| smoothed committor (no chromatin) | .81 / .91 / .89 | .78 / .83 / −.71 | .94 / .86 / .61 |
| **FINAL** | .81 / .91 / .83 | **.80** / .83 / **−.27** | .95 / .86 / .60 |
| FINAL with shuffled-ATAC lead | .75 / .91 / .73 | .64 / .83 / −.28 | .94 / .86 / .61 |
| λ = 1 | .80 / .91 / .90 | .81 / .83 / −.46 | .95 / .86 / .60 |
| agreement gate instead of coupling gate | .85 / .90 / .95 | .81 / .83 / −.72 | — |
| … same, shuffled-ATAC lead | .83 / .90 / .92 | .80 / .83 / −.70 | — |

FINAL reverses ≤ 0.3 % of confident committor arrows.

**Arrow-integrated pseudotime** (least-squares potential, `pseudotime_from_arrows`), ρ with the reference:

| | committor q | arrows: u | arrows: smoothed u | arrows: FINAL | arrows: lead alone |
|---|---|---|---|---|---|
| hema | .73 | .75 | .75 | .75 | .57 |
| aml89 | .93 | .93 | .93 | .93 | .48 |
| paul15 | .52 | .50 | .50 | .49 | .40 |

### 4.4 Negative results worth reporting

- **Anchoring the lead at the root is wrong.** The root has the *largest* residual (priming): 0.6–1.2× a typical cell's.
- **Sink anchoring gains are partly the anchor.** The shuffled-ATAC null reaches .61 / .49 agreement with the committor.
- **An agreement gate ("use the lead only where it agrees with the committor") is circular.** A shuffled-ATAC lead passes it almost as well.
- **Null-calibrating the lead's magnitude fails.** Real leads are always more coherent than shuffled ones, including on aml89, where they are uninformative.
- **Colour weighting never helps. λ = ∞ (replace) hurts.**

---

## 5. What to claim / not claim

**Claim:**
1. An unsupervised chromatin-defined origin, robust to the clustering scale (consensus over K), that places the root correctly on all three datasets.
2. A committor pseudotime from it: ρ = .73 / .93 / .52.
3. A diagnosis: naive ATAC − RNA leads (and centred variants) are dominated by a regression-to-the-mean artifact. Shown with nulls.
4. A corrected, offset-free chromatin lead that beats shuffled ATAC and independently corroborates the committor: its confident arrows agree 100 % on hema and paul15.
5. A principled fusion. The committor is used where it is determined; chromatin fills only where the committor is undetermined (plateaus); confident committor arrows are untouched.

**Do not claim:**
- That chromatin improves pseudotime accuracy (it does not change integrated pseudotime).
- That the fusion beats smoothing on every metric. On paul15 plateaus it moves arrows toward the reference but away from the committor's general direction.
- That the aml89 sign is inferred. It is an assumption.

---

## 6. Next steps

### 6.1 Push: implemented on branch `trajectory-final`

* **`comet/utils/trajectory.py`** (new) holds the whole formulation:
  `origin_score`, `consensus_root`, `consensus_sinks`, `matched_pseudobulk`, `residual_lead`, `tangent_project`,
  `forward_cone`, `chromatin_lead`, and `final_trajectory(b, Z, S_mom, sign)`.
  * `final_trajectory` returns `q`, the root / sink masks, `u`, `u_bar`, `lead`, `w` and `V`.
  * It reproduces `final_trajectory_testing.ipynb` exactly: identical roots and sinks, max |Δq| = 0, cos(V) = 1.000000 on all three datasets.
* **`comet/data/load.py`**: `Bundle.ga_atac_cells` keeps the raw per-cell gene activity (per ATAC cell if unpaired, per RNA cell for paired ma2020), so the pseudo-bulk can be rebuilt at other resolutions. `build_local_atac(..., paired=True)` averages each cluster's own paired ATAC instead of matching.
* **`comet/config.py`**: the `TRAJ_*` hyperparameters (§3.8), `ORIGIN_SIGN` (−1 for aml89) and `TRAIN_GRN`.
  * `OUT_DIR = "comet_out_trajectory_final"`, so old cached runs are not reused.
* **`comet/training/training_loop.run_comet`**:
  * The trajectory (root, sinks, q, V) comes from `final_trajectory`. `V_pred.npy` = V.
  * Also saved: `traj_u_bar.npy`, `traj_lead.npy`, `traj_w.npy`, `root_freq.npy`, `trajectory_mode.txt`.
  * `run_comet(..., train=False)` computes the trajectory only; otherwise `train_grn` then fits the GRN dictionary.
* **Removed** (no longer used by the method): the legacy top-10 % root (`committor_field`), the trained velocity head
  (`DictVelocityHead`), the disequilibrium / tilt orientation and the velocity, ATAC-direction and committor losses.
  The GRN dictionary, cell encoder, linear decoder and their losses are unchanged; the velocity loss used to backpropagate
  into the GRN during phase 2, and that coupling is gone with it.
* **`comet/train.py`**: `--trajectory-only` flag. Example: `python train.py --datasets hema aml89 paul15 --trajectory-only`.
  Usage, outputs and settings are documented in the top-level `README.md`.

### 6.2 Benchmark (same three datasets, same references; add held-out datasets)

**Pseudotime / root (ρ with the reference, root purity):**
- Repo default (top-10 % chromatin potential root → committor).
- Diffusion pseudotime (DPT), rooted by CytoTRACE and by the RNA-entropy maximum.
- Palantir (same roots), plus its fate probabilities.
- CellRank 2: CytoTRACE kernel + pseudotime kernel; absorption probabilities vs our committor.
- Slingshot / Monocle 3, given the same root (sanity check of the backbone).

**Velocity field (2D/gene cos with the reference direction; arrow-integrated pseudotime ρ; fraction of backward arrows):**
- scVelo (stochastic and dynamical), where spliced/unspliced counts exist.
- MultiVelo, which is chromatin-aware; it needs paired multiome, so use it on held-out paired data.
- Chromatin potential (Ma et al. 2020) as the closest prior art for ATAC → RNA leads. Show the regression-to-mean issue on unpaired data.

**Internal ablations** (already in the notebook): committor only, smoothed committor, shuffled-ATAC lead, free vs fixed lead, λ, plateau threshold, no gate.

**Held-out datasets (no retuning — freeze the method first):**
- `ma2020` (SHARE-seq skin, paired). The old rule's root landed on Medulla, not TAC; that came from NN-matching the paired ATAC. With the paired pseudo-bulk: root TAC-1 79 %, sinks Medulla / Hair Shaft / IRS, ρ(q, TAC-1-rooted DPT) = .91 (NN-matched: .35). Coupling .08, so the lead is gated off.
- `nestorowa` (full 1,654 HSPCs, rebuilt from GSE81682 + BEELINE; ATAC prior shared with paul15). Root LT-HSC 59 %, ρ(q, stage) = .60, ρ(q, BEELINE Slingshot) = .87 / .86 / .80 (E / GM / L).
- If time allows, a 10x multiome haematopoiesis dataset, where a paired chromatin lead can be validated directly.

### 6.3 Figures for the paper

1. Schematic of the method: origin → committor → plateau fill.
2. Origin consensus panel (`final_trajectory_testing/01_origin_consensus.png`): purity vs K, root frequency map, q.
3. Regression-to-mean diagnosis: anchored-lead quivers plus the constant-vector control (`new_trajectory_experiments/05f_*`).
4. "Colourful ⇒ correct" curves for the residual lead vs the repo lead vs the null (`05f_lead_precision_vs_coverage.png`).
5. Final field and where it differs from the committor (`final_trajectory_testing/02_final_trajectory_field.png`).
6. Benchmark table vs the baselines.

### 6.4 Open risks

- hema root purity .46 under the consensus rule (CLOUD at low FACS pseudotime). Report it with FACS pseudotime, not only purity.
- paul15's reference is label-derived and disagrees with the committor on sink-less branches.
- aml89 depends on the flipped assumption, and its chromatin does not support the flip.
- Everything was developed on these three datasets. The held-out evaluation in §6.2 is what makes the claims credible.

---

## 7. Files

| file | content |
|---|---|
| `trajectory_benchmarking/final_trajectory_testing.ipynb` | clean final pipeline, K-stability test, lead + nulls, fusion ablation, pseudotime check, decision |
| `trajectory_benchmarking/new_trajectory_experiments.ipynb` | development: origin sweeps (§5–5e), lead formulations (§5f), origin anchoring (§5g), combination sweeps (§6, §7) |
| `trajectory_benchmarking/root_finding_method.md` | write-up of the cluster-median root (single-K version) |
| `comet_out_beta_sweep/final_trajectory_testing/` | final figures |
| `comet_out_beta_sweep/new_trajectory_experiments/` | development figures (`05e_*`, `05f_*`, `05g_*`, `07_*`) |
| `comet/utils/trajectory.py` | the implemented method (`final_trajectory`); how to run it: top-level `README.md` |
| `comet/data/load.py` | loaders, including the aml89 fix (peak overlap mapping, patient-only ATAC) |
