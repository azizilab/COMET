# Committor origin: the cluster-median root selector

`new_trajectory_experiments.ipynb` §5e. Method label: **`ATAC120 | cluster K6 → medoid 5% | spread sinks`** (`BEST_5E`).
From §5e onward, this is the notebook default (`BEST_ORIGIN = BEST_5E`).

## 1. What problem this solves

COMET's committor field needs a root set R (where differentiation starts, q = 0) and a sink set T (terminal states, q = 1). The data contain no markers or labels for these, so both have to come from the data. The repo default (`utils/committor.py:committor_field`) takes the root as the top 10 % of a chromatin "potential" score. On the three benchmark datasets this puts the root in the wrong population for at least one dataset, whatever score is used (§5–5d):

| dataset | where the root should be (validation only) | what went wrong before |
|---|---|---|
| hema | HSC / MPP, low FACS pseudotime | root partly in CLOUD, slightly too high (mean FACS pt 0.36) |
| aml89 | `immature` | root always on `blast0` (purity 0.00), even with the sign flipped |
| paul15 | 7MEP (/ 9–10 GMP) | root spread into 6Ery etc. (purity ≈ 0.46) |

The new rule puts the root in the right population on all three datasets, with **one definition** shared by all of them. The only per-dataset setting is the sign of the chromatin assumption.

## 2. Data used

All of these are already produced by the loader / `setup_analytic`. No labels, pseudotime or markers are used.

* `X` (`b.X`): log-normalised RNA, N cells × G genes (hema 1035 × 3016, aml89 5000 × 3198, paul15 2671 × 1850).
* Raw per-ATAC-cell gene activity and the Decipher latent `Z_dec`, captured during loading (`RAW_ATAC`).
* **State space** `Zst` = Decipher latent z (10-dim, `config.STATE_SPACE = "decipher"` for all three). The committor graph, KMeans and the root/sink geometry all live here.
* Committor graphs on `Zst` with k = `COMMIT_GRAPH_K` = 15 (symmetrised kNN): a distance graph W^d (edge length = Euclidean distance; used for geodesics) and a Gaussian affinity graph W (w_ij = exp(−d_ij² / 2σ²), σ = median edge length; used for the harmonic solve).

Validation only (never inside the method): hema FACS pseudotime, aml89 blast stage (immature 0 → blast3 4), paul15 per-branch stage rank, and the cell-type labels for origin purity.

## 3. The method

### 3.1 Chromatin score at finer resolution ("ATAC120")

The datasets are unpaired, so each RNA cell gets a pseudo-bulk ATAC profile (`utils/geometry.build_local_atac`):

1. KMeans RNA cells into C clusters in `Z_dec`.
2. For each cluster c, take the k_atac = 8 ATAC cells whose gene activity is most cosine-similar to the cluster's mean RNA, and average them.
3. Every cell in the cluster inherits that profile.
4. Apply a per-gene min-max scaling, then a 15-NN smooth in `Z_dec`.

The repo uses **C = 40**, so the score is essentially piecewise constant over 40 blocks. ATAC120 uses **C = 120** (`ga_local(n, 120, 8, 15)`), so the score has three times the spatial resolution. Call the result a_i ∈ ℝ^G.

The score is the same formula as `chromatin_potential`, applied to that matrix:

$$
H_i = -\sum_g p_{ig}\log p_{ig},\quad p_{ig} = \frac{\max(a_{ig},0)}{\sum_{g'}\max(a_{ig'},0)}
\qquad
\beta_i = \frac1G\sum_g \mathbb 1\big[a_{ig} > \mathrm{median}_j\, a_{jg}\big]
$$

$$
\phi_i = s_d\,\big[z(H)_i + z(\beta)_i\big],\qquad z(v) = (v-\bar v)/\mathrm{sd}(v)
$$

- H = chromatin entropy (how evenly accessibility is spread over genes).
- β = breadth (fraction of genes above their dataset median).
- φ is root-high.
- **s_d is the only per-dataset choice:** s = +1 for hema and paul15 (origin = most open / highest entropy), **s = −1 for aml89**. That is the "flipped" assumption: in this leukaemia, the least open, lowest-entropy region is taken as the origin, and the committor is built from there.

### 3.2 Stage 1: pick a cluster by its *median* score

$$
\ell = \mathrm{KMeans}_{K=6}(\mathrm{Zst})\quad(\texttt{n\_init}=10,\ \texttt{random\_state}=0),
\qquad
c^\* = \arg\max_{c}\ \mathrm{median}\{\phi_i : \ell_i = c\}
$$

### 3.3 Stage 2: a compact root around the cluster medoid

$$
m = \arg\min_{i:\ \ell_i=c^\*}\ \lVert \mathrm{Zst}_i - \overline{\mathrm{Zst}}_{c^\*}\rVert^2,
\qquad
R = \text{the } \lfloor 0.05N \rceil \text{ cells of } c^\* \text{ with the smallest geodesic distance } d_{W^d}(m,\cdot)
$$

This gives 52 / 250 / 134 root cells for hema / aml89 / paul15. The root is one contiguous patch at the centre of the chosen cluster.

### 3.4 Sinks: spread over branch tips (`sinks_spread`)

$$
\delta_i = \min_{r\in R} d_{W^d}(r,i),\qquad
\mathcal C = \{\, i\notin R:\ \phi_i < P_{40}(\phi),\ \delta_i \ge P_{75}(\delta)\,\}
$$

- KMeans the candidates 𝒞 into 4 groups in `Zst`.
- In each group, take the ⌊0.06N/4⌋ cells with the largest δ.
- Together they form T (`COMMIT_SINK_FRAC` = 0.06). If |𝒞| < 16, fall back to the default sink rule.

Every branch tip gets sinks. The default rule (the most graph-eccentric low-φ cells) can spend all of them on one far branch.

### 3.5 Committor and its gene-space direction (unchanged)

The committor is the harmonic solve on W, with the graph Laplacian L = D − W:

$$
(Lq)_i = 0\ \ (i\notin R\cup T),\qquad q|_R = 0,\qquad q|_T = 1
$$

The gene-space forward direction u (`committor_gene_direction`) is a unit vector, with plateau cells filled from valid neighbours:

$$
u_i \propto \sum_{j\in N_{15}(i)} (q_j-q_i)\,\big(\tilde x_j-\tilde x_i\big)
$$

Here $\tilde x$ is the per-gene z-scored X.

## 4. What changed relative to before

| | repo default (`committor_field`) | §5 "compact" (previous best) | **§5e cluster-median (new)** |
|---|---|---|---|
| ATAC resolution | 40 pseudo-bulk clusters | 40 or 120 | **120** |
| sign | `ENTROPY_ROOT_TRUST` (−1 aml89 in config) | same | `FLIP`: −1 aml89 only |
| root selector | top 10 % of φ (can be scattered) | seed = argmax of 30-NN-smoothed φ; 5 % nearest cells | **cluster with highest median φ (K = 6); 5 % nearest its medoid, inside it** |
| sinks | eccentric low-φ cells | default or spread | **spread** |
| committor / u | harmonic / kNN | same | same |

The key change is **stage 1: a cluster's median instead of extreme cells**. On aml89 (flipped), `immature` has the highest *median* score of any population (smoothed percentile 0.71–0.84). But the single most extreme cells are in `blast0` (or `blast3`, for entropy only). `immature` is a small side branch off `blast0`: graph distance 4 from `blast0`, 14 from `blast3`, at a graph tip (eccentricity percentile 0.91). Anything driven by the argmax or a top quantile (quantile, compact) therefore lands on `blast0`.

Geometric fixes do not help either. "Prefer tips" and "farthest from the sinks" both fail, because the progenitors in the other two datasets are central (eccentricity percentile: paul15 7MEP 0.05, hema HSC 0.26). The median over a coarse cluster ignores scattered extreme cells and picks the population that is *consistently* most root-like.

Stage 2 (medoid, not argmax) keeps the root in the cluster's interior. It also stops the root from following extreme cells to the cluster's edge, which is where the old hema root drifted up into CLOUD.

## 5. Results (validation only)

| dataset | origin purity | root composition | ρ(q, reference) | 2D cos(u, reference) |
|---|---|---|---|---|
| hema | 0.63 (was 0.46) | MPP 35 %, HSC 29 %, CLOUD 29 %; root mean FACS pt **0.27** (was 0.36; the HSC/MPP mean is 0.35) | 0.73 | 0.92 |
| aml89 | **1.00** (was 0.00) | immature 100 % | 0.93 | 0.87 |
| paul15 | 0.75 (was 0.46) | 7MEP 72 %, 6Ery 22 % | 0.52 | 0.61 |

Sinks: hema MEP / GMP / CLOUD; aml89 blast3 (96 %); paul15 1–2Ery, 14–15Mo, 16Neu.

- Runner-up: `cluster K10 → far 2%`. It takes the seed in the chosen cluster farthest from provisional sinks and uses a 2 % root. Purity is 0.62 / 0.85 / **0.96**, and the hema root is more HSC-heavy.
- Using entropy instead of ATAC120 gives nearly identical results.

## 6. Robustness and caveats

* **K must be in a band.** Over K = 5–12 × 10 KMeans seeds:
  - aml89 is correct for every K = 5–10 and every seed. It fails for K ≥ 11, where `blast0` and `immature` start sharing clusters.
  - paul15 fails at K = 5: the progenitor cluster absorbs too much Ery.
  - K = 7 is seed-dependent.
  - Stable settings: K = 6 medoid 5 % (worst seed min-purity 0.60) and K = 8–10 far 2 % (worst 0.48–0.52).

  So the rule needs clusters coarse enough to keep the immature branch whole, but fine enough to separate 7MEP from 6Ery. It is not K-free.
* The aml89 sign is an **assumption, not something inferred**: the origin at the *lowest* chromatin entropy. §5f found that the chromatin lead does not support it. The ATAC residual and the ATAC state point against the flipped committor in aml89, and ATAC↔RNA coupling is near zero there.
* hema's root still contains 29 % CLOUD. CLOUD sits just above HSC/MPP in FACS pseudotime (0.41).
* The validation references are weak for paul15 (label-based stage ranks), so ρ there is a coarse check.

## 7. Where it lives in code

`new_trajectory_experiments.ipynb`:

- §5e cell: `make_cluster_selector`, `sinks_spread`, `committor_from_root`, `SCORES_5E["ATAC120"]`, `FLIP`.
- Robustness table: `ROBUST_5E`.
- `BEST_ORIGIN = BEST_5E` is set in the cell after the §5e results.

The repo pipeline now uses the consensus version of this rule (K ∈ [3, 10] × 10 seeds instead of K = 6):
`comet/utils/trajectory.py:consensus_root`, called by `final_trajectory`. See `final_method_writeup.md` §3.2.
