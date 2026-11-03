"""Supplementary figure: forward transport of the committor-only field, committor + chromatin lead, the GRN tilt with
scrambled networks, and COMET (GRN tilt with the learned network; five training seeds), per dataset (panels) and graph
space (rows). Redraws the figure from the precomputed values in data/grn_tilt/; writes figs_final/
supp_grn_tilt_significance.{pdf,png,svg} and data/grn_tilt/summary.csv (with Holm-adjusted p over the 10 tests)."""
import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import norm, skew

HERE = os.path.dirname(os.path.abspath(__file__))
SIG, FIG = f"{HERE}/data/grn_tilt", f"{HERE}/figs_final"
DATASETS = {"paul15": "Paul et al. 2015", "hema": "Velten et al. 2017", "nestorowa": "Nestorowa et al. 2016",
            "ma2020": "Ma et al. 2020", "aml89": "Nazaret et al. 2025"}   # order and names as in the overview / CytoTRACE figures
SPACES = ["Decipher latent", "gene PCA"]
COMET_C, NULL_C, INK, MUTED = "#C9027A", "#BDBDBD", "#1F1F1F", "#6B6B6B"
mpl.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def stars(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def null_values(ds, sp): return pd.read_csv(f"{SIG}/{ds}_{sp.replace(' ', '_')}_null_values.csv").transport.values


S = pd.concat([pd.read_csv(f"{SIG}/{ds}.csv") for ds in DATASETS])
S["perm_p_holm"] = holm(S.perm_p.values)
# the permutation p is floored at 1/(1+N) = 0.0101 for N = 98 scrambles, which Holm over 10 tests cannot get below
# 0.05; the scrambled values are tightly and symmetrically distributed, so we also test against their normal fit
S["z_vs_null"] = (S.full - S.null_mean) / S.null_sd
S["p_z"] = norm.sf(S.z_vs_null); S["p_z_holm"] = holm(S.p_z.values)
S["null_skew"] = [skew(null_values(d, sp)) for d, sp in zip(S.dataset, S.space)]
S["exceeds_all_nulls"] = S.full > S.null_max
S.to_csv(f"{SIG}/summary.csv", index=False)

fig, axes = plt.subplots(2, len(DATASETS), figsize=(7.2, 3.3), squeeze=False)
X = {"committor only": 0, "V0": 1, "scrambled GRN tilt": 2, "COMET": 3}
LEAD_OFF = {"aml89", "ma2020"}
rng = np.random.default_rng(0)
for r, sp in enumerate(SPACES):
    for c, ds in enumerate(DATASETS):
        ax = axes[r, c]; row = S[(S.dataset == ds) & (S.space == sp)].iloc[0]
        nv = null_values(ds, sp)
        ax.scatter(X["scrambled GRN tilt"] + rng.uniform(-0.16, 0.16, len(nv)), nv, s=4, color=NULL_C, lw=0, zorder=2)
        # no per-marker CIs: every comparison is paired on the same cells (paired CIs in summary.csv / caption)
        ax.plot(X["committor only"], row.committor_only, "o", ms=4.5, mfc="white", mec=INK, mew=0.9, zorder=3)
        ax.plot(X["V0"], row.V0, "s", ms=4.2, mfc="white", mec=MUTED, mew=0.9, zorder=3)
        if ds in LEAD_OFF:
            ax.text(X["V0"], row.V0, "lead\noff", ha="center", va="bottom", fontsize=4.6, color=MUTED,
                    transform=mpl.transforms.offset_copy(ax.transData, fig=fig, y=5, units="points"))
        seedf = f"{SIG}/{ds}_seeds.csv"; sv = np.array([row.full])
        if os.path.exists(seedf):
            sd = pd.read_csv(seedf); sv = sd[f"transport_{sp}"].values
            ax.plot(X["COMET"] + rng.uniform(-0.12, 0.12, len(sv)), sv, "o", ms=5, color=COMET_C, mec="white", mew=0.5, ls="", zorder=4)
        else:
            ax.plot(X["COMET"], row.full, "o", ms=5, color=COMET_C, mec="white", mew=0.5, zorder=4)
        top = max(sv.max(), row.full, nv.max(), row.committor_only, row.V0); bot = min(row.committor_only, row.V0, nv.min(), sv.min())
        pad = 0.08 * (top - bot)
        ax.text(X["COMET"], top + pad, stars(row.p_z_holm), ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_ylim(bot - pad, top + 3.2 * pad); ax.set_xlim(-0.5, 3.5)
        ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4)); ax.tick_params(axis="y", labelsize=5.6)
        ax.grid(axis="y", color="#ECECEC", lw=0.4, zorder=0)
        if r == 0:
            ax.set_title(DATASETS[ds], fontsize=6.6, fontweight="bold", color=INK, pad=4)
            ax.text(-0.30, 1.13, "abcdefgh"[c], transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
        if c == 0: ax.set_ylabel(f"forward transport\n({sp} graph)", fontsize=6.3, color=INK)
fig.tight_layout(w_pad=0.9, h_pad=0.8, rect=(0, 0, 1, 0.90))
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=INK, mew=0.9, label="committor only"),
                    Line2D([], [], ls="", marker="s", ms=4.2, mfc="white", mec=MUTED, mew=0.9, label="committor + chromatin lead ($V^0$)"),
                    Line2D([], [], ls="", marker="o", ms=3, color=NULL_C, label="committor + lead + GRN tilt, scrambled network"),
                    Line2D([], [], ls="", marker="o", ms=5, color=COMET_C, label="COMET (committor + lead + GRN tilt, learned network)")],
           loc="upper center", ncol=2, frameon=False, fontsize=6.0, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.6)
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"{FIG}/supp_grn_tilt_significance.{ext}", dpi=300)
print(S[["dataset", "space", "committor_only", "null_mean", "null_max", "full", "exceeds_all_nulls", "perm_p", "z_vs_null", "p_z_holm", "null_skew",
         "diff_vs_committor", "diff_vs_committor_lo", "diff_vs_committor_hi", "diff_vs_null", "diff_vs_null_lo",
         "diff_vs_null_hi", "R2_real", "R2_null_mean"]].round(4).to_string())
