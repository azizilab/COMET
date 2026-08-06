"""
Mask-conditional GRN evaluation — the corrected counterpart to run_bleval.py.

WHY THIS EXISTS
---------------
run_bleval.py stages a ranked edge list for BEELINE's BLEvaluator, which scores
it against the *full* gene x TF pair space. But every GRN this model can emit is
masked: A_dict = tanh(W_raw) * M and A_base = tanh(W_base) * M, so the support is
always exactly M — the same ~808k motif edges, whatever the weights are. BEELINE
therefore spends almost all of its discriminative power separating in-mask pairs
from the ~1.6M out-of-mask pairs it never sees, which is a property of the HOMER
motif prior and is IDENTICAL across every run. Measured on the mHSC-E setup,
uniform random weights sprinkled on the mask scored AUROC 0.750 / EPr 1.19,
matching or beating a fully trained model — the metric simply cannot resolve the
model.

This script holds the mask fixed and scores only what the model actually
controls: the ORDERING of edges within the mask.

  * candidate set  = in-mask (TF, gene) pairs inside the panel universe
  * ground truth   = reference edges that fall inside that candidate set
                     (an edge the mask cannot express is not a miss the weights
                     could ever have fixed, so it is excluded rather than
                     counted as a false negative)
  * random control = uniform random scores on the same candidate set, N seeds,
                     reported as mean +/- sd so "better than chance" is a
                     statistical claim and not a vibe

Everything is computed directly (sklearn) rather than via BLEvaluator, because
BLEvaluator has no notion of a restricted candidate set. AUROC / average
precision / early-precision-ratio follow BEELINE's definitions: EPr is the
precision among the top-k edges where k = |ground truth|, divided by prevalence.

NORMALIZATION (--normalize)
---------------------------
Raw |A| carries two nuisance scales that are anti-correlated with truth:

  1. init_atoms_from_soft_targets divides each TF column by the max |LFC| over
     that TF's own motif targets, so a TF with more targets is scaled DOWN.
  2. predict_rna has no intercept and expression is non-negative, so a highly
     expressed TF needs SMALL weights to reach a given target level.

Both push weight away from well-expressed, many-target TFs — which are exactly
the hubs of the ChIP-seq references. --normalize tf-z removes the per-TF scale
(z-score each column within its own mask support) and exposes the edge-level
signal underneath; tf-gene-z removes the per-gene scale as well. Compare all
three: a method whose score only survives at 'none' is reporting an artifact.

USAGE
-----
  python scripts/run_bleval2.py --run /home/elicanter/<output_dir>
  python scripts/run_bleval2.py --run <dir> --panel 1000 --normalize tf-z
  python scripts/run_bleval2.py --run <dir> --all-normalizations   # side by side

  # add --grn learned|base|mixture (default learned)
  # add --diagnostics to print the nuisance-correlation table
Outputs: <run>/bleval2/metrics.csv  and  <run>/bleval2/figures/<label>_summary.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

NET_DIR = "/home/elicanter/datasets/_beeline_raw/networks/Networks/mouse"
NETWORKS = {
    "mHSC-ChIP-seq": "mHSC-ChIP-seq-network.csv",
    "Non-Specific-ChIP": "Non-Specific-ChIP-seq-network.csv",
    "STRING": "STRING-network.csv",
}
DEFAULT_GENE_ORDERING = (
    "/home/elicanter/datasets/_beeline_raw/BEELINE-data/"
    "inputs/scRNA-Seq/mHSC-E/GeneOrdering.csv"
)
NORMALIZATIONS = ("none", "tf-z", "tf-gene-z")


# ---------------------------------------------------------------- loading ----
def load_grn(run_dir, which):
    """Returns (genes, tfs, |W|) with names upper-cased, matching run_bleval.py."""
    A_genes = np.load(f"{run_dir}/A_genes.npy", allow_pickle=True).astype(str)
    A_tfs = np.load(f"{run_dir}/A_tfs.npy", allow_pickle=True).astype(str)
    if which == "learned":
        W = np.load(f"{run_dir}/A_learned.npy")
    elif which == "base":
        W = np.load(f"{run_dir}/A_base.npy")
    elif which == "mixture":
        A_dict = np.load(f"{run_dir}/A_dict.npy")
        w = np.load(f"{run_dir}/mix_weights.npy").mean(0)
        W = np.tensordot(w, A_dict, axes=([0], [0]))
    else:
        raise ValueError(which)
    assert W.shape == (len(A_genes), len(A_tfs))
    genes = np.array([s.upper() for s in A_genes])
    tfs = np.array([s.upper() for s in A_tfs])
    return genes, tfs, np.abs(W).astype(np.float64)


def load_mask(run_dir, absW, mask_path):
    """Mask support. Defaults to the GRN's own nonzeros, which for this model is
    exactly M (tanh() is never exactly 0 off the mask). Pass --mask explicitly if
    the saved GRN has been pruned, otherwise the pruning gets silently scored as
    if it were part of the prior."""
    if mask_path:
        M = np.load(mask_path)
        if M.shape != absW.shape:
            sys.exit(f"--mask shape {M.shape} != GRN shape {absW.shape}")
        return M != 0
    return absW != 0


def make_universe(genes, tfs, panel_n, gene_ordering):
    """Gene universe to score over — identical convention to run_bleval.py."""
    genes_u, tfs_u = set(genes), set(tfs)
    if not panel_n:
        return genes_u | tfs_u, "full universe"
    go = pd.read_csv(gene_ordering, index_col=0)
    var_col = "Variance" if "Variance" in go.columns else go.columns[-1]
    ranked = [str(x).upper() for x in go.sort_values(var_col, ascending=False).index]
    top = [x for x in ranked if x in genes_u][:panel_n]
    return set(top) | tfs_u, f"TFs + top {panel_n} HVGs"


# ---------------------------------------------------------- normalization ----
def _z_within(vals):
    return (vals - vals.mean()) / (vals.std() + 1e-9)


def normalize_scores(absW, mask, how):
    """Strip nuisance scales. Each z-score is taken WITHIN a row/column's own mask
    support, so genes and TFs with few candidates are not compared against zeros
    they never had a chance to fill."""
    if how == "none":
        return absW
    out = np.zeros_like(absW)
    for t in range(absW.shape[1]):                     # per-TF (column)
        m = mask[:, t]
        if m.sum() > 2:
            out[m, t] = _z_within(absW[m, t])
    if how == "tf-z":
        return out
    out2 = np.zeros_like(out)
    for g in range(out.shape[0]):                      # then per-gene (row)
        m = mask[g]
        if m.sum() > 2:
            out2[g, m] = _z_within(out[g, m])
    return out2


# ----------------------------------------------------------------- metrics ---
def early_precision_ratio(y, s, rng):
    """Precision among the top-k, k = |GT|, over prevalence. Ties are broken
    randomly — otherwise a near-constant score (e.g. the all-ties mask itself)
    gets ranked by array order and reports a meaningless number."""
    k = int(y.sum())
    order = np.lexsort((rng.random(len(s)), -s))[:k]
    return y[order].mean() / y.mean()


def score(y, s, rng):
    return {
        "AUROC": roc_auc_score(y, s),
        "AP": average_precision_score(y, s),
        "AP_ratio": average_precision_score(y, s) / y.mean(),
        "EPr": early_precision_ratio(y, s, rng),
    }


def random_control(y, n_pairs, n_seeds):
    runs = [score(y, np.random.default_rng(1000 + i).random(n_pairs),
                  np.random.default_rng(2000 + i)) for i in range(n_seeds)]
    keys = runs[0].keys()
    return ({k: float(np.mean([r[k] for r in runs])) for k in keys},
            {k: float(np.std([r[k] for r in runs])) for k in keys})


# ------------------------------------------------------------- diagnostics ---
def diagnostics(absW, mask, gidx, tidx, y, genes, tfs):
    """Is the ranking driven by edge identity, or by a per-TF / per-gene scale?
    A per-TF or per-gene score that beats the real thing means the model is
    ranking hubs, not regulatory relationships."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    n_t = np.maximum(mask.sum(0), 1)
    n_g = np.maximum(mask.sum(1), 1)
    tf_mean = absW.sum(0) / n_t
    gene_mean = absW.sum(1) / n_g

    rows = []
    for name, s in [
        ("full |A|", absW[gidx, tidx]),
        ("per-TF mean only", np.broadcast_to(tf_mean[None, :], absW.shape)[gidx, tidx]),
        ("per-gene mean only", np.broadcast_to(gene_mean[:, None], absW.shape)[gidx, tidx]),
    ]:
        rows.append((name, score(y, s, rng)["AUROC"]))

    # does weight track the reference's own hub structure, and in which direction?
    outdeg = np.zeros(len(tfs))
    for t_i, t in enumerate(tfs):
        outdeg[t_i] = y[tfs[tidx] == t].sum()
    corr_deg = spearmanr(tf_mean, outdeg).statistic
    corr_cand = spearmanr(tf_mean, mask.sum(0).astype(float)).statistic
    return rows, corr_deg, corr_cand


# ----------------------------------------------------------------- figure ----
def make_summary_figure(df, label, panel_tag, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nets = df["network"].tolist()
    x = np.arange(len(nets)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(3.0 + 1.8 * len(nets), 4.2))
    for ax, (col, rnd_col, chance, name) in zip(axes, [
            ("AUROC", "AUROC_random", 0.5, "AUROC"),
            ("EPr", "EPr_random", 1.0, "Early-precision ratio")]):
        ax.bar(x - w / 2, df[col], w, label=label, color="#2a9d8f")
        ax.bar(x + w / 2, df[rnd_col], w, label="random on mask", color="#adb5bd")
        ax.axhline(chance, ls="--", lw=1, color="0.35")
        for i, v in enumerate(df[col]):
            ax.text(i - w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(nets, rotation=20, ha="right")
        ax.set_title(name); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("within-mask score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{label} — mask-conditional  ({panel_tag})")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    fig.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description="Mask-conditional GRN evaluation.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--grn", default="learned", choices=["learned", "base", "mixture"])
    ap.add_argument("--label", default="COSMOS")
    ap.add_argument("--panel", type=int, default=0,
                    help="0=full universe; 500/1000=all TFs + top-N varying genes")
    ap.add_argument("--normalize", default="none", choices=NORMALIZATIONS,
                    help="strip per-TF / per-gene nuisance scale before ranking")
    ap.add_argument("--all-normalizations", action="store_true",
                    help="report every --normalize setting side by side")
    ap.add_argument("--mask", default=None,
                    help="explicit motif mask .npy; default = the GRN's nonzeros")
    ap.add_argument("--assayed-tfs-only", action="store_true",
                    help="drop TFs with zero reference edges (never assayed). "
                         "Strongly recommended: without it the score is dominated "
                         "by which TFs the reference happens to cover.")
    ap.add_argument("--seeds", type=int, default=20, help="random-control repeats")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--gene-ordering", default=DEFAULT_GENE_ORDERING)
    ap.add_argument("--net-dir", default=NET_DIR)
    ap.add_argument("--workdir", default=None, help="default <run>/bleval2")
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir or os.path.join(args.run, "bleval2"))
    os.makedirs(workdir, exist_ok=True)

    genes, tfs, absW = load_grn(args.run, args.grn)
    mask = load_mask(args.run, absW, args.mask)
    universe, panel_tag = make_universe(genes, tfs, args.panel, args.gene_ordering)

    # candidate set: in-mask, inside the panel, no self-edges
    gidx0, tidx0 = np.nonzero(mask)
    keep = ((genes[gidx0] != tfs[tidx0])
            & np.isin(genes[gidx0], list(universe))
            & np.isin(tfs[tidx0], list(universe)))
    gidx0, tidx0 = gidx0[keep], tidx0[keep]

    print(f"{args.label} GRN '{args.grn}': {len(genes)} genes x {len(tfs)} TFs")
    print(f"panel = {panel_tag}; candidate set = {len(gidx0)} in-mask pairs "
          f"({mask.mean():.1%} mask density)")
    print("scoring the ORDERING within the mask — the mask itself is held fixed")
    if args.assayed_tfs_only:
        print("restricted to assayed TFs — see --assayed-tfs-only")
    print()

    modes = list(NORMALIZATIONS) if args.all_normalizations else [args.normalize]
    scored = {m: normalize_scores(absW, mask, m) for m in modes}

    rows = []
    for net, fn in NETWORKS.items():
        ref = pd.read_csv(os.path.join(args.net_dir, fn))
        edges = {(str(a).upper(), str(b).upper())
                 for a, b in zip(ref["Gene1"], ref["Gene2"])}
        y_all = np.fromiter(((tfs[t], genes[g]) in edges
                             for g, t in zip(gidx0, tidx0)), bool, len(gidx0))
        gidx, tidx, y = gidx0, tidx0, y_all
        if args.assayed_tfs_only:
            # A TF with zero reference edges was almost certainly never assayed,
            # not "assayed and found to regulate nothing". Keeping its whole
            # column makes the task "guess which TFs were assayed", which TF
            # expression level answers with no GRN at all -- on mHSC-ChIP the
            # top 10% of TFs hold 99.9% of the edges, so the unrestricted AUROC
            # is a TF-ranking score wearing an edge-ranking costume.
            assayed = np.zeros(len(tfs), bool)
            assayed[np.unique(tidx0[y_all])] = True
            sel = assayed[tidx0]
            gidx, tidx, y = gidx0[sel], tidx0[sel], y_all[sel]
            print(f"  [{net}] {assayed.sum()}/{len(tfs)} TFs have reference edges; "
                  f"candidate set {len(y)} pairs")
        if y.sum() < 10:
            print(f"{net}: only {y.sum()} reference edges inside the mask, skipping")
            continue
        rmean, rsd = random_control(y, len(gidx), args.seeds)
        print(f"=== {net}: {y.sum()} reference edges inside mask "
              f"(prevalence {y.mean():.4f}) ===")
        print(f"{'normalize':>14s} {'AUROC':>8s} {'AP':>9s} {'APratio':>8s} "
              f"{'EPr':>7s} {'AUROC z':>9s}")
        for m in modes:
            sc = score(y, scored[m][gidx, tidx], np.random.default_rng(7))
            z = (sc["AUROC"] - rmean["AUROC"]) / (rsd["AUROC"] + 1e-12)
            print(f"{m:>14s} {sc['AUROC']:8.4f} {sc['AP']:9.5f} "
                  f"{sc['AP_ratio']:8.3f} {sc['EPr']:7.3f} {z:+9.1f}")
            rows.append(dict(network=net, normalize=m, n_gt=int(y.sum()),
                             prevalence=float(y.mean()), **sc,
                             AUROC_random=rmean["AUROC"], AUROC_random_sd=rsd["AUROC"],
                             EPr_random=rmean["EPr"], EPr_random_sd=rsd["EPr"],
                             AUROC_z=z))
        print(f"{'RANDOM':>14s} {rmean['AUROC']:8.4f} {rmean['AP']:9.5f} "
              f"{rmean['AP_ratio']:8.3f} {rmean['EPr']:7.3f}"
              f"   (sd: AUROC {rsd['AUROC']:.4f}, EPr {rsd['EPr']:.3f}"
              f", {args.seeds} seeds)")

        if args.diagnostics:
            diag, corr_deg, corr_cand = diagnostics(absW, mask, gidx, tidx, y, genes, tfs)
            print("  -- diagnostics (raw |A|) ---------------------------------")
            for name, auroc in diag:
                print(f"     AUROC from {name:<20s} {auroc:.4f}")
            print(f"     Spearman(per-TF mean |A|, reference out-degree) {corr_deg:+.3f}")
            print(f"     Spearman(per-TF mean |A|, # motif candidates)   {corr_cand:+.3f}")
            print("     negative correlations = weight flows AWAY from hub TFs")
        print()

    if not rows:
        sys.exit("no network had enough reference edges inside the mask")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(workdir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    fig_dir = os.path.join(workdir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{args.label}_summary.png")
    plot_mode = args.normalize if not args.all_normalizations else "tf-z"
    make_summary_figure(df[df["normalize"] == plot_mode].reset_index(drop=True),
                        args.label, f"{panel_tag}, normalize={plot_mode}", fig_path)

    print(f"Metrics: {csv_path}")
    print(f"Figure:  {fig_path}  (+ .pdf)")
    print("\nRead it this way: AUROC must clear the random control by several sd "
          "to mean anything,\nand a score that only survives at normalize=none is "
          "a per-TF/per-gene scale artifact.")


if __name__ == "__main__":
    main()
