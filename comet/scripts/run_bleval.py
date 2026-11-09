"""
Set up and run the OFFICIAL BEELINE evaluator (BLEval) on a COMET GRN run,
then draw a clean single-method summary figure.

Unlike scripts/eval_grn_beeline.py (a lightweight reimplementation for fast
iteration), this stages the directory layout the official tool expects and
invokes Murali-group/Beeline's real BLEvaluator.py CLI, so AUPRC / AUROC /
EarlyPrecision use BEELINE's exact conventions and are comparable to published
results. BEELINE's own BLPlotter is built for many-algorithm x many-dataset
grids, so a single-method summary figure is drawn here instead.

Gene panel (--panel N)
----------------------
The BEELINE convention for real scRNA-seq is "all significantly-varying TFs +
the top-N most-varying genes" (N = 500 and 1000). Pass --panel 500 or 1000 to
score on that panel (targets = top-N varying genes that COMET models, plus all
TFs; regulators = all TFs). --panel 0 (default) uses COMET's full gene universe.
Top-N ranking comes from BEELINE's GeneOrdering.csv (--gene-ordering).

SIMPLE ONE-COMMAND USAGE
------------------------
  python scripts/run_bleval.py --run comet_out_trajectory_final/<dataset> --panel 1000

  # add --grn learned|base|mixture (default learned)
  # add --label NAME  (name shown in the figure; default COMET)
Outputs: <run>/bleval/  ->  outputs/.../<label>/... (metric CSVs)
                            figures/<label>_summary.png  (+ .pdf)
Any env with pandas/sklearn/pyyaml/matplotlib works (e.g. decipher-env).
Docker is NOT needed for evaluation.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

# BEELINE reference data (networks/ and BEELINE-data/ from https://github.com/Murali-group/Beeline):
# $BEELINE_DATA, or <repo>/datasets/_beeline_raw by default.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEELINE_DATA = os.environ.get("BEELINE_DATA", os.path.join(REPO_ROOT, "datasets", "_beeline_raw"))
NET_DIR = os.path.join(BEELINE_DATA, "networks", "Networks", "mouse")
NETWORKS = {
    "mHSC-ChIP-seq": "mHSC-ChIP-seq-network.csv",
    "Non-Specific-ChIP": "Non-Specific-ChIP-seq-network.csv",
    "STRING": "STRING-network.csv",
}
DEFAULT_GENE_ORDERING = os.path.join(BEELINE_DATA, "BEELINE-data", "inputs", "scRNA-Seq", "mHSC-E", "GeneOrdering.csv")


def load_grn(run_dir, which):
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
    return A_genes, A_tfs, np.abs(W).astype(np.float64)


def build_ranked_edges(A_genes, A_tfs, absW):
    g = np.array([s.upper() for s in A_genes])
    t = np.array([s.upper() for s in A_tfs])
    gi, ti = np.nonzero(absW)
    df = pd.DataFrame({"Gene1": t[ti], "Gene2": g[gi], "EdgeWeight": absW[gi, ti]})
    df = df[df["Gene1"] != df["Gene2"]]
    return df.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)


def make_universe(A_genes, A_tfs, panel_n, gene_ordering):
    """Gene universe to score over. panel_n=0 -> full COMET universe;
    else all TFs + top-N most-varying genes (BEELINE convention)."""
    genes_u = set(np.char.upper(A_genes.astype(str)))
    tfs_u = set(np.char.upper(A_tfs.astype(str)))
    if not panel_n:
        return genes_u | tfs_u, "full universe"
    go = pd.read_csv(gene_ordering, index_col=0)
    var_col = "Variance" if "Variance" in go.columns else go.columns[-1]
    ranked = [str(x).upper() for x in go.sort_values(var_col, ascending=False).index]
    top = [x for x in ranked if x in genes_u][:panel_n]     # top-N genes COMET models
    return set(top) | tfs_u, f"TFs + top {panel_n} HVGs"


def filter_network(csv_path, universe):
    ref = pd.read_csv(csv_path)
    ref["Gene1"] = ref["Gene1"].astype(str).str.upper()
    ref["Gene2"] = ref["Gene2"].astype(str).str.upper()
    m = ref["Gene1"].isin(universe) & ref["Gene2"].isin(universe)
    out = ref.loc[m, ["Gene1", "Gene2"]].drop_duplicates()
    return out[out["Gene1"] != out["Gene2"]]


def read_metric(net_out_dir, name, label):
    p = os.path.join(net_out_dir, name)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0)
    return float(df.loc[label].iloc[0]) if label in df.index else None


def make_summary_figure(rows, label, panel_tag, fig_path):
    """rows: list of dict(net, auprc_ratio, epr). Clean grouped-bar figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nets = [r["net"] for r in rows]
    x = np.arange(len(nets)); w = 0.38
    aupr = [r["auprc_ratio"] for r in rows]
    epr = [r["epr"] for r in rows]
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(nets), 4.2))
    b1 = ax.bar(x - w / 2, aupr, w, label="AUPRC ratio", color="#2a9d8f")
    b2 = ax.bar(x + w / 2, epr, w, label="EPr", color="#e76f51")
    ax.axhline(1.0, ls="--", lw=1, color="0.4")
    ax.text(len(nets) - 0.5, 1.02, "random", color="0.4", va="bottom", ha="right", fontsize=8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(nets)
    ax.set_ylabel("ratio vs random  (>1 = better)")
    ax.set_title(f"{label} — BEELINE mHSC-E  ({panel_tag})")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    fig.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--grn", default="learned", choices=["learned", "base", "mixture"])
    ap.add_argument("--label", default="COMET", help="method name shown in the figure")
    ap.add_argument("--panel", type=int, default=0,
                    help="0=full universe; 500/1000=all TFs + top-N varying genes")
    ap.add_argument("--gene-ordering", default=DEFAULT_GENE_ORDERING)
    ap.add_argument("--beeline-repo", default=os.environ.get("BEELINE_REPO"),
                    help="checkout of https://github.com/Murali-group/Beeline (default: $BEELINE_REPO)")
    ap.add_argument("--net-dir", default=NET_DIR)
    ap.add_argument("--workdir", default=None, help="default <run>/bleval")
    args = ap.parse_args()
    if not args.beeline_repo:
        ap.error("--beeline-repo (or $BEELINE_REPO) is required")

    label = args.label
    workdir = os.path.abspath(args.workdir or os.path.join(args.run, "bleval"))
    in_dir = os.path.join(workdir, "inputs")
    out_dir = os.path.join(workdir, "outputs")
    exp = "cosmos"

    A_genes, A_tfs, absW = load_grn(args.run, args.grn)
    universe, panel_tag = make_universe(A_genes, A_tfs, args.panel, args.gene_ordering)
    ranked = build_ranked_edges(A_genes, A_tfs, absW)
    print(f"{label} GRN '{args.grn}': {len(A_genes)} genes x {len(A_tfs)} TFs "
          f"-> {len(ranked)} nonzero edges; panel={panel_tag} ({len(universe)} genes)")

    datasets, dens = [], {}
    for net, fn in NETWORKS.items():
        gt = filter_network(os.path.join(args.net_dir, fn), universe)
        if gt.empty:
            print(f"  {net}: no edges within panel, skipping"); continue
        ds_in = os.path.join(in_dir, net); os.makedirs(ds_in, exist_ok=True)
        gt.to_csv(os.path.join(ds_in, "refNetwork.csv"), index=False)
        algo_out = os.path.join(out_dir, exp, net, "run1", label)
        os.makedirs(algo_out, exist_ok=True)
        ranked.to_csv(os.path.join(algo_out, "rankedEdges.csv"), sep="\t", index=False)
        P = len(set(gt["Gene1"]) | set(gt["Gene2"]))
        dens[net] = len(gt) / (P * (P - 1))            # all-pairs random baseline for AUPRC
        print(f"  {net}: {len(gt)} GT edges over {P} genes  (density {dens[net]:.5f})")
        datasets.append({"dataset_id": net, "should_run": [True],
                         "groundTruthNetwork": "refNetwork.csv",
                         "runs": [{"run_id": "run1"}]})

    config = {
        "input_settings": {"input_dir": in_dir,
                           "algorithms": [{"algorithm_id": label, "should_run": [True]}],
                           "datasets": datasets},
        "output_settings": {"output_dir": out_dir, "experiment_id": exp},
    }
    cfg_path = os.path.join(workdir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # --- official CLI: AUPRC / AUROC / (all-pairs) EarlyPrecision ------------
    env = dict(os.environ, PYTHONPATH=args.beeline_repo)
    print("\n$ python BLEvaluator.py -c config.yaml -a -e   (official CLI)")
    subprocess.run([sys.executable, "BLEvaluator.py", "-c", cfg_path, "-a", "-e"],
                   cwd=args.beeline_repo, env=env, check=True)
    for net in dens:
        p = os.path.join(out_dir, exp, net, "EarlyPrecision.csv")
        if os.path.exists(p):
            os.replace(p, os.path.join(out_dir, exp, net, "EarlyPrecision_allpairs.csv"))

    # --- TF->gene EarlyPrecision (real-data convention) ---------------------
    sys.path.insert(0, args.beeline_repo)
    from BLEval.data import EvaluationData
    from BLEval.EarlyPrecision import EarlyPrecision
    EarlyPrecision(tf_edges=True)(EvaluationData(config, root=None))
    for net in dens:
        p = os.path.join(out_dir, exp, net, "EarlyPrecision.csv")
        if os.path.exists(p):
            os.replace(p, os.path.join(out_dir, exp, net, "EarlyPrecision_TFedges.csv"))

    # --- consolidate + print + figure --------------------------------------
    print(f"\n=== OFFICIAL BLEval — {label} (grn={args.grn}, panel={panel_tag}) ===")
    hdr = f"{'network':20s} {'AUPRC':>9s} {'AUPRCr':>8s} {'AUROC':>8s} {'EPr(TF->gene)':>14s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for net in dens:
        nod = os.path.join(out_dir, exp, net)
        auprc = read_metric(nod, "AUPRC.csv", label)
        auroc = read_metric(nod, "AUROC.csv", label)
        epr = read_metric(nod, "EarlyPrecision_TFedges.csv", label)
        auprc_ratio = auprc / dens[net] if auprc is not None else float("nan")
        f = lambda v, s="{:.4f}": s.format(v) if v is not None else "  n/a"
        print(f"{net:20s} {f(auprc):>9s} {auprc_ratio:>8.2f} {f(auroc):>8s} {f(epr):>14s}")
        rows.append({"net": net, "auprc_ratio": auprc_ratio,
                     "epr": epr if epr is not None else float("nan")})

    fig_dir = os.path.join(workdir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{label}_summary.png")
    make_summary_figure(rows, label, panel_tag, fig_path)
    print(f"\nMetric CSVs: {os.path.join(out_dir, exp)}/<network>/")
    print(f"Figure:      {fig_path}  (+ .pdf)")


if __name__ == "__main__":
    main()
