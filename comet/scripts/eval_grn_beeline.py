"""
Score a COMET GRN run against BEELINE gold-standard reference networks.

Evaluates the GRN-inference stage in isolation (the ranked TF->gene edge list),
independent of the trajectory/velocity machinery, using the same metrics as
BEELINE (Pratapa et al., Nat Methods 2020): AUROC, AUPRC, AUPRC-ratio vs a
random predictor, and early-precision (EP) + EP-ratio.

Predicted network
-----------------
A COMET run dir (e.g. .../run0) saves A_learned.npy (G genes x T TFs, signed),
with row/col labels A_genes.npy / A_tfs.npy. An edge TF t -> gene g is scored by
|A_learned[g, t]| (magnitude = confidence). --grn selects the matrix:
  learned  A_learned                                  (default)
  base     A_base
  mixture  cell-averaged atom mixture, tensordot(mean mix_weights, A_dict)

Candidate universe (BEELINE convention for ChIP networks)
---------------------------------------------------------
  regulators = A_tfs  n  (TFs that appear as Gene1 in the reference network)
  targets    = A_genes,  optionally n the dataset's BEELINE expression panel
               (--panel, read from the h5ad's uns['beeline_genes'])
Self-edges excluded. Ground-truth positives are reference edges lying inside the
candidate universe. Matching is case-insensitive (BEELINE symbols are uppercase).

Usage:
  python scripts/eval_grn_beeline.py --run comet_out_trajectory_final/nestorowa \
      --h5ad ../datasets/nestorowa/rna.h5ad \
      [--panel] [--grn learned]
Networks are read from $BEELINE_DATA/networks/Networks/mouse (default BEELINE_DATA: <repo>/datasets/_beeline_raw).
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

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


def load_grn(run_dir: str, which: str):
    A_genes = np.load(f"{run_dir}/A_genes.npy", allow_pickle=True).astype(str)
    A_tfs = np.load(f"{run_dir}/A_tfs.npy", allow_pickle=True).astype(str)
    if which == "learned":
        W = np.load(f"{run_dir}/A_learned.npy")
    elif which == "base":
        W = np.load(f"{run_dir}/A_base.npy")
    elif which == "mixture":
        A_dict = np.load(f"{run_dir}/A_dict.npy")           # (D, G, T)
        w = np.load(f"{run_dir}/mix_weights.npy").mean(0)   # (D,)
        W = np.tensordot(w, A_dict, axes=([0], [0]))        # (G, T)
    else:
        raise ValueError(which)
    assert W.shape == (len(A_genes), len(A_tfs))
    return A_genes, A_tfs, np.abs(W).astype(np.float64)


def score_against(ref_csv, A_genes, A_tfs, absW, panel_genes=None):
    ref = pd.read_csv(ref_csv)
    ref["Gene1"] = ref["Gene1"].astype(str).str.upper()
    ref["Gene2"] = ref["Gene2"].astype(str).str.upper()

    genes_u = np.array([g.upper() for g in A_genes])
    tfs_u = np.array([t.upper() for t in A_tfs])

    # candidate targets: our genes, optionally restricted to the BEELINE panel
    tgt_mask = np.ones(len(genes_u), bool)
    if panel_genes is not None:
        panel = set(g.upper() for g in panel_genes)
        tgt_mask = np.array([g in panel for g in genes_u])
    tgt_idx = np.where(tgt_mask)[0]
    tgt_set = set(genes_u[tgt_idx])

    # candidate regulators: our TFs that the reference network actually knows about
    ref_regs = set(ref["Gene1"])
    reg_idx = np.array([i for i, t in enumerate(tfs_u) if t in ref_regs], dtype=int)
    if len(reg_idx) == 0 or len(tgt_idx) == 0:
        return None

    reg_names = tfs_u[reg_idx]
    # positives inside the candidate universe (exclude self-edges)
    gt = set()
    reg_set = set(reg_names)
    for t, g in zip(ref["Gene1"].values, ref["Gene2"].values):
        if t in reg_set and g in tgt_set and t != g:
            gt.add((t, g))

    # build flat candidate arrays: scores + labels
    scores, labels = [], []
    tgt_names = genes_u[tgt_idx]
    for ci, ti in enumerate(reg_idx):
        t = reg_names[ci]
        col = absW[tgt_idx, ti]                     # scores for this TF over targets
        for gi, g in enumerate(tgt_names):
            if t == g:
                continue
            scores.append(col[gi])
            labels.append(1 if (t, g) in gt else 0)
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    n_pos, n_cand = int(labels.sum()), len(labels)
    if n_pos == 0:
        return {"n_reg": len(reg_idx), "n_tgt": len(tgt_idx), "n_cand": n_cand,
                "n_pos": 0, "note": "no ground-truth positives in universe"}

    density = n_pos / n_cand
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    # early precision: precision among top-n_pos ranked predictions
    order = np.argsort(-scores)
    topk = order[:n_pos]
    ep = labels[topk].mean()
    return {
        "n_reg": len(reg_idx), "n_tgt": len(tgt_idx), "n_cand": n_cand,
        "n_pos": n_pos, "density": density,
        "AUROC": auroc, "AUPRC": auprc, "AUPRC_ratio": auprc / density,
        "EP": ep, "EP_ratio": ep / density,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="COMET run dir with A_learned.npy etc.")
    ap.add_argument("--h5ad", default=None, help="dataset h5ad (for --panel gene list)")
    ap.add_argument("--panel", action="store_true",
                    help="restrict targets to the BEELINE expression panel")
    ap.add_argument("--grn", default="learned", choices=["learned", "base", "mixture"])
    ap.add_argument("--net-dir", default=NET_DIR)
    args = ap.parse_args()

    A_genes, A_tfs, absW = load_grn(args.run, args.grn)

    panel_genes = None
    if args.panel:
        import anndata as ad
        adata = ad.read_h5ad(args.h5ad)
        panel_genes = list(np.asarray(adata.uns["beeline_genes"]).astype(str))

    print(f"run: {args.run}")
    print(f"GRN: {args.grn}  ({len(A_genes)} genes x {len(A_tfs)} TFs)"
          f"{'  [targets restricted to BEELINE panel]' if args.panel else ''}\n")
    hdr = f"{'network':20s} {'reg':>4s} {'tgt':>5s} {'pos':>6s} {'AUROC':>7s} " \
          f"{'AUPRC':>7s} {'AUPRCr':>7s} {'EP':>6s} {'EPr':>6s}"
    print(hdr); print("-" * len(hdr))
    for name, fn in NETWORKS.items():
        r = score_against(os.path.join(args.net_dir, fn), A_genes, A_tfs, absW, panel_genes)
        if r is None or r.get("n_pos", 0) == 0:
            print(f"{name:20s}  (no scorable edges)")
            continue
        print(f"{name:20s} {r['n_reg']:>4d} {r['n_tgt']:>5d} {r['n_pos']:>6d} "
              f"{r['AUROC']:>7.3f} {r['AUPRC']:>7.4f} {r['AUPRC_ratio']:>7.2f} "
              f"{r['EP']:>6.3f} {r['EP_ratio']:>6.2f}")
    print("\nAUPRCr / EPr are ratios vs a random predictor (>1 = better than chance).")


if __name__ == "__main__":
    main()
