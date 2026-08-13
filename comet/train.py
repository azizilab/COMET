import argparse
import os

import matplotlib.pyplot as plt

from config import RUN_DATASETS, BASES, OUT_DIR, STATE_SPACE, DISPLAY_EMBEDDINGS
from data.load import LOADERS
from training.training_loop import run_comet
from utils.geometry import project_to_2d
from viz.visualization import clean_quiver, make_embeddings

EMB_ORDER = ["decipher2d", "pca2d", "umap_decipher", "umap_pca"]


def run(datasets=None, state_space=None, display_embeddings=None, out_dir=None):
    """Load, train, and plot COMET for the given datasets.

    datasets:           list of dataset names ("hema", "aml89", "ma2020"); defaults to
                         config.RUN_DATASETS.
    state_space:        optional {name: "decipher"|"pca"} override merged over config.STATE_SPACE.
    display_embeddings: optional {name: [embedding names]} override merged over
                         config.DISPLAY_EMBEDDINGS.
    """
    names = list(datasets) if datasets else list(RUN_DATASETS)
    out_root = out_dir or OUT_DIR
    os.makedirs(out_root, exist_ok=True)

    if state_space:
        STATE_SPACE.update(state_space)
    if display_embeddings:
        DISPLAY_EMBEDDINGS.update(display_embeddings)

    bundles = {}
    for name in names:
        print(f"Loading {name}...")
        bundles[name] = LOADERS[name](BASES[name])
        b = bundles[name]
        print(f"{name:8s} | paired={b.paired} | N={b.N} G={b.G} T={b.T} | "
              f"types={sorted(set(b.cell_types))}")

    results = {}
    for name in names:
        results[name] = run_comet(bundles[name], out_dir=os.path.join(out_root, name))
    print("\nTraining complete for:", list(results))

    # ---- clean_quiver figure: same gene-space velocity, projected onto each dataset's
    # requested display embeddings (decipher2d/pca2d always; UMAP if `umap` is installed) ----
    embeddings, wanted = {}, {}
    for name in names:
        req = [e for e in DISPLAY_EMBEDDINGS.get(name, EMB_ORDER) if e in EMB_ORDER]
        embeddings[name] = make_embeddings(bundles[name], wanted=req)
        wanted[name] = [e for e in req if e in embeddings[name]]

    nrow = len(names); ncol = max(len(wanted[n]) for n in names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow), squeeze=False)
    for r, name in enumerate(names):
        b, R, cols = bundles[name], results[name], wanted[name]
        for c in range(ncol):
            if c < len(cols):
                en = cols[c]; emb, src = embeddings[name][en]
                V_proj2d = project_to_2d(R["V_pred"], emb, b.X, search_space=src, k=45, scale=0.4,
                                         drift_correct=True)
                clean_quiver(axes[r, c], V_proj2d, emb, b.cell_types,
                             title=f"{name} — {en}", root_mask=R["root_mask"])
            else:
                axes[r, c].axis("off")
    plt.suptitle("COMET velocity (clean_quiver): committor-oriented, GRN-propagated ATAC lead",
                 fontweight="bold", y=1.002)
    plt.tight_layout()
    plt.savefig(f"{out_root}/velocity_clean_quiver_all_embeddings.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_root}/velocity_clean_quiver_all_embeddings.png")

    return bundles, results


def _parse_kv_list(items):
    return dict(item.split("=", 1) for item in items) if items else None


def _parse_args():
    p = argparse.ArgumentParser(description="Run the COMET GRN + committor-anchored velocity pipeline.")
    p.add_argument("--datasets", nargs="+", choices=list(BASES.keys()), default=None,
                   help="Datasets to run (default: config.RUN_DATASETS).")
    p.add_argument("--state-space", nargs="+", default=None, metavar="NAME=SPACE",
                   help="Override STATE_SPACE, e.g. --state-space hema=pca ma2020=decipher")
    p.add_argument("--display-embeddings", nargs="+", default=None, metavar="NAME=EMB,EMB,...",
                   help="Override DISPLAY_EMBEDDINGS, e.g. --display-embeddings hema=decipher2d,pca2d")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    state_space = _parse_kv_list(args.state_space)
    de_raw = _parse_kv_list(args.display_embeddings)
    display_embeddings = {k: v.split(",") for k, v in de_raw.items()} if de_raw else None
    run(args.datasets, state_space=state_space, display_embeddings=display_embeddings)
