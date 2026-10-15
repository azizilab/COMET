import argparse
import functools
import os

import matplotlib.pyplot as plt

from config import (
    RUN_DATASETS, BASES, OUT_DIR, STATE_SPACE, DISPLAY_EMBEDDINGS,
    ORIGIN_SIGN, VEL_MOMENT_KNN,
)
from data.load import LOADERS, load_generic
from training.training_loop import run_comet
from utils.geometry import project_to_2d
from utils.seed import set_seed
from viz.visualization import clean_quiver, make_embeddings, palette_for, plot_celltypes_pseudotime

EMB_ORDER = ["decipher2d", "pca2d", "umap_decipher", "umap_pca"]

# --new-dataset keys passed through to load_generic (see README_DATASETS.md)
_LOADER_KEYS = ("cell_type_col", "pseudotime_col", "rna_layer",
                "rna_file", "atac_mtx", "atac_peaks_file", "decipher_npz")
# --new-dataset hyperparameter overrides: key -> (config dict, cast); omitted keys use the defaults
_HYPERPARAM_KEYS = {
    "origin_sign":    (ORIGIN_SIGN, float),
    "vel_moment_knn": (VEL_MOMENT_KNN, int),
    "state_space":    (STATE_SPACE, str),
}


def register_new_dataset(spec):
    """Register a non-built-in dataset through the generic loader (data/load.py:load_generic; file layout in
    README_DATASETS.md). `spec` is a flat {key: str} dict as parsed from --new-dataset KEY=VALUE.
    Required: name, base. Returns the registered dataset name."""
    name, base = spec.get("name"), spec.get("base")
    if not name or not base:
        raise ValueError("--new-dataset requires name=... and base=... (see README_DATASETS.md)")
    if name in BASES:
        raise ValueError(f"'{name}' is already a registered dataset name")

    loader_kwargs = {k: spec[k] for k in _LOADER_KEYS if k in spec}
    BASES[name] = base
    LOADERS[name] = functools.partial(load_generic, name=name, **loader_kwargs)

    for key, (cfg_dict, cast) in _HYPERPARAM_KEYS.items():
        if key in spec:
            cfg_dict[name] = cast(spec[key])
    if "display_embeddings" in spec:
        DISPLAY_EMBEDDINGS[name] = spec["display_embeddings"].split(",")
    return name


def run(datasets=None, state_space=None, display_embeddings=None, out_dir=None, train=None):
    """Load, run and plot COMET for the given datasets; returns (bundles, results).

    datasets:           registered dataset names (default config.RUN_DATASETS); call register_new_dataset first
                        for anything not in config.BASES.
    state_space:        optional {name: "decipher"|"pca"} merged over config.STATE_SPACE.
    display_embeddings: optional {name: [embedding names]} merged over config.DISPLAY_EMBEDDINGS.
    out_dir:            output root (default config.OUT_DIR); each dataset writes to <out_dir>/<name>/.
    train:              None = config.TRAIN_GRN; False = trajectory only (no GRN training).
    """
    set_seed()

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
        results[name] = run_comet(bundles[name], out_dir=os.path.join(out_root, name), train=train)
    print("\nFinished:", list(results))

    # ---- figures: the gene-space velocity projected onto each display embedding (clean_quiver), and cell types
    # next to the committor pseudotime; per dataset, plus all datasets in one clean_quiver figure ----
    embeddings, wanted = {}, {}
    for name in names:
        req = [e for e in DISPLAY_EMBEDDINGS.get(name, EMB_ORDER) if e in EMB_ORDER]
        embeddings[name] = make_embeddings(bundles[name], wanted=req, out_dir=os.path.join(out_root, name))
        wanted[name] = [e for e in req if e in embeddings[name]]

    def quiver_row(axes_row, name):
        b, R, cols = bundles[name], results[name], wanted[name]
        for c, ax in enumerate(axes_row):
            if c < len(cols):
                en = cols[c]; emb, src = embeddings[name][en]
                V_proj2d = project_to_2d(R["V_pred"], emb, b.X, search_space=src, k=45, scale=0.4,
                                         drift_correct=True)
                clean_quiver(ax, V_proj2d, emb, b.cell_types, title=f"{name} — {en}", root_mask=R["root_mask"])
            else:
                ax.axis("off")

    title = "COMET trajectory (clean_quiver): consensus chromatin origin -> committor + chromatin-lead fill-in"
    for name in names:
        ds_dir = os.path.join(out_root, name); ncol = len(wanted[name])
        fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.6), squeeze=False)
        quiver_row(axes[0], name)
        pal = palette_for(bundles[name].cell_types)
        axes[0, -1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in pal.items()],
                           loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
        plt.suptitle(title, fontweight="bold", y=1.02); plt.tight_layout()
        plt.savefig(f"{ds_dir}/{name}_clean_quiver.png", dpi=140, bbox_inches="tight"); plt.close(fig)
        R = results[name]
        plot_celltypes_pseudotime(bundles[name], embeddings[name], R["q_commit"], R["root_mask"], R["sink_mask"],
                                  f"{ds_dir}/{name}_celltypes_pseudotime.png")

    nrow = len(names); ncol = max(len(wanted[n]) for n in names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow), squeeze=False)
    for r, name in enumerate(names):
        quiver_row(axes[r], name)
    plt.suptitle(title, fontweight="bold", y=1.002)
    plt.tight_layout()
    plt.savefig(f"{out_root}/velocity_clean_quiver_all_embeddings.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures -> {out_root}/velocity_clean_quiver_all_embeddings.png and <name>/<name>_*.png")

    return bundles, results


def _parse_kv_list(items):
    return dict(item.split("=", 1) for item in items) if items else None


def _parse_args():
    p = argparse.ArgumentParser(description="Run COMET: chromatin-origin committor trajectory (+ GRN dictionary).")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Registered datasets to run (default: config.RUN_DATASETS). For anything not in "
                        "config.BASES use --new-dataset.")
    p.add_argument("--new-dataset", nargs="+", default=None, metavar="KEY=VALUE",
                   help="Register and run a dataset through the generic loader (file layout: "
                        "README_DATASETS.md). Required: name= base=. Loader overrides: cell_type_col= "
                        "pseudotime_col= rna_layer= rna_file= atac_mtx= atac_peaks_file= decipher_npz=. "
                        "Hyperparameters: origin_sign= vel_moment_knn= state_space= "
                        "display_embeddings=decipher2d,pca2d.")
    p.add_argument("--state-space", nargs="+", default=None, metavar="NAME=SPACE",
                   help="Override STATE_SPACE, e.g. --state-space hema=pca ma2020=decipher")
    p.add_argument("--out-dir", default=None,
                   help="Output root (default: config.OUT_DIR); each dataset writes to <out-dir>/<name>/.")
    p.add_argument("--trajectory-only", action="store_true",
                   help="Compute only the trajectory (committor pseudotime + velocity field); skip GRN training.")
    p.add_argument("--display-embeddings", nargs="+", default=None, metavar="NAME=EMB,EMB,...",
                   help="Override DISPLAY_EMBEDDINGS, e.g. --display-embeddings hema=decipher2d,pca2d")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    state_space = _parse_kv_list(args.state_space)
    de_raw = _parse_kv_list(args.display_embeddings)
    display_embeddings = {k: v.split(",") for k, v in de_raw.items()} if de_raw else None

    datasets = list(args.datasets) if args.datasets else None
    new_ds_spec = _parse_kv_list(args.new_dataset)
    if new_ds_spec:
        new_name = register_new_dataset(new_ds_spec)
        datasets = (datasets + [new_name]) if datasets else [new_name]

    run(datasets, state_space=state_space, display_embeddings=display_embeddings, out_dir=args.out_dir,
        train=False if args.trajectory_only else None)
