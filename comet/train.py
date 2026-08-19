import argparse
import functools
import os

import matplotlib.pyplot as plt

from config import (
    RUN_DATASETS, BASES, OUT_DIR, STATE_SPACE, DISPLAY_EMBEDDINGS,
    ENTROPY_ROOT_TRUST, COMMIT_W_SCALE, VEL_MOMENT_KNN, BETA_PERP, MARKER_GENES,
)
from data.load import LOADERS, load_generic
from training.training_loop import run_comet
from utils.geometry import project_to_2d
from viz.visualization import clean_quiver, make_embeddings

EMB_ORDER = ["decipher2d", "pca2d", "umap_decipher", "umap_pca"]

# fields --new-dataset accepts that just pass through as loader kwargs (see load_generic /
# README_DATASETS.md for what each means and the canonical filenames they default to)
_LOADER_KEYS = ("cell_type_col", "pseudotime_col", "rna_layer",
                "rna_file", "atac_mtx", "atac_peaks_file", "decipher_npz")
# hyperparameter overrides --new-dataset accepts; only set if the user passes them, so anything
# omitted keeps falling back to the same defaults the existing datasets use (see the .get(name,
# default) calls in training_loop.py / train.py) -- (config dict, cast) pairs
_HYPERPARAM_KEYS = {
    "entropy_root_trust": (ENTROPY_ROOT_TRUST, float),
    "commit_w_scale":     (COMMIT_W_SCALE, float),
    "vel_moment_knn":     (VEL_MOMENT_KNN, int),
    "beta_perp":          (BETA_PERP, float),
    "state_space":        (STATE_SPACE, str),
}


def register_new_dataset(spec):
    """Register + wire up a dataset that isn't one of the built-in defaults (hema/aml89/ma2020/
    nestorowa/paul15). Always goes through the generic loader (data/load.py: load_generic) --
    the dataset's files must follow the canonical layout documented in README_DATASETS.md.
    `spec` is a flat {key: str} dict (as produced by parsing --new-dataset KEY=VALUE args).
    Required: name, base. Returns the registered dataset name.
    """
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
    if "marker_genes" in spec:
        MARKER_GENES[name] = spec["marker_genes"].split(",")
    if "display_embeddings" in spec:
        DISPLAY_EMBEDDINGS[name] = spec["display_embeddings"].split(",")
    return name


def run(datasets=None, state_space=None, display_embeddings=None, out_dir=None):
    """Load, train, and plot COMET for the given datasets.

    datasets:           list of dataset names, e.g. ["hema", "aml89"] or ["my_new_dataset"];
                         must already be registered in config.BASES/data.load.LOADERS (built-in
                         datasets are pre-registered; call register_new_dataset(...) first for
                         anything else). Defaults to config.RUN_DATASETS.
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
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Built-in datasets to run (default: config.RUN_DATASETS). Must be names "
                        "already in config.BASES -- for a dataset that isn't one of the 5 "
                        "defaults, use --new-dataset instead.")
    p.add_argument("--new-dataset", nargs="+", default=None, metavar="KEY=VALUE",
                   help="Register + run a dataset that isn't one of the 5 defaults, via the "
                        "generic loader (canonical file layout only -- see README_DATASETS.md). "
                        "Required: name=... base=.... Optional loader overrides: "
                        "cell_type_col= pseudotime_col= rna_layer= rna_file= atac_mtx= "
                        "atac_peaks_file= decipher_npz=. Optional hyperparameter overrides "
                        "(see README_DATASETS.md for defaults/guidance): entropy_root_trust= "
                        "commit_w_scale= vel_moment_knn= beta_perp= state_space= "
                        "marker_genes=GENE1,GENE2 display_embeddings=decipher2d,pca2d. "
                        "Example: --new-dataset name=my_ds base=/path/to/my_ds "
                        "cell_type_col=cell_type pseudotime_col=pseudotime")
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

    datasets = list(args.datasets) if args.datasets else None
    new_ds_spec = _parse_kv_list(args.new_dataset)
    if new_ds_spec:
        new_name = register_new_dataset(new_ds_spec)
        # --new-dataset alone runs just that dataset; combined with --datasets, runs both.
        datasets = (datasets + [new_name]) if datasets else [new_name]

    run(datasets, state_space=state_space, display_embeddings=display_embeddings)
