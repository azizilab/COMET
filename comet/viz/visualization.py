import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from config import RANDOM_STATE, OUT_DIR, FORCE_EMB
from utils.common import zscore_cols


def palette_for(cts):
    u = np.unique(cts); cmap = plt.cm.tab20 if len(u) > 10 else plt.cm.tab10
    return {c: cmap(i / max(1, len(u)-1)) for i, c in enumerate(u)}

def scatter_types(ax, V, cts, pal, s=10, alpha=0.35, legend=False):
    for c in np.unique(cts):
        m = cts == c
        ax.scatter(V[m,0], V[m,1], c=[pal[c]], s=s, alpha=alpha, edgecolors='none', label=c, rasterized=True)
    if legend: ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), frameon=False, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def align_to(emb2d, ref2d):
    """Rotate / reflect emb2d onto ref2d (orthogonal Procrustes on the centred coordinates; no scaling, no shear), so
    an embedding with an arbitrary orientation (UMAP, PCA) is shown in the reference's orientation. The result is
    re-centred on the reference's centre."""
    X = np.asarray(emb2d, np.float64); Y = np.asarray(ref2d, np.float64)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    return (Xc @ (U @ Vt) + Y.mean(0)).astype(np.float32)


def clean_quiver(ax, V2D, emb, cts, title, pal=None, n_grid=22, root_mask=None):
    pal = pal or palette_for(cts); scatter_types(ax, emb, cts, pal, s=12, alpha=0.30)
    xmin, xmax = emb[:,0].min(), emb[:,0].max(); ymin, ymax = emb[:,1].min(), emb[:,1].max()
    xs, ys = np.linspace(xmin, xmax, n_grid), np.linspace(ymin, ymax, n_grid); r = (xmax-xmin)/n_grid*1.2
    gx, gy, gu, gv = [], [], [], []
    for xi in xs:
        for yi in ys:
            near = (emb[:,0]-xi)**2 + (emb[:,1]-yi)**2 < r**2
            if near.sum() > 3:
                gx.append(xi); gy.append(yi); gu.append(V2D[near,0].mean()); gv.append(V2D[near,1].mean())
    u, v = np.array(gu), np.array(gv); sp_ = np.sqrt(u**2 + v**2) + 1e-8
    ax.quiver(gx, gy, u/sp_, v/sp_, sp_, cmap='magma', scale=28, width=0.006, headwidth=4, zorder=4)
    if root_mask is not None and root_mask.any():
        rc = emb[root_mask].mean(0); ax.scatter(*rc, marker="*", s=420, c="lime", edgecolors="k", zorder=6)
    ax.set_title(title, fontsize=10, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])


def make_embeddings(b, seed=RANDOM_STATE, wanted=None, out_dir=None, force=None):
    """Display embeddings: {name: (emb2d, kNN_source_space)}. These are DISPLAY projections only — the same
    gene-space velocity is projected onto each; they do not change the trajectory. `wanted` (a set/list of names)
    restricts which are built so UMAP isn't computed if unused. decipher2d + pca2d are always available; the UMAP
    variants require `umap` (skipped if absent).

    decipher2d is Decipher's own 2D v, unchanged. Every other embedding is rotated / reflected onto it (align_to), so
    all panels share the Decipher orientation and look the same across reruns. The raw 2D coordinates are cached to
    {out_dir}/emb2d_{b.name}_seed{seed}.npz (out_dir defaults to {OUT_DIR}/{b.name}) and reused unless `force` (or
    config.FORCE_EMB) is True, so UMAP is computed once."""
    want = None if wanted is None else set(wanted)
    need = lambda n: (want is None) or (n in want)
    force = FORCE_EMB if force is None else force

    ds_dir = out_dir or os.path.join(OUT_DIR, b.name)
    cache_path = os.path.join(ds_dir, f"emb2d_{b.name}_seed{seed}.npz")
    cached2d = {}
    if not force and os.path.exists(cache_path):
        with np.load(cache_path) as f:
            cached2d = {k: f[k] for k in f.files}
    raw2d = {}

    def get2d(key, compute_fn):
        raw2d[key] = cached2d[key] if key in cached2d else np.asarray(compute_fn(), np.float32)
        return raw2d[key] if key == "decipher2d" else align_to(raw2d[key], b.V_dec2)

    E = {}
    Xz = zscore_cols(b.X)
    Xp = PCA(n_components=min(30, b.G - 1, b.N - 1), random_state=seed).fit_transform(Xz).astype(np.float32)

    if need("decipher2d"):
        E["decipher2d"] = (get2d("decipher2d", lambda: b.V_dec2), b.Z_dec)
    if need("pca2d"):
        E["pca2d"] = (get2d("pca2d", lambda: Xp[:, :2]), Xp)
    if need("umap_decipher") or need("umap_pca"):
        try:
            import umap
            if need("umap_decipher"):
                E["umap_decipher"] = (get2d("umap_decipher", lambda: umap.UMAP(
                    random_state=seed, n_neighbors=15, min_dist=0.3).fit_transform(b.Z_dec)), b.Z_dec)
            if need("umap_pca"):
                E["umap_pca"] = (get2d("umap_pca", lambda: umap.UMAP(
                    random_state=seed, n_neighbors=15, min_dist=0.3).fit_transform(Xp)), Xp)
        except Exception as e:
            print(f"  [umap unavailable, plotting decipher2d + pca2d only: {e}]")

    new2d = {k: v for k, v in raw2d.items() if k not in cached2d}
    if new2d:
        os.makedirs(ds_dir, exist_ok=True)
        np.savez(cache_path, **{**cached2d, **new2d})

    return E


def plot_celltypes_pseudotime(b, embeddings, q, root_mask, sink_mask, path, names=("decipher2d", "umap_decipher")):
    """One row per display embedding: left = cell types (labelled at their medians), right = committor pseudotime q
    with the root (green, star = centroid) and sink (red) cells outlined."""
    names = [n for n in names if n in embeddings]
    cts = b.cell_types; pal = palette_for(cts); types = np.unique(cts); s = 14 if b.N < 3000 else 6
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 4.75 * len(names)), squeeze=False)
    for r, en in enumerate(names):
        emb = embeddings[en][0]; ax = axes[r, 0]
        for t in types:
            m = cts == t
            ax.scatter(emb[m, 0], emb[m, 1], s=s, c=[pal[t]], alpha=.75, edgecolors="none", label=f"{t} ({m.sum()})",
                       rasterized=True)
        for t in types:
            cx, cy = np.median(emb[cts == t], 0)
            ax.text(cx, cy, t, fontsize=8, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=pal[t], alpha=.85))
        ax.set_title(f"{b.name} — {en}: cell types", fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        if r == len(names) - 1:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.02), ncol=4, frameon=False, fontsize=7, markerscale=2)
        ax = axes[r, 1]
        sc_ = ax.scatter(emb[:, 0], emb[:, 1], c=q, s=s, cmap="viridis", edgecolors="none", rasterized=True)
        ax.scatter(emb[sink_mask, 0], emb[sink_mask, 1], s=s * 1.6, facecolors="none", edgecolors="red", lw=.6,
                   label="sinks")
        ax.scatter(emb[root_mask, 0], emb[root_mask, 1], s=s * 1.6, facecolors="none", edgecolors="lime", lw=.6,
                   label="root cells")
        ax.scatter(*emb[root_mask].mean(0), marker="*", s=380, c="lime", edgecolors="k", zorder=5)
        plt.colorbar(sc_, ax=ax, fraction=.04, label="committor pseudotime q")
        ax.set_title(f"{b.name} — {en}: COMET pseudotime", fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
