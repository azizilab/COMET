import io
import os

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from config import RANDOM_STATE
from utils.common import zscore_cols
from utils.committor import palette_for, scatter_types


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


def make_embeddings(b, seed=RANDOM_STATE, wanted=None):
    """Display embeddings: {name: (emb2d, kNN_source_space)}. These are DISPLAY projections only —
    the same gene-space velocity is projected onto each; they do not change the trajectory.
    `wanted` (a set/list of names) restricts which are built so UMAP isn't computed if unused.
    decipher2d + pca2d are always available; the UMAP variants require `umap` (skipped if absent)."""
    want = None if wanted is None else set(wanted)
    need = lambda n: (want is None) or (n in want)
    E = {}
    E["decipher2d"] = (b.V_dec2, b.Z_dec)
    Xz = zscore_cols(b.X)
    Xp = PCA(n_components=min(30, b.G - 1, b.N - 1), random_state=seed).fit_transform(Xz)
    E["pca2d"] = (Xp[:, :2].astype(np.float32), Xp.astype(np.float32))
    if need("umap_decipher") or need("umap_pca"):
        try:
            import umap
            if need("umap_decipher"):
                E["umap_decipher"] = (umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.3)
                                      .fit_transform(b.Z_dec).astype(np.float32), b.Z_dec)
            if need("umap_pca"):
                E["umap_pca"]      = (umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.3)
                                      .fit_transform(Xp).astype(np.float32), Xp.astype(np.float32))
        except Exception as e:
            print(f"  [umap unavailable, plotting decipher2d + pca2d only: {e}]")
    return E


class VelocityGIFMaker:
    def __init__(self, dpi=100, mode="Decipher"):
        self.dpi = dpi
        self.frames = []
        self.mode = mode
 
    def _render_frame(self, X_2d, V_2d, pseudo_norm, epoch):
 
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['svg.fonttype'] = 'none'
 
        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=150)
 
        sc = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=pseudo_norm, cmap="viridis",
            s=20, alpha=0.6, edgecolors='none', rasterized=True
        )
 
        # ── Grid setup — MUST MATCH manual plot ──────────────────────────────
        density = 2
        grid_n  = int(60 * density)
 
        x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
        y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
 
        x_grid = np.linspace(x_min, x_max, grid_n)
        y_grid = np.linspace(y_min, y_max, grid_n)
        Xg, Yg = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[Xg.ravel(), Yg.ravel()]
 
        # ── Sparse sampling ───────────────────────────────────────────────────
        n_cells = min(3000, len(X_2d))
        cell_idx = np.random.choice(len(X_2d), n_cells, replace=False)
        X_small = X_2d[cell_idx]
        V_small = V_2d[cell_idx]
 
        # ── kNN projection onto grid ──────────────────────────────────────────
        k = 20
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_small)
        _, idx = nbrs.kneighbors(grid_points)
        V_grid = V_small[idx].mean(axis=1)
 
        # ── Cell support mask — MUST USE SAME PERCENTILE AS MANUAL PLOT ──────
        nbrs_cells = NearestNeighbors(n_neighbors=1).fit(X_2d)
        dist_to_cells, _ = nbrs_cells.kneighbors(grid_points)
        dist_to_cells = dist_to_cells.ravel()
        
        # KEY FIX: 40th percentile — same as the manual scVelo-style plot
        support_threshold = np.percentile(dist_to_cells, 40)
        support_mask = dist_to_cells < support_threshold
 
        # ── Velocity magnitude mask — MUST MATCH ─────────────────────────────
        speed = np.linalg.norm(V_grid, axis=1)
        cutoff_speed = np.percentile(speed, 35)
        velocity_mask = speed > cutoff_speed
 
        # ── Combined mask ─────────────────────────────────────────────────────
        final_mask = support_mask & velocity_mask
        if final_mask.sum() == 0:
            final_mask = np.ones(len(grid_points), dtype=bool)
 
        grid_points_f = grid_points[final_mask]
        V_grid_f      = V_grid[final_mask]
 
        # ── Reproject onto structured grid ────────────────────────────────────
        tree = cKDTree(grid_points_f)
        _, nn = tree.query(np.c_[Xg.ravel(), Yg.ravel()], k=1)
 
        U = V_grid_f[nn, 0].reshape(Xg.shape)
        W = V_grid_f[nn, 1].reshape(Yg.shape)
 
        # Apply support mask to structured grid
        grid_dist, _ = nbrs_cells.kneighbors(np.c_[Xg.ravel(), Yg.ravel()])
        grid_mask = grid_dist.ravel() < support_threshold
 
        U_flat = U.ravel(); W_flat = W.ravel()
        U_flat[~grid_mask] = 0.0
        W_flat[~grid_mask] = 0.0
        U = U_flat.reshape(Xg.shape)
        W = W_flat.reshape(Yg.shape)
 
        # ── Gaussian smoothing ────────────────────────────────────────────────
        U = gaussian_filter(U, sigma=1.2)
        W = gaussian_filter(W, sigma=1.2)
 
        # ── Streamplot ────────────────────────────────────────────────────────
        ax.streamplot(
            Xg, Yg, U, W,
            color="black", density=1.2,
            linewidth=1, arrowsize=1, arrowstyle='-|>',
            minlength=0.2, integration_direction='both', zorder=3
        )
 
        mode_label = self.mode
        ax.set_xlabel(f"{mode_label} 1", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"{mode_label} 2", fontsize=11, fontweight='bold')
        ax.set_title(f"Velocity ({mode_label}) @ Epoch {epoch}",
                     fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("Pseudotime", fontsize=10)
        cbar.outline.set_visible(False)
 
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
 
    def add_frame(self, X_2d, V_2d_smooth, pseudo_norm, epoch):
        img = self._render_frame(X_2d, V_2d_smooth, pseudo_norm, epoch)
        self.frames.append(img)
 
    def save(self, path="velocity_training.gif", fps=4):
        if not self.frames:
            print("No frames to save.")
            return
        duration_ms = int(1000 / fps)
        self.frames[0].save(
            path, save_all=True, append_images=self.frames[1:],
            loop=0, duration=duration_ms,
        )
        print(f"Saved GIF ({len(self.frames)} frames): {path}")
 
    def save_slide_png(self, path="velocity_final_frame.png"):
        if not self.frames:
            print("No frames."); return
        last = self.frames[-1]
        w, h = last.size
        last.resize((w * 2, h * 2), Image.LANCZOS).save(path, dpi=(300, 300))
        print(f"Saved slide PNG: {path}")
 