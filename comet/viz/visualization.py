import io
import os

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
 
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
 
        x_min, x_max = float(X_2d[:, 0].min()), float(X_2d[:, 0].max())
        y_min, y_max = float(X_2d[:, 1].min()), float(X_2d[:, 1].max())
 
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
 