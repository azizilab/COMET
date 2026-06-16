import heapq

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class MarkovProgressionField(nn.Module):
    """
    Phase 3 learned object — D scalars total.

    BIPOLAR ψ DESIGN (replaces old P_HIGH ≥ 0 scheme):
      P[origin_atom] = -1  (fixed — the unique global minimum)
      P[d]           = +1 + ALPHA * tanh(eps_raw[d])   for d != origin_atom

    psi_raw(i) = mix_w[i] @ P

    With the old (0, P_HIGH) scheme, if origin atom has mean mixing weight ~0.07,
    psi spans only ~0.07 * P_HIGH per cell — all cells cluster near P_HIGH * 0.93
    with tiny variation. With bipolar (-1, +1), cells with any origin-atom weight
    get a -1 contribution, creating a real gradient from origin territory (psi ≈ -1)
    to branch territory (psi ≈ +1). Dynamic range scales with mix_w contrast.

    ALPHA = 0.15 keeps all downstream atoms at P ∈ [+0.85, +1.15] — branches
    sit at roughly equal height so noisy discordance differences among downstream
    atoms don't produce spurious inter-branch ordering.
    """
    ALPHA = 0.15

    def __init__(self, D: int, origin_atom: int):
        super().__init__()
        self.D = D
        self.origin_atom = origin_atom
        downstream = [d for d in range(D) if d != origin_atom]
        self.register_buffer("downstream_atoms", torch.tensor(downstream, dtype=torch.long))
        # eps_raw starts at 0 → all downstream atoms at P = +1.0
        self.eps_raw = nn.Parameter(torch.zeros(D - 1))

    def atom_progression(self) -> torch.Tensor:
        """Returns P (D,): P[origin_atom] = -1, P[downstream] ≈ +1."""
        P = torch.full((self.D,), -1.0, device=self.eps_raw.device)
        P[self.downstream_atoms] = 1.0 + self.ALPHA * torch.tanh(self.eps_raw)
        return P

    def psi_raw(self, mix_weights: torch.Tensor) -> torch.Tensor:
        """mix_weights (N, D) -> psi_raw (N,)"""
        return mix_weights @ self.atom_progression()


def compute_global_disc_order(grn, X_full_t, gene_access_full_t, encoder,
                               top_k_genes=100):
    """
    Compute a stable, global atom discordance ordering from the entire dataset.
    Called once after Phase 2 (when atoms + encoder are stable).

    HIGH disc = ATAC open, RNA silent = UPSTREAM (progenitor).
    LOW  disc = RNA expressed, chromatin closing = DOWNSTREAM (committed).

    Returns:
      direction_matrix (D, D): relu(disc[a] - disc[b])
      disc (D,): per-atom discordance score
      dom_atom_full (N,): dominant atom per cell
    """
    D = grn.D
    grn.eval(); encoder.eval()
    with torch.no_grad():
        mix_w_full = encoder(X_full_t, temperature=0.1)               # (N, D)
        w_T        = mix_w_full.T                                      # (D, N)
        w_sum      = w_T.sum(dim=1, keepdim=True).clamp(min=1e-8)

        atom_gene_str = grn.A_dict.abs().sum(dim=2)                    # (D, G)
        centroids_rna = (w_T @ X_full_t) / w_sum                      # (D, G)

        ga    = gene_access_full_t                                     # (N, G)
        ga_01 = (ga - ga.min(dim=0).values) / \
                (ga.max(dim=0).values - ga.min(dim=0).values + 1e-8)

        disc = torch.zeros(D)
        for d in range(D):
            top_idx = atom_gene_str[d].topk(top_k_genes).indices       # (K,)
            cell_w  = w_T[d] / (w_T[d].sum() + 1e-8)                  # (N,)
            atac_d  = (cell_w @ ga_01[:, top_idx])                     # (K,)
            rna_d   = (cell_w @ X_full_t[:, top_idx])                  # (K,)
            rna_max = centroids_rna[:, top_idx].max(dim=0).values.clamp(min=1e-8)
            disc[d] = (atac_d - rna_d / rna_max).mean()

        dom_atom_full = mix_w_full.argmax(dim=1)                       # (N,)

    direction_matrix = torch.relu(disc.unsqueeze(1) - disc.unsqueeze(0))  # (D, D)

    print("Global discordance scores (higher = more upstream):")
    for d in range(D):
        print(f"  Atom {d}: disc={disc[d]:.4f}")
    print(f"  Upstream→downstream order: {disc.argsort(descending=True).tolist()}")
    print(f"  Spread: {disc.max() - disc.min():.4f}")

    grn.train(); encoder.train()
    return direction_matrix.detach(), disc.detach(), dom_atom_full.detach()


def psi_polarity_loss(psi: torch.Tensor, dom_atom_full: torch.Tensor,
                       origin_atom: int, margin: float = 1.0) -> torch.Tensor:
    """
    Hinge loss: cells whose dominant atom is origin_atom should have psi
    at least margin below cells whose dominant atom is anything else.

    With bipolar ψ the theoretical gap between a pure-origin and a pure-branch
    cell is 2.0. margin=1.0 is half the maximum gap: easy to satisfy for
    well-separated atoms but still trains eps_raw when mixing weights are noisy.
    """
    is_origin = dom_atom_full == origin_atom
    if is_origin.sum() == 0 or (~is_origin).sum() == 0:
        return torch.zeros((), device=psi.device)
    gap = psi[~is_origin].mean() - psi[is_origin].mean()
    return torch.relu(margin - gap)


def psi_fine_order_loss(psi: torch.Tensor, dom_atom_full: torch.Tensor,
                         direction_matrix: torch.Tensor, D: int,
                         margin: float = 0.1) -> torch.Tensor:
    """
    Secondary signal: for downstream-atom pairs (a, b) where
    direction_matrix[a, b] > 0 (a is more upstream than b), nudge
    mean(psi | atom b) above mean(psi | atom a) by margin, weighted
    by the discordance gap.

    direction_matrix[a, b] = relu(disc[a] - disc[b]): positive means a is
    more upstream, so b should have higher psi.

    This is intentionally disabled (W3_FINE_ORDER = 0) by default: discordance
    differences among downstream atoms are within noise and ordering them creates
    asymmetric ψ that causes cross-branch arrows.
    """
    terms = []
    for a in range(D):
        for b in range(D):
            if a == b:
                continue
            gap_disc = float(direction_matrix[a, b].item())
            if gap_disc <= 1e-4:
                continue
            ma = dom_atom_full == a
            mb = dom_atom_full == b
            if ma.sum() == 0 or mb.sum() == 0:
                continue
            terms.append(gap_disc * torch.relu(margin - (psi[mb].mean() - psi[ma].mean())))
    if not terms:
        return torch.zeros((), device=psi.device)
    return torch.stack(terms).mean()


def build_2d_neighbor_index(X_2d: np.ndarray, k: int) -> np.ndarray:
    """
    Build a kNN graph in 2D space for arrow geometry only.

    Arrow displacement vectors X_2d[j] - X_2d[i] are geometrically meaningful
    only when j is found in the same 2D space as the arrows. Latent-space
    neighbors can be geometrically far away in 2D, producing spurious arrows.
    Do NOT use this index for ψ smoothing — use latent-space kNN there.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_2d)   # +1 because self is included
    _, idx = nbrs.kneighbors(X_2d)
    return idx[:, 1:]   # exclude self


def compute_markov_velocity_2d(psi: np.ndarray,
                                X_2d: np.ndarray,
                                neighbor_idx_2d,
                                n_hops: int = 4,
                                beta: float = 0.4,
                                eps: float = 1e-6):
    """
    Build the 2D Markov velocity field from psi using the ReLU kernel.

    Kernel: T[i→j] ∝ ReLU(ψ(j)−ψ(i)) — strictly zero for downhill neighbours.

    The softmax kernel cancels ψ(i) algebraically (exp(-γψ(i)) appears in both
    numerator and denominator), making T independent of the reference cell. The
    ReLU kernel is strictly zero for downhill neighbours, shrinks to zero when ψ
    is flat (committed cells), and is large when ψ rises sharply (transition zones),
    implementing true gradient ascent of ψ.

    T is row-normalised to a proper row-stochastic matrix so multi-hop diffusion
    V_{h+1} = (1-β)V_h + β T V_h is a valid convex combination. The old code
    used unnormalised T, causing V to shrink to zero after a few hops.

    neighbor_idx_2d must be a 2D-space kNN index (from build_2d_neighbor_index),
    NOT the latent-space index used for ψ smoothing.
    """
    N   = psi.shape[0]
    idx = neighbor_idx_2d.numpy() if torch.is_tensor(neighbor_idx_2d) else np.asarray(neighbor_idx_2d)

    rows_list, cols_list, vals_list = [], [], []
    V_raw = np.zeros((N, 2), dtype=np.float32)

    for i in range(N):
        nbr_idx   = idx[i]
        delta_psi = psi[nbr_idx] - psi[i]
        w         = np.maximum(delta_psi, 0.0)
        w_sum     = w.sum()

        if w_sum < eps:
            continue   # local psi-maximum -> V[i] stays 0

        w_norm = w / w_sum
        disp   = X_2d[nbr_idx] - X_2d[i]
        V_raw[i] = (w_norm[:, np.newaxis] * disp).sum(axis=0)

        for jj, j in enumerate(nbr_idx):
            if w_norm[jj] > 0:
                rows_list.append(i); cols_list.append(int(j)); vals_list.append(float(w_norm[jj]))

    # Build row-stochastic transition matrix (row sums = 1 for cells with outgoing flow)
    T_np = csr_matrix((vals_list, (rows_list, cols_list)), shape=(N, N), dtype=np.float64)
    row_sums = np.asarray(T_np.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    T_norm = csr_matrix(T_np.multiply(1.0 / row_sums[:, np.newaxis]))

    # Multi-hop diffusion along T_norm — spreads directionally consistent signal
    # into sparse CLOUD regions. beta controls the blend per hop.
    V_diff = V_raw.copy()
    for _ in range(n_hops):
        V_diff = (1.0 - beta) * V_diff + beta * (T_norm @ V_diff)

    return V_raw, V_diff.astype(np.float32), T_norm.tocsr()


def infer_velocity_pseudotime(V_dec2: np.ndarray,
                               psi_arr: np.ndarray,
                               neighbor_idx_2d: np.ndarray,
                               cell_types: np.ndarray = None,
                               root_pct: float = 0.05) -> np.ndarray:
    """
    Pseudotime via Dijkstra shortest-path on the ψ-gradient graph.

    WHY NOT MFPT ON T_markov:
    - T_markov is NOT doubly stochastic: zero rows at ψ-maxima make
      (I - T) pt = 1 singular.
    - The source-score heuristic (out/in degree) fails for the same reason:
      cells at ψ-maxima have zero out-degree → neighbors' in-degree inflated
      → wrong roots spread across CLOUD, GMP, HSC, MEP simultaneously.

    ALGORITHM (principled, label-free):
    1. Root detection: lowest root_pct% of ψ. Guaranteed correct because
       P[origin_atom] = -1 and ψ(origin cells) ≈ -1 by construction.
       No degree heuristic needed.
    2. Directed graph: edge i→j iff ψ(j) > ψ(i).
       Cost = (ψ(j) - ψ(i)) / dist(i, j) — same topology as ReLU velocity field.
    3. Multi-source Dijkstra from all root cells simultaneously.
       Shortest-path distance from roots = pseudotime.
       Handles disconnected subgraphs naturally; no singularity issues.
    4. Normalise to [0, 1] after clipping the 99th percentile.
    """
    N   = V_dec2.shape[0]
    idx = neighbor_idx_2d

    psi_thresh = np.percentile(psi_arr, root_pct * 100)
    root_mask  = psi_arr <= psi_thresh
    root_idx   = np.where(root_mask)[0]
    n_roots    = root_mask.sum()

    print(f"Root cells (lowest {root_pct*100:.1f}% of ψ): n={n_roots}")
    print(f"  ψ threshold = {psi_thresh:.4f}  "
          f"(global range: [{psi_arr.min():.4f}, {psi_arr.max():.4f}])")
    if cell_types is not None and len(cell_types) == N:
        print(f"  Root cell types: {np.unique(cell_types[root_idx])}")

    # Build directed graph: uphill edges only (same as ReLU kernel)
    graph = [[] for _ in range(N)]
    for i in range(N):
        for j in idx[i]:
            delta = psi_arr[j] - psi_arr[i]
            if delta > 0:
                dist_ij = np.linalg.norm(V_dec2[j] - V_dec2[i]) + 1e-8
                cost    = delta / dist_ij   # cost ∝ ψ gradient magnitude
                graph[i].append((cost, int(j)))

    INF  = float('inf')
    dist = np.full(N, INF, dtype=np.float64)
    dist[root_idx] = 0.0
    heap = [(0.0, int(r)) for r in root_idx]
    heapq.heapify(heap)

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist[u]:
            continue   # stale entry
        for cost_uv, v in graph[u]:
            d_v = dist[u] + cost_uv
            if d_v < dist[v]:
                dist[v] = d_v
                heapq.heappush(heap, (d_v, v))

    finite_mask = dist < INF
    if not finite_mask.all():
        n_disc = (~finite_mask).sum()
        print(f"  Warning: {n_disc} cells not reachable — assigning max pseudotime.")
        dist[~finite_mask] = dist[finite_mask].max() * 1.05

    pt = np.clip(dist, 0, np.percentile(dist, 99))
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
    return pt.astype(np.float32)
