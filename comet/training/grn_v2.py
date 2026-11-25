"""COMET v2 GRN dictionary: the v28 "T3_auto" recipe (trajectory_benchmarking/v28/lib/t28.py), made self-contained so that
`python train.py` fits it from the same inputs as before. Selected by config.GRN_MODEL = "v2" (training_loop.run_comet).

Model (per cell i; y = gene-wise standardised log expression, u = standardised TF log expression):
    y_hat_i = b_base + sum_d pi_id b_d + W_base u_i + sum_d pi_id W_d u_i
    W_base = tanh(P_base) * M_base,  W_d = tanh(P_d) * M_d       (M_d: motif prior x module licensing; no self edges)
    pi_i = softmax(enc(x_i))                                     (CellStateEncoder, warm-started to the soft targets omega)
T3 training recipe (unchanged from v28):
    - count splitting (eps .5): each epoch counts -> halves A / B; encoder and TF inputs read A, masked targets are B
    - W_base dropout .3: each step, with probability .3, W_base and b_base are left out of the forward pass
    - trajectory-anchored soft targets omega: k-means (D = 4) on standardised (tau, F)
    - flow penalty .1: ||pi_i - pi_j||^2, j drawn from the transition matrix P
    - published weights / schedule: MASK 2/1, SPARSE 5e-3, ANCHOR 1/.5 annealed, AdamW 5e-4 cosine, 80 epochs,
      best-epoch restore on an internal held-out-molecule validation split
    - auto init: fit cold (soft-LFC atoms) and warm (closed-form ridge per module), keep the lower validation MSE

Inputs that v28 took from the benchmarking pipelines are built here from the run's own trajectory:
    - P:   scVelo velocity graph of V_pred (30 neighbours in the state space), .8 P_velocity + .2 uniform kNN
    - tau: rank of the committor q;  F: absorption probabilities into the sink regions (connected components of
           traj sink_mask cells in the kNN graph); single sink region -> F = committor root -> sink
    - licensing: edge (TF -> g) is licensed in module d if module d's omega-weighted mean accessibility of g
                 (b.ga_matched) is >= .5 x its maximum over modules (v20 jc.gating rule; gene-level accessibility here,
                 v28 used per-edge linked-peak curves where a dataset had them)
    - counts: raw counts kept by the loaders (b.counts, b.lib)
"""
import copy, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

T3 = dict(cs=.5, p_base=.3, lam_flow=.1, D=4, epochs=80, phase1=40, lr=5e-4, seed=0, patience=10, mask_frac=.4,
          init="auto", knn=30, mix=.2, min_det=.05, min_sink=10)
W1 = dict(MASK=2.0, SPARSE=5e-3, ANCHOR=1.0)
W2 = dict(MASK=1.0, SPARSE=5e-3, ANCHOR=.5)


def zs(A):
    A = np.asarray(A, np.float32); return ((A - A.mean(0)) / (A.std(0) + 1e-8)).astype(np.float32)


# ------------------------------------------------------------------ trajectory inputs
def transition_matrix(Y, V, Z, knn=30, mix=.2):
    """-> P (N, N) row-stochastic, A (N, N) symmetric kNN adjacency"""
    import anndata as ad, scanpy as sc, scvelo as scv
    N = len(Z); a = ad.AnnData(np.asarray(Y, np.float32)); a.layers["Ms"] = a.X.copy()
    a.layers["velocity"] = np.asarray(V, np.float32); a.obsm["X_rep"] = np.asarray(Z, np.float32)
    sc.pp.neighbors(a, n_neighbors=knn, use_rep="X_rep", random_state=0)
    scv.settings.verbosity = 0; scv.tl.velocity_graph(a, vkey="velocity", xkey="Ms", n_jobs=1)
    Pv = sp.csr_matrix(scv.utils.get_transition_matrix(a, vkey="velocity"))
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=knn + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    A = sp.csr_matrix((np.ones(N * knn), (np.repeat(np.arange(N), knn), idx.ravel())), shape=(N, N))
    A = ((A + A.T) > 0).astype(np.float64).tocsr(); U = sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
    rs = np.asarray(Pv.sum(1)).ravel(); Pv = sp.diags(np.where(rs > 0, 1 / np.maximum(rs, 1e-300), 0)) @ Pv
    P = (1 - mix) * Pv + mix * U
    if (rs <= 0).any(): P = P + sp.diags((rs <= 0) * (1 - mix)) @ U
    P = sp.csr_matrix(P); P.sort_indices(); return P, A


def absorb(P, sets):
    """F (N, K) = absorption probabilities into the K disjoint cell sets"""
    N = P.shape[0]; K = len(sets); bnd = np.zeros(N, bool); FB = np.zeros((N, K))
    for k, s in enumerate(sets): bnd[s] = True; FB[s, k] = 1
    I = np.where(~bnd)[0]; Bn = np.where(bnd)[0]; P = sp.csr_matrix(P)
    F = FB.copy(); F[I] = splu((sp.eye(len(I)) - P[I][:, I]).tocsc()).solve(np.asarray(P[I][:, Bn] @ FB[Bn]))
    return np.clip(F, 0, 1)


def fates(P, A, q, root_mask, sink_mask, min_sink=10):
    """sink regions = connected components (in A) of the sink cells with >= min_sink cells -> F (N, K), sets"""
    from scipy.sparse.csgraph import connected_components
    s = np.where(sink_mask)[0]
    n, lab = connected_components(A[s][:, s], directed=False)
    sets = [s[lab == k] for k in range(n) if (lab == k).sum() >= min_sink] or [s]
    if len(sets) == 1:
        root = np.where(root_mask)[0] if root_mask.any() else np.where(q <= np.quantile(q, .05))[0]
        root = np.setdiff1d(root, sets[0]); return absorb(P, [root, sets[0]])[:, [1]], sets
    return absorb(P, sets), sets


def traj_omega(F, tau, D):
    """k-means (D, seed 0) on standardised (tau, F); omega_id = softmax_d(-||(tau,F)_i - c_d||^2 / T), T = .5 median d^2"""
    from sklearn.cluster import KMeans
    feat = np.c_[tau, F]; fz = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    km = KMeans(D, random_state=0, n_init=10).fit(fz)
    d2 = ((fz[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1); T_ = np.median(d2) * .5 + 1e-8
    q = np.exp(-(d2 - d2.min(1, keepdims=True)) / T_); return (q / q.sum(1, keepdims=True)).astype(np.float32)


def licence(ga, om, tgt):
    """licensed[d, e] for edge e with target tgt[e] (gene-level accessibility, jc.gating threshold .5 x max)"""
    Abar = (om.T.astype(np.float64) @ np.asarray(ga, np.float64)) / (om.sum(0)[:, None] + 1e-12)       # (D, G)
    mx = Abar.max(0, keepdims=True); lic = (Abar >= .5 * mx) | (mx <= 0)
    return lic[:, tgt]


# ------------------------------------------------------------------ initialisations
def fit_dict(U, Y, pi, lic, src, tgt, alpha_rel=.1):
    """closed-form warm start: per-module weighted ridge (with intercept) of each target on its licensed TF parents"""
    G, T = Y.shape[1], U.shape[1]; D = pi.shape[1]; Wc = np.zeros((D, G, T), np.float32); bc = np.zeros((D, G), np.float32)
    Ud = U.astype(np.float64); Yd = Y.astype(np.float64)
    for d in range(D):
        w = pi[:, d].astype(np.float64); ws = w / w.sum(); um = ws @ Ud; ym = ws @ Yd; Uc = Ud - um; neff = w.sum()
        Gm = (Uc * w[:, None]).T @ Uc; Bm = (Uc * w[:, None]).T @ (Yd - ym); a = alpha_rel * neff
        e = np.where(lic[d])[0]; e = e[np.argsort(tgt[e], kind="stable")]; tg = tgt[e]
        cuts = np.r_[0, np.where(np.diff(tg))[0] + 1, len(tg)]; bc[d] = ym
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            if s1 <= s0: continue
            g = tg[s0]; p = src[e[s0:s1]]
            beta = np.linalg.solve(Gm[np.ix_(p, p)] + a * np.eye(len(p)), Bm[p, g]); Wc[d, g, p] = beta; bc[d, g] = ym[g] - um[p] @ beta
    mass = pi.sum(0) / pi.sum(); Ab = np.tensordot(mass, Wc, 1); bb = mass @ bc
    return Ab.astype(np.float32), (Wc - Ab[None]).astype(np.float32), bb.astype(np.float32), (bc - bb[None]).astype(np.float32)


def lfc_init(logX, om, Md):
    """published soft-LFC atom init (model.grn.init_atoms_from_soft_targets) on log expression and module masks"""
    out = np.zeros_like(Md)
    for d in range(om.shape[1]):
        w = om[:, d]; mi = (logX * w[:, None]).sum(0) / (w.sum() + 1e-8); mo = (logX * (1 - w)[:, None]).sum(0) / ((1 - w).sum() + 1e-8)
        L = (np.log1p(mi) - np.log1p(mo))[:, None] * Md[d]; out[d] = L / (np.abs(L).max(0, keepdims=True) + 1e-8) * .3
    return out


def count_split(C, lib, rows, eps, seed):
    """binomial split of the counts of `rows` -> (A, B) standardised log halves, each normalised by its expected library"""
    C = C[rows]; rng = np.random.default_rng(seed); A = C.copy()
    A.data = rng.binomial(C.data.astype(np.int64), eps).astype(np.float64); Bh = C - A; out = []
    for H, e in ((A, eps), (Bh, 1 - eps)):
        H = (sp.diags(1e4 / (e * lib[rows])) @ H).tocsr(); H.data = np.log1p(H.data); out.append(zs(H.toarray()))
    return out


# ------------------------------------------------------------------ training (t28.train, T3 switches)
def _train(d, c, warm, log=print):
    import torch, torch.nn.functional as Fnn
    from model.encoder import CellStateEncoder
    torch.manual_seed(c["seed"]); t0 = time.time()
    N, G, T = d["N"], d["G"], d["T"]; rows = np.arange(N)
    perm = np.random.default_rng(28000 + c["seed"]).permutation(rows); nv = max(20, len(rows) // 10)
    val = np.sort(perm[:nv]); tr = np.sort(perm[nv:])
    om, Mb, Md, Y, U, C, lib = d["om"], d["Mb"], d["Md"], d["Y"], d["U"], d["C"], d["lib"]; D = om.shape[1]
    tfg = torch.tensor(d["tfg"], dtype=torch.long); cs = c["cs"] if C is not None else None
    if warm:
        Ab0, Ad0, bb0, bd0 = fit_dict(U[tr], Y[tr], om[tr], d["lic"], d["src"], d["tgt"])
    else:
        Ad0 = lfc_init(d["logX"][tr], om[tr], Md); Ab0 = np.zeros((G, T), np.float32); bb0 = Y[tr].mean(0); bd0 = np.zeros((D, G), np.float32)
    clip = lambda a: np.arctanh(np.clip(a, -.95, .95)).astype(np.float32)
    Mbt, Mdt = torch.tensor(Mb), torch.tensor(Md)
    pAb = torch.nn.Parameter(torch.tensor(clip(Ab0))); pAd = torch.nn.Parameter(torch.tensor(clip(Ad0)))
    pbb = torch.nn.Parameter(torch.tensor(bb0, dtype=torch.float32)); pbd = torch.nn.Parameter(torch.tensor(bd0, dtype=torch.float32))
    fAb = lambda: torch.tanh(pAb) * Mbt; fAd = lambda: torch.tanh(pAd) * Mdt
    enc = CellStateEncoder(G=G, D=D)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()): enc.init_from_soft_targets(Y[tr], om[tr], n_epochs=40)
    params = [pAb, pAd, pbb, pbd]; params_all = params + list(enc.parameters())
    qt = torch.tensor(om); Yt = torch.tensor(Y)
    P = d["P"]; Pind, Pptr, Pdat = P.indices, P.indptr, P.data; cums = [None] * N; rng = np.random.default_rng(c["seed"] + 1)

    def nbr_of(bi):
        out = np.empty(len(bi), int)
        for k, i in enumerate(bi):
            a, b = Pptr[i], Pptr[i + 1]
            if b <= a: out[k] = i; continue
            if cums[i] is None: cc = np.cumsum(Pdat[a:b]); cums[i] = cc / cc[-1]
            out[k] = Pind[a + min(np.searchsorted(cums[i], rng.random()), b - a - 1)]
        return out

    if cs: vA, vB = count_split(C, lib, val, .5, 99)
    else: vA = vB = Y[val]
    vAt, vBt = torch.tensor(vA), torch.tensor(vB)

    def forward(xtf, mix, drop_base=False):
        out = mix @ pbd + torch.einsum("bt,dgt,bd->bg", xtf, fAd(), mix)
        return out if drop_base else out + xtf @ fAb().T + pbb

    def val_mse():
        enc.eval()
        with torch.no_grad(): return float(((forward(vAt[:, tfg], enc(vAt)) - vBt) ** 2).mean())

    opt = torch.optim.AdamW(params_all, lr=c["lr"]); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"], eta_min=1e-5)
    gen = torch.Generator(); gen.manual_seed(c["seed"]); BS = 256; nb = int(np.ceil(len(tr) / BS))
    st1 = max(1, min(nb, int(np.ceil(2048 / BS)))); st2 = max(1, min(nb, int(np.ceil(16384 / BS))))
    hist = []; best = None; since = 0; trpos = -np.ones(N, int); trpos[tr] = np.arange(len(tr))
    for e in range(c["epochs"]):
        ph = 1 if e < c["phase1"] else 2; Wt = W1 if ph == 1 else W2; steps = st1 if ph == 1 else st2; aw = max(.2, 1. - .8 * e / c["epochs"])
        if cs: XA, XB = count_split(C, lib, tr, cs, 1000 * c["seed"] + e); XAt, XBt = torch.tensor(XA), torch.tensor(XB)
        enc.train(); el = []; order = torch.randperm(len(tr), generator=gen).numpy(); pos = 0
        for s in range(steps):
            if pos >= len(tr): order = torch.randperm(len(tr), generator=gen).numpy(); pos = 0
            bi = tr[order[pos:pos + BS]]; pos += BS; nbi = len(bi)
            mask = torch.rand(nbi, G) < c["mask_frac"]
            if cs:
                k = torch.tensor(trpos[bi]); xin = XAt[k]; tgt = XBt[k]; xtf = xin[:, tfg].clone()
            else:
                xin = Yt[bi]; tgt = xin; xtf = xin[:, tfg].clone() * (~mask[:, tfg]).float()
            mix = enc(xin * (~mask).float())
            drop = c["p_base"] > 0 and rng.random() < c["p_base"]
            l_mask = Fnn.mse_loss(forward(xtf, mix, drop)[mask], tgt[mask])
            l_sp = fAd().abs().mean() + fAb().abs().mean()
            qb = qt[bi]; l_anc = (qb * (torch.log(qb + 1e-8) - torch.log(mix + 1e-8))).sum(1).mean()
            loss = Wt["MASK"] * l_mask + Wt["SPARSE"] * l_sp + Wt["ANCHOR"] * aw * l_anc
            if c["lam_flow"] > 0:
                js = nbr_of(bi); xj = Yt[js].clone()
                if cs:
                    kj = trpos[js]; okj = kj >= 0
                    if okj.any(): xj[torch.tensor(okj)] = XAt[torch.tensor(kj[okj])]
                loss = loss + c["lam_flow"] * ((mix - enc(xj)) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params_all, 1.0); opt.step(); el.append(float(l_mask))
        sch.step(); vl = val_mse(); hist.append((e, float(np.mean(el)), vl))
        if e % 20 == 0 or e == c["epochs"] - 1: log(f"  [{'warm' if warm else 'cold'}] epoch {e:03d}  train {np.mean(el):.4f}  val {vl:.4f}")
        if best is None or vl < best[0]:
            best = (vl, e, [p.detach().clone() for p in params], copy.deepcopy(enc.state_dict())); since = 0
        else: since += 1
        if since >= c["patience"] and e >= c["phase1"]: break
    with torch.no_grad():
        for p, v in zip(params, best[2]): p.copy_(v)
    enc.load_state_dict(best[3]); enc.eval()
    with torch.no_grad():
        pi = enc(Yt).numpy(); W = dict(Ab=fAb().numpy(), Ad=fAd().numpy(), bb=pbb.numpy().copy(), bd=pbd.numpy().copy(), pi=pi.astype(np.float32))
    diag = dict(init="warm" if warm else "cold", best_epoch=int(best[1]), best_val=float(best[0]), epochs_run=len(hist),
                count_split=bool(cs), train_s=time.time() - t0)
    return W, diag, enc, np.array(hist)


def predict(W, U, pi):
    V = W["bb"][None] + pi @ W["bd"] + U @ W["Ab"].T
    for d in range(W["Ad"].shape[0]): V += pi[:, [d]] * (U @ W["Ad"][d].T)
    return V


def train_grn_v2(b, Zst, traj, cfg=None, log=print):
    """Fit COMET v2 on Bundle b. traj: utils.trajectory.final_trajectory output (V, q, root_mask, sink_mask).
    Returns the same keys as training_loop.train_grn (A_* over all b.A_tfs; TFs below min_det have zero columns),
    plus b_base, b_dict, fates, omega and diag."""
    c = dict(T3); c.update(cfg or {})
    logX = np.asarray(b.X, np.float32); N, G = logX.shape; det = (logX > 0).mean(0)
    tcols = np.where(det[b.tf_indices_in_G] >= c["min_det"])[0]; tfg = np.asarray(b.tf_indices_in_G)[tcols]; T = len(tcols)
    tg, tt = np.nonzero(np.asarray(b.M)[:, tcols] > 0); ok = tfg[tt] != tg; tgt, src = tg[ok], tt[ok]
    Y = zs(logX); U = Y[:, tfg]
    C = getattr(b, "counts", None); lib = getattr(b, "lib", None)
    if C is None: log("  [grn v2] WARNING: loader kept no raw counts -> no count splitting (published masked targets)")
    else: C = sp.csr_matrix(C).astype(np.float64); lib = np.asarray(lib, np.float64).ravel()
    q = np.asarray(traj["q"], np.float64); tau = (np.argsort(np.argsort(q)) / (N - 1)).astype(np.float64)
    log(f"  [grn v2] N={N} G={G} TFs={T}/{len(b.A_tfs)} edges={len(tgt)}; building transition matrix")
    P, A = transition_matrix(Y, traj["V"], Zst, c["knn"], c["mix"])
    F, sets = fates(P, A, q, np.asarray(traj["root_mask"], bool), np.asarray(traj["sink_mask"], bool), c["min_sink"])
    om = traj_omega(F, tau, c["D"]); lic = licence(b.ga_matched, om, tgt)
    Md = np.zeros((c["D"], G, T), np.float32)
    for d_ in range(c["D"]): Md[d_, tgt[lic[d_]], src[lic[d_]]] = 1
    Md[:, tfg, np.arange(T)] = 0; Mb = (Md.sum(0) > 0).astype(np.float32)
    log(f"  [grn v2] sink regions {[len(s) for s in sets]}; omega mass {np.round(om.mean(0), 2).tolist()}; licensed {lic.mean():.2f}")
    d = dict(N=N, G=G, T=T, tfg=tfg, src=src, tgt=tgt, om=om, lic=lic, Mb=Mb, Md=Md, Y=Y, U=U, logX=logX, C=C, lib=lib, P=P)
    inits = {"auto": (False, True), "cold": (False,), "warm": (True,)}[c["init"]]
    fits = [_train(d, c, w, log) for w in inits]
    W, diag, enc, hist = min(fits, key=lambda f: f[1]["best_val"])
    diag["auto"] = {f[1]["init"]: f[1]["best_val"] for f in fits}
    log(f"  [grn v2] kept {diag['init']} (best epoch {diag['best_epoch']}, val {diag['best_val']:.4f}; {diag['auto']})")
    Tall = len(b.A_tfs); A_base = np.zeros((G, Tall), np.float32); A_dict = np.zeros((c["D"], G, Tall), np.float32)
    A_base[:, tcols] = W["Ab"]; A_dict[:, :, tcols] = W["Ad"]; pi = W["pi"]
    return dict(grn=None, cell_encoder=enc, mix=pi, A_base=A_base, A_dict=A_dict,
                A_learned=A_base + np.tensordot(pi.mean(0), A_dict, 1), steady_state=predict(W, U, pi).astype(np.float32),
                loss_history=hist[:, 2], b_base=W["bb"], b_dict=W["bd"], fates=F.astype(np.float32), omega=om, diag=diag)
