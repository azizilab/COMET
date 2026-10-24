import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from sklearn.decomposition import PCA

from config import (
    D_ATOMS, N_EPOCHS, PHASE1_EPOCHS, BATCH_SIZE, GEO_TAU_SCALE,
    TARGET_CELLS_PER_EPOCH, TARGET_CELLS_PER_EPOCH_PHASE2, FORCE_RERUN, ORIGIN_SIGN,
    STATE_SPACE, VEL_MOMENT_KNN, K_MOMENT, RANDOM_STATE, W1, W2, LD_HIDDEN, LR_GRN, TRAIN_GRN, GRN_TILT_LAMBDA,
)
from utils.common import zscore_cols
from utils.committor import moment_operator
from utils.seed import set_seed
from utils.trajectory import final_trajectory, regulatory_consistency_tilt
from model.grn import GRNDictionary, select_archetypes_fps, geometric_soft_targets, init_atoms_from_soft_targets
from model.encoder import CellStateEncoder
from model.ld_vae import LinearDecoder
from losses.losses import masked_reconstruction_loss

_TRAJ_FILES = ["V_pred.npy", "committor_q.npy", "root_mask.npy", "sink_mask.npy", "phi.npy",
               "traj_lead.npy", "traj_w.npy"]
_GRN_FILES = ["mix_weights.npy", "A_learned.npy", "A_dict.npy"]


def _save(out_dir, **arrays):
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)


def _load_cached_run(out_dir, train):
    """Results dict from a finished run's saved arrays, or None if the run is incomplete."""
    need = _TRAJ_FILES + (_GRN_FILES if train else [])
    if not all(os.path.exists(os.path.join(out_dir, f)) for f in need):
        return None
    L = lambda f: np.load(os.path.join(out_dir, f)) if os.path.exists(os.path.join(out_dir, f)) else None
    return dict(V_pred=L("V_pred.npy"), q_commit=L("committor_q.npy"),
                root_mask=L("root_mask.npy").astype(bool), sink_mask=L("sink_mask.npy").astype(bool),
                phi=L("phi.npy"), traj_lead=L("traj_lead.npy"), traj_w=L("traj_w.npy"),
                mix=L("mix_weights.npy"), A_learned=L("A_learned.npy"), A_dict=L("A_dict.npy"),
                grn=None, cell_encoder=None)


def state_space(b):
    """The latent the trajectory (and the GRN's archetypes) live in: config.STATE_SPACE[b.name]."""
    if STATE_SPACE.get(b.name, "decipher") == "pca":
        return PCA(n_components=min(30, b.G - 1, b.N - 1),
                   random_state=RANDOM_STATE).fit_transform(zscore_cols(b.X)).astype(np.float32)
    return b.Z_dec


def train_grn(b, Zst):
    """Train the GRN dictionary (shared base + D_ATOMS residual atoms, mixed per cell by the cell encoder).

    Phase 1 and phase 2 differ only in the loss weights (config.W1 / W2) and in how many cells an epoch sees.
    Returns the trained modules and the arrays saved by run_comet.
    """
    print(f"  GRN training: {PHASE1_EPOCHS} phase-1 epochs + {N_EPOCHS - PHASE1_EPOCHS} phase-2 epochs")

    # ---- archetypes + geometric soft mixing targets ----
    arch_idx = select_archetypes_fps(Zst, D_ATOMS, seed=0)
    q_target = geometric_soft_targets(Zst, arch_idx, tau_scale=GEO_TAU_SCALE)

    # ---- ATAC accessibility priors ----
    G, T = b.G, b.T
    TF_access = np.zeros(T, np.float32)
    for i, tf in enumerate(b.A_tfs):
        if tf in b.sub_g2i: TF_access[i] = float(b.ga_matched[:, b.sub_g2i[tf]].mean())
    lo, hi = TF_access.min(), TF_access.max(); TF_access = (TF_access - lo) / (hi - lo + 1e-8)
    gene_access_t = torch.tensor(b.gene_access, dtype=torch.float32)

    # ---- tensors + loader ----
    # TF inputs are standardised so a GRN weight is "RNA per SD of TF" (comparable across TFs); in raw units
    # |A[:, t]| scales like 1/expression(t) and edge rankings invert.
    X_TF = (b.X_TF - b.X_TF.mean(0)) / (b.X_TF.std(0) + 1e-8)
    X_t    = torch.tensor(b.X,  dtype=torch.float32)
    X_TF_t = torch.tensor(X_TF, dtype=torch.float32)
    q_t    = torch.tensor(q_target, dtype=torch.float32)
    tf_indices_in_G = torch.tensor(b.tf_indices_in_G, dtype=torch.long)
    dataset = TensorDataset(X_t, X_TF_t, torch.arange(b.N))
    loader_gen = torch.Generator(); loader_gen.manual_seed(RANDOM_STATE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        sampler=RandomSampler(dataset, replacement=False, generator=loader_gen), drop_last=False)

    # ---- models ----
    grn = GRNDictionary(G=G, T=T, M=b.M, D=D_ATOMS)
    grn.TF_access = torch.tensor(TF_access, dtype=torch.float32); grn.Gene_access = gene_access_t.clone()
    # positive, typical-magnitude TF input for diversity_loss (the standardised X_TF has mean 0)
    grn.mean_tf = torch.tensor(np.abs(X_TF).mean(0), dtype=torch.float32)
    grn.init_from_atoms(init_atoms_from_soft_targets(b.X, q_target, b.M, b.A_tfs))
    cell_encoder = CellStateEncoder(G=G, D=D_ATOMS); cell_encoder.init_from_soft_targets(b.X, q_target, n_epochs=40)
    assert LD_HIDDEN == cell_encoder.h_out, f"config.LD_HIDDEN={LD_HIDDEN} must equal {cell_encoder.h_out}"
    ld_decoder = LinearDecoder(hidden=LD_HIDDEN, G=G, ld_dim=2)

    grn_params = list(grn.parameters()) + list(cell_encoder.parameters()) + list(ld_decoder.parameters())
    opt_grn = torch.optim.AdamW(grn_params, lr=LR_GRN)
    sch_grn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_grn, T_max=N_EPOCHS, eta_min=1e-5)

    n_batches = len(loader)
    def _steps(cells_target):
        return n_batches if cells_target is None \
            else max(1, min(n_batches, int(np.ceil(cells_target / BATCH_SIZE))))
    steps_per_epoch_p1 = _steps(TARGET_CELLS_PER_EPOCH)
    steps_per_epoch_p2 = _steps(TARGET_CELLS_PER_EPOCH_PHASE2)
    print(f"  {n_batches} batches/epoch available; using {steps_per_epoch_p1} in phase 1, "
          f"{steps_per_epoch_p2} in phase 2")

    loss_history = []
    for epoch in range(N_EPOCHS):
        phase = 1 if epoch < PHASE1_EPOCHS else 2
        if epoch == PHASE1_EPOCHS:
            print("  -- Phase 2 --")
        anchor_scale = max(0.2, 1.0 - 0.8 * epoch / N_EPOCHS)
        grn.train(); cell_encoder.train()
        steps_per_epoch = steps_per_epoch_p1 if phase == 1 else steps_per_epoch_p2
        Wt = W1 if phase == 1 else W2

        epoch_losses = []
        for step, (rna_b, tf_b, bidx) in enumerate(loader):
            if step >= steps_per_epoch: break
            mix = cell_encoder(rna_b, temperature=1.0)
            l_mask = masked_reconstruction_loss(grn, mix, rna_b, tf_b, tf_indices_in_G, 0.40)
            q_b = q_t[bidx]
            l_anchor = (q_b * (torch.log(q_b + 1e-8) - torch.log(mix + 1e-8))).sum(1).mean()
            h_b = cell_encoder.encoder(rna_b)
            loss_grn = (Wt['MASK'] * l_mask + Wt['SPARSE'] * grn.sparsity_loss() +
                        Wt['ACCESS'] * grn.tf_access_consistency_loss() +
                        Wt['ACCESS_GENE'] * grn.gene_access_consistency_loss() +
                        Wt['DIVERSE'] * grn.diversity_loss() + Wt['ANCHOR'] * anchor_scale * l_anchor +
                        Wt['LD'] * ld_decoder.reconstruction_loss(h_b, rna_b) +
                        Wt['LD_REG'] * ld_decoder.ld_regularization_loss(h_b))

            opt_grn.zero_grad(); loss_grn.backward()
            torch.nn.utils.clip_grad_norm_(grn_params, 1.0)
            opt_grn.step(); epoch_losses.append(float(loss_grn.item()))

        sch_grn.step()
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            avg = float(np.mean(epoch_losses)); loss_history.append(avg)
            print(f"  [P{phase}] epoch {epoch:03d}  loss={avg:.4f}  anchor_scale={anchor_scale:.2f}")

    # ---- inference ----
    grn.eval(); cell_encoder.eval()
    with torch.no_grad():
        mix_full = cell_encoder(X_t, temperature=1.0)
        steady_full = grn.predict_rna(X_TF_t, mix_full).cpu().numpy()
        A_dict_np   = grn.A_dict.cpu().numpy(); A_base_np = grn.A_base.cpu().numpy()
        A_learned   = grn.effective_A(mix_full.mean(0)).cpu().numpy()
        all_mix     = mix_full.cpu().numpy()

    return dict(grn=grn, cell_encoder=cell_encoder, mix=all_mix, A_learned=A_learned, A_dict=A_dict_np,
                A_base=A_base_np, steady_state=steady_full, loss_history=np.array(loss_history))


def run_comet(b, out_dir, train=None):
    """Run COMET on one Bundle, save its arrays to out_dir and return a results dict.

    The trajectory (pseudotime q, root / sinks, velocity field V_pred) comes from utils/trajectory.final_trajectory
    and involves no training. With train=True (default: config.TRAIN_GRN) the GRN dictionary is also trained, and
    (if config.GRN_TILT_LAMBDA > 0) V_pred becomes the regulatory-consistency-tilted field
    (utils/trajectory.regulatory_consistency_tilt); the GRN-independent field is kept as V_pred_grn_independent.
    """
    os.makedirs(out_dir, exist_ok=True)
    train = TRAIN_GRN if train is None else bool(train)

    if not FORCE_RERUN:
        cached = _load_cached_run(out_dir, train)
        if cached is not None:
            print(f"\n::: {b.name} ::: found finished run in {out_dir} -> loading (set FORCE_RERUN=True to rerun).")
            # a finished run from before the tilt: apply it to the saved GRN (nothing is retrained)
            if train and GRN_TILT_LAMBDA > 0 and not os.path.exists(os.path.join(out_dir, "V_pred_grn_independent.npy")):
                Zst = state_space(b); S_mom = moment_operator(Zst, int(VEL_MOMENT_KNN.get(b.name, K_MOMENT)))
                A_base = np.load(os.path.join(out_dir, "A_base.npy"))
                tilt = regulatory_consistency_tilt(b, cached["V_pred"], A_base, cached["A_dict"], cached["mix"], S_mom, Zst)
                _save(out_dir, V_pred=tilt["V"], V_pred_grn_independent=cached["V_pred"],
                      grn_activity_velocity=tilt["activity"], grn_explained_fraction=tilt["explained"])
                cached["V_pred"] = tilt["V"]
            return cached

    set_seed()
    sign = float(ORIGIN_SIGN.get(b.name, 1.0))
    Zst  = state_space(b)
    mknn = int(VEL_MOMENT_KNN.get(b.name, K_MOMENT))
    print(f"\n::: {b.name} :::  (origin_sign={sign:+.0f}, state_space={STATE_SPACE.get(b.name, 'decipher')}, "
          f"moment_knn={mknn}, train_grn={train})")

    # ---- trajectory ----
    S_mom = moment_operator(Zst, mknn)
    traj = final_trajectory(b, Zst, S_mom, sign=sign)
    _save(out_dir, V_pred=traj["V"], committor_q=traj["q"], root_mask=traj["root_mask"], sink_mask=traj["sink_mask"],
          phi=traj["phi"], root_freq=traj["root_freq"], sink_freq=traj["sink_freq"], traj_u_bar=traj["u_bar"], traj_lead=traj["lead"],
          traj_w=traj["w"])
    with open(os.path.join(out_dir, "trajectory_mode.txt"), "w") as fh:
        fh.write(f"final\ncoupling={traj['coupling']:.4f} gate={traj['gate']:.0f} "
                 f"k_range={traj['k_range'][0]}-{traj['k_range'][-1]}\n")
    out = dict(V_pred=traj["V"], q_commit=traj["q"], root_mask=traj["root_mask"], sink_mask=traj["sink_mask"],
               phi=traj["phi"], traj_lead=traj["lead"], traj_w=traj["w"],
               mix=None, A_learned=None, A_dict=None, grn=None, cell_encoder=None)

    # ---- GRN ----
    if train:
        g = train_grn(b, Zst)
        _save(out_dir, A_dict=g["A_dict"], A_base=g["A_base"], A_learned=g["A_learned"], mix_weights=g["mix"],
              steady_state=g["steady_state"], A_genes=b.A_genes, A_tfs=b.A_tfs, loss_history=g["loss_history"])
        out.update(grn=g["grn"], cell_encoder=g["cell_encoder"], mix=g["mix"], A_learned=g["A_learned"],
                   A_dict=g["A_dict"])
        if GRN_TILT_LAMBDA > 0:
            tilt = regulatory_consistency_tilt(b, traj["V"], g["A_base"], g["A_dict"], g["mix"], S_mom, Zst)
            _save(out_dir, V_pred=tilt["V"], V_pred_grn_independent=traj["V"], grn_activity_velocity=tilt["activity"],
                  grn_explained_fraction=tilt["explained"])
            out["V_pred"] = tilt["V"]
    print(f"  saved -> {out_dir}")
    return out
