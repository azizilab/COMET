import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from sklearn.decomposition import PCA

from config import (
    D_ATOMS, N_EPOCHS, PHASE1_EPOCHS, BATCH_SIZE, GEO_TAU_SCALE, LAMBDA_GRN,
    TARGET_CELLS_PER_EPOCH, FORCE_RERUN, ENTROPY_ROOT_TRUST, COMMIT_LOSS_W,
    COMMIT_W_SCALE, STATE_SPACE, VEL_MOMENT_KNN, SMOOTH_VELOCITY, BETA_PERP,
    K_MOMENT, RANDOM_STATE, W1, W2, GRN_PRUNE_FRAC, LD_HIDDEN, LR_GRN, LR_VEL,
)
from utils.common import zscore_cols
from utils.committor import moment_operator, committor_field, committor_gene_direction
from model.grn import GRNDictionary, select_archetypes_fps, geometric_soft_targets, init_atoms_from_soft_targets
from model.encoder import CellStateEncoder
from model.velocity import DictVelocityHead
from model.ld_vae import LinearDecoder
from losses.losses import (
    masked_reconstruction_loss, velocity_consistency_loss, atac_direction_loss,
    committor_consistency_loss, orient_disequilibrium,
)


def _load_cached_run(out_dir):
    """Rebuild the results dict from a finished run's saved arrays (no model objects needed
    downstream). Returns None if the run is incomplete."""
    need = ["V_pred.npy", "mix_weights.npy", "A_learned.npy", "A_dict.npy",
            "committor_q.npy", "root_mask.npy", "sink_mask.npy", "phi.npy"]
    if not all(os.path.exists(os.path.join(out_dir, f)) for f in need):
        return None
    L = lambda f: np.load(os.path.join(out_dir, f))
    return dict(V_pred=L("V_pred.npy"), mix=L("mix_weights.npy"), A_learned=L("A_learned.npy"),
                A_dict=L("A_dict.npy"), q_commit=L("committor_q.npy"),
                root_mask=L("root_mask.npy").astype(bool), sink_mask=L("sink_mask.npy").astype(bool),
                phi=L("phi.npy"), grn=None, cell_encoder=None, vel_head=None)


def run_comet(b, out_dir):
    """Train the full COMET model on one Bundle; save arrays; return a results dict."""
    os.makedirs(out_dir, exist_ok=True)

    # ---- resume: skip datasets that already finished ----
    if not FORCE_RERUN:
        cached = _load_cached_run(out_dir)
        if cached is not None:
            print(f"\n::: {b.name} ::: found finished run in {out_dir} -> loading & skipping training "
                  f"(set FORCE_RERUN=True to retrain).")
            return cached

    trust    = float(ENTROPY_ROOT_TRUST.get(b.name, 1.0))
    commit_w = COMMIT_LOSS_W * abs(trust) * float(COMMIT_W_SCALE.get(b.name, 1.0))
    beta     = float(BETA_PERP.get(b.name, 0.5))

    # ---- pick the LATENT STATE SPACE the trajectory is built in ----------------
    # This is the only embedding choice that changes the gene-space velocity: committor graph,
    # archetypes, geometric targets and the projection search space all live here.
    space = STATE_SPACE.get(b.name, "decipher")
    if space == "pca":
        Zst = PCA(n_components=min(30, b.G - 1, b.N - 1),
                  random_state=RANDOM_STATE).fit_transform(zscore_cols(b.X)).astype(np.float32)
    else:
        Zst = b.Z_dec
    # first-moment denoiser in the state space (cleans the noisy per-cell ATAC lead; k larger for
    # the paired ma2020 dataset). Built once and reused for the lead and the final velocity.
    mknn  = int(VEL_MOMENT_KNN.get(b.name, K_MOMENT))
    S_mom = moment_operator(Zst, mknn)
    print(f"\n::: {b.name} :::  (entropy_trust={trust:+.2f}, beta_perp={beta:.2f}, "
          f"state_space={space}, moment_knn={mknn}, committor_reinf_w={commit_w:.2f})")

    # ---- shared lead signals + committor field (built once, in the state space) ----
    # Pool the z-scored ATAC/RNA lead over state-space neighborhoods before differencing — this is
    # the first-moment denoising that removes the per-cell jitter (the ma2020 messiness).
    atac_z = np.asarray(S_mom @ zscore_cols(b.ga_matched), dtype=np.float32)
    rna_z  = np.asarray(S_mom @ zscore_cols(b.X),          dtype=np.float32)
    atac_z_t = torch.tensor(atac_z, dtype=torch.float32)
    rna_z_t  = torch.tensor(rna_z,  dtype=torch.float32)

    q_commit, root_mask, sink_mask, phi = committor_field(b, entropy_trust=trust, Z=Zst)
    commit_dir, commit_valid = committor_gene_direction(b, q_commit, Z=Zst)
    commit_dir_t = torch.tensor(commit_dir, dtype=torch.float32)
    commit_valid_t = torch.tensor(commit_valid, dtype=torch.bool)
    print(f"  committor: root={int(root_mask.sum())} cells, sinks={int(sink_mask.sum())}, "
          f"valid-direction cells={int(commit_valid.sum())}/{b.N}")

    # ---- archetypes + geometric soft mixing targets (in the state space) ----
    arch_idx = select_archetypes_fps(Zst, D_ATOMS, seed=0)
    q_target = geometric_soft_targets(Zst, arch_idx, tau_scale=GEO_TAU_SCALE)

    # ---- ATAC accessibility priors ----
    G, T = b.G, b.T
    TF_access = np.zeros(T, np.float32)
    for i, tf in enumerate(b.A_tfs):
        if tf in b.sub_g2i: TF_access[i] = float(b.ga_matched[:, b.sub_g2i[tf]].mean())
    lo, hi = TF_access.min(), TF_access.max(); TF_access = (TF_access - lo) / (hi - lo + 1e-8)
    TF_access_t   = torch.tensor(TF_access, dtype=torch.float32)
    gene_access_t = torch.tensor(b.gene_access, dtype=torch.float32)

    # ---- tensors + loader ----
    # standardise the TF inputs: predict_rna is X_TF @ A.T with no intercept, so a weight is
    # "RNA per unit of TF expression". In raw units that makes |A[:, t]| scale like
    # 1/expression(t): a highly expressed TF needs a SMALL weight to reach a given target level
    # and a barely expressed one needs a LARGE weight. Ranking edges by |A| therefore ranks TFs
    # roughly in reverse order of how active they are — measured on the mHSC-E run,
    # Spearman(per-TF mean |A|, TF expression) = -0.73, which put edge recovery *below* random
    # because well-expressed TFs are exactly the hubs of the ChIP references.
    #
    # Dividing each TF column by its own SD makes a weight "RNA per SD of TF", i.e. a
    # standardised coefficient, which is unit-free and directly comparable across TFs. A_learned
    # is saved in those units, so downstream edge rankings are comparable without any post-hoc
    # rescaling. Ported from cleanup.
    X_TF = (b.X_TF - b.X_TF.mean(0)) / (b.X_TF.std(0) + 1e-8)
    X_t    = torch.tensor(b.X,  dtype=torch.float32)
    X_TF_t = torch.tensor(X_TF, dtype=torch.float32)
    q_t    = torch.tensor(q_target, dtype=torch.float32)
    cell_idx_t = torch.arange(b.N)
    tf_indices_in_G = torch.tensor(b.tf_indices_in_G, dtype=torch.long)
    dataset = TensorDataset(X_t, X_TF_t, cell_idx_t)
    # drop_last=False: on small bundles, dropping the ragged tail every epoch costs whole
    # optimizer steps (e.g. at 1071 cells / batch 512, drop_last=True leaves only 2 steps/epoch).
    # Every loss term here is batch-size agnostic, so a short final batch is safe. Ported from
    # cleanup.
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=RandomSampler(dataset, replacement=False), drop_last=False)

    # ---- models ----
    grn = GRNDictionary(G=G, T=T, M=b.M, D=D_ATOMS)
    grn.TF_access = TF_access_t.clone(); grn.Gene_access = gene_access_t.clone()
    # Representative TF input for diversity_loss's prediction-space term, which evaluates each
    # atom on it and penalises atoms whose OUTPUTS agree. X_TF is now standardised so its column
    # mean is 0 by construction, and feeding that in would make every atom predict ~0 and the
    # term vanish silently. Mean absolute deviation keeps a per-TF, positive, typical-magnitude
    # input in the same units the model now consumes. Ported from cleanup.
    grn.mean_tf = torch.tensor(np.abs(X_TF).mean(0), dtype=torch.float32)
    grn.init_from_atoms(init_atoms_from_soft_targets(b.X, q_target, b.M, b.A_tfs))
    cell_encoder = CellStateEncoder(G=G, D=D_ATOMS); cell_encoder.init_from_soft_targets(b.X, q_target, n_epochs=40)
    vel_head = DictVelocityHead(G=G)
    # LinearDecoder consumes the encoder's hidden output directly, so LD_HIDDEN is not free --
    # it has to match cell_encoder.h_out (the encoder's hidden // 2).
    assert LD_HIDDEN == cell_encoder.h_out, (
        f"config.LD_HIDDEN={LD_HIDDEN} must equal the encoder's output width "
        f"({cell_encoder.h_out})"
    )
    ld_decoder = LinearDecoder(hidden=LD_HIDDEN, G=G, ld_dim=2)

    opt_grn = torch.optim.AdamW(
        list(grn.parameters()) + list(cell_encoder.parameters()) + list(ld_decoder.parameters()), lr=LR_GRN)
    opt_vel = torch.optim.AdamW(vel_head.parameters(), lr=LR_VEL)
    sch_grn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_grn, T_max=N_EPOCHS, eta_min=1e-5)
    sch_vel = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vel, T_max=N_EPOCHS - PHASE1_EPOCHS, eta_min=1e-5)

    # cap minibatches/epoch so per-epoch cost is independent of #cells (unbiased SGD subsampling)
    n_batches = len(loader)
    steps_per_epoch = n_batches if TARGET_CELLS_PER_EPOCH is None \
        else max(1, min(n_batches, int(np.ceil(TARGET_CELLS_PER_EPOCH / BATCH_SIZE))))
    print(f"  {n_batches} batches/epoch available; using {steps_per_epoch}/epoch "
          f"({steps_per_epoch*BATCH_SIZE} cells/epoch of {b.N}).")

    loss_history = []
    for epoch in range(N_EPOCHS):
        phase = 1 if epoch < PHASE1_EPOCHS else 2
        if epoch == PHASE1_EPOCHS: print("  -- Phase 2 (joint velocity + committor) --")
        anchor_scale = max(0.2, 1.0 - 0.8 * epoch / N_EPOCHS)
        grn.train(); cell_encoder.train(); vel_head.train(phase == 2)
        for p in vel_head.parameters(): p.requires_grad_(phase == 2)

        epoch_losses = []
        for step, (rna_b, tf_b, bidx) in enumerate(loader):
            if step >= steps_per_epoch: break                  # fresh random subset each epoch
            mix = cell_encoder(rna_b, temperature=1.0)
            l_mask    = masked_reconstruction_loss(grn, mix, rna_b, tf_b, tf_indices_in_G, 0.40)
            l_sparse  = grn.sparsity_loss(); l_access = grn.tf_access_consistency_loss()
            l_access_gene = grn.gene_access_consistency_loss()
            l_diverse = grn.diversity_loss()
            q_b       = q_t[bidx]
            l_anchor  = (q_b * (torch.log(q_b + 1e-8) - torch.log(mix + 1e-8))).sum(1).mean()
            h_b       = cell_encoder.encoder(rna_b)
            l_ld      = ld_decoder.reconstruction_loss(h_b, rna_b)
            l_ld_reg  = ld_decoder.ld_regularization_loss(h_b)

            if phase == 1:
                loss_grn = (W1['MASK']*l_mask + W1['SPARSE']*l_sparse + W1['ACCESS']*l_access +
                            W1['ACCESS_GENE']*l_access_gene +
                            W1['DIVERSE']*l_diverse + W1['ANCHOR']*anchor_scale*l_anchor +
                            W1['LD']*l_ld + W1['LD_REG']*l_ld_reg)
                opt_grn.zero_grad(); loss_grn.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(grn.parameters()) + list(cell_encoder.parameters()) + list(ld_decoder.parameters()), 1.0)
                opt_grn.step(); epoch_losses.append(float(loss_grn.item()))
            else:
                # ATAC->RNA led disequilibrium (the gene program + per-cell magnitude)
                az, rz     = atac_z_t[bidx], rna_z_t[bidx]
                chrom_lead = az - rz
                tf_lead    = az[:, tf_indices_in_G] - rz[:, tf_indices_in_G]
                prod       = grn.production_from_lead(tf_lead, mix.detach(), prune_frac=GRN_PRUNE_FRAC)
                disequil   = chrom_lead + LAMBDA_GRN * prod
                # committor-orient it: lock the forward sign, keep GRN magnitude + orthogonal tilt.
                # This is what the head sees, so the committor can actually steer the flow.
                disequil_or = orient_disequilibrium(
                    disequil, commit_dir_t[bidx], commit_valid_t[bidx], beta=beta)
                velocity   = vel_head(rna_b, disequil_or)

                l_vel    = velocity_consistency_loss(velocity, disequil_or.detach())
                ga_b     = gene_access_t if gene_access_t.dim() == 1 else gene_access_t[bidx]
                l_atac   = atac_direction_loss(velocity, ga_b)
                l_commit = committor_consistency_loss(velocity, commit_dir_t[bidx], commit_valid_t[bidx])

                loss_vel = W2['VEL']*l_vel + W2['ATAC']*l_atac + commit_w*l_commit
                loss_grn = (W2['MASK']*l_mask + W2['SPARSE']*l_sparse + W2['ACCESS']*l_access +
                            W2['ACCESS_GENE']*l_access_gene +
                            W2['DIVERSE']*l_diverse + W2['ANCHOR']*anchor_scale*l_anchor +
                            W2['LD']*l_ld + W2['LD_REG']*l_ld_reg)

                opt_grn.zero_grad(); opt_vel.zero_grad()
                loss_grn.backward(retain_graph=True); loss_vel.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(grn.parameters()) + list(cell_encoder.parameters()) + list(ld_decoder.parameters()), 1.0)
                torch.nn.utils.clip_grad_norm_(vel_head.parameters(), 1.0)
                opt_grn.step(); opt_vel.step()
                epoch_losses.append(float((loss_grn + loss_vel).item()))

        sch_grn.step()
        if phase == 2: sch_vel.step()
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            avg = float(np.mean(epoch_losses)); loss_history.append(avg)
            print(f"  [P{phase}] epoch {epoch:03d}  loss={avg:.4f}  anchor_scale={anchor_scale:.2f}")

    # ---- inference + save ----
    grn.eval(); vel_head.eval(); cell_encoder.eval()
    with torch.no_grad():
        mix_full      = cell_encoder(X_t, temperature=1.0)
        az_all        = atac_z_t - rna_z_t
        tf_lead_all   = az_all[:, tf_indices_in_G]
        disequil_full = az_all + LAMBDA_GRN * grn.production_from_lead(tf_lead_all, mix_full, prune_frac=GRN_PRUNE_FRAC)
        disequil_full = orient_disequilibrium(disequil_full, commit_dir_t, commit_valid_t, beta=beta)
        V_pred        = vel_head(X_t, disequil_full).cpu().numpy()
        steady_full   = grn.predict_rna(X_TF_t, mix_full).cpu().numpy()
        A_dict_np     = grn.A_dict.cpu().numpy(); A_base_np = grn.A_base.cpu().numpy()
        A_learned     = grn.effective_A(mix_full.mean(0)).cpu().numpy()
        all_mix       = mix_full.cpu().numpy()

    # first-moment pooling of the final gene-space velocity (denoise before projection). The arrow
    # of time is committor-locked, so this cannot reverse the flow — it only removes per-cell jitter.
    if SMOOTH_VELOCITY:
        V_pred = np.asarray(S_mom @ V_pred, dtype=np.float32)

    np.save(f"{out_dir}/V_pred.npy", V_pred);          np.save(f"{out_dir}/A_dict.npy", A_dict_np)
    np.save(f"{out_dir}/A_base.npy", A_base_np);        np.save(f"{out_dir}/A_learned.npy", A_learned)
    np.save(f"{out_dir}/mix_weights.npy", all_mix);     np.save(f"{out_dir}/steady_state.npy", steady_full)
    np.save(f"{out_dir}/A_genes.npy", b.A_genes);       np.save(f"{out_dir}/A_tfs.npy", b.A_tfs)
    np.save(f"{out_dir}/committor_q.npy", q_commit);    np.save(f"{out_dir}/loss_history.npy", np.array(loss_history))
    np.save(f"{out_dir}/root_mask.npy", root_mask);     np.save(f"{out_dir}/sink_mask.npy", sink_mask)
    np.save(f"{out_dir}/phi.npy", phi)
    print(f"  saved -> {out_dir}")

    return dict(V_pred=V_pred, mix=all_mix, A_learned=A_learned, A_dict=A_dict_np,
                q_commit=q_commit, root_mask=root_mask, sink_mask=sink_mask, phi=phi,
                grn=grn, cell_encoder=cell_encoder, vel_head=vel_head)
