"""training/training_loop.py — ground truth: model_5_30_26_rerun.md notebook."""

import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.preprocessing import StandardScaler

from model.encoder import CellStateEncoder
from model.grn import (GRNDictionary,
                        initialize_grn_dictionary_from_clusters,
                        masked_reconstruction_loss)
from model.ld_vae import LinearDecoder
from model.velocity import (DictVelocityHead,
                             velocity_consistency_loss,
                             atac_direction_loss,
                             atac_direction_loss_asymmetric,
                             velocity_smoothness_loss,
                             density_gradient_loss,
                             divergence_loss,
                             project_to_2d)
from utils.graph import build_neighbor_index
from utils.geometry import build_local_atac
from model.ld_vae import plot_ld_embedding


def run_training(data: dict, model_config: dict, train_config: dict) -> dict:
    """
    Two-phase training loop identical to model_5_30_26_rerun.md.

    Phase 1 (epochs 0..PHASE1_EPOCHS-1): GRN + encoder + LD-decoder.
    Phase 2 (epochs PHASE1_EPOCHS..N_EPOCHS-1): adds DictVelocityHead.
    No psi, no Phase 3, no MarkovProgressionField.
    """
    # ── Unpack data ───────────────────────────────────────────────────────────
    X             = data["X"]
    X_TF          = data["X_TF"]
    M             = data["M"]
    gene_activity = data["gene_activity"]
    gene_access   = data["gene_access"]
    TF_access_prior = data["TF_access_prior"]
    A_genes       = data["A_genes"]
    A_tfs         = data["A_tfs"]
    gene_idx      = data["gene_idx"]
    tf_idx        = data["tf_idx"]
    sub_g2i       = data["sub_g2i"]
    Z_dec         = data["Z_dec"]
    V_dec2        = data["V_dec2"]
    rna_adata     = data["rna_adata"]

    # ── Unpack config ─────────────────────────────────────────────────────────
    NUM_RUNS      = train_config["NUM_RUNS"]
    N_EPOCHS      = train_config["N_EPOCHS"]
    PHASE1_EPOCHS = train_config["PHASE1_EPOCHS"]
    BATCH_SIZE    = train_config["BATCH_SIZE"]
    CLIP_NORM     = train_config.get("CLIP_NORM", 1.0)
    PLOT_EVERY    = train_config["PLOT_EVERY"]
    base_path     = train_config["OUTPUT_DIR"]
    USE_LOCAL_ATAC = train_config["USE_LOCAL_ATAC"]
    D_ATOMS       = model_config["D_ATOMS"]
    LR_GRN        = train_config["LR_GRN"]
    LR_VEL        = train_config["LR_VEL"]
    ETA_MIN       = train_config.get("ETA_MIN", 1e-5)
    NEIGHBOR_SPACE = train_config.get("NEIGHBOR_SPACE", "latent")

    W1 = train_config["WEIGHTS_PHASE1"]
    W2 = train_config["WEIGHTS_PHASE2"]

    _pt_key = train_config.get("OBS_PSEUDOTIME_KEY", None)

    # ── TF indices within A_genes (for masked reconstruction) ────────────────
    tf_indices_in_G = torch.tensor(
        [sub_g2i[t] if t in sub_g2i else -1 for t in A_tfs], dtype=torch.long
    )

    # ── Tensors ───────────────────────────────────────────────────────────────
    G = X.shape[1]; T = X_TF.shape[1]
    X_t        = torch.tensor(X,    dtype=torch.float32)
    X_TF_t     = torch.tensor(X_TF, dtype=torch.float32)
    TF_access_t = torch.tensor(TF_access_prior, dtype=torch.float32)
    cell_idx_t  = torch.arange(X_t.size(0))

    # ── kNN for velocity smoothness loss ──────────────────────────────────────
    neighbor_indices_t = build_neighbor_index(X, V_dec2, Z_dec, mode=NEIGHBOR_SPACE, k=15)

    # ── Gene access tensor ────────────────────────────────────────────────────
    if USE_LOCAL_ATAC:
        local_access, _ = build_local_atac(
            X, gene_activity, V_dec2, n_clusters=train_config.get("N_CLUSTERS", 20)
        )
        gene_access_t = torch.tensor(local_access, dtype=torch.float32)  # (N, G)
    else:
        gene_access_t = torch.tensor(gene_access, dtype=torch.float32)   # (G,)

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = TensorDataset(X_t, X_TF_t, cell_idx_t)
    _n_workers = min(os.cpu_count() // 2, train_config.get("MAX_DATALOADER_WORKERS", 8))
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=RandomSampler(dataset, replacement=False),
        drop_last=True,
        num_workers=_n_workers,
        pin_memory=True,
        persistent_workers=(_n_workers > 0),
    )

    # ── Optional: FACS pseudotime for GIF coloring ────────────────────────────
    pseudo_norm = None
    if _pt_key and _pt_key in rna_adata.obs.columns:
        pseudo = rna_adata.obs[_pt_key].astype(np.float32).values
        pseudo_norm = (pseudo - pseudo.min()) / (pseudo.max() - pseudo.min() + 1e-8)

    os.makedirs(base_path, exist_ok=True)

    all_loss_histories = {}

    for run_i in range(NUM_RUNS):
        run_dir = os.path.join(base_path, f"run{run_i}")

        # ── Checkpoint skip ───────────────────────────────────────────────────
        if os.path.exists(os.path.join(run_dir, "V_pred.npy")):
            print(f"\n::: Run {run_i} — checkpoint found, skipping. :::")
            continue

        print(f"\n::: Starting Run {run_i} :::")
        print(f"Phase 1: {PHASE1_EPOCHS} epochs  Phase 2: {N_EPOCHS - PHASE1_EPOCHS} epochs")

        # ── Model construction ────────────────────────────────────────────────
        grn = GRNDictionary(G=G, T=T, M=M, D=D_ATOMS)
        grn.TF_access = TF_access_t.clone()
        grn.mean_tf   = torch.tensor(X_TF.mean(axis=0), dtype=torch.float32)

        cell_encoder = CellStateEncoder(G=G, D=D_ATOMS,
                                         hidden=model_config.get("ENCODER_HIDDEN", 128))
        vel_head     = DictVelocityHead(G=G, hidden=model_config.get("VEL_HIDDEN", 256))
        ld_decoder   = LinearDecoder(hidden=model_config.get("LD_HIDDEN", 64), G=G, ld_dim=2)

        # ── Cluster-based initialization ──────────────────────────────────────
        cluster_labels, W_init = initialize_grn_dictionary_from_clusters(
            X=X, V_dec2=V_dec2, M=M, D=D_ATOMS, A_tfs=A_tfs, sub_g2i=sub_g2i
        )
        grn.init_from_clusters(W_init)
        cell_encoder.init_from_cluster_labels(X, cluster_labels, n_epochs=20)

        # ── Optimizers ────────────────────────────────────────────────────────
        params_grn = (list(grn.parameters()) +
                      list(cell_encoder.parameters()) +
                      list(ld_decoder.parameters()))
        opt_grn = torch.optim.AdamW(params_grn, lr=LR_GRN)
        opt_vel = torch.optim.AdamW(vel_head.parameters(), lr=LR_VEL)
        sch_grn = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_grn, T_max=N_EPOCHS, eta_min=ETA_MIN
        )
        sch_vel = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_vel, T_max=max(1, N_EPOCHS - PHASE1_EPOCHS), eta_min=ETA_MIN
        )

        # GIF maker imported lazily to avoid PIL dependency at import time
        from viz.visualization import VelocityGIFMaker
        gif_maker = VelocityGIFMaker(dpi=100)

        loss_history = {"epoch": [], "total": [], "phase": []}

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(N_EPOCHS):
            phase = 1 if epoch < PHASE1_EPOCHS else 2
            temperature = max(0.1, 0.5 - 0.4 * epoch / N_EPOCHS)

            grn.train(); cell_encoder.train(); ld_decoder.train()
            if phase == 1:
                vel_head.eval()
                for p in vel_head.parameters(): p.requires_grad_(False)
            else:
                if epoch == PHASE1_EPOCHS:
                    print("\n── Switching to Phase 2 (joint training) ──")
                vel_head.train()
                for p in vel_head.parameters(): p.requires_grad_(True)

            epoch_losses = []

            for rna_b, tf_b, batch_idx in loader:
                mix_w = cell_encoder(rna_b, temperature=temperature)

                l_mask    = masked_reconstruction_loss(
                    grn=grn, mix_weights=mix_w, rna_batch=rna_b,
                    tf_batch=tf_b, tf_indices_in_G=tf_indices_in_G, mask_frac=0.40,
                )
                l_sparse  = grn.sparsity_loss()
                l_access  = grn.tf_access_consistency_loss()
                l_diverse = grn.diversity_loss()
                l_dirich  = -torch.sum(torch.log(mix_w + 1e-8), dim=-1).mean()
                H_norm    = (-torch.sum(mix_w * torch.log(mix_w + 1e-8), dim=-1).mean()
                             / np.log(D_ATOMS))

                h_b    = cell_encoder.encoder(rna_b)
                l_ld   = ld_decoder.reconstruction_loss(h_b, rna_b)
                l_ld_r = ld_decoder.ld_regularization_loss(h_b)

                if phase == 1:
                    loss = (W1["MASK"]    * l_mask    +
                            W1["SPARSE"]  * l_sparse  +
                            W1["ACCESS"]  * l_access  +
                            W1["DIVERSE"] * l_diverse +
                            W1["ENTROPY"] * H_norm    +
                            W1["DIRICH"]  * l_dirich  +
                            W1["LD"]      * l_ld      +
                            W1["LD_REG"]  * l_ld_r)
                    opt_grn.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params_grn, CLIP_NORM)
                    opt_grn.step()

                else:  # phase 2
                    steady   = grn.predict_rna(tf_b, mix_w.detach())
                    disequil = steady - rna_b
                    velocity = vel_head(rna_b, disequil)

                    l_vel  = velocity_consistency_loss(velocity, rna_b, steady.detach())
                    if USE_LOCAL_ATAC:
                        l_atac = atac_direction_loss_asymmetric(velocity, gene_access_t[batch_idx])
                    else:
                        l_atac = atac_direction_loss(velocity, gene_access_t)
                    l_smo  = velocity_smoothness_loss(velocity, batch_idx, neighbor_indices_t)
                    l_den  = density_gradient_loss(velocity)
                    l_div  = divergence_loss(velocity, batch_idx, rna_b, neighbor_indices_t, X_t)

                    loss_grn = (W2["MASK"]    * l_mask    +
                                W2["SPARSE"]  * l_sparse  +
                                W2["ACCESS"]  * l_access  +
                                W2["DIVERSE"] * l_diverse +
                                W2["ENTROPY"] * H_norm    +
                                W2["DIRICH"]  * l_dirich  +
                                W2["LD"]      * l_ld      +
                                W2["LD_REG"]  * l_ld_r)
                    loss_vel = (W2["VEL"]    * l_vel  +
                                W2["ATAC"]   * l_atac +
                                W2["SMOOTH"] * l_smo  +
                                W2["DENSE"]  * l_den  +
                                W2["DIV"]    * l_div)

                    opt_grn.zero_grad(); opt_vel.zero_grad()
                    loss_grn.backward(retain_graph=True)
                    loss_vel.backward()
                    torch.nn.utils.clip_grad_norm_(params_grn, CLIP_NORM)
                    torch.nn.utils.clip_grad_norm_(vel_head.parameters(), CLIP_NORM)
                    opt_grn.step(); opt_vel.step()
                    loss = loss_grn + loss_vel

                epoch_losses.append(float(loss.item()))

            sch_grn.step()
            if phase == 2:
                sch_vel.step()

            avg = float(np.mean(epoch_losses))
            loss_history["epoch"].append(epoch)
            loss_history["total"].append(avg)
            loss_history["phase"].append(phase)

            if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
                print(f"  [P{phase}] Epoch {epoch:03d}  loss={avg:.4f}")

            if (epoch % PLOT_EVERY == 0 or epoch == N_EPOCHS - 1) and pseudo_norm is not None:
                grn.eval(); vel_head.eval(); cell_encoder.eval()
                with torch.no_grad():
                    mw_s   = cell_encoder(X_t, temperature=0.1)
                    ss_s   = grn.predict_rna(X_TF_t, mw_s)
                    V_snap = vel_head(X_t, ss_s - X_t).numpy()
                V_snap_2d = project_to_2d(V_snap, V_dec2, X, k=12, scale=12.0)
                gif_maker.add_frame(V_dec2, V_snap_2d, pseudo_norm, epoch)
                grn.train(); vel_head.train(); cell_encoder.train()

        # ── Save outputs ──────────────────────────────────────────────────────
        grn.eval(); vel_head.eval(); cell_encoder.eval()
        with torch.no_grad():
            full_w  = cell_encoder(X_t, temperature=0.1)
            ss_full = grn.predict_rna(X_TF_t, full_w)
            V_pred  = vel_head(X_t, ss_full - X_t).cpu().numpy()
            A_dict_np = grn.A_dict.cpu().numpy()
            w_np    = full_w.cpu().numpy()
            ss_np   = ss_full.cpu().numpy()
            h_full  = cell_encoder.encoder(X_t)
            z_2d_np, _ = ld_decoder(h_full)
            z_2d_np = z_2d_np.cpu().numpy()
            gene_loadings = ld_decoder.get_gene_loadings()

        os.makedirs(run_dir, exist_ok=True)
        np.save(f"{run_dir}/V_pred.npy",        V_pred)
        np.save(f"{run_dir}/A_dict.npy",         A_dict_np)
        np.save(f"{run_dir}/mix_weights.npy",    w_np)
        np.save(f"{run_dir}/steady_state.npy",   ss_np)
        np.save(f"{run_dir}/A_genes.npy",        A_genes)
        np.save(f"{run_dir}/A_tfs.npy",          A_tfs)
        np.save(f"{run_dir}/gene_idx.npy",       gene_idx)
        np.save(f"{run_dir}/tf_idx.npy",         tf_idx)
        np.save(f"{run_dir}/z_ld_2d.npy",        z_2d_np)
        np.save(f"{run_dir}/gene_loadings.npy",  gene_loadings)
        np.save(f"{run_dir}/V_dec2.npy",         V_dec2)
        np.save(f"{run_dir}/Z_dec.npy",          Z_dec)
        np.save(f"{run_dir}/neighbor_indices.npy", neighbor_indices_t.numpy())
        np.save(f"{run_dir}/gene_activity.npy",  gene_activity)
        with open(f"{run_dir}/loss_history.json", "w") as f:
            json.dump(loss_history, f)

        gif_maker.save(f"{run_dir}/velocity_training.gif", fps=3)
        gif_maker.save_slide_png(f"{run_dir}/velocity_slide_final.png")
        print(f"Run {run_i} complete → {run_dir}")

        all_loss_histories[f"run{run_i}"] = loss_history

    return {"loss_histories": all_loss_histories}
