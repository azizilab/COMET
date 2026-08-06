from model.encoder import CellStateEncoder
from model.grn import GRNDictionary, init_atoms_from_soft_targets, masked_reconstruction_loss1
from model.velocity import DictVelocityHead, velocity_consistency_loss, atac_direction_loss, atac_direction_loss
from model.ld_vae import LinearDecoder, plot_ld_embedding
from utils.graph import build_neighbor_index
from utils.geometry import project_to_2d, build_local_atac, select_archetypes_fps, geometric_soft_targets, zscore_cols, knn_smooth
from viz.visualization import VelocityGIFMaker, plot_velocity_stream

import os
from collections import defaultdict

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

def run_training(data, model_config, train_config):

    X = data["X"] # normalised scRNA-seq expression matrix (C x G)
    X_TF_raw = data["X_TF"] # TF-expression sub-matrix (C x T), raw normalised units

    # ── standardise the TF inputs ────────────────────────────────────────────
    # predict_rna is X_TF @ A.T with no intercept, so a weight is "RNA per unit of
    # TF expression". In raw units that makes |A[:, t]| scale like 1/expression(t):
    # a highly expressed TF needs a SMALL weight to reach a given target level and
    # a barely expressed one needs a LARGE weight. Ranking edges by |A| therefore
    # ranks TFs roughly in reverse order of how active they are — measured on the
    # mHSC-E run, Spearman(per-TF mean |A|, TF expression) = -0.73, which put edge
    # recovery *below* random because well-expressed TFs are exactly the hubs of
    # the ChIP references.
    #
    # Dividing each TF column by its own SD makes a weight "RNA per SD of TF", i.e.
    # a standardised coefficient, which is unit-free and directly comparable across
    # TFs. A_learned is saved in those units, so downstream edge rankings are
    # comparable without any post-hoc rescaling.
    X_TF = (X_TF_raw - X_TF_raw.mean(axis=0)) / (X_TF_raw.std(axis=0) + 1e-8)

    M = data["M"] # binary motif mask (G x T), M_gt=1 iff TF t has a binding motif in a peak linked to gene g

    gene_activity = data["gene_activity"] # per-cell local chromatin accessibility (C x G) 
    gene_access = data["gene_access"] # population-mean chromatin accessibility for each gene (G,)
    gene_access_t = data["gene_access_t"] # as tensor

    TF_access_prior = data["TF_access_prior"] # population-mean chromatin accessibility for each TF locus (T,)
    TF_access_t = torch.tensor(TF_access_prior, dtype=torch.float32)

    # identity sets for genes and TFs (i.e. active "universe")
    A_genes = data["A_genes"]
    A_tfs = data["A_tfs"]

    # index mappings
    gene_idx = data["gene_idx"]
    tf_idx = data["tf_idx"]
    sub_g2i = data["sub_g2i"]

    rna_adata = data["rna_adata"] # raw AnnData object for RNA, used for pseudotime and plotting only (not model input)

    
    # Run once before starting training
    # Find the local index of each TF within the A_genes list
    tf_indices_in_G = []
    for tf_name in A_tfs:
        if tf_name in sub_g2i:
            tf_indices_in_G.append(sub_g2i[tf_name])
        else:
            # This shouldn't happen based on A_tfs filtering, but for safety:
            tf_indices_in_G.append(-1) 

    tf_indices_in_G = torch.tensor(tf_indices_in_G, dtype=torch.long)


    decipher_data = data["decipher_outputs"]
    Z_dec   = decipher_data["z"].astype(np.float32)   # (C, 10)
    V_dec2  = decipher_data["v"].astype(np.float32)   # (C, 2)

    USE_LOCAL_ATAC = train_config["USE_LOCAL_ATAC"]
    K_MOMENT = train_config["K_MOMENT"]  # kNN smoothing strength for the ATAC/RNA lead

    neighbor_indices_t = build_neighbor_index(X, V_dec2, Z_dec, mode=train_config["NEIGHBOR_SPACE"], k=15)

    if USE_LOCAL_ATAC:
        # This assumes 'gene_activity' is your (C_atac, G) matrix
        local_access = build_local_atac(X, gene_activity, Z_dec, n_clusters=train_config["N_CLUSTERS"], k_atac=8)
        gene_access_t = torch.tensor(local_access, dtype=torch.float32)
        # Denoise the raw per-cell ATAC estimate over its latent-space neighborhood.
        # This replaces the old displacement-vector smoothing loss as the pipeline's
        # single denoising step, moved from output to input.
        local_access = knn_smooth(local_access, neighbor_indices_t.numpy(), k=K_MOMENT)
    else:
        # Fallback to your existing global vector (G,)
        gene_access_t = torch.tensor(gene_access, dtype=torch.float32)

    # ── shared, per-gene z-scored lead signal (computed once) ─────────────────────
    # atac_z - rna_z is the ATAC->RNA lead: open-but-quiet genes are primed to rise.
    # RNA is smoothed at the SAME neighborhood scale as ATAC (K_MOMENT) before
    # z-scoring: smoothing only one side leaves a variance imbalance that becomes a
    # spurious pull-to-the-population-mean once the two are z-scored and differenced.
    X_smooth = knn_smooth(X, neighbor_indices_t.numpy(), k=K_MOMENT)

    atac_z = zscore_cols(local_access)   # (N, G) — already kNN-smoothed above
    rna_z  = zscore_cols(X_smooth)       # (N, G)
    lead   = knn_smooth(atac_z - rna_z, neighbor_indices_t.numpy(), k=K_MOMENT)   # (N, G)
    lead_t = torch.tensor(lead, dtype=torch.float32)
    print(f"Lead signal ready: atac_z - rna_z (per-cell, per-gene), symmetric kNN-denoise k={K_MOMENT}.")

    # --- SET NUMBER OF RUNS & PHASES HERE ---
    NUM_RUNS = train_config["NUM_RUNS"]
    N_EPOCHS = train_config["N_EPOCHS"]
    PHASE1_EPOCHS = train_config["PHASE1_EPOCHS"] # GRN + encoder warmup (no velocity)
    BATCH_SIZE = train_config["BATCH_SIZE"]
    D_ATOMS = model_config["D_ATOMS"] # number of GRN dictionary atoms
    # ----------------------------------------
    

    # ── data prep ────────────────────────────────────────────────────────────
    G = X.shape[1]
    T = X_TF.shape[1]
    
    X_t        = torch.tensor(X,    dtype=torch.float32)
    X_TF_t     = torch.tensor(X_TF, dtype=torch.float32)
    cell_idx_t = torch.arange(X_t.size(0))

    #pseudo = rna_adata.obs["facs_pseudotime"].astype(np.float32)
    #pseudo = rna_adata.obs["dpt_pseudotime"].astype(np.float32)
    pseudo_norm = np.zeros(X_t.size(0), dtype=np.float32)
    #pseudo_norm = (pseudo.values - pseudo.min()) / (np.ptp(pseudo.values) + 1e-8)
    pseudo_norm_t = torch.tensor(pseudo_norm, dtype=torch.float32)
    
    dataset = TensorDataset(X_t, X_TF_t, cell_idx_t)
    
    # drop_last=False: with 1071 cells the old drop_last=True discarded the ragged
    # tail every epoch (47 cells at batch 128, 47 at batch 512) and, more to the
    # point, cost whole optimizer steps — at batch 512 it left only 2 steps per
    # epoch, 240 over a 120-epoch run. Every loss term here is batch-size agnostic,
    # so a short final batch is safe.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        sampler=RandomSampler(dataset, replacement=False),
                        drop_last=False)
    
    W1 = train_config["WEIGHTS_PHASE1"]
    W2 = train_config["WEIGHTS_PHASE2"]

    # ── loss weights (Phase 1: GRN + Encoder warmup) ─────────────────────────
    W1_MASK    = W1["MASK"]
    W1_SPARSE  = W1["SPARSE"]
    W1_ACCESS  = W1["ACCESS"]
    W1_ACCESS_GENE = W1["ACCESS_GENE"]
    W1_DIVERSE = W1["DIVERSE"] # HIGH — forces atoms apart from the start
    W1_LD      = W1["LD"] # reconstruction weight — raised 10×
    W1_LD_REG  = W1["LD_REG"] # orthogonality + variance regulariser

    # ── loss weights (Phase 2: Joint) ────────────────────────────────────────
    W2_MASK    = W2["MASK"]
    W2_VEL     = W2["VEL"]
    W2_ATAC    = W2["ATAC"]
    W2_SPARSE  = W2["SPARSE"]
    W2_ACCESS  = W2["ACCESS"]
    W2_ACCESS_GENE = W2["ACCESS_GENE"]
    W2_DIVERSE = W2["DIVERSE"]
    W2_LD      = W2["LD"]
    W2_LD_REG  = W2["LD_REG"]
    
    PLOT_EVERY = train_config["PLOT_EVERY"]
    base_path = train_config["OUTPUT_DIR"]  # Defined here so the check can use it
    os.makedirs(base_path, exist_ok=True)

    for run_i in range(NUM_RUNS):
        # ── CHECK IF RUN ALREADY EXISTS ────────────────────────────────────────
        if run_i == 0:
            # Run 0 could exist directly in the main folder (if previously run with NUM_RUNS=1)
            # or inside its own subfolder 'run0'
            exists_in_main = os.path.exists(os.path.join(base_path, "V_pred.npy"))
            exists_in_sub  = os.path.exists(os.path.join(base_path, "run0", "V_pred.npy"))
            run_exists = exists_in_main or exists_in_sub
        else:
            # Runs 1, 2, 3+ will always be in their respective subfolders
            run_exists = os.path.exists(os.path.join(base_path, f"run{run_i}", "V_pred.npy"))

        if run_exists:
            print(f"\n::: Run {run_i} already exists. Skipping... :::")
            # Set 'd' as a fallback so the plotting code at the very end doesn't throw a NameError
            d = os.path.join(base_path, f"run{run_i}") if run_i > 0 else base_path
            continue
        # ───────────────────────────────────────────────────────────────────────

        print(f"\n::: Starting Run {run_i} :::")
        print(f"Phase 1: GRN+Encoder warmup for {PHASE1_EPOCHS} epochs")
        print(f"Phase 2: Joint training for {N_EPOCHS - PHASE1_EPOCHS} epochs")
    

        # ── model construction ────────────────────────────────────────────────
        grn = GRNDictionary(G=G, T=T, M=M, D=D_ATOMS)
        grn.TF_access = TF_access_t.clone()
        grn.Gene_access = torch.tensor(gene_access, dtype=torch.float32)
        # Representative TF input for diversity_loss's prediction-space term, which
        # evaluates each atom on it and penalises atoms whose OUTPUTS agree. Now that
        # X_TF is standardised its column mean is 0 by construction, and feeding that
        # in would make every atom predict ~0 and the term vanish silently. Mean
        # absolute deviation keeps a per-TF, positive, typical-magnitude input in the
        # same units the model now consumes.
        grn.mean_tf = torch.tensor(np.abs(X_TF).mean(axis=0), dtype=torch.float32)
    
        cell_encoder = CellStateEncoder(G=G, D=D_ATOMS)
    
        vel_head = DictVelocityHead(G=G)
        # Linear decoder for interpretable 2D embedding
        ld_decoder = LinearDecoder(hidden=64, G=G, ld_dim=2)

        # ── cluster-based initialization ──────────────────────────────────────

        arch_idx = select_archetypes_fps(Z_dec, D_ATOMS, seed=0)
        q_target = geometric_soft_targets(Z_dec, arch_idx, tau_scale=train_config["GEO_TAU_SCALE"])
        q_t     = torch.tensor(q_target, dtype=torch.float32)


        _ent = -(q_target * np.log(q_target + 1e-8)).sum(1).mean()
        print(f"Archetype cells: {arch_idx.tolist()}")
        print(f"Soft-target entropy: {_ent:.3f} / {np.log(D_ATOMS):.3f} max "
            f"(want clearly between 0 and max)")


        grn.init_from_atoms(init_atoms_from_soft_targets(X, q_target, M, A_tfs))

        A_dict_init_np = grn.A_dict.detach().cpu().numpy().copy() # for analysis
    
        # Pre-train encoder to recognize clusters
        cell_encoder.init_from_soft_targets(X, q_target, n_epochs=40)
    
        # ── optimizers ────────────────────────────────────────────────────────
        opt_grn = torch.optim.AdamW(
            list(grn.parameters()) + 
            list(cell_encoder.parameters()) + 
            list(ld_decoder.parameters()),
            lr=train_config["LR_GRN"]
        )
        
        opt_vel = torch.optim.AdamW(vel_head.parameters(), lr=train_config["LR_VEL"])
    
        sch_grn = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_grn, T_max=N_EPOCHS, eta_min=train_config["LR_SCHED"]["ETA_MIN"]
        )
        sch_vel = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_vel, T_max=N_EPOCHS - PHASE1_EPOCHS, eta_min=train_config["LR_SCHED"]["ETA_MIN"]
        )
    
        gif_maker = VelocityGIFMaker(dpi=100)
        loss_history = []
    
        for epoch in range(N_EPOCHS):
            phase = 1 if epoch < PHASE1_EPOCHS else 2
    
            if epoch == PHASE1_EPOCHS:
                print(f"\n── Switching to Phase 2 (joint training) ──")

            anchor_scale = max(0.2, 1.0 - 0.8 * epoch / N_EPOCHS)
    
            grn.train(); cell_encoder.train()
            if phase == 1:
                vel_head.eval()
                for p in vel_head.parameters(): p.requires_grad_(False)
            else:
                vel_head.train()
                for p in vel_head.parameters(): p.requires_grad_(True)
    
            epoch_losses = []
            epoch_terms  = defaultdict(list)   # raw (unweighted) per-term diagnostics

            for rna_b, tf_b, batch_idx in loader:
                # Compute mix_weights ONCE — shared by all loss terms
                mix_weights = cell_encoder(rna_b, temperature=1.0)

                # ── GRN + Encoder losses (both phases) ────────────────────────
                l_mask = masked_reconstruction_loss1(
                    grn=grn, mix_weights=mix_weights,          # <-- pass pre-computed
                    rna_batch=rna_b, tf_batch=tf_b,
                    tf_indices_in_G=tf_indices_in_G.to(rna_b.device),
                    mask_frac=0.40
                )
                l_sparse  = grn.sparsity_loss()
                l_access  = grn.tf_access_consistency_loss()
                l_access_gene = grn.gene_access_consistency_loss()
                l_diverse = grn.diversity_loss()
                q_b       = q_t[batch_idx]
                l_anchor  = (q_b * (torch.log(q_b + 1e-8) - torch.log(mix_weights + 1e-8))).sum(1).mean()

                # LD loss: linear decoder reconstruction
                h_batch  = cell_encoder.encoder(rna_b)        # (B, 64)
                l_ld     = ld_decoder.reconstruction_loss(h_batch, rna_b)
                l_ld_reg = ld_decoder.ld_regularization_loss(h_batch)

                epoch_terms['mask'].append(float(l_mask.item()))
                epoch_terms['sparse'].append(float(l_sparse.item()))
                epoch_terms['access'].append(float(l_access.item()))
                epoch_terms['access_gene'].append(float(l_access_gene.item()))
                epoch_terms['diverse'].append(float(l_diverse.item()))
                epoch_terms['anchor'].append(float(l_anchor.item()))
                epoch_terms['ld'].append(float(l_ld.item()))
                epoch_terms['ld_reg'].append(float(l_ld_reg.item()))

                if phase == 1:
                    loss_grn = (W1_MASK    * l_mask    +
                                W1_SPARSE  * l_sparse  +
                                W1_ACCESS  * l_access  +
                                W1_ACCESS_GENE * l_access_gene +
                                W1_DIVERSE * l_diverse +
                                W1['ANCHOR'] * anchor_scale * l_anchor +
                                W1_LD      * l_ld      +
                                W1_LD_REG  * l_ld_reg)
                
                    opt_grn.zero_grad()
                    loss_grn.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(grn.parameters()) + list(cell_encoder.parameters()) +
                        list(ld_decoder.parameters()), 1.0
                    )
                    opt_grn.step()
                    epoch_losses.append(float(loss_grn.item()))

                else:  # phase 2
                    # ── ATAC->RNA led disequilibrium (the arrow of time) ──────────
                    # Reverted to the two-term formula: the raw gene-level lead is a
                    # fixed, non-learnable vector that anchors disequil's overall scale.
                    # Dropping it (GRN-propagated term only) removes that anchor, and
                    # the shared A_base/A_dict then has every incentive to inflate
                    # toward tanh saturation to satisfy the velocity magnitude loss —
                    # nothing else in the loss restrains it strongly enough. Keeping the
                    # fixed term bounds disequil's scale independent of GRN weights, so
                    # that runaway can't happen.
                    chrom_lead = lead_t[batch_idx]                       # (B, G)
                    tf_lead    = chrom_lead[:, tf_indices_in_G]          # (B, T)
                    prod       = grn.production_from_lead(tf_lead, mix_weights.detach(),
                                                           prune_frac=train_config["GRN_PRUNE_FRAC"])  # (B, G)
                    disequil   = chrom_lead + train_config["LAMBDA_GRN"] * prod

                    # Velocity forward — pass disequil directly, no encoder re-call
                    velocity = vel_head(rna_b, disequil)                     # (B, G)

                    l_vel    = velocity_consistency_loss(velocity, disequil.detach())

                    if USE_LOCAL_ATAC:
                        ga_batch = gene_access_t if gene_access_t.dim() == 1 else gene_access_t[batch_idx]
                        l_atac   = atac_direction_loss(velocity, ga_batch)
                    else:
                        l_atac   = atac_direction_loss(velocity, gene_access_t)

                    #l_dense  = density_gradient_loss(velocity, batch_idx, V_dec2)
                    #l_div    = divergence_loss(velocity, batch_idx, rna_b, neighbor_indices_t, X_t)

                    epoch_terms['vel'].append(float(l_vel.item()))
                    epoch_terms['atac'].append(float(l_atac.item()))
                    epoch_terms['disequil_norm'].append(float(disequil.norm(dim=1).mean().item()))
                    epoch_terms['velocity_norm'].append(float(velocity.norm(dim=1).mean().item()))

                    loss_vel = (W2_VEL    * l_vel    +
                                W2_ATAC   * l_atac   )

                    loss_grn = (W2_MASK    * l_mask    +
                                W2_SPARSE  * l_sparse  +
                                W2_ACCESS  * l_access  +
                                W2_ACCESS_GENE * l_access_gene +
                                W2_DIVERSE * l_diverse +
                                W2['ANCHOR'] * anchor_scale * l_anchor +
                                W2_LD      * l_ld      +
                                W2_LD_REG  * l_ld_reg)

                    opt_grn.zero_grad()
                    opt_vel.zero_grad()
                    loss_grn.backward(retain_graph=True)
                    loss_vel.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(grn.parameters()) + list(cell_encoder.parameters()) +
                        list(ld_decoder.parameters()), 1.0
                    )
                    torch.nn.utils.clip_grad_norm_(vel_head.parameters(), 1.0)
                    opt_grn.step()
                    opt_vel.step()
                    epoch_losses.append(float((loss_grn + loss_vel).item()))
                    
            sch_grn.step()
            if phase == 2:
                sch_vel.step()
    
            if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
                avg = np.mean(epoch_losses)
                loss_history.append(avg)
                print(f"  [P{phase}] epoch {epoch:03d}  loss={avg:.4f}  anchor_scale={anchor_scale:.2f}")
                term_avgs = {k: np.mean(v) for k, v in epoch_terms.items()}
                print("           " + "  ".join(f"{k}={v:.4f}" for k, v in term_avgs.items()))
    
            if epoch % PLOT_EVERY == 0 or epoch == N_EPOCHS - 1:
                grn.eval(); vel_head.eval(); cell_encoder.eval()
                with torch.no_grad():
                    mw_snap  = cell_encoder(X_t, temperature=1.0)
                    tf_lead_all = lead_t[:, tf_indices_in_G]
                    diseq_snap = lead_t + train_config["LAMBDA_GRN"] * grn.production_from_lead(
                        tf_lead_all, mw_snap, prune_frac=train_config["GRN_PRUNE_FRAC"])
                    V_snap   = vel_head(X_t, diseq_snap).numpy()
                # X_full passed here MUST be in the same z-scored space as the velocity
                # (rna_z) — passing raw X silently reweights every gene by its own
                # standard deviation and rotates the projected arrow off-direction.
                V_snap_2d = project_to_2d(V_snap, V_dec2, rna_z, search_space=Z_dec, k=15, scale=1.0)
                gif_maker.add_frame(V_dec2, V_snap_2d, pseudo_norm, epoch)
                grn.train(); vel_head.train(); cell_encoder.train()
    
        # ── Saving Outputs ────────────────────────────────────────────────────
        grn.eval(); vel_head.eval(); cell_encoder.eval()
        with torch.no_grad():
            full_weights = cell_encoder(X_t, temperature=1.0)
            tf_lead_all = lead_t[:, tf_indices_in_G]
            disequil_full = lead_t + train_config["LAMBDA_GRN"] * grn.production_from_lead(
                tf_lead_all, full_weights, prune_frac=train_config["GRN_PRUNE_FRAC"])
            V_pred       = vel_head(X_t, disequil_full).cpu().numpy()
            steady_full = grn.predict_rna(X_TF_t, full_weights).cpu().numpy()
            A_dict_np    = grn.A_dict.cpu().numpy()
            A_base_np = grn.A_base.cpu().numpy()
            w_mean = full_weights.mean(0)
            A_learned = grn.effective_A(w_mean).cpu().numpy()           # (G, T) population GRN
            all_mix_weights = full_weights.cpu().numpy()

            h_full   = cell_encoder.encoder(X_t)           # (C, 64)
            z_2d_full, _ = ld_decoder.forward(h_full)
            z_2d_np  = z_2d_full.cpu().numpy()
            gene_loadings = ld_decoder.get_gene_loadings()  # (G, 2)

        # ── Final 2D streamline projection (Step 16 of the reference notebook) ──
        # Drift-corrected correlation-kernel projection into Decipher 2D, then a
        # light 70/30 self/neighbor blend for a clean field — no masking here,
        # this only averages the already-learned per-cell vectors.
        V_proj2d = project_to_2d(V_pred, V_dec2, rna_z, search_space=Z_dec, k=20,
                                  scale=1.0, drift_correct=True)
        _, idx_vis = NearestNeighbors(n_neighbors=31).fit(V_dec2).kneighbors(V_dec2)
        idx_vis = idx_vis[:, 1:]
        V_stream_2d = 0.7 * V_proj2d + 0.3 * V_proj2d[idx_vis].mean(1)

        target_dirs = [base_path] if NUM_RUNS == 1 else [os.path.join(base_path, f"run{run_i}")]
        if NUM_RUNS > 1 and run_i == 0:
            target_dirs.append(base_path)

        for d in target_dirs:
            os.makedirs(d, exist_ok=True)
            np.save(f"{d}/V_pred.npy",        V_pred)
            np.save(f"{d}/A_dict.npy",        A_dict_np)          # (D, G, T) atom bank
            np.save(f"{d}/A_base.npy", A_base_np)
            np.save(f"{d}/A_learned.npy", A_learned)
            np.save(f"{d}/mix_weights.npy",   all_mix_weights)    # (C, D) per-cell
            np.save(f"{d}/steady_state.npy",  steady_full)
            np.save(f"{d}/A_genes.npy",       A_genes)
            np.save(f"{d}/A_tfs.npy",         A_tfs)
            np.save(f"{d}/gene_idx.npy",      gene_idx)
            np.save(f"{d}/tf_idx.npy",        tf_idx)
            np.save(f"{d}/loss_history.npy",  np.array(loss_history))
            np.save(f"{d}/z_ld_2d.npy",      z_2d_np)
            np.save(f"{d}/gene_loadings.npy", gene_loadings)
            np.save(f"{d}/A_dict_init.npy",   A_dict_init_np) # for analysis
            gif_maker.save(f"{d}/velocity_training.gif", fps=3)
            gif_maker.save_slide_png(f"{d}/velocity_slide_final.png")
            plot_velocity_stream(V_dec2, V_stream_2d, pseudo_norm,
                                  save_path=f"{d}/velocity_stream.png")

        print(f"Run {run_i} complete. Outputs saved to: {', '.join(target_dirs)}")

    with torch.no_grad():
        # --- RNA cells ---
        h_rna = cell_encoder.encoder(X_t)           # (C_rna, 64)
        z_rna, _ = ld_decoder(h_rna)
        z_rna = z_rna.cpu().numpy()                  # (C_rna, 2)

        # --- ATAC cells: project gene_activity through the same encoder ---
        # gene_activity is (C_atac, G) — same gene space as X, so encoder applies directly
        gene_act_t = torch.tensor(gene_activity, dtype=torch.float32)
        h_atac = cell_encoder.encoder(gene_act_t)    # (C_atac, 64)
        z_atac, _ = ld_decoder(h_atac)
        z_atac = z_atac.cpu().numpy()                # (C_atac, 2)

        # --- Center and scale for plotting (doesn't affect saved z_2d_np) ---
        # We fit scaler on RNA only so RNA occupies the center; ATAC is transformed consistently
        scaler = StandardScaler()
        z_rna_plot  = scaler.fit_transform(z_rna)
        z_atac_plot = scaler.transform(z_atac)
        
        gene_loadings = ld_decoder.get_gene_loadings()   # (G, 2)
        
        # Scale loadings to match the standardised space
        # (W_dec maps from 2D to gene space; the transpose gives gene directions in 2D)
        # Magnitude of each gene's loading arrow
        loading_scale = 2.0 / (np.abs(gene_loadings).max() + 1e-8)
        gene_loadings_plot = gene_loadings * loading_scale
        
        plot_ld_embedding(
            z_2d_rna    = z_rna_plot,
            pseudo_norm = pseudo_norm,
            gene_loadings = gene_loadings_plot,
            A_genes     = A_genes,
            z_2d_atac   = z_atac_plot,
            highlight_genes = ['CD34', 'MPO', 'LYZ', 'GATA1', 'SPI1', 'CEBPA'],
            save_path   = f"{d}/ld_embedding_biplot.png"
        )
        
        
        print(f"RNA z_2d std per dim: {z_rna.std(axis=0).round(3)}")
        print(f"ATAC z_2d std per dim: {z_atac.std(axis=0).round(3)}")
        print("If stds are similar, both dims are active. If one is near 0, collapse persists.")

    
    print("Training complete.")

    return {
        "V_pred": V_pred,
        "A_dict": A_dict_np,
        "loss_history": loss_history,
    }