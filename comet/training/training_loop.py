import json
import os

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from model.encoder import CellStateEncoder
from model.grn import GRNDictionary, initialize_grn_dictionary_from_clusters, masked_reconstruction_loss
from model.velocity import (
    MarkovProgressionField,
    compute_global_disc_order,
    psi_polarity_loss,
    psi_fine_order_loss,
    build_2d_neighbor_index,
    compute_markov_velocity_2d,
)
from model.ld_vae import LinearDecoder, plot_ld_embedding
from utils.graph import build_neighbor_index, graph_smooth
from utils.geometry import build_local_atac


def run_training(data, model_config, train_config):

    X        = data["X"]           # normalised scRNA-seq expression matrix (C x G)
    X_TF     = data["X_TF"]        # TF-expression sub-matrix (C x T)
    M        = data["M"]           # binary motif mask (G x T)

    gene_activity    = data["gene_activity"]    # per-cell local chromatin accessibility (C x G)
    gene_access      = data["gene_access"]      # population-mean accessibility (G,)
    TF_access_prior  = data["TF_access_prior"]  # population-mean TF locus accessibility (T,)
    TF_access_t      = torch.tensor(TF_access_prior, dtype=torch.float32)

    A_genes   = data["A_genes"]
    A_tfs     = data["A_tfs"]
    gene_idx  = data["gene_idx"]
    tf_idx    = data["tf_idx"]
    sub_g2i   = data["sub_g2i"]
    rna_adata = data["rna_adata"]

    # ── TF index within A_genes ───────────────────────────────────────────────
    tf_indices_in_G = []
    for tf_name in A_tfs:
        tf_indices_in_G.append(sub_g2i[tf_name] if tf_name in sub_g2i else -1)
    tf_indices_in_G = torch.tensor(tf_indices_in_G, dtype=torch.long)

    decipher_data = data["decipher_outputs"]
    Z_dec  = decipher_data["z"].astype(np.float32)   # (C, 10) latent space
    V_dec2 = decipher_data["v"].astype(np.float32)   # (C, 2)  vis space

    # ── Config ───────────────────────────────────────────────────────────────
    NUM_RUNS      = train_config["NUM_RUNS"]
    N_EPOCHS      = train_config["N_EPOCHS"]
    PHASE1_EPOCHS = train_config["PHASE1_EPOCHS"]
    BATCH_SIZE    = train_config["BATCH_SIZE"]
    D_ATOMS       = model_config["D_ATOMS"]
    CLIP_NORM     = train_config.get("CLIP_NORM", 5.0)
    PLOT_EVERY    = train_config["PLOT_EVERY"]
    base_path     = train_config["OUTPUT_DIR"]
    USE_LOCAL_ATAC = train_config["USE_LOCAL_ATAC"]

    # Phase 3 config
    PHASE3_EPOCHS     = train_config.get("PHASE3_EPOCHS", 500)
    PHASE3_LR         = train_config.get("PHASE3_LR", 1e-1)
    K_PSI_SMOOTH      = train_config.get("K_PSI_SMOOTH", 30)
    N_PSI_SMOOTH_HOPS = train_config.get("N_PSI_SMOOTH_HOPS", 3)
    W3                = train_config.get("WEIGHTS_PHASE3", {})
    W3_POLARITY       = W3.get("POLARITY",   1.0)
    W3_FINE_ORDER     = W3.get("FINE_ORDER", 0.0)
    W3_EPS_REG        = W3.get("EPS_REG",    0.1)
    ETA_MIN_P3        = train_config.get("LR_SCHED", {}).get("ETA_MIN", 1e-3)

    W1 = train_config["WEIGHTS_PHASE1"]
    W2 = train_config["WEIGHTS_PHASE2"]

    W1_MASK    = W1["MASK"];    W2_MASK    = W2["MASK"]
    W1_SPARSE  = W1["SPARSE"];  W2_SPARSE  = W2["SPARSE"]
    W1_ACCESS  = W1["ACCESS"];  W2_ACCESS  = W2["ACCESS"]
    W1_DIVERSE = W1["DIVERSE"]; W2_DIVERSE = W2["DIVERSE"]
    W1_ENTROPY = W1["ENTROPY"]; W2_DIV     = W2["DIV"]
    W1_DIRICH  = W1["DIRICH"];  W2_DIRICH  = W2["DIRICH"]
    W1_LD      = W1["LD"];      W2_LD      = W2["LD"]
    W1_LD_REG  = W1["LD_REG"]; W2_LD_REG  = W2["LD_REG"]

    os.makedirs(base_path, exist_ok=True)

    # ── Local ATAC / global fallback ──────────────────────────────────────────
    if USE_LOCAL_ATAC:
        local_access, _ = build_local_atac(
            X, gene_activity, V_dec2, n_clusters=train_config["N_CLUSTERS"]
        )
        gene_access_t = torch.tensor(local_access, dtype=torch.float32)   # (C, G)
    else:
        gene_access_t = torch.tensor(gene_access, dtype=torch.float32)    # (G,)

    # ── Latent-space kNN for ψ smoothing (K_PSI_SMOOTH neighbors) ────────────
    # Uses Decipher latent Z (10-D) for higher fidelity than 2D vis space.
    neighbor_indices_t = build_neighbor_index(
        X, V_dec2, Z_dec, mode="latent", k=K_PSI_SMOOTH
    )
    print(f"Latent-space kNN (for psi smoothing): {neighbor_indices_t.shape}")

    # ── Tensors and DataLoader ────────────────────────────────────────────────
    G = X.shape[1]
    T = X_TF.shape[1]

    X_t        = torch.tensor(X,    dtype=torch.float32)
    X_TF_t     = torch.tensor(X_TF, dtype=torch.float32)
    cell_idx_t = torch.arange(X_t.size(0))

    dataset = TensorDataset(X_t, X_TF_t, cell_idx_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=RandomSampler(dataset, replacement=False),
                         drop_last=True)

    # Pseudotime for optional LD-embedding plot at end of training
    pseudo_norm = None
    if "facs_pseudotime" in rna_adata.obs.columns:
        pseudo = rna_adata.obs["facs_pseudotime"].astype(np.float32)
        pseudo_norm = (pseudo.values - pseudo.min()) / (pseudo.values.max() - pseudo.values.min() + 1e-8)

    all_run_loss_histories = {}
    all_run_flow_histories = {}

    for run_i in range(NUM_RUNS):
        # ── Skip if already complete ──────────────────────────────────────────
        run_dir = os.path.join(base_path, f"run{run_i}")
        psi_path = os.path.join(run_dir, "psi.npy")
        if os.path.exists(psi_path):
            print(f"\n::: Run {run_i} already exists. Skipping. :::")
            continue

        print(f"\n::: Starting Run {run_i} :::")
        print(f"Phase 1: {PHASE1_EPOCHS} epochs (GRN + encoder warmup)")
        print(f"Phase 2: {N_EPOCHS - PHASE1_EPOCHS} epochs (GRN + encoder + LD-decoder refinement)")
        print(f"Phase 3: {PHASE3_EPOCHS} full-batch steps (Markov progression field)")

        # ── Model construction ────────────────────────────────────────────────
        grn          = GRNDictionary(G=G, T=T, M=M, D=D_ATOMS)
        grn.TF_access = TF_access_t.clone()
        grn.mean_tf   = torch.tensor(X_TF.mean(axis=0), dtype=torch.float32)
        cell_encoder  = CellStateEncoder(G=G, D=D_ATOMS)
        ld_decoder    = LinearDecoder(hidden=model_config.get("LD_HIDDEN", 64), G=G, ld_dim=2)

        # ── Cluster-based initialization ──────────────────────────────────────
        cluster_labels, W_init = initialize_grn_dictionary_from_clusters(
            X=X, V_dec2=V_dec2, M=M, D=D_ATOMS, A_tfs=A_tfs, sub_g2i=sub_g2i
        )
        grn.init_from_clusters(W_init)
        cell_encoder.init_from_cluster_labels(X, cluster_labels, n_epochs=30)

        # ── Optimizers — separate Phase 1 / Phase 2 LRs ──────────────────────
        params_grn = (list(grn.parameters()) + list(cell_encoder.parameters())
                      + list(ld_decoder.parameters()))
        opt_p1 = torch.optim.AdamW(params_grn, lr=train_config["LR_GRN"], weight_decay=1e-4)
        opt_p2 = torch.optim.AdamW(params_grn, lr=train_config["LR_GRN"] * 0.4, weight_decay=1e-4)

        loss_history = {k: [] for k in ["epoch", "total", "mask", "diversity", "phase"]}

        # ── Phase 1 + Phase 2: GRN / encoder / LD-decoder ────────────────────
        # Phase 2 does NOT train any velocity head. Its only job is to deliver
        # stable, well-separated mix_weights to Phase 3.
        for epoch in range(N_EPOCHS):
            phase = 1 if epoch < PHASE1_EPOCHS else 2

            # Temperature anneals over Phase 1 only; Phase 2 is fixed at 0.2
            if phase == 1:
                temp = max(0.1, 0.5 - 0.4 * epoch / PHASE1_EPOCHS)
                opt  = opt_p1
                W_MASK, W_SPARSE, W_ACCESS, W_DIVERSE = W1_MASK, W1_SPARSE, W1_ACCESS, W1_DIVERSE
                W_ENT, W_DIRICH, W_LD, W_LD_REG = W1_ENTROPY, W1_DIRICH, W1_LD, W1_LD_REG
                W_DIV_PHASE = None   # Phase 1 uses normalized entropy (W_ENT * H/log(D))
            else:
                temp = 0.2
                opt  = opt_p2
                W_MASK, W_SPARSE, W_ACCESS, W_DIVERSE = W2_MASK, W2_SPARSE, W2_ACCESS, W2_DIVERSE
                W_ENT, W_DIRICH, W_LD, W_LD_REG = None, W2_DIRICH, W2_LD, W2_LD_REG
                W_DIV_PHASE = W2_DIV   # Phase 2 uses DIV weight on normalized entropy

            if epoch == PHASE1_EPOCHS:
                print("\n── Switching to Phase 2 (GRN/encoder refinement, no velocity) ──")

            grn.train(); cell_encoder.train(); ld_decoder.train()

            epoch_losses = {"total": 0.0, "mask": 0.0, "diversity": 0.0, "n_batches": 0}

            for rna_b, tf_b, batch_idx in loader:
                mix_w = cell_encoder(rna_b, temperature=temp)

                l_mask    = masked_reconstruction_loss(
                    grn=grn, mix_weights=mix_w, rna_batch=rna_b, tf_batch=tf_b,
                    tf_indices_in_G=tf_indices_in_G, mask_frac=0.30
                )
                l_sparse  = grn.sparsity_loss()
                l_diverse = grn.diversity_loss()
                l_access  = grn.tf_access_consistency_loss()
                l_dirich  = cell_encoder.dirichlet_sharpening_loss(mix_w)

                H      = -torch.sum(mix_w * torch.log(mix_w + 1e-8), dim=-1).mean()
                H_norm = H / np.log(D_ATOMS)   # normalized entropy

                h_enc     = cell_encoder.encoder(rna_b)
                l_ld      = ld_decoder.reconstruction_loss(h_enc, rna_b)
                l_ld_reg  = ld_decoder.ld_regularization_loss(h_enc)

                if phase == 1:
                    loss = (W_MASK    * l_mask    +
                            W_SPARSE  * l_sparse  +
                            W_ACCESS  * l_access  +
                            W_DIVERSE * l_diverse +
                            W_ENT     * H_norm    +
                            W_DIRICH  * l_dirich  +
                            W_LD      * l_ld      +
                            W_LD_REG  * l_ld_reg)
                else:
                    loss = (W_MASK    * l_mask    +
                            W_SPARSE  * l_sparse  +
                            W_ACCESS  * l_access  +
                            W_DIVERSE * l_diverse +
                            W_DIV_PHASE * H_norm  +
                            W_DIRICH  * l_dirich  +
                            W_LD      * l_ld      +
                            W_LD_REG  * l_ld_reg)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_grn, max_norm=CLIP_NORM)
                opt.step()

                epoch_losses["total"]     += float(loss.item())
                epoch_losses["mask"]      += float(l_mask.item())
                epoch_losses["diversity"] += float(l_diverse.item())
                epoch_losses["n_batches"] += 1

            nb = epoch_losses.pop("n_batches")
            for k_ in epoch_losses:
                epoch_losses[k_] /= nb

            loss_history["epoch"].append(epoch)
            for k_ in ["total", "mask", "diversity"]:
                loss_history[k_].append(epoch_losses[k_])
            loss_history["phase"].append(phase)

            if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
                tag = "P1" if phase == 1 else "P2"
                print(f"  [{tag}] Epoch {epoch:03d}  total={epoch_losses['total']:.1f}  "
                      f"mask={epoch_losses['mask']:.1f}  diversity={epoch_losses['diversity']:.3f}")

        all_run_loss_histories[f"run{run_i}"] = loss_history

        # ── Phase 3: Markov Progression Field ────────────────────────────────
        print("\n── Phase 3 (Markov progression field) ──")
        print("  Computing global atom discordance order...")
        grn.eval(); cell_encoder.eval()

        global_disc_order, global_disc, dom_atom_full = compute_global_disc_order(
            grn=grn, X_full_t=X_t, gene_access_full_t=gene_access_t,
            encoder=cell_encoder, top_k_genes=100,
        )
        origin_atom = int(global_disc.argmax().item())
        print(f"  origin_atom = {origin_atom}  "
              f"(largest discordance gap → psi global minimum)")

        with torch.no_grad():
            mix_w_full = cell_encoder(X_t, temperature=0.1)   # (N, D), frozen for Phase 3

        mvf    = MarkovProgressionField(D=D_ATOMS, origin_atom=origin_atom)
        opt_p3 = torch.optim.Adam(mvf.parameters(), lr=PHASE3_LR)
        sched_p3 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_p3, T_max=PHASE3_EPOCHS, eta_min=ETA_MIN_P3
        )

        flow_history = {"step": [], "total": [], "polarity": [], "fine_order": []}
        for step in range(PHASE3_EPOCHS):
            psi_raw_full = mvf.psi_raw(mix_w_full)
            psi_full     = graph_smooth(psi_raw_full, neighbor_indices_t,
                                        n_hops=N_PSI_SMOOTH_HOPS)

            l_pol  = psi_polarity_loss(psi_full, dom_atom_full, origin_atom)
            l_fine = (torch.zeros((), device=psi_full.device)
                      if W3_FINE_ORDER == 0 else
                      psi_fine_order_loss(psi_full, dom_atom_full,
                                          global_disc_order, D_ATOMS))
            l_eps  = (mvf.eps_raw ** 2).mean()
            loss3  = W3_POLARITY * l_pol + W3_FINE_ORDER * l_fine + W3_EPS_REG * l_eps

            opt_p3.zero_grad()
            loss3.backward()
            opt_p3.step()
            sched_p3.step()

            flow_history["step"].append(step)
            flow_history["total"].append(float(loss3.item()))
            flow_history["polarity"].append(float(l_pol.item()))
            flow_history["fine_order"].append(float(l_fine.item()))

            if step % 50 == 0 or step == PHASE3_EPOCHS - 1:
                with torch.no_grad():
                    P_now = mvf.atom_progression().numpy()
                    psi_check = graph_smooth(mvf.psi_raw(mix_w_full), neighbor_indices_t,
                                             n_hops=N_PSI_SMOOTH_HOPS)
                    origin_mean = float(psi_check[dom_atom_full == origin_atom].mean())
                    branch_mean = float(psi_check[dom_atom_full != origin_atom].mean())
                print(f"  [P3] Step {step:03d}  total={loss3.item():.4f}  "
                      f"polarity={l_pol.item():.4f}  "
                      f"ψ_origin={origin_mean:.3f}  ψ_branches={branch_mean:.3f}  "
                      f"P={P_now.round(3)}")

        all_run_flow_histories[f"run{run_i}"] = flow_history

        # ── Compute final outputs ─────────────────────────────────────────────
        grn.eval(); cell_encoder.eval()
        with torch.no_grad():
            psi_raw_final = mvf.psi_raw(mix_w_full)
            psi_final_t   = graph_smooth(psi_raw_final, neighbor_indices_t,
                                          n_hops=N_PSI_SMOOTH_HOPS)
            P_final       = mvf.atom_progression().numpy()

            # Population-weighted mean GRN across all cells
            w_np = mix_w_full.detach().cpu().numpy()           # (N, D)
            A_dict_list = [grn.A_dict[d].detach().cpu().numpy() for d in range(D_ATOMS)]
            A_learned = sum(
                w_np[:, d:d+1, np.newaxis] * A_dict_list[d][np.newaxis]
                for d in range(D_ATOMS)
            ).mean(axis=0)   # (G, T)

            # LD embedding for optional plotting
            h_full    = cell_encoder.encoder(X_t)
            z_2d_full, _ = ld_decoder(h_full)
            z_2d_np   = z_2d_full.detach().cpu().numpy()
            gene_loadings = ld_decoder.get_gene_loadings()    # (G, 2)

        # ── Save outputs ──────────────────────────────────────────────────────
        os.makedirs(run_dir, exist_ok=True)

        np.save(f"{run_dir}/psi.npy",         psi_final_t.detach().numpy())
        np.save(f"{run_dir}/A_learned.npy",   A_learned)
        np.save(f"{run_dir}/mix_weights.npy", w_np)
        np.save(f"{run_dir}/gene_idx.npy",    gene_idx)
        np.save(f"{run_dir}/tf_idx.npy",      tf_idx)
        np.save(f"{run_dir}/A_genes.npy",     A_genes)
        np.save(f"{run_dir}/A_tfs.npy",       A_tfs)
        np.save(f"{run_dir}/z_ld_2d.npy",    z_2d_np)
        np.save(f"{run_dir}/gene_loadings.npy", gene_loadings)
        np.save(f"{run_dir}/loss_history.npy",
                np.array(loss_history["total"]))

        with open(f"{run_dir}/flow_field_params.json", "w") as fjson:
            json.dump({
                "origin_atom":      origin_atom,
                "P":                P_final.tolist(),
                "disc":             global_disc.numpy().tolist(),
                "direction_matrix": global_disc_order.numpy().tolist(),
            }, fjson, indent=2)

        print(f"\nRun {run_i} complete. Outputs saved to: {run_dir}")

        # ── Optional LD-VAE biplot (uses pseudotime if available) ────────────
        if pseudo_norm is not None:
            with torch.no_grad():
                h_rna = cell_encoder.encoder(X_t)
                z_rna, _ = ld_decoder(h_rna)
                z_rna = z_rna.cpu().numpy()

                gene_act_t = torch.tensor(gene_activity, dtype=torch.float32)
                h_atac = cell_encoder.encoder(gene_act_t)
                z_atac, _ = ld_decoder(h_atac)
                z_atac = z_atac.cpu().numpy()

            scaler     = StandardScaler()
            z_rna_plot  = scaler.fit_transform(z_rna)
            z_atac_plot = scaler.transform(z_atac)

            loading_scale = 2.0 / (np.abs(gene_loadings).max() + 1e-8)
            gene_loadings_plot = gene_loadings * loading_scale

            plot_ld_embedding(
                z_2d_rna      = z_rna_plot,
                pseudo_norm   = pseudo_norm,
                gene_loadings = gene_loadings_plot,
                A_genes       = A_genes,
                z_2d_atac     = z_atac_plot,
                highlight_genes = ['CD34', 'MPO', 'LYZ', 'GATA1', 'SPI1', 'CEBPA'],
                save_path     = f"{run_dir}/ld_embedding_biplot.png"
            )

    print("Training complete.")

    # Return psi and mix_weights from the last run for downstream use
    return {
        "psi":             psi_final_t.detach().numpy(),
        "A_learned":       A_learned,
        "mix_weights":     w_np,
        "loss_histories":  all_run_loss_histories,
        "flow_histories":  all_run_flow_histories,
    }
