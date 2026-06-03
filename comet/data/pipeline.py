def build_pipeline(config):
    raw = load_datasets(config)

    universe = build_universe(
        raw["rna_adata"],
        raw["motif_tf_names"],
        raw["peak_gene"],
        raw["peaks"],
        raw["peaks_atac"],
    )

    M_bundle = build_M(
        raw["peak_gene"],
        raw["peak_tf"],
        universe["X"],
        universe["A_genes"],
        universe["A_tfs"],
        universe["peak_to_idx"],
        universe["sub_g2i"],
        universe["gene_to_idx_full"],
        universe["motif_tf_to_idx"]
    )

    prior_bundle = build_gene_priors(
        universe,
        raw["atac_X"],
    )

    tf_prior = build_tf_access_prior(
        prior_bundle["gene_activity"],
        universe["A_tfs"],
        universe["gene_to_idx_full"],
        universe["gene_names_all"],
        universe["sub_g2i"]
    )

    return {
        **universe,
        **M_bundle,
        **prior_bundle,
        "TF_access_prior": tf_prior,
    }