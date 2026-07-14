from config import DATA, MODEL, TRAIN
from data.load import load_datasets, build_universe, build_M, build_gene_priors, build_tf_access_prior
from training.training_loop import run_training

def main():

    print("beginning data loading...")

    raw = load_datasets(DATA)

    print("Data loaded.")

    # gene / tf universe
    universe = build_universe(
        raw["rna_adata"],
        raw["motif_tf_names"],
        raw["peak_gene"],
        raw["peaks"],
        raw["peaks_atac"],
        rna_layer=DATA["rna_layer"],
    )

    print("universe built.")

    # accessibility mask M
    M_bundle = build_M(
        raw["peaks"],
        raw["peak_gene"],
        raw["peak_tf"],
        universe["X_full"],
        universe["A_genes"],
        universe["A_tfs"],
        universe["peak_to_idx"],
        universe["sub_g2i"],
        universe["gene_to_idx_full"],
        universe["motif_tf_to_idx"],
    )

    print("Mask constructed.")

    # gene accessibility prior
    gene_prior_bundle = build_gene_priors(
        universe["peak_names"],
        universe["A_genes"],
        raw["peak_gene"],
        universe["peak_to_idx_atac"],
        universe["X"],
        raw["atac_X"],
        universe["sub_g2i"],
    )

    print("Gene priors constructed.")

    # tf accessibility prior
    TF_access_prior = build_tf_access_prior(
        gene_prior_bundle["gene_activity"],
        M_bundle["A_tfs"],     
        universe["gene_to_idx_full"],
        universe["sub_g2i"],
    )

    print("TF priors constructed.")

    data = {
        **raw,
        **universe,
        **M_bundle,
        **gene_prior_bundle,
        "TF_access_prior": TF_access_prior,
    }
    
    print("Data loading complete.")

    outputs = run_training(data, MODEL, TRAIN)

    return outputs


if __name__ == "__main__":
    data = main()