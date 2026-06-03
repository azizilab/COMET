from config import DATA, TRAIN, MODEL
from data.load import *

#from model.grn import GRNDictionary
#from model.encoder import CellStateEncoder
#from model.velocity import DictVelocityHead
#from losses.grn_losses import masked_reconstruction_loss
#from losses.velocity_losses import velocity_consistency_loss

def main():

    raw = load_datasets(DATA)

    # gene / tf universe
    universe = build_universe(
        raw["rna_adata"],
        raw["motif_tf_names"],
        raw["peak_gene"],
        raw["peaks"],
        raw["peaks_atac"],
    )

    
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

    # tf accessibility prior
    TF_access_prior = build_tf_access_prior(
        gene_prior_bundle["gene_activity"],
        M_bundle["A_tfs"],     
        universe["gene_to_idx_full"],
        universe["sub_g2i"],
    )

    data = {
        **universe,
        **M_bundle,
        **gene_prior_bundle,
        "TF_access_prior": TF_access_prior,
    }
    print("Data loading complete.")



if __name__ == "__main__":
    data = main()