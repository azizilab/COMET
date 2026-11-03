import torch
import torch.nn.functional as F


def sample_gene_mask(rna_batch, mask_frac=0.40):
    """(B, G) Bernoulli(mask_frac) gene mask; True = masked (hidden from the encoder and the TF inputs, and scored)"""
    return torch.rand(rna_batch.shape, device=rna_batch.device) < mask_frac


def masked_reconstruction_loss(grn, mix_weights, rna_batch, tf_batch, tf_indices_in_G, mask):
    """MSE on the masked entries; each masked TF gene's input is zeroed. mix_weights must come from the encoder applied
    to the masked profile (rna_batch * ~mask), so no masked target reaches the prediction."""
    B = rna_batch.shape[0]; device = rna_batch.device
    tf_input_masked = tf_batch.clone(); batch_idx = torch.arange(B, device=device).unsqueeze(1)
    tf_input_masked[batch_idx, torch.arange(len(tf_indices_in_G), device=device)] *= \
        (~mask[batch_idx, tf_indices_in_G]).float()
    rna_pred = grn.predict_rna(tf_input_masked, mix_weights)
    if mask.sum() == 0: return rna_pred.mean() * 0.0
    return F.mse_loss(rna_pred[mask], rna_batch[mask])
