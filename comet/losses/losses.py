import torch
import torch.nn.functional as F


def masked_reconstruction_loss(grn, mix_weights, rna_batch, tf_batch, tf_indices_in_G, mask_frac=0.40):
    B, Gg = rna_batch.shape; device = rna_batch.device
    mask = torch.rand(B, Gg, device=device) < mask_frac
    tf_input_masked = tf_batch.clone(); batch_idx = torch.arange(B, device=device).unsqueeze(1)
    tf_input_masked[batch_idx, torch.arange(len(tf_indices_in_G), device=device)] *= \
        (~mask[batch_idx, tf_indices_in_G]).float()
    rna_pred = grn.predict_rna(tf_input_masked, mix_weights)
    if mask.sum() == 0: return rna_pred.mean() * 0.0
    return F.mse_loss(rna_pred[mask], rna_batch[mask])

def velocity_consistency_loss(velocity, disequil):
    """Align displacement with the chromatin-led disequilibrium; guard magnitude."""
    v = F.normalize(velocity, dim=1); d = F.normalize(disequil, dim=1)
    cos = (v * d).sum(1)
    mag = torch.abs(velocity.norm(dim=1) - disequil.norm(dim=1).detach()).mean()
    return (1.0 - cos).mean() + 0.1 * mag

def atac_direction_loss(velocity, gene_access_batch):
    """A closed locus should not be pushed up (cheap consistency guard)."""
    closed = 1.0 - gene_access_batch.to(velocity.device)
    return (closed * torch.relu(velocity)).mean()

def committor_consistency_loss(velocity, commit_dir, valid):
    """Direction-only: velocity should follow the committor's forward (increasing-q) direction.
    Invalid (flat-committor) cells are skipped so root/terminal plateaus don't inject noise.
    NOTE: with the committor now baked into the velocity's INPUT (orient_disequilibrium below),
    this term is only a light reinforcement — not the mechanism that steers the flow."""
    if valid.sum() == 0: return velocity.sum() * 0.0
    v = F.normalize(velocity[valid], dim=1); d = F.normalize(commit_dir[valid], dim=1)
    return (1.0 - (v * d).sum(1)).mean()


def orient_disequilibrium(disequil, commit_dir, valid, beta=0.5):
    """Committor-oriented arrow of time — the fix that makes the committor actually steer.

    The raw ATAC→RNA / GRN disequilibrium carries the gene program and per-cell magnitude, but its
    SIGN is locally ambiguous (az−rz can point forward or backward per cell), which is what produces
    the "arrows randomly reversed to meet in the middle" failure. Here we DECOMPOSE the disequilibrium
    into its component along the committor's forward direction u and its orthogonal remainder, then
    rebuild a strictly-forward vector:

        oriented_i = ||d_i|| · normalize( u_i + beta · ĥ(d_i^perp) )

    so the arrow is guaranteed to have a positive committor component (no backflow / no reversal),
    keeps the GRN/ATAC speed ||d_i||, and is tilted inside the forward cone by the committor-
    orthogonal, GRN-derived structure (branch selection the scalar committor cannot express).
    Cells with a flat/invalid committor keep the raw disequilibrium."""
    u    = F.normalize(commit_dir, dim=1)                     # committor forward unit (per cell)
    dot  = (disequil * u).sum(1, keepdim=True)                # projection onto committor axis
    dperp = disequil - dot * u                                # GRN-orthogonal refinement
    dperp_hat = F.normalize(dperp, dim=1)
    mag  = disequil.norm(dim=1, keepdim=True)                 # keep GRN/ATAC per-cell speed
    oriented = F.normalize(u + beta * dperp_hat, dim=1) * mag # strictly-forward cone, GRN-tilted
    return torch.where(valid.view(-1, 1), oriented, disequil)
