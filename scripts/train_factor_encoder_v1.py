# scripts/train_factor_encoder_v1.py
# Minimal training loop for the FactorEncoder with on-the-fly scVI normalization.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from utils.scvi_stream import ScviOnTheFly
from models.factor_encoder import FactorEncoder, FactorEncoderConfig, build_W0_from_membership

def batch_index_loader(n_obs: int, batch_size: int, shuffle: bool, seed: int = 13):
    """
    Load a batch of cell indices (not data) for n_obs total rows.
    """
    sampler = RandomSampler(range(n_obs), generator=torch.Generator().manual_seed(seed)) if shuffle \
              else SequentialSampler(range(n_obs))
    return BatchSampler(sampler, batch_size=batch_size, drop_last=False)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-type", required=True)
    ap.add_argument("--model-dir", default=None, help="scVI model dir (default: artifacts/scvi_<CELLTYPE>/)")
    ap.add_argument("--scvi-input", default=None, help="scVI input h5ad (default: artifacts/scvi_input_<CELLTYPE>.h5ad)")
    ap.add_argument("--z-parquet", default=None, help="Latent parquet (default: artifacts/scvi_z_<CELLTYPE>.parquet)")
    ap.add_argument("--membership-npy", required=True, help="Path to membership matrix M.npy (F_anchor x G)")
    ap.add_argument("--epochs-warm", type=int, default=5)
    ap.add_argument("--epochs-joint", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr-enc", type=float, default=1e-3)
    ap.add_argument("--lr-W", type=float, default=1e-4)
    ap.add_argument("--lambda-prior", type=float, default=1e-3)
    ap.add_argument("--lambda-W", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cell = args.cell_type
    model_dir  = args.model_dir  or f"artifacts/scvi_{cell}"
    scvi_input = args.scvi_input or f"artifacts/scvi_input_{cell}.h5ad"
    z_path     = args.z_parquet  or f"artifacts/scvi_z_{cell}.parquet"

    # Load z latents (float32) and align to AnnData order
    z_df = pd.read_parquet(z_path)
    z_df = z_df.astype(np.float32)
    z = z_df.values
    z_idx = z_df.index

    # scVI streaming helper
    scvi_stream = ScviOnTheFly(model_dir=model_dir, scvi_input_h5ad=scvi_input, library_size=1e4)
    # Reindex z to match scVI anndata order (required!)
    pos = scvi_stream.align_z_index(z_idx)
    if (pos < 0).any():
        missing = int((pos < 0).sum())
        raise RuntimeError(f"{missing} z rows not found in scVI AnnData obs_names.")
    z = z[pos, :]  # reorder to match scVI adata
    n_obs, z_dim = z.shape
    G = scvi_stream.n_vars

    # Load membership matrix M (F_anchor x G)
    M = np.load(args.membership_npy).astype(np.float32)
    if M.shape[1] != G:
        raise ValueError(f"M has {M.shape[1]} genes but scVI has {G}. Ensure same gene order.")
    M_t = torch.from_numpy(M).to(args.device)

    # Build W0 and mask; init model
    W0, anchor_mask = build_W0_from_membership(M_t, n_free=16, add_junk=True)
    cfg = FactorEncoderConfig(
        n_genes=G, n_anchor=M.shape[0], n_free=16, add_junk=True,
        mlp_hidden=64, mlp_layers=2,
        alpha_cons=1.0, lambda_ridge=0.1,
        # turn off L1-style anchors for v1
        lambda_out=0.0, lambda_in=0.0,
        beta_recon=1e-2, gamma_cov=1e-3,
        # NEW: simple dense anchor regularization
        lambda_prior=args.lambda_prior, lambda_W=args.lambda_W,
    )

    model = FactorEncoder(W0=W0, anchor_mask=anchor_mask, z_dim=z_dim, cfg=cfg).to(args.device)

    # Optims
    opt_enc = torch.optim.AdamW(list(model.encoder.parameters()) + [model.a_bias], lr=args.lr_enc, weight_decay=1e-4)
    opt_W   = torch.optim.AdamW([model.W_param], lr=args.lr_W, weight_decay=1e-4)

    def run_epoch(phase: str, freeze_W: bool):
        model.train()
        if freeze_W:
            for p in [model.W_param]:
                p.requires_grad_(False)
        else:
            model.W_param.requires_grad_(True)

        total = 0.0
        loader = batch_index_loader(n_obs, args.batch_size, shuffle=True)
        for idx_batch in loader:
            idx = np.fromiter(idx_batch, dtype=np.int64)
            zb = torch.from_numpy(z[idx]).to(args.device, non_blocking=True)

            # Stream x̄ for just these rows (NumPy -> torch)
            xbar_np = scvi_stream.get_xbar(indices=idx, return_numpy=True)
            xbar = torch.from_numpy(xbar_np).to(args.device, non_blocking=True)

            # (Optional) quick per-gene z-score using running stats or precomputed stats.
            # For v1, skip or compute a simple scale on-the-fly:
            xbar_scaled = None

            # Zero grads
            opt_enc.zero_grad()
            if not freeze_W:
                opt_W.zero_grad()

            loss, parts = model.compute_losses(z=zb, xbar=xbar, xbar_scaled=xbar_scaled, freeze_W=freeze_W)
            loss.backward()
            opt_enc.step()
            if not freeze_W:
                opt_W.step()

            total += float(parts["L_total"].cpu().item())

        return total

    # Warm-start
    for e in range(args.epochs_warm):
        tot = run_epoch("warm", freeze_W=True)
        print(f"[warm] epoch {e+1}/{args.epochs_warm} total={tot:.4f}")

    # Joint
    for e in range(args.epochs_joint):
        tot = run_epoch("joint", freeze_W=False)
        print(f"[joint] epoch {e+1}/{args.epochs_joint} total={tot:.4f}")

    # Save learned W and a small report
    os.makedirs("artifacts", exist_ok=True)
    W_final = model.forward(torch.from_numpy(z[:1]).to(args.device))["W"].detach().cpu().numpy()
    np.save(f"artifacts/factor_W_{cell}.npy", W_final)
    print(f"[OK] saved W to artifacts/factor_W_{cell}.npy")

if __name__ == "__main__":
    main()
