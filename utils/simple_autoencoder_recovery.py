#!/usr/bin/env python3
"""
simple_autoencoder_signal_recovery.py

Train a tiny autoencoder on a single AnnData:
- Input: concat([log1p(counts), one-hot(target_gene)]).
- Target: log1p(counts).
- Output: writes a new AnnData with predictions replacing .X.
  Optionally write predictions in "counts" space via expm1.

Example:
  python simple_autoencoder_signal_recovery.py \
    --input /path/to/input.h5ad \
    --output /path/to/output_pred.h5ad \
    --latent-dim 32 --hidden 256 --epochs 5 --batch-size 2048 --output-space counts
"""

import argparse
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# Utilities
# ---------------------------

def to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)

def one_hot_from_obs(obs_series: pd.Series):
    cats = obs_series.astype("category").cat.categories
    idx = obs_series.astype("category").cat.codes.values  # -1 for NaN
    n = len(obs_series)
    k = len(cats)
    oh = np.zeros((n, k), dtype=np.float32)
    mask = idx >= 0
    oh[np.where(mask)[0], idx[mask]] = 1.0
    return oh, list(map(str, cats))

class AE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, latent=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .h5ad")
    ap.add_argument("--output", required=True, help="Output .h5ad (predictions replace .X)")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--output-space", choices=["counts", "log1p"], default="counts",
                    help="Write predictions back as counts (expm1) or in log1p space.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load
    A = ad.read_h5ad(args.input)
    genes = A.var_names.astype(str)
    obs = A.obs.copy()
    if "target_gene" not in obs.columns:
        raise ValueError("adata.obs['target_gene'] is required.")

    # Build inputs
    X = to_dense(A.X).astype(np.float32)
    X_log = np.log1p(X, dtype=np.float32)

    # One-hot for target_gene (strings ok)
    tgt_oh, tgt_cats = one_hot_from_obs(obs["target_gene"])
    # Concatenate features: [log1p(X) | one-hot(target_gene)]
    X_in = np.concatenate([X_log, tgt_oh], axis=1).astype(np.float32)
    Y = X_log  # target is log1p counts

    # Dataset / loader
    ds = TensorDataset(torch.from_numpy(X_in), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_in.shape[1]
    out_dim = Y.shape[1]
    model = AE(in_dim, out_dim, hidden=args.hidden, latent=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # L1 is robust for this; MSE works too
    loss_fn = nn.L1Loss()

    # Train
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n_batches = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        print(f"[epoch {epoch}] L1={total / max(1, n_batches):.4f}")

    # Predict (full batch for simplicity)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_in).to(device)
        yhat_log = model(xb).cpu().numpy().astype(np.float32)

    # Post-process for output
    if args.output_space == "counts":
        yhat = np.expm1(yhat_log, dtype=np.float32)
        np.maximum(yhat, 0.0, out=yhat)  # clamp small negatives
        # Keep memory reasonable: store CSR
        X_out = sparse.csr_matrix(yhat)
    else:
        X_out = sparse.csr_matrix(yhat_log)

    # Write new AnnData with SAME fields as input, just replacing .X
    A_out = ad.AnnData(
        X=X_out,
        obs=obs,          # unchanged
        var=A.var.copy(), # unchanged
        uns=A.uns.copy() if A.uns is not None else None,
        obsm=A.obsm.copy() if A.obsm is not None else None,
        varm=A.varm.copy() if A.varm is not None else None,
        layers={k: v.copy() for k, v in (A.layers.items() if A.layers is not None else [])},
    )
    A_out.var_names = genes

    # Save
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    A_out.write_h5ad(args.output, compression="lzf")
    print(f"[done] wrote predictions to: {args.output}")

if __name__ == "__main__":
    main()
