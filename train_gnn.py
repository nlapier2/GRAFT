
#!/usr/bin/env python3
"""
Skeleton trainer for the GRAFT GNN model (v1).
This wires: Step-0, a simple state propagator, mediated + direct heads, and core losses.
Most components are stubs with clear TODOs to fill in.
"""
import os, argparse, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from graft.data.dataset import GraftDataset
from graft.data.samplers import LabBalancedSampler
from graft.models.step0 import StepZeroClamp
from graft.models.gnn_core import StatePropagator
from graft.models.heads import MediatedHead, SparseDirectHead
from graft.losses.distribution import sliced_wasserstein
from graft.losses.invariance import risk_extrapolation
from graft.losses.consistency import target_knockdown_consistency
from graft.utils.common import load_z_and_meta

def load_U(path):
    U = np.load(path).astype(np.float32)
    # Expect shape (F, G) from factor encoder; convert to (G, F)
    if U.shape[0] < U.shape[1]:
        U = U  # F x G
        U = U.T  # G x F
    else:
        pass
    return torch.from_numpy(U)

class GraftCore(nn.Module):
    def __init__(self, z_dim: int, G: int, U: torch.Tensor, hidden=256, gnn_layers=2, use_factor=False, a_dim=0, n_labs=1):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)  # freeze for v1
        self.propagator = StatePropagator(z_dim, hidden=hidden, layers=gnn_layers)
        self.med = MediatedHead(z_dim, F=U.size(1), hidden=hidden, use_factor_feats=use_factor, a_dim=a_dim)
        self.dir = SparseDirectHead(z_dim, G=G, hidden=hidden)
        self.step0 = StepZeroClamp(z_dim=z_dim, n_labs=n_labs, hidden=64, init_eff=0.9)

    def forward(self, z, x0, lab_ids, target_idx, a=None):
        z_ref = self.propagator(z)
        # Step-0 clamp
        x_clamped, eff = self.step0(x0, z_ref, lab_ids, target_idx)
        # Heads
        m = self.med(z_ref, a)
        dx_med = torch.matmul(m, self.U.T)  # (B, F) * (F, G) = (B, G)
        dx_dir = self.dir(z_ref)
        x_pred = x_clamped + dx_med + dx_dir
        return x_pred, {"eff": eff, "m": m, "dx_med": dx_med, "dx_dir": dx_dir, "x_clamped": x_clamped}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))

    # Load inputs
    z, meta = load_z_and_meta(cfg["paths"]["scvi_latent"], cfg["paths"]["scvi_meta"])
    # Expect z index = cell ids; meta has lab_id/dataset_id/tech_batch_id
    train_df = pd.read_parquet(cfg["paths"]["train_perturb"])
    val_df = pd.read_parquet(cfg["paths"]["val_perturb"])

    # Gene list from scVI input (optional; placeholder)
    genes = np.array([f"gene_{i}" for i in range(1000)])

    train = GraftDataset(train_df, z, meta, genes)
    val = GraftDataset(val_df, z, meta, genes)

    labs = np.unique(train.labs)
    lab_to_idx = train.split_by_lab()
    sampler = LabBalancedSampler(lab_to_idx, batch_size=cfg["training"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = load_U(cfg["paths"]["factor_U"]).to(device)  # (G, F)

    z_dim = train.z.shape[1]
    G = len(genes)
    model = GraftCore(z_dim=z_dim, G=G, U=U, hidden=cfg["model"]["hidden"], gnn_layers=cfg["model"]["gnn_layers"],
                      use_factor=cfg["model"]["use_factor_features"], a_dim=0, n_labs=len(labs)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    def batch_to_tensors(idx):
        # TODO: fetch real tensors
        B = len(idx)
        z_b = torch.from_numpy(train.z.values[idx]).to(device)
        x0 = torch.zeros(B, G, device=device)  # TODO: scVI decoded controls/pre-state
        y = torch.zeros(B, G, device=device)   # TODO: post-state normalized expression
        lab_ids = torch.zeros(B, dtype=torch.long, device=device)  # TODO: map lab strings to ints
        target_idx = torch.full((B,), -1, dtype=torch.long, device=device)  # -1 = control
        return z_b, x0, y, lab_ids, target_idx

    max_epochs = cfg["training"]["max_epochs"]
    for epoch in range(1, max_epochs+1):
        for step, idx in zip(range(100), sampler):  # limit steps per epoch for skeleton
            z_b, x0, y, lab_ids, target_idx = batch_to_tensors(idx)
            y_pred, aux = model(z_b, x0, lab_ids, target_idx)

            L_dist = sliced_wasserstein(y_pred, y, n_proj=32)
            L_kd = target_knockdown_consistency(y_pred, y, target_idx, weight=cfg["training"]["kd_consistency_weight"])

            L = cfg["training"]["dist_loss_weight"] * L_dist + L_kd
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        print(f"[epoch {epoch}] loss={float(L):.4f}  (dist={float(L_dist):.4f}, kd={float(L_kd):.4f})")

    print("[DONE] skeleton training loop finished (stub data). Fill data fetching and losses as discussed.")

if __name__ == "__main__":
    main()
