#!/usr/bin/env python3
# gnn_fit_panel.py
import argparse, math, os, sys, random
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

# ----------------------------
# Utilities
# ----------------------------
def to_numpy(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)

def build_target_to_gene_index(adata: ad.AnnData, target_label: str) -> Dict[str, int]:
    """
    Map each perturbation label to a gene index IF the label is a gene present in var_names.
    Non-gene labels will be ignored (they can still be part of the training set, but Step0 will
    clamp nothing for that sample). For your panel, we expect labels == gene symbols.
    """
    varset = set(adata.var_names)
    t2i = {}
    for t in adata.obs[target_label].unique():
        if t in varset:
            t2i[t] = int(np.where(adata.var_names == t)[0][0])
    return t2i

def sample_minibatch(
    X_ctrl: np.ndarray,
    X_pert: np.ndarray,
    pert_labels: np.ndarray,
    control_label: str,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Returns batch_x_ctrl (B,G), batch_x_pert (B,G), batch_targets (list of labels).
    Matches each perturbed cell to a random control cell.
    """
    B = batch_size
    # indices for perturbed cells (exclude controls)
    pert_mask = pert_labels != control_label
    pert_idx = np.where(pert_mask)[0]
    if len(pert_idx) < B:
        choice = rng.choice(pert_idx, size=B, replace=True)
    else:
        choice = rng.choice(pert_idx, size=B, replace=False)
    # random controls
    ctrl_idx = np.where(pert_labels == control_label)[0]
    rand_ctrl = rng.choice(ctrl_idx, size=B, replace=True)
    bx_ctrl = torch.from_numpy(X_ctrl[rand_ctrl]).float()
    bx_pert = torch.from_numpy(X_pert[choice]).float()
    btargets = pert_labels[choice].tolist()
    return bx_ctrl, bx_pert, btargets

def make_base_adjacency(G: int, self_loops: bool = True) -> torch.Tensor:
    """
    Dense fully-connected adjacency (uniform), normalized row-wise.
    We'll mask rows per-sample to forbid inbound messages to the target.
    """
    A = torch.ones(G, G)
    if not self_loops:
        A.fill_diagonal_(0.0)
    # row-normalize so each node aggregates an average of neighbors
    A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
    return A

# ----------------------------
# Model: Step0 + MPNN + Readout
# ----------------------------
class Step0Clamp(nn.Module):
    """
    Simple Step-0: clamp the target node toward an anchor 'tau' with learnable efficacy alpha in (0,1).
    For CRISPRi-like behavior, tau=0.0 (in normalized space).
    """
    def __init__(self, tau: float = 0.0):
        super().__init__()
        # single learnable alpha (could be made per-target or small MLP if desired)
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid -> ~0.5 initially
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 when unknown (i.e., target label not a gene)
        """
        B, G = x_ctrl.shape
        x0 = x_ctrl.clone()
        alpha = torch.sigmoid(self.logit_alpha)  # (scalar)
        if (target_idx >= 0).any():
            bmask = (target_idx >= 0)
            rows = torch.arange(B, device=x_ctrl.device)[bmask]
            cols = target_idx[bmask]
            # x_t := (1 - alpha) * x_ctrl_t + alpha * tau
            x0[rows, cols] = (1.0 - alpha) * x_ctrl[rows, cols] + alpha * self.tau
        return x0

class MPNNLayer(nn.Module):
    """
    Basic MPNN layer with dense adjacency.
    h_in -> aggregate (A @ h_in) -> update with residual
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.upd = nn.Linear(2 * hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, A_batch: torch.Tensor, h_t_frozen: torch.Tensor) -> torch.Tensor:
        """
        h:        (B,G,C)
        A_batch:  (B,G,G) row-normalized, with row[target]=0 for each sample
        h_t_frozen: (B,1,C) the clamped target embedding to re-impose after update
        """
        # messages
        m = torch.matmul(A_batch, self.msg(h))  # (B,G,C)
        h_new = self.act(self.upd(torch.cat([h, m], dim=-1)))  # (B,G,C)
        # residual
        h_out = h + h_new
        # re-impose frozen target state
        # gather: replace the row corresponding to target with frozen
        # h_t_frozen is provided already extracted as h[:, t, :].unsqueeze(1) after Step-0 embed
        # We assume caller already zeroed inbound to t in A_batch.
        # Concatenate by slicing to avoid scatter for speed on small G
        # (But we need indices; we’ll do it in the caller for clarity.)
        return h_out

class GeneMPNN(nn.Module):
    def __init__(self, G: int, hidden: int = 128, T: int = 2, tau: float = 0.0):
        super().__init__()
        self.G = G
        self.hidden = hidden
        self.T = T
        # per-node input is scalar (expression); use a shared linear to lift to hidden
        self.embed = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden) for _ in range(T)])
        self.readout = nn.Linear(hidden, 1)
        self.step0 = Step0Clamp(tau=tau)

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor, A_base: torch.Tensor) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 where unknown
        A_base: (G,G) dense row-normalized base adjacency
        """
        device = x_ctrl.device
        B, G = x_ctrl.shape
        assert G == self.G

        # Step-0 clamp in expression space
        x0 = self.step0(x_ctrl, target_idx)  # (B,G)

        # Initial hidden state (shared 1->hidden linear applied per gene)
        h = self.embed(x0.unsqueeze(-1))  # (B,G,hidden)

        # Prepare per-sample adjacency (block inbound to target)
        # Start from base A, then zero the row 't' per sample.
        A_batch = A_base.unsqueeze(0).repeat(B, 1, 1).to(device)  # (B,G,G)
        # keep a copy of target embeddings to re-impose after each layer
        # If target_idx == -1, we won’t freeze anything; we’ll handle with a mask.
        freeze_mask = (target_idx >= 0)
        if freeze_mask.any():
            rows = torch.arange(B, device=device)[freeze_mask]
            cols = target_idx[freeze_mask]
            A_batch[rows, cols, :] = 0.0  # zero inbound to target (row=t)

        # Save the frozen target embedding (after Step-0 embed)
        # If some samples lack known target, we’ll just skip the replacement.
        h_t0 = torch.zeros(B, 1, self.hidden, device=device)
        if freeze_mask.any():
            h_t0[freeze_mask] = h[rows, cols].unsqueeze(1)

        # Run T layers with reimposition of target state
        for layer in self.layers:
            h = layer(h, A_batch, h_t0)
            if freeze_mask.any():
                # put frozen target embedding back
                h[rows, cols] = h_t0[freeze_mask, 0]

        # Readout back to expression space
        y = self.readout(h).squeeze(-1)  # (B,G)
        return y, x0  # return x0 for optional locality loss

# ----------------------------
# Losses
# ----------------------------
def mse_loss(yhat, y):
    return F.mse_loss(yhat, y)

def target_consistency_loss(yhat, x_ctrl, target_idx, mode="knockdown", margin=0.0):
    """
    Encourage correct direction at the target:
      knockdown: yhat[t] <= x_ctrl[t] - margin
      activation: yhat[t] >= x_ctrl[t] + margin
    """
    if (target_idx < 0).sum() == target_idx.numel():
        return yhat.new_tensor(0.0)
    rows = torch.arange(target_idx.numel(), device=yhat.device)[target_idx >= 0]
    cols = target_idx[target_idx >= 0]
    y_t = yhat[rows, cols]
    x_t = x_ctrl[rows, cols]
    if mode == "activation":
        # hinge: max(0, (x+margin) - y)
        return F.relu((x_t + margin) - y_t).mean()
    # default knockdown
    return F.relu(y_t - (x_t - margin)).mean()

def locality_damping(yhat, x0, target_idx, k_mask=None, weight=1.0):
    """
    Penalize changes far from target. Simplest form: L1 over all non-target genes.
    You can pass a boolean k-hop mask (B,G) with True where penalty applies less (or zero near t).
    For now, just exclude the target index itself.
    """
    B, G = yhat.shape
    loss = 0.0
    for b in range(B):
        t = int(target_idx[b].item())
        if t >= 0:
            mask = torch.ones(G, dtype=torch.bool, device=yhat.device)
            mask[t] = False
            loss = loss + (yhat[b, mask] - x0[b, mask]).abs().mean()
    return (loss / max((target_idx >= 0).sum().item(), 1)) * weight

# ----------------------------
# Training
# ----------------------------
def train(
    adata: ad.AnnData,
    target_label: str,
    control_label: str,
    hidden: int = 128,
    T: int = 2,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_target: float = 0.1,
    weight_local: float = 0.0,
    seed: int = 0,
    tau: float = 0.0,
    device: str = "cuda",
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Data arrays
    X = to_numpy(adata.X).astype(np.float32)  # assume normalized/log space already
    labels = adata.obs[target_label].astype(str).values
    G = adata.n_vars

    # Index pools
    ctrl_mask = labels == control_label
    if ctrl_mask.sum() == 0:
        raise ValueError("No control cells found.")
    # perturbed pool includes all non-controls (even if target gene not found)
    pert_mask = ~ctrl_mask
    if pert_mask.sum() == 0:
        raise ValueError("No perturbed cells found.")

    X_ctrl = X  # we’ll pick rows via indices
    X_pert = X
    pert_labels = labels

    # Map perturbation label -> gene index (for Step-0); unknown => -1
    t2gi = build_target_to_gene_index(adata, target_label)
    # Precompute a tensor of target indices per cell
    tgt_idx = np.full(adata.n_obs, -1, dtype=np.int64)
    for i, lab in enumerate(labels):
        tgt_idx[i] = t2gi.get(lab, -1)

    # Model
    model = GeneMPNN(G=G, hidden=hidden, T=T, tau=tau).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Base adjacency (fully connected, row-normalized)
    A_base = make_base_adjacency(G, self_loops=True).to(device)

    # Simple schedule
    steps_per_epoch = math.ceil(pert_mask.sum() / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"mse": 0.0, "targ": 0.0, "loc": 0.0, "tot": 0.0}
        for step in range(steps_per_epoch):
            bx_ctrl, bx_pert, btargets = sample_minibatch(
                X_ctrl=X_ctrl, X_pert=X_pert, pert_labels=pert_labels,
                control_label=control_label, batch_size=batch_size, rng=rng
            )
            # per-sample target index tensor
            tidx = torch.tensor([t2gi.get(t, -1) for t in btargets], dtype=torch.long)

            bx_ctrl = bx_ctrl.to(device)
            bx_pert = bx_pert.to(device)
            tidx = tidx.to(device)

            yhat, x0 = model(bx_ctrl, tidx, A_base)

            loss_mse = mse_loss(yhat, bx_pert)
            loss_t = target_consistency_loss(yhat, bx_ctrl, tidx, mode="knockdown", margin=0.0)
            loss_loc = locality_damping(yhat, x0, tidx, weight=1.0) if weight_local > 0 else yhat.new_tensor(0.0)

            loss = loss_mse + weight_target * loss_t + weight_local * loss_loc

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["mse"]  += float(loss_mse.item())
            running["targ"] += float(loss_t.item())
            running["loc"]  += float(loss_loc.item())
            running["tot"]  += float(loss.item())

        denom = max(steps_per_epoch, 1)
        print(f"[epoch {epoch:03d}] "
              f"mse={running['mse']/denom:.5f}  "
              f"targ={running['targ']/denom:.5f}  "
              f"loc={running['loc']/denom:.5f}  "
              f"total={running['tot']/denom:.5f}")

    return model

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--T", type=int, default=2, help="Number of message-passing steps.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_target", type=float, default=0.1)
    ap.add_argument("--weight_local", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0, help="Step-0 anchor (e.g., 0.0 for CRISPRi).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.in_h5ad)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway

    model = train(
        adata=adata,
        target_label=args.target_label,
        control_label=args.control_label,
        hidden=args.hidden,
        T=args.T,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_target=args.weight_target,
        weight_local=args.weight_local,
        seed=args.seed,
        tau=args.tau,
        device=args.device,
    )

    # Optional: save weights
    out_path = os.path.splitext(args.in_h5ad)[0] + f".mpnn_hidden{args.hidden}_T{args.T}.pt"
    torch.save({"state_dict": model.state_dict(),
                "G": model.G,
                "hidden": model.hidden,
                "T": model.T}, out_path)
    print(f"[done] saved model to {out_path}")

if __name__ == "__main__":
    main()
