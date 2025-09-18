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

def sample_minibatch_knn(
    X: np.ndarray,
    labels: np.ndarray,
    control_label: str,
    batch_size: int,
    rng: np.random.Generator,
    knn_k: int = 32,
    knn_temp: float = 0.1,
    metric: str = "l2",
    pre_norm_ctrl: np.ndarray | None = None,
    pre_norm_pert: np.ndarray | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Choose B perturbed cells; for each, sample one control from its top-k nearest controls
    with softmax(-dist / temp). Supports L2 or cosine. Small-panel friendly (no index lib).
    """
    # index pools
    ctrl_idx = np.where(labels == control_label)[0]
    pert_idx = np.where(labels != control_label)[0]
    if len(pert_idx) == 0 or len(ctrl_idx) == 0:
        raise ValueError("Need both control and perturbed cells.")
    # choose pert rows
    if len(pert_idx) < batch_size:
        sel_pert = rng.choice(pert_idx, size=batch_size, replace=True)
    else:
        sel_pert = rng.choice(pert_idx, size=batch_size, replace=False)
    Xp = X[sel_pert]  # (B,G)
    Xc = X[ctrl_idx]  # (Nc,G)
    # distances (B,Nc)
    if metric == "cosine":
        # use pre-normalized rows if provided
        P = pre_norm_pert if pre_norm_pert is not None else (Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8))
        C = pre_norm_ctrl if pre_norm_ctrl is not None else (Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8))
        # cosine distance = 1 - cosine similarity
        dists = 1.0 - (P @ C.T)
    else:  # l2
        # (a-b)^2 = a^2  b^2 - 2ab
        a2 = np.sum(Xp * Xp, axis=1, keepdims=True)        # (B,1)
        b2 = np.sum(Xc * Xc, axis=1, keepdims=True).T      # (1,Nc)
        dists = a2 + b2 - 2.0 * (Xp @ Xc.T)                # (B,Nc)
        dists = np.maximum(dists, 0.0)
    # top-k and softmax sampling
    B = sel_pert.shape[0]
    sel_ctrl = np.empty(B, dtype=int)
    for i in range(B):
        di = dists[i]
        k = min(knn_k, di.shape[0])
        cand = np.argpartition(di, k-1)[:k]      # indices of k smallest
        # softmax over -dist / temp  (smaller distance => higher prob)
        logits = -di[cand] / max(knn_temp, 1e-6)
        logits -= logits.max()                   # stabilize
        p = np.exp(logits)
        p /= p.sum()
        sel_ctrl[i] = rng.choice(cand, p=p)
    # pack tensors
    bx_ctrl = torch.from_numpy(Xc[sel_ctrl]).float()
    bx_pert = torch.from_numpy(Xp).float()
    btargets = labels[sel_pert].tolist()
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

# Add near your utilities
def collapse_to_pseudobulk(adata, target_label: str):
    """Return a new AnnData with one row per label (perturbation + control)."""
    import pandas as pd
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    df = pd.DataFrame(X, columns=adata.var_names).groupby(labels).mean()
    from anndata import AnnData
    ad_bulk = AnnData(df.values.astype(np.float32))
    ad_bulk.var_names = adata.var_names.copy()
    ad_bulk.obs[target_label] = df.index.astype(str)
    return ad_bulk

# ----------------------------
# Model: Step0 + MPNN + Readout
# ----------------------------
class Step0Clamp(nn.Module):
    """
    Simple Step-0: clamp the target node toward an anchor 'tau' with learnable efficacy alpha in (0,1).
    For CRISPRi-like behavior, tau=0.0 (in normalized space).
    """
    def __init__(self, tau: float = 0.0, num_perts: int = None):
        super().__init__()
        # global alpha by default; if num_perts is given, use per-pert embedding for alpha
        if num_perts is not None:
            self.alpha_table = nn.Embedding(num_perts, 1)
            nn.init.zeros_(self.alpha_table.weight)  # sigmoid(0)=0.5
        else:
            self.alpha_table = None
            self.logit_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid -> ~0.5 initially
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor, pert_rowidx: torch.Tensor = None,
                alpha_override: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 when unknown (i.e., target label not a gene)
        """
        B, G = x_ctrl.shape
        x0 = x_ctrl.clone()
        if alpha_override is not None:
            alpha = alpha_override  # (B,)
        elif self.alpha_table is not None and pert_rowidx is not None:
             alpha = torch.sigmoid(self.alpha_table(pert_rowidx)).view(-1)  # (B,)
        else:
            alpha = torch.sigmoid(self.logit_alpha).expand(B)  # (B,)
        if (target_idx >= 0).any():
            bmask = (target_idx >= 0)
            rows = torch.arange(B, device=x_ctrl.device)[bmask]
            cols = target_idx[bmask]
            # multiplicative knockdown on counts: x0_t = log1p( m * (exp(x_ctrl_t)-1) )
            # where m = (1 - alpha) ∈ (0,1). With tau≈0 counts, this is the correct semantics.
            ctrl_lin = torch.expm1(x_ctrl[rows, cols].clamp_min(0.0))
            m = (1.0 - alpha).expand_as(rows.float()) if alpha.dim() == 0 else (1.0 - alpha[bmask])
            x0_lin = m * ctrl_lin                       # tau=0 → just multiply counts
            x0[rows, cols] = torch.log1p(x0_lin)
        return x0

class PrototypeGenerator(nn.Module):
    """
    Gene-conditioned generator:
      - learns a per-gene embedding e_t in R^d
      - maps e_t to a prototype mean-effect vector b_t in R^G
      - predicts a Step-0 efficacy alpha_t in (0,1) from e_t (and optional meta covariates)
    """
    def __init__(self, G: int, d: int = 64):
        super().__init__()
        self.G = G
        self.E = nn.Embedding(G, d)
        nn.init.normal_(self.E.weight, std=0.02)
        # e_t -> gene-space prototype
        self.W_out = nn.Linear(d, G, bias=False)
        # e_t (+ meta) -> alpha
        self.alpha_head = nn.Sequential(
            nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, target_idx: torch.LongTensor, meta: torch.Tensor | None = None):
        """
        target_idx: (B,) gene indices (>=0); if -1, we still produce something but it won't be used.
        meta: optional (B, M) covariates; ignored in this minimal version.
        Returns:
          b: (B,G) prototype vector
          alpha: (B,) efficacy in (0,1)
          e_t: (B,d) gene embeddings
        """
        e_t = self.E(torch.clamp(target_idx, min=0))  # (B,d); clamp just to index safely
        b = self.W_out(e_t)                           # (B,G)
        alpha = torch.sigmoid(self.alpha_head(e_t)).squeeze(-1)  # (B,)
        return b, alpha, e_t


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
        self.norm = nn.LayerNorm(hidden_dim)
        self.beta = nn.Parameter(torch.tensor(0.5))  # start with a gentle mix of neighbor info

    def forward(self, h: torch.Tensor, A_batch: torch.Tensor, h_t_frozen: torch.Tensor) -> torch.Tensor:
        """
        h:        (B,G,C)
        A_batch:  (B,G,G) row-normalized, with row[target]=0 for each sample
        h_t_frozen: (B,1,C) the clamped target embedding to re-impose after update
        """
        # messages
        m = self.beta * torch.matmul(A_batch, self.msg(h))  # (B,G,C)
        h_new = self.act(self.upd(torch.cat([h, m], dim=-1)))  # (B,G,C)
        # residual
        h_out = self.norm(h + h_new)
        # re-impose frozen target state
        # gather: replace the row corresponding to target with frozen
        # h_t_frozen is provided already extracted as h[:, t, :].unsqueeze(1) after Step-0 embed
        # We assume caller already zeroed inbound to t in A_batch.
        # Concatenate by slicing to avoid scatter for speed on small G
        # (But we need indices; we’ll do it in the caller for clarity.)
        return h_out

class GeneMPNN(nn.Module):
    def __init__(self, G: int, hidden: int = 128, T: int = 2, tau: float = 0.0, num_perts: int = None):
        super().__init__()
        self.G = G
        self.hidden = hidden
        self.T = T
        # per-node input is scalar (expression); use a shared linear to lift to hidden
        self.embed = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden) for _ in range(T)])
        self.readout = nn.Linear(hidden, 1)
        # gene-conditioned prototype & alpha; FiLM from gene embedding
        self.proto = PrototypeGenerator(G=G, d=64)
        self.film_gamma = nn.Linear(64, hidden)
        self.film_beta  = nn.Linear(64, hidden)
        self.step0 = Step0Clamp(tau=tau, num_perts=None)  # no per-pert table anymore

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor, A_base: torch.Tensor, pert_rowidx: torch.Tensor = None) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 where unknown
        A_base: (G,G) dense row-normalized base adjacency
        """
        device = x_ctrl.device
        B, G = x_ctrl.shape
        assert G == self.G

        # Gene-conditioned prototype & efficacy
        b_proto, alpha_t, e_t = self.proto(target_idx, meta=None)   # (B,G), (B,), (B,64)
        # Step-0 clamp in expression space using alpha_t
        x0 = self.step0(x_ctrl, target_idx, pert_rowidx=None, alpha_override=alpha_t)  # (B,G)

        # Initial hidden state (shared 1->hidden linear applied per gene)
        h = self.embed(x0.unsqueeze(-1))  # (B,G,hidden)
        # FiLM condition on target gene embedding (broadcast across genes)
        gamma = torch.tanh(self.film_gamma(e_t)).unsqueeze(1)  # (B,1,H)
        beta  = self.film_beta(e_t).unsqueeze(1)               # (B,1,H)
        h = h * (1 + gamma) + beta

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
        # add gene-conditioned prototype mean-effect
        y = y + b_proto
        # Preserve Step-0 at the target: y_t := x0_t
        freeze_mask = (target_idx >= 0)
        if freeze_mask.any():
            rows = torch.arange(B, device=device)[freeze_mask]
            cols = target_idx[freeze_mask]
            y[rows, cols] = x0[rows, cols]
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
    w_proto: float = 0.2,
    seed: int = 0,
    tau: float = 0.0,
    device: str = "cuda",
    match_controls: str = "knn",  # or "random"
    knn_k: int = 32,
    knn_temp: float = 0.1,
    knn_metric: str = "l2",  # or "cosine"
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
    # Precompute normalized rows for cosine kNN (one-time)
    pre_norm_ctrl = pre_norm_pert = None

    # Map perturbation label -> gene index (for Step-0); unknown => -1
    t2gi = build_target_to_gene_index(adata, target_label)
    # Precompute a tensor of target indices per cell
    tgt_idx = np.full(adata.n_obs, -1, dtype=np.int64)
    for i, lab in enumerate(labels):
        tgt_idx[i] = t2gi.get(lab, -1)

    # Model
    # Build stable mapping from label -> embedding row
    pert_names_unique = sorted(set(labels.tolist()))
    pert2row = {p: i for i, p in enumerate(pert_names_unique)}
    num_perts = len(pert_names_unique)

    ctrl_mean_np = X[ctrl_mask].mean(axis=0).astype(np.float32)
    ctrl_mean = torch.from_numpy(ctrl_mean_np).to(device)

    # Model
    model = GeneMPNN(G=G, hidden=hidden, T=T, tau=tau, num_perts=num_perts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Base adjacency (fully connected, row-normalized)
    A_base = make_base_adjacency(G, self_loops=True).to(device)

    # Simple schedule
    steps_per_epoch = math.ceil(pert_mask.sum() / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"mse": 0.0, "targ": 0.0, "loc": 0.0, "proto": 0.0, "tot": 0.0}
        for step in range(steps_per_epoch):
            if match_controls == "knn":
                # lazily build cosine norms if requested
                if knn_metric == "cosine":
                    if pre_norm_ctrl is None:
                        ctrl_idx_all = np.where(labels == control_label)[0]
                        pre_norm_ctrl = X[ctrl_idx_all] / (np.linalg.norm(X[ctrl_idx_all], axis=1, keepdims=True) + 1e-8)
                        pre_norm_pert = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                bx_ctrl, bx_pert, btargets = sample_minibatch_knn(
                    X=X, labels=labels, control_label=control_label,
                    batch_size=batch_size, rng=rng,
                    knn_k=knn_k, knn_temp=knn_temp, metric=knn_metric,
                    pre_norm_ctrl=pre_norm_ctrl, pre_norm_pert=pre_norm_pert
                )
            else:
                bx_ctrl, bx_pert, btargets = sample_minibatch(
                    X_ctrl=X_ctrl, X_pert=X_pert, pert_labels=pert_labels,
                    control_label=control_label, batch_size=batch_size, rng=rng
                )
            # per-sample target index tensor
            tidx = torch.tensor([t2gi.get(t, -1) for t in btargets], dtype=torch.long)

            bx_ctrl = bx_ctrl.to(device)
            bx_pert = bx_pert.to(device)
            tidx = tidx.to(device)

            # per-sample perturbation row indices for embedding / FiLM
            pert_rowidx = torch.tensor([pert2row[t] for t in btargets], dtype=torch.long, device=device)
            yhat, x0 = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx)

            # --- Prototype / bulk-delta loss (aligns to PDS) ---
            # Group rows by perturbation label in this batch
            from collections import defaultdict
            by_lbl = defaultdict(list)
            for i, lbl in enumerate(btargets):
                by_lbl[lbl].append(i)

            loss_proto = yhat.new_tensor(0.0)
            groups = 0
            for lbl, idxs in by_lbl.items():
                if len(idxs) < 1:
                    continue
                idx = torch.tensor(idxs, device=yhat.device, dtype=torch.long)
                # predicted / true deltas relative to the model's own baseline x0
                d_pred = (yhat[idx] - x0[idx]).mean(dim=0)      # (G,)
                d_true = (bx_pert[idx] - x0[idx]).mean(dim=0)   # (G,)
                # optional: exclude this perturbation's target gene from the loss
                # (only if label matches a gene in the panel)
                t = t2gi.get(lbl, -1) if 't2gi' in locals() else -1
                if t >= 0:
                    mask = torch.ones_like(d_pred, dtype=torch.bool)
                    mask[t] = False
                d_pred = (yhat[idx].mean(dim=0) - ctrl_mean)    # (G,)
                d_true = (bx_pert[idx].mean(dim=0) - ctrl_mean) # (G,)
                loss_proto = loss_proto + torch.mean(torch.abs(d_pred - d_true))
                groups += 1
            if groups > 0:
                loss_proto = loss_proto / groups

            loss_mse = mse_loss(yhat - x0, bx_pert - x0)
            loss_t = target_consistency_loss(yhat, bx_ctrl, tidx, mode="knockdown", margin=0.0)
            loss_loc = locality_damping(yhat, x0, tidx, weight=1.0) if weight_local > 0 else yhat.new_tensor(0.0)

            loss = loss_mse + weight_target * loss_t + weight_local * loss_loc + w_proto * loss_proto

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["mse"]  += float(loss_mse.item())
            running["targ"] += float(loss_t.item())
            running["loc"]  += float(loss_loc.item())
            running["proto"] += float(loss_proto.item())
            running["tot"]  += float(loss.item())

        denom = max(steps_per_epoch, 1)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[epoch {epoch:03d}] "
                f"mse={running['mse']/denom:.5f}  "
                f"targ={running['targ']/denom:.5f}  "
                f"loc={running['loc']/denom:.5f}  "
                f"proto={running['proto']/denom:.5f}  "
                f"total={running['tot']/denom:.5f}")

    return model


@torch.no_grad()
def predict_all_perturbations(
    adata: ad.AnnData,
    model: nn.Module,
    target_label: str,
    control_label: str,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 0,
):
    """
    For every perturbed cell, match a random control, run model, and collect predictions.
    Returns:
      pred_mat: (N_pert, G) predicted expressions (aligned to perturbed rows)
      true_mat: (N_pert, G) true perturbed expressions
      pert_names: list[str] of length N_pert (labels for each row)
      ctrl_mean: (G,) global control pseudobulk (mean of all control cells)
    """
    rng = np.random.default_rng(seed)
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    G = adata.n_vars

    # pools
    ctrl_idx = np.where(labels == control_label)[0]
    pert_idx = np.where(labels != control_label)[0]
    if len(ctrl_idx) == 0 or len(pert_idx) == 0:
        raise ValueError("Need both control and perturbed cells for evaluation.")

    # control pseudobulk (global)
    ctrl_mean = X[ctrl_idx].mean(axis=0)

    # target mapping (label -> gene index), -1 if not a gene in panel
    t2gi = build_target_to_gene_index(adata, target_label)

    # adjacency
    A_base = make_base_adjacency(G, self_loops=True).to(device)

    # allocate
    Np = len(pert_idx)
    pred_mat = np.zeros((Np, G), dtype=np.float32)
    true_mat = X[pert_idx]  # (Np,G)
    pert_names = labels[pert_idx].tolist()

    # stable mapping label -> row index (should match training order if same labels set)
    pert_names_unique = sorted(set(labels.tolist()))
    pert2row = {p: i for i, p in enumerate(pert_names_unique)}

    # batched forward with random control matches
    model.eval()
    for start in range(0, Np, batch_size):
        end = min(start + batch_size, Np)
        b_idx = np.arange(start, end)
        # random controls (with replacement)
        rand_ctrl = rng.choice(ctrl_idx, size=len(b_idx), replace=True)

        bx_ctrl = torch.from_numpy(X[rand_ctrl]).float().to(device)
        tidx = torch.tensor([t2gi.get(p, -1) for p in pert_names[start:end]], dtype=torch.long, device=device)
        pert_rowidx = torch.tensor([pert2row[p] for p in pert_names[start:end]], dtype=torch.long, device=device)

        yhat, _ = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx)
        pred_mat[b_idx] = yhat.detach().cpu().numpy()

    return pred_mat, true_mat, pert_names, ctrl_mean, pert_idx


def evaluate_model(
    adata: ad.AnnData,
    model: nn.Module,
    target_label: str,
    control_label: str,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 0,
):
    """
    Computes:
      - per-perturbation MAE
      - knockdown efficiency (abs & %) for true vs predicted at the target gene
      - perturbation similarity: mean & min pairwise Pearson corr between predicted mean effect vectors
      - PDS (Perturbation Discrimination Score): mean over perturbations
    Prints a concise report and returns a dict with all metrics.
    """
    pred_mat, true_mat, pert_names, ctrl_mean, pert_idx = predict_all_perturbations(
        adata, model, target_label, control_label, device=device, batch_size=batch_size, seed=seed
    )
    G = adata.n_vars
    df_obs = adata.obs
    labels = df_obs[target_label].astype(str).values

    # group indices by perturbation (excluding control)
    perts = sorted(set(pert_names))
    # target mapping
    t2gi = build_target_to_gene_index(adata, target_label)

    # per-pert pseudobulks (pred & true) and MAE
    pred_bulk = {}
    true_bulk = {}
    mae_per_pert = {}

    # map pert_names (length Np) to row indices for quick grouping
    from collections import defaultdict
    rows_by_pert = defaultdict(list)
    for i, p in enumerate(pert_names):
        rows_by_pert[p].append(i)

    for p in perts:
        rows = rows_by_pert[p]
        yhat_p = pred_mat[rows]  # (n_p, G)
        ytrue_p = true_mat[rows] # (n_p, G)
        pred_bulk[p] = yhat_p.mean(axis=0)
        true_bulk[p] = ytrue_p.mean(axis=0)
        mae_per_pert[p] = np.mean(np.abs(yhat_p - ytrue_p))

    # knockdown efficiency at target gene (abs & %), true vs predicted
    # uses GLOBAL control pseudobulk as the "control" reference
    eps = 1e-8
    kd_eff = {}  # p -> dict
    for p in perts:
        t = t2gi.get(p, -1)
        if t < 0:
            kd_eff[p] = {"target_gene": None,
                         "true_abs": np.nan, "true_pct": np.nan,
                         "pred_abs": np.nan, "pred_pct": np.nan}
            continue
        ctrl_t = float(ctrl_mean[t])
        true_t = float(true_bulk[p][t])
        pred_t = float(pred_bulk[p][t])

        # absolute "knockdown" (positive if below control)
        true_abs = ctrl_t - true_t
        pred_abs = ctrl_t - pred_t
        # percentage relative to control level
        true_pct = true_abs / (ctrl_t + eps)
        pred_pct = pred_abs / (ctrl_t + eps)

        kd_eff[p] = {"target_gene": adata.var_names[t],
                     "true_abs": true_abs, "true_pct": true_pct,
                     "pred_abs": pred_abs, "pred_pct": pred_pct}

    # perturbation similarity (correlations between predicted mean effect vectors)
    # use predicted (pred_bulk[p] - ctrl_mean) as effect vector
    effect_vecs = []
    for p in perts:
        effect_vecs.append(pred_bulk[p] - ctrl_mean)
    effect_mat = np.stack(effect_vecs, axis=0)  # (K,G)
    # pairwise Pearson correlation matrix
    K = effect_mat.shape[0]
    # normalize
    em = effect_mat - effect_mat.mean(axis=1, keepdims=True)
    denom = np.sqrt((em ** 2).sum(axis=1, keepdims=True)) + 1e-8
    emn = em / denom
    corr_mat = emn @ emn.T  # (K,K)
    # take upper triangle excluding diagonal
    iu = np.triu_indices(K, k=1)
    mean_corr = float(corr_mat[iu].mean()) if iu[0].size > 0 else np.nan
    min_corr = float(corr_mat[iu].min()) if iu[0].size > 0 else np.nan

    # PDS (Perturbation Discrimination Score)
    # Distance between predicted pseudobulk for p and true pseudobulks for t
    # Exclude the target gene of *each* perturbation in the distance (both p's and t's, if present).
    # Rank of true t==p among all t (ascending distance). PDS_p = 1 - (rank-1)/(K-1).
    # Overall PDS = mean over p.
    # Build target indexes per perturbation (or -1 if N/A)
    t_idx_per_pert = {p: t2gi.get(p, -1) for p in perts}
    true_bulk_mat = np.stack([true_bulk[p] for p in perts], axis=0)  # (K,G)
    pred_bulk_mat = np.stack([pred_bulk[p] for p in perts], axis=0)  # (K,G)

    # precompute masks per pair to exclude targets
    PDS_scores = []
    for i, p in enumerate(perts):
        # distances to every t
        dists = []
        for j, tname in enumerate(perts):
            mask = np.ones(G, dtype=bool)
            # ti = t_idx_per_pert[p]
            tj = t_idx_per_pert[tname]
            # if ti >= 0: mask[ti] = False
            if tj >= 0: mask[tj] = False
            # L1 distance over masked genes
            d = np.abs(pred_bulk_mat[i, mask] - true_bulk_mat[j, mask]).sum()
            dists.append(d)
        dists = np.asarray(dists)
        # rank of the true target (j==i) in ascending distances
        order = np.argsort(dists)
        rank = int(np.where(order == i)[0][0]) + 1  # 1-based
        Kp = len(perts)
        PDS_p = 1.0 if Kp == 1 else (1.0 - (rank - 1) / (Kp - 1))
        PDS_scores.append(PDS_p)

    PDS_mean = float(np.mean(PDS_scores)) if len(PDS_scores) > 0 else np.nan

    # ---- Print concise report ----
    print("\n=== Evaluation ===")
    print(f"Per-perturbation MAE (mean ± sd): {np.mean(list(mae_per_pert.values())):.5f} ± {np.std(list(mae_per_pert.values())):.5f}")
    print(f"Perturbation similarity (pred mean effects): mean corr={mean_corr:.4f}, min corr={min_corr:.4f}")
    print(f"PDS (mean over perts): {PDS_mean:.4f}")
    print("\nKnockdown efficiency per perturbation (target gene, true_abs, true_pct, pred_abs, pred_pct):")
    # show a few lines sorted by true_abs descending
    preview = sorted(kd_eff.items(), key=lambda kv: (np.nan_to_num(kv[1]['true_abs'], nan=-1e9)), reverse=True)
    for p, d in preview[: min(10, len(preview))]:
        tg = d['target_gene'] or "N/A"
        print(f"  {p:20s}  tg={tg:12s}  true_abs={d['true_abs']:.4f}  true_pct={d['true_pct']:.2%}  "
              f"pred_abs={d['pred_abs']:.4f}  pred_pct={d['pred_pct']:.2%}")

    return {
        "mae_per_pert": mae_per_pert,
        "kd_eff": kd_eff,
        "mean_corr_pred_effects": mean_corr,
        "min_corr_pred_effects": min_corr,
        "PDS_mean": PDS_mean,
        "PDS_scores": dict(zip(perts, PDS_scores)),
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--out_pred_h5ad", type=str, default="",
                    help="If set, write an AnnData with predictions for the evaluation split.")
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
    ap.add_argument("--use_pseudobulk", action="store_true",
                    help="Collapse to one mean row per perturbation (incl. control).")
    ap.add_argument("--match_controls", choices=["random", "knn"], default="knn",
                    help="How to choose a control for each perturbed cell.")
    ap.add_argument("--knn_k", type=int, default=32, help="Top-k controls to sample from.")
    ap.add_argument("--knn_temp", type=float, default=0.1, help="Softmax temperature over distances.")
    ap.add_argument("--knn_metric", choices=["l2", "cosine"], default="l2",
                    help="Distance metric for kNN control matching.")
    ap.add_argument("--test_pct_perts", type=float, default=0.0,
                    help="Fraction of perturbation labels (excluding control) to hold out for testing. 0.0 = no holdout.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.in_h5ad)
    if args.use_pseudobulk:
        adata = collapse_to_pseudobulk(adata, args.target_label)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway

    # ---------------------------
    # Split perts into train/test (leave-perturbations-out)
    # ---------------------------
    labels_all = adata.obs[args.target_label].astype(str).values
    rng = np.random.default_rng(args.seed)
    # all perturbations excluding the control label
    all_perts = sorted({lbl for lbl in labels_all if lbl != args.control_label})
    n_test = int(round(args.test_pct_perts * len(all_perts)))
    if n_test > 0:
        test_perts = set(rng.choice(np.array(all_perts), size=n_test, replace=False).tolist())
    else:
        test_perts = set()
    train_perts = [p for p in all_perts if p not in test_perts]

    # build train and (optional) test AnnData views
    mask_train = adata.obs[args.target_label].isin([args.control_label] + train_perts)
    adata_train = adata[mask_train].copy()
    if n_test > 0:
        mask_test = adata.obs[args.target_label].isin([args.control_label] + list(test_perts))
        adata_test = adata[mask_test].copy()
    else:
        adata_test = None

    print("=== Split summary ===")
    print(f"Total perts (excl. control): {len(all_perts)}  |  Held-out test perts: {len(test_perts)}")
    if n_test > 0:
        print(f"Test perts: {sorted(test_perts)[:10]}{' ...' if len(test_perts) > 10 else ''}")
    print(f"Train cells: {adata_train.n_obs}, Test cells: {adata_test.n_obs if adata_test is not None else 0}")

    model = train(
        adata=adata_train,
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
        match_controls=args.match_controls,
        knn_k=args.knn_k,
        knn_temp=args.knn_temp,
        knn_metric=args.knn_metric,
    )

    # Evaluate: if holding out perts, evaluate on test split; else on training split
    eval_adata = adata_test if adata_test is not None else adata_train
    print("\n=== Evaluation on {} set ===".format("TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)"))

    # Evaluate on the same panel (training fit quality)
    eval_metrics = evaluate_model(
        adata=eval_adata,
        model=model,
        target_label=args.target_label,
        control_label=args.control_label,
        device=args.device,
        batch_size=512,
        seed=args.seed,
    )

    # Optional: save weights
    out_path = os.path.splitext(args.in_h5ad)[0] + f".mpnn_hidden{args.hidden}_T{args.T}.pt"
    torch.save({"state_dict": model.state_dict(),
                "G": model.G,
                "hidden": model.hidden,
                "T": model.T}, out_path)

    # ---------------------------
    # Optional: write predictions AnnData for the evaluation split
    # ---------------------------
    if args.out_pred_h5ad:
        print(f"\n[write] Generating predictions AnnData → {args.out_pred_h5ad}")
        # run the same batched predictor to get per-pert predictions + their row indices
        pred_mat, _, pert_names_eval, _, pert_idx = predict_all_perturbations(
            eval_adata, model, args.target_label, args.control_label,
            device=args.device, batch_size=512, seed=args.seed
        )
        # start from a copy of eval_adata.X and replace perturbed rows with predictions
        from anndata import AnnData
        X_eval = to_numpy(eval_adata.X).astype(np.float32, copy=True)
        X_eval[pert_idx, :] = pred_mat  # controls remain unchanged
        ad_pred = AnnData(X_eval, obs=eval_adata.obs.copy(), var=eval_adata.var.copy())
        ad_pred.write_h5ad(args.out_pred_h5ad, compression="lzf")
        eval_adata.write_h5ad(os.path.splitext(args.out_pred_h5ad)[0] + ".true.h5ad", compression="lzf")
        print(f"[done] Wrote {args.out_pred_h5ad} (cells={ad_pred.n_obs}, genes={ad_pred.n_vars})")

    print(f"[done] saved model to {out_path}")

if __name__ == "__main__":
    main()
