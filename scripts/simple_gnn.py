#!/usr/bin/env python3
# gnn_fit_panel.py
from email import parser
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

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graft.losses.distribution import sliced_wasserstein, mmd_rbf, energy_distance

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
    fixed_label: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Returns batch_x_ctrl (B,G), batch_x_pert (B,G), batch_targets (list of labels).
    Matches each perturbed cell to a random control cell.
    """
    B = batch_size
    # indices for perturbed cells (exclude controls)
    if fixed_label is None:
        pert_idx = np.where(pert_labels != control_label)[0]
    else:
        pert_idx = np.where(pert_labels == fixed_label)[0]
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
    fixed_label: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Choose B perturbed cells; for each, sample one control from its top-k nearest controls
    with softmax(-dist / temp). Supports L2 or cosine. Small-panel friendly (no index lib).
    """
    # index pools
    ctrl_idx = np.where(labels == control_label)[0]
    if fixed_label is None:
        pert_idx = np.where(labels != control_label)[0]
    else:
        pert_idx = np.where(labels == fixed_label)[0]
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

def make_pretrain_pseudobulk_from_adata(adata: ad.AnnData, target_label: str, control_label: str, dataset_id: str) -> ad.AnnData:
    """One-row-per-pert pseudobulk of the provided AnnData (already normalized/log1p).
    Keeps columns: dataset_id, is_control, target_idx, target_present, cell_type (UNK), tech_batch_id (UNK), lab_id (UNK)."""
    groups = adata.obs[target_label].astype(str).values
    uniq = np.unique(groups)
    X_rows = []
    obs_rows = []
    # map gene -> index once
    gene_to_pos = {g: i for i, g in enumerate(adata.var_names)}
    for p in uniq:
        m = (groups == p)
        X_rows.append(np.asarray(adata[m].X.mean(axis=0)).squeeze())
        if p == control_label:
            pert_type = "control"; t_idx = -1; t_present = False
        else:
            pos = gene_to_pos.get(p, None)
            if pos is None:
                pert_type = "non_gene"; t_idx = -1; t_present = False
            else:
                pert_type = "gene"; t_idx = int(pos); t_present = True
        obs_rows.append({
            "dataset_id": dataset_id,
            "is_control": (p == control_label),
            target_label: p,
            "pert_type": pert_type,
            "target_idx": t_idx,
            "target_present": t_present,
            "cell_type": "UNK",
            "tech_batch_id": "UNK",
            "lab_id": "UNK",
        })
    X_bulk = np.stack(X_rows, axis=0)
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"{dataset_id}::{r[target_label]}" for _, r in obs.iterrows()]
    return ad.AnnData(X=X_bulk, obs=obs, var=adata.var.copy())

def prep_external_data(pb: ad.AnnData, target_label: str, control_label: str, adata_train: ad.AnnData) -> ad.AnnData:
    if 'target_present' in pb.obs.columns:
        pb = pb[pb.obs["target_present"] | (pb.obs[target_label] == control_label), :].copy()
    # Clean placeholders before normalization
    X = pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # Replace NaN / -1 placeholders with 0 counts
    if np.isnan(X).any():
        np.nan_to_num(X, copy=False, nan=0.0)
    if (X == -1).any():
        X[X == -1] = 0.0
    pb.X = X
    # Normalize only rows with positive sums; leave zero-sum rows as zeros
    row_sums = X.sum(axis=1)
    if (row_sums > 0).any():
        mask = row_sums > 0
        # normalize a temporary AnnData view to avoid Scanpy warnings on zero-sum rows
        tmp = pb[mask].copy()
        sc.pp.normalize_total(tmp, target_sum=None, inplace=True)
        sc.pp.log1p(tmp)
        pb.X[mask] = tmp.X
    # (rows with sum==0 remain zero; perfectly fine for Stage-1 since we mask missing genes)
    assert list(pb.var_names) == list(adata_train.var_names), "Pseudobulk genes/order must match target dataset."
    return pb


# ----------------------------
# Model: Step0 + MPNN + Readout
# ----------------------------
class Step0Head(nn.Module):
    """
    Predict efficacy alpha_t from the target embedding and APPLY the counts-space clamp.
    Returns the clamped input x0 and the efficacy vector alpha_t.
    """
    def __init__(self, tau: float = 0.0, d: int = 64):
        super().__init__()
        self.alpha_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.alpha_mean_train: torch.Tensor | None = None  # set externally if available
        self.tau = tau

    def forward(
        self,
        x_ctrl: torch.Tensor,             # (B,G), log1p
        target_idx: torch.Tensor,         # (B,), -1 for control
        e_t: torch.Tensor,                # (B,d) target embedding
        alpha_cap: float = 1.0,
        mean_shrink: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Learn alpha_t = sigmoid(MLP(e_t)), optionally shrink to train mean,
        then clamp the target coordinate in counts space and map back to log1p.
        """
        # 1) predict efficacy
        alpha_t = torch.sigmoid(self.alpha_head(e_t)).squeeze(-1)     # (B,)
        if self.alpha_mean_train is not None and mean_shrink > 0.0:
            alpha_t = (1 - mean_shrink) * alpha_t + mean_shrink * self.alpha_mean_train
        alpha_t = alpha_t.clamp(0.0, float(alpha_cap))

        # 2) counts-space clamp at the target
        x0 = x_ctrl.clone()
        mask = (target_idx >= 0)
        if mask.any():
            rows = torch.arange(x_ctrl.shape[0], device=x_ctrl.device)[mask]
            cols = target_idx[mask]
            counts = torch.expm1(x0[rows, cols].clamp_min(0.0))
            m = (1.0 - alpha_t[mask]).clamp(0.0, 1.0)
            new_counts = counts * m
            x0[rows, cols] = torch.log1p(new_counts.clamp_min(0.0))
        return x0, alpha_t

class PrototypeGenerator(nn.Module):
    """
    Gene-conditioned generator:
      - learns a per-gene embedding e_t in R^d
      - maps e_t to a prototype mean-effect vector b_t in R^G
      - predicts a Step-0 efficacy alpha_t in (0,1) from e_t (and optional meta covariates)
    """
    def __init__(self, G: int, d: int = 64, extra_cond_dim: int = 0):
        super().__init__()
        self.E = nn.Embedding(G, d)
        nn.init.normal_(self.E.weight, std=0.02)
        # MLP to produce a hidden code from [e_t, alpha, (optional z_d, z_ct)]
        self.phi = nn.Sequential(
            nn.Linear(d + 1 + extra_cond_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.W_out = nn.Linear(d, G, bias=False)
        # kept for backward-compat; may not be used when external alpha is provided
        self.alpha_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

    def forward(self, target_idx: torch.LongTensor, meta: torch.Tensor | None = None, 
                external_alpha: torch.Tensor | None = None, z_extra: torch.Tensor | None = None):
        """
        target_idx: (B,) gene indices (>=0); if -1, we still produce something but it won't be used.
        meta: optional (B, M) covariates; ignored in this minimal version.
        Returns:
          b: (B,G) prototype vector
          alpha: (B,) efficacy in (0,1)
          e_t: (B,d) gene embeddings
        """
        e_t = self.E(torch.clamp(target_idx, min=0))    # (B,d)
        alpha = (torch.sigmoid(self.alpha_head(e_t)).squeeze(-1)
                 if external_alpha is None else external_alpha)       # (B,)
        # Nonlinear conditioning on efficacy (+ optional dataset/celltype embedding)
        if z_extra is None:
            z = torch.cat([e_t, alpha.unsqueeze(-1)], dim=-1)         # (B,d+1)
        else:
            z = torch.cat([e_t, alpha.unsqueeze(-1), z_extra], dim=-1) # (B,d+1+extra)
        h = self.phi(z)                                               # (B,d)
        b = self.W_out(h)                                             # (B,G)
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
    def __init__(self, G: int, hidden: int = 128, T: int = 2, tau: float = 0.0, alpha_cap: float = 1.0, prior_dim: int | None = None,
                 dset_vocab: int = 0, dset_dim: int = 0, ct_vocab: int = 0, ct_dim: int = 0):
        super().__init__()
        self.G = G
        self.hidden = hidden
        self.T = T
        self.alpha_cap = alpha_cap
        # per-gene node embeddings (used by all nodes, every batch)
        self.node_dim = 64
        self.node_E = nn.Embedding(G, self.node_dim)
        nn.init.normal_(self.node_E.weight, std=0.02)
        # Optional small projector: prior (R) -> node_dim
        self.prior_proj = nn.Linear(prior_dim, self.node_dim, bias=False) if prior_dim is not None else None
        # input becomes [log1p expression, node embedding] per gene
        self.embed = nn.Linear(1 + self.node_dim, hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden) for _ in range(T)])
        self.readout = nn.Linear(hidden, 1)
        # Optional dataset/cell-type embeddings
        self.dset_E = nn.Embedding(dset_vocab, dset_dim) if dset_vocab > 0 and dset_dim > 0 else None
        self.ct_E   = nn.Embedding(ct_vocab,   ct_dim)   if ct_vocab   > 0 and ct_dim   > 0 else None
        extra_cond_dim = (dset_dim if self.dset_E is not None else 0) + (ct_dim if self.ct_E is not None else 0)
        # FiLM will see [e_t, alpha, (z_d), (z_ct)]
        self.film_gamma = nn.Linear(self.node_dim + 1 + extra_cond_dim, hidden)
        self.film_beta  = nn.Linear(self.node_dim + 1 + extra_cond_dim, hidden)
        # Step-0 head: predict efficiency and apply the clamp
        self.step0_head = Step0Head(tau=tau, d=self.node_dim)
        # gene-conditioned prototype & alpha; FiLM from gene embedding
        self.proto = PrototypeGenerator(G=G, d=self.node_dim, extra_cond_dim=extra_cond_dim)
        # Tie prototype embedding to the node embedding so both share the same e_t
        self.proto.E = self.node_E

    @torch.no_grad()
    def init_from_prior(self, W_meta: torch.Tensor):
        """
        W_meta: (R, G) tensor matching var_names order. Projects columns to node_dim and uses
        as an initialization for node_E. Requires prior_proj to be present.
        """
        assert self.prior_proj is not None, "prior_proj not initialized (set prior_dim when constructing GeneMPNN)."
        # Project each gene's R-dim vector to node_dim
        # W_meta.T: (G, R) -> (G, node_dim)
        E0 = self.prior_proj(W_meta.T)  # (G, node_dim)
        self.node_E.weight.copy_(E0)

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor, A_base: torch.Tensor, pert_rowidx: torch.Tensor = None,
                dset_idx: torch.Tensor | None = None, ct_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 where unknown
        A_base: (G,G) dense row-normalized base adjacency
        """
        device = x_ctrl.device
        B, G = x_ctrl.shape
        assert G == self.G

        # --- Step-0: learn efficacy and apply counts-space clamp in one place ---
        e_t = self.node_E(torch.clamp(target_idx, min=0))                       # (B,64)
        # hand mean alpha to the head if we have it
        if hasattr(self, "alpha_mean_train"):
            self.step0_head.alpha_mean_train = self.alpha_mean_train
        # For non-gene perts (target_idx == -1), skip clamp and set alpha=0
        if (target_idx >= 0).any():
            x0, alpha_t = self.step0_head(x_ctrl, target_idx, e_t, alpha_cap=self.alpha_cap)
        else:
            x0 = x_ctrl
            alpha_t = torch.zeros(x_ctrl.size(0), device=x_ctrl.device)

        # Initial hidden state (shared 1->hidden linear applied per gene)
        # assemble per-node embeddings for all genes
        idx_all = torch.arange(G, device=device)
        node_feats = self.node_E(idx_all).unsqueeze(0).expand(B, G, -1)  # (B,G,node_dim)
        x_in = torch.cat([x0.unsqueeze(-1), node_feats], dim=-1)         # (B,G,1+node_dim)
        h = self.embed(x_in)
        # FiLM conditions on both the identity (e_t) and strength (alpha_t) of the hit
        # Gather optional conditioning from dataset / cell-type
        z_list = []
        if self.dset_E is not None and dset_idx is not None:
            z_list.append(self.dset_E(dset_idx))                      # (B, dset_dim)
        if self.ct_E is not None and ct_idx is not None:
            z_list.append(self.ct_E(ct_idx))                          # (B, ct_dim)
        z_extra = torch.cat(z_list, dim=-1) if len(z_list) > 0 else None
        # FiLM conditions on [e_t, alpha_t, (z_d), (z_ct)]
        e_aug = torch.cat([e_t, alpha_t.unsqueeze(-1)], dim=-1) if z_extra is None else torch.cat([e_t, alpha_t.unsqueeze(-1), z_extra], dim=-1)
        gamma = torch.tanh(self.film_gamma(e_aug)).unsqueeze(1)       # (B,1,H)
        beta  = self.film_beta(e_aug).unsqueeze(1)                    # (B,1,H)
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
        # add gene-conditioned prototype mean-effect (now nonlinear in alpha_t and optional z_extra)
        b_proto, _, _ = self.proto(target_idx, meta=None, external_alpha=alpha_t, z_extra=z_extra)
        y = y + b_proto
        # Preserve Step-0 at the target: y_t := x0_t
        freeze_mask = (target_idx >= 0)
        if freeze_mask.any():
            rows = torch.arange(B, device=device)[freeze_mask]
            cols = target_idx[freeze_mask]
            y[rows, cols] = x0[rows, cols]
        return y, x0, alpha_t

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
    weight_target: float = 0.2,
    weight_local: float = 0.0,
    weight_mse: float = 0.0,
    weight_proto: float = 0.2,
    seed: int = 0,
    tau: float = 0.0,
    device: str = "cuda",
    match_controls: str = "knn",  # or "random"
    knn_k: int = 32,
    knn_temp: float = 0.1,
    knn_metric: str = "l2",  # or "cosine"
    dist_loss: str = "mmd",  # or "swd" or "energy"
    swd_projections: int = 128,
    weight_dist: float = 1.0,
    single_pert_batches: bool = False,
    W_meta: np.ndarray | None = None,
    init_from_meta: bool = False,
    weight_prior: float = 0.0,
    meta_topk: int = 0,
    model: nn.Module | None = None,
    pretrain_mode: bool = False,
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
    # unique perturbation labels (exclude control)
    pert_unique = sorted({l for l in labels if l != control_label})

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

    ctrl_mean_np = X[ctrl_mask].mean(axis=0).astype(np.float32)
    ctrl_mean = torch.from_numpy(ctrl_mean_np).to(device)

    # Dataset / cell-type categorical codes (only if present; safe to ignore otherwise)
    dset_codes = None
    ct_codes = None
    if "dataset_id" in adata.obs.columns:
        dset_codes = pd.Categorical(adata.obs["dataset_id"]).codes.astype(np.int64)
    if "cell_type" in adata.obs.columns:
        ct_col = adata.obs["cell_type"]
        # Casting to str avoids Categorical fillna errors for unseen categories
        if isinstance(ct_col.dtype, pd.CategoricalDtype):
            ct_col = ct_col.astype(str)
        else:
            ct_col = ct_col.astype(str)
        # Normalize missing/mixed to UNK
        ct_vals = ct_col.replace({"<NA>": "UNK", "mixed": "UNK"}).fillna("UNK")
        ct_codes = pd.Categorical(ct_vals).codes.astype(np.int64)

    # Model (reuse if provided)
    if model is None:
        prior_dim = W_meta.shape[0] if W_meta is not None else None
        model = GeneMPNN(G=G, hidden=hidden, T=T, tau=tau, prior_dim=prior_dim).to(device)
        # Optionally initialize node embeddings from prior
        if (W_meta is not None) and init_from_meta:
            Wm_torch = torch.from_numpy(W_meta.astype(np.float32)).to(device)  # (R,G)
            model.init_from_prior(Wm_torch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Base adjacency: either dense (as before) or prior-based top-k cosine graph
    if (W_meta is not None) and (meta_topk > 0):
        # build kNN in prior space (cosine), symmetric, row-normalized
        Wm = W_meta.astype(np.float32)  # (R,G)
        # cosine over columns
        V = Wm.T  # (G,R)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        S = Vn @ Vn.T  # (G,G) cosine similarity
        # for each row, keep top-k (including self), set others to 0
        k = min(meta_topk, G)
        A = np.zeros_like(S, dtype=np.float32)
        idx = np.argpartition(-S, kth=k-1, axis=1)[:, :k]
        rows = np.repeat(np.arange(G)[:, None], k, axis=1)
        A[rows, idx] = S[rows, idx]
        # symmetrize by max
        A = np.maximum(A, A.T)
        # row-normalize
        A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
        A_base = torch.from_numpy(A).to(device)
        print(f"[graph] Using prior kNN graph (top-k={k}) from M_meta.")
    else:
        # dense fully-connected adjacency (previous behavior)
        A_base = make_base_adjacency(G, self_loops=True).to(device)

    # Simple schedule
    steps_per_epoch = math.ceil(pert_mask.sum() / batch_size)

    # Stage-1: build per-dataset control pseudobulks (if available)
    ctrl_by_dset = None
    if pretrain_mode and ("dataset_id" in adata.obs.columns):
        ctrl_by_dset = {}
        dsets = adata.obs["dataset_id"].astype(str).values
        for d in sorted(set(dsets)):
            m = (labels == control_label) & (dsets == d)
            if m.any():
                ctrl_by_dset[d] = X[m].mean(axis=0).astype(np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"mse": 0.0, "targ": 0.0, "loc": 0.0, "proto": 0.0, "dist": 0.0, "prior": 0.0, "tot": 0.0}
        for step in range(steps_per_epoch):
            # if requested, choose a single perturbation label for this batch
            fixed_label = None
            if single_pert_batches:
                fixed_label = rng.choice(np.array(pert_unique)) if len(pert_unique) > 0 else None

            if pretrain_mode and (ctrl_by_dset is not None):
                # --- Stage-1 pseudobulk controls: match control by dataset_id, not by sampling ---
                # choose perturbed rows for this batch
                if fixed_label is None:
                    pert_idx = np.where(labels != control_label)[0]
                else:
                    pert_idx = np.where(labels == fixed_label)[0]
                if len(pert_idx) < batch_size:
                    sel_pert = rng.choice(pert_idx, size=batch_size, replace=True)
                else:
                    sel_pert = rng.choice(pert_idx, size=batch_size, replace=False)
                bx_pert = torch.from_numpy(X[sel_pert]).float()
                # build per-row control vector from the SAME dataset; fallback to global control mean
                dsets = adata.obs["dataset_id"].astype(str).values
                bx_ctrl_rows = []
                for j in sel_pert:
                    dj = dsets[j]
                    if (dj in ctrl_by_dset):
                        bx_ctrl_rows.append(ctrl_by_dset[dj])
                    else:
                        bx_ctrl_rows.append(ctrl_mean_np)
                bx_ctrl = torch.from_numpy(np.stack(bx_ctrl_rows, axis=0)).float()
                btargets = labels[sel_pert].tolist()
            elif match_controls == "knn":
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
                    pre_norm_ctrl=pre_norm_ctrl, pre_norm_pert=pre_norm_pert,
                    fixed_label=fixed_label
                )
            else:
                bx_ctrl, bx_pert, btargets = sample_minibatch(
                    X_ctrl=X_ctrl, X_pert=X_pert, pert_labels=pert_labels,
                    control_label=control_label, batch_size=batch_size, rng=rng,
                    fixed_label=fixed_label
                )
            # per-sample target index tensor
            tidx = torch.tensor([t2gi.get(t, -1) for t in btargets], dtype=torch.long)

            bx_ctrl = bx_ctrl.to(device)
            bx_pert = bx_pert.to(device)
            tidx = tidx.to(device)

            # dataset / cell-type indices pulled from the SAME rows as bx_pert
            z_d, z_ct = None, None
            # identify the row indices that produced bx_pert
            if 'sel_pert' in locals():           # pretrain_mode branch (or random sampler)
                idx_rows = np.asarray(sel_pert, dtype=int)
            elif 'pert_rowidx' in locals() and pert_rowidx is not None:
                # KNN sampler should provide these indices (may be a CUDA tensor)
                if torch.is_tensor(pert_rowidx):
                    idx_rows = pert_rowidx.detach().cpu().numpy().astype(int)
                else:
                    idx_rows = np.asarray(pert_rowidx, dtype=int)
            else:
                idx_rows = None
            if idx_rows is not None:
                if "dataset_id" in adata.obs.columns:
                    z_d = torch.tensor(pd.Categorical(adata.obs.iloc[idx_rows]["dataset_id"]).codes,
                                        device=device)
                if "cell_type" in adata.obs.columns:
                    ct_slice = adata.obs.iloc[idx_rows]["cell_type"]
                    # Cast to str before fill/replace to avoid Categorical category errors
                    if isinstance(ct_slice.dtype, pd.CategoricalDtype):
                        ct_slice = ct_slice.astype(str)
                    else:
                        ct_slice = ct_slice.astype(str)
                    ct_slice = ct_slice.replace({"<NA>": "UNK", "mixed": "UNK"}).fillna("UNK")
                    z_ct = torch.tensor(pd.Categorical(ct_slice).codes, device=device)

            # per-sample perturbation row indices for embedding / FiLM
            pert_rowidx = torch.tensor([pert2row[t] for t in btargets], dtype=torch.long, device=device)
            yhat, x0, alpha_vec = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx, dset_idx=z_d, ct_idx=z_ct)

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
                # Use the group's own control mean for this batch (crucial in Stage-1)
                ctrl_mean_grp = bx_ctrl[idx].mean(dim=0)        # (G,)
                d_pred = (yhat[idx].mean(dim=0) - ctrl_mean_grp)
                d_true = (bx_pert[idx].mean(dim=0) - ctrl_mean_grp)
                if pretrain_mode:
                    # mask missing genes from external pseudobulk: NaN or -1 placeholders
                    mnan = torch.isnan(d_true)
                    mneg1 = (d_true == -1)
                    mask_cols = ~(mnan | mneg1)
                    if mask_cols.any():
                        loss_proto = loss_proto + torch.mean(torch.abs(d_pred[mask_cols] - d_true[mask_cols]))
                else:
                    loss_proto = loss_proto + torch.mean(torch.abs(d_pred - d_true))
                groups += 1
            if groups > 0:
                loss_proto = loss_proto / groups

            # In Stage-1 pretraining, pseudobulk may contain NaNs/-1 for missing genes; skip MSE entirely.
            loss_mse = (yhat - x0).new_tensor(0.0) if pretrain_mode else mse_loss(yhat - x0, bx_pert - x0)
            loss_loc = locality_damping(yhat, x0, tidx, weight=1.0) if weight_local > 0 else yhat.new_tensor(0.0)

            # --- Efficacy supervision (repurpose "target" loss): per-cell estimate in counts space ---
            loss_t = yhat.new_tensor(0.0)
            if (tidx >= 0).any():
                mask = (tidx >= 0)
                rows = torch.arange(tidx.numel(), device=yhat.device)[mask]
                cols = tidx[mask]
                # counts for target gene in matched control vs perturbed cell
                ctrl_cnt = torch.expm1(bx_ctrl[rows, cols].clamp_min(0.0))
                pert_cnt = torch.expm1(bx_pert[rows, cols].clamp_min(0.0))
                true_alpha = (1.0 - pert_cnt / (ctrl_cnt + 1e-8)).clamp(0.0, 1.0)
                loss_t = F.mse_loss(alpha_vec[mask], true_alpha)

            # --- Distribution loss on per-perturbation deltas (optional) ---
            loss_dist = yhat.new_tensor(0.0)
            if (not pretrain_mode) and (dist_loss != "none"):
                from collections import defaultdict
                by_lbl = defaultdict(list)
                for i, lbl in enumerate(btargets):
                    by_lbl[lbl].append(i)
                for lbl, idxs in by_lbl.items():
                    if len(idxs) < 2:
                        continue
                    idx = torch.tensor(idxs, device=yhat.device, dtype=torch.long)
                    # deltas vs per-sample Step-0 baseline (robust to control matching)
                    d_pred = (yhat[idx] - x0[idx])         # (n_p, G)
                    d_true = (bx_pert[idx] - x0[idx])      # (n_p, G)
                    # mask target gene column
                    t = t2gi.get(lbl, -1)
                    if t >= 0:
                        d_pred = torch.cat([d_pred[:, :t], d_pred[:, t+1:]], dim=1)
                        d_true = torch.cat([d_true[:, :t], d_true[:, t+1:]], dim=1)
                    # choose loss
                    if dist_loss == "mmd":
                        loss_dist = loss_dist + mmd_rbf(d_pred, d_true)
                    elif dist_loss == "swd":
                        loss_dist = loss_dist + sliced_wasserstein(d_pred, d_true, num_proj=swd_projections)
                    elif dist_loss == "energy":
                        loss_dist = loss_dist + energy_distance(d_pred, d_true)
                # average across present perts
                n_groups = sum(1 for v in by_lbl.values() if len(v) >= 2)
                if n_groups > 0:
                    loss_dist = loss_dist / n_groups

            # Optional proximity loss: keep node_E near projected prior
            loss_prior = yhat.new_tensor(0.0)
            if (weight_prior > 0.0) and (W_meta is not None) and (model.prior_proj is not None):
                # compute current projected prior: (G,node_dim)
                with torch.no_grad():
                    Wm_torch = torch.from_numpy(W_meta.astype(np.float32)).to(yhat.device)
                E_prior = model.prior_proj(Wm_torch.T)  # (G,node_dim)
                loss_prior = F.mse_loss(model.node_E.weight, E_prior)

            if weight_mse == 0.0:
                loss_mse = loss_mse * 0.0
            if weight_target == 0.0:
                loss_t = loss_t * 0.0
            if weight_local == 0.0:
                loss_loc = loss_loc * 0.0
            if weight_proto == 0.0:
                loss_proto = loss_proto * 0.0
            if weight_dist == 0.0:
                loss_dist = loss_dist * 0.0
            if weight_prior == 0.0:
                loss_prior = loss_prior * 0.0

            loss = weight_mse * loss_mse \
                 + weight_target * loss_t \
                 + weight_local * loss_loc \
                 + weight_proto * loss_proto \
                 + weight_dist * loss_dist \
                 + weight_prior * loss_prior

            # loss = loss_mse + weight_target * loss_t + weight_local * loss_loc + w_proto * loss_proto

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["mse"]  += float(loss_mse.item())
            running["targ"] += float(loss_t.item())
            running["loc"]  += float(loss_loc.item())
            running["proto"] += float(loss_proto.item())
            running["dist"] += float(loss_dist.item())
            running["prior"] += float(loss_prior.item())
            running["tot"]  += float(loss.item())

        denom = max(steps_per_epoch, 1)
        do_print = True
        if pretrain_mode and epoch != 1 and epoch % 20 != 0 and epoch != epochs:
            do_print = False
        if do_print:
            print(f"[epoch {epoch:03d}] "
                f"mse={running['mse']/denom:.5f}  "
                f"targ={running['targ']/denom:.5f}  "
                f"loc={running['loc']/denom:.5f}  "
                f"proto={running['proto']/denom:.5f}  "
                f"dist={running['dist']/denom:.5f}  "
                f"prior={running['prior']/denom:.5f}  "
                f"total={running['tot']/denom:.5f}")

    # estimate mean alpha on training targets (linear KD from pseudobulk)
    with torch.no_grad():
        # quick calc using control pseudobulk + training perts
        X = to_numpy(adata.X).astype(np.float32)
        labels = adata.obs[target_label].astype(str).values
        ctrl_mean = X[labels == control_label].mean(axis=0)
        t2gi = build_target_to_gene_index(adata, target_label)
        train_perts = sorted({l for l in labels if l != control_label})
        alphas = []
        for p in train_perts:
            idx = np.where(labels == p)[0]
            if len(idx)==0: continue
            t = t2gi.get(p, -1)
            if t < 0: continue
            true_t = np.expm1(X[idx, t]).mean()
            ctrl_t = np.expm1(ctrl_mean[t])
            a = np.clip(1.0 - true_t/(ctrl_t + 1e-8), 0.0, 1.0)
            alphas.append(a)
        model.register_buffer("alpha_mean_train", torch.tensor(float(np.mean(alphas) if alphas else 0.8)))

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

        yhat, _, _ = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx)
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
    bulk_mae_per_pert = {}

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
        # per-cell MAE (cells+genes)
        mae_per_pert[p] = np.mean(np.abs(yhat_p - ytrue_p))
        # pseudobulk MAE (genes only)
        bulk_mae_per_pert[p] = float(np.mean(np.abs(pred_bulk[p] - true_bulk[p])))

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
    # - use absolute deltas vs control
    # - exclude only the TRUE target gene for expression data, by name
    # - zero-based rank normalized by N (not N-1): PDS_p = 1 - rank/N
    # absolute deltas vs global control mean
    true_bulk_mat = np.stack([np.abs(true_bulk[p] - ctrl_mean) for p in perts], axis=0)  # (K,G)
    pred_bulk_mat = np.stack([np.abs(pred_bulk[p] - ctrl_mean) for p in perts], axis=0)  # (K,G)
    t_idx_per_pert = {p: t2gi.get(p, -1) for p in perts}

    # precompute masks per pair to exclude targets
    from sklearn.metrics import pairwise_distances
    Kp = len(perts)
    PDS_scores = []
    for i, p in enumerate(perts):
        # build include mask: exclude target gene IF its name equals the perturbation label
        mask = np.ones(G, dtype=bool)
        tj = t_idx_per_pert[p]
        if tj >= 0:
            mask[tj] = False
        # distances from ALL real effects to this predicted effect
        dists = pairwise_distances(
            true_bulk_mat[:, mask],    # (K, G')
            pred_bulk_mat[i, mask][None, :],  # (1, G')
            metric="manhattan",
        ).ravel()
        order = np.argsort(dists)          # ascending
        # rank of the correct perturbation (zero-based)
        p_index = i  # same ordering
        rank0 = int(np.flatnonzero(order == p_index)[0])
        # normalize by K (not K-1), then invert
        PDS_scores.append(1.0 - rank0 / Kp)

    PDS_mean = float(np.mean(PDS_scores)) if len(PDS_scores) > 0 else np.nan

    # ---- Print concise report ----
    print("\n=== Evaluation ===")
    # print(f"Per-cell MAE (mean ± sd over perts): {np.mean(list(mae_per_pert.values())):.5f} ± {np.std(list(mae_per_pert.values())):.5f}")
    print(f"Pseudobulk MAE (mean over perts):   {np.mean(list(bulk_mae_per_pert.values())):.5f}")
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
        "bulk_mae_per_pert": bulk_mae_per_pert,
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
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional path to a separate test AnnData. If set, overrides --test_pct_perts.")
    ap.add_argument("--dist_loss", choices=["none","mmd","swd","energy"], default="mmd",
                    help="Distribution loss between predicted and true deltas per perturbation.")
    ap.add_argument("--weight_dist", type=float, default=1.0, help="Weight for distribution loss.")
    ap.add_argument("--swd_projections", type=int, default=128, help="Num random projections for SWD.")
    ap.add_argument("--single_pert_batches", action="store_true",
                    help="If set, each batch contains cells from a single perturbation label.")
    ap.add_argument("--meta_path", type=str, default="",
                    help="Path to M_meta.npy produced by embed_pathways.py (shape R x G; columns aligned to var_names).")
    ap.add_argument("--init_from_meta", action="store_true",
                    help="If set, initialize node embeddings from the projected pathway prior.")
    ap.add_argument("--weight_prior", type=float, default=0.0,
                    help="L2 proximity loss weight to keep node embeddings near the projected pathway prior (Phase 1).")
    ap.add_argument("--meta_topk", type=int, default=0,
                    help="If >0, build a top-k cosine kNN adjacency from the pathway prior instead of dense A.")
    ap.add_argument("--pretrain_pseudobulk", type=str, default="",
                    help="Path to a pseudobulk .h5ad for Stage-1 pretraining; empty = skip Stage-1")
    ap.add_argument("--pretrain_pseudobulk_list", type=str, default="",
                        help="Text file with one pseudobulk .h5ad path per line; blank/comment lines ignored")
    ap.add_argument("--include_target_pseudobulk", action="store_true",
                        help="Also pseudobulk the target dataset and include it in Stage-1 pretraining")
    ap.add_argument("--pretrain_epochs", type=int, default=10,
                    help="Epochs to run Stage-1 pseudobulk pretraining")
    ap.add_argument("--use_dset_embed", action="store_true",
                    help="Enable dataset_id embeddings (used in FiLM/proto conditioning)")
    ap.add_argument("--use_celltype_embed", action="store_true",
                    help="Enable cell_type embeddings (used in FiLM/proto conditioning)")
    ap.add_argument("--dset_embed_dim", type=int, default=16,
                    help="Dimensionality of dataset embedding (if enabled)")
    ap.add_argument("--ct_embed_dim", type=int, default=16,
                    help="Dimensionality of cell_type embedding (if enabled)")
    ap.add_argument("--missing_gene_fill", type=str, default="nan", choices=["nan", "-1"],
                        help="Placeholder used in pseudobulk for missing genes; masked in Stage-1 losses")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.in_h5ad)
    pb_target = None  # pseudobulked target data for Stage-1 pretraining
    if args.include_target_pseudobulk:
        pb_target = make_pretrain_pseudobulk_from_adata(adata, args.target_label, args.control_label, dataset_id="target_all")
        sc.pp.normalize_total(pb_target, inplace=True)
        sc.pp.log1p(pb_target)
    if args.use_pseudobulk:  # stage 2 pseudobulk
        args.batch_size = 1  # enforce single-row batches
        adata = collapse_to_pseudobulk(adata, args.target_label)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway

    # ---------------------------
    # Train/Test setup
    # If --test_h5ad is provided, use that file for evaluation and ignore --test_pct_perts.
    # Otherwise, do the leave-perturbations-out split as before.
    # ---------------------------
    if args.test_h5ad:
        print(f"=== Using external TEST set: {args.test_h5ad} (overrides --test_pct_perts) ===")
        adata_train = adata
        adata_test = ad.read_h5ad(args.test_h5ad)
        # (Optional) apply the same pseudobulk collapse if requested
        if args.use_pseudobulk:
            adata_test = collapse_to_pseudobulk(adata_test, args.target_label)
        sc.pp.normalize_total(adata_test, inplace=True)
        sc.pp.log1p(adata_test)
        if sparse.isspmatrix(adata_test.X) and not sparse.isspmatrix_csr(adata_test.X):
            adata_test.X = adata_test.X.tocsr()  # nicer slicing, though we load to numpy anyway
        # Sanity check: same genes / order (as guaranteed by user)
        assert np.array_equal(adata_train.var_names.values, adata_test.var_names.values), \
            "Train and test var_names differ or are out of order."
        print(f"Train cells: {adata_train.n_obs}, Test cells: {adata_test.n_obs}")
    else:
        # Leave-perturbations-out split (previous behavior)
        labels_all = adata.obs[args.target_label].astype(str).values
        rng = np.random.default_rng(args.seed)
        all_perts = sorted({lbl for lbl in labels_all if lbl != args.control_label})
        n_test = int(round(args.test_pct_perts * len(all_perts)))
        test_perts = set(rng.choice(np.array(all_perts), size=n_test, replace=False).tolist()) if n_test > 0 else set()
        train_perts = [p for p in all_perts if p not in test_perts]
        mask_train = adata.obs[args.target_label].isin([args.control_label] + train_perts)
        adata_train = adata[mask_train].copy()
        if pb_target is not None:  # also filter pseudobulk to training perts only
            pb_target = pb_target[pb_target.obs[args.target_label].isin([args.control_label] + train_perts)].copy()
        adata_test = adata[adata.obs[args.target_label].isin([args.control_label] + list(test_perts))].copy() if n_test > 0 else None
        print("=== Split summary ===")
        print(f"Total perts (excl. control): {len(all_perts)}  |  Held-out test perts: {len(test_perts)}")
        if n_test > 0:
            print(f"Test perts: {sorted(test_perts)[:10]}{' ...' if len(test_perts) > 10 else ''}")
        print(f"Train cells: {adata_train.n_obs}, Test cells: {adata_test.n_obs if adata_test is not None else 0}")

    # ----- Optional: load pathway prior M_meta.npy (R x G) -----
    W_meta = None
    if args.meta_path:
        print(f"[prior] Loading pathway meta from {args.meta_path}")
        W_meta = np.load(args.meta_path)
        assert W_meta.ndim == 2 and W_meta.shape[1] == adata_train.n_vars, \
            f"M_meta shape mismatch: got {W_meta.shape}, expected (R,{adata_train.n_vars}) aligned to var_names."
        
    # Optional Stage-1: pseudobulk pretraining (reuses the same train() loop)
    model = None
    pb_paths = []
    pbs = []
    if pb_target is not None:
        pbs.append(pb_target)
    if args.pretrain_pseudobulk:
        pb_paths.append(args.pretrain_pseudobulk)
    if args.pretrain_pseudobulk_list:
        with open(args.pretrain_pseudobulk_list, "r") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                pb_paths.append(s)

    if len(pb_paths) > 0:
        for p in pb_paths:
            pb_i = ad.read_h5ad(p)
            pb_i = prep_external_data(pb_i, args.target_label, args.control_label, adata_train)
            pbs.append(pb_i)
        # Concatenate all pseudobulk rows
        pb_all = ad.concat(pbs, axis=0, join="outer", merge="same")
        pb_all.obs = pb_all.obs.copy()  # ensure contiguous
        print(f"=== Stage-1: pretraining on {len(pbs)} pseudobulk sources; total rows: {pb_all.n_obs} ===")

        model = train(
            adata=pb_all,
            target_label=args.target_label,
            control_label=args.control_label,
            hidden=args.hidden,
            T=args.T,
            epochs=args.pretrain_epochs,
            batch_size=1,
            lr=args.lr,
            weight_target=args.weight_target,     # keep α supervision for gene perts
            weight_local=0.0,
            seed=args.seed,
            tau=args.tau,
            device=args.device,
            match_controls="random",              # ignored in pretrain_mode due to per-dataset controls
            knn_k=args.knn_k,
            knn_temp=args.knn_temp,
            knn_metric=args.knn_metric,
            dist_loss="none",                     # no distribution loss in Stage-1
            weight_dist=0.0,
            swd_projections=args.swd_projections,
            single_pert_batches=False,
            W_meta=W_meta,
            init_from_meta=args.init_from_meta,
            weight_prior=args.weight_prior,
            meta_topk=args.meta_topk,
            model=None,
            pretrain_mode=True,
        )

    print(f"=== Stage-2: training on {'train+test' if adata_test is not None else 'train'} set ===")
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
        dist_loss=args.dist_loss,
        weight_dist=args.weight_dist,
        swd_projections=args.swd_projections,
        single_pert_batches=args.single_pert_batches,
        W_meta=W_meta,
        init_from_meta=args.init_from_meta,
        weight_prior=args.weight_prior,
        meta_topk=args.meta_topk,
        model=model,  # continue from Stage-1 if done
        pretrain_mode=False,
    )

    # Evaluate: external test if provided, else held-out split, else train split
    eval_adata = adata_test if adata_test is not None else adata_train
    print("\n=== Evaluation on {} set ===".format("TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)"))
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
