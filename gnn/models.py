#!/usr/bin/env python3
import argparse, math, os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import pairwise_distances


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
    def __init__(self, G: int, hidden: int = 128, T: int = 2, tau: float = 0.0, alpha_cap: float = 1.0, node_dim: int = 128,
                 prior_dim: int | None = None, dset_vocab: int = 0, dset_dim: int = 0, ct_vocab: int = 0, ct_dim: int = 0, proj_dim: int = 128):
        super().__init__()
        self.G = G
        self.hidden = hidden
        self.T = T
        self.alpha_cap = alpha_cap
        self.proj_dim = proj_dim
        # per-gene node embeddings (used by all nodes, every batch)
        self.node_dim = node_dim
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
        # --- contrastive projection head: R^G -> R^{proj_dim} ---
        self.delta_proj = nn.Sequential(
            nn.Linear(self.G, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )

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

    # --- helper to project deltas and L2-normalize for cosine similarity ---
    @torch.no_grad()
    def project_key(self, delta_vec: torch.Tensor) -> torch.Tensor:
        """
        Args: delta_vec: (..., G)
        Returns: (..., D) L2-normalized key embedding (no grad)
        """
        k = self.delta_proj(delta_vec)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-8)
        return k

    def project_query(self, delta_vec: torch.Tensor) -> torch.Tensor:
        """
        Same as project_key but with grad (for queries from predicted deltas).
        """
        q = self.delta_proj(delta_vec)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        return q
