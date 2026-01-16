"""
Distribution-level losses
=========================

This module provides **mini-batch, distribution-matching losses** that compare
two *sets* of vectors (e.g., predicted vs observed normalized expression) **within
a dataset/environment** during training.

Implemented losses
------------------
- `sliced_wasserstein(X, Y, n_proj=32, p=2)`
    *Random-projection Wasserstein-p (SWD).* Projects onto `n_proj` random unit
    directions, computes the 1D Wasserstein-p distance by **sorting** each
    projected sample and averaging across projections.

- `mmd_rbf(X, Y, sigma=None, n_sigma=5)`
    *Maximum Mean Discrepancy with an RBF kernel.* Uses either a fixed `sigma`
    or a **median heuristic** to build a small mixture of RBF kernels.

Usage in training
-----------------
Per dataset slice `m`: we call, e.g., `sliced_wasserstein(y_pred[m], y_true[m])`
and average across datasets equally. This favors distributional matching without
requiring per-cell pairings.

SWD vs MMD: when to use which?
------------------------------
- **Sliced Wasserstein (SWD)**
  * Pros: captures **geometric structure** and heavy tails; *linear* memory; cost
    ~ `O(n_proj * B * G + n_proj * B log B)`; robust with a modest `n_proj` (e.g., 32–128);
    no kernel bandwidth to tune.
  * Cons: uses **sorting**, which gives piecewise-constant gradients (usually fine);
    needs enough batch size to get stable order statistics.
- **MMD (RBF)**
  * Pros: smooth gradients; simple to implement; can detect subtle shifts if
    kernel bandwidths match the scale.
  * Cons: requires choosing/tuning **sigma**; `O(B^2)` pairwise cost; can over-smooth
    or under-detect tails / multimodality if sigma is off.

In high-dimensional gene space with large G, SWD with 32–128 projections typically
gives a strong, stable signal and scales better than MMD. MMD can be useful as a
secondary term or for small-B validation.

Notes
-----
- Input shapes are `(B, G)`; the functions handle **unequal** sample sizes by
  *subsampling* the larger set down to `min(Bx, By)` (no grad through the index choice).
- All ops are PyTorch; gradients flow through values, not through sampling indices or random projections.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def _match_sizes(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample the larger of (X, Y) along batch dim to match the smaller size.
    Used by all losses here to handle unequal batch sizes.
    """
    Bx, _ = X.shape
    By, _ = Y.shape
    if Bx == By:
        return X, Y
    B = min(Bx, By)
    # Randomly subsample without replacement
    if Bx > B:
        indices = torch.randperm(Bx, device=X.device)[:B]
        X = X[indices]
    if By > B:
        indices = torch.randperm(By, device=Y.device)[:B]
        Y = Y[indices]
    return X, Y


def _unit_random_projections(G: int, K: int, device=None, dtype=None) -> torch.Tensor:
    """
    Sample K random unit vectors in R^G. Used by sliced_wasserstein.
    """
    v = torch.randn(G, K, device=device, dtype=dtype)
    v = v / (v.norm(dim=0, keepdim=True) + 1e-12)
    return v  # (G, K)


def sliced_wasserstein(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_proj: int = 128,
    p: int = 2,
) -> torch.Tensor:
    """
    Sliced Wasserstein-p distance between two batches X, Y of shape (B, G).
    Computes an approximation to the Wasserstein-p distance by projecting
    onto `n_proj` random 1D directions and averaging the 1D Wasserstein distances.

    Steps:
      1) Sample K = n_proj random unit directions v_k in R^G.
      2) Project: x_k = X @ v_k , y_k = Y @ v_k  -> (B, K)
      3) Sort each column and compute 1D W_p by pairing order statistics.
      4) Average over K and return a scalar.

    Returns
    -------
    scalar tensor (>=0)
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be (B, G)")
    X, Y = _match_sizes(X, Y)
    B, G = X.shape
    device = X.device
    dtype = X.dtype

    V = _unit_random_projections(G, n_proj, device=device, dtype=dtype)   # (G, K)
    x = X @ V  # (B, K)
    y = Y @ V  # (B, K)

    # Sort along batch dim for each projection
    x_sorted, _ = torch.sort(x, dim=0)
    y_sorted, _ = torch.sort(y, dim=0)

    # 1D Wasserstein-p per projection
    if p == 1:
        diff = torch.abs(x_sorted - y_sorted)
        w = diff.mean(dim=0)   # average over quantiles
    elif p == 2:
        diff = x_sorted - y_sorted
        w = torch.mean(diff * diff, dim=0) ** 0.5
    else:
        diff = torch.abs(x_sorted - y_sorted) ** p
        w = torch.mean(diff, dim=0) ** (1.0 / p)

    return w.mean()


def _pairwise_sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances between rows of A (N,d) and B (M,d).
    Returns (N, M).
    """
    a2 = (A * A).sum(dim=1, keepdim=True)         # (N,1), l2 norms of rows of A
    b2 = (B * B).sum(dim=1, keepdim=True).T       # (1,M), l2 norms of rows of B
    return torch.clamp(a2 + b2 - 2.0 * (A @ B.T), min=0.0)  # l2 squared distance for each row pair from A and B


def mmd_poly2(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    MMD with polynomial-degree-2 kernel: k(x,y) = (x·y)^2.
    This kernel matches means and (uncentered) covariances in expectation.
    Returns a biased MMD^2 estimator (fine for optimization).
    """
    X, Y = _match_sizes(X, Y)
    Kxx = (X @ X.t()) ** 2
    Kyy = (Y @ Y.t()) ** 2
    Kxy = (X @ Y.t()) ** 2
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


def mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: Optional[float] = None,
    n_sigma: int = 5,
    sigma_scale: float = 2.0,
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with an RBF kernel (mixture of bandwidths).

    If `sigma` is None, uses the **median heuristic** on the pooled sample to pick
    a base bandwidth, then builds a geometric grid of `n_sigma` values around it.

    Returns a **biased** MMD^2 estimator (sufficient for optimization).
    """
    X, Y = _match_sizes(X, Y)
    Z = torch.cat([X, Y], dim=0)  # (2B, G)
    B = X.size(0)

    # Pairwise squared distances
    Kxx = _pairwise_sqdist(X, X)
    Kyy = _pairwise_sqdist(Y, Y)
    Kxy = _pairwise_sqdist(X, Y)

    # Bandwidths
    if sigma is None:
        # Use median distance heuristic to set a base bandwidth if sigma not provided
        with torch.no_grad():
            D = _pairwise_sqdist(Z[:min(1024, Z.size(0))], Z[:min(1024, Z.size(0))]).detach()  # use a subset for efficiency
            med = torch.median(D[D > 0])
            base = torch.sqrt(med + 1e-8)  # sqrt since D is squared distance
            base = float(base.item()) if torch.isfinite(base) else 1.0
        sigmas = [base * (sigma_scale ** i) for i in range(-(n_sigma // 2), (n_sigma // 2) + 1)]
    else:
        sigmas = [float(sigma)]

    mmd2 = 0.0
    for s in sigmas:
        s2 = (s ** 2) + 1e-12
        kxx = torch.exp(-Kxx / (2 * s2))
        kyy = torch.exp(-Kyy / (2 * s2))
        kxy = torch.exp(-Kxy / (2 * s2))
        # Biased estimator (includes diagonal terms); good enough for training and faster
        mmd2 += (kxx.mean() + kyy.mean() - 2.0 * kxy.mean())

    return mmd2 / len(sigmas)


def _pairwise_euclidean(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between rows of A (N,d) and B (M,d).
    Returns (N, M). Uses squared distances + sqrt for stability/speed.
    """
    D2 = _pairwise_sqdist(A, B)
    return torch.sqrt(D2 + 1e-12)


def energy_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Energy distance between two batches X, Y (shape: (B, G)).

    Definition (sample version):
        ED(X,Y) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||
    We use the finite-sample analogue with means over all pairs (biased estimator).
    Returns a non-negative scalar; zero iff distributions match (in the limit).

    Complexity: O(B^2). Parameter-free; no projections or kernel bandwidth.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be (B, G)")
    X, Y = _match_sizes(X, Y)
    B = X.size(0)

    d_xy = _pairwise_euclidean(X, Y)         # (B, B) pairwise distances between cells from the two different batches
    d_xx = _pairwise_euclidean(X, X)         # (B, B) pairwise distances between cells from batch X
    d_yy = _pairwise_euclidean(Y, Y)         # (B, B) pairwise distances between cells from batch Y

    term1 = 2.0 * d_xy.mean()
    term2 = d_xx.mean()
    term3 = d_yy.mean()
    return term1 - term2 - term3

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

def target_efficacy_loss(yhat, bx_ctrl, bx_pert, target_idx, alpha_vec):
    # per-cell target MSE loss vs true efficacy
    loss_t = yhat.new_tensor(0.0)
    if (target_idx >= 0).any():
        mask = (target_idx >= 0)
        rows = torch.arange(target_idx.numel(), device=yhat.device)[mask]
        cols = target_idx[mask]
        # counts for target gene in matched control vs perturbed cell
        ctrl_cnt = torch.expm1(bx_ctrl[rows, cols].clamp_min(0.0))
        pert_cnt = torch.expm1(bx_pert[rows, cols].clamp_min(0.0))
        true_alpha = (1.0 - pert_cnt / (ctrl_cnt + 1e-8)).clamp(0.0, 1.0)
        loss_t = F.mse_loss(alpha_vec[mask], true_alpha)
    return loss_t

def target_efficacy_batch_loss(bx_ctrl, bx_pert, target_idx, alpha_vec, batch_labels):
    """
    Batch-level target loss (pseudobulk α):
      For each perturbation label present in the batch, compute the target-gene
      counts from the batch mean control vs batch mean perturbed, derive a
      pseudobulk true_alpha, and compare to the MEAN predicted alpha over rows
      of that label. Averaged across labels present.
    Args:
      alpha_vec: (B,) predicted per-row alphas
      bx_ctrl, bx_pert: (B,G) log1p controls/perturbed
      target_idx: (B,) target gene indices (-1 if unknown)
      batch_labels: list[str] length B with the perturbation labels
    """
    device = bx_ctrl.device
    B = alpha_vec.shape[0]
    # group row indices by label
    by_lbl = defaultdict(list)
    for i, lbl in enumerate(batch_labels):
        by_lbl[lbl].append(i)
    eps = 1e-8
    losses = []
    for lbl, idxs in by_lbl.items():
        idx = torch.tensor(idxs, device=device, dtype=torch.long)
        # choose the (majority) target index among rows; require at least one known target
        t_rows = target_idx[idx]
        if (t_rows >= 0).any():
            # take the first known target index (they should be identical within a label)
            t = int(t_rows[t_rows >= 0][0].item())
            # pseudobulk counts at target gene
            ctrl_cnt = torch.expm1(bx_ctrl[idx, t].clamp_min(0.0)).mean()
            pert_cnt = torch.expm1(bx_pert[idx, t].clamp_min(0.0)).mean()
            true_alpha = (1.0 - pert_cnt / (ctrl_cnt + eps)).clamp(0.0, 1.0)
            pred_alpha = alpha_vec[idx].mean()
            losses.append((pred_alpha - true_alpha).pow(2))
    if len(losses) == 0:
        return alpha_vec.new_tensor(0.0)
    return torch.stack(losses).mean()

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

def prototype_loss(yhat, x0, bx_pert, bx_ctrl, by_lbl, t2gi, pretrain_mode=False):
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
    return loss_proto

def prior_loss(yhat, model, W_meta, weight_prior):
    # Optional proximity loss: keep node_E near projected prior
    loss_prior = yhat.new_tensor(0.0)
    if (weight_prior > 0.0) and (W_meta is not None) and (model.prior_proj is not None):
        # compute current projected prior: (G,node_dim)
        with torch.no_grad():
            Wm_torch = torch.from_numpy(W_meta.astype(np.float32)).to(yhat.device)
        E_prior = model.prior_proj(Wm_torch.T)  # (G,node_dim)
        loss_prior = F.mse_loss(model.node_E.weight, E_prior)
    return loss_prior

def compute_distance_loss(yhat, x0, bx_pert, btargets, by_lbl, t2gi, dist_loss, pretrain_mode=False, swd_projections=64):
    loss_dist = yhat.new_tensor(0.0)
    if (not pretrain_mode) and (dist_loss != "none"):
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
                loss_dist = loss_dist + sliced_wasserstein(d_pred, d_true, n_proj=swd_projections)
            elif dist_loss == "energy":
                loss_dist = loss_dist + energy_distance(d_pred, d_true)
        # average across present perts
        n_groups = sum(1 for v in by_lbl.values() if len(v) >= 2)
        if n_groups > 0:
            loss_dist = loss_dist / n_groups
    return loss_dist

def info_nce_loss(q: torch.Tensor,
                  k_pos: torch.Tensor,
                  neg_bank: torch.Tensor,
                  tau: float = 0.1,
                  neg_mask: torch.Tensor = None) -> torch.Tensor:
    """
    InfoNCE over cosine similarities.
    Args:
      q:      (B,D) queries (L2-normalized)
      k_pos:  (B,D) positives (L2-normalized)   -- we use B=1 for single-pert batches
      neg_bank: (N,D) negatives (L2-normalized)
      tau: temperature
      neg_mask: (N,) bool mask; True=keep as negative, False=drop
    """
    assert q.ndim == 2 and k_pos.ndim == 2
    B, D = q.shape
    if neg_bank is None or neg_bank.numel() == 0:
        # fall back to just positive (avoid NaN early in training)
        # loss = -log(exp(sim_pos/tau)/exp(sim_pos/tau)) = 0
        return q.new_tensor(0.0)
    if neg_mask is not None:
        neg_bank = neg_bank[neg_mask]
        if neg_bank.numel() == 0:
            return q.new_tensor(0.0)
    # cos sims
    sim_pos = (q * k_pos).sum(dim=-1, keepdim=True)              # (B,1)
    sim_neg = q @ neg_bank.t()                                    # (B,N)
    logits = torch.cat([sim_pos, sim_neg], dim=1) / max(tau, 1e-8)
    labels = q.new_zeros(B, dtype=torch.long)  # positives at index 0
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return loss