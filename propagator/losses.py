"""
This script contains code for the loss functions used in the model.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def mse_loss(yhat, y):
    return F.mse_loss(yhat, y)

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
