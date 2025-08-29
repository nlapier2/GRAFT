"""
Consistency losses
==================

Losses that tie model predictions to **known per-gene effects** when a perturbation
targets a specific gene. These are lightweight, numerically stable, and designed
to work with the batching pattern used in `train_gnn.py`.

Main API
--------
- target_knockdown_consistency(y_pred, y_true, target_idx, ..., mode="mse")

Rationale
---------
Even if the main objective is **distributional** (e.g., sliced-Wasserstein per dataset),
we often know the direct target gene for each perturbed cell. This loss adds a
per-example scalar constraint on the **target gene coordinate** only, which:
  * provides a strong, low-variance supervision signal,
  * doesn't overfit the entire expression vector,
  * scales to huge G (thousands of genes) without extra memory.

It cleanly ignores controls (target_idx = -1). If a batch contains zero valid
targets, the functions return a zero loss to avoid NaNs.

Notes
-----
- All losses are computed in **normalized gene space** (same space as y_pred/y_true).
- We include a numerically-stable **Huber** option for robustness to outliers.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def _gather_target_cols(Y: torch.Tensor, target_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather the per-row target column from Y using target_idx.

    Parameters
    ----------
    Y : (B, G) tensor
    target_idx : (B,) int tensor with -1 for controls / no target, otherwise [0..G-1]

    Returns
    -------
    y_tgt : (M,) tensor of gathered target values for M valid rows
    mask  : (B,) boolean mask indicating which rows were valid (target_idx >= 0)
    """
    if Y.ndim != 2:
        raise ValueError("Y must be (B, G)")
    if target_idx.ndim != 1 or target_idx.size(0) != Y.size(0):
        raise ValueError("target_idx must be shape (B,) matching batch rows")

    device = Y.device
    B, G = Y.size()
    mask = target_idx >= 0
    if not torch.any(mask):
        # Return empty tensors on correct device/dtype
        return Y.new_zeros((0,)), mask

    # clamp indices to [0, G-1] to keep gather well-defined; mask will exclude invalids anyway
    idx = torch.clamp(target_idx[mask], 0, G - 1)
    y_sel = torch.gather(Y[mask], dim=1, index=idx.view(-1, 1)).squeeze(1)
    return y_sel, mask


def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0:
        return a.new_tensor(0.0)
    return F.mse_loss(a, b)


def _huber(a: torch.Tensor, b: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss between vectors.
    """
    if a.numel() == 0:
        return a.new_tensor(0.0)
    return F.huber_loss(a, b, delta=delta)


@torch.no_grad()
def _infer_direction_from_pair(y_true: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Heuristic direction: +1 if (y_true - x0) > 0 else -1.
    Used only if user requests a margin loss without providing directions.

    Returns
    -------
    dir_sign : (M,) in {+1, -1}
    """
    d = torch.sign(torch.clamp(y_true - x0, min=-1e9, max=1e9))
    d[d == 0] = -1.0  # default to knockdown if equal
    return d


def target_knockdown_consistency(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    weight: float = 1.0,
    mode: str = "mse",
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """
    Consistency loss on the *target gene only* for each perturbed cell.

    Parameters
    ----------
    y_pred : (B, G) predicted normalized expression
    y_true : (B, G) observed normalized expression (same decoding / transform_batch)
    target_idx : (B,) int tensor with -1 for controls; otherwise gene index
    weight : scalar multiplier
    mode : "mse" | "huber"
        - "mse": L2 on the target coordinate (default)
        - "huber": robust to outliers
    huber_delta : delta parameter for Huber loss

    Returns
    -------
    scalar loss tensor
    """
    y_pred_tgt, mask = _gather_target_cols(y_pred, target_idx)
    y_true_tgt, _    = _gather_target_cols(y_true, target_idx)

    if mode == "mse":
        base = _mse(y_pred_tgt, y_true_tgt)
    elif mode == "huber":
        base = _huber(y_pred_tgt, y_true_tgt, delta=huber_delta)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return weight * base


def target_margin_directional(
    y_pred: torch.Tensor,
    x0: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    direction: Optional[torch.Tensor] = None,
    margin: float = 0.1,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Enforce a **directional margin** at the target gene relative to a pre-state proxy x0.
    Useful for CRISPRi (down) vs CRISPRa (up) semantics.

    Parameters
    ----------
    y_pred : (B, G) predicted normalized expression
    x0 : (B, G) pre-state proxy normalized expression
    target_idx : (B,) int tensor (-1 for controls)
    direction : optional (M,) tensor in {+1 (up), -1 (down)} per valid row.
                If None, infer from (y_true - x0) at runtime (weak heuristic).
    margin : required margin size in normalized units (e.g., 0.1 on 1e4 scale)
    weight : scalar multiplier

    Returns
    -------
    scalar hinge loss encouraging:
        - down: y_pred <= x0 - margin
        - up:   y_pred >= x0 + margin
    """
    y_pred_tgt, mask = _gather_target_cols(y_pred, target_idx)
    x0_tgt, _        = _gather_target_cols(x0, target_idx)
    if y_pred_tgt.numel() == 0:
        return y_pred.new_tensor(0.0)

    if direction is None:
        dir_sign = _infer_direction_from_pair(y_pred_tgt.detach(), x0_tgt.detach())
    else:
        dir_sign = direction.to(y_pred_tgt).view(-1)
        dir_sign = torch.sign(dir_sign)
        dir_sign[dir_sign == 0] = -1.0

    # For down: want y_pred <= x0 - m  -> violation: y_pred - (x0 - m)
    # For up:   want y_pred >= x0 + m  -> violation: (x0 + m) - y_pred
    margin_vec = torch.full_like(y_pred_tgt, float(margin))
    down_mask = (dir_sign < 0)
    up_mask = ~down_mask

    viol = torch.zeros_like(y_pred_tgt)
    if torch.any(down_mask):
        viol[down_mask] = y_pred_tgt[down_mask] - (x0_tgt[down_mask] - margin_vec[down_mask])
    if torch.any(up_mask):
        viol[up_mask] = (x0_tgt[up_mask] + margin_vec[up_mask]) - y_pred_tgt[up_mask]

    hinge = torch.clamp(viol, min=0.0)
    return weight * torch.mean(hinge)
