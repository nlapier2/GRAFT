
from __future__ import annotations
import torch
import torch.nn.functional as F

def target_knockdown_consistency(pred_x, true_x, target_idx, weight=1.0):
    """
    Penalize mismatch on the target gene only (when present).
    pred_x, true_x: (B, G)
    target_idx: (B,) with -1 when no target (controls)
    """
    mask = target_idx >= 0
    if not mask.any():
        return pred_x.new_tensor(0.0)
    b = torch.arange(pred_x.size(0), device=pred_x.device)[mask]
    g = target_idx[mask]
    return weight * F.mse_loss(pred_x[b, g], true_x[b, g])
