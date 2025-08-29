
from __future__ import annotations
import torch

def risk_extrapolation(per_env_losses):
    """
    REx: mean loss + variance of per-environment losses.
    per_env_losses: list[Tensor] (scalar each)
    """
    L = torch.stack(per_env_losses)  # (E,)
    return L.mean() + L.var(unbiased=False)
