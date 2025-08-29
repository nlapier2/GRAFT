
from __future__ import annotations
import torch
import torch.nn.functional as F

def sliced_wasserstein(x, y, n_proj: int = 64):
    """
    Max-sliced Wasserstein (approx): sample random projections and average 1D W1.
    x, y: (B, G)
    """
    device = x.device
    G = x.size(1)
    dirs = torch.randn(n_proj, G, device=device)
    dirs = F.normalize(dirs, p=2, dim=1)
    xs = x @ dirs.T  # (B, P)
    ys = y @ dirs.T
    xs, _ = torch.sort(xs, dim=0)
    ys, _ = torch.sort(ys, dim=0)
    return torch.mean(torch.abs(xs - ys))
