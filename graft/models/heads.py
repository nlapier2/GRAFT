
from __future__ import annotations
import torch
import torch.nn as nn

class MediatedHead(nn.Module):
    """
    Δx_med = U @ m_theta(state, env)
    U: (G, F) fixed dictionary; m_theta outputs (B, F).
    """
    def __init__(self, z_dim: int, F: int, hidden: int = 256, use_factor_feats: bool = False, a_dim: int = 0):
        super().__init__()
        in_dim = z_dim + (a_dim if (use_factor_feats and a_dim>0) else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, F),
            nn.ReLU()  # nonnegative program coeffs
        )

    def forward(self, z, a=None):
        if a is not None:
            z = torch.cat([z, a], dim=-1)
        m = self.net(z)  # (B, F)
        return m

class SparseDirectHead(nn.Module):
    """
    Δx_dir: gene-wise sparse update conditioned on target gene.
    For v1 we implement a per-target small MLP whose output is (B, G) masked to be sparse via L1.
    TODO: replace with learned per-target indices or top-k gating.
    """
    def __init__(self, z_dim: int, G: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, G)
        )

    def forward(self, z):
        return self.net(z)  # (B, G)
