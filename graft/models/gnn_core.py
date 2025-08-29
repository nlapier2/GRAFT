
from __future__ import annotations
import torch
import torch.nn as nn

class StatePropagator(nn.Module):
    """
    Placeholder for message passing over a latent state graph.
    For v1, we approximate with a gated MLP residual block on (z) to produce a refined state z'.
    TODO: swap with a proper GNN over a kNN graph in state space (e.g., PyG).
    """
    def __init__(self, z_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        blocks = []
        d = z_dim
        for _ in range(layers):
            blocks += [
                nn.Linear(d, hidden), nn.ReLU(),
                nn.Linear(hidden, d)
            ]
        self.mlp = nn.Sequential(*blocks)
        self.gate = nn.Sigmoid()

    def forward(self, z):
        dz = self.mlp(z)
        alpha = self.gate(dz)
        return z + alpha * dz  # residual gated update
