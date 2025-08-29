
from __future__ import annotations
import torch
import torch.nn as nn

class StepZeroClamp(nn.Module):
    """
    Soft 'do' clamp at t=0 on the target gene (if known).

    y0 = x (normalized)
    For a target gene g, apply: x'_g = (1 - e) * x_g, where efficacy e in (0,1).
    Efficacy is predicted by a small MLP conditioned on pre-state z and lab embedding.
    """
    def __init__(self, z_dim: int, n_labs: int, hidden: int = 64, init_eff: float = 0.9):
        super().__init__()
        self.lab_emb = nn.Embedding(num_embeddings=max(n_labs,1), embedding_dim=16)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 16, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        # initialize near init_eff
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
        self.init_eff = init_eff

    def forward(self, x, z, lab_ids, target_idx):
        """
        x: (B, G) normalized pre-state expressions
        z: (B, d)
        lab_ids: (B,) int in [0, n_labs)
        target_idx: (B,) int gene indices (or -1 for control)
        """
        lab_vec = self.lab_emb(lab_ids) if self.lab_emb.num_embeddings>1 else torch.zeros(z.size(0), 16, device=z.device)
        h = torch.cat([z, lab_vec], dim=-1)
        eff = self.mlp(h).squeeze(-1)  # (B,) in (0,1)
        x_clamped = x.clone()
        mask = target_idx >= 0
        if mask.any():
            idx = target_idx[mask]
            b = torch.arange(x.size(0), device=x.device)[mask]
            x_clamped[b, idx] = (1.0 - eff[mask]) * x[b, idx]
        return x_clamped, eff
