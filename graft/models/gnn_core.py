"""
GNN Core
========

`StatePropagator` transforms the raw scVI latent `z` into a **refined, message-passed
state** `z_ref`. It’s “GNN-ish” in that we apply multi-step gated residual mixing
in *state space* with optional conditioning on **environment** (dataset) and
**target identity**, keeping compute light while approximating causal propagation.

Connections
-----------
- `StepZeroClamp` uses `z_ref` to compute target efficacy at time step 0.
- `MediatedHead` / `SparseDirectHead` consume `z_ref` to produce mediated and
  direct gene-space deltas.
- Invariance and distribution losses act on decoded predictions, while conditioning
  here helps transfer across datasets.

Design
------
- **FiLM**: feature-wise affine modulation per environment, initialized to identity.
- **TargetEmbed**: small embedding for the current target gene (with a no-target token).
- **ResidualBlock**: LayerNorm → MLP → gated residual (sigmoid gate) with dropout.
- **Steps**: repeat the residual stack for a few steps (weights shared across steps).

The public API is:
    z_ref = StatePropagator(...)(z, target_idx=None, env_codes=None)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helpers / initialization
# -----------------------------------------------------------------------------
def _xavier_small_(w: torch.Tensor, gain: float = 0.1) -> None:
    nn.init.xavier_uniform_(w, gain=gain)


# -----------------------------------------------------------------------------
# Conditioning modules
# -----------------------------------------------------------------------------
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    y = (1 + s * gamma_e) ⊙ x + s * beta_e

    where gamma_e, beta_e are learned embeddings for environment e and s is a
    small scale so the layer starts near identity.
    """
    def __init__(self, n_envs: int, dim: int, scale: float = 1e-2):
        super().__init__()
        self.gamma = nn.Embedding(n_envs, dim)
        self.beta  = nn.Embedding(n_envs, dim)
        self.scale = float(scale)
        # Init near zero so 1 + s*gamma ≈ 1 and s*beta ≈ 0 at start
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, env_codes: Optional[torch.Tensor]) -> torch.Tensor:
        if env_codes is None:
            return x
        g = self.gamma(env_codes)  # (B, D)
        b = self.beta(env_codes)   # (B, D)
        return (1.0 + self.scale * g) * x + self.scale * b


class TargetEmbed(nn.Module):
    """
    Embed a target gene index into a small vector. Controls map to a special token.
    """
    def __init__(self, n_genes: int, dim: int):
        super().__init__()
        self.n_genes = int(n_genes)
        self.emb = nn.Embedding(self.n_genes + 1, dim)  # last index = no-target
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, target_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if target_idx is None:
            # Use the no-target token for all rows if no indices provided
            return self.emb.weight[-1:].expand(1, -1)  # caller should expand to batch size if needed
        # Map -1 -> no-target index
        idx = torch.where(target_idx >= 0, target_idx, torch.full_like(target_idx, self.n_genes))
        return self.emb(idx)


# -----------------------------------------------------------------------------
# Gated residual mixing
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Gated residual MLP block.

    x -> LN -> Linear(h) -> GELU -> Dropout -> Linear(d) -> h
    gate = sigmoid(LN(x) -> Linear(d))
    out = x + gate ⊙ h
    """
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden)
        self.fc2  = nn.Linear(hidden, dim)
        self.gate_preact = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        # Conservative init
        _xavier_small_(self.fc1.weight, gain=0.2); nn.init.zeros_(self.fc1.bias)
        _xavier_small_(self.fc2.weight, gain=0.2); nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.gate_preact.weight); nn.init.zeros_(self.gate_preact.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in = self.norm(x)
        h = self.fc1(h_in)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate_preact(h_in))
        return x + g * h


# -----------------------------------------------------------------------------
# StatePropagator
# -----------------------------------------------------------------------------
class StatePropagator(nn.Module):
    """
    Multi-step gated residual mixing in state space with optional conditioning.
    """
    def __init__(
        self,
        z_dim: int,
        hidden: int = 256,
        layers: int = 2,
        steps: int = 2,
        dropout: float = 0.0,
        use_env_film: bool = True,
        use_target_cond: bool = False,
        target_embed_dim: int = 32,
        n_envs: int = 1,
        n_genes: Optional[int] = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.layers = int(layers)
        self.steps = int(steps)
        self.use_env_film = use_env_film
        self.use_target_cond = use_target_cond

        # Conditioning adapters
        self.film = FiLM(n_envs, z_dim) if use_env_film else None
        if use_target_cond:
            if n_genes is None:
                raise ValueError("n_genes is required when use_target_cond=True")
            self.tok = TargetEmbed(n_genes=n_genes, dim=target_embed_dim)
            self.fuse = nn.Linear(z_dim + target_embed_dim + 1, z_dim)
            _xavier_small_(self.fuse.weight, gain=0.2); nn.init.zeros_(self.fuse.bias)
        else:
            self.tok = None
            self.fuse = None

        # Residual stack (shared across steps for stability/efficiency)
        self.blocks = nn.ModuleList([ResidualBlock(z_dim, hidden, dropout=dropout) for _ in range(self.layers)])

    def forward(
        self,
        z: torch.Tensor,
        target_idx: Optional[torch.Tensor] = None,
        eff: Optional[torch.Tensor] = None,
        env_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Produce a refined state embedding `z_ref` for downstream heads.
        """
        x = z

        # Optional target conditioning: concatenate token and project back to z_dim
        if self.use_target_cond:
            if target_idx is None:
                # use no-target embedding for the whole batch
                t = self.tok(None).to(x.device).expand(x.shape[0], -1)
            else:
                t = self.tok(target_idx)
            if eff is None:
                eff = torch.zeros_like(x[:, :1]) # Fallback if eff not provided
            x = torch.cat([x, t, eff.view(-1, 1)], dim=1)
            x = self.fuse(x)

        # Multi-step propagation (weight sharing across steps)
        for _ in range(self.steps):
            if self.use_env_film:
                x = self.film(x, env_codes)
            for blk in self.blocks:
                x = blk(x)

        return x
