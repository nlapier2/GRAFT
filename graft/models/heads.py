"""
Model heads for GRAFT
=====================

We separate two additively-composed effect heads:

1) MediatedHead
   -------------
   Predicts **factor/program activations** m(z[, a]) in R^{F}_{>=0} from the
   batch cell-state representation `z` (and optionally external factor features `a`).
   These activations are later mapped to gene deltas via a fixed or learnable
   dictionary U (genes x programs), outside this module.

   Design choices:
   - Tiny MLP, gated non-negativity (Softplus) to interpret `m` as program "usage".
   - Optional concatenation of precomputed factor features `a` (e.g., factor encoder output).
   - Lightweight LayerNorm for stability.

2) SparseDirectHead
   -----------------
   Predicts **direct gene deltas** Δx_dir(z) in R^{G}. This captures residual
   gene-specific effects not explained by mediated programs (e.g., crisp target
   gene changes, idiosyncratic edges).
   - Small MLP z → hidden → G with conservative init to keep this term modest.
   - Sparsity is enforced **via losses** (e.g., L1 penalty in the trainer).

Both heads are intentionally minimal for v1 and avoid assumptions about graph
structure; the graph propagation happens in the upstream `StatePropagator`.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_small_(w: torch.Tensor, gain: float = 0.1) -> None:
    """
    Xavier/Glorot init scaled down to keep early predictions small.
    """
    nn.init.xavier_uniform_(w, gain=gain)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * max(0, layers - 1) + [out_dim]
        mods = []
        for i in range(len(dims) - 2):
            lin = nn.Linear(dims[i], dims[i + 1])
            xavier_small_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods += [lin, nn.GELU(), nn.Dropout(dropout)]
        lin = nn.Linear(dims[-2], dims[-1])
        xavier_small_(lin.weight)
        nn.init.zeros_(lin.bias)
        mods += [lin]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MediatedHead(nn.Module):
    """
    z (and optional a) -> non-negative program activations m in R^{F}_{>=0}.

    Args
    ----
    z_dim : int
        Dimension of state embedding.
    F : int
        Number of programs/factors.
    hidden : int
        Hidden size for the MLP.
    use_factor_feats : bool
        If True, concatenate external factor features `a` (dimension a_dim).
    a_dim : int
        Dimension of external factor features (required if use_factor_feats).
    nonneg : bool
        If True, apply Softplus to outputs to keep m >= 0.
    dropout : float
        Dropout inside the MLP.
    """
    def __init__(
        self,
        z_dim: int,
        F: int,
        hidden: int = 256,
        use_factor_feats: bool = False,
        a_dim: int = 0,
        nonneg: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_factor_feats = use_factor_feats
        self.nonneg = nonneg
        in_dim = z_dim + (a_dim if use_factor_feats and a_dim > 0 else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, F, hidden=hidden, layers=2, dropout=dropout)
        self.out_scale = nn.Parameter(torch.tensor(1.0))  # simple global scale if needed

    def forward(self, z: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_factor_feats:
            if a is None:
                raise ValueError("MediatedHead expected factor features `a`, got None")
            x = torch.cat([z, a], dim=1)
        else:
            x = z
        x = self.norm(x)
        m = self.mlp(x) * self.out_scale
        if self.nonneg:
            m = F.softplus(m)  # non-negative program activations
        return m


class SparseDirectHead(nn.Module):
    """
    z -> Δx_dir in gene space (R^{G}).

    Args
    ----
    z_dim : int
        Dimension of state embedding.
    G : int
        Number of genes.
    hidden : int
        Hidden size for the MLP.
    dropout : float
        Dropout inside the MLP.
    bound : Optional[float]
        If provided, clamp outputs to [-bound, bound] to avoid extreme jumps.
    """
    def __init__(
        self,
        z_dim: int,
        G: int,
        hidden: int = 256,
        dropout: float = 0.0,
        bound: Optional[float] = None,
    ):
        super().__init__()
        self.bound = bound
        self.norm = nn.LayerNorm(z_dim)
        # Keep the head compact; a single hidden layer often suffices
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, G),
        )
        # Conservative init
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                xavier_small_(m.weight)
                nn.init.zeros_(m.bias)

        # Small output scaling to start with very small direct effects
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.norm(z)
        dx = self.mlp(x) * self.out_scale
        if self.bound is not None:
            dx = torch.clamp(dx, min=-self.bound, max=self.bound)
        return dx
