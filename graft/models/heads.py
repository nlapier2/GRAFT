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


class TrueSparseDirectHead(nn.Module):
    """
    Predicts direct gene deltas Δx_dir by conditioning on the intervention
    and the pre-perturbation state. This is a more causally faithful model
    of a "direct" effect than decoding from the post-perturbation state z_ref.

    Mechanism:
        - Embeds the target gene's identity.
        - Concatenates the pre-state z_q, target embedding, and clamp effectiveness eff.
        - An MLP maps this combined representation directly to a sparse gene-space delta.
    
    Args
    ----
    z_dim : int
        Dimension of the pre-perturbation state embedding (z_q).
    G : int
        Number of genes.
    n_genes : int
        Total number of genes in the vocabulary for the embedding layer.
    hidden : int
        Hidden size for the MLP.
    target_embed_dim : int
        Dimension of the target gene embedding.
    dropout : float
        Dropout inside the MLP.
    bound : Optional[float]
        If provided, clamp outputs to [-bound, bound].
    """
    def __init__(
        self,
        z_dim: int,
        n_genes: int,
        hidden: int = 256,
        target_embed_dim: int = 32,
        dropout: float = 0.0,
        bound: Optional[float] = None,
    ):
        super().__init__()
        self.bound = bound
        self.n_genes = n_genes

        # Embedding for the target gene's identity (+1 for a no-target token)
        self.target_embed = nn.Embedding(n_genes + 1, target_embed_dim)
        nn.init.normal_(self.target_embed.weight, mean=0.0, std=0.02)
        
        # Normalization layer for the input state
        self.norm = nn.LayerNorm(z_dim)

        # The MLP maps from the combined context to the gene-space delta
        mlp_in_dim = z_dim + target_embed_dim + 1  # +1 for the scalar `eff`
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_genes),
        )
        
        # Initialize MLP weights conservatively
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                xavier_small_(m.weight)
                nn.init.zeros_(m.bias)

        # Small output scaling to start with modest direct effects
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        z_q: torch.Tensor,
        target_idx: torch.Tensor,
        eff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce the direct-effect delta, Δx_dir.
        
        Note: The output is zero for control cells (where target_idx < 0).
        """
        # Map target indices to embeddings. Controls (idx=-1) map to the last token.
        idx = torch.where(target_idx >= 0, target_idx, torch.full_like(target_idx, self.n_genes))
        t_embed = self.target_embed(idx)
        
        # Normalize the pre-perturbation state
        z_norm = self.norm(z_q)
        
        # Combine all context information
        x = torch.cat([z_norm, t_embed, eff.view(-1, 1)], dim=1)
        
        # Predict the change in gene expression
        dx = self.mlp(x) * self.out_scale
        
        # Ensure the direct effect is zero for control cells
        control_mask = (target_idx < 0).float().view(-1, 1)
        dx = dx * (1.0 - control_mask)

        if self.bound is not None:
            dx = torch.clamp(dx, min=-self.bound, max=self.bound)
            
        return dx
