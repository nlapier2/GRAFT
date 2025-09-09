"""
State Propagator for Causal Perturbation Modeling
=================================================

This module defines the `StatePropagator`, the core component of the GRAFT
architecture responsible for modeling the dynamic response of a cell to a genetic
perturbation.

Purpose
-------
The primary goal of this module is to transform an initial, pre-perturbation
cell state embedding (`z_q`) into a refined, post-perturbation state (`z_ref`).
This is achieved by simulating a discrete-time dynamical process where the
cell's latent state is iteratively updated.

The model is "GNN-ish" (Graph Neural Network-like) in that it uses multi-step
updates to propagate information. However, instead of passing messages between
distinct nodes in a graph, it mixes information within the dimensions of the
latent state vector itself. This approach maintains the spirit of causal
propagation while remaining computationally efficient and scalable.

Key Components
--------------
- **Conditioning Adapters**: `TargetEmbed` and `FiLM` modules inject critical
  context about the specific perturbation and experimental environment, allowing
  the model to produce context-aware predictions.
- **Gated Residual Mixing**: The `ResidualBlock` serves as the core computational
  unit, providing a stable and adaptive mechanism for updating the cell state.

The final output, `z_ref`, is a rich, causally-informed embedding that serves
as a sufficient summary of the cell's post-perturbation state for all
downstream prediction heads.
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
    """
    Xavier/Glorot initialization with a small gain factor.
    
    This conservative initialization ensures that the outputs of a layer are
    small at the beginning of training, promoting stability, especially in
    deep networks or residual architectures.
    """
    nn.init.xavier_uniform_(w, gain=gain)


# -----------------------------------------------------------------------------
# Conditioning modules
# -----------------------------------------------------------------------------
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    This module conditions the network's behavior on an external categorical
    variable, such as the experimental environment (e.g., dataset, lab). It
    learns a unique affine transformation (a scaling `gamma` and a shift `beta`)
    for each category.

    The transformation is defined as:
        y = (1 + s * gamma_e) ⊙ x + s * beta_e
    
    where `gamma_e` and `beta_e` are learned embeddings for environment `e`, and `s`
    is a small, fixed scale factor to ensure the layer starts near the identity
    transform for stable training.

    Args:
        n_envs (int): The number of unique environments (e.g., datasets).
        dim (int): The feature dimension of the input tensor `x`.
        scale (float): A small scalar to keep the initial transformation
                       close to the identity function.
    """
    def __init__(self, n_envs: int, dim: int, scale: float = 1e-2):
        super().__init__()
        self.gamma = nn.Embedding(n_envs, dim)
        self.beta  = nn.Embedding(n_envs, dim)
        self.scale = float(scale)
        # Initialize gamma and beta to zero. This makes the FiLM layer act as an
        # identity function at the start of training (y = 1*x + 0), which is a
        # safe and stable starting point.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, env_codes: Optional[torch.Tensor]) -> torch.Tensor:
        # If no environment codes are provided, act as a passthrough.
        if env_codes is None:
            return x
        
        # Look up the gamma and beta vectors for each item in the batch.
        g = self.gamma(env_codes)  # (B, D)
        b = self.beta(env_codes)   # (B, D)
        
        # Apply the feature-wise affine transformation.
        return (1.0 + self.scale * g) * x + self.scale * b


class TargetEmbed(nn.Module):
    """
    Embeds a discrete target gene index into a continuous vector representation.

    This module converts a categorical input (the index of the perturbed gene)
    into a dense, learnable vector. This allows the model to reason about
    gene identity in a continuous space. A special token is reserved for
    control samples that have no target.

    Args:
        n_genes (int): The total number of genes in the vocabulary.
        dim (int): The dimensionality of the output embedding vector.
    """
    def __init__(self, n_genes: int, dim: int):
        super().__init__()
        self.n_genes = int(n_genes)
        # The embedding table has n_genes + 1 rows: one for each gene, plus
        # one extra token to represent the "no-target" (control) condition.
        self.emb = nn.Embedding(self.n_genes + 1, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, target_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if target_idx is None:
            # If no indices are provided, return the "no-target" embedding.
            # The caller is responsible for expanding it to the batch size if needed.
            return self.emb.weight[-1:].expand(1, -1)
        
        # Map control samples (indexed as -1) to the last row of the embedding
        # table, which corresponds to the dedicated "no-target" token.
        idx = torch.where(target_idx >= 0, target_idx, torch.full_like(target_idx, self.n_genes))
        return self.emb(idx)


# -----------------------------------------------------------------------------
# Gated residual mixing
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    A gated residual MLP block for state refinement.

    This block updates a state vector `x` using a combination of a standard
    MLP, a residual connection, and a learned gate. This structure is designed
    for stable and effective training of deep or recurrent architectures.

    The update rule is:
        h_in = LayerNorm(x)
        h    = MLP(h_in)
        gate = sigmoid(Linear(h_in))
        out  = x + gate ⊙ h

    Args:
        dim (int): The dimension of the input and output state vector.
        hidden (int): The dimension of the hidden layer within the MLP.
        dropout (float): The dropout rate applied within the MLP.
    """
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden)
        self.fc2  = nn.Linear(hidden, dim)
        self.gate_preact = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        # Conservative initialization helps stabilize training by ensuring that
        # the initial updates from this block are small.
        _xavier_small_(self.fc1.weight, gain=0.2); nn.init.zeros_(self.fc1.bias)
        _xavier_small_(self.fc2.weight, gain=0.2); nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.gate_preact.weight); nn.init.zeros_(self.gate_preact.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the input for stability.
        h_in = self.norm(x)
        
        # Main transformation path.
        h = self.fc1(h_in)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        
        # Gating path: learns to control the flow of information.
        g = torch.sigmoid(self.gate_preact(h_in))
        
        # Combine with a gated residual connection.
        return x + g * h


# -----------------------------------------------------------------------------
# StatePropagator
# -----------------------------------------------------------------------------
class StatePropagator(nn.Module):
    """
    Transforms an initial state `z` into a refined post-perturbation state `z_ref`
    through multi-step, conditioned, gated residual mixing.

    This module simulates the cell's dynamic response by iteratively applying a
    stack of `ResidualBlock`s. The process is conditioned on the perturbation
    identity and the experimental environment to produce a context-aware result.

    Args:
        z_dim (int): The dimension of the input and output latent state.
        hidden (int): The hidden dimension for the `ResidualBlock`s.
        layers (int): The number of `ResidualBlock`s in the stack per step.
        steps (int): The number of times the entire stack is applied. Simulates
                     discrete time steps in the propagation process.
        dropout (float): Dropout rate for the `ResidualBlock`s.
        use_env_film (bool): If True, apply FiLM conditioning for the environment.
        use_target_cond (bool): If True, condition on the perturbation target.
        target_embed_dim (int): The dimension for the target gene embedding.
        n_envs (int): The total number of unique environments for FiLM.
        n_genes (Optional[int]): The total number of genes, required if
                                 `use_target_cond` is True.
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

        # --- Conditioning Adapters ---
        self.film = FiLM(n_envs, z_dim) if use_env_film else None
        if use_target_cond:
            if n_genes is None:
                raise ValueError("n_genes is required when use_target_cond=True")
            self.tok = TargetEmbed(n_genes=n_genes, dim=target_embed_dim)
            # This linear layer fuses the state, target, and effectiveness info.
            self.fuse = nn.Linear(z_dim + target_embed_dim + 1, z_dim)
            _xavier_small_(self.fuse.weight, gain=0.2); nn.init.zeros_(self.fuse.bias)
        else:
            self.tok = None
            self.fuse = None

        # --- Core Propagation Stack ---
        # The stack of residual blocks is shared across all propagation steps.
        self.blocks = nn.ModuleList([ResidualBlock(z_dim, hidden, dropout=dropout) for _ in range(self.layers)])

    def forward(
        self,
        z: torch.Tensor,
        target_idx: Optional[torch.Tensor] = None,
        eff: Optional[torch.Tensor] = None,
        env_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute the forward pass to produce a refined state embedding `z_ref`.

        Args:
            z (torch.Tensor): The initial pre-perturbation state (B, z_dim).
            target_idx (Optional[torch.Tensor]): The target gene index for each
                sample in the batch (B,). Use -1 for controls.
            eff (Optional[torch.Tensor]): The learned clamp effectiveness for
                each sample (B,).
            env_codes (Optional[torch.Tensor]): The environment code for each
                sample (B,).

        Returns:
            torch.Tensor: The refined, post-perturbation state `z_ref` (B, z_dim).
        """
        x = z

        # 1. Initial Conditioning on Perturbation Identity & Strength
        if self.use_target_cond:
            if target_idx is None:
                # Use no-target embedding for the whole batch if not specified.
                t = self.tok(None).to(x.device).expand(x.shape[0], -1)
            else:
                t = self.tok(target_idx)
            
            if eff is None:
                # Use a zero effectiveness if not provided (e.g., for controls).
                eff = torch.zeros_like(x[:, :1])
            
            # Concatenate all information and fuse it into a single vector.
            x = torch.cat([x, t, eff.view(-1, 1)], dim=1)
            x = self.fuse(x)

        # 2. Multi-Step Propagation Loop
        # This simulates the dynamics of the cell's response over time.
        for _ in range(self.steps):
            # 2a. Apply environment-specific modulation.
            if self.use_env_film:
                x = self.film(x, env_codes)
            
            # 2b. Pass through the core state update blocks.
            for blk in self.blocks:
                x = blk(x)

        return x
