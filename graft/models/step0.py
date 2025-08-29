"""
Step 0: Target clamp / effectiveness
====================================

This module implements the **time-step-0** semantics for targeted perturbations:
when a perturbation targets a specific gene, we apply a **learned immediate effect**
to that gene before any graph/message passing or mediated decoding.

Intuition
--------
- For CRISPRi, the target gene should be **knocked down** first, and only then do
  downstream effects propagate via cell state and programs.
- For controls (no target), do nothing.
- Effectiveness can vary by **environment** (dataset) and **cell state**.

API
---
class StepZeroClamp(nn.Module):
    def __init__(z_dim, n_labs, hidden=64, init_eff=0.9, mode="down")
    forward(x0, z_ref, env_codes, target_idx, direction=None)

Inputs
------
- x0 : (B, G) pre-state proxy in normalized space (e.g., matched control x̄)
- z_ref : (B, d) state embedding after propagation (or plain z)
- env_codes : (B,) int-encoded dataset IDs
- target_idx : (B,) int gene indices; -1 for controls
- direction : optional (B,) in {+1 (up), -1 (down)}; if None, uses `mode`.

Outputs
-------
- x_clamped : (B, G) where target coordinate is modified by a multiplicative factor
- eff : (B,) inferred effectiveness in [0,1] for diagnostics

Details
-------
Effectiveness model:
    eff_b = sigmoid( w_env[env_b] + MLP(z_ref_b) )

Clamp rule (normalized space):
    if down: x_tgt' = x_tgt * (1 - eff_b)
    if up:   x_tgt' = x_tgt * (1 + eff_b)

We keep it multiplicative (scale the mean) to be compatible with later decoding
and to avoid negative values. For safety, we cap the up-clamp to a modest range
(1 + eff ≤ 2.0) in v1; adjust as needed for CRISPRa datasets.

Note: This module does **not** enforce the direction from metadata; pass `direction`
if available. Otherwise `mode` sets a global default ("down" for CRISPRi-only v1).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepZeroClamp(nn.Module):
    def __init__(
        self,
        z_dim: int,
        n_labs: int,
        hidden: int = 64,
        init_eff: float = 0.9,
        mode: str = "down",
    ):
        super().__init__()
        self.mode = mode  # "down" or "up"; can be overridden per-sample via `direction`
        # Per-environment bias (captures protocol- and dataset-specific efficacy)
        self.env_bias = nn.Embedding(num_embeddings=n_labs, embedding_dim=1)
        with torch.no_grad():
            # Initialize toward desired init effectiveness through sigmoid inverse
            # eff0 ~ init_eff => preact ~ logit(init_eff)
            logit = torch.log(torch.tensor(init_eff) / (1.0 - torch.tensor(init_eff)))
            self.env_bias.weight.fill_(float(logit))

        # Small state-dependent adjustment
        self.state_mlp = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.state_mlp[-1].bias)  # start with small adjustment

    def forward(
        self,
        x0: torch.Tensor,
        z_ref: torch.Tensor,
        env_codes: torch.Tensor,
        target_idx: torch.Tensor,
        direction: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the clamp and return (x_clamped, eff). Controls are passed through.
        """
        B, G = x0.shape
        device = x0.device
        x = x0.clone()

        # Build per-sample effectiveness in (0,1)
        env_term = self.env_bias(env_codes.view(-1))  # (B,1)
        state_term = self.state_mlp(z_ref)            # (B,1)
        eff = torch.sigmoid(env_term + state_term).view(-1)  # (B,)

        # Determine direction per sample
        if direction is None:
            dir_sign = x0.new_full((B,), -1.0 if self.mode == "down" else 1.0)  # default all down for v1
        else:
            dir_sign = torch.sign(direction.to(x0)).view(-1)
            dir_sign[dir_sign == 0] = -1.0

        # Apply clamp only where target_idx >= 0
        mask = target_idx >= 0
        if torch.any(mask):
            rows = torch.nonzero(mask, as_tuple=False).view(-1)
            cols = torch.clamp(target_idx[mask], 0, G - 1)
            # Gather current target values
            x_tgt = x[rows, cols]
            e = eff[rows]
            d = dir_sign[rows]

            # down: scale by (1 - e), up: scale by (1 + e) but cap to 2x
            scale = torch.where(d < 0, 1.0 - e, torch.clamp(1.0 + e, max=2.0))
            x[rows, cols] = x_tgt * scale

        return x, eff
