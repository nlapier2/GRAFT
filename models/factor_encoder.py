# models/factor_encoder.py
# Minimal pathway-anchored factor encoder (PyTorch, v1)
# - W (F x G): factor->gene dictionary (nonneg, column-normalized)
# - a(z): per-cell factor activations from an MLP on scVI z (nonneg)
# - Losses: ridge-consistency, masked L1 (outside), gentle anchor (inside), optional recon, optional decorrelation

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- utilities -----------------------------

def colnorm_nonneg(W_param: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Softplus to enforce nonnegativity, then column-normalize (sum_f W_fg = 1).
    W_param: (F, G) unconstrained
    returns W: (F, G) nonneg, col-normalized
    """
    W_pos = F.softplus(W_param)
    colsum = W_pos.sum(dim=0, keepdim=True) + eps
    return W_pos / colsum


def ridge_project_batch(xbar: torch.Tensor, W: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute a_lin = argmin_a ||x - W^T a||^2 + lam||a||^2 for a batch (ridge reconstruction target for batch).
    xbar: (B, G), W: (F, G). We use A = W (detached).
    a_lin = xbar @ ( (K^{-1} A)^T ), where K = A A^T + lam I, and solve K M = A.
    """
    A = W.detach()                                  # (F, G)
    # K = A A^T + lam I_F
    K = A @ A.T
    Fdim = K.shape[0]
    K = K + lam * torch.eye(Fdim, device=K.device, dtype=K.dtype)
    # Solve K M = A  (M: F x G)
    M = torch.linalg.solve(K, A)                    # (F, G)
    # a_lin = xbar @ M^T
    return xbar @ M.T                               # (B, F)


def offdiag_penalty(C: torch.Tensor) -> torch.Tensor:
    """Sum of squares of off-diagonal elements (optional, used to decorrelate factors if desired)."""
    return (C - torch.diag(torch.diag(C))).pow(2).sum()


# ------------------------ initializers / config -----------------------

@dataclass
class FactorEncoderConfig:
    n_genes: int
    n_anchor: int                    # number of anchored factors (rows from membership)
    n_free: int = 64                 # number of free factors
    add_junk: bool = True            # one junk factor to absorb uncovered genes
    mlp_hidden: int = 64
    mlp_layers: int = 2
    # losses / weights
    alpha_cons: float = 1.0          # ||a(z) - a_lin(xbar)||^2
    lambda_ridge: float = 0.1        # ridge for projection
    lambda_out: float = 5e-3         # L1 outside-pathway
    lambda_in: float = 1e-4          # gentle anchor inside
    beta_recon: float = 1e-2         # optional recon on scaled xbar
    gamma_cov: float = 0             # decorrelate factor activations
    # numerics
    eps_norm: float = 1e-8
    # Simple anchor regularization
    lambda_prior: float = 1e-3   # pull W toward W0 (dense prior)
    lambda_W: float = 1e-5       # small ridge on W


def build_W0_from_membership(
    M: torch.Tensor,
    n_free: int = 16,
    add_junk: bool = True,
    init_free_scale: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build initial W0 and anchor_mask from a membership matrix.
    M: (F_anchor, G) continuous or binary in [0,1].
    Returns:
      W0: (F, G) column-normalized nonneg init
      anchor_mask: (F, G) binary mask where M>0 for anchored rows; 0 elsewhere (free/junk rows all zero)
    """
    assert M.dim() == 2, "M must be (F_anchor, G)"
    F_anchor, G = M.shape
    extra = n_free + (1 if add_junk else 0)
    F_total = F_anchor + extra

    # Initialize with zeros, fill anchored rows from M
    W0 = torch.zeros((F_total, G), dtype=M.dtype, device=M.device)
    W0[:F_anchor, :] = torch.clamp(M, min=0.0)

    # Free factors: tiny random positives
    if n_free > 0:
        W0[F_anchor:F_anchor + n_free, :] = init_free_scale * torch.rand((n_free, G), dtype=M.dtype, device=M.device)

    # Junk factor: start tiny positive mass everywhere (will absorb genes outside any set after colnorm)
    if add_junk:
        W0[-1, :] = init_free_scale

    # Column-normalize
    W0 = torch.clamp(W0, min=0.0)
    W0 = W0 / (W0.sum(dim=0, keepdim=True) + 1e-8)

    # Anchor mask: only for anchored rows where M>0
    anchor_mask = torch.zeros_like(W0, dtype=torch.bool)
    anchor_mask[:F_anchor, :] = (M > 0)

    return W0, anchor_mask


# ------------------------------ model -------------------------------

class FactorEncoder(nn.Module):
    """
    Pathway-anchored factor encoder.
    Inputs per batch:
      z    : (B, d) scVI latent
      xbar : (B, G) denoised gene means (optionally scaled before calling)
    Parameters:
      W_param : (F, G) unconstrained (softplus + colnorm -> W)
      MLP on z -> a(z) >= 0
    Losses:
      ridge-consistency (a vs a_lin), masked L1 (outside), gentle inside anchor, optional recon, optional decorrelation.
    """

    def __init__(self, W0: torch.Tensor, anchor_mask: torch.Tensor, z_dim: int, cfg: FactorEncoderConfig):
        super().__init__()
        assert W0.shape == anchor_mask.shape, "W0 and anchor_mask must match"
        F_total, G = W0.shape
        self.cfg = cfg
        self.F = F_total
        self.G = G
        self.z_dim = z_dim

        # Store W0 for anchoring (buffer, not a parameter)
        self.register_buffer("W0", W0.clone())
        self.register_buffer("anchor_mask", anchor_mask.to(torch.bool))

        # Learnable unconstrained parameter for W
        # Start from inverse-softplus(W0) so that softplus(W_param) ≈ W0 before colnorm
        with torch.no_grad():
            W0_eps = torch.clamp(W0, min=1e-8)
            W0_inv = torch.log(torch.exp(W0_eps) - 1.0)  # inverse softplus
        self.W_param = nn.Parameter(W0_inv)

        # MLP: z -> a(z) >= 0 (softplus)
        layers = []
        in_dim = z_dim
        for _ in range(max(0, cfg.mlp_layers - 1)):
            layers.append(nn.Linear(in_dim, cfg.mlp_hidden))
            layers.append(nn.ReLU())
            in_dim = cfg.mlp_hidden
        layers.append(nn.Linear(in_dim, self.F))
        self.encoder = nn.Sequential(*layers)

        # Small trainable per-factor bias on activations (nonneg enforced via softplus)
        self.a_bias = nn.Parameter(torch.zeros(self.F))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass without losses (for inference):
          returns:
            a    : (B, F) nonneg activations
            W    : (F, G) nonneg, column-normalized
            xhat : (B, G) reconstructed means
        """
        W = colnorm_nonneg(self.W_param, eps=self.cfg.eps_norm)        # (F, G)
        a = F.softplus(self.encoder(z)) + F.softplus(self.a_bias)      # (B, F)
        xhat = a @ W                                                   # (B, G)
        return {"a": a, "W": W, "xhat": xhat}

    def compute_losses(
        self,
        z: torch.Tensor,
        xbar: torch.Tensor,
        xbar_scaled: Optional[torch.Tensor] = None,
        freeze_W: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and parts for a batch.
        Set freeze_W=True during warm-start (consistency only updates encoder).
        """
        out = self.forward(z)
        a, W, xhat = out["a"], out["W"], out["xhat"]

        # a_lin: ridge projection target (detach W inside ridge solve)
        a_lin = ridge_project_batch(xbar, W, lam=self.cfg.lambda_ridge)

        # Consistency loss
        L_cons = self.cfg.alpha_cons * F.mse_loss(a, a_lin)

        # --- Anchor regularization: simple, dense L2-to-prior (no masks) ---
        L_prior = torch.tensor(0.0, device=z.device)
        L_W = torch.tensor(0.0, device=z.device)
        if not freeze_W:
            W0 = self.W0.to(W.device, dtype=W.dtype)
            if self.cfg.lambda_prior > 0:
                L_prior = self.cfg.lambda_prior * torch.sum((W - W0) ** 2)
            if self.cfg.lambda_W > 0:
                L_W = self.cfg.lambda_W * torch.sum(W ** 2)

        # Optional reconstruction (use scaled xbar if provided)
        L_recon = torch.tensor(0.0, device=z.device)
        if self.cfg.beta_recon > 0.0:
            target = xbar_scaled if xbar_scaled is not None else xbar
            L_recon = self.cfg.beta_recon * F.mse_loss(xhat, target)

        # Optional decorrelation of activations
        L_cov = torch.tensor(0.0, device=z.device)
        if self.cfg.gamma_cov > 0.0 and a.shape[0] > 1:
            a_center = a - a.mean(dim=0, keepdim=True)
            C = (a_center.T @ a_center) / (a_center.shape[0] - 1 + 1e-6)
            L_cov = self.cfg.gamma_cov * offdiag_penalty(C)

        # Total
        L = L_cons + L_prior + L_W + L_recon + L_cov

        parts = {
            "L_total": L.detach(),
            "L_cons": L_cons.detach(),
            "L_prior": L_prior.detach(),
            "L_W": L_W.detach(),
            "L_recon": L_recon.detach(),
            "L_cov": L_cov.detach(),
        }
        return L, parts
