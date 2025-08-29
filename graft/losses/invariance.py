"""
Invariance penalties
====================

This module implements two popular environment-invariance regularizers to improve
out-of-environment generalization across **datasets** in GRAFT:

1) Risk Extrapolation (REx)
---------------------------
Encourages the **per-environment risk** to be similar by penalizing the variance
of environment losses. Simple, stable, cheap.

    L_rex = Var( {L_e}_e )

Use when you can compute a **scalar loss per environment** (e.g., per-dataset
distribution loss). At training time you typically add:

    L_total = mean({L_e}) + λ * Var({L_e})

2) Invariant Risk Minimization (IRM, IRMv1 penalty)
---------------------------------------------------
Encourages each environment's loss to be **minimized by the same predictor**
by penalizing the gradient of the env loss w.r.t. a **dummy scalar** that scales
the predictor's outputs (Arjovsky et al., 2019). Zero gradient implies the same
optimum across environments for that representation/predictor.

    penalty_e = || ∂ L_e( scale * f(x) , y ) / ∂ scale |_{scale=1} ||^2
    L_irm = sum_e penalty_e

In practice we compute this with autograd and a per-env scalar `scale`.
You can apply IRM to:
  - the **target-gene coordinate** (recommended for CRISPRi/a semantics), or
  - the **whole vector** (heavier; often noisy in high dimensions).

API overview
------------
- risk_extrapolation(per_env_losses, unbiased=False) -> scalar variance
- rex_objective(per_env_losses, mean_weight=1.0, var_weight=1.0) -> mean + λ * var
- irmv1_penalty(y_pred, y_true, env_codes, mode="mse", use_target_only=False, target_idx=None)

Notes
-----
- `env_codes` should be int-encoded dataset IDs, one per row.
- For IRM on target-only, provide `target_idx` (B,) with -1 for controls.
- IRM adds second-order graph (grad-of-grad). Keep batch sizes moderate.
"""

from __future__ import annotations
from typing import Optional, Iterable, Tuple

import torch
import torch.nn.functional as F


# -----------------------------
# Risk Extrapolation (REx)
# -----------------------------
def risk_extrapolation(per_env_losses: Iterable[torch.Tensor], unbiased: bool = False) -> torch.Tensor:
    """
    Variance of per-environment scalar losses.

    Parameters
    ----------
    per_env_losses : iterable of scalar tensors (one per environment)
    unbiased : use unbiased variance (N-1 in denominator). Default False for stability.

    Returns
    -------
    scalar tensor with variance
    """
    losses = torch.stack([l for l in per_env_losses], dim=0)
    return losses.var(unbiased=unbiased)


def rex_objective(
    per_env_losses: Iterable[torch.Tensor],
    mean_weight: float = 1.0,
    var_weight: float = 1.0,
    unbiased_var: bool = False,
) -> torch.Tensor:
    """
    Combine mean loss with a variance penalty (REx objective).

    L = mean_weight * mean({L_e}) + var_weight * Var({L_e})
    """
    losses = torch.stack([l for l in per_env_losses], dim=0)
    return mean_weight * losses.mean() + var_weight * losses.var(unbiased=unbiased_var)


# -----------------------------
# IRM (IRMv1-style penalty)
# -----------------------------
def _gather_target_cols(Y: torch.Tensor, target_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather target coordinate per row (ignoring -1).
    Returns (y_tgt, mask) where y_tgt is (M,) and mask is (B,) bool.
    """
    B, G = Y.shape
    mask = target_idx >= 0
    if not torch.any(mask):
        return Y.new_zeros((0,)), mask
    idx = torch.clamp(target_idx[mask], 0, G - 1)
    y_sel = torch.gather(Y[mask], dim=1, index=idx.view(-1, 1)).squeeze(1)
    return y_sel, mask


def irmv1_penalty(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    env_codes: torch.Tensor,
    *,
    mode: str = "mse",
    use_target_only: bool = False,
    target_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    IRMv1 penalty: sum over environments of the squared gradient (w.r.t. a dummy
    scalar `scale`) of the environment loss.

    Parameters
    ----------
    y_pred : (B, G) predicted normalized expression
    y_true : (B, G) observed normalized expression
    env_codes : (B,) int tensor of dataset codes
    mode : "mse" | "huber"   (loss inside the penalty)
    use_target_only : if True, apply IRM to **target gene coordinate** only
    target_idx : required if use_target_only=True; (B,) with -1 for controls

    Returns
    -------
    scalar IRM penalty
    """
    penalty = 0.0
    for env_id in torch.unique(env_codes).tolist():
        m = (env_codes == env_id)
        if torch.count_nonzero(m) == 0:
            continue

        # Dummy scalar with grad
        scale = torch.tensor(1.0, device=y_pred.device, requires_grad=True)

        if use_target_only:
            if target_idx is None:
                raise ValueError("target_idx is required when use_target_only=True")
            y_pred_t, mask = _gather_target_cols(y_pred[m], target_idx[m])
            y_true_t, _    = _gather_target_cols(y_true[m], target_idx[m])
            if y_pred_t.numel() == 0:
                continue
            if mode == "mse":
                loss_e = F.mse_loss(scale * y_pred_t, y_true_t)
            elif mode == "huber":
                loss_e = F.huber_loss(scale * y_pred_t, y_true_t, delta=0.1)
            else:
                raise ValueError(f"Unknown mode '{mode}'")
        else:
            # full-vector penalty (heavier; usually not necessary)
            if mode == "mse":
                loss_e = F.mse_loss(scale * y_pred[m], y_true[m])
            elif mode == "huber":
                loss_e = F.huber_loss(scale * y_pred[m], y_true[m], delta=0.1)
            else:
                raise ValueError(f"Unknown mode '{mode}'")

        # grad wrt scale
        g = torch.autograd.grad(loss_e, scale, create_graph=True)[0]
        penalty = penalty + g.pow(2)

    return penalty
