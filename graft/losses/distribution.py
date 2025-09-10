"""
Distribution-level losses
=========================

This module provides **mini-batch, distribution-matching losses** that compare
two *sets* of vectors (e.g., predicted vs observed normalized expression) **within
a dataset/environment** during training.

Implemented losses
------------------
- `sliced_wasserstein(X, Y, n_proj=32, p=2)`
    *Random-projection Wasserstein-p (SWD).* Projects onto `n_proj` random unit
    directions, computes the 1D Wasserstein-p distance by **sorting** each
    projected sample and averaging across projections.

- `mmd_rbf(X, Y, sigma=None, n_sigma=5)`
    *Maximum Mean Discrepancy with an RBF kernel.* Uses either a fixed `sigma`
    or a **median heuristic** to build a small mixture of RBF kernels.

Usage in training
-----------------
Per dataset slice `m`: we call, e.g., `sliced_wasserstein(y_pred[m], y_true[m])`
and average across datasets equally. This favors distributional matching without
requiring per-cell pairings.

SWD vs MMD: when to use which?
------------------------------
- **Sliced Wasserstein (SWD)**
  * Pros: captures **geometric structure** and heavy tails; *linear* memory; cost
    ~ `O(n_proj * B * G + n_proj * B log B)`; robust with a modest `n_proj` (e.g., 32–128);
    no kernel bandwidth to tune.
  * Cons: uses **sorting**, which gives piecewise-constant gradients (usually fine);
    needs enough batch size to get stable order statistics.
- **MMD (RBF)**
  * Pros: smooth gradients; simple to implement; can detect subtle shifts if
    kernel bandwidths match the scale.
  * Cons: requires choosing/tuning **sigma**; `O(B^2)` pairwise cost; can over-smooth
    or under-detect tails / multimodality if sigma is off.

In high-dimensional gene space with large G, SWD with 32–128 projections typically
gives a strong, stable signal and scales better than MMD. MMD can be useful as a
secondary term or for small-B validation.

Notes
-----
- Input shapes are `(B, G)`; the functions handle **unequal** sample sizes by
  *subsampling* the larger set down to `min(Bx, By)` (no grad through the index choice).
- All ops are PyTorch; gradients flow through values, not through sampling indices or random projections.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def _match_sizes(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample the larger of (X, Y) along batch dim to match the smaller size.
    Used by all losses here to handle unequal batch sizes.
    """
    Bx, _ = X.shape
    By, _ = Y.shape
    if Bx == By:
        return X, Y
    B = min(Bx, By)
    # Randomly subsample without replacement
    if Bx > B:
        indices = torch.randperm(Bx, device=X.device)[:B]
        X = X[indices]
    if By > B:
        indices = torch.randperm(By, device=Y.device)[:B]
        Y = Y[indices]
    return X, Y


def _unit_random_projections(G: int, K: int, device=None, dtype=None) -> torch.Tensor:
    """
    Sample K random unit vectors in R^G. Used by sliced_wasserstein.
    """
    v = torch.randn(G, K, device=device, dtype=dtype)
    v = v / (v.norm(dim=0, keepdim=True) + 1e-12)
    return v  # (G, K)


def sliced_wasserstein(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_proj: int = 128,
    p: int = 2,
) -> torch.Tensor:
    """
    Sliced Wasserstein-p distance between two batches X, Y of shape (B, G).
    Computes an approximation to the Wasserstein-p distance by projecting
    onto `n_proj` random 1D directions and averaging the 1D Wasserstein distances.

    Steps:
      1) Sample K = n_proj random unit directions v_k in R^G.
      2) Project: x_k = X @ v_k , y_k = Y @ v_k  -> (B, K)
      3) Sort each column and compute 1D W_p by pairing order statistics.
      4) Average over K and return a scalar.

    Returns
    -------
    scalar tensor (>=0)
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be (B, G)")
    X, Y = _match_sizes(X, Y)
    B, G = X.shape
    device = X.device
    dtype = X.dtype

    V = _unit_random_projections(G, n_proj, device=device, dtype=dtype)   # (G, K)
    x = X @ V  # (B, K)
    y = Y @ V  # (B, K)

    # Sort along batch dim for each projection
    x_sorted, _ = torch.sort(x, dim=0)
    y_sorted, _ = torch.sort(y, dim=0)

    # 1D Wasserstein-p per projection
    if p == 1:
        diff = torch.abs(x_sorted - y_sorted)
        w = diff.mean(dim=0)   # average over quantiles
    elif p == 2:
        diff = x_sorted - y_sorted
        w = torch.mean(diff * diff, dim=0) ** 0.5
    else:
        diff = torch.abs(x_sorted - y_sorted) ** p
        w = torch.mean(diff, dim=0) ** (1.0 / p)

    return w.mean()


def _pairwise_sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances between rows of A (N,d) and B (M,d).
    Returns (N, M).
    """
    a2 = (A * A).sum(dim=1, keepdim=True)         # (N,1), l2 norms of rows of A
    b2 = (B * B).sum(dim=1, keepdim=True).T       # (1,M), l2 norms of rows of B
    return torch.clamp(a2 + b2 - 2.0 * (A @ B.T), min=0.0)  # l2 squared distance for each row pair from A and B


def mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma: Optional[float] = None,
    n_sigma: int = 5,
    sigma_scale: float = 2.0,
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with an RBF kernel (mixture of bandwidths).

    If `sigma` is None, uses the **median heuristic** on the pooled sample to pick
    a base bandwidth, then builds a geometric grid of `n_sigma` values around it.

    Returns a **biased** MMD^2 estimator (sufficient for optimization).
    """
    X, Y = _match_sizes(X, Y)
    Z = torch.cat([X, Y], dim=0)  # (2B, G)
    B = X.size(0)

    # Pairwise squared distances
    Kxx = _pairwise_sqdist(X, X)
    Kyy = _pairwise_sqdist(Y, Y)
    Kxy = _pairwise_sqdist(X, Y)

    # Bandwidths
    if sigma is None:
        # Use median distance heuristic to set a base bandwidth if sigma not provided
        with torch.no_grad():
            D = _pairwise_sqdist(Z[:min(1024, Z.size(0))], Z[:min(1024, Z.size(0))]).detach()  # use a subset for efficiency
            med = torch.median(D[D > 0])
            base = torch.sqrt(med + 1e-8)  # sqrt since D is squared distance
            base = float(base.item()) if torch.isfinite(base) else 1.0
        sigmas = [base * (sigma_scale ** i) for i in range(-(n_sigma // 2), (n_sigma // 2) + 1)]
    else:
        sigmas = [float(sigma)]

    mmd2 = 0.0
    for s in sigmas:
        s2 = (s ** 2) + 1e-12
        kxx = torch.exp(-Kxx / (2 * s2))
        kyy = torch.exp(-Kyy / (2 * s2))
        kxy = torch.exp(-Kxy / (2 * s2))
        # Biased estimator (includes diagonal terms); good enough for training and faster
        mmd2 += (kxx.mean() + kyy.mean() - 2.0 * kxy.mean())

    return mmd2 / len(sigmas)


def _pairwise_euclidean(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between rows of A (N,d) and B (M,d).
    Returns (N, M). Uses squared distances + sqrt for stability/speed.
    """
    D2 = _pairwise_sqdist(A, B)
    return torch.sqrt(D2 + 1e-12)


def energy_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Energy distance between two batches X, Y (shape: (B, G)).

    Definition (sample version):
        ED(X,Y) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||
    We use the finite-sample analogue with means over all pairs (biased estimator).
    Returns a non-negative scalar; zero iff distributions match (in the limit).

    Complexity: O(B^2). Parameter-free; no projections or kernel bandwidth.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be (B, G)")
    X, Y = _match_sizes(X, Y)
    B = X.size(0)

    d_xy = _pairwise_euclidean(X, Y)         # (B, B) pairwise distances between cells from the two different batches
    d_xx = _pairwise_euclidean(X, X)         # (B, B) pairwise distances between cells from batch X
    d_yy = _pairwise_euclidean(Y, Y)         # (B, B) pairwise distances between cells from batch Y

    term1 = 2.0 * d_xy.mean()
    term2 = d_xx.mean()
    term3 = d_yy.mean()
    return term1 - term2 - term3
