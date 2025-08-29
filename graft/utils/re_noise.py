
from __future__ import annotations
import numpy as np

def sample_nb(mu: np.ndarray, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample NB counts independently per gene given mean mu and dispersion theta (gene-wise).
    Variance = mu + mu^2 / theta. Parameterize via Gamma-Poisson mixture.
    """
    eps = 1e-8
    shape = theta + eps
    rate = theta / (mu + eps)
    lam = rng.gamma(shape=shape, scale=1.0/(rate+eps))
    return rng.poisson(lam).astype(np.int32)

def renoise_batch(norm_pred: np.ndarray, libsize: np.ndarray, theta_gene: np.ndarray, rng=None) -> np.ndarray:
    """
    norm_pred: (B, G) normalized means (sum ~ library_size_ref)
    libsize: (B,) sampled library sizes
    theta_gene: (G,) NB dispersion (from scVI)
    """
    if rng is None:
        rng = np.random.default_rng(13)
    mu = norm_pred * libsize[:, None]
    out = np.zeros_like(mu, dtype=np.int32)
    for i in range(mu.shape[0]):
        out[i, :] = sample_nb(mu[i], theta_gene, rng)
    return out
