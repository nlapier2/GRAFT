"""
Re-noising utilities
====================
Convert model predictions in **normalized gene space** (e.g., scVI normalized means)
back to **count space** by sampling from a Negative Binomial (NB) model whose
mean scales with per-cell library size and whose overdispersion can be supplied
(from scVI) or estimated from controls.

Design goals
------------
- **Library-aware:** scale normalized means by a sampled or provided library size.
- **Flexible dispersion:** use provided per-gene NB `theta` (r in NB), or estimate
  alpha in Var = mu + alpha*mu^2 from control counts (theta = 1/alpha).
- **Lightweight:** NumPy + SciPy sparse; no scVI dependency here.
- **Deterministic option:** return expected counts (no sampling) for fast evaluation.

Key conventions
---------------
Let `xbar` be a (B, G) matrix of **normalized means** produced with a reference
library size `L_ref` (e.g., 1e4). For a target library size `L` per cell, the
expected counts are `mu_counts = xbar * (L / L_ref)`.

NB parameterization
-------------------
We use the (mean, theta) parameterization:
    Var[count] = mu + mu^2 / theta
Sampling is implemented via Gamma-Poisson mixture:
    rate ~ Gamma(shape=theta, scale=mu/theta)
    count ~ Poisson(rate)

If `theta` is None, we fall back to **Poisson**.

Typical usage
-------------
from graft.utils.re_noise import ReNoiser, estimate_alpha_from_counts, write_anndata

# 1) Fit alphas on control counts scaled to L_ref (optional if you already have theta)
alpha = estimate_alpha_from_counts(counts_control, lib_sizes_control, L_ref=1e4, clip=(1e-4, 1e3))
theta = 1.0 / np.clip(alpha, 1e-8, None)

# 2) Initialize renoiser with empirical library sizes per dataset
rn = ReNoiser(L_ref=1e4, theta=theta, lib_sizes_by_dataset={"dsA": lib_sizes_A, "dsB": lib_sizes_B})

# 3) Sample counts for a batch of predictions
C = rn.sample_counts(xbar_pred, dataset_ids=batch_dataset_ids, mode="empirical")

# 4) Optionally write an AnnData for evaluation
adata = write_anndata(C, var_names=genes, obs_df=obs_df)

"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad


def estimate_alpha_from_counts(
    counts: np.ndarray,
    lib_sizes: np.ndarray,
    L_ref: float = 1e4,
    clip: Tuple[float, float] = (1e-5, 1e5),
    robust: bool = True,
) -> np.ndarray:
    """
    Estimate per-gene overdispersion alpha in Var = mu + alpha * mu^2 from **control** counts.

    Parameters
    ----------
    counts : (N, G) array of integer counts (controls)
    lib_sizes : (N,) array of total UMI per cell for the same rows
    L_ref : reference library size used for normalized means
    clip : clamp alpha to a sensible range to avoid extreme values
    robust : if True, use median-of-ratios style robust moments

    Returns
    -------
    alpha : (G,) np.ndarray with non-negative values
    """
    counts = np.asarray(counts, dtype=np.float64)
    lib_sizes = np.asarray(lib_sizes, dtype=np.float64)
    eps = 1e-8
    # scale counts to the reference library
    scale = (L_ref / np.maximum(lib_sizes, eps))[:, None]
    X = counts * scale  # (N, G) "normalized" counts at L_ref

    if robust:
        mu = np.median(X, axis=0)
        var = np.median((X - mu[None, :]) ** 2, axis=0) * 1.4826**2  # approx robust variance
    else:
        mu = X.mean(axis=0)
        var = X.var(axis=0, ddof=1)

    # alpha = max( (var - mu) / mu^2 , 0 )
    alpha = (var - mu) / np.maximum(mu * mu, eps)
    alpha = np.maximum(alpha, 0.0)
    alpha = np.clip(alpha, clip[0], clip[1])
    return alpha.astype(np.float32, copy=False)


class ReNoiser:
    """
    Convert normalized predictions back to counts with a Negative Binomial model.
    You can provide per-gene theta (1/alpha), or let the sampler fall back to Poisson.

    Parameters
    ----------
    L_ref : float
        Reference library size at which xbar means are expressed (e.g., 1e4).
    theta : Optional[np.ndarray], shape (G,)
        Per-gene NB dispersion theta (r in NB). If None, Poisson fallback is used.
    lib_sizes_by_dataset : Optional[Dict[str, np.ndarray]]
        Empirical library size arrays per dataset for sampling.
    """

    def __init__(
        self,
        L_ref: float = 1e4,
        theta: Optional[np.ndarray] = None,
        lib_sizes_by_dataset: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.L_ref = float(L_ref)
        self.theta = None if theta is None else np.asarray(theta, dtype=np.float32)
        self.lib_sizes_by_dataset = lib_sizes_by_dataset or {}

    def sample_library_sizes(
        self,
        dataset_ids: Sequence[str],
        mode: str = "empirical",
        fixed_L: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Produce a library size per row in the batch.

        mode:
          - "empirical": resample from the dataset's empirical lib-size array (with replacement).
          - "fixed": use fixed_L for all rows.
          - "match": expects `dataset_ids` already contain desired library sizes cast to str;
                     only useful if you pass raw numeric strings as dataset_ids (niche).

        Returns
        -------
        libs : (B,) float32
        """
        rng = rng or np.random.default_rng(13)
        B = len(dataset_ids)
        libs = np.zeros(B, dtype=np.float32)
        if mode == "fixed":
            if fixed_L is None:
                raise ValueError("fixed mode requires fixed_L.")
            libs.fill(float(fixed_L))
            return libs

        # empirical mode per dataset
        for i, ds in enumerate(dataset_ids):
            arr = self.lib_sizes_by_dataset.get(str(ds), None)
            if arr is None or len(arr) == 0:
                # fallback to reference
                libs[i] = self.L_ref
            else:
                libs[i] = float(rng.choice(arr))
        return libs

    def _nb_sample(self, mu: np.ndarray, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Sample NB counts given mean mu and theta per gene with Gamma-Poisson mixture.
        mu: (B, G) expected counts
        theta: (G,) dispersion
        """
        B, G = mu.shape
        # Expand theta to (B, G)
        th = np.broadcast_to(theta.reshape(1, G), (B, G))
        # Gamma(shape=theta, scale=mu/theta)
        rate = rng.gamma(shape=th, scale=np.maximum(mu, 0.0) / np.maximum(th, 1e-8))
        # Poisson
        return rng.poisson(rate).astype(np.int32)

    def sample_counts(
        self,
        xbar: np.ndarray,
        dataset_ids: Sequence[str],
        mode: str = "empirical",
        fixed_L: Optional[float] = None,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
        chunk_size: int = 5000,
    ) -> np.ndarray:
        """
        Convert normalized predictions xbar (B, G) into integer counts (B, G).

        Parameters
        ----------
        xbar : (B, G) normalized means at L_ref
        dataset_ids : iterable of dataset IDs (strings) used to sample library sizes
        mode : "empirical" | "fixed"
        fixed_L : library size if mode == "fixed"
        deterministic : if True, return expected counts (no NB sampling)
        rng : optional numpy Generator

        Returns
        -------
        counts : (B, G) int32
        """
        rng = rng or np.random.default_rng(13)
        xbar = np.asarray(xbar, dtype=np.float32)
        
        # The deterministic path is less memory-intensive, but we can chunk it too for consistency.
        if deterministic or self.theta is None:
            libs = self.sample_library_sizes(dataset_ids, mode=mode, fixed_L=fixed_L, rng=rng)
            scale = (libs / self.L_ref).reshape(-1, 1)
            mu = np.maximum(xbar * scale, 0.0)
            return sparse.csr_matrix(np.rint(mu).astype(np.int32))

        # ---- REFACTORED CHUNKING LOGIC ----
        # Process everything in chunks to avoid creating any large intermediate matrices.
        B = xbar.shape[0]
        count_chunks = []
        for i in range(0, B, chunk_size):
            end = min(i + chunk_size, B)
            
            # 1. Slice the inputs for the current chunk
            xbar_chunk = xbar[i:end]
            dataset_ids_chunk = dataset_ids[i:end]
            
            # 2. Calculate libs, scale, and mu only for this smaller chunk
            libs_chunk = self.sample_library_sizes(dataset_ids_chunk, mode=mode, fixed_L=fixed_L, rng=rng)
            scale_chunk = (libs_chunk / self.L_ref).reshape(-1, 1)
            mu_chunk = np.maximum(xbar_chunk * scale_chunk, 0.0)
            
            # 3. Sample counts for the chunk (dense)
            counts_chunk_dense = self._nb_sample(mu_chunk, self.theta, rng=rng)
            
            # 4. Convert to sparse immediately and append
            count_chunks.append(sparse.csr_matrix(counts_chunk_dense))
            
        # 5. Vertically stack the sparse chunks to create the final matrix
        return sparse.vstack(count_chunks, format="csr")

    def expected_counts(
        self,
        xbar: np.ndarray,
        dataset_ids: Sequence[str],
        mode: str = "empirical",
        fixed_L: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return expected counts without sampling (float32).
        """
        xbar = np.asarray(xbar, dtype=np.float32)
        libs = self.sample_library_sizes(dataset_ids, mode=mode, fixed_L=fixed_L).astype(np.float32)
        scale = (libs / self.L_ref).reshape(-1, 1)
        return xbar * scale


def write_anndata(
    counts: np.ndarray,
    var_names: Sequence[str],
    obs_df: Optional[pd.DataFrame] = None,
    X_fmt: str = "csr",
) -> ad.AnnData:
    """
    Build an AnnData from a counts matrix and metadata.

    Parameters
    ----------
    counts : (N, G) int array
    var_names : list/array of gene names length G
    obs_df : optional dataframe with N rows; index becomes obs_names
    X_fmt : "csr" or "csc"

    Returns
    -------
    AnnData with integer sparse counts in .X
    """
    counts = np.asarray(counts, dtype=np.int32)
    if X_fmt == "csr":
        X = sparse.csr_matrix(counts, copy=False)
    else:
        X = sparse.csc_matrix(counts, copy=False)

    if obs_df is None:
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(counts.shape[0])])
    else:
        obs = obs_df.copy()
        if "cell_id" in obs.columns:
            obs = obs.set_index("cell_id", drop=True)

    var = pd.DataFrame(index=np.asarray(var_names, dtype=str))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata
