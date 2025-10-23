#!/usr/bin/env python3
"""
simulate_sem_perturbations.py

Generate simple single-cell perturbation datasets from a tiny linear SEM:

A ─┐
   ├─> D ──> E
B ─┘
C ─────────> E

Edges carry *fractional-effect* coefficients (default 0.5). We model knockdowns
as fractional reductions of targeted genes (default efficiency 0.5 = 50%).
Effects propagate multiplicatively on the *fraction of baseline expression*:

Let fX be the fractional change of gene X relative to its baseline for a cell.
Then
    fD = fD_direct + w_AD * fA + w_BD * fB
    fE = fE_direct + w_DE * fD + w_CE * fC
where fA, fB, fC, fD_direct, fE_direct are set by the perturbation (only the
target has a direct fractional change = -efficiency; others are 0).

New expression for gene X is:
    X' = X_baseline * (1 + fX) + noise_X

Controls are drawn from the *stationary SEM covariance*:
   Sigma = (I - B)^{-1} diag(resid_var) (I - B)^{-T},
then per-cell size factors are added and *Negative Binomial* counts are sampled
(Gamma–Poisson). Perturbations apply fractional shifts to the *mean* before NB sampling.

Outputs
-------
Writes 4 AnnData .h5ad files to the provided output directory:

  - train_hard.h5ad : controls + perts {A,B,C}
  - test_hard.h5ad  : controls + perts {D,E}
  - train_easy.h5ad : controls + perts {A,B,C,E}
  - test_easy.h5ad  : controls + perts {D}

All objects include all control cells. obs['target_gene'] gives the label:
{'non-targeting','A','B','C','D','E'}.

Example
-------
python simulate_sem_perturbations.py --outdir ./sim_out \
  --n_controls 2000 --n_per_pert 500 --efficiency 0.5 \
  --mean 10000 --var 1000

Requires: anndata, numpy, pandas, scipy (for truncated normals).
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import anndata as ad

from numpy.linalg import inv


GENES = ["A","B","C","D","E"]

def _build_B(weights: Dict[Tuple[str,str], float]) -> np.ndarray:
    """Parent->child weighted adjacency B in the order GENES."""
    idx = {g:i for i,g in enumerate(GENES)}
    B = np.zeros((len(GENES), len(GENES)), dtype=float)
    for (u,v), w in weights.items():
        if u in idx and v in idx:
            B[idx[u], idx[v]] = float(w)
    return B

def _stationary_cov(B: np.ndarray, resid_var: float) -> np.ndarray:
    """Sigma = (I-B)^{-1} diag(resid_var) (I-B)^{-T}."""
    G = B.shape[0]
    I = np.eye(G)
    R = np.full(G, float(resid_var), dtype=float)
    A = inv(I - B)
    Sigma = A @ np.diag(R) @ A.T
    return Sigma

def _draw_size_factors(n: int, mu: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """LogNormal size factors (library sizes)."""
    return rng.lognormal(mean=mu, sigma=sigma, size=n)

def _nb_sample(mu: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """
    Negative Binomial via Gamma–Poisson. phi: dispersion (>0).
    Var = mu + phi * mu^2. Implemented as:
      lambda ~ Gamma(shape=1/phi, scale=phi*mu) ;  X ~ Poisson(lambda)
    """
    mu = np.clip(mu, 1e-8, None)
    shape = 1.0 / float(phi)
    scale = float(phi) * mu
    lam = rng.gamma(shape=shape, scale=scale, size=mu.shape)
    X = rng.poisson(lam)
    return X

def _sample_control_means_mvn(n: int, mean: float, Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Draw baseline *means* for controls from MVN(mean*1, Sigma). Clip small negatives.
    Returns (n x G) matrix of mean expression before size factors.
    """
    G = len(GENES)
    mu_vec = np.full(G, float(mean), dtype=float)
    M = rng.multivariate_normal(mean=mu_vec, cov=Sigma, size=n)
    return np.clip(M, 1e-6, None)

def propagate_fractional_changes(
    baseline_means: np.ndarray,
    target: str | None,
    efficiency: float,
    w_AD: float, w_BD: float, w_DE: float, w_CE: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Given baseline expression (n_cells x 5) for (A..E), apply a single perturbation label
    to all rows and return perturbed expression (n_cells x 5).

    Fractional change fX is applied relative to baseline for each cell. Only the targeted
    gene gets a direct fractional change of -efficiency. Effects propagate as:
        fD = fD_direct + w_AD * fA + w_BD * fB
        fE = fE_direct + w_DE * fD + w_CE * fC
    fA, fB, fC are direct only if targeted; otherwise 0.
    """
    n, g = baseline_means.shape
    assert g == 5

    # Direct fractional changes from knockdown:
    fA = -efficiency if target == "A" else 0.0
    fB = -efficiency if target == "B" else 0.0
    fC = -efficiency if target == "C" else 0.0
    fD_direct = -efficiency if target == "D" else 0.0
    fE_direct = -efficiency if target == "E" else 0.0

    # Propagate along the DAG (instantaneous):
    fD = fD_direct + w_AD * fA + w_BD * fB
    fE = fE_direct + w_DE * fD + w_CE * fC

    # All cells share the same fractional changes for a given perturbation condition.
    F = np.tile(np.array([fA, fB, fC, fD, fE], dtype=float), (n,1))
    M = baseline_means * (1.0 + F)              # new *means* after perturbation
    return np.clip(M, 1e-8, None)

def make_adata_from_matrix(X: np.ndarray, target_gene: str) -> 'ad.AnnData':
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var.index = pd.Index(GENES, name="gene")
    adata.obs["target_gene"] = target_gene
    adata.obs_names = pd.Index([f"{target_gene}_{i}" for i in range(adata.n_obs)])  # ensure unique obs names
    return adata

def create_split(
    controls_X: np.ndarray,
    n_per_pert: int,
    efficiency: float,
    weights: Dict[Tuple[str,str], float],
    mean: float,
    var: float,
    rng: np.random.Generator,
    sizefactor_mu: float = 0.0,
    sizefactor_sigma: float = 0.25,
    phi: float = 0.1,
) -> Dict[str, ad.AnnData]:
    """
    Build all condition-specific AnnData objects for each perturbation and combine into
    two pairs of train/test splits (hard and easy).

    Returns dict with keys: 'train_hard','test_hard','train_easy','test_easy'
    """
    w_AD = weights.get(("A","D"), 0.5)
    w_BD = weights.get(("B","D"), 0.5)
    w_DE = weights.get(("D","E"), 0.5)
    w_CE = weights.get(("C","E"), 0.5)

    # Build SEM covariance for controls
    B = _build_B({("A","D"):w_AD, ("B","D"):w_BD, ("D","E"):w_DE, ("C","E"):w_CE})
    Sigma = _stationary_cov(B, resid_var=var)

    # Prepare baseline for each perturbation condition
    perts = ["A","B","C","D","E"]
    adatas: Dict[str, ad.AnnData] = {}

    # Controls (shared across all splits) — resample counts from MVN means + size factors + NB
    ctrl_means = _sample_control_means_mvn(controls_X.shape[0], mean, Sigma, rng)
    s_ctrl = _draw_size_factors(ctrl_means.shape[0], mu=sizefactor_mu, sigma=sizefactor_sigma, rng=rng)
    ctrl_counts = _nb_sample(ctrl_means * s_ctrl[:, None], phi=phi, rng=rng)
    ctrl = make_adata_from_matrix(ctrl_counts, "non-targeting")

    # Per perturbation matrices
    for p in perts:
        base_means = _sample_control_means_mvn(n_per_pert, mean, Sigma, rng)
        Mp = propagate_fractional_changes(
            baseline_means=base_means, target=p, efficiency=efficiency,
            w_AD=w_AD, w_BD=w_BD, w_DE=w_DE, w_CE=w_CE, rng=rng
        )
        s = _draw_size_factors(n_per_pert, mu=sizefactor_mu, sigma=sizefactor_sigma, rng=rng)
        counts = _nb_sample(Mp * s[:, None], phi=phi, rng=rng)
        adatas[p] = make_adata_from_matrix(counts, p)

    # Assemble splits
    # Hard: train on {A,B,C} (+controls); test on {D,E} (+controls)
    train_hard = ad.concat([ctrl, adatas["A"], adatas["B"], adatas["C"]], join="outer", axis=0, label=None)
    test_hard  = ad.concat([ctrl.copy(), adatas["D"], adatas["E"]], join="outer", axis=0, label=None)

    # Easy: train on {A,B,C,E} (+controls); test on {D} (+controls)
    train_easy = ad.concat([ctrl.copy(), adatas["A"], adatas["B"], adatas["C"], adatas["E"]], join="outer", axis=0, label=None)
    test_easy  = ad.concat([ctrl.copy(), adatas["D"]], join="outer", axis=0, label=None)

    # Add small metadata on uns
    for name, A in {"train_hard":train_hard, "test_hard":test_hard, "train_easy":train_easy, "test_easy":test_easy}.items():
        A.uns["sem_graph"] = {
            "edges": {"A->D": w_AD, "B->D": w_BD, "D->E": w_DE, "C->E": w_CE},
            "efficiency": efficiency,
            "note": "Fractional-effect SEM; fD = fD_direct + w_AD*fA + w_BD*fB; fE = fE_direct + w_DE*fD + w_CE*fC",
            "covariance": "stationary SEM Sigma = (I-B)^-1 diag(resid_var) (I-B)^-T",
            "nb_dispersion_phi": float(phi),
            "sizefactor_lognormal": {"mu": float(sizefactor_mu), "sigma": float(sizefactor_sigma)}
        }

    return {
        "train_hard": train_hard,
        "test_hard": test_hard,
        "train_easy": train_easy,
        "test_easy": test_easy
    }

def main():
    p = argparse.ArgumentParser(description="Simulate tiny single-cell perturbation datasets from a 5-gene SEM.")
    p.add_argument("--outdir", required=True, help="Output directory for .h5ad files.")
    p.add_argument("--seed", type=int, default=1337, help="Random seed.")
    p.add_argument("--n_controls", type=int, default=2000, help="Number of control cells.")
    p.add_argument("--n_per_pert", type=int, default=500, help="Number of cells per perturbation condition.")
    p.add_argument("--mean", type=float, default=10000.0, help="Baseline per-gene mean counts for controls.")
    p.add_argument("--var", type=float, default=1000.0, help="Per-gene variance (used for both baseline and additive noise).")
    p.add_argument("--phi", type=float, default=0.1, help="Negative Binomial dispersion phi (>0): Var=mu+phi*mu^2.")
    p.add_argument("--sizefactor_mu", type=float, default=0.0, help="LogNormal mean for size factors (log-space).")
    p.add_argument("--sizefactor_sigma", type=float, default=0.25, help="LogNormal sigma for size factors (log-space).")
    p.add_argument("--efficiency", type=float, default=0.5, help="Knockdown efficiency (fractional), e.g., 0.5 = 50%%.")
    p.add_argument("--w_AD", type=float, default=0.5, help="Fractional-effect weight from A to D.")
    p.add_argument("--w_BD", type=float, default=0.5, help="Fractional-effect weight from B to D.")
    p.add_argument("--w_DE", type=float, default=0.5, help="Fractional-effect weight from D to E.")
    p.add_argument("--w_CE", type=float, default=0.5, help="Fractional-effect weight from C to E.")
    args = p.parse_args()

    if not (0.0 <= args.efficiency <= 1.0):
        raise SystemExit("--efficiency must be in [0,1].")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Placeholder array for control count of rows; values not used after refactor
    controls_X = np.empty((args.n_controls, len(GENES)), dtype=float)

    splits = create_split(
        controls_X=controls_X,
        n_per_pert=args.n_per_pert,
        efficiency=args.efficiency,
        weights={("A","D"):args.w_AD, ("B","D"):args.w_BD, ("D","E"):args.w_DE, ("C","E"):args.w_CE},
        mean=args.mean,
        var=args.var,
        rng=rng,
        sizefactor_mu=args.sizefactor_mu,
        sizefactor_sigma=args.sizefactor_sigma,
        phi=args.phi,
    )

    # Save
    paths = {}
    for name, adata in splits.items():
        path = os.path.join(args.outdir, f"{name}.h5ad")
        adata.write(path)
        paths[name] = path

    print("Wrote files:")
    for k,v in paths.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
