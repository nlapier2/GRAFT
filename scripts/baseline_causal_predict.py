#!/usr/bin/env python3
"""
causal_skeleton_predict.py

A tiny baseline for causal discovery + inference tailored to the 5-gene SEM toy:
A->D, B->D, D->E, C->E. It uses train/test AnnData (from the provided simulator)
and produces a predicted test AnnData with the same number of cells per
perturbation as the original test, suitable for your evaluation pipeline.

Pipeline
--------
1) From TRAIN:
   - Extract control cells ("non-targeting").
   - Z-score all genes using control mean/std; keep these stats for TEST.
   - Learn an undirected skeleton on controls via GraphicalLassoCV.
   - Orient edges using train interventions: for each observed perturbation P,
     compute pseudobulk delta (mean_z(pert P) - mean_z(controls)).
     If delta[Y] is large when P==X (and vice versa), orient X->Y.
     Break ties by absolute delta magnitude; leave unoriented if ambiguous.
   - Fit linear SEM (OLS) for each node on its parents using CONTROL cells only,
     yielding a weighted adjacency matrix B (parent->child).

2) Knockdown size (z units):
   - Estimate a single global knockdown size from TRAIN interventions as the
     mean self-delta magnitude across available perts.

3) Predict TEST:
   - Compute per-target total z-shift Δ = (I - B)^{-1} * s, where s has s_T = -knockdown_z.
   - For each TEST perturbation condition with n cells:
        - Sample n control cells *from TEST controls* (with replacement) as
          the base z vectors, add Δ, then inverse-z to counts using TEST
          control mean/std. Clip negatives to 0.
   - Copy controls over unchanged.
   - Concatenate and save.

Notes
-----
- Keeps deps minimal: anndata, numpy, pandas, scikit-learn.
- Designed to be robust on the toy setting. No guarantee for general datasets.
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
try:
    import anndata as ad
except Exception as e:
    raise SystemExit("This script requires 'anndata'. Try: pip install anndata") from e

from sklearn.covariance import GraphicalLassoCV
from sklearn.linear_model import LinearRegression
import scanpy as sc


def _get_controls(adata: ad.AnnData) -> ad.AnnData:
    if "target_gene" not in adata.obs:
        raise ValueError("Expected obs['target_gene'] to exist.")
    return adata[adata.obs["target_gene"] == "non-targeting"]


def _zscore_from_controls(adata: ad.AnnData, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    X = adata.X.astype(np.float64)
    if hasattr(X, "toarray"):
        X = X.toarray()
    Z = (X - mu) / sd
    return Z


def _compute_control_stats(adata_ctrl: ad.AnnData) -> Tuple[np.ndarray, np.ndarray]:
    X = adata_ctrl.X.astype(np.float64)
    if hasattr(X, "toarray"):
        X = X.toarray()
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return mu, sd


def _learn_skeleton_glasso(Z_ctrl: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    model = GraphicalLassoCV()
    model.fit(Z_ctrl)
    precision = model.precision_
    A = (np.abs(precision) > tol).astype(int)
    np.fill_diagonal(A, 0)
    A = ((A + A.T) > 0).astype(int)
    return A


def _pseudobulk_deltas(adata: ad.AnnData, mu_ctrl: np.ndarray, sd_ctrl: np.ndarray) -> Dict[str, np.ndarray]:
    Z_all = _zscore_from_controls(adata, mu_ctrl, sd_ctrl)
    is_ctrl = (adata.obs["target_gene"].values == "non-targeting")
    ctrl_mean = Z_all[is_ctrl].mean(axis=0)
    deltas: Dict[str, np.ndarray] = {}
    for tg in np.unique(adata.obs["target_gene"].values):
        if tg == "non-targeting":
            continue
        mask = (adata.obs["target_gene"].values == tg)
        if mask.sum() == 0:
            continue
        mean_tg = Z_all[mask].mean(axis=0)
        deltas[tg] = (mean_tg - ctrl_mean)
    return deltas


def _skeleton_from_deltas(deltas: dict, genes: list[str], thresh: float) -> np.ndarray:
    """Build an undirected skeleton where either direction shows a sizable intervention effect."""
    G = len(genes)
    A = np.zeros((G, G), dtype=int)
    for i in range(G):
        gi = genes[i]
        di = deltas.get(gi, None)
        if di is None:
            continue
        for j in range(G):
            if i == j:
                continue
            if abs(di[j]) > thresh:
                A[i, j] = 1
    # symmetrize to undirected
    A = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


def _skeleton_from_comovement(
    deltas: dict, genes: list[str], corr_thresh: float = 0.6, min_perts: int = 2, min_std: float = 1e-4
) -> np.ndarray:
    """
    Add undirected edges for pairs (i,j) whose z-mean shifts co-move across train interventions.
    For each pair, collect vectors [Δ_i(P)]_P and [Δ_j(P)]_P over available perts P and add edge if:
      - number of perts >= min_perts, and
      - std of both vectors >= min_std, and
      - |corr(Δ_i, Δ_j)| >= corr_thresh.
    """
    G = len(genes)
    # Build a matrix M of shape (num_perts, G): row = delta vector for that pert
    perts = [p for p in deltas.keys() if p in genes]
    if len(perts) == 0:
        return np.zeros((G, G), dtype=int)
    M = np.vstack([deltas[p] for p in perts])  # rows: perts, cols: genes
    A = np.zeros((G, G), dtype=int)
    for i in range(G):
        xi = M[:, i]
        if np.std(xi) < min_std:
            continue
        for j in range(i + 1, G):
            xj = M[:, j]
            if np.std(xj) < min_std:
                continue
            if len(perts) < min_perts:
                continue
            r = np.corrcoef(xi, xj)[0, 1]
            if np.isfinite(r) and abs(r) >= corr_thresh:
                A[i, j] = A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A


def _orient_edges_from_interventions(
    undirected_adj: np.ndarray,
    genes: List[str],
    deltas: Dict[str, np.ndarray],
    thresh: float = 0.25,
) -> np.ndarray:
    G = len(genes)
    idx = {g:i for i,g in enumerate(genes)}
    Bmask = np.zeros((G,G), dtype=int)

    for i in range(G):
        for j in range(i+1, G):
            if undirected_adj[i,j] == 0:
                continue
            gi, gj = genes[i], genes[j]
            d_i_on_j = np.abs(deltas.get(gi, np.zeros(G)))[j]
            d_j_on_i = np.abs(deltas.get(gj, np.zeros(G)))[i]

            vote_i_to_j = d_i_on_j > thresh
            vote_j_to_i = d_j_on_i > thresh

            if vote_i_to_j and not vote_j_to_i:
                Bmask[i,j] = 1
            elif vote_j_to_i and not vote_i_to_j:
                Bmask[j,i] = 1
            elif vote_i_to_j and vote_j_to_i:
                if d_i_on_j > d_j_on_i:
                    Bmask[i,j] = 1
                elif d_j_on_i > d_i_on_j:
                    Bmask[j,i] = 1
                # else leave undecided

    return Bmask


def _fit_linear_sem_controls(Z_ctrl: np.ndarray, Bmask: np.ndarray) -> np.ndarray:
    G = Z_ctrl.shape[1]
    B = np.zeros((G,G), dtype=float)
    for child in range(G):
        parents = np.where(Bmask[:,child] == 1)[0]
        if len(parents) == 0:
            continue
        X = Z_ctrl[:, parents]
        y = Z_ctrl[:, child]
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        B[parents, child] = reg.coef_
    return B


def _estimate_knockdown_size_from_train_self(deltas: Dict[str, np.ndarray], genes: List[str]) -> float:
    vals = []
    idx = {g:i for i,g in enumerate(genes)}
    for tg, dv in deltas.items():
        if tg in idx:
            vals.append(abs(dv[idx[tg]]))
    if len(vals) == 0:
        return 0.5
    return float(np.mean(vals))


def _fallback_orient_by_regression(
    undirected_adj: np.ndarray,
    genes: List[str],
    deltas: Dict[str, np.ndarray],
    Bmask: np.ndarray,
    coef_thresh: float = 0.05
) -> np.ndarray:
    G = len(genes)
    # Build matrix M: rows = perts in deltas, cols = genes
    perts = [p for p in deltas.keys() if p in genes]
    if len(perts) == 0:
        return Bmask
    M = np.vstack([deltas[p] for p in perts])  # shape (P, G)
    name_to_idx = {g:i for i,g in enumerate(genes)}

    # Helper to get neighbors from skeleton (undirected)
    def neighbors(k: int) -> List[int]:
        return [j for j in range(G) if j != k and undirected_adj[k, j] == 1]

    for i in range(G):
        for j in range(i+1, G):
            if undirected_adj[i, j] == 0:
                continue
            # Skip if already oriented
            if Bmask[i, j] == 1 or Bmask[j, i] == 1:
                continue
            # Build two regressions on perts x genes deltas
            # V ~ U + N(V\U)
            V = M[:, j]
            U = M[:, i]
            Nv = [n for n in neighbors(j) if n != i]
            X1 = U[:, None]
            if len(Nv) > 0:
                X1 = np.hstack([X1, M[:, Nv]])
            try:
                reg1 = LinearRegression(fit_intercept=False).fit(X1, V)
                c_uv = abs(reg1.coef_[0])
            except Exception:
                c_uv = 0.0
            # U ~ V + N(U\V)
            Nu = [n for n in neighbors(i) if n != j]
            X2 = V[:, None]
            if len(Nu) > 0:
                X2 = np.hstack([X2, M[:, Nu]])
            try:
                reg2 = LinearRegression(fit_intercept=False).fit(X2, U)
                c_vu = abs(reg2.coef_[0])
            except Exception:
                c_vu = 0.0
            # Decide
            if c_uv > c_vu and c_uv >= coef_thresh:
                Bmask[i, j] = 1
            elif c_vu > c_uv and c_vu >= coef_thresh:
                Bmask[j, i] = 1
            # else remain undecided
    return Bmask


def _predict_test(
    train: ad.AnnData,
    test: ad.AnnData,
    genes: List[str] = None,
    thresh: float = 0.25,
    corr_thresh: float = 0.6,
) -> ad.AnnData:
    if genes is None:
        genes = list(test.var_names.astype(str))

    ctrl_train = _get_controls(train)
    mu_train, sd_train = _compute_control_stats(ctrl_train)
    Z_ctrl_train = _zscore_from_controls(ctrl_train, mu_train, sd_train)

    A = _learn_skeleton_glasso(Z_ctrl_train)
    deltas_train = _pseudobulk_deltas(train, mu_train, sd_train)
    # Build delta-based and co-movement skeletons, then union with Glasso
    A_from_delta = _skeleton_from_deltas(deltas_train, genes, thresh=thresh)
    A_comove = _skeleton_from_comovement(deltas_train, genes, corr_thresh=corr_thresh)
    if A.sum() == 0:
        A = ((A_from_delta + A_comove) > 0).astype(int)
    else:
        A = ((A + A_from_delta + A_comove) > 0).astype(int)
    Bmask = _orient_edges_from_interventions(A, genes, deltas_train, thresh=thresh)
    Bmask = _fallback_orient_by_regression(A, genes, deltas_train, Bmask, coef_thresh=0.05)
    B = _fit_linear_sem_controls(Z_ctrl_train, Bmask)
    kd_z = _estimate_knockdown_size_from_train_self(deltas_train, genes)

    ctrl_test = _get_controls(test)
    mu_test, sd_test = _compute_control_stats(ctrl_test)

    G = len(genes)
    I = np.eye(G)
    try:
        resolvent = np.linalg.inv(I - B)
    except np.linalg.LinAlgError:
        resolvent = np.linalg.inv(I - B + 1e-6 * I)

    # --- Log learned structure/weights (genes, skeleton, orientations, B, kd_z) ---
    G = len(genes)
    genes_arr = np.array(genes)
    skel_edges = [(genes_arr[i], genes_arr[j]) for i in range(G) for j in range(i+1, G) if A[i, j] == 1]
    comove_edges = [(genes[i], genes[j]) for i in range(G) for j in range(i+1, G) if A_comove[i, j] == 1]
    oriented = [(genes_arr[i], genes_arr[j]) for i in range(G) for j in range(G) if Bmask[i, j] == 1]
    learned_log = {
        "genes": list(genes_arr),
        "skeleton_edges": skel_edges,          # undirected pairs from Graphical Lasso
        "comovement_edges": comove_edges,      # undirected pairs from co-movement test
        "oriented_edges": oriented,            # (parent, child)
        "B_weights": B.tolist(),               # parent->child matrix
        "kd_z": float(kd_z),
        "thresh": float(thresh),
        "corr_thresh": float(corr_thresh),
        "fallback_coef_thresh": 0.05,
    }

    pred_adatas: List[ad.AnnData] = []
    pred_adatas.append(ctrl_test.copy())

    Z_ctrl_test = _zscore_from_controls(ctrl_test, mu_test, sd_test)
    rng = np.random.default_rng(1337)

    for tg in sorted(set(test.obs["target_gene"]) - {"non-targeting"}):
        mask = (test.obs["target_gene"].values == tg)
        n = int(mask.sum())
        if n == 0:
            continue

        s = np.zeros(G, dtype=float)
        try:
            t_idx = list(genes).index(tg)
        except ValueError:
            continue
        s[t_idx] = -kd_z
        delta = resolvent @ s

        idxs = rng.integers(low=0, high=Z_ctrl_test.shape[0], size=n)
        Z_base = Z_ctrl_test[idxs].copy()
        Z_pred = Z_base + delta[None, :]

        X_pred = (Z_pred * sd_test[None, :]) + mu_test[None, :]
        X_pred[X_pred < 0] = 0.0

        adata_pred = ad.AnnData(X=X_pred.astype(np.float32))
        adata_pred.var.index = test.var_names.copy()
        adata_pred.obs["target_gene"] = tg
        adata_pred.obs_names = pd.Index([f"{tg}_pred_{i}" for i in range(n)])
        pred_adatas.append(adata_pred)

    pred_test = ad.concat(pred_adatas, axis=0, join="outer")
    pred_test = pred_test[:, test.var_names].copy()
    pred_test.uns["learned_graph"] = learned_log
    return pred_test


def main():
    import anndata as ad
    import os
    parser = argparse.ArgumentParser(description="Simple skeleton-based causal predictor for toy SEM.")
    parser.add_argument("--train_h5ad", required=True, help="Path to training AnnData (.h5ad).")
    parser.add_argument("--test_h5ad", required=True, help="Path to test AnnData (.h5ad).")
    parser.add_argument("--out_h5ad", required=True, help="Where to write predicted test AnnData (.h5ad).")
    parser.add_argument("--thresh", type=float, default=0.25, help="Z-delta threshold to orient edges.")
    parser.add_argument("--corr_thresh", type=float, default=0.6, help="Absolute correlation threshold for co-movement skeleton.")
    args = parser.parse_args()

    train = ad.read_h5ad(args.train_h5ad)
    test = ad.read_h5ad(args.test_h5ad)
    # sc.pp.normalize_total(train, target_sum=1)
    # sc.pp.normalize_total(test, target_sum=1)
    # sc.pp.log1p(train)
    # sc.pp.log1p(test)

    if not np.array_equal(train.var_names.values, test.var_names.values):
        genes_common = [g for g in test.var_names if g in set(train.var_names)]
        train = train[:, genes_common].copy()
        test = test[:, genes_common].copy()

    pred = _predict_test(train, test, genes=list(test.var_names.astype(str)), thresh=args.thresh, corr_thresh=args.corr_thresh)
    # Print a compact log of the learned graph/weights
    lg = pred.uns.get("learned_graph", {})
    if lg:
        print("=== Learned skeleton edges (undirected) ===")
        print(lg.get("skeleton_edges", []))
        print("=== Oriented edges (parent -> child) ===")
        print(lg.get("oriented_edges", []))
        print("=== B (parent->child) weights ===")
        print(np.array(lg.get("B_weights", [])))
        print(f"Estimated knockdown (z): {lg.get('kd_z')}  |  orientation threshold: {lg.get('thresh')}  |  co-move corr thresh: {lg.get('corr_thresh')}")

    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    pred.write(args.out_h5ad)
    print(f"Wrote predicted test AnnData to: {args.out_h5ad}")


if __name__ == "__main__":
    main()
