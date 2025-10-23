#!/usr/bin/env python3
"""
causal_pc_ridge_predict.py

PC (constraint-based) causal discovery on CONTROL cells only (ignores perturbation labels),
followed by ridge regressions to estimate edge weights (parent -> child), and prediction of
test perturbations via linear SEM resolvent.
"""
from __future__ import annotations
import argparse
import itertools
import math
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import anndata as ad

from sklearn.covariance import LedoitWolf
from sklearn.linear_model import RidgeCV, LinearRegression
from scipy.stats import norm

def _get_controls(adata: ad.AnnData) -> ad.AnnData:
    if "target_gene" not in adata.obs:
        raise ValueError("Expected obs['target_gene'] to exist.")
    return adata[adata.obs["target_gene"] == "non-targeting"]

def _log1p(X: np.ndarray) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.log1p(X.astype(np.float64))

def _rank_gaussian(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float64)
    n, p = X.shape
    out = np.zeros_like(X)
    eps = 1e-12
    for j in range(p):
        x = X[:, j] + rng.normal(0, 1e-9, size=n)
        ranks = x.argsort().argsort().astype(np.float64) + 1.0
        u = ranks / (n + 1.0)
        u = np.clip(u, eps, 1 - eps)
        out[:, j] = norm.ppf(u)
    return out

def _compute_z_stats_from_controls(adata_ctrl: ad.AnnData, npn: bool, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    X = adata_ctrl.X
    X = _log1p(X)
    if npn:
        X = _rank_gaussian(X, rng)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return mu, sd

def _zscore_apply(adata: ad.AnnData, mu: np.ndarray, sd: np.ndarray, npn: bool, rng: np.random.Generator) -> np.ndarray:
    X = adata.X
    X = _log1p(X)
    if npn:
        X = _rank_gaussian(X, rng)
    Z = (X - mu) / sd
    return Z

def _pseudobulk_deltas(adata: ad.AnnData, mu: np.ndarray, sd: np.ndarray, npn: bool, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    Z_all = _zscore_apply(adata, mu, sd, npn=npn, rng=rng)
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

def _fisher_z_test(r: float, n: int, cond_size: int, alpha: float) -> bool:
    r = np.clip(r, -0.999999, 0.999999)
    if n - cond_size - 3 <= 0:
        return False
    z = 0.5 * np.log((1 + r) / (1 - r)) * math.sqrt(max(1.0, n - cond_size - 3))
    return abs(z) < norm.ppf(1 - alpha / 2.0)

def _partial_corr(i: int, j: int, S: List[int], Z: np.ndarray) -> float:
    cols = [i, j] + list(S)
    Xs = Z[:, cols]
    lw = LedoitWolf().fit(Xs)
    P = lw.precision_
    r = -P[0, 1] / math.sqrt(P[0, 0] * P[1, 1])
    return float(r)

def pc_stable(Z: np.ndarray, alpha: float) -> Tuple[np.ndarray, Dict[Tuple[int, int], Set[int]]]:
    p = Z.shape[1]
    G = np.ones((p, p), dtype=int)
    np.fill_diagonal(G, 0)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    l = 0
    cont = True
    while cont:
        cont = False
        pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if G[i, j] == 1]
        for (i, j) in pairs:
            adj_i = [k for k in range(p) if k != j and k != i and G[i, k] == 1]
            if len(adj_i) < l:
                continue
            for S in itertools.combinations(adj_i, l):
                r = _partial_corr(i, j, list(S), Z)
                if _fisher_z_test(r, Z.shape[0], len(S), alpha):
                    G[i, j] = G[j, i] = 0
                    sep_sets[(i, j)] = set(S)
                    sep_sets[(j, i)] = set(S)
                    cont = True
                    break
        l += 1
    def orient(u: int, v: int):
        G[u, v] = 2
        G[v, u] = 0
    for k in range(p):
        nbrs = [i for i in range(p) if G[i, k] == 1 and G[k, i] == 1]
        for i, j in itertools.combinations(nbrs, 2):
            if G[i, j] == 0 and G[j, i] == 0:
                S = sep_sets.get((i, j), set())
                if k not in S:
                    orient(i, k)
                    orient(j, k)
    changed = True
    while changed:
        changed = False
        for i in range(p):
            for k in range(p):
                if G[i, k] == 2:
                    for j in range(p):
                        if j == i or j == k:
                            continue
                        if (G[k, j] == 1 and G[j, k] == 1) and (G[i, j] == 0 and G[j, i] == 0):
                            orient(k, j)
                            changed = True
    return G, sep_sets

def _has_cycle_after_orient(D: np.ndarray, u: int, v: int) -> bool:
    p = D.shape[0]
    if D[u, v] == 1:
        return False
    visited = [False] * p
    stack = [v]
    while stack:
        x = stack.pop()
        if x == u:
            return True
        for y in range(p):
            if D[x, y] == 1 and not visited[y]:
                visited[y] = True
                stack.append(y)
    return False

def _bic_for_child(y_idx: int, parents: List[int], Z: np.ndarray) -> float:
    n = Z.shape[0]
    if len(parents) == 0:
        y = Z[:, y_idx]
        resid = y
        rss = float((resid ** 2).sum())
        k = 0
    else:
        X = Z[:, parents]
        y = Z[:, y_idx]
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        resid = y - reg.predict(X)
        rss = float((resid ** 2).sum())
        k = len(parents)
    rss = max(rss, 1e-12)
    return n * math.log(rss / n) + k * math.log(max(n, 1))

def cpdag_to_dag_greedy(G_cpd: np.ndarray, Z_ctrl: np.ndarray) -> np.ndarray:
    p = G_cpd.shape[0]
    D = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if G_cpd[i, j] == 2:
                D[i, j] = 1
    undirected = [(i, j) for i in range(p) for j in range(i + 1, p) if G_cpd[i, j] == 1 and G_cpd[j, i] == 1]
    def parents_of(node: int) -> List[int]:
        return [i for i in range(p) if D[i, node] == 1]
    for (i, j) in undirected:
        cyc_ij = _has_cycle_after_orient(D, i, j)
        cyc_ji = _has_cycle_after_orient(D, j, i)
        score_ij = score_ji = float('inf')
        if not cyc_ij:
            parents_j = parents_of(j) + [i]
            score_ij = _bic_for_child(j, parents_j, Z_ctrl)
        if not cyc_ji:
            parents_i = parents_of(i) + [j]
            score_ji = _bic_for_child(i, parents_i, Z_ctrl)
        if score_ij < score_ji and not cyc_ij:
            D[i, j] = 1
        elif score_ji < score_ij and not cyc_ji:
            D[j, i] = 1
    return D

def fit_ridge_B(D: np.ndarray, Z_ctrl: np.ndarray, alphas: List[float]) -> Tuple[np.ndarray, Dict[int, float]]:
    p = D.shape[0]
    B = np.zeros((p, p), dtype=float)
    alpha_map: Dict[int, float] = {}
    for j in range(p):
        parents = [i for i in range(p) if D[i, j] == 1]
        if len(parents) == 0:
            alpha_map[j] = 0.0
            continue
        X = Z_ctrl[:, parents]
        y = Z_ctrl[:, j]
        rc = RidgeCV(alphas=alphas, fit_intercept=False, cv=None, scoring=None)
        rc.fit(X, y)
        coef = rc.coef_
        B[parents, j] = coef
        alpha_map[j] = float(rc.alpha_)
    return B, alpha_map

def _estimate_knockdown_size_from_train_self(train: ad.AnnData, mu: np.ndarray, sd: np.ndarray, npn: bool, rng: np.random.Generator, genes: List[str]) -> float:
    deltas = _pseudobulk_deltas(train, mu, sd, npn=npn, rng=rng)
    idx = {g: i for i, g in enumerate(genes)}
    vals = []
    for tg, dv in deltas.items():
        if tg in idx:
            vals.append(abs(dv[idx[tg]]))
    if len(vals) == 0:
        return 0.5
    return float(np.mean(vals))

def predict_test_from_B(train: ad.AnnData, test: ad.AnnData, genes: List[str], B: np.ndarray, kd_z: float, mu_test: np.ndarray, sd_test: np.ndarray) -> ad.AnnData:
    ctrl_test = _get_controls(test)
    pred_list: List[ad.AnnData] = [ctrl_test.copy()]
    p = len(genes)
    I = np.eye(p)
    try:
        resolvent = np.linalg.inv(I - B)
    except np.linalg.LinAlgError:
        resolvent = np.linalg.inv(I - B + 1e-6 * I)
    rng = np.random.default_rng(1337)
    Xc = _log1p(ctrl_test.X)
    Zc = (Xc - mu_test) / sd_test
    for tg in sorted(set(test.obs["target_gene"]) - {"non-targeting"}):
        mask = (test.obs["target_gene"].values == tg)
        n = int(mask.sum())
        if n == 0:
            continue
        s = np.zeros(p, dtype=float)
        try:
            t_idx = genes.index(tg)
        except ValueError:
            continue
        s[t_idx] = -kd_z
        delta = resolvent @ s
        idxs = rng.integers(low=0, high=Zc.shape[0], size=n)
        Z_base = Zc[idxs].copy()
        Z_pred = Z_base + delta[None, :]
        X_pred = (Z_pred * sd_test[None, :]) + mu_test[None, :]
        X_pred[X_pred < 0] = 0.0
        A = ad.AnnData(X=X_pred.astype(np.float32))
        A.var.index = pd.Index(genes, name="gene")
        A.obs["target_gene"] = tg
        A.obs_names = pd.Index([f"{tg}_pred_{i}" for i in range(n)])
        pred_list.append(A)
    pred_test = ad.concat(pred_list, axis=0, join="outer")
    pred_test = pred_test[:, genes].copy()
    return pred_test

def main():
    parser = argparse.ArgumentParser(description="PC (controls) + ridge SEM predictor for toy 5-gene data.")
    parser.add_argument("--train_h5ad", required=True)
    parser.add_argument("--test_h5ad", required=True)
    parser.add_argument("--out_h5ad", required=True)
    parser.add_argument("--alpha", type=float, default=0.05, help="PC CI-test significance level.")
    parser.add_argument("--npn", action="store_true", help="Apply rank-Gaussian (nonparanormal) before z-scoring.")
    parser.add_argument("--ridge_alphas", type=str, default="0.01,0.1,1.0,10.0", help="Comma-separated ridge alphas grid.")
    args = parser.parse_args()
    train = ad.read_h5ad(args.train_h5ad)
    test = ad.read_h5ad(args.test_h5ad)
    genes = list(test.var_names.astype(str))
    train = train[:, genes].copy()
    rng = np.random.default_rng(1234)
    ctrl_train = _get_controls(train)
    mu_tr, sd_tr = _compute_z_stats_from_controls(ctrl_train, npn=args.npn, rng=rng)
    Z_ctrl = _zscore_apply(ctrl_train, mu_tr, sd_tr, npn=args.npn, rng=rng)
    G_cpd, sep_sets = pc_stable(Z_ctrl, alpha=args.alpha)
    D = cpdag_to_dag_greedy(G_cpd, Z_ctrl)
    alphas = [float(x) for x in args.ridge_alphas.split(",")]
    B, alpha_map = fit_ridge_B(D, Z_ctrl, alphas=alphas)
    kd_z = _estimate_knockdown_size_from_train_self(train, mu_tr, sd_tr, npn=args.npn, rng=rng, genes=genes)
    ctrl_test = _get_controls(test)
    mu_te, sd_te = _compute_z_stats_from_controls(ctrl_test, npn=False, rng=rng)
    pred = predict_test_from_B(train, test, genes, B, kd_z, mu_te, sd_te)
    def edgelist_from_adj(A: np.ndarray, mode: str="cpdag") -> List[Tuple[str, str, str]]:
        out = []
        p = A.shape[0]
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if mode == "cpdag":
                    if A[i, j] == 2:
                        out.append((genes[i], "->", genes[j]))
                    elif A[i, j] == 1 and A[j, i] == 1 and i < j:
                        out.append((genes[i], "-", genes[j]))
                elif mode == "dag":
                    if A[i, j] == 1:
                        out.append((genes[i], "->", genes[j]))
        return out
    pred.uns["pc_ridge_log"] = {
        "genes": genes,
        "alpha": float(args.alpha),
        "npn": bool(args.npn),
        "ridge_alphas": [float(x) for x in args.ridge_alphas.split(",")],
        "kd_z": float(kd_z),
        "cpdag_edges": edgelist_from_adj(G_cpd, mode="cpdag"),
        "dag_edges": edgelist_from_adj(D, mode="dag"),
        "B_weights": B.tolist(),
        "chosen_ridge_alpha_per_child": {genes[j]: alpha_map[j] for j in range(len(genes))},
    }
    print("=== PC (controls) → CPDAG edges ===")
    print(pred.uns["pc_ridge_log"]["cpdag_edges"])
    print("=== Greedy DAG edges ===")
    print(pred.uns["pc_ridge_log"]["dag_edges"])
    print("=== B (parent->child) ===")
    print(np.array(pred.uns["pc_ridge_log"]["B_weights"]))
    print(f"alpha={args.alpha}, npn={args.npn}, kd_z={pred.uns['pc_ridge_log']['kd_z']}")
    print("ridge chosen alphas:", pred.uns["pc_ridge_log"]["chosen_ridge_alpha_per_child"])
    import os
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    pred.write(args.out_h5ad)
    print(f"Wrote predicted test AnnData to: {args.out_h5ad}")
if __name__ == "__main__":
    main()
