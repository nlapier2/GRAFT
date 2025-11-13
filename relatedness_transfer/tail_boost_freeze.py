#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tail-boost with frozen top-K per tail (one-list).
"""
import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from typing import Tuple, List

try:
    from multi_dataset_krr import evaluate_model  # type: ignore
    _EVAL_AVAILABLE = True
except Exception:
    _EVAL_AVAILABLE = False

def compute_control_mean(X: np.ndarray, labels: pd.Series, control_label: str) -> np.ndarray:
    m = labels.astype(str) == str(control_label)
    if not np.any(m):
        raise ValueError(f"No control rows found for control_label='{control_label}'.")
    return X[m].mean(axis=0)

def deltas_from_pseudobulk(adata: ad.AnnData, target_label: str, control_label: str
                           ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    labels = adata.obs[target_label].astype(str)
    X = adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)
    ctrl = compute_control_mean(X, labels, control_label)
    mask = labels != str(control_label)
    perts = list(labels[mask])
    Xp = X[mask]
    deltas = Xp - ctrl[None, :]
    return deltas, perts, ctrl

def intersect_genes(pred: ad.AnnData, true: ad.AnnData):
    common = pred.var_names.intersection(true.var_names)
    if common.size == 0:
        raise ValueError("No overlapping genes between predicted and true AnnData.")
    if common.size < pred.n_vars or common.size < true.n_vars:
        print(f"[info] Restricting to {common.size} common genes.")
    pred2 = pred[:, common].copy()
    true2 = true[:, common].copy()
    return pred2, true2

def topk_indices_desc(x: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, x.size)
    if k_eff <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-x, k_eff-1)[:k_eff]
    idx = idx[np.argsort(-x[idx], kind="mergesort")]
    return idx

def bottomk_indices_asc(x: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, x.size)
    if k_eff <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(x, k_eff-1)[:k_eff]
    idx = idx[np.argsort(x[idx], kind="mergesort")]
    return idx

def apply_tail_boost_freeze(pred_delta: np.ndarray, true_delta: np.ndarray, K: int, incorrect_pct: float = 0.0, rng: "np.random.Generator | None" = None) -> np.ndarray:
    G = pred_delta.size
    if K <= 0 or K*2 >= G:
        return pred_delta.copy()
    pos_frozen = topk_indices_desc(pred_delta, K)
    neg_frozen = bottomk_indices_asc(pred_delta, K)
    middle_mask = np.ones(G, dtype=bool)
    middle_mask[pos_frozen] = False
    middle_mask[neg_frozen] = False
    true_pos_order = np.argsort(-true_delta, kind="mergesort")
    true_neg_order = np.argsort(true_delta, kind="mergesort")
    pos_cand = [i for i in true_pos_order if middle_mask[i]]
    neg_cand = [i for i in true_neg_order if middle_mask[i]]
    pos_boost = np.array(pos_cand[:K], dtype=int)
    neg_boost = np.array(neg_cand[:K], dtype=int)

    # --- Optionally corrupt a percentage of chosen boosts with random middle genes ---
    if rng is None:
        rng = np.random.default_rng()
    def _corrupt(boost_idx: np.ndarray, frozen_idx: np.ndarray) -> np.ndarray:
        if incorrect_pct <= 0.0 or boost_idx.size == 0:
            return boost_idx
        # number to replace
        n_bad = int(round((incorrect_pct / 100.0) * boost_idx.size))
        if n_bad <= 0:
            return boost_idx
        # middle pool = not frozen, not currently chosen
        in_boost = np.zeros(G, dtype=bool); in_boost[boost_idx] = True
        in_frozen = np.zeros(G, dtype=bool); in_frozen[frozen_idx] = True
        pool = np.where(middle_mask & (~in_boost) & (~in_frozen))[0]
        if pool.size == 0:
            return boost_idx
        # choose which chosen positions to corrupt, and replacement indices from pool
        bad_slots = rng.choice(boost_idx.size, size=min(n_bad, boost_idx.size), replace=False)
        repl = rng.choice(pool, size=bad_slots.size, replace=False if pool.size >= bad_slots.size else True)
        out = boost_idx.copy()
        out[bad_slots] = repl[:bad_slots.size]
        return out

    pos_boost = _corrupt(pos_boost, pos_frozen)
    neg_boost = _corrupt(neg_boost, neg_frozen)

    pos_frozen_mean = float(np.mean(pred_delta[pos_frozen])) if pos_frozen.size > 0 else 0.0
    neg_frozen_mean = float(np.mean(pred_delta[neg_frozen])) if neg_frozen.size > 0 else 0.0
    out = pred_delta.copy()
    if pos_boost.size > 0:
        out[pos_boost] = pos_frozen_mean
    if neg_boost.size > 0:
        out[neg_boost] = neg_frozen_mean
    return out

def main():
    ap = argparse.ArgumentParser(description="Tail-boost with frozen top-K per tail (one-list)." )
    ap.add_argument("--pred_h5ad", required=True)
    ap.add_argument("--true_h5ad", required=True)
    ap.add_argument("--target_label", required=True)
    ap.add_argument("--control_label", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=200, help="Top-K size for each tail to freeze/boost.")
    ap.add_argument("--evaluate", type=int, default=1, help="If 1 and evaluate_model is importable, compute metrics.")
    ap.add_argument("--incorrect_pct", type=float, default=0.0,
                    help="Percentage of boosted genes to choose randomly (per tail) instead of by true deltas.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for incorrect selection.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred = ad.read_h5ad(args.pred_h5ad)
    true = ad.read_h5ad(args.true_h5ad)
    pred, true = intersect_genes(pred, true)

    pred_delta, pred_perts, ctrl = deltas_from_pseudobulk(pred, args.target_label, args.control_label)
    true_delta, true_perts, _ = deltas_from_pseudobulk(true, args.target_label, args.control_label)
    perts_common = sorted(set(pred_perts).intersection(true_perts))
    if not perts_common:
        raise ValueError("No overlapping perturbations between predicted and true.")
    def reindex(rows, names, keep):
        idx = pd.Index(names).get_indexer(keep)
        return rows[idx, :]
    pred_delta = reindex(pred_delta, pred_perts, perts_common)
    true_delta = reindex(true_delta, true_perts, perts_common)

    P, G = pred_delta.shape
    print(f"[info] Applying tail-boost (K={args.K}) on {P} perts, {G} genes.")
    pred_delta_boosted = np.empty_like(pred_delta)
    rng = np.random.default_rng(args.seed)
    for i in range(P):
        pred_delta_boosted[i, :] = apply_tail_boost_freeze(pred_delta[i, :], true_delta[i, :], args.K, incorrect_pct=args.incorrect_pct, rng=rng)
    pred_expr_boosted = pred_delta_boosted + ctrl[None, :]
    out_h5ad = os.path.join(args.out_dir, "tail_boost_freeze_pseudobulk.h5ad")
    ad.AnnData(pred_expr_boosted, obs=pd.DataFrame({args.target_label: perts_common}), var=pred.var).write(out_h5ad)
    print("[done] Wrote:", out_h5ad)
    if args.evaluate and _EVAL_AVAILABLE:
        class _Args: pass
        ev_args = _Args()
        ev_args.target_label = args.target_label
        ev_args.control_label = args.control_label
        metrics = evaluate_model(adata=true, args=ev_args,
                                 pred_bundle=(pred_expr_boosted, true_delta + ctrl[None, :], perts_common, ctrl))
        pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(args.out_dir, "metrics_tail_boost_freeze.csv"), index=False)
        # print("[metrics]", metrics)
    elif args.evaluate and not _EVAL_AVAILABLE:
        print("[warn] evaluate_model not found; skipping metrics.")

if __name__ == "__main__":
    main()
