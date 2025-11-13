#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tail-boost pipeline with clean train/test separation (selector currently random).
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
    idx = idx[np.argsort(x[idx], kind="mergesort")][::1]
    return idx

class RandomTailSelector:
    def __init__(self, K: int, rng: np.random.Generator):
        self.K = K
        self.rng = rng
    def fit(self, pred_delta_train: np.ndarray):
        return self
    def select_middle_indices(self, pred_delta_vec: np.ndarray,
                              pos_frozen: np.ndarray, neg_frozen: np.ndarray):
        G = pred_delta_vec.size
        middle_mask = np.ones(G, dtype=bool)
        middle_mask[pos_frozen] = False
        middle_mask[neg_frozen] = False
        middle = np.where(middle_mask)[0]
        k_eff = min(self.K, middle.size)
        if k_eff <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        pos_boost = self.rng.choice(middle, size=k_eff, replace=(middle.size < k_eff))
        neg_boost = self.rng.choice(middle, size=k_eff, replace=(middle.size < k_eff))
        return np.array(pos_boost, dtype=int), np.array(neg_boost, dtype=int)

def apply_boost_once(pred_delta: np.ndarray,
                     pos_frozen: np.ndarray, neg_frozen: np.ndarray,
                     pos_boost: np.ndarray, neg_boost: np.ndarray) -> np.ndarray:
    out = pred_delta.copy()
    pos_mean = float(np.mean(pred_delta[pos_frozen])) if pos_frozen.size > 0 else 0.0
    neg_mean = float(np.mean(pred_delta[neg_frozen])) if neg_frozen.size > 0 else 0.0
    if pos_boost.size > 0:
        out[pos_boost] = pos_mean
    if neg_boost.size > 0:
        out[neg_boost] = neg_mean
    return out

def main():
    ap = argparse.ArgumentParser(description="Tail-boost train/test with random selector (no leakage)." )
    ap.add_argument("--pred_h5ad", required=True)
    ap.add_argument("--true_h5ad", required=True)
    ap.add_argument("--target_label", required=True)
    ap.add_argument("--control_label", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=1500, help="Top-K size for each tail to freeze and to boost.")
    ap.add_argument("--test_pct_pert", type=float, default=0.2, help="Fraction of perturbations in TEST (0..1)." )
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--evaluate", type=int, default=1, help="If 1 and evaluate_model is importable, compute TEST metrics.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pred = ad.read_h5ad(args.pred_h5ad)
    true = ad.read_h5ad(args.true_h5ad)
    pred, true = intersect_genes(pred, true)

    pred_delta_all, pred_perts_all, ctrl = deltas_from_pseudobulk(pred, args.target_label, args.control_label)
    true_delta_all, true_perts_all, _ = deltas_from_pseudobulk(true, args.target_label, args.control_label)

    perts_common = np.array(sorted(set(pred_perts_all).intersection(true_perts_all)))
    if perts_common.size == 0:
        raise ValueError("No overlapping perturbations between predicted and true.")
    def reindex(rows, names, keep):
        idx = pd.Index(names).get_indexer(keep)
        return rows[idx, :]
    pred_delta_all = reindex(pred_delta_all, pred_perts_all, perts_common)
    true_delta_all = reindex(true_delta_all, true_perts_all, perts_common)

    P, G = pred_delta_all.shape
    perm = rng.permutation(P)
    n_test = int(round(args.test_pct_pert * P))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    perts_train = list(perts_common[train_idx])
    perts_test = list(perts_common[test_idx])
    print(f"[split] P={P} -> TRAIN={len(train_idx)} TEST={len(test_idx)} (test_pct_pert={args.test_pct_pert})" )

    selector = RandomTailSelector(K=args.K, rng=rng).fit(pred_delta_all[train_idx, :])

    pred_delta_test = pred_delta_all[test_idx, :].copy()
    pred_delta_test_boosted = np.empty_like(pred_delta_test)
    for i in range(pred_delta_test.shape[0]):
        x = pred_delta_test[i, :]
        pos_frozen = topk_indices_desc(x, args.K)
        neg_frozen = bottomk_indices_asc(x, args.K)
        pos_boost, neg_boost = selector.select_middle_indices(x, pos_frozen, neg_frozen)
        pred_delta_test_boosted[i, :] = apply_boost_once(x, pos_frozen, neg_frozen, pos_boost, neg_boost)

    pred_expr_test_boosted = pred_delta_test_boosted + ctrl[None, :]
    out_h5ad = os.path.join(args.out_dir, "tail_boost_TEST_only_pseudobulk.h5ad")
    ad.AnnData(pred_expr_test_boosted, obs=pd.DataFrame({args.target_label: perts_test}), var=pred.var).write(out_h5ad)
    print("[done] Wrote:", out_h5ad)

    if args.evaluate and _EVAL_AVAILABLE:
        true_delta_test = true_delta_all[test_idx, :]
        metrics = evaluate_model(
            adata=true,
            args=type("A", (), {"target_label": args.target_label, "control_label": args.control_label})(),
            pred_bundle=(pred_expr_test_boosted, true_delta_test + ctrl[None, :], perts_test, ctrl)
        )
        pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(args.out_dir, "metrics_TEST_only.csv"), index=False)
        # print("[metrics][TEST]", metrics)
    elif args.evaluate and not _EVAL_AVAILABLE:
        print("[warn] evaluate_model not found; skipping TEST metrics.")

    pd.DataFrame({"pert": perts_train, "split": "train"}).to_csv(os.path.join(args.out_dir, "perts_train.csv"), index=False)
    pd.DataFrame({"pert": perts_test,  "split": "test"}).to_csv(os.path.join(args.out_dir, "perts_test.csv"),  index=False)

if __name__ == "__main__":
    main()
