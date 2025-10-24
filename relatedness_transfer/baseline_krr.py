#!/usr/bin/env python3
import argparse, math, os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import pairwise_distances

from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.isotonic import IsotonicRegression

from utils import *
from losses import *


def parse_arguments():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    # Basic and I/O options
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--external_h5ad", required=True, help="Path to the external pseudobulked AnnData object.")
    ap.add_argument("--out_pred_h5ad", type=str, default="",
                    help="If set, write an AnnData with predictions for the evaluation split.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)

    # Train/test split and eval options
    ap.add_argument("--test_pct_perts", type=float, default=0.0,
                    help="Fraction of perturbation labels (excluding control) to hold out for testing. 0.0 = no holdout.")
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional path to a separate test AnnData. If set, overrides --test_pct_perts.")
    ap.add_argument('--eval_on_train', action='store_true', help='Evaluate on training set in addition to test set')
    ap.add_argument('--write_test', action='store_true', help='Write true test set')

    # Method + KRR hyperparams
    ap.add_argument("--method", type=str, default="krr",
                    choices=["krr"], help="Which transfer method to run.")
    ap.add_argument("--krr_lambda", type=float, default=1e-2,
                    help="Ridge regularization λ for KRR on perturbation kernel.")
    ap.add_argument("--kernel_metric", type=str, default="corr",
                    choices=["corr", "cosine"],
                    help="How to build the perturbation kernel from the external dataset.")
    ap.add_argument("--iso_calibrate", action="store_true",
                        help="Apply isotonic calibration of external similarity to match target similarity on training perts.")

    args = ap.parse_args()
    return args

def intersect_datasets(adata_source, adata_target, target_label, control_label):
    """
    Subsets two AnnData objects to their common genes and perturbations.

    Args:
        adata_source: The source (external) AnnData object.
        adata_target: The target AnnData object.
        target_label: The obs column containing perturbation labels.
        control_label: The label for control samples.

    Returns:
        A tuple of (subsetted source AnnData, subsetted target AnnData).
    """
    print("Finding intersection of genes and perturbations...")
    # First, get a list of valid genes from the source (not all NaN), then intersect.
    common_genes = np.intersect1d(
        adata_source.var_names[~np.isnan(to_numpy(adata_source.X)).all(axis=0)],
        adata_target.var_names
    )

    source_perts = set(adata_source.obs[target_label].unique())
    target_perts = set(adata_target.obs[target_label].unique())
    common_perts = sorted(list(source_perts.intersection(target_perts)))

    # Ensure the control label is always kept, even if it's not in the intersection
    if control_label not in common_perts:
        if control_label in source_perts and control_label in target_perts:
            common_perts.append(control_label)
    
    print(f"  Found {len(common_genes)} common genes.")
    print(f"  Found {len(common_perts) - 1} common perturbations (plus control).")

    adata_source_sub = adata_source[adata_source.obs[target_label].isin(common_perts), common_genes].copy()
    adata_target_sub = adata_target[adata_target.obs[target_label].isin(common_perts), common_genes].copy()

    return adata_source_sub, adata_target_sub

def compute_deltas(adata, target_label, control_label):
    """
    Computes the delta (perturbation - control) vectors for a pseudobulked dataset.

    Args:
        adata: A pseudobulked AnnData object.
        target_label: The obs column containing perturbation labels.
        control_label: The label for control samples.

    Returns:
        A dictionary mapping perturbation labels to their delta vectors.
    """
    control_mask = adata.obs[target_label] == control_label
    control_mean = adata[control_mask].X.mean(axis=0)

    pert_adata = adata[~control_mask]
    
    deltas = {
        pert: pert_adata[pert_adata.obs[target_label] == pert].X.flatten() - control_mean
        for pert in pert_adata.obs[target_label].unique()
    }
    return deltas, control_mean

def _row_standardize(M: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm per row (safe for sparse/np arrays)."""
    M = np.asarray(M, dtype=np.float32)
    M = M - M.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    return M / denom

def _fit_isotonic_on_pairs(S_ext_OO: np.ndarray, Y_O: np.ndarray) -> "IsotonicRegression|None":
    """
    Fit isotonic regression mapping external similarity -> target similarity,
    using only training perts (O). Returns a fitted IsotonicRegression or None.
    """
    # target similarity among O (using correlation-style similarity of DELTAS)
    Zt = _row_standardize(Y_O)  # (|O|, G)
    S_tgt_OO = Zt @ Zt.T
    # take off-diagonal upper triangle pairs
    iu, ju = np.triu_indices(S_ext_OO.shape[0], k=1)
    x = S_ext_OO[iu, ju].astype(np.float64)
    y = S_tgt_OO[iu, ju].astype(np.float64)
    # Guard against degenerate cases
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        print("[iso] Degenerate pairwise similarities; skipping isotonic calibration.")
        return None
    iso = IsotonicRegression(y_min=-1.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(x, y)
    return iso

def _apply_isotonic_matrix(iso: "IsotonicRegression|None", S: np.ndarray) -> np.ndarray:
    """Apply fitted isotonic regressor elementwise to a similarity matrix; symmetrize and fix diag."""
    if iso is None:
        return S
    S_flat = S.ravel()
    S_cal = iso.predict(S_flat).reshape(S.shape)
    S_cal = 0.5 * (S_cal + S_cal.T)
    np.fill_diagonal(S_cal, 1.0)
    return S_cal

def _pert_list(adata: ad.AnnData, target_label: str, control_label: str) -> list[str]:
    perts_all = list(map(str, adata.obs[target_label].values))
    # If pseudobulked, each row is a single pert; otherwise fallback to unique order.
    # In either case, we evaluate on NON-control perts only.
    uniq = list(dict.fromkeys(perts_all))  # stable unique
    return [p for p in uniq if p != control_label]

def krr_predict_from_external(
    adata_source: ad.AnnData,
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    krr_lambda: float = 1e-2,
    kernel_metric: str = "corr",
    ctrl_mean_target: np.ndarray | None = None,
    iso_calibrate: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    KRR over perturbation index with kernel built from EXTERNAL deltas (train+eval perts).
    Returns the bundle expected by your evaluator: (pred_mat, true_mat, pert_names, ctrl_mean),
    where **pred_mat and true_mat are EXPRESSION LEVELS**, not deltas.
    ctrl_mean is the TRAIN-split control mean, matching your evaluation protocol.
    """
    G = adata_train.n_vars
    # ---- use TRAIN control mean (for adding back to deltas & evaluator's baseline) ----
    if ctrl_mean_target is None:
        train_mask = np.asarray(adata_train.obs[target_label] == control_label)
        ctrl_mean_target = np.asarray(adata_train.X)[train_mask].mean(axis=0).reshape(-1)

    # ---- define sets ----
    O = _pert_list(adata_train, target_label, control_label)  # observed perts
    U = _pert_list(adata_eval,  target_label, control_label)  # to predict/evaluate
    perts_all = O + [p for p in U if p not in O]
    G = adata_train.n_vars

    # indices for O and U inside perts_all
    idx = {p: i for i, p in enumerate(perts_all)}
    iO = np.array([idx[p] for p in O], dtype=int)
    iU = np.array([idx[p] for p in U], dtype=int)

    # ---- target deltas: Y_O (|O| x G) and Y_true_U (|U| x G) ----
    # build deltas against the SAME ctrl_mean_target
    def _delta_mat(adataX: ad.AnnData, perts: list[str]) -> np.ndarray:
        rows = []
        for p in perts:
            v = adataX[adataX.obs[target_label] == p].X
            v = np.asarray(v).reshape(-1, G).mean(axis=0)  # pseudobulk row for this pert
            rows.append(v - ctrl_mean_target)
        return np.stack(rows, axis=0)

    Y_O = _delta_mat(adata_train, O)   # (|O|, G)


    # ---- external deltas & similarity over ALL perts (O∪U) ----
    del_src, _ = compute_deltas(adata_source, target_label, control_label)  # pert -> delta row
    Delta_src = np.stack([np.asarray(del_src[p]).ravel() for p in perts_all], axis=0)  # (P,G)
    if kernel_metric == "corr":
        Z = _row_standardize(Delta_src)
        S_ext = Z @ Z.T
    else:  # "cosine"
        Z = Delta_src / (np.linalg.norm(Delta_src, axis=1, keepdims=True) + 1e-8)
        S_ext = Z @ Z.T
    S_ext = 0.5 * (S_ext + S_ext.T)
    np.fill_diagonal(S_ext, 1.0)

    # ---- isotonic calibration on training pairs (O×O) ----
    if iso_calibrate:
        print("[iso] Fitting isotonic calibration on training perts...")
        iso = _fit_isotonic_on_pairs(S_ext[np.ix_(iO, iO)], Y_O)
        S_cal = _apply_isotonic_matrix(iso, S_ext)
    else:
        S_cal = S_ext

    # ---- build kernel K from calibrated similarity ----
    K = S_cal
    K = 0.5 * (K + K.T)
    K += np.eye(K.shape[0], dtype=K.dtype) * 1e-6  # nudge toward PSD

    KOO = K[np.ix_(iO, iO)]
    KUO = K[np.ix_(iU, iO)]

    # ---- KRR: \hat Y_U = K_{UO} (K_{OO} + λI)^{-1} Y_O ----
    A = np.linalg.solve(KOO + krr_lambda * np.eye(KOO.shape[0], dtype=KOO.dtype),
                        Y_O)  # (|O|, G)
    Y_U_hat_delta = KUO @ A  # (|U|, G) deltas

    # ---- Convert to EXPRESSION levels expected by evaluator ----
    pred_mat = Y_U_hat_delta + ctrl_mean_target[None, :]   # (|U|, G)

    # ---- True expression rows and pert_names from EVAL split (like your example) ----
    test_pert_mask = adata_eval.obs[target_label].isin(U)
    true_mat = np.asarray(adata_eval.X)[np.asarray(test_pert_mask)]
    pert_names = adata_eval.obs.loc[test_pert_mask, target_label].astype(str).tolist()

    # ctrl_mean returned in the bundle = TRAIN control mean (global baseline)
    return pred_mat, true_mat, pert_names, np.asarray(ctrl_mean_target).ravel()

def evaluate_model(
    adata: ad.AnnData,
    args,
    pred_bundle: tuple[np.ndarray, np.ndarray, list[str], np.ndarray],
):
    """
    Computes:
      - per-perturbation MAE
      - knockdown efficiency (abs & %) for true vs predicted at the target gene
      - perturbation similarity: mean & min pairwise Pearson corr between predicted mean effect vectors
      - PDS (Perturbation Discrimination Score): mean over perturbations
    Prints a concise report and returns a dict with all metrics.
    """
    pred_mat, true_mat, pert_names, ctrl_mean = pred_bundle
    G = adata.n_vars
    df_obs = adata.obs
    labels = df_obs[args.target_label].astype(str).values

    # group indices by perturbation (excluding control)
    perts = sorted(set(pert_names))
    # target mapping
    t2gi = build_target_to_gene_index(adata, args.target_label)

    # per-pert pseudobulks (pred & true) and MAE
    pred_bulk = {}
    true_bulk = {}
    mae_per_pert = {}
    bulk_mae_per_pert = {}

    # map pert_names (length Np) to row indices for quick grouping
    rows_by_pert = defaultdict(list)
    for i, p in enumerate(pert_names):
        rows_by_pert[p].append(i)

    for p in perts:
        rows = rows_by_pert[p]
        yhat_p = pred_mat[rows]  # (n_p, G)
        ytrue_p = true_mat[rows] # (n_p, G)
        pred_bulk[p] = yhat_p.mean(axis=0)
        true_bulk[p] = ytrue_p.mean(axis=0)
        # per-cell MAE (cells+genes)
        mae_per_pert[p] = np.mean(np.abs(yhat_p - ytrue_p))
        # pseudobulk MAE (genes only)
        bulk_mae_per_pert[p] = float(np.mean(np.abs(pred_bulk[p] - true_bulk[p])))

    # knockdown efficiency at target gene (abs & %), true vs predicted
    # uses GLOBAL control pseudobulk as the "control" reference
    eps = 1e-8
    kd_eff = {}  # p -> dict
    for p in perts:
        t = t2gi.get(p, -1)
        if t < 0:
            kd_eff[p] = {"target_gene": None,
                         "true_abs": np.nan, "true_pct": np.nan,
                         "pred_abs": np.nan, "pred_pct": np.nan}
            continue
        ctrl_t = float(ctrl_mean[t])
        true_t = float(true_bulk[p][t])
        pred_t = float(pred_bulk[p][t])

        # absolute "knockdown" (positive if below control)
        true_abs = ctrl_t - true_t
        pred_abs = ctrl_t - pred_t
        # percentage relative to control level
        true_pct = true_abs / (ctrl_t + eps)
        pred_pct = pred_abs / (ctrl_t + eps)

        kd_eff[p] = {"target_gene": adata.var_names[t],
                     "true_abs": true_abs, "true_pct": true_pct,
                     "pred_abs": pred_abs, "pred_pct": pred_pct}

    # perturbation similarity (correlations between predicted mean effect vectors)
    # use predicted (pred_bulk[p] - ctrl_mean) as effect vector
    effect_vecs = []
    for p in perts:
        effect_vecs.append(pred_bulk[p] - ctrl_mean)
    effect_mat = np.stack(effect_vecs, axis=0)  # (K,G)
    # pairwise Pearson correlation matrix
    K = effect_mat.shape[0]
    # normalize
    em = effect_mat - effect_mat.mean(axis=1, keepdims=True)
    denom = np.sqrt((em ** 2).sum(axis=1, keepdims=True)) + 1e-8
    emn = em / denom
    corr_mat = emn @ emn.T  # (K,K)
    # take upper triangle excluding diagonal
    iu = np.triu_indices(K, k=1)
    mean_corr = float(corr_mat[iu].mean()) if iu[0].size > 0 else np.nan
    min_corr = float(corr_mat[iu].min()) if iu[0].size > 0 else np.nan

    # PDS (Perturbation Discrimination Score)
    # - use absolute deltas vs control
    # - exclude only the TRUE target gene for expression data, by name
    # - zero-based rank normalized by N (not N-1): PDS_p = 1 - rank/N
    # absolute deltas vs global control mean
    true_bulk_mat = np.stack([np.abs(true_bulk[p] - ctrl_mean) for p in perts], axis=0)  # (K,G)
    pred_bulk_mat = np.stack([np.abs(pred_bulk[p] - ctrl_mean) for p in perts], axis=0)  # (K,G)
    t_idx_per_pert = {p: t2gi.get(p, -1) for p in perts}

    # precompute masks per pair to exclude targets
    Kp = len(perts)
    PDS_scores = []
    for i, p in enumerate(perts):
        # build include mask: exclude target gene IF its name equals the perturbation label
        mask = np.ones(G, dtype=bool)
        tj = t_idx_per_pert[p]
        if tj >= 0:
            mask[tj] = False
        # distances from ALL real effects to this predicted effect
        dists = pairwise_distances(
            true_bulk_mat[:, mask],    # (K, G')
            pred_bulk_mat[i, mask][None, :],  # (1, G')
            metric="manhattan",
        ).ravel()
        order = np.argsort(dists)          # ascending
        # rank of the correct perturbation (zero-based)
        p_index = i  # same ordering
        rank0 = int(np.flatnonzero(order == p_index)[0])
        # normalize by K (not K-1), then invert
        PDS_scores.append(1.0 - rank0 / Kp)

    PDS_mean = float(np.mean(PDS_scores)) if len(PDS_scores) > 0 else np.nan

    # ---- Print concise report ----
    print("\n=== Evaluation ===")
    # print(f"Per-cell MAE (mean ± sd over perts): {np.mean(list(mae_per_pert.values())):.5f} ± {np.std(list(mae_per_pert.values())):.5f}")
    print(f"Pseudobulk MAE (mean over perts):   {np.mean(list(bulk_mae_per_pert.values())):.5f}")
    print(f"Perturbation similarity (pred mean effects): mean corr={mean_corr:.4f}, min corr={min_corr:.4f}")
    print(f"PDS (mean over perts): {PDS_mean:.4f}")
    print("\nKnockdown efficiency per perturbation (target gene, true_abs, true_pct, pred_abs, pred_pct):")
    # show a few lines sorted by true_abs descending
    preview = sorted(kd_eff.items(), key=lambda kv: (np.nan_to_num(kv[1]['true_abs'], nan=-1e9)), reverse=True)
    for p, d in preview[: min(10, len(preview))]:
        tg = d['target_gene'] or "N/A"
        print(f"  {p:20s}  tg={tg:12s}  true_abs={d['true_abs']:.4f}  true_pct={d['true_pct']:.2%}  "
              f"pred_abs={d['pred_abs']:.4f}  pred_pct={d['pred_pct']:.2%}")

    return {
        "mae_per_pert": mae_per_pert,
        "bulk_mae_per_pert": bulk_mae_per_pert,
        "kd_eff": kd_eff,
        "mean_corr_pred_effects": mean_corr,
        "min_corr_pred_effects": min_corr,
        "PDS_mean": PDS_mean,
        "PDS_scores": dict(zip(perts, PDS_scores)),
    }


def main():
    args = parse_arguments()
    # Both baselines require pseudobulked data, so we enforce it.
    args.use_pseudobulk = True

    # ---------------------------
    # 1. Read and Prepare Data
    # ---------------------------
    print("Reading and preparing data...")
    adata_target = ad.read_h5ad(args.in_h5ad)
    adata_source = ad.read_h5ad(args.external_h5ad)

    # Subset both datasets to their intersection of genes and perturbations
    adata_source, adata_target = intersect_datasets(
        adata_source, adata_target, args.target_label, args.control_label
    )

    # Process and pseudobulk both datasets
    for adata in [adata_source, adata_target]:
        # Normalize and log1p. We assume source is already pseudobulked.
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
    
    # Pseudobulk the target data if it's not already
    if adata_target.n_obs > len(adata_target.obs[args.target_label].unique()):
        print("Collapsing target data to pseudobulk...")
        adata_target = collapse_to_pseudobulk(adata_target, args.target_label)

    # Split the TARGET data into train/test sets
    adata_train, adata_test = train_test_split(args, adata_target)
    eval_adata = adata_test if adata_test is not None else adata_train

    # ---------------------------
    # 3. Evaluate on the Test Set
    # ---------------------------
    print("\n=== Evaluation on {} set ===".format("TEST" if adata_test is not None else "TRAIN"))
    if args.method == "krr":
        eval_pred_bundle = krr_predict_from_external(
            adata_source=adata_source,
            adata_train=adata_train,
            adata_eval=eval_adata,
            target_label=args.target_label,
            control_label=args.control_label,
            krr_lambda=args.krr_lambda,
            kernel_metric=args.kernel_metric,
            ctrl_mean_target=None,
            iso_calibrate=args.iso_calibrate,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    evaluate_model(adata=eval_adata, args=args, pred_bundle=eval_pred_bundle)

    # ---------------------------
    # 4. (Optional) Evaluate on the Train Set
    # ---------------------------
    if args.eval_on_train and (adata_test is not None):
        print("\n=== Evaluation on TRAIN set ===")
        train_pred_bundle = krr_predict_from_external(
            adata_source=adata_source,
            adata_train=adata_train,
            adata_eval=adata_train,
            target_label=args.target_label,
            control_label=args.control_label,
            krr_lambda=args.krr_lambda,
            kernel_metric=args.kernel_metric,
            ctrl_mean_target=None,
            iso_calibrate=args.iso_calibrate,
        )
        evaluate_model(adata=adata_train, args=args, pred_bundle=train_pred_bundle)

    # ---------------------------
    # 5. (Optional) Write Output Files
    # ---------------------------
    if args.out_pred_h5ad:
        print(f"\nWriting prediction outputs to {args.out_pred_h5ad}...")
        write_pred_true_h5ads(
            eval_adata=eval_adata,
            pred_bundle=eval_pred_bundle,
            out_pred_h5ad=args.out_pred_h5ad,
            target_label=args.target_label,
            control_label=args.control_label,
        )
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()
