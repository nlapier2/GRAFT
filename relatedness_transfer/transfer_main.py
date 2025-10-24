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
    ap.add_argument("--intersect_genes", action="store_true", default=False,
                    help="If set, intersect genes across source and target. Otherwise, only intersect perts (default).")
    ap.add_argument("--already_logged", action="store_true", 
                    help="Set if inputs are already log1p-normalized; otherwise apply log1p to raw counts.")
    ap.add_argument("--keep_oov_perts", action="store_true", 
                    help="If set, keep perts that are not in the source∩target intersection (left as AverageKnown baseline). "
                         "If unset, drop those rows before evaluation/output.")

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
    # Neighbor sharpening (perturbation-space)
    ap.add_argument("--kernel_gamma", type=float, default=1.0,
                    help="Power sharpening for K_UO rows; >1 sharpens (e.g., 1.4). 1.0 disables.")
    ap.add_argument("--topk", type=int, default=0,
                    help="Keep only top-k neighbors per row of K_UO; 0 disables.")
    # Subspace boosting (gene-space)
    ap.add_argument("--boost_pcs", type=int, default=0,
                    help="Number of PCA components (from Y_O) to boost in predictions; 0 disables.")
    ap.add_argument("--boost_gamma", type=float, default=0.6,
                    help="Boost strength along PCA subspace (e.g., 0.3–1.0).")

    args = ap.parse_args()
    return args

def build_truth_bundle(adata_split: ad.AnnData, target_label: str, control_label: str):
    """
    Returns (true_mat, pert_names) for this split.
    true_mat rows match pert_names order; controls excluded.
    """
    G = adata_split.n_vars
    perts = list(dict.fromkeys(map(str, adata_split.obs[target_label].values)))
    perts = [p for p in perts if p != control_label]
    rows = []
    for p in perts:
        m = (adata_split.obs[target_label] == p).values
        rows.append(np.asarray(adata_split.X)[m].reshape(-1, G).mean(axis=0))
    true_mat = np.stack(rows, axis=0) if rows else np.zeros((0, adata_split.n_vars))
    return true_mat, perts

def build_average_known_baseline(
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    ctrl_mean_global: np.ndarray,
):
    """
    Build AverageKnown predictions for TRAIN & EVAL:
      1) Compute TRAIN mean delta vs GLOBAL control mean.
      2) For each split, pred_mat_split := ctrl_mean_global - mean_delta_train (broadcasted).
    Returns:
      (pred_train, true_train, names_train), (pred_eval, true_eval, names_eval)
    """
    # TRAIN truths / names
    true_train, names_train = build_truth_bundle(adata_train, target_label, control_label)
    # EVAL truths / names
    true_eval, names_eval = build_truth_bundle(adata_eval, target_label, control_label)
    # Mean TRAIN delta vs global ctrl
    if true_train.shape[0] > 0:
        mean_delta_train = (true_train - ctrl_mean_global[None, :]).mean(axis=0, keepdims=True)
    else:
        mean_delta_train = np.zeros((1, ctrl_mean_global.shape[0]), dtype=float)
    # Broadcast to splits (expression space)
    pred_train = np.repeat(ctrl_mean_global[None, :] - mean_delta_train, repeats=len(names_train), axis=0) \
                 if len(names_train) else mean_delta_train[:0]
    pred_eval = np.repeat(ctrl_mean_global[None, :] - mean_delta_train, repeats=len(names_eval), axis=0) \
                if len(names_eval) else mean_delta_train[:0]
    return (pred_train, true_train, names_train), (pred_eval, true_eval, names_eval)

def apply_target_overwrite_and_clamp(
    pred_mat: np.ndarray,
    pert_names: list[str],
    adata_train: ad.AnnData,
    target_label: str,
    control_label: str,
    ctrl_mean_global: np.ndarray,
):
    """Overwrite target gene expression via avg KD efficiency from TRAIN; then clamp >=0."""
    var_to_idx = {g: i for i, g in enumerate(adata_train.var_names.astype(str))}
    global_eff, per_gene_eff = _compute_avg_kd_efficiencies(
        adata_train=adata_train, O=_pert_list(adata_train, target_label, control_label),
        target_label=target_label, control_label=control_label, ctrl_mean_target=ctrl_mean_global
    )
    for row, pert in enumerate(pert_names):
        gi = var_to_idx.get(pert, None)  # assume pert name equals target gene name when present
        if gi is None:
            continue
        eff = per_gene_eff.get(pert, global_eff)
        pred_mat[row, gi] = ctrl_mean_global[gi] * (1.0 - eff)
    np.maximum(pred_mat, 0.0, out=pred_mat)

def intersect_datasets(adata_source, adata_target, target_label, control_label, intersect_genes=False):
    """
    Subsets two AnnData objects to their common genes and perturbations.

    Args:
        adata_source: The source (external) AnnData object.
        adata_target: The target AnnData object.
        target_label: The obs column containing perturbation labels.
        control_label: The label for control samples.
        intersect_genes: If True, intersect genes across datasets; else only perts.

    Returns:
        A tuple of (subsetted source AnnData, subsetted target AnnData).
    """
    print("Finding intersection of genes and perturbations...")

    source_perts = set(adata_source.obs[target_label].unique())
    target_perts = set(adata_target.obs[target_label].unique())
    common_perts = sorted(list(source_perts.intersection(target_perts)))

    # Ensure the control label is always kept, even if it's not in the intersection
    if control_label not in common_perts:
        if control_label in source_perts and control_label in target_perts:
            common_perts.append(control_label)
    
    if intersect_genes:
        # First, get a list of valid genes from the source (not all NaN), then intersect.
        common_genes = np.intersect1d(
            adata_source.var_names[~np.isnan(to_numpy(adata_source.X)).all(axis=0)],
            adata_target.var_names
        )
        print(f"  Found {len(common_genes)} common genes.")
    else:
        common_genes = None
    print(f"  Found {len(common_perts) - 1} common perturbations (plus control).")

    if common_genes is None:
        # Intersect perts only; keep original gene spaces
        adata_source_sub = adata_source[adata_source.obs[target_label].isin(common_perts), :].copy()
        adata_target_sub = adata_target[adata_target.obs[target_label].isin(common_perts), :].copy()
    else:
        # Intersect both perts and genes
        adata_source_sub = adata_source[adata_source.obs[target_label].isin(common_perts), common_genes].copy()
        adata_target_sub = adata_target[adata_target.obs[target_label].isin(common_perts), common_genes].copy()

    # Remove columns in source with all-NaN values (genes not in its dataset)
    if not intersect_genes:
        adata_source_sub = adata_source_sub[:, ~np.isnan(to_numpy(adata_source_sub.X)).all(axis=0)].copy()

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

def _compute_avg_kd_efficiencies(
    adata_train: ad.AnnData,
    O: list[str],
    target_label: str,
    control_label: str,
    ctrl_mean_target: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Estimate knockdown efficiency e in [0,1] from TRAIN perts:
      e = clip((ctrl - pert_expr) / max(ctrl, eps), 0, 1)
    Returns (global_avg, per_gene_avg).
    If a gene never appears as a target in O, fall back to global_avg.
    """
    G = adata_train.n_vars
    var_to_idx = {g: i for i, g in enumerate(adata_train.var_names.astype(str))}
    effs_by_gene: dict[str, list[float]] = {}
    eps = 1e-8
    for p in O:
        g = str(p)  # assume pert name equals target gene symbol
        if g not in var_to_idx:
            continue
        gi = var_to_idx[g]
        v = adata_train[adata_train.obs[target_label] == p].X
        v = np.asarray(v).reshape(-1, G).mean(axis=0)
        ctrl = float(ctrl_mean_target[gi])
        if ctrl <= eps:
            continue
        eff = (ctrl - float(v[gi])) / max(ctrl, eps)
        eff = float(np.clip(eff, 0.0, 1.0))
        effs_by_gene.setdefault(g, []).append(eff)
    # per-gene and global averages
    per_gene_avg = {g: float(np.mean(vals)) for g, vals in effs_by_gene.items()}
    all_effs = [e for vals in effs_by_gene.values() for e in vals]
    global_avg = float(np.mean(all_effs)) if all_effs else 0.5  # sensible fallback
    return global_avg, per_gene_avg

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

def _sharpen_neighbors(K_UO: np.ndarray, tau: float = 1.0, topk: int = 0) -> np.ndarray:
    """
    A4: Neighbor sharpening. Elementwise power on similarities then optional top-k per row.
    Operates ONLY on K_UO (cross block) to avoid changing the fit on O.
    """
    if tau <= 1.0 and (topk is None or topk <= 0):
        return K_UO
    Kp = np.maximum(K_UO, 0.0).astype(np.float32)
    if tau > 1.0:
        Kp = np.power(Kp, tau, dtype=np.float32)
    if topk and topk > 0:
        topk = min(topk, Kp.shape[1])
        # threshold each row to its k-th largest value
        part = np.partition(Kp, Kp.shape[1] - topk, axis=1)
        thresh = part[:, Kp.shape[1] - topk : Kp.shape[1] - topk + 1]
        Kp[Kp < thresh] = 0.0
    Kp_sum = Kp.sum(axis=1, keepdims=True) + 1e-8
    Kp /= Kp_sum
    return Kp

def _subspace_boost(Y_U_hat_delta: np.ndarray, Y_O_delta: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """
    A3: Subspace boosting in gene space. Boost components along the top-k PCs
    computed from training deltas Y_O (|O| x G). Uses SVD to avoid extra deps.
    """
    if k <= 0 or gamma <= 0:
        return Y_U_hat_delta
    k = min(k, min(Y_O_delta.shape[0], Y_O_delta.shape[1]))
    # Center across perts before SVD to focus on between-pert variation
    Yc = Y_O_delta - Y_O_delta.mean(axis=0, keepdims=True)
    # thin SVD: Yc = U S Vt ; Vt is (G x G) truncated to k
    try:
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Y_U_hat_delta  # fall back safely if SVD fails
    P = Vt[:k, :].T  # G x k
    proj = (Y_U_hat_delta @ P) @ P.T  # project into top-k subspace
    boosted = Y_U_hat_delta + gamma * proj
    return boosted

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
    kernel_gamma: float = 1.0,
    topk: int = 0,
    boost_pcs: int = 0,
    boost_gamma: float = 0.6,
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
    A = np.linalg.solve(KOO + krr_lambda * np.eye(KOO.shape[0], dtype=KOO.dtype), Y_O)  # (|O|, G)
    # sharpen neighbor mixing at prediction time
    KUO_sharp = _sharpen_neighbors(KUO, tau=kernel_gamma, topk=topk)
    Y_U_hat_delta = KUO_sharp @ A  # (|U|, G)

    # subspace boosting along perturbation-contrast directions from Y_O
    if boost_pcs and boost_pcs > 0 and boost_gamma > 0:
        Y_U_hat_delta = _subspace_boost(Y_U_hat_delta, Y_O, k=boost_pcs, gamma=boost_gamma)

    # ---- Convert to EXPRESSION levels expected by evaluator ----
    pred_mat = Y_U_hat_delta + ctrl_mean_target[None, :]   # (|U|, G)

    # --- Target gene overwrite using avg KD efficiency from TRAIN perts ---
    var_to_idx = {g: i for i, g in enumerate(adata_train.var_names.astype(str))}
    global_eff, per_gene_eff = _compute_avg_kd_efficiencies(
        adata_train=adata_train, O=O, target_label=target_label,
        control_label=control_label, ctrl_mean_target=ctrl_mean_target
    )
    for row, pert in enumerate(U):
        g = str(pert)
        gi = var_to_idx.get(g, None)
        if gi is None:
            continue
        eff = per_gene_eff.get(g, global_eff)  # [0,1]
        # predicted target-gene expression = ctrl_mean * (1 - eff)
        pred_mat[row, gi] = ctrl_mean_target[gi] * (1.0 - eff)

    # --- Non-negativity clamp: no expression should be < 0 ---
    np.maximum(pred_mat, 0.0, out=pred_mat)

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
    # Read and Prepare Data
    # ---------------------------
    print("Reading and preparing data...")
    adata_target = ad.read_h5ad(args.in_h5ad)
    adata_source = ad.read_h5ad(args.external_h5ad)
    adata_source = adata_source[:, ~np.isnan(to_numpy(adata_source.X)).all(axis=0)].copy()  # filter all-NaN genes
    if not args.already_logged:
        sc.pp.normalize_total(adata_target, inplace=True)
        sc.pp.log1p(adata_target)
        sc.pp.normalize_total(adata_source, inplace=True)
        sc.pp.log1p(adata_source)

    # Compute a SINGLE global control mean from ALL target controls (fixed across splits)
    ctrl_mask_full = (adata_target.obs[args.target_label] == args.control_label).values
    ctrl_mean_global = np.asarray(adata_target.X)[ctrl_mask_full].mean(axis=0).reshape(-1)

    # Split TARGET into train/test (controls appear in both; controls themselves are never modified)
    adata_train, adata_test = train_test_split(args, adata_target)
    eval_adata = adata_test if adata_test is not None else adata_train

    # Build AverageKnown baselines and truths BEFORE any intersection
    (pred_tr, true_tr, names_tr), (pred_ev, true_ev, names_ev) = build_average_known_baseline(
        adata_train, eval_adata, args.target_label, args.control_label, ctrl_mean_global
    )

    # Now intersect datasets (perts always; genes optional). This will also strip all-NaN external genes if needed.
    adata_source, adata_target_int = intersect_datasets(
        adata_source, adata_target, args.target_label, args.control_label, intersect_genes=args.intersect_genes
    )
    # Keep split views in intersected target
    adata_train_int = adata_target_int[adata_target_int.obs.index.isin(adata_train.obs.index)].copy()
    eval_adata_int  = adata_target_int[adata_target_int.obs.index.isin(eval_adata.obs.index)].copy()

    # If genes were intersected, align baseline tensors and ctrl_mean to the intersected gene order
    if args.intersect_genes:
        gene_order = adata_target_int.var_names
        idx_in_full = pd.Index(adata_target.var_names).get_indexer(gene_order)
        # Slice baselines and truths to intersected genes
        if pred_tr.shape[0] > 0:
            pred_tr  = pred_tr[:, idx_in_full]
            true_tr  = true_tr[:, idx_in_full]
        if pred_ev.shape[0] > 0:
            pred_ev  = pred_ev[:, idx_in_full]
            true_ev  = true_ev[:, idx_in_full]
        # Slice the global control mean as well
        ctrl_mean_global = ctrl_mean_global[idx_in_full]

    # ---------------------------
    # Evaluate on the Test Set
    # ---------------------------
    print("\n=== Evaluation on {} set ===".format("TEST" if adata_test is not None else "TRAIN"))
    if args.method != "krr":
        raise ValueError(f"Unknown method: {args.method}")
    # Run KRR ONLY on the intersected views; get predictions for intersected perts
    pred_krr_ev, _true_krr_ev, names_krr_ev, _ctrl_ignored = krr_predict_from_external(
        adata_source=adata_source,
        adata_train=adata_train_int,
        adata_eval=eval_adata_int,
        target_label=args.target_label,
        control_label=args.control_label,
        krr_lambda=args.krr_lambda,
        kernel_metric=args.kernel_metric,
        ctrl_mean_target=ctrl_mean_global,  # fixed control mean
        iso_calibrate=args.iso_calibrate,
        kernel_gamma=args.kernel_gamma,
        topk=args.topk,
        boost_pcs=args.boost_pcs,
        boost_gamma=args.boost_gamma,
    )
    # Overwrite rows (by pert name) into the AverageKnown baseline (eval split)
    name2row_ev = {p: i for i, p in enumerate(names_ev)}
    for j, p in enumerate(names_krr_ev):
        if p in name2row_ev:
            pred_ev[name2row_ev[p], :] = pred_krr_ev[j, :]
    # Optionally DROP OOV perts (not present in intersection)
    if not args.keep_oov_perts:
        keep = np.array([p in set(names_krr_ev) for p in names_ev])
        pred_ev, true_ev = pred_ev[keep], true_ev[keep]
        names_ev = [p for (p, k) in zip(names_ev, keep) if k]
    # Use the AnnData with matching gene space for target overwrite indexing
    ad_train_for_eff = adata_train_int if args.intersect_genes else adata_train
    # Post-processing on the FULL eval predictions (target overwrite + clamp)
    apply_target_overwrite_and_clamp(
        pred_mat=pred_ev, pert_names=names_ev,
        adata_train=ad_train_for_eff,  # efficiencies from TRAIN perts
        target_label=args.target_label, control_label=args.control_label,
        ctrl_mean_global=ctrl_mean_global
    )
    # Evaluate using our assembled bundle (fixed control mean)
    eval_adata_for_eval = eval_adata_int if args.intersect_genes else eval_adata
    evaluate_model(adata=eval_adata_for_eval, args=args, pred_bundle=(pred_ev, true_ev, names_ev, ctrl_mean_global))


    # ---------------------------
    # (Optional) Evaluate on the Train Set
    # ---------------------------
    if args.eval_on_train and (adata_test is not None):
        print("\n=== Evaluation on TRAIN set ===")
        pred_krr_tr, _true_krr_tr, names_krr_tr, _ = krr_predict_from_external(
            adata_source=adata_source,
            adata_train=adata_train_int,
            adata_eval=adata_train_int,
            target_label=args.target_label,
            control_label=args.control_label,
            krr_lambda=args.krr_lambda,
            kernel_metric=args.kernel_metric,
            ctrl_mean_target=ctrl_mean_global,
            iso_calibrate=args.iso_calibrate,
            kernel_gamma=args.kernel_gamma,
            topk=args.topk,
            boost_pcs=args.boost_pcs,
            boost_gamma=args.boost_gamma,
        )
        # Overwrite into train baseline
        name2row_tr = {p: i for i, p in enumerate(names_tr)}
        for j, p in enumerate(names_krr_tr):
            if p in name2row_tr:
                pred_tr[name2row_tr[p], :] = pred_krr_tr[j, :]
        if not args.keep_oov_perts:
            keep = np.array([p in set(names_krr_tr) for p in names_tr])
            pred_tr, true_tr = pred_tr[keep], true_tr[keep]
            names_tr = [p for (p, k) in zip(names_tr, keep) if k]
        # Use the AnnData with matching gene space for target overwrite indexing
        ad_train_for_eff = adata_train_int if args.intersect_genes else adata_train
        apply_target_overwrite_and_clamp(
            pred_mat=pred_tr, pert_names=names_tr,
            adata_train=ad_train_for_eff, target_label=args.target_label,
            control_label=args.control_label, ctrl_mean_global=ctrl_mean_global
        )
        train_adata_for_eval = adata_train_int if args.intersect_genes else adata_train
        evaluate_model(adata=train_adata_for_eval, args=args, pred_bundle=(pred_tr, true_tr, names_tr, ctrl_mean_global))

    # ---------------------------
    # (Optional) Write Output Files
    # ---------------------------
    if args.out_pred_h5ad:
        print(f"\nWriting prediction outputs to {args.out_pred_h5ad}...")
        write_pred_true_h5ads(
            eval_adata=(eval_adata_int if args.intersect_genes else eval_adata),
            pred_bundle=(pred_ev, true_ev, names_ev, ctrl_mean_global),
            out_pred_h5ad=args.out_pred_h5ad,
            target_label=args.target_label,
            control_label=args.control_label,
        )
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()
