#!/usr/bin/env python3
import argparse, math, os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional, List

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.vectors import ListVector, IntVector
from rpy2.robjects.conversion import localconverter

from sklearn.linear_model import Ridge
import networkx as nx
from cdt.causality.graph import GIES

from utils import *


def parse_arguments():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--out_pred_h5ad", type=str, default="",
                    help="If set, write an AnnData with predictions for the evaluation split.")
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_pct_perts", type=float, default=0.0,
                    help="Fraction of perturbation labels (excluding control) to hold out for testing. 0.0 = no holdout.")
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional path to a separate test AnnData. If set, overrides --test_pct_perts.")
    ap.add_argument('--eval_on_train', action='store_true', help='Evaluate on training set in addition to test set')
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--causal_backend", choices=["cdt", "python", "auto"], default="auto",
                    help="GIES+IDA backend: 'cdt' (R pcalg via CDT), 'python' (GES/IGSP-style), or 'auto'")
    ap.add_argument("--standardize", action="store_true",
                    help="Z-score genes before causal discovery (recommended).")
    ap.add_argument("--hvg_topk", type=int, default=0,
                    help="If >0, restrict causal discovery to top-K highly variable genes to stabilize.")
    ap.add_argument("--knn_k_pert", type=int, default=5,
                    help="Fallback NN-pert baseline: neighbors averaged if causal backend unavailable.")

    args = ap.parse_args()
    return args


def _bundle_from_effects_df(
    adata: ad.AnnData,
    effects_df: pd.DataFrame,
    target_label: str,
    control_label: str,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Convert a (perts x genes) DataFrame of predicted *mean effect vectors* (log1p space),
    aligned to adata.var_names, into the tuple expected by evaluate_model:
        (pred_mat, true_mat, pert_names, ctrl_mean)
    It broadcasts each perturbation's predicted pseudobulk (ctrl_mean + effect)
    to all rows of that perturbation in `adata`.
    """
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    ctrl_mask = labels == control_label
    if ctrl_mask.sum() == 0:
        raise ValueError("No control rows found to compute ctrl_mean.")
    ctrl_mean = X[ctrl_mask].mean(axis=0).astype(np.float32)
    # Sanity: ensure columns match var_names order
    if list(effects_df.columns) != list(adata.var_names):
        effects_df = effects_df.reindex(columns=adata.var_names)
    # Rows to fill (exclude controls)
    pert_idx = np.where(~ctrl_mask)[0]
    pert_names = labels[pert_idx].tolist()
    # Build prediction rows by label lookup in effects_df
    pred_rows: List[np.ndarray] = []
    true_rows: List[np.ndarray] = []
    for i in pert_idx:
        p = labels[i]
        # predicted pseudobulk = ctrl_mean + predicted effect for this label
        if p not in effects_df.index:
            raise KeyError(f"Predicted effects missing for label: {p}")
        pred_pb = ctrl_mean + effects_df.loc[p].values.astype(np.float32)
        pred_rows.append(pred_pb)
        true_rows.append(X[i])
    pred_mat = np.stack(pred_rows, axis=0)
    true_mat = np.stack(true_rows, axis=0)
    return pred_mat, true_mat, pert_names, ctrl_mean

def _write_pred_true_h5ads(
    eval_adata: ad.AnnData,
    pred_bundle: tuple[np.ndarray, np.ndarray, list[str], np.ndarray],
    out_pred_h5ad: str,
    target_label: str,
    control_label: str,
):
    """
    Write two AnnData files for the *full evaluation split* (controls INCLUDED):
      - predicted (.h5ad): X has predictions on perturbation rows; control rows are copied from eval_adata.
      - true (.true.h5ad): exact copy of eval_adata (ground truth).
    """
    pred_mat, _true_mat_nc, pert_names, _ctrl_mean = pred_bundle
    labels = eval_adata.obs[target_label].astype(str).values
    nonctrl_mask = labels != control_label
    pert_idx = np.where(nonctrl_mask)[0]  # order matches pred_mat rows

    # Start from the true matrix for ALL rows; then replace only the pert rows with predictions
    X_true_all = np.asarray(eval_adata.X).astype(np.float32, copy=False)
    X_pred_all = X_true_all.copy()
    X_pred_all[pert_idx, :] = pred_mat.astype(np.float32)

    # Build AnnData objects with the full eval obs/var/uns
    ad_pred = ad.AnnData(
        X=X_pred_all,
        obs=eval_adata.obs.copy(),
        var=eval_adata.var.copy(),
        uns=eval_adata.uns.copy(),
    )
    ad_true = ad.AnnData(
        X=X_true_all,
        obs=eval_adata.obs.copy(),
        var=eval_adata.var.copy(),
        uns=eval_adata.uns.copy(),
    )
    # Optional breadcrumbs
    ad_pred.obs[f"{target_label}_predicted_for"] = labels  # includes control_label entries
    ad_true.obs[f"{target_label}_true_for"] = labels

    os.makedirs(os.path.dirname(out_pred_h5ad) or ".", exist_ok=True)
    ad_pred.write_h5ad(out_pred_h5ad)
    ad_true.write_h5ad(out_pred_h5ad + ".true.h5ad")


def run_causal_baseline_igsp_train_eval(
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    alpha_ci: float = 1e-3,        # CI test threshold
    alpha_inv: float = 1e-3,       # invariance test threshold
    ridge_alpha: float = 1.0,      # edge-weight ridge
    standardize: bool = True,      # z-score on TRAIN before learning
    hvgs: int | None = None,       # optional: limit structure learning to top-HVGs
    scale_by_on_target: bool = False,  # optional magnitude calibration
) -> pd.DataFrame:
    """
    Pure-Python IGSP baseline with causaldag:
      1) Build observational (controls) + interventional settings from TRAIN.
      2) Run UT-IGSP to get a DAG structure (directional).
      3) Fit linear weights per node via ridge on TRAIN -> B (p x p).
      4) Total-effect matrix A = (I - B^T)^(-1).
      5) For EVAL perts, predicted effect = column t of A (optionally scaled by α_t).

    Returns a DataFrame (rows = eval perts, cols = genes in var_names order).
    """
    import numpy as np, pandas as pd
    import causaldag as cd
    from causaldag import igsp

    from conditional_independence import (
        MemoizedCI_Tester,
        partial_correlation_suffstat,
        partial_correlation_test,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
    )
    from sklearn.linear_model import Ridge

    # ----- prepare TRAIN data -----
    Xtr = to_numpy(adata_train.X).astype(np.float64)   # (n_train, p)
    genes = list(map(str, adata_train.var_names))
    assert list(adata_eval.var_names) == genes, "Train/Eval var_names must match & be aligned."
    p = Xtr.shape[1]

    # Optional: restrict structure learning to HVGs (effects returned for all genes)
    if hvgs is not None and 0 < hvgs < p:
        var = Xtr.var(axis=0)
        hvg_idx = np.argsort(var)[::-1][:hvgs]
        hvg_idx = np.sort(hvg_idx)
    else:
        hvg_idx = np.arange(p)

    # Standardize (fit on TRAIN only)
    mu = Xtr[:, hvg_idx].mean(axis=0) if standardize else np.zeros(len(hvg_idx))
    sd = Xtr[:, hvg_idx].std(axis=0); sd[sd < 1e-9] = 1.0
    Ztr_hvg = (Xtr[:, hvg_idx] - mu) / sd if standardize else Xtr[:, hvg_idx]
    genes_hvg = [genes[i] for i in hvg_idx]
    gene_pos_hvg = {g: i for i, g in enumerate(genes_hvg)}

    # ----- build settings from TRAIN -----
    labels_tr = adata_train.obs[target_label].astype(str).values
    ctrl_mask_tr = (labels_tr == control_label)

    # Observational samples (controls)
    obs_samples = Ztr_hvg[ctrl_mask_tr]
    # Interventional settings: one per pert label present in TRAIN (rows with that label)
    train_perts = sorted({lab for lab in labels_tr if lab != control_label})
    iv_samples_list = []
    setting_list = []  # for UT-IGSP API
    # known_interventions lists use *positions in the HVG subspace*; if a pert gene not in HVGs, set empty -> observational-like
    for plab in train_perts:
        rows = (labels_tr == plab)
        if rows.sum() == 0:
            continue
        iv_samples = Ztr_hvg[rows]
        iv_samples_list.append(iv_samples)
        if plab in gene_pos_hvg:
            tgt = [gene_pos_hvg[plab]]
        else:
            tgt = []  # pert gene not in HVGs -> treat as observational for structure learning
        # Provide BOTH keys to be compatible with different package variants
        setting_list.append({
            "interventions": tgt,           # required by graphical_model_learning.algorithms.dag.gsp.igsp
            "known_interventions": tgt,     # used by unknown_target_igsp and some older APIs
        })

    # Edge case: if no control rows or no IV settings, bail early with zeros
    if obs_samples.shape[0] == 0 or len(iv_samples_list) == 0:
        eval_perts = sorted({lab for lab in adata_eval.obs[target_label].astype(str).values if lab != control_label})
        return pd.DataFrame(np.zeros((len(eval_perts), p)), index=eval_perts, columns=genes)

    # ----- UT-IGSP inputs: CI & invariance testers -----
    obs_suff = partial_correlation_suffstat(obs_samples)
    inv_suff = gauss_invariance_suffstat(obs_samples, iv_samples_list)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suff, alpha=alpha_ci)
    inv_tester = MemoizedInvarianceTester(gauss_invariance_test, inv_suff, alpha=alpha_inv)

    # Nodes are 0..(p_hvg-1)
    nodes = set(range(len(hvg_idx)))

    # ----- run UT-IGSP to get a DAG on HVGs -----
    _igsp_out = igsp(
        setting_list=setting_list,
        nodes=nodes,
        ci_tester=ci_tester,
        invariance_tester=inv_tester
    )  # est_dag_hvg is a cd.DAG over nodes
    est_dag_hvg = _igsp_out[0] if isinstance(_igsp_out, tuple) else _igsp_out

    # ----- refit edge weights (linear ridge) on TRAIN (HVG subspace) -----
    B_hvg = np.zeros((len(hvg_idx), len(hvg_idx)), dtype=np.float64)
    Xt = Ztr_hvg
    for j in range(len(hvg_idx)):
        parents = list(est_dag_hvg.parents_of(j))
        if not parents:
            continue
        X_par = Xt[:, parents]
        y = Xt[:, j]
        coef = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(X_par, y).coef_
        B_hvg[parents, j] = coef
    np.fill_diagonal(B_hvg, 0.0)

    # ----- total effects on HVGs -----
    I_h = np.eye(B_hvg.shape[0], dtype=np.float64)
    try:
        A_hvg = np.linalg.inv(I_h - B_hvg.T)
    except np.linalg.LinAlgError:
        A_hvg = np.linalg.inv(I_h - B_hvg.T + 1e-6 * I_h)

    # Expand to all genes (zeros outside HVG set)
    A_full = np.zeros((p, p), dtype=np.float64)
    A_full[np.ix_(hvg_idx, hvg_idx)] = A_hvg

    # Optional α_t scaling from TRAIN (original space)
    alpha_by_pert = {}
    if scale_by_on_target:
        Xtr_full = to_numpy(adata_train.X).astype(np.float64)
        ctrl_mean = Xtr_full[ctrl_mask_tr].mean(axis=0, dtype=np.float64) if ctrl_mask_tr.any() else np.zeros(p)
        for plab in train_perts:
            if plab in genes:
                t_idx = genes.index(plab)
                pb_mean = Xtr_full[labels_tr == plab].mean(axis=0, dtype=np.float64)
                base = np.expm1(np.maximum(ctrl_mean[t_idx], 0.0))
                obs  = np.expm1(np.maximum(pb_mean[t_idx], 0.0))
                alpha_by_pert[plab] = float(np.clip(1.0 - (obs / base) if base > 0 else 0.0, 0.0, 1.5))

    # ----- build effects for ALL perts seen in TRAIN ∪ EVAL (no refit needed later) -----
    labels_tr = adata_train.obs[target_label].astype(str).values
    labels_ev = adata_eval.obs[target_label].astype(str).values
    train_perts = {lab for lab in labels_tr if lab != control_label}
    eval_perts  = {lab for lab in labels_ev if lab != control_label}
    all_perts   = sorted(train_perts | eval_perts)

    effects = np.zeros((len(all_perts), p), dtype=np.float64)
    for i, plab in enumerate(all_perts):
        if plab in genes:
            t = genes.index(plab)
            delta = A_full[:, t].copy()
            if scale_by_on_target and plab in alpha_by_pert:
                delta *= alpha_by_pert[plab]
            effects[i, :] = delta
    return pd.DataFrame(effects, index=all_perts, columns=genes)

def run_causal_baseline_precision_train_eval(
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    shrinkage: str = "lw",  # "lw" or "oas"
    scale_by_on_target: bool = False,
) -> pd.DataFrame:
    import numpy as np, pandas as pd
    from sklearn.covariance import LedoitWolf, OAS

    Xtr = to_numpy(adata_train.X).astype(np.float64)
    genes = list(map(str, adata_train.var_names))
    assert list(adata_eval.var_names) == genes

    # Standardize on TRAIN
    mu = Xtr.mean(axis=0, dtype=np.float64)
    sd = Xtr.std(axis=0, dtype=np.float64); sd[sd < 1e-9] = 1.0
    Ztr = (Xtr - mu) / sd

    # Precision estimate
    prec = (OAS().fit(Ztr).precision_ if shrinkage == "oas"
            else LedoitWolf().fit(Ztr).precision_)
    Theta = prec

    # Regression coefficients from precision
    B = -Theta / (np.diag(Theta)[:, None])
    np.fill_diagonal(B, 0.0)

    # Total effects
    I = np.eye(B.shape[0])
    try:
        A = np.linalg.inv(I - B.T)
    except np.linalg.LinAlgError:
        A = np.linalg.inv(I - B.T + 1e-6 * I)

    # Optional α_t scaling from TRAIN (original space)
    alpha_by_pert = {}
    if scale_by_on_target:
        labels_tr = adata_train.obs[target_label].astype(str).values
        ctrl_mask_tr = (labels_tr == control_label)
        if ctrl_mask_tr.sum() > 0:
            ctrl_mean = Xtr[ctrl_mask_tr].mean(axis=0, dtype=np.float64)
            for p_label in np.unique(labels_tr[~ctrl_mask_tr]):
                if p_label in genes:
                    t_idx = genes.index(p_label)
                    pb_mean = Xtr[labels_tr == p_label].mean(axis=0, dtype=np.float64)
                    base = np.expm1(np.maximum(ctrl_mean[t_idx], 0.0))
                    obs  = np.expm1(np.maximum(pb_mean[t_idx], 0.0))
                    alpha = float(np.clip(1.0 - (obs / base) if base > 0 else 0.0, 0.0, 1.5))
                    alpha_by_pert[p_label] = alpha

    labels_ev = adata_eval.obs[target_label].astype(str).values
    eval_perts = sorted({lab for lab in labels_ev if lab != control_label})
    effects = np.zeros((len(eval_perts), len(genes)), dtype=np.float64)
    for i, p_label in enumerate(eval_perts):
        if p_label in genes:
            t = genes.index(p_label)
            delta = A[:, t].copy()
            if scale_by_on_target and p_label in alpha_by_pert:
                delta *= alpha_by_pert[p_label]
            effects[i, :] = delta
    return pd.DataFrame(effects, index=eval_perts, columns=genes)

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


# ----------------------------
# CLI
# ----------------------------
def main():
    args = parse_arguments()

    # ---------------------------
    # Read input data
    # ---------------------------
    adata = ad.read_h5ad(args.in_h5ad)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    adata_train, adata_test = train_test_split(args, adata)

    eval_adata = adata_test if adata_test is not None else adata_train
    print("\n=== Building causal-baseline predictions ===")
    # effects_df = run_causal_baseline_train_eval(
    #     adata_train=adata_train,
    #     adata_eval=eval_adata,
    #     target_label=args.target_label,
    #     control_label=args.control_label,
    #     hvgs=None,                 # adjust or None
    #     scale_by_on_target=False,  # optional
    # )
    effects_all_df = run_causal_baseline_igsp_train_eval(  # run_causal_baseline_precision_train_eval(
        adata_train=adata_train,
        adata_eval=eval_adata,
        target_label=args.target_label,
        control_label=args.control_label,
        scale_by_on_target=False,
        alpha_ci=5e-2, alpha_inv=5e-2, ridge_alpha=1.0, standardize=True, hvgs=None,
    )

    # 2) Convert effects_df -> (pred_mat, true_mat, pert_names, ctrl_mean)
    # pred_bundle = _bundle_from_effects_df(
    #     eval_adata, effects_df, args.target_label, args.control_label
    # )
    # --- Slice rows for the EVAL split perts ---
    eval_labels = eval_adata.obs[args.target_label].astype(str).values
    eval_perts  = sorted({lab for lab in eval_labels if lab != args.control_label})
    effects_eval_df = effects_all_df.loc[[p for p in eval_perts if p in effects_all_df.index]]
    pred_bundle = _bundle_from_effects_df(eval_adata, effects_eval_df, args.target_label, args.control_label)


    # 3) Evaluate with your existing metrics
    print("\n=== Evaluation on {} set ===".format(
        "TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)")
    )
    _ = evaluate_model(adata=eval_adata, args=args, pred_bundle=pred_bundle)

    # 5) (Optional) Evaluate on TRAIN split as well (fit on TRAIN, eval on TRAIN)
    if args.eval_on_train and (adata_test is not None):
        print("\n=== Evaluation on TRAIN set (fit on TRAIN) ===")
        # Build predictions for TRAIN split using the same precision baseline
        # effects_df_tr = run_causal_baseline_precision_train_eval(
        #     adata_train=adata_train,
        #     adata_eval=adata_train,
        #     target_label=args.target_label,
        #     control_label=args.control_label,
        #     scale_by_on_target=False,
        # )
        train_labels = adata_train.obs[args.target_label].astype(str).values
        train_perts  = sorted({lab for lab in train_labels if lab != args.control_label})
        effects_df_tr = effects_all_df.loc[[p for p in train_perts if p in effects_all_df.index]]
        pred_bundle_tr = _bundle_from_effects_df(
            adata_train, effects_df_tr, args.target_label, args.control_label
        )
        _ = evaluate_model(adata=adata_train, args=args, pred_bundle=pred_bundle_tr)

    if args.out_pred_h5ad:
        if hasattr(eval_adata.X, "toarray"):
            eval_adata.X = eval_adata.X.toarray()
        _write_pred_true_h5ads(
            eval_adata=eval_adata,
            pred_bundle=pred_bundle,
            out_pred_h5ad=args.out_pred_h5ad,
            target_label=args.target_label,
            control_label=args.control_label,
        )

    # if args.eval_on_train and (adata_test is not None):
    #     print("\n=== (Optional) Evaluate same baseline on TRAIN set ===")
    #     effects_df_tr = run_causal_baseline(
    #         adata_train, target_label=args.target_label, control_label=args.control_label
    #     )
    #     pred_bundle_tr = _bundle_from_effects_df(
    #         adata_train, effects_df_tr, args.target_label, args.control_label
    #     )
    #     _ = evaluate_model(adata=adata_train, args=args, pred_bundle=pred_bundle_tr)

if __name__ == "__main__":
    main()
