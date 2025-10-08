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


def run_causal_baseline(
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    hvgs: int | None = None,
    scale_by_on_target: bool = False,
) -> pd.DataFrame:
    """
    GIES (+ interventional score) -> IDA total effects, via rpy2 + R pcalg.
    Returns a DataFrame of predicted *effects* (rows=pert labels in eval set,
    cols=genes in adata.var_names order).

    Assumptions:
      - adata_eval is pseudobulked & log1p-normalized.
      - Each row corresponds to a context (control or a single-gene perturbation).
      - obs[target_label] gives the perturbed gene; control rows equal control_label.

    Params:
      hvgs: if set (e.g., 1000–3000), restrict the graph learning to top-HVGs
            (improves stability when contexts << genes). Effects are returned
            over *all* genes by placing HVG effects and zeros elsewhere.
      scale_by_on_target: if True, multiply IDA effects for each pert by an
            estimated efficacy α_t computed from the observed on-target shift
            (pseudobulk - control mean). Off by default for “pure” linear effects.

    Dependencies in your env:
      pip install rpy2
      In R: install.packages("pcalg"); install.packages("graph")
    """

    # --- Prepare data ---
    X = to_numpy(adata_eval.X).astype(np.float64)  # (B, G)
    genes_all = list(map(str, adata_eval.var_names))
    labels = adata_eval.obs[target_label].astype(str).values
    ctrl_mask = (labels == control_label)
    if ctrl_mask.sum() == 0:
        raise ValueError("No control contexts found; needed to compute sanity checks / α_t if scaling.")

    # optional HVG restriction for structure learning (effects returned over all genes)
    if hvgs is not None and hvgs > 0 and hvgs < X.shape[1]:
        # simple HVG by variance (since pseudobulked)
        var = X.var(axis=0)
        hvg_idx = np.argsort(var)[::-1][:hvgs]
        hvg_idx = np.sort(hvg_idx)
    else:
        hvg_idx = np.arange(X.shape[1])

    X_hvg = X[:, hvg_idx]
    genes_hvg = [genes_all[i] for i in hvg_idx]

    # z-score columns for Gaussian score
    eps = 1e-9
    mu = X_hvg.mean(axis=0, dtype=np.float64)
    sd = X_hvg.std(axis=0, dtype=np.float64)
    sd[sd < eps] = 1.0  # avoid zero-div
    Xz = (X_hvg - mu) / sd

    # Build interventional SETTINGS (unique labels) for this dataset
    gene_to_pos_hvg = {g: i for i, g in enumerate(genes_hvg)}
    unique_labels = pd.Index(labels).astype(str).unique().tolist()
    def label_to_iv(lbl: str):
        if lbl == control_label:
            return IntVector([])
        return IntVector([gene_to_pos_hvg[lbl] + 1]) if lbl in gene_to_pos_hvg else IntVector([])
    targets_unique = [label_to_iv(lbl) for lbl in unique_labels]
    label_to_sid = {lbl: i+1 for i, lbl in enumerate(unique_labels)}
    target_index = IntVector([label_to_sid[lbl] for lbl in labels])

    # Estimate per-pert on-target efficacy α_t if requested (in *z-scored HVG space* for consistency)
    alpha_by_pert = {}
    if scale_by_on_target:
        ctrl_mean = X[ctrl_mask].mean(axis=0, dtype=np.float64)  # in original space
        for p in np.unique(labels[~ctrl_mask]):
            # only define α_t if the gene is present in var_names
            if p in genes_all:
                t_idx_all = genes_all.index(p)
                pb_mean_p = X[labels == p].mean(axis=0, dtype=np.float64)
                # crude efficacy proxy: fractional reduction at the target (clamped)
                base = np.expm1(np.maximum(ctrl_mean[t_idx_all], 0.0))
                obs = np.expm1(np.maximum(pb_mean_p[t_idx_all], 0.0))
                if base <= 0:
                    alpha = 0.0
                else:
                    alpha = float(np.clip(1.0 - (obs / base), 0.0, 1.5))
                alpha_by_pert[p] = alpha

    # --- rpy2: ship data to R and run GIES + IDA ---
    ro.r("suppressPackageStartupMessages(library(pcalg))")
    ro.r("suppressPackageStartupMessages(library(graph))")

    # Assign variables in R global env (with explicit conversion context)
    with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
        ro.globalenv["Xz"] = Xz
        ro.globalenv["targets_unique"] = ListVector({str(i+1): iv for i, iv in enumerate(targets_unique)})
        ro.globalenv["target_index"] = target_index

    # Build score and run GIES
    ro.r("""
        # Xz: rows = contexts, cols = genes (HVG set), standardized
        # targets_unique: list of targets per *setting*; target_index: setting id per row
        score_obj <- new("GaussL0penIntScore", data = Xz,
                         targets = targets_unique, target.index = target_index)
        gies_fit <- gies(score_obj)
        cpdag_obj <- gies_fit$essgraph  # CPDAG as graphNEL
        # sample covariance (on standardized data => ~correlation)
        S <- stats::cov(Xz)
    """)

    # For each *pert gene present in HVG set* we compute IDA total effects
    effects_hvg = []
    eval_perts = sorted({p for p in labels if p != control_label})
    for p in eval_perts:
        if p not in gene_to_pos_hvg:
            # Not in HVG set: predict zero effects on all genes (filled later)
            effects_hvg.append((p, None))
            continue
        x_pos = gene_to_pos_hvg[p] + 1  # 1-based
        # idaFast returns total effects from x to *all* variables in CPDAG
        with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
            ro.globalenv["x_pos"] = x_pos
            beta = ro.r("""
                p <- ncol(S_tr)
                eff <- numeric(p)
                for (y in 1:p) {
                    be <- tryCatch(
                            ida(x.pos = x_pos, y.pos = y,
                                graph = cpdag_obj,  covMat = S_tr,
                                method = "local", type = "cpdag"),
                            error = function(e)
                            ida(x.pos = x_pos, y.pos = y,
                                graphEst = cpdag_obj, CovMat = S_tr,
                                method = "local", type = "cpdag")
                        )
                    # ida() can return multiple values (different valid parent-sets).
                    eff[y] <- mean(be)
                }
                eff
            """)
            beta = np.asarray(beta, dtype=np.float64)
        # Optional magnitude calibration by α_t (on HVG subset only; later broadcast)
        if scale_by_on_target and p in alpha_by_pert:
            beta = beta * float(alpha_by_pert[p])
        effects_hvg.append((p, beta))

    # --- Assemble final DataFrame over *all* genes in var_names order ---
    # Initialize with zeros everywhere (conservative); fill HVG entries where available
    effects_mat = np.zeros((len(eval_perts), len(genes_all)), dtype=np.float64)
    row_index = []
    for i, (p, beta_h) in enumerate(effects_hvg):
        row_index.append(p)
        if beta_h is None:
            continue
        # place into all-genes vector
        full = np.zeros(len(genes_all), dtype=np.float64)
        full[hvg_idx] = beta_h
        effects_mat[i, :] = full

    effects_pred = pd.DataFrame(effects_mat, index=row_index, columns=genes_all)

    # Ensure we return rows in the evaluation set order (stable)
    return effects_pred.reindex(row_index)


def run_causal_baseline_train_eval(
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    hvgs: int | None = None,
    scale_by_on_target: bool = False,
) -> pd.DataFrame:
    """
    Fit GIES (+ interventional score) on TRAIN, then compute IDA total effects
    for the perts present in EVAL. Returns a DataFrame (rows=eval perts, cols=genes).
    """

    # ----- TRAIN side: build Xz_train and targets_list_train -----
    Xtr = to_numpy(adata_train.X).astype(np.float64)
    genes_all = list(map(str, adata_train.var_names))
    if list(adata_eval.var_names) != genes_all:
        raise ValueError("Train and eval var_names must match and be identically ordered.")
    labels_tr = adata_train.obs[target_label].astype(str).values
    ctrl_mask_tr = (labels_tr == control_label)
    if ctrl_mask_tr.sum() == 0:
        raise ValueError("Training split must include at least one control context.")

    # HVG restriction for *structure learning* only
    if hvgs is not None and 0 < hvgs < Xtr.shape[1]:
        var = Xtr.var(axis=0)
        hvg_idx = np.argsort(var)[::-1][:hvgs]
        hvg_idx = np.sort(hvg_idx)
    else:
        hvg_idx = np.arange(Xtr.shape[1])
    genes_hvg = [genes_all[i] for i in hvg_idx]
    gene_to_pos_hvg = {g: i for i, g in enumerate(genes_hvg)}

    Xtr_hvg = Xtr[:, hvg_idx]
    mu = Xtr_hvg.mean(axis=0, dtype=np.float64)
    sd = Xtr_hvg.std(axis=0, dtype=np.float64); sd[sd < 1e-9] = 1.0
    Xz_tr = (Xtr_hvg - mu) / sd

    # ---- Build interventional SETTINGS for TRAIN ----
    # Each setting corresponds to a label value (control or a pert).
    # targets_unique : list of IntVector (length = n_settings)
    # target_index_tr: IntVector of length n_rows mapping each row to its setting (1-based).
    unique_labels_tr = pd.Index(labels_tr).astype(str).unique().tolist()
    def label_to_iv(lbl: str):
        if lbl == control_label:
            return IntVector([])  # observational/control setting
        return IntVector([gene_to_pos_hvg[lbl] + 1]) if lbl in gene_to_pos_hvg else IntVector([])
    targets_unique = [label_to_iv(lbl) for lbl in unique_labels_tr]
    # map each row's label to its setting id (1-based for R)
    label_to_sid = {lbl: i+1 for i, lbl in enumerate(unique_labels_tr)}
    target_index_tr = IntVector([label_to_sid[lbl] for lbl in labels_tr])

    # Optional efficacy scaling computed from TRAIN only (original space)
    alpha_by_pert = {}
    if scale_by_on_target:
        ctrl_mean_tr = Xtr[ctrl_mask_tr].mean(axis=0, dtype=np.float64)
        for p in np.unique(labels_tr[~ctrl_mask_tr]):
            if p in genes_all:
                t_idx = genes_all.index(p)
                pb_p = Xtr[labels_tr == p].mean(axis=0, dtype=np.float64)
                base = np.expm1(np.maximum(ctrl_mean_tr[t_idx], 0.0))
                obs  = np.expm1(np.maximum(pb_p[t_idx], 0.0))
                alpha = float(np.clip(1.0 - (obs / base) if base > 0 else 0.0, 0.0, 1.5))
                alpha_by_pert[p] = alpha

    # ----- R: fit on TRAIN only -----
    ro.r("suppressPackageStartupMessages(library(pcalg))")
    ro.r("suppressPackageStartupMessages(library(graph))")
    with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
        ro.globalenv["Xz_tr"] = Xz_tr
        ro.globalenv["targets_unique"] = ListVector({str(i+1): iv for i, iv in enumerate(targets_unique)})
        ro.globalenv["target_index_tr"] = target_index_tr
    ro.r("""
        score_tr <- new("GaussL0penIntScore", data = Xz_tr, targets = targets_unique, target.index = target_index_tr)
        gies_fit <- gies(score_tr)
        cpdag_obj <- gies_fit$essgraph
        S_tr <- stats::cov(Xz_tr)
    """)

    # ----- EVAL side: request IDA effects for EVAL perts (even if unseen in train) -----
    labels_ev = adata_eval.obs[target_label].astype(str).values
    eval_perts = sorted({p for p in labels_ev if p != control_label})
    effects_mat = np.zeros((len(eval_perts), len(genes_all)), dtype=np.float64)
    for i, p in enumerate(eval_perts):
        if p in gene_to_pos_hvg:
            x_pos = gene_to_pos_hvg[p] + 1
            # Version-agnostic ida(): pick correct arg names (graph vs graphEst,
            # cov.mat vs CovMat vs covMat), then loop over all y and average parent-sets.
            with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
                ro.globalenv["x_pos"] = x_pos
                beta = ro.r("""
                    ida_one <- function(xpos, ypos) {
                      f <- formals(ida)
                      # Choose graph argument name
                      garg <- if ("graph"   %in% names(f)) "graph" else "graphEst"
                      # Choose covariance argument name
                      carg <- if ("cov.mat" %in% names(f)) "cov.mat" else if ("CovMat" %in% names(f)) "CovMat" else "covMat"
                      # Build the call programmatically
                      a <- list(x.pos = xpos, y.pos = ypos, method = "local", type = "cpdag")
                      a[[garg]] <- cpdag_obj
                      a[[carg]] <- S_tr
                      be <- do.call(ida, a)
                      # ida can return a vector (multiple parent-sets); take the mean
                      mean(as.numeric(be))
                    }
                    p <- ncol(S_tr)
                    eff <- vapply(1:p, function(y) ida_one(x_pos, y), numeric(1))
                    eff
                """)
                beta = np.asarray(beta, dtype=np.float64)
            if scale_by_on_target and p in alpha_by_pert:
                beta = beta * float(alpha_by_pert[p])
            full = np.zeros(len(genes_all), dtype=np.float64)
            full[hvg_idx] = beta
            effects_mat[i, :] = full
        else:
            # pert gene not in HVG set: leave zeros (conservative)
            pass
    effects_pred = pd.DataFrame(effects_mat, index=eval_perts, columns=genes_all)
    return effects_pred

def run_causal_baseline_cdt_notears_train_eval(
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    max_iter: int = 200,
    l1: float = 0.01,           # NOTEARS sparsity (lambda)
    ridge_alpha: float = 1.0,   # weight fitting (per-node ridge)
    scale_by_on_target: bool = False,
) -> pd.DataFrame:
    """
    Pure-Python baseline:
      1) Learn DAG structure on TRAIN via CDT's NOTEARS (linear).
      2) For each node j, fit ridge regression X_j ~ X_parents(j) on TRAIN to get edge weights.
      3) Build B (p x p) from those weights; total effects = (I - B^T)^(-1).
      4) For each eval perturbation t, predict effect as column t of that inverse (optionally scaled by α_t).

    Returns a DataFrame (rows = eval perturbation labels, cols = genes in var_names order), in log1p space.
    """

    # --- Prepare TRAIN data (standardize) ---
    Xtr = to_numpy(adata_train.X).astype(np.float64)  # (n_train, p)
    genes = list(map(str, adata_train.var_names))
    assert list(adata_eval.var_names) == genes, "Train/eval var_names must match (order too)."
    n, p = Xtr.shape

    mu = Xtr.mean(axis=0, dtype=np.float64)
    sd = Xtr.std(axis=0, dtype=np.float64); sd[sd < 1e-9] = 1.0
    Ztr = (Xtr - mu) / sd
    df_tr = pd.DataFrame(Ztr, columns=genes)

    # --- 1) Structure learning with NOTEARS (CDT, pure Python) ---
    # Keep it simple & deterministic-ish
    #algo = NOTEARS(l1=l1, max_iter=max_iter, verbose=False)
    algo = GIES()
    # Returns a weighted DAG as networkx.DiGraph (edge weights are NOTEARS' parameters)
    dag_nx: nx.DiGraph = algo.predict(df_tr)

    # --- 2) Edge-weight refit via ridge on TRAIN (parents -> child) ---
    # Build B (p x p) so that X ≈ B^T X + eps; i.e., for child j, parents P, coeffs on P go into B[P, j]
    B = np.zeros((p, p), dtype=np.float64)
    Xt = Ztr  # standardized features
    for j, gene_j in enumerate(genes):
        # parents are nodes with edges i -> j
        parents = [u for u, v in dag_nx.in_edges(gene_j)]
        if not parents:
            continue
        P_idx = [genes.index(g) for g in parents]
        X_par = Xt[:, P_idx]
        y = Xt[:, j]
        # Small ridge to stabilize
        coef = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(X_par, y).coef_
        B[P_idx, j] = coef

    # Zero diagonal (safety) and tiny thresholding
    np.fill_diagonal(B, 0.0)
    thr = np.percentile(np.abs(B), 95) * 1e-8 + 1e-10
    B[np.abs(B) < thr] = 0.0

    # --- 3) Total-effect matrix A = (I - B^T)^(-1) ---
    I = np.eye(p, dtype=np.float64)
    try:
        A = np.linalg.inv(I - B.T)
    except np.linalg.LinAlgError:
        A = np.linalg.inv(I - B.T + 1e-6 * I)

    # --- 4) Optional α_t scaling estimated from TRAIN (original space) ---
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

    # --- Build per-pert predicted effects for EVAL ---
    labels_ev = adata_eval.obs[target_label].astype(str).values
    eval_perts = sorted({lab for lab in labels_ev if lab != control_label})
    effects = np.zeros((len(eval_perts), p), dtype=np.float64)
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
    effects_df = run_causal_baseline_cdt_notears_train_eval(
        adata_train=adata_train,
        adata_eval=eval_adata,
        target_label=args.target_label,
        control_label=args.control_label,
        max_iter=200,
        l1=0.01,                # increase for sparser graph; decrease for denser
        ridge_alpha=1.0,
        scale_by_on_target=False
    )
    # 2) Convert effects_df -> (pred_mat, true_mat, pert_names, ctrl_mean)
    pred_bundle = _bundle_from_effects_df(
        eval_adata, effects_df, args.target_label, args.control_label
    )
    # 3) Evaluate with your existing metrics
    print("\n=== Evaluation on {} set ===".format(
        "TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)")
    )
    _ = evaluate_model(adata=eval_adata, args=args, pred_bundle=pred_bundle)

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
