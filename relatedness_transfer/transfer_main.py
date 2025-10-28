#!/usr/bin/env python3
import warnings
# Suppress annoying FutureWarning from scanpy
warnings.filterwarnings('ignore', category=FutureWarning)
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
from transforms import *


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
    ap.add_argument("--test_predict_out", type=str, default="",
                    help="Path to write cell-level predicted test AnnData (.h5ad).")

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
    # PDS sharpening (post-processing on predicted effects)
    ap.add_argument("--pds_sharpen", type=str, default="none",
                    choices=["none", "power", "topk", "sigmoid"],
                    help="Post-process predicted effects to boost large signals and shrink small ones.")
    ap.add_argument("--pds_gamma", type=float, default=1.5,
                    help="Exponent for power mode (>|1| boosts large |Δ|, shrinks small).")
    ap.add_argument("--pds_topk_frac", type=float, default=0.1,
                    help="Fraction (0-1) of largest-|Δ| genes to inflate in topk mode.")
    ap.add_argument("--pds_alpha", type=float, default=0.3,
                    help="Inflation factor for topk mode (Δ_topk *= (1+alpha)).")
    ap.add_argument("--pds_beta", type=float, default=0.2,
                    help="Shrink factor for non-topk in topk mode (Δ_else *= (1-beta)).")
    ap.add_argument("--pds_sigmoid_B", type=float, default=0.7,
                    help="Slope B in Δ' = A*tanh(B*Δ). A is auto-scaled to preserve a high-percentile.")
    ap.add_argument("--pds_preserve_quantile", type=float, default=0.95,
                    help="Quantile of |Δ| whose magnitude is preserved by the transform.")

    # Confidence-weighted amplification of predicted deltas (pre-PDS-sharpening)
    ap.add_argument("--conf_boost_alpha", type=float, default=0.0,
                    help="Amplify high-confidence (low-variance) genes. 0.0 = disabled.")
    ap.add_argument("--conf_shrink_alpha", type=float, default=0.0,
                    help="Optionally shrink low-confidence (high-variance) genes. 0.0 = no shrink.")
    ap.add_argument("--conf_min_var", type=float, default=1e-6,
                    help="Lower variance bound when converting var->confidence.")
    ap.add_argument("--conf_max_var", type=float, default=1.0,
                    help="Upper variance bound when converting var->confidence.")

    # Single-cell synthesis using predictive variance
    ap.add_argument("--scell_use_var", action="store_true",
                    help="If set, sample per-cell deltas using predictive variance instead of applying one fixed delta.")
    ap.add_argument("--scell_var_scale", type=float, default=1.0,
                    help="Global multiplier on per-gene stddev when sampling per-cell deltas.")
    ap.add_argument("--scell_clip_zero", action="store_true",
                    help="Clamp synthesized cells to >=0 after applying sampled deltas.")

    args = ap.parse_args()
    return args

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
    # NEW: confidence boost args
    conf_boost_alpha: float = 0.0,
    conf_shrink_alpha: float = 0.0,
    conf_min_var: float = 1e-6,
    conf_max_var: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Returns:
      pred_mat        (|U| x G) predicted EXPRESSION for eval perts U
      true_mat        (|U| x G) true EXPRESSION rows for eval perts U
      pert_names      list[str] length |U|
      ctrl_mean       (G,)
      pred_delta_var  (|U| x G) predictive VARIANCE on the DELTA space (before ctrl_mean added)
    """
    G = adata_train.n_vars
    # ---- use TRAIN control mean (for adding back to deltas & evaluator's baseline) ----
    if ctrl_mean_target is None:
        train_mask = np.asarray(adata_train.obs[target_label] == control_label)
        ctrl_mean_target = np.asarray(adata_train.X)[train_mask].mean(axis=0).reshape(-1)

    # ---- define sets ----
    O = pert_list(adata_train, target_label, control_label)  # observed perts
    U = pert_list(adata_eval,  target_label, control_label)  # to predict/evaluate
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
        Z = row_standardize(Delta_src)
        S_ext = Z @ Z.T
    else:  # "cosine"
        Z = Delta_src / (np.linalg.norm(Delta_src, axis=1, keepdims=True) + 1e-8)
        S_ext = Z @ Z.T
    S_ext = 0.5 * (S_ext + S_ext.T)
    np.fill_diagonal(S_ext, 1.0)

    # ---- isotonic calibration on training pairs (O×O) ----
    if iso_calibrate:
        print("[iso] Fitting isotonic calibration on training perts...")
        iso = fit_isotonic_on_pairs(S_ext[np.ix_(iO, iO)], Y_O)
        S_cal = apply_isotonic_matrix(iso, S_ext)
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
    KUO_sharp = sharpen_neighbors(KUO, tau=kernel_gamma, topk=topk)
    Y_U_hat_delta = KUO_sharp @ A  # (|U|, G)

    # --- Estimate per-gene residual variance on training perts (noise model)
    Y_O_hat = KOO @ A                           # (|O|, G), fitted deltas for O
    resid = Y_O - Y_O_hat                       # (|O|, G)
    sigma2_gene = (resid ** 2).mean(axis=0)     # (G,)

    # --- GP-style scalar uncertainty per eval pert based on kernel geometry
    # s2_raw[u] = k_uu - k_uO @ (KOO+λI)^(-1) @ k_Ou
    KOO_reg_inv = np.linalg.inv(KOO + krr_lambda * np.eye(KOO.shape[0], dtype=KOO.dtype))
    # precompute for speed: M = KOO_reg_inv @ K_Ou for each u
    # We'll just loop since |U| is usually not huge
    s2_raw_list = []
    for row_u in range(KUO.shape[0]):
        k_uO = KUO[row_u, :].reshape(1, -1)     # (1, |O|)
        k_Ou = k_uO.T                           # (|O|, 1)
        k_uu = float(K[iU[row_u], iU[row_u]])   # scalar
        middle = KOO_reg_inv @ k_Ou             # (|O|,1)
        s2_raw = k_uu - (k_uO @ middle).item()    # scalar
        if s2_raw < 0:
            # small negative due to numerics; clip
            s2_raw = 0.0
        s2_raw_list.append(s2_raw)
    s2_raw_arr = np.asarray(s2_raw_list, dtype=np.float32)  # (|U|,)

    # Broadcast to per-gene predictive variance
    pred_delta_var = s2_raw_arr[:, None] * sigma2_gene[None, :]  # (|U|, G)

    # subspace boosting along perturbation-contrast directions from Y_O
    if boost_pcs and boost_pcs > 0 and boost_gamma > 0:
        Y_U_hat_delta = subspace_boost(Y_U_hat_delta, Y_O, k=boost_pcs, gamma=boost_gamma)

    # --- Confidence-weighted amplification of deltas BEFORE adding ctrl_mean ---
    Y_U_hat_delta = apply_confidence_boost(
        pred_delta_mat = Y_U_hat_delta,
        pred_delta_var = pred_delta_var,
        conf_boost_alpha = conf_boost_alpha,
        conf_shrink_alpha = conf_shrink_alpha,
        conf_min_var = conf_min_var,
        conf_max_var = conf_max_var,
    )

    # ---- Convert to EXPRESSION levels expected by evaluator ----
    pred_mat = Y_U_hat_delta + ctrl_mean_target[None, :]   # (|U|, G)

    # --- Target gene overwrite using avg KD efficiency from TRAIN perts ---
    var_to_idx = {g: i for i, g in enumerate(adata_train.var_names.astype(str))}
    global_eff, per_gene_eff = compute_avg_kd_efficiencies(
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
    return pred_mat, true_mat, pert_names, np.asarray(ctrl_mean_target).ravel(), pred_delta_var

def write_cell_level_predictions(
    adata_test_orig: ad.AnnData,
    eval_gene_names,
    pred_mat_eval: np.ndarray,
    pred_delta_var_eval: np.ndarray,
    names_eval: list[str],
    ctrl_mean_eval: np.ndarray,
    target_label: str,
    control_label: str,
    out_path: str,
    scell_use_var: bool = False,
    scell_var_scale: float = 1.0,
    scell_clip_zero: bool = True,
    random_state: int | None = None,
):
    """
    Build a cell-level predicted AnnData for the TEST set by:
      - Copying control cells from adata_test_orig unchanged.
      - For each perturbation p in adata_test_orig (excluding control), sampling N_p control cells
        and subtracting the learned delta vector delta_p = ctrl_mean - pred_expr[p].
      - Clamping to >= 0 and writing to out_path.
    The resulting AnnData has (controls + synthesized perts) and matches per-pert cell counts in adata_test_orig.
    """
    if out_path is None or out_path == "":
        return
    rng = np.random.default_rng(random_state)

    # --- Align gene space: use the evaluation gene order (columns of pred_mat_eval) ---
    eval_genes = np.array(eval_gene_names, dtype=str)
    test_genes = np.array(adata_test_orig.var_names, dtype=str)
    # map eval_genes into adata_test_orig
    take_idx = pd.Index(test_genes).get_indexer(eval_genes)
    if np.any(take_idx < 0):
        # intersect
        common = np.intersect1d(eval_genes, test_genes)
        if common.size == 0:
            raise ValueError("No overlapping genes between eval gene space and adata_test_orig.")
        print(f"[cells] Restricting to {common.size} common genes for cell-level synthesis.")
        # remap everything to 'common'
        # positions in eval
        pos_eval = pd.Index(eval_genes).get_indexer(common)
        # positions in test
        pos_test = pd.Index(test_genes).get_indexer(common)
        eval_genes = common
        pred_mat_eval = pred_mat_eval[:, pos_eval]
        ctrl_mean_eval = ctrl_mean_eval[pos_eval]
        take_idx = pos_test  # now all >= 0 by construction
    else:
        # same order as eval
        pass

    # Pull the control pool from adata_test_orig (in eval gene order)
    ctrl_mask = (adata_test_orig.obs[target_label].astype(str) == control_label).values
    if not ctrl_mask.any():
        raise ValueError("No control cells found in adata_test_orig; cannot synthesize perts from control pool.")
    X_ctrl = to_numpy(adata_test_orig.X)[:, take_idx]
    X_ctrl = X_ctrl[ctrl_mask]  # (n_ctrl, G_eval)
    n_ctrl, G = X_ctrl.shape

    # Compute per-pert delta vectors from predicted pseudobulk expression (eval space)
    # delta_p = ctrl_mean - pred_expr[p], so x_pert ≈ x_ctrl - delta_p
    name_to_row = {p: i for i, p in enumerate(names_eval)}
    name_to_varrow = {p: i for i, p in enumerate(names_eval)}

    # Prepare outputs
    obs_rows = []
    X_rows = []
    var_df = adata_test_orig.var.loc[test_genes[take_idx]].copy()
    var_df.index = eval_genes  # ensure matching names/order

    # 1) copy original control cells (unaltered) into output
    ctrl_obs = adata_test_orig.obs.loc[ctrl_mask].copy()
    X_rows.append(X_ctrl)  # unchanged
    obs_rows.append(ctrl_obs)

    # 2) for each perturbation present in adata_test_orig, synthesize cells
    perts_in_test = adata_test_orig.obs[target_label].astype(str).unique().tolist()
    perts_in_test = [p for p in perts_in_test if p != control_label]
    for p in perts_in_test:
        n_p = int((adata_test_orig.obs[target_label].astype(str) == p).sum())
        if n_p == 0:
            continue
        row = name_to_row.get(p, None)
        if row is None:
            # No predicted vector for this pert; skip (or you could choose to leave original cells)
            print(f"[cells] WARNING: no prediction for pert '{p}' in pred_mat_eval; skipping synthesis for this pert.")
            continue
        pred_expr = pred_mat_eval[row]          # (G,)
        delta_p_mean = ctrl_mean_eval - pred_expr    # (G,)

        # per-gene predictive variance for this pert in delta space
        var_row = name_to_varrow.get(p, None)
        if var_row is None:
            var_vec = np.zeros_like(delta_p_mean)
        else:
            var_vec = pred_delta_var_eval[var_row]  # (G,)

        # sample control indices (with replacement if needed)
        replace = n_p > n_ctrl
        idx = rng.choice(n_ctrl, size=n_p, replace=replace)
        X_base = X_ctrl[idx]                    # (n_p, G)

        if scell_use_var:
            # sample a different delta for each synthetic cell
            sampled_deltas = sample_cell_level_deltas(
                mean_delta_vec = delta_p_mean,
                var_delta_vec  = var_vec,
                n_cells        = n_p,
                var_scale      = scell_var_scale,
                rng            = rng,
            )  # (n_p, G)
            X_syn = X_base - sampled_deltas
        else:
            # old deterministic behavior
            X_syn = X_base - delta_p_mean[None, :]

        if scell_clip_zero:
            np.maximum(X_syn, 0.0, out=X_syn)       # clamp

        # clone obs rows from sampled controls but set pert label to p
        obs_p = ctrl_obs.iloc[idx].copy()
        obs_p[target_label] = p
        X_rows.append(X_syn)
        obs_rows.append(obs_p)

    # Concatenate
    X_out = np.vstack(X_rows) if len(X_rows) else np.zeros((0, G), dtype=float)
    obs_out = pd.concat(obs_rows, axis=0) if len(obs_rows) else adata_test_orig.obs.iloc[:0].copy()
    obs_out = obs_out.loc[:, ~obs_out.columns.duplicated(keep="first")]
    # Build AnnData and write
    ad_out = ad.AnnData(X_out, obs=obs_out, var=var_df.copy())
    ad_out.layers = {}  # keep minimal; add if you want raw etc.
    print(f"[cells] Writing synthesized test predictions: {out_path} "
          f"(controls={ctrl_mask.sum()}, synthesized={X_out.shape[0] - ctrl_mask.sum()}, genes={G})")
    ad_out.write_h5ad(out_path)

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
    true_bulk_mat = np.stack([true_bulk[p] - ctrl_mean for p in perts], axis=0)  # (K,G)
    pred_bulk_mat = np.stack([pred_bulk[p] - ctrl_mean for p in perts], axis=0)  # (K,G)
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
    adata_train, adata_test, adata_test_orig = train_test_split(args, adata_target)
    if args.test_h5ad != "":  # if external test set provided, merge into overall target dataset
        tmp = adata_test[adata_test.obs[args.target_label] != args.control_label].copy()
        adata_target = ad.concat([adata_train, tmp])
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
    pred_krr_ev, _true_krr_ev, names_krr_ev, _ctrl_ignored, pred_delta_var_ev = krr_predict_from_external(
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
        conf_boost_alpha=args.conf_boost_alpha,
        conf_shrink_alpha=args.conf_shrink_alpha,
        conf_min_var=args.conf_min_var,
        conf_max_var=args.conf_max_var,
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

    # Optional PDS sharpening in effect space (pred → Δ → sharpen → pred)
    if args.pds_sharpen != "none" and pred_ev.shape[0] > 0:
        pred_ev = sharpen_effects(
            pred_mat=pred_ev, ctrl_mean=ctrl_mean_global, mode=args.pds_sharpen,
            gamma=args.pds_gamma, topk_frac=args.pds_topk_frac,
            alpha=args.pds_alpha, beta=args.pds_beta,
            sigmoid_B=args.pds_sigmoid_B, preserve_q=args.pds_preserve_quantile
        )

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
        pred_krr_tr, _true_krr_tr, names_krr_tr, _ctrl_ignored, _pred_delta_var_tr = krr_predict_from_external(
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
            conf_boost_alpha=args.conf_boost_alpha,
            conf_shrink_alpha=args.conf_shrink_alpha,
            conf_min_var=args.conf_min_var,
            conf_max_var=args.conf_max_var,
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

        # Optional sharpening for train split too (so metrics are consistent if you eval on train)
        if args.pds_sharpen != "none" and pred_tr.shape[0] > 0:
            pred_tr = sharpen_effects(
                pred_mat=pred_tr, ctrl_mean=ctrl_mean_global, mode=args.pds_sharpen,
                gamma=args.pds_gamma, topk_frac=args.pds_topk_frac,
                alpha=args.pds_alpha, beta=args.pds_beta,
                sigmoid_B=args.pds_sigmoid_B, preserve_q=args.pds_preserve_quantile
            )

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

    # single cell predictions and output
    if (args.test_predict_out is not None and args.test_predict_out != "") and (adata_test_orig is not None):
        # Use the same evaluation gene space that pred_ev uses
        eval_adata_for_eval = eval_adata_int if args.intersect_genes else eval_adata
        eval_gene_names = eval_adata_for_eval.var_names

        write_cell_level_predictions(
            adata_test_orig=adata_test_orig,
            eval_gene_names=eval_gene_names,
            pred_mat_eval=pred_ev,
            pred_delta_var_eval=pred_delta_var_ev,
            names_eval=names_ev,
            ctrl_mean_eval=ctrl_mean_global,
            target_label=args.target_label,
            control_label=args.control_label,
            out_path=args.test_predict_out,
            scell_use_var=args.scell_use_var,
            scell_var_scale=args.scell_var_scale,
            scell_clip_zero=args.scell_clip_zero,
            random_state=getattr(args, "seed", None),
        )

    print("\n✨ Done!")

if __name__ == "__main__":
    main()
