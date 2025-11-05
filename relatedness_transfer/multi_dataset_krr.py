#!/usr/bin/env python3
import warnings
# Suppress annoying FutureWarning from scanpy
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse, math, os
import pickle as pkl
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
from sklearn.metrics import pairwise_distances

from utils import *
from losses import *
from transforms import *
from load_pathways import load_pathway_sources, make_pathway_matrix  # YAML + per-source matrix loaders


def parse_arguments():
    ap = argparse.ArgumentParser(description="Multi dataset KRR for relatedness transfer.")
    # Basic and I/O options
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--external_h5ad", default="", help="Path to the external pseudobulked AnnData object.")
    ap.add_argument("--external_list", type=str, default="",
                    help="Optional text file with one external .h5ad path per line. If set, overrides --external_h5ad.")
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
    ap.add_argument("--external_as_tsv_deltas", action="store_true",
                    help="If set, entries in --external_list are TSV files with genes as rows and perts as columns; values are already deltas..")
    ap.add_argument("--run_diagnostics", action="store_true", help="Run diagnostic analyses on kernel relatedness.")

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
    ap.add_argument("--kernel_agg", type=str, choices=["mean", "max", "wmean", "pwmean"], default="mean",
                    help="How to aggregate per-dataset kernels when --external_list is used.")
    ap.add_argument("--kernel_weight_gamma", type=float, default=1.0,
                    help="Exponent gamma for global reliability weights (w_i ∝ score_i^gamma).")
    ap.add_argument("--pw_topk", type=int, default=10,
                    help="Top-k neighbors in O used to transfer per-pert weights to U.")
    ap.add_argument("--pw_pair_rule", type=str, choices=["geom", "min"], default="geom",
                    help="Combine per-pert weights into pair weights: geometric mean or min.")
    ap.add_argument("--pw_gamma", type=float, default=1.0,
                    help="Exponent gamma for per-pert reliability scores before normalization.")
    ap.add_argument("--pw_floor", type=float, default=0.0,
                    help="Small floor added to per-pert weights before normalization to avoid brittle zeros.")
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
    
    # arguments for using gene embeddings / pathway info
    ap.add_argument("--embeddings_yaml", type=str, default="",
                    help="YAML config with embedding sources (each entry has file/gene_col/pathway_col/format).")
    ap.add_argument("--emb_metric", type=str, choices=["cosine", "corr", "rbf"], default="cosine",
                    help="Similarity for embedding sources: cosine (default), Pearson corr, or RBF.")
    ap.add_argument("--emb_pca_dim", type=int, default=0,
                    help="Optional PCA on embedding features before similarity (0=off).")
    ap.add_argument("--emb_rbf_gamma", type=float, default=0.0,
                    help="RBF gamma; if 0, use median heuristic.")

    # low rank mode
    ap.add_argument("--lowrank_pca_r", type=int, default=0,
                    help="If >0, run KRR in a rank-r PCA space learned from target TRAIN deltas, then reconstruct.")
    ap.add_argument("--lowrank_gene_zscore", action="store_true",
                    help="Z-score genes across TRAIN perts before PCA (undo after reconstruction).")
    ap.add_argument("--kernel_pc_space_r", type=int, default=0,
                    help="If >0, build per-dataset kernels in the target TRAIN PCA space of rank r before aggregation.")
    ap.add_argument("--kernel_pc_gene_zscore", action="store_true",
                    help="Z-score genes across TRAIN perts when fitting PCA (apply same transform to externals before projection).")

    args = ap.parse_args()
    return args

def create_kernel_from_tsv_deltas(
    fname: str,
    target_perts: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Read a TSV where rows=genes, columns=perts, values=delta (already computed).
    Intersect columns with target_perts, compute perts×perts correlation via df.corr().
    Returns (K, perts) with K float32, symmetrized, diag=1.
    """
    df = pd.read_csv(fname, sep="\t", index_col=0)
    # columns are perts in this format
    cols = [str(c) for c in df.columns]
    keep = [p for p in target_perts if p in cols]
    if len(keep) < 2:
        return np.zeros((0, 0), dtype=np.float32), []
    K = df[keep].corr().to_numpy(dtype=np.float32)
    K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return K, keep

# ---------- 1) corr-of-corrs (per-pert) ----------
def report_corr_of_corrs(
    K_true: np.ndarray,
    perts_true: List[str],
    externals: List[Tuple[np.ndarray, List[str], str]],
    topn_per_pert: int = 5,
) -> None:
    """
    For each perturbation p (from perts_true), compute Pearson correlation between:
      - the true kernel row K_true[p, *] (excluding self),
      - each external's kernel row K_ext[p, *] aligned to the same perts (excluding self).
    Prints per-pert sorted scores to stdout.

    externals: list of (K_ext, perts_ext, name)
    """
    perts_true = list(perts_true)
    pos_true = {p: i for i, p in enumerate(perts_true)}
    P = len(perts_true)

    def _row_corr(x: np.ndarray, y: np.ndarray) -> float:
        # Pearson, robust to constants / NaNs
        m = np.isfinite(x) & np.isfinite(y)
        m_sum = m.sum()
        if m_sum < 3:
            return np.nan
        xa, ya = x[m], y[m]
        vx = xa.var()
        vy = ya.var()
        if vx <= 1e-12 or vy <= 1e-12:
            return np.nan
        return float(np.corrcoef(xa, ya)[0, 1])

    print("\n=== corr-of-corrs per perturbation (higher = better proxy) ===")
    for p in perts_true:
        i = pos_true[p]
        # true reference row excluding self
        x = K_true[i, :].astype(np.float64)
        x = np.delete(x, i)  # drop self
        labels_ref = perts_true[:i] + perts_true[i+1:]

        scores = []
        for K_ext, perts_ext, name in externals:
            # align row p to the same non-self column set as in the true vector
            if p not in perts_ext:
                scores.append((name, np.nan))
                continue
            pos_ext = {q: j for j, q in enumerate(perts_ext)}
            j = pos_ext[p]
            # collect ext row over labels_ref intersection
            cols = [pos_ext[q] for q in labels_ref if q in pos_ext]  # keep true order, skip missing
            if len(cols) < max(3, int(0.05 * len(labels_ref))):  # too little overlap -> skip
                scores.append((name, np.nan))
                continue
            y = K_ext[j, cols].astype(np.float64)
            # also cut x to those same perts
            m_idx = [k for k, q in enumerate(labels_ref) if q in pos_ext]
            x_cut = x[m_idx]
            cij = _row_corr(x_cut, y)
            scores.append((name, cij))

        # sort, print top-N
        scores_sorted = sorted(scores, key=lambda t: (-(t[1] if np.isfinite(t[1]) else -np.inf)))
        head = scores_sorted[:topn_per_pert]
        sline = ", ".join([f"{nm}:{cij:.3f}" if np.isfinite(cij) else f"{nm}:nan" for nm, cij in head])
        print(f"{p:>12s}  |  {sline}")

# ---------- 2) Manhattan distances (true pairwise; predicted vs true cross) ----------
def save_manhattan_distances(
    Y_true: np.ndarray,
    perts_true: List[str],
    out_true_csv: str = "true_pairwise_l1.csv",
    Y_pred: Optional[np.ndarray] = None,
    perts_pred: Optional[List[str]] = None,
    out_cross_csv: str = "pred_true_l1.csv",
) -> Tuple[str, Optional[str]]:
    """
    Saves two CSVs:
      - out_true_csv: P_true x P_true Manhattan (L1) distances between rows of Y_true (aligned to perts_true).
      - out_cross_csv: if Y_pred is provided, P_pred x P_true L1 distances between Y_pred rows and Y_true rows
                       over the INTERSECTION of perts (by name) and the SAME gene set (assumed already aligned).

    Returns paths to the written CSVs (second may be None).
    """
    perts_true = list(perts_true)
    # 2a) pairwise L1 on true
    # Efficient: ||x - y||_1 = sum |x| + sum |y| - 2*sum min(x,y)  (but absolute is fine with vectorization)
    Yt = np.asarray(Y_true, dtype=np.float32)
    P_true = Yt.shape[0]
    # Broadcasted absolute differences: might be large; do it in blocks if needed.
    block = max(1, 4096 // max(1, Yt.shape[1]))  # simple heuristic to limit memory
    D_true = np.zeros((P_true, P_true), dtype=np.float32)
    for start in range(0, P_true, block):
        end = min(P_true, start + block)
        # (end-start, 1, G) vs (1, P_true, G) -> (end-start, P_true, G)
        diffs = np.abs(Yt[start:end, None, :] - Yt[None, :, :])
        D_true[start:end, :] = diffs.sum(axis=2)
    df_true = pd.DataFrame(D_true, index=perts_true, columns=perts_true)
    df_true.to_csv(out_true_csv)

    # 2b) cross L1: predicted (rows) vs true (cols)
    out2 = None
    if Y_pred is not None and perts_pred is not None:
        perts_pred = list(perts_pred)
        # align to intersection of perts by name (keep true order on columns, pred order on rows)
        set_true = set(perts_true)
        rows_keep = [i for i, p in enumerate(perts_pred) if p in set_true]
        cols_keep = [j for j, p in enumerate(perts_true) if p in set(perts_pred)]
        if rows_keep and cols_keep:
            Yp = np.asarray(Y_pred, dtype=np.float32)[rows_keep, :]
            Yt_sub = Yt[cols_keep, :]
            # Compute cross distances (P_pred_int x P_true_int)
            block_r = max(1, 4096 // max(1, Yp.shape[1]))
            D_cross = np.zeros((Yp.shape[0], Yt_sub.shape[0]), dtype=np.float32)
            for start in range(0, Yp.shape[0], block_r):
                end = min(Yp.shape[0], start + block_r)
                diffs = np.abs(Yp[start:end, None, :] - Yt_sub[None, :, :])
                D_cross[start:end, :] = diffs.sum(axis=2)
            row_names = [perts_pred[i] for i in rows_keep]
            col_names = [perts_true[j] for j in cols_keep]
            df_cross = pd.DataFrame(D_cross, index=row_names, columns=col_names)
            df_cross.to_csv(out_cross_csv)
            out2 = out_cross_csv
        else:
            print("[save_manhattan_distances] Warning: no shared perts between predicted and true; skipping cross CSV.")

    print(f"[save_manhattan_distances] wrote: {out_true_csv}" + (f", {out2}" if out2 else ""))
    return out_true_csv, out2


def create_kernel_pc_space(
    adata_panel: ad.AnnData,
    *,
    target_label: str,
    control_label: str,
    W_r: np.ndarray,           # (G, r) PCA loadings from target TRAIN
    col_mean: np.ndarray,      # (G,) TRAIN per-gene mean
    col_std: np.ndarray,       # (G,) TRAIN per-gene std (ones if not zscored)
    genes_target: list[str],   # gene order used to fit W_r / col_mean / col_std
    metric: str = "corr",      # "corr" or "cosine"
) -> tuple[np.ndarray, list[str]]:
    """
    Project panel deltas into target TRAIN PCA space and compute a similarity kernel
    between perturbations in that space.
    Returns: (K_pc, perts_list)
    """
    # 1) deltas in panel gene space
    deltas_panel, _ = compute_deltas(adata_panel, target_label, control_label)  # {pert: (G_panel,)}
    perts = [p for p in adata_panel.obs[target_label].astype(str).unique().tolist()
             if p != control_label and p in deltas_panel]
    if not perts:
        return np.zeros((0, 0), dtype=np.float32), []

    # 2) align panel genes -> target PCA gene order (subset on intersection)
    genes_panel = adata_panel.var_names.astype(str).tolist()
    pos_t = {g: i for i, g in enumerate(genes_target)}
    pos_p = {g: i for i, g in enumerate(genes_panel)}
    common = [g for g in genes_target if g in pos_p]          # preserve target order
    if len(common) < 10:
        # not enough overlap to build a meaningful PC kernel
        return np.zeros((0, 0), dtype=np.float32), []

    Jt = np.array([pos_t[g] for g in common], dtype=int)
    Jp = np.array([pos_p[g] for g in common], dtype=int)

    # slice the stats/loadings to common genes
    Wc = W_r[Jt, :]                  # (G_common, r)
    mc = col_mean[Jt]                # (G_common,)
    sc = col_std[Jt]                 # (G_common,)

    # 3) stack panel deltas in common-gene order
    Yp = np.stack([np.asarray(deltas_panel[p]).ravel()[Jp] for p in perts], axis=0).astype(np.float32)  # (P, Gc)

    # 4) apply target TRAIN centering / (optional) z-scoring
    Yc = Yp - mc[None, :]
    sc_safe = np.where(sc > 1e-8, sc, 1.0).astype(np.float32)
    Yc = Yc / sc_safe[None, :]

    # 5) project to PC space and build similarity
    T = (Yc @ Wc).astype(np.float32)   # (P, r)
    if metric == "corr":
        # corr between rows of T
        K = np.corrcoef(T)
    elif metric == "cosine":
        from sklearn.metrics import pairwise_distances
        K = 1.0 - pairwise_distances(T, metric="cosine")
    else:
        raise ValueError("metric must be 'corr' or 'cosine'")
    K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return K, perts

def _build_pca_basis_from_train(
    adata_train_target: ad.AnnData,
    target_label: str,
    control_label: str,
    r: int,
    gene_zscore: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      W_r : (G, r) PCA loadings with orthonormal columns (approx; TruncatedSVD components^T)
      T_O : (|O|, r) component scores for TRAIN perts (Y_O projected onto W_r)
      col_mean : (G,) per-gene mean across TRAIN perts (added back later if gene_zscore=False)
      col_std  : (G,) per-gene std across TRAIN perts (if gene_zscore=True; used to unscale)
      perts_O_order : list[str] TRAIN perts used (order matches T_O rows)
    """
    # collect TRAIN deltas as (|O|, G)
    deltas_O, _ = compute_deltas(adata_train_target, target_label, control_label)
    perts_O_order = [p for p in adata_train_target.obs[target_label].astype(str).unique().tolist()
                     if p != control_label and p in deltas_O]
    if len(perts_O_order) == 0:
        raise ValueError("No TRAIN deltas found to build PCA basis.")
    Y_O = np.stack([np.asarray(deltas_O[p]).ravel() for p in perts_O_order], axis=0).astype(np.float32)  # (|O|, G)

    # center and (optionally) z-score genes across TRAIN perts
    col_mean = Y_O.mean(axis=0, dtype=np.float64).astype(np.float32)
    Yc = Y_O - col_mean[None, :]
    if gene_zscore:
        col_std = Yc.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
        col_std = np.where(col_std > 1e-8, col_std, 1.0)
        Yc = Yc / col_std[None, :]
    else:
        col_std = np.ones_like(col_mean, dtype=np.float32)

    r_eff = max(1, min(r, Yc.shape[0], Yc.shape[1]))
    # fast truncated SVD (no full covariance). components_: (r, G)
    try:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=r_eff, random_state=0)
        T_O = svd.fit_transform(Yc).astype(np.float32)              # (|O|, r)
        W_r = svd.components_.T.astype(np.float32, copy=True)       # (G, r)
    except Exception:
        # fallback to dense SVD
        U, S, VT = np.linalg.svd(Yc, full_matrices=False)
        T_O = (U[:, :r_eff] * S[:r_eff]).astype(np.float32)         # (|O|, r)
        W_r = VT[:r_eff, :].T.astype(np.float32)                    # (G, r)

    return W_r, T_O, col_mean, col_std, perts_O_order

def create_kernel(
    adata_source_int: ad.AnnData,
    adata_train_target: ad.AnnData,
    target_label: str,
    control_label: str,
    kernel_metric: str = "corr",     # {"corr","cosine"}
    iso_calibrate: bool = False,     # match current flag semantics
    eps: float = 1e-6,               # tiny PSD nudge
) -> tuple[np.ndarray, list[str]]:
    """
    Build a perturbation kernel from ONE external dataset, with optional isotonic calibration
    using TARGET train deltas (on the overlapping training perts).

    Returns:
        K_src  : (P_i x P_i) numpy array (float32), symmetric with diag=1
        perts  : List[str] of NON-control perturbations in this external (row/col order of K_src)
    """
    # --- collect perts in this external (exclude control) ---
    perts_src_all = list(map(str, adata_source_int.obs[target_label].values))
    perts = [p for p in dict.fromkeys(perts_src_all) if p != control_label]  # stable unique, no control
    if len(perts) == 0:
        # return a degenerate 1x1 kernel if nothing to contribute; caller can ignore it
        return np.ones((0, 0), dtype=np.float32), []

    # --- compute SOURCE deltas matrix: rows=perts, cols=genes ---
    deltas_src_dict, _ = compute_deltas(adata_source_int, target_label, control_label)
    # Keep only perts that truly exist in this source's delta dict (paranoia)
    perts = [p for p in perts if p in deltas_src_dict]
    D_src = np.stack([np.asarray(deltas_src_dict[p]).ravel() for p in perts], axis=0).astype(np.float32)

    # --- similarity on source deltas ---
    if kernel_metric == "corr":
        # Pearson correlation between rows
        # np.corrcoef expects rows as variables if rowvar=True (default)
        S_src = np.corrcoef(D_src)
        # numerical guard: NaNs can happen if a row is constant
        S_src = np.nan_to_num(S_src, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    elif kernel_metric == "cosine":
        # similarity = 1 - cosine distance
        S_src = 1.0 - pairwise_distances(D_src, metric="cosine")
        S_src = np.nan_to_num(S_src, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported kernel_metric={kernel_metric!r}. Use 'corr' or 'cosine'.")

    # --- optional isotonic calibration on O×O overlap (training perts only) ---
    if iso_calibrate:
        # Training perts present in BOTH this external and the target TRAIN split
        perts_O = [p for p in pert_list(adata_train_target, target_label, control_label) if p in set(perts)]
        if len(perts_O) >= 4:
            # target deltas for O (rows aligned to perts_O) -> target similarity on O×O
            deltas_tgt_O, _ = compute_deltas(adata_train_target, target_label, control_label)
            D_tgt_O = np.stack([np.asarray(deltas_tgt_O[p]).ravel() for p in perts_O], axis=0).astype(np.float32)
            if kernel_metric == "corr":
                S_tgt_OO = np.corrcoef(D_tgt_O)
            else:
                S_tgt_OO = 1.0 - pairwise_distances(D_tgt_O, metric="cosine")
            S_tgt_OO = np.nan_to_num(S_tgt_OO, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # map perts_O to indices in the source kernel
            idx_src = {p: i for i, p in enumerate(perts)}
            iO = np.array([idx_src[p] for p in perts_O], dtype=int)
            S_src_OO = S_src[np.ix_(iO, iO)]

            # fit an increasing isotonic map: src_sim -> tgt_sim, using upper triangle (i<j)
            iu = np.triu_indices(len(perts_O), k=1)
            x = S_src_OO[iu].ravel()
            y = S_tgt_OO[iu].ravel()
            # clamp to [-1,1] for safety
            x = np.clip(x, -1.0, 1.0)
            y = np.clip(y, -1.0, 1.0)

            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and x.size >= 8:
                iso = IsotonicRegression(y_min=-1.0, y_max=1.0, increasing=True, out_of_bounds="clip")
                iso.fit(x, y)
                # apply to the whole matrix
                S_src = iso.predict(np.clip(S_src, -1.0, 1.0).reshape(-1)).reshape(S_src.shape).astype(np.float32)
            # else: quietly keep S_src as-is (too few/ill-conditioned pairs)

    # --- symmetrize, set diag=1, tiny PSD nudge ---
    S_src = 0.5 * (S_src + S_src.T)
    np.fill_diagonal(S_src, 1.0)
    S_src = S_src + eps * np.eye(Src := S_src.shape[0], dtype=np.float32)

    return S_src.astype(np.float32), perts

def _row_l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / nrm

def _median_heuristic_gamma(X: np.ndarray) -> float:
    # sample up to 4096 rows for a quick median distance
    rng = np.random.default_rng(0)
    n = X.shape[0]
    take = min(n, 4096)
    idx = rng.choice(n, size=take, replace=False) if n > take else np.arange(n)
    Xa = X[idx]
    # pairwise squared distances
    d2 = ((Xa[:, None, :] - Xa[None, :, :]) ** 2).sum(axis=2)
    # take upper triangle median
    iu = np.triu_indices(d2.shape[0], k=1)
    med = np.median(d2[iu]) if iu[0].size > 0 else 1.0
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    # gamma = 1/(2*sigma^2) with sigma^2 = med
    return float(1.0 / (2.0 * med))

def create_embedding_kernel_from_df(
    df_gene_by_feat: "pd.DataFrame",
    perts_candidates: list[str],
    metric: str = "cosine",
    pca_dim: int = 0,
    rbf_gamma: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a pert-pert kernel from a gene-by-feature matrix (rows=genes).
    Returns (K, perts_present_in_df).
    """
    # intersect available perts (gene symbols) with df index
    genes_in_df = set(map(str, df_gene_by_feat.index))
    perts = [p for p in perts_candidates if p in genes_in_df]
    if len(perts) < 2:
        return np.zeros((0, 0), dtype=np.float32), []

    X = df_gene_by_feat.loc[perts].to_numpy(dtype=np.float32, copy=True)  # (P, F)

    # optional PCA for stability / denoising
    if pca_dim and pca_dim > 0 and X.shape[1] > pca_dim:
        try:
            from sklearn.decomposition import PCA
            X = PCA(n_components=pca_dim, random_state=0).fit_transform(X)
        except Exception:
            # keep original X if sklearn not available
            pass

    metric = str(metric).lower()
    if metric == "cosine":
        Xn = _row_l2_normalize(X)
        K = Xn @ Xn.T
    elif metric == "corr":
        Xm = X - X.mean(axis=1, keepdims=True)
        denom = np.sqrt((Xm ** 2).sum(axis=1, keepdims=True)) + 1e-8
        Xz = Xm / denom
        K = Xz @ Xz.T
    elif metric == "rbf":
        if rbf_gamma and rbf_gamma > 0.0:
            gamma = float(rbf_gamma)
        else:
            gamma = _median_heuristic_gamma(X)
        # pairwise squared distances
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        K = np.exp(-gamma * d2, dtype=np.float32)
    else:
        raise ValueError(f"Unknown emb_metric {metric}")

    # symmetrize, diag=1, tiny ridge
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    K = K + 1e-6 * np.eye(K.shape[0], dtype=np.float32)
    return K.astype(np.float32, copy=False), perts

def build_embedding_kernels_from_yaml(
    embeddings_yaml: str,
    var_names_target: list[str],
    perts_O: list[str],
    perts_U: list[str],
    emb_metric: str = "cosine",
    emb_pca_dim: int = 0,
    emb_rbf_gamma: float = 0.0,
) -> list[tuple[np.ndarray, list[str]]]:
    """
    Parse YAML of embedding sources, materialize gene-by-feature matrices using your loader,
    and convert each into a (K_i, perts_i) kernel to append to kernels_and_perts.
    """
    if not embeddings_yaml:
        return []

    # 1) YAML → dict of sources
    #    Required keys per entry: file, gene_col, pathway_col, format
    sources = load_pathway_sources(embeddings_yaml)  # dict[name] -> meta
    if not sources:
        return []

    kernels = []
    perts_candidates = list(dict.fromkeys(list(perts_O) + list(perts_U)))  # O∪U order-preserving

    for name, meta in sources.items():
        file_name   = meta["file"]
        gene_col    = meta["gene_col"]
        pathway_col = meta["pathway_col"]
        fmt         = meta["format"]  # "tsv" or "presage"

        # 2) Make pathway/feature matrix aligned to target gene set
        #    Returns a DataFrame with rows=genes, cols=features (the helpers handle TSV/PRESAGE shapes). 
        #    - TSV: parses, de-URLs, pivots, fills missing genes, reorders to var_names, returns .T
        #    - PRESAGE: loads pickle, transposes, intersects + fills, reorders, returns .T
        df_gene_by_feat = make_pathway_matrix(
            file_name=file_name,
            gene_col=gene_col,
            pathway_col=pathway_col,
            format=fmt,
            var_names=list(map(str, var_names_target)),
        )

        # 3) Build (K_i, perts_i) for this source over perts present in df
        K_i, perts_i = create_embedding_kernel_from_df(
            df_gene_by_feat=df_gene_by_feat,
            perts_candidates=perts_candidates,
            metric=emb_metric,
            pca_dim=emb_pca_dim,
            rbf_gamma=emb_rbf_gamma,
        )
        if K_i.size and len(perts_i):
            kernels.append((K_i, perts_i))
            # (optional) print a tiny summary for visibility
            print(f"[emb] source '{name}': {K_i.shape} over {len(perts_i)} perts "
                  f"(metric={emb_metric}, pca_dim={emb_pca_dim})")

    return kernels

def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a symmetric kernel."""
    n = K.shape[0]
    one = np.ones((n, 1), dtype=K.dtype) / n
    row_mean = K @ one
    col_mean = (one.T @ K).T
    grand = (one.T @ K @ one)[0, 0]
    return K - row_mean - col_mean + grand


def _kernel_alignment(K: np.ndarray, S: np.ndarray) -> float:
    """
    Centered kernel alignment (a.k.a. HSIC normalized, aka KTA):
    <Kc, Sc>_F / (||Kc||_F * ||Sc||_F)
    """
    Kc = _center_kernel(K)
    Sc = _center_kernel(S)
    num = np.sum(Kc * Sc)
    den = np.linalg.norm(Kc, ord="fro") * np.linalg.norm(Sc, ord="fro")
    if den <= 0 or not np.isfinite(den):
        return 0.0
    val = float(num / den)
    # clip tiny numeric drift
    if not np.isfinite(val):
        return 0.0
    return val


def compute_global_kernel_weights(
    kernels_and_perts: list[tuple[np.ndarray, list[str]]],
    adata_train_target: ad.AnnData,
    target_label: str,
    control_label: str,
    perts_O: list[str],
    gamma: float = 1.0,
    min_common: int = 4,
) -> tuple[list[float], list[float]]:
    """
    Compute one reliability weight per dataset using alignment of K_i^{OO} to the
    target train similarity S_tgt^{OO}. Returns (weights, raw_scores) where
    weights are normalized to sum to 1. If all scores are 0, falls back to uniform.
    """
    # Build target train similarity on O×O from target deltas (corr)
    deltas_tgt_O, _ = compute_deltas(adata_train_target, target_label, control_label)
    # Keep only perts we actually have in training
    perts_O = [p for p in perts_O if p in deltas_tgt_O]
    if len(perts_O) < min_common:
        # degenerate: not enough training perts
        return [1.0 / max(1, len(kernels_and_perts))] * len(kernels_and_perts), [0.0] * len(kernels_and_perts)

    D_tgt_O = np.stack([np.asarray(deltas_tgt_O[p]).ravel() for p in perts_O], axis=0).astype(np.float32)
    # Pearson corr similarity on O×O
    S_tgt = np.corrcoef(D_tgt_O)
    S_tgt = np.nan_to_num(S_tgt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    scores = []
    for (K_i, perts_i) in kernels_and_perts:
        # intersect perts of this dataset with O
        common = [p for p in perts_O if p in set(perts_i)]
        if len(common) < min_common:
            scores.append(0.0)
            continue
        # align K_i to the same ordering
        idx_map = {p: j for j, p in enumerate(perts_i)}
        I = np.array([idx_map[p] for p in common], dtype=int)
        K_OO = K_i[np.ix_(I, I)].astype(np.float32)
        # Build target similarity on the same subset order
        j_map = {p: j for j, p in enumerate(perts_O)}
        J = np.array([j_map[p] for p in common], dtype=int)
        S_sub = S_tgt[np.ix_(J, J)]
        s = _kernel_alignment(K_OO, S_sub)
        # non-negative score; negative alignment shouldn’t get weight
        scores.append(max(0.0, s))

    # turn into weights with exponent gamma
    if np.allclose(scores, 0.0):
        w = [1.0 / len(kernels_and_perts)] * len(kernels_and_perts)
        return w, scores
    sc = np.array(scores, dtype=np.float64)
    sc = np.power(sc, max(0.0, float(gamma)))
    sc_sum = sc.sum()
    w = (sc / sc_sum).tolist()
    return w, scores

def _row_corr2(a: np.ndarray, b: np.ndarray) -> float:
    """Return squared Pearson correlation between two 1D arrays, guarding NaNs."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    am = a - a.mean()
    bm = b - b.mean()
    denom = float(np.linalg.norm(am) * np.linalg.norm(bm))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    r = float((am @ bm) / denom)
    if not np.isfinite(r):
        return 0.0
    r2 = r * r
    return max(0.0, r2)


def compute_per_pert_weights_on_O(
    kernels_and_perts: list[tuple[np.ndarray, list[str]]],
    adata_train_target: ad.AnnData,
    target_label: str,
    control_label: str,
    perts_O: list[str],
    gamma: float = 1.0,
    floor: float = 0.0,
    min_common: int = 4,
) -> list[dict[str, float]]:
    """
    For each dataset i, compute a dict w_i(p) for p∈O, from corr^2 between K_i[p, O∩P_i]
    and S_tgt[p, same subset]. Then, for each p, normalize weights across datasets
    that contain p (apply floor and exponent gamma before normalization).
    Returns: list of length M, each is {pert -> weight} for p∈O∩P_i.
    """
    # Build target similarity S_tgt over O using deltas (corr)
    deltas_tgt_O, _ = compute_deltas(adata_train_target, target_label, control_label)
    perts_O = [p for p in perts_O if p in deltas_tgt_O]
    if len(perts_O) < min_common:
        return [{p: 1.0 for p in perts_O}] * len(kernels_and_perts)

    D_tgt_O = np.stack([np.asarray(deltas_tgt_O[p]).ravel() for p in perts_O], axis=0).astype(np.float32)
    S_tgt = np.corrcoef(D_tgt_O)
    S_tgt = np.nan_to_num(S_tgt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pos = {p: j for j, p in enumerate(perts_O)}

    # raw scores r_i(p)
    raw_scores_per_ds: list[dict[str, float]] = []
    for (K_i, perts_i) in kernels_and_perts:
        avail = set(perts_i)
        idx_i = {p: j for j, p in enumerate(perts_i)}
        scores: dict[str, float] = {}
        for p in perts_O:
            if p not in avail:
                continue
            # common O perts this dataset has (including p)
            common = [q for q in perts_O if q in avail]
            if len(common) < min_common:
                continue
            I = np.array([idx_i[q] for q in common], dtype=int)
            J = np.array([pos[q] for q in common], dtype=int)
            row_i = K_i[idx_i[p], I]
            row_t = S_tgt[pos[p], J]
            r2 = _row_corr2(row_i, row_t)
            scores[p] = r2
        raw_scores_per_ds.append(scores)

    # normalize per-pert across datasets that contain that pert
    M = len(kernels_and_perts)
    w_per_ds: list[dict[str, float]] = [dict() for _ in range(M)]
    for p in perts_O:
        vals = []
        which = []
        for i in range(M):
            if p in raw_scores_per_ds[i]:
                vals.append(max(0.0, raw_scores_per_ds[i][p]))
                which.append(i)
        if not which:
            continue
        vals = np.asarray(vals, dtype=np.float64)
        vals = np.power(vals + float(floor), max(0.0, float(gamma)))
        s = float(vals.sum())
        if s <= 0 or not np.isfinite(s):
            vals = np.ones_like(vals) / len(vals)
        else:
            vals = vals / s
        for v, i in zip(vals.tolist(), which):
            w_per_ds[i][p] = float(v)
    return w_per_ds


def estimate_weights_for_U_by_neighbors(
    perts_U: list[str],
    perts_O: list[str],
    base_kernel: np.ndarray,
    base_perts: list[str],
    w_per_ds_O: list[dict[str, float]],
    topk: int = 10,
) -> list[dict[str, float]]:
    """
    For each u∈U, estimate w_i(u) by averaging weights of top-k neighbors in O,
    neighbors chosen using base_kernel[u, O] in base_perts ordering.
    Returns: list of dicts per dataset i with entries for u that exist in base_perts.
    """
    idx = {p: j for j, p in enumerate(base_perts)}
    O_in_base = [p for p in perts_O if p in idx]
    if not O_in_base:
        # degenerate: return empty dicts (caller should fall back gracefully)
        return [dict() for _ in w_per_ds_O]

    jO = np.array([idx[p] for p in O_in_base], dtype=int)
    M = len(w_per_ds_O)
    w_per_ds_U: list[dict[str, float]] = [dict() for _ in range(M)]

    for u in perts_U:
        if u not in idx:
            continue
        ju = idx[u]
        sims = base_kernel[ju, jO].astype(np.float32)
        # keep positive sims only (avoid noisy negatives)
        sims = np.where(np.isfinite(sims), sims, 0.0)
        sims[sims < 0] = 0.0
        if sims.sum() <= 0:
            continue
        # pick top-k
        if topk > 0 and topk < sims.size:
            top_idx = np.argpartition(sims, -topk)[-topk:]
            sims_k = sims[top_idx]
            O_k = [O_in_base[t] for t in top_idx]
        else:
            sims_k = sims
            O_k = O_in_base
        sw = float(sims_k.sum())
        if sw <= 0:
            continue
        weights_O_norm = (sims_k / sw).tolist()
        # transfer average per-pert weights
        for i in range(M):
            acc = 0.0
            for q, alpha in zip(O_k, weights_O_norm):
                acc += alpha * float(w_per_ds_O[i].get(q, 0.0))
            if acc > 0:
                w_per_ds_U[i][u] = acc
    return w_per_ds_U


def aggregate_kernels_pwmean(
    kernels_and_perts: list[tuple[np.ndarray, list[str]]],
    perts_union: list[str],
    w_per_ds_O: list[dict[str, float]],
    w_per_ds_U: list[dict[str, float]],
    pair_rule: str = "geom",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Per-perturbation weighted mean aggregation over union perts.
    For each dataset i, we prepare a per-pert weight vector W_i over union P,
    then form pair weights W_i(p,q) via sqrt(W_i(p)W_i(q)) or min rule.
    Coverage-aware normalization per pair.
    """
    P = len(perts_union)
    index = {p: j for j, p in enumerate(perts_union)}
    M = len(kernels_and_perts)

    # Build W (M x P): per-dataset weight per pert (0 if unknown)
    W = np.zeros((M, P), dtype=np.float32)
    for i in range(M):
        for p, w in w_per_ds_O[i].items():
            if p in index:
                W[i, index[p]] = max(0.0, float(w))
        for u, w in w_per_ds_U[i].items():
            if u in index:
                W[i, index[u]] = max(W[i, index[u]], max(0.0, float(w)))  # prefer positive estimate

    # Build per-dataset availability masks, and align kernels to union
    K_aligned = []
    A_masks = []  # per-pert availability (P,)
    for (K_i, perts_i) in kernels_and_perts:
        avail = np.zeros(P, dtype=bool)
        idx_i = {p: j for j, p in enumerate(perts_i)}
        sel = [index[p] for p in perts_i if p in index]
        A = np.full((P, P), np.nan, dtype=np.float32)
        if sel:
            J = np.array([idx_i[perts_union[s]] for s in sel], dtype=int)
            A[np.ix_(sel, sel)] = K_i[np.ix_(J, J)].astype(np.float32)
            avail[sel] = True
        K_aligned.append(A)
        A_masks.append(avail)

    # Aggregate
    num = np.zeros((P, P), dtype=np.float32)
    den = np.zeros((P, P), dtype=np.float32)
    for i in range(M):
        wi = W[i, :]  # (P,)
        if pair_rule == "geom":
            Wi = np.sqrt(np.maximum(0.0, wi)[:, None] * np.maximum(0.0, wi)[None, :])
        else:  # "min"
            Wi = np.minimum(wi[:, None], wi[None, :])
        mask = np.outer(A_masks[i], A_masks[i])  # pairs available in dataset i
        Ki = np.where(mask, K_aligned[i], np.nan)
        # coverage-aware weighted mean
        num += np.nan_to_num(Wi * Ki, nan=0.0)
        den += np.where(np.isnan(Ki), 0.0, Wi)

    with np.errstate(invalid="ignore", divide="ignore"):
        K_agg = num / den
    K_agg = np.nan_to_num(K_agg, nan=0.0, posinf=0.0, neginf=0.0)

    # finalize
    K_agg = 0.5 * (K_agg + K_agg.T)
    np.fill_diagonal(K_agg, 1.0)
    K_agg = K_agg + eps * np.eye(P, dtype=np.float32)
    return K_agg.astype(np.float32)

def aggregate_kernels(
    kernels_and_perts: list[tuple[np.ndarray, list[str]]],
    method: str = "mean",     
    eps: float = 1e-6,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate multiple per-dataset kernels into a single kernel over the union of perts.
    NaN-aware reduction:
      - If multiple datasets define a pair, reduce via mean/max.
      - If only one defines it, that value is used.
      - If none define it, the entry stays NaN (caller can decide to drop those perts).
    Returns:
        K_agg : (P_union x P_union) float32 kernel (diag=1, symmetrized, eps*I added)
        perts_union : row/col order
    """
    # filter out empties
    items = [(K, perts) for (K, perts) in kernels_and_perts if K is not None and len(perts) > 0]
    if len(items) == 0:
        return np.ones((0, 0), dtype=np.float32), []

    # stable union order: appearance order across inputs
    perts_union = []
    seen = set()
    for _, perts in items:
        for p in perts:
            if p not in seen:
                seen.add(p)
                perts_union.append(p)
    P = len(perts_union)
    index = {p: i for i, p in enumerate(perts_union)}

    # accumulator cube with NaNs
    stacks = []  # list of (P x P) arrays with NaNs where undefined
    for K_i, perts_i in items:
        A = np.full((P, P), np.nan, dtype=np.float32)
        idx_i = [index[p] for p in perts_i]
        A[np.ix_(idx_i, idx_i)] = K_i.astype(np.float32)
        stacks.append(A)

    S = np.stack(stacks, axis=0)  # (M, P, P)

    if method == "mean":
        K_agg = np.nanmean(S, axis=0)
    elif method == "max":
        K_agg = np.nanmax(S, axis=0)
    elif method == "wmean":
        if weights is None or len(weights) != len(items):
            raise ValueError("aggregate_kernels(method='wmean') requires weights with length == #datasets.")
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)  # (M,1,1)
        mask = ~np.isnan(S)      # (M,P,P)
        num = np.nansum(np.where(mask, w * S, 0.0), axis=0)  # (P,P)
        den = np.sum(np.where(mask, w, 0.0), axis=0)         # (P,P)
        with np.errstate(invalid="ignore", divide="ignore"):
            K_agg = num / den
    else:
        raise ValueError(f"Unknown aggregation method {method!r}; use 'mean' or 'max'.")

    # Any pairs never observed by any dataset remain NaN -> set to 0 (neutral-ish)
    K_agg = np.nan_to_num(K_agg, nan=0.0, posinf=0.0, neginf=0.0)

    # symmetrize, set diag=1, and nudge
    K_agg = 0.5 * (K_agg + K_agg.T)
    np.fill_diagonal(K_agg, 1.0)
    K_agg = K_agg + eps * np.eye(P, dtype=np.float32)

    return K_agg.astype(np.float32), perts_union

def krr_predict_from_external(
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    target_label: str,
    control_label: str,
    krr_lambda: float = 1e-2,
    ctrl_mean_target: np.ndarray | None = None,
    kernel_gamma: float = 1.0,
    topk: int = 0,
    boost_pcs: int = 0,
    boost_gamma: float = 0.6,
    conf_boost_alpha: float = 0.0,
    conf_shrink_alpha: float = 0.0,
    conf_min_var: float = 1e-6,
    conf_max_var: float = 1.0,
    K_full: np.ndarray = None,
    perts_all: List[str] = None,
    pca_r: int = 0,
    pca_gene_zscore: bool = False,
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

    # ---- define sets (KEEP kernel order; do not rebuild perts_all) ----
    if K_full is None or perts_all is None:
        raise ValueError("krr_predict_from_external requires a precomputed kernel: pass K_full and perts_all.")
    O_raw = pert_list(adata_train, target_label, control_label)  # observed perts
    U_raw = pert_list(adata_eval,  target_label, control_label)  # to predict/evaluate
    # map provided kernel order
    idx = {p: i for i, p in enumerate(perts_all)}
    # keep only perts present in the kernel (order preserved from O_raw / U_raw)
    O = [p for p in O_raw if p in idx]
    U = [p for p in U_raw if p in idx]
    if len(O) == 0 or len(U) == 0:
        raise ValueError("After aligning to kernel perts, O or U is empty. Check kernel construction / intersections.")
    iO = np.asarray([idx[p] for p in O], dtype=int)
    iU = np.asarray([idx[p] for p in U], dtype=int)

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

    # ---- use the provided aggregated kernel (over O∪U) ----
    if K_full is None or perts_all is None:
        raise ValueError("krr_predict_from_external now requires a precomputed kernel: pass K_full and perts_all.")
    K = K_full.astype(np.float32, copy=False)

    KOO = K[np.ix_(iO, iO)]
    KUO = K[np.ix_(iU, iO)]

    # ---- KRR solve & prediction (full rank or low-rank PCA) ----
    if pca_r and pca_r > 0:
        # Learn PCA on TRAIN deltas, project to component scores
        W_r, T_O, col_mean, col_std, perts_O_pca = _build_pca_basis_from_train(
            adata_train_target=adata_train,
            target_label=target_label,
            control_label=control_label,
            r=int(pca_r),
            gene_zscore=bool(pca_gene_zscore),
        )  # W_r:(G,r), T_O:(|O|,r)
        # Align T_O rows to the O-order used to form KOO/KUO
        if perts_O_pca != O:
            pos = {p:i for i,p in enumerate(perts_O_pca)}
            T_O = T_O[np.asarray([pos[p] for p in O], dtype=int), :]
        # Solve once for all components and predict scores for U
        A_comp = np.linalg.solve(KOO + krr_lambda * np.eye(KOO.shape[0], dtype=KOO.dtype), T_O)  # (|O|, r)
        KUO_sharp = sharpen_neighbors(KUO, tau=kernel_gamma, topk=topk)
        Z_U = KUO_sharp @ A_comp                                                                  # (|U|, r)
        # Reconstruct gene-space deltas for U
        Y_U_hat_centered = Z_U @ W_r.T                                                            # (|U|, G)
        if pca_gene_zscore:
            Y_U_hat_delta = (Y_U_hat_centered * col_std[None, :]) + col_mean[None, :]
        else:
            Y_U_hat_delta = Y_U_hat_centered + col_mean[None, :]
        # Training fit (for variance) in gene space
        Y_O_hat_centered = (KOO @ A_comp) @ W_r.T                                                 # (|O|, G)
        if pca_gene_zscore:
            Y_O_hat = (Y_O_hat_centered * col_std[None, :]) + col_mean[None, :]
        else:
            Y_O_hat = Y_O_hat_centered + col_mean[None, :]
        resid = Y_O - Y_O_hat                                                                      # (|O|, G)
        sigma2_gene = (resid ** 2).mean(axis=0)                                                    # (G,)
    else:
        # Original full-rank path on genes
        A = np.linalg.solve(KOO + krr_lambda * np.eye(KOO.shape[0], dtype=KOO.dtype), Y_O)         # (|O|, G)
        KUO_sharp = sharpen_neighbors(KUO, tau=kernel_gamma, topk=topk)
        Y_U_hat_delta = KUO_sharp @ A                                                              # (|U|, G)
        Y_O_hat = KOO @ A                                                                          # (|O|, G)
        resid = Y_O - Y_O_hat
        sigma2_gene = (resid ** 2).mean(axis=0)

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

    # ---- True expression rows and pert_names (just the U we actually predicted) ----
    # We return rows in the SAME order as 'U' (the eval perts aligned to kernel)
    # Build true_mat by averaging per pert from adata_eval (pseudobulk rows).
    rows = []
    for p in U:
        v = adata_eval[adata_eval.obs[target_label] == p].X
        rows.append(np.asarray(v).reshape(-1, G).mean(axis=0))
    true_mat = np.stack(rows, axis=0)
    pert_names = list(U)

    # ctrl_mean returned in the bundle = TRAIN control mean (global baseline)
    return pred_mat, true_mat, pert_names, np.asarray(ctrl_mean_target).ravel(), pred_delta_var

def write_cell_level_predictions(
    adata_test_orig: ad.AnnData,
    eval_gene_names,
    pred_mat_eval: np.ndarray,
    names_eval: list[str],
    ctrl_mean_eval: np.ndarray,
    target_label: str,
    control_label: str,
    out_path: str,
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
        delta_p = ctrl_mean_eval - pred_expr    # (G,)
        # sample control indices (with replacement if needed)
        replace = n_p > n_ctrl
        idx = rng.choice(n_ctrl, size=n_p, replace=replace)
        X_base = X_ctrl[idx]                    # (n_p, G)
        X_syn = X_base - delta_p[None, :]       # subtract learned delta
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

def collect_top_neighbors_per_u(
    *,
    adata_train_target: ad.AnnData,
    adata_eval_target: ad.AnnData,
    target_label: str,
    control_label: str,
    perts_U: list[str],
    kernels_and_perts: list[tuple[np.ndarray, list[str]]],
    dataset_names: list[str] | None = None,
    topk: int = 10,
    true_metric: str = "corr",   # similarity for TRUE (target) side: "corr" or "cosine"
) -> dict[str, dict]:
    """
    Build an inspection object for each test perturbation u in perts_U:
      {
        u: {
          "true_top":      [(pert, sim), ...],        # from target deltas vs TRAIN perts
          "datasets": {                               # per precomputed kernel (no recomputation)
              <ds_name>: [(pert, kernel_val), ...],
              ...
          }
        },
        ...
      }

    Notes:
      - Only TRAIN perts (O) are eligible as neighbors.
      - For 'true_top', we compute similarity between the eval delta of u and train deltas of O from the TARGET data.
      - For each kernel, we use the PRECOMPUTED kernel entries directly (no recomputation).
      - If a dataset lacks u or lacks a given train pert, it simply won't contribute to that list.
      - All lists are sorted descending by similarity and truncated to 'topk'.
    """
    # -------- 0) collect O and basic maps --------
    def _pert_list(adata):
        return [p for p in adata.obs[target_label].astype(str).unique().tolist() if p != control_label]

    perts_O = _pert_list(adata_train_target)
    perts_U = [str(p) for p in perts_U if p != control_label]
    if dataset_names is None:
        dataset_names = [f"ds{i}" for i in range(len(kernels_and_perts))]
    assert len(dataset_names) == len(kernels_and_perts), "dataset_names must match kernels_and_perts length"

    # -------- 1) TRUE similarities: y_u (from eval) vs Y_O (from train) --------
    # Compute deltas once
    deltas_O, _ = compute_deltas(adata_train_target, target_label, control_label)  # {pert -> (G,)}
    deltas_U, _ = compute_deltas(adata_eval_target,  target_label, control_label)  # {pert -> (G,)}

    # Build a matrix Y_O in a stable O order (only keep perts actually present)
    perts_O_present = [p for p in perts_O if p in deltas_O]
    if len(perts_O_present) == 0:
        raise ValueError("No training deltas available to compute 'true_top' neighbors.")

    Y_O = np.stack([np.asarray(deltas_O[p]).ravel() for p in perts_O_present], axis=0).astype(np.float32)  # (|O|, G)
    # Normalize once depending on metric
    if true_metric == "corr":
        # row-wise z-score
        Yo = Y_O - Y_O.mean(axis=1, keepdims=True)
        Yo /= (np.linalg.norm(Yo, axis=1, keepdims=True) + 1e-8)
    elif true_metric == "cosine":
        Yo = Y_O / (np.linalg.norm(Y_O, axis=1, keepdims=True) + 1e-8)
    else:
        raise ValueError("true_metric must be 'corr' or 'cosine'.")

    def _true_top_for_u(u: str) -> list[tuple[str, float]]:
        if u not in deltas_U:
            return []
        yu = np.asarray(deltas_U[u]).ravel().astype(np.float32)
        if true_metric == "corr":
            yu = yu - yu.mean()
            denom = np.linalg.norm(yu) + 1e-8
            yu = yu / denom
        else:  # cosine
            yu = yu / (np.linalg.norm(yu) + 1e-8)
        sims = (Yo @ yu).astype(np.float32)  # (|O|,)
        order = np.argsort(-sims)[:topk]
        return [(perts_O_present[i], float(sims[i])) for i in order]

    # -------- 2) Kernel-based neighbors per dataset (use precomputed values) --------
    # For each dataset, align its kernel to (u vs O) if available
    ds_index_maps = []
    for K_i, perts_i in kernels_and_perts:
        ds_index_maps.append({p: j for j, p in enumerate(perts_i)})

    def _kernel_top_for_u(ds_idx: int, u: str) -> list[tuple[str, float]]:
        K_i, perts_i = kernels_and_perts[ds_idx]
        idx = ds_index_maps[ds_idx]
        if u not in idx:
            return []
        ju = idx[u]
        # consider only O perts present in this dataset
        cand = [(p, idx[p]) for p in perts_O if p in idx]
        if not cand:
            return []
        cols = np.array([j for _, j in cand], dtype=int)
        vals = K_i[ju, cols].astype(np.float32)
        order = np.argsort(-vals)[:topk]
        # map back to pert names
        return [(cand[i][0], float(vals[i])) for i in order]

    # -------- 3) Assemble result --------
    out: dict[str, dict] = {}
    for u in perts_U:
        entry = {
            "true_top": _true_top_for_u(u),
            "datasets": {}
        }
        for i, name in enumerate(dataset_names):
            entry["datasets"][name] = _kernel_top_for_u(i, u)
        out[u] = entry

    return out

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
    print("\nPDS per perturbation:")
    print(dict(zip(perts, PDS_scores)))
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
    if args.external_h5ad != "":
        adata_source = ad.read_h5ad(args.external_h5ad)
    elif args.external_list == "":
        raise ValueError("Either --external_h5ad or --external_list must be provided.")

    # Normalize/log target if requested
    if not args.already_logged:
        sc.pp.normalize_total(adata_target, inplace=True)
        sc.pp.log1p(adata_target)

    # Split TARGET into train/test (controls appear in both; controls themselves are never modified)
    adata_train, adata_test, adata_test_orig = train_test_split(args, adata_target)
    if args.test_h5ad != "":  # if external test set provided, merge into overall target dataset
        tmp = adata_test[adata_test.obs[args.target_label] != args.control_label].copy()
        adata_target = ad.concat([adata_train, tmp])
    eval_adata = adata_test if adata_test is not None else adata_train

    # Load externals:
    # If --external_list is provided and non-empty, it *overrides* --external_h5ad.
    external_paths = []
    if args.external_list:
        with open(args.external_list, "r") as f:
            external_paths = [ln.strip() for ln in f if ln.strip()]

    using_external_list = len(external_paths) > 0
    if using_external_list:
        if getattr(args, "external_as_tsv_deltas", False):
            print(f"Found {len(external_paths)} external TSV-delta file(s) from list (target unchanged)...")
            tsv_sources_list = list(external_paths)  # keep as paths; we build kernels later
            adata_sources_list = []                  # unused in TSV mode
            # Set a placeholder so downstream branches that expect 'adata_source' don’t fail
            adata_source = None
        else:
            print(f"Found {len(external_paths)} external dataset(s) from list. Reading & intersecting perts (target unchanged)...")
            adata_sources_list = [
                read_and_intersect(
                    ext_path=p,
                    adata_target=adata_target,
                    target_label=args.target_label,
                    control_label=args.control_label,
                    already_logged=args.already_logged,
                )
                for p in external_paths
            ]
            if len(adata_sources_list) == 0:
                raise ValueError("`--external_list` was provided but no valid paths were found.")
            adata_source = adata_sources_list[0]
    else:
        # Legacy single-external path, but route through read_and_intersect for consistency.
        adata_source = read_and_intersect(
            ext_path=args.external_h5ad,
            adata_target=adata_target,
            target_label=args.target_label,
            control_label=args.control_label,
            already_logged=args.already_logged,
        )

    # Compute a SINGLE global control mean from ALL target controls (fixed across splits)
    ctrl_mask_full = (adata_target.obs[args.target_label] == args.control_label).values
    ctrl_mean_global = np.asarray(adata_target.X)[ctrl_mask_full].mean(axis=0).reshape(-1)

    # Build AverageKnown baselines and truths BEFORE any intersection
    (pred_tr, true_tr, names_tr), (pred_ev, true_ev, names_ev) = build_average_known_baseline(
        adata_train, eval_adata, args.target_label, args.control_label, ctrl_mean_global
    )

    # Intersection handling:
    # - With --external_list we already intersected perts per-external and we DO NOT modify the target.
    # - Otherwise, keep your existing two-sided intersection.
    if using_external_list:
        adata_target_int = adata_target  # target remains unchanged; genes unchanged
        # (adata_source is already per-pert intersected by read_and_intersect)
    else:
        # Existing behavior (two-sided intersect)
        adata_source, adata_target_int = intersect_datasets(
            adata_source, adata_target, args.target_label, args.control_label, intersect_genes=args.intersect_genes
        )
    # Keep split views in intersected target
    adata_train_int = adata_target_int[adata_target_int.obs.index.isin(adata_train.obs.index)].copy()
    eval_adata_int  = adata_target_int[adata_target_int.obs.index.isin(eval_adata.obs.index)].copy()

    # --- Target TRAIN PCA basis for kernel-in-PC-space (optional) ---
    if args.kernel_pc_space_r and args.kernel_pc_space_r > 0:
        W_r_pc, T_O_pc, col_mean_pc, col_std_pc, perts_O_pc = _build_pca_basis_from_train(
            adata_train_target=adata_train_int,     # use your train split
            target_label=args.target_label,
            control_label=args.control_label,
            r=int(args.kernel_pc_space_r),
            gene_zscore=bool(args.kernel_pc_gene_zscore),
        )
        genes_target_pc = adata_train_int.var_names.astype(str).tolist()
        print(f"[pc-kernel] Using target TRAIN PCA rank={args.kernel_pc_space_r} "
              f"({T_O_pc.shape[0]} perts, {len(genes_target_pc)} genes).")
    else:
        W_r_pc = col_mean_pc = col_std_pc = None
        genes_target_pc = []

    # ---------------------------
    # Build per-dataset kernels and aggregate
    # ---------------------------
    perts_O = pert_list(adata_train_int, args.target_label, args.control_label)
    perts_U = pert_list(eval_adata_int,  args.target_label, args.control_label)
    # which externals to use
    if using_external_list:
        if getattr(args, "external_as_tsv_deltas", False):
            src_list = tsv_sources_list  # list of file paths (TSV deltas)
        else:
            src_list = adata_sources_list  # list of AnnData externals
    else:
        src_list = [adata_source]

    kernels_and_perts = []
    for src_i in src_list:
        if using_external_list and getattr(args, "external_as_tsv_deltas", False):
            # Build kernel directly from a TSV of precomputed deltas (rows=genes, cols=perts)
            # Intersect to *target* perts (O∪U) so ordering matches downstream expectations.
            perts_all = list(dict.fromkeys(list(perts_O) + list(perts_U)))  # O∪U, order-preserving
            K_i, perts_i = create_kernel_from_tsv_deltas(
                fname=src_i,
                target_perts=perts_all,
            )
        else:
            if args.kernel_pc_space_r and args.kernel_pc_space_r > 0:
                K_i, perts_i = create_kernel_pc_space(
                    adata_panel=src_i,
                    adata_train_target=adata_train_int,
                    target_label=args.target_label,
                    control_label=args.control_label,
                    kernel_metric=args.kernel_metric,
                    iso_calibrate=args.iso_calibrate,
                    W_r_pc=W_r_pc, T_O_pc=T_O_pc, col_mean_pc=col_mean_pc, col_std_pc=col_std_pc,
                    perts_O_pc=perts_O_pc, genes_target_pc=genes_target_pc,
                    gene_zscore=bool(args.kernel_pc_gene_zscore),
                )
            else:
                K_i, perts_i = create_kernel(
                    adata_source_int=src_i,
                    adata_train_target=adata_train_int,   # isotonic is per-dataset vs target train
                    target_label=args.target_label,
                    control_label=args.control_label,
                    kernel_metric=args.kernel_metric,
                    iso_calibrate=args.iso_calibrate,
                )
        if K_i.size and len(perts_i):
            kernels_and_perts.append((K_i, perts_i))

    # --- Append embedding kernels from YAML, if provided ---
    if args.embeddings_yaml and args.embeddings_yaml != "":
        emb_kernels = build_embedding_kernels_from_yaml(
            embeddings_yaml=args.embeddings_yaml,
            var_names_target=list(map(str, adata_target_int.var_names)),
            perts_O=perts_O, perts_U=perts_U,
            emb_metric=args.emb_metric, emb_pca_dim=args.emb_pca_dim, emb_rbf_gamma=args.emb_rbf_gamma,
        )
        kernels_and_perts.extend(emb_kernels)

    if len(kernels_and_perts) == 0:
        raise ValueError("No valid per-dataset kernels were constructed (neither h5ad nor embedding sources).")

    if args.kernel_agg == "wmean":
        # global weights from alignment with target train O×O
        perts_O = pert_list(adata_train_int, args.target_label, args.control_label)
        weights, raw_scores = compute_global_kernel_weights(
            kernels_and_perts=kernels_and_perts,
            adata_train_target=adata_train_int,
            target_label=args.target_label,
            control_label=args.control_label,
            perts_O=perts_O,
            gamma=args.kernel_weight_gamma,
        )
        print(f"[wmean] raw_scores per dataset: {np.round(np.array(raw_scores), 4).tolist()}")
        print(f"[wmean] normalized weights    : {np.round(np.array(weights), 4).tolist()}")
        K_full, perts_union = aggregate_kernels(
            kernels_and_perts,
            method="wmean",
            weights=weights,
        )
    elif args.kernel_agg == "pwmean":
        # 1) Union order (same as other aggregators)
        #    Reuse aggregate_kernels to get perts_union cheaply (we'll overwrite K afterward)
        K_tmp, perts_union = aggregate_kernels(kernels_and_perts, method="mean")
        # 2) Build a base kernel for neighbor transfer to U
        perts_O = pert_list(adata_train_int, args.target_label, args.control_label)
        # Try global weighted mean for base if multiple datasets; else mean
        if len(kernels_and_perts) > 1:
            weights_base, scores_base = compute_global_kernel_weights(
                kernels_and_perts=kernels_and_perts,
                adata_train_target=adata_train_int,
                target_label=args.target_label,
                control_label=args.control_label,
                perts_O=perts_O,
                gamma=args.kernel_weight_gamma,
            )
            K_base, perts_union_base = aggregate_kernels(
                kernels_and_perts, method="wmean", weights=weights_base
            )
        else:
            K_base, perts_union_base = K_tmp, perts_union
        if perts_union_base != perts_union:
            # align base to union order
            idx_base = {p: j for j, p in enumerate(perts_union_base)}
            sel = [idx_base[p] for p in perts_union if p in idx_base]
            K_base = K_base[np.ix_(sel, sel)]
        # 3) Per-pert weights on O
        w_per_ds_O = compute_per_pert_weights_on_O(
            kernels_and_perts=kernels_and_perts,
            adata_train_target=adata_train_int,
            target_label=args.target_label,
            control_label=args.control_label,
            perts_O=perts_O,
            gamma=args.pw_gamma,
            floor=args.pw_floor,
        )
        # 4) Transfer weights to U via neighbors in base kernel
        perts_U = pert_list(eval_adata_int, args.target_label, args.control_label)
        w_per_ds_U = estimate_weights_for_U_by_neighbors(
            perts_U=perts_U,
            perts_O=perts_O,
            base_kernel=K_base,
            base_perts=perts_union,
            w_per_ds_O=w_per_ds_O,
            topk=args.pw_topk,
        )
        # 5) Final per-pert weighted aggregation
        K_full = aggregate_kernels_pwmean(
            kernels_and_perts=kernels_and_perts,
            perts_union=perts_union,
            w_per_ds_O=w_per_ds_O,
            w_per_ds_U=w_per_ds_U,
            pair_rule=args.pw_pair_rule,
        )
    else:
        K_full, perts_union = aggregate_kernels(
            kernels_and_perts,
            method=args.kernel_agg,   # {"mean","max"}
        )
    print(f"Aggregated kernel: {K_full.shape} over {len(perts_union)} perts.")

    # Build "core" splits that *only* contain perts covered by the kernel union (plus controls).
    # These are used for the KRR call so indices/names match the kernel exactly.
    kernel_pert_set = set(perts_union)
    def _mask_in_kernel(adx):
        lab = adx.obs[args.target_label].astype(str)
        return lab.isin(kernel_pert_set) | (lab == args.control_label)

    adata_train_core = adata_train_int[_mask_in_kernel(adata_train_int)].copy()
    eval_adata_core  = eval_adata_int[_mask_in_kernel(eval_adata_int)].copy()

    # If the user *doesn't* want to keep OOV perts, also shrink the "public" splits
    # so all downstream metrics only reflect the perts covered by the kernel.
    if not args.keep_oov_perts:
        adata_train_int = adata_train_core
        eval_adata_int  = eval_adata_core

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
        adata_train=adata_train_core,
        adata_eval=eval_adata_core,
        target_label=args.target_label,
        control_label=args.control_label,
        krr_lambda=args.krr_lambda,
        ctrl_mean_target=ctrl_mean_global,  # fixed control mean
        kernel_gamma=args.kernel_gamma,
        topk=args.topk,
        boost_pcs=args.boost_pcs,
        boost_gamma=args.boost_gamma,
        conf_boost_alpha=args.conf_boost_alpha,
        conf_shrink_alpha=args.conf_shrink_alpha,
        conf_min_var=args.conf_min_var,
        conf_max_var=args.conf_max_var,
        K_full=K_full,
        perts_all=perts_union,
        pca_r=args.lowrank_pca_r,
        pca_gene_zscore=args.lowrank_gene_zscore,
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
            adata_train=adata_train_core,
            adata_eval=adata_train_core,
            target_label=args.target_label,
            control_label=args.control_label,
            krr_lambda=args.krr_lambda,
            ctrl_mean_target=ctrl_mean_global,
            kernel_gamma=args.kernel_gamma,
            topk=args.topk,
            boost_pcs=args.boost_pcs,
            boost_gamma=args.boost_gamma,
            conf_boost_alpha=args.conf_boost_alpha,
            conf_shrink_alpha=args.conf_shrink_alpha,
            conf_min_var=args.conf_min_var,
            conf_max_var=args.conf_max_var,
            K_full=K_full,
            perts_all=perts_union,
            pca_r=args.lowrank_pca_r,
            pca_gene_zscore=args.lowrank_gene_zscore,
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
            names_eval=names_ev,
            ctrl_mean_eval=ctrl_mean_global,
            target_label=args.target_label,
            control_label=args.control_label,
            out_path=args.test_predict_out,
            random_state=getattr(args, "seed", None),
        )

    # Optional human-readable names for each kernel (same order as kernels_and_perts):
    dataset_names = []
    # Fill dataset_names if you have names; otherwise they auto-name as ds0, ds1, ...

    neighbor_obj = collect_top_neighbors_per_u(
        adata_train_target=adata_train_core,
        adata_eval_target=eval_adata_core,
        target_label=args.target_label,
        control_label=args.control_label,
        perts_U=perts_U,
        kernels_and_perts=kernels_and_perts,
        dataset_names=dataset_names or None,
        topk=10,                 # tweak as you like
        true_metric="corr",      # "corr" or "cosine" for the TRUE side
    )
    with(open('kernel_similarity.pkl', 'wb')) as f:
        pkl.dump(neighbor_obj, f)

    if args.run_diagnostics:
        print("\n=== Running Diagnostics ===")
        # ---------------------------
        # [DIAG] Per-pert "corr-of-corrs" versus target ALL-perts kernel
        # ---------------------------
        try:
            # Build TRUE kernel on ALL perts (train + eval) from target mean-effects
            Y_all = np.vstack([
                (true_tr - ctrl_mean_global[None, :]).astype(np.float32),
                (true_ev - ctrl_mean_global[None, :]).astype(np.float32),
            ])  # (|O|+|U|, G)
            names_all = list(names_tr) + list(names_ev)

            if args.kernel_metric == "corr":
                Z = Y_all - Y_all.mean(axis=1, keepdims=True)
                Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
            else:  # cosine
                Z = Y_all / (np.linalg.norm(Y_all, axis=1, keepdims=True) + 1e-8)
            K_true_all = Z @ Z.T
            np.fill_diagonal(K_true_all, 1.0)
            K_true_all = 0.5 * (K_true_all + K_true_all.T)

            # Labeled external kernels (unchanged)
            labeled_externals = []
            for i, (Ki, pertsi) in enumerate(kernels_and_perts):
                label = (dataset_names[i] if i < len(dataset_names) else f"ds{i}")
                labeled_externals.append((Ki, pertsi, label))

            # Print corr-of-corrs for ALL perts
            report_corr_of_corrs(
                K_true=K_true_all,
                perts_true=names_all,
                externals=labeled_externals,
                topn_per_pert=5,
            )
        except Exception as e:
            print(f"[corr-of-corrs] skipped: {e}")

        # ---------------------------
        # [DIAG] Manhattan (L1) distance CSVs (ALL perts)
        # ---------------------------
        try:
            # All-perts true deltas
            Y_true_all = Y_all  # already centered above
            # 1) Pairwise L1 among ALL perts
            save_manhattan_distances(
                Y_true=Y_true_all, perts_true=names_all,
                out_true_csv="true_L1_all.csv",
            )

            # 2) Pred (eval-only) vs TRUE (ALL-perts) cross L1, with columns = ALL perts
            if pred_ev is not None:
                Y_pred_allrows = (pred_ev - ctrl_mean_global[None, :]).astype(np.float32)  # (|U|, G)
                # block-wise compute |U| x (|O|+|U|) L1
                U = Y_pred_allrows.shape[0]
                P = Y_true_all.shape[0]
                D = np.zeros((U, P), dtype=np.float32)
                # simple block to avoid big temporary tensors
                block = max(1, 4096 // max(1, Y_true_all.shape[1]))
                for s in range(0, U, block):
                    e = min(U, s + block)
                    diffs = np.abs(Y_pred_allrows[s:e, None, :] - Y_true_all[None, :, :])
                    D[s:e, :] = diffs.sum(axis=2)
                df_cross = pd.DataFrame(D, index=list(names_ev), columns=list(names_all))
                df_cross.to_csv("pred_true_L1_all.csv")
            print("[manhattan] wrote true_L1_all.csv" + (", pred_true_L1_all.csv" if pred_ev is not None else ""))
        except Exception as e:
            print(f"[manhattan] skipped: {e}")

    print("\n✨ Done!")

if __name__ == "__main__":
    main()