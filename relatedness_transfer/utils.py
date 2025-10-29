# --- utility functions for GNN training and evaluation ---
import os
from typing import Tuple, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


def to_numpy(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)

def build_target_to_gene_index(adata: ad.AnnData, target_label: str) -> Dict[str, int]:
    """
    Map each perturbation label to a gene index IF the label is a gene present in var_names.
    Non-gene labels will be ignored (they can still be part of the training set, but Step0 will
    clamp nothing for that sample). For your panel, we expect labels == gene symbols.
    """
    varset = set(adata.var_names)
    t2i = {}
    for t in adata.obs[target_label].unique():
        if t in varset:
            t2i[t] = int(np.where(adata.var_names == t)[0][0])
    return t2i

def prep_external_data(pb: ad.AnnData, target_label: str, control_label: str, adata_train: ad.AnnData, remove_non_genes: bool) -> ad.AnnData:
    if 'target_present' in pb.obs.columns and remove_non_genes:
        pb = pb[pb.obs["target_present"] | (pb.obs[target_label] == control_label), :].copy()
    # Clean placeholders before normalization
    X = pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # Replace NaN / -1 placeholders with 0 counts
    if np.isnan(X).any():
        np.nan_to_num(X, copy=False, nan=0.0)
    if (X == -1).any():
        X[X == -1] = 0.0
    pb.X = X
    # Normalize only rows with positive sums; leave zero-sum rows as zeros
    row_sums = X.sum(axis=1)
    if (row_sums > 0).any():
        mask = row_sums > 0
        # normalize a temporary AnnData view to avoid Scanpy warnings on zero-sum rows
        tmp = pb[mask].copy()
        sc.pp.normalize_total(tmp, target_sum=None, inplace=True)
        sc.pp.log1p(tmp)
        pb.X[mask] = tmp.X
    # (rows with sum==0 remain zero; perfectly fine for Stage-1 since we mask missing genes)
    assert list(pb.var_names) == list(adata_train.var_names), "Pseudobulk genes/order must match target dataset."
    return pb

def prep_pb_all(pb_target, adata_train, args):
    pb_paths = []
    pbs = []
    pb_all = None
    if pb_target is not None:
        pbs.append(pb_target)

    if args.external_list:
        with open(args.external_list, "r") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                pb_paths.append(s)

    if len(pb_paths) > 0:
        for p in pb_paths:
            pb_i = ad.read_h5ad(p)
            pb_i = prep_external_data(pb_i, args.target_label, args.control_label, adata_train, args.remove_non_gene_perts)
            pbs.append(pb_i)
        # Concatenate all pseudobulk rows
        pb_all = ad.concat(pbs, axis=0, join="outer", merge="same")
        pb_all.obs = pb_all.obs.copy()  # ensure contiguous
    return pb_all, len(pbs)

def train_test_split(args, adata):
    # ---------------------------
    # Train/Test setup
    # If --test_h5ad is provided, use that file for evaluation and ignore --test_pct_perts.
    # Otherwise, do the leave-perturbations-out split as before.
    # ---------------------------
    adata_test_orig = None
    if args.test_h5ad:
        print(f"=== Using external TEST set: {args.test_h5ad} (overrides --test_pct_perts) ===")
        adata_train = adata
        adata_test = ad.read_h5ad(args.test_h5ad)
        adata_test_orig = adata_test.copy()
        if args.use_pseudobulk:
            adata_test = collapse_to_pseudobulk(adata_test, args.target_label)
        sc.pp.normalize_total(adata_test, target_sum=56903, inplace=True)
        sc.pp.log1p(adata_test)
        if sparse.isspmatrix(adata_test.X) and not sparse.isspmatrix_csr(adata_test.X):
            adata_test.X = adata_test.X.tocsr()  # nicer slicing, though we load to numpy anyway
        sc.pp.normalize_total(adata_test_orig, target_sum=56903, inplace=True)
        sc.pp.log1p(adata_test_orig)
        if sparse.isspmatrix(adata_test_orig.X) and not sparse.isspmatrix_csr(adata_test_orig.X):
            adata_test_orig.X = adata_test_orig.X.tocsr()  # nicer slicing, though we load to numpy anyway
        # Sanity check: same genes / order (as guaranteed by user)
        assert np.array_equal(adata_train.var_names.values, adata_test.var_names.values), \
            "Train and test var_names differ or are out of order."
        print(f"Train cells: {adata_train.n_obs}, Test cells: {adata_test.n_obs}")
    else:
        # Leave-perturbations-out split (previous behavior)
        labels_all = adata.obs[args.target_label].astype(str).values
        rng = np.random.default_rng(args.seed)
        all_perts = sorted({lbl for lbl in labels_all if lbl != args.control_label})
        n_test = int(round(args.test_pct_perts * len(all_perts)))
        test_perts = set(rng.choice(np.array(all_perts), size=n_test, replace=False).tolist()) if n_test > 0 else set()
        train_perts = [p for p in all_perts if p not in test_perts]
        mask_train = adata.obs[args.target_label].isin([args.control_label] + train_perts)
        adata_train = adata[mask_train].copy()
        adata_test = adata[adata.obs[args.target_label].isin([args.control_label] + list(test_perts))].copy() if n_test > 0 else None
        print("=== Split summary ===")
        print(f"Total perts (excl. control): {len(all_perts)}  |  Held-out test perts: {len(test_perts)}")
        if n_test > 0:
            print(f"Test perts: {sorted(test_perts)[:10]}{' ...' if len(test_perts) > 10 else ''}")
        print(f"Train cells: {adata_train.n_obs}, Test cells: {adata_test.n_obs if adata_test is not None else 0}")
    return adata_train, adata_test, adata_test_orig


def write_pred_true_h5ads(
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

def collapse_to_pseudobulk(adata, target_label: str):
    """Return a new AnnData with one row per label (perturbation + control)."""
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    df = pd.DataFrame(X, columns=adata.var_names).groupby(labels).mean()
    ad_bulk = ad.AnnData(df.values.astype(np.float32))
    ad_bulk.var_names = adata.var_names.copy()
    ad_bulk.obs[target_label] = df.index.astype(str)
    return ad_bulk

def _variance_to_confidence(pred_delta_var: np.ndarray,
                            vmin: float,
                            vmax: float) -> np.ndarray:
    """
    Map per-(pert,gene) predictive variance to [0,1] confidence.
    Lower var -> confidence ~1.
    Higher var -> confidence ~0.
    """
    # normalize var into [0,1]
    norm = (pred_delta_var - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    conf = 1.0 - norm
    return conf


def apply_confidence_boost(pred_delta_mat: np.ndarray,
                            pred_delta_var: np.ndarray,
                            conf_boost_alpha: float = 0.0,
                            conf_shrink_alpha: float = 0.0,
                            conf_min_var: float = 1e-6,
                            conf_max_var: float = 1.0) -> np.ndarray:
    """
    Take the (|U| x G) predicted delta matrix from KRR and a matching (|U| x G)
    predictive variance matrix, and scale each delta entry based on confidence.

        scale = 1
              + conf_boost_alpha  * conf
              - conf_shrink_alpha * (1 - conf)

    where conf is in [0,1] from _variance_to_confidence().

    Setting conf_shrink_alpha=0 leaves low-confidence genes mostly unchanged.
    Setting conf_boost_alpha>0 amplifies confident hits (PDS-friendly).
    """
    if (conf_boost_alpha == 0.0) and (conf_shrink_alpha == 0.0):
        return pred_delta_mat

    conf = _variance_to_confidence(
        pred_delta_var,
        vmin=conf_min_var,
        vmax=conf_max_var,
    )  # same shape as pred_delta_mat

    scale = 1.0 + conf_boost_alpha * conf - conf_shrink_alpha * (1.0 - conf)
    boosted = pred_delta_mat * scale
    return boosted


def sample_cell_level_deltas(mean_delta_vec: np.ndarray,
                             var_delta_vec: np.ndarray,
                             n_cells: int,
                             var_scale: float = 1.0,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """
    For a single perturbation:
        mean_delta_vec: (G,) mean predicted (pert - ctrl) delta
        var_delta_vec:  (G,) predictive variance for that perturbation
        n_cells:        how many synthetic cells to make
        var_scale:      multiplier on stddev

    Returns (n_cells x G) array, where each row is a sampled delta to subtract
    from a sampled control cell. Low-variance genes -> similar deltas across cells.
    High-variance genes -> more heterogeneity.
    """
    if rng is None:
        rng = np.random.default_rng()

    std_vec = np.sqrt(np.clip(var_delta_vec, 0.0, None)) * float(var_scale)
    # Broadcast mean_delta_vec and std_vec to per-cell Gaussian draws
    deltas = rng.normal(
        loc=mean_delta_vec[None, :],
        scale=std_vec[None, :],
        size=(n_cells, mean_delta_vec.shape[0]),
    )
    return deltas

def compute_avg_kd_efficiencies(
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

def pert_list(adata: ad.AnnData, target_label: str, control_label: str) -> list[str]:
    perts_all = list(map(str, adata.obs[target_label].values))
    # If pseudobulked, each row is a single pert; otherwise fallback to unique order.
    # In either case, we evaluate on NON-control perts only.
    uniq = list(dict.fromkeys(perts_all))  # stable unique
    return [p for p in uniq if p != control_label]

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
    global_eff, per_gene_eff = compute_avg_kd_efficiencies(
        adata_train=adata_train, O=pert_list(adata_train, target_label, control_label),
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

    print('Target perturbations not found in target:', sorted(list(target_perts - set(common_perts))))

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

def row_standardize(M: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm per row (safe for sparse/np arrays)."""
    M = np.asarray(M, dtype=np.float32)
    M = M - M.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    return M / denom
