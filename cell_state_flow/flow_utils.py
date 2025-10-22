# --- utility functions for training and evaluation ---
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
from collections import defaultdict
from sklearn.metrics import pairwise_distances


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

def make_base_adjacency(G: int, self_loops: bool = True) -> torch.Tensor:
    """
    Dense fully-connected adjacency (uniform), normalized row-wise.
    We'll mask rows per-sample to forbid inbound messages to the target.
    """
    A = torch.ones(G, G)
    if not self_loops:
        A.fill_diagonal_(0.0)
    # row-normalize so each node aggregates an average of neighbors
    A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
    return A

def make_adjacency_prior(W_meta: np.ndarray, meta_topk: int, G: int, device: str) -> torch.Tensor:
    # build kNN in prior space (cosine), symmetric, row-normalized
    Wm = W_meta.astype(np.float32)  # (R,G)
    # cosine over columns
    V = Wm.T  # (G,R)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    S = Vn @ Vn.T  # (G,G) cosine similarity
    # for each row, keep top-k (including self), set others to 0
    k = min(meta_topk, G)
    A = np.zeros_like(S, dtype=np.float32)
    idx = np.argpartition(-S, kth=k-1, axis=1)[:, :k]
    rows = np.repeat(np.arange(G)[:, None], k, axis=1)
    A[rows, idx] = S[rows, idx]
    # symmetrize by max
    A = np.maximum(A, A.T)
    # row-normalize
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    A_base = torch.from_numpy(A).to(device)
    return A_base, k

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

def train_test_split(args, adata, pb_target):
    # ---------------------------
    # Train/Test setup
    # If --test_h5ad is provided, use that file for evaluation and ignore --test_pct_perts.
    # Otherwise, do the leave-perturbations-out split as before.
    # ---------------------------
    if args.test_h5ad:
        print(f"=== Using external TEST set: {args.test_h5ad} (overrides --test_pct_perts) ===")
        adata_train = adata
        adata_test = ad.read_h5ad(args.test_h5ad)
        # (Optional) apply the same pseudobulk collapse if requested
        if args.use_pseudobulk:
            adata_test = collapse_to_pseudobulk(adata_test, args.target_label)
        sc.pp.normalize_total(adata_test, inplace=True)
        sc.pp.log1p(adata_test)
        if sparse.isspmatrix(adata_test.X) and not sparse.isspmatrix_csr(adata_test.X):
            adata_test.X = adata_test.X.tocsr()  # nicer slicing, though we load to numpy anyway
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
        if pb_target is not None:  # also filter pseudobulk to training perts only
            pb_target = pb_target[pb_target.obs[args.target_label].isin([args.control_label] + train_perts)].copy()
        adata_test = adata[adata.obs[args.target_label].isin([args.control_label] + list(test_perts))].copy() if n_test > 0 else None
        print("=== Split summary ===")
        print(f"Total perts (excl. control): {len(all_perts)}  |  Held-out test perts: {len(test_perts)}")
        if n_test > 0:
            print(f"Test perts: {sorted(test_perts)[:10]}{' ...' if len(test_perts) > 10 else ''}")
        print(f"Train cells: {adata_train.n_obs}, Test cells: {adata_test.n_obs if adata_test is not None else 0}")
    return adata_train, adata_test, pb_target

@torch.no_grad()
def print_edge_weight_stats(model, prefix="edges"):
    """
    Prints mean/std/min/max for learned edge weights.
    - For sparse SpMM: over E stored edges (CSR).
    - For dense learned edges: over all GxG entries.

    It prints both raw probabilities (sigmoid(logit)) and the row-normalized version
    that the layer actually uses in forward.
    """
    printed_any = False

    # --- Sparse SpMM weights (E edges) ---
    if hasattr(model, "edge_logit") and model.edge_logit is not None:
        logits = model.edge_logit
        G = int(model.csr_rowptr.numel() - 1)
        E = int(logits.numel())
        w = torch.sigmoid(logits)  # (E,)

        # row-normalize like in forward
        rows = model.csr_rows
        row_sums = torch.zeros(G, dtype=w.dtype, device=w.device)
        row_sums.index_add_(0, rows, w)
        w_norm = w / row_sums[rows].clamp_min(1e-8)

        def _stats(x):
            return (x.mean().item(), x.std(unbiased=False).item(),
                    x.min().item(), x.max().item())

        m, s, mn, mx = _stats(w)
        mN, sN, mnN, mxN = _stats(w_norm)

        print(f"[{prefix}:sparse] G={G}  E={E}")
        print(f"  raw    σ(logit): mean={m:.5f}  std={s:.5f}  min={mn:.3e}  max={mx:.5f}")
        print(f"  row-norm used : mean={mN:.5f} std={sN:.5f} min={mnN:.3e} max={mxN:.5f}")
        printed_any = True

    # --- Dense learned weights (GxG) ---
    if hasattr(model, "dense_edge_logit") and model.dense_edge_logit is not None:
        W_raw = torch.sigmoid(model.dense_edge_logit)   # (G,G)
        # row-normalize
        W = W_raw / W_raw.sum(dim=1, keepdim=True).clamp_min(1e-8)

        def _stats2(x):
            return (x.mean().item(), x.std(unbiased=False).item(),
                    x.amin().item(), x.amax().item())

        m, s, mn, mx = _stats2(W_raw)
        mN, sN, mnN, mxN = _stats2(W)

        G = W_raw.shape[0]
        print(f"[{prefix}:dense ] G={G}  entries={G*G}")
        print(f"  raw    σ(logit): mean={m:.5f}  std={s:.5f}  min={mn:.3e}  max={mx:.5f}")
        print(f"  row-norm used : mean={mN:.5f} std={sN:.5f} min={mnN:.3e} max={mxN:.5f}")
        printed_any = True


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
