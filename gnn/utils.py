from email import parser
import argparse, math, os, sys, random
from typing import Tuple, Dict, List
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

def sample_minibatch(
    X_ctrl: np.ndarray,
    X_pert: np.ndarray,
    pert_labels: np.ndarray,
    control_label: str,
    batch_size: int,
    rng: np.random.Generator,
    fixed_label: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Returns batch_x_ctrl (B,G), batch_x_pert (B,G), batch_targets (list of labels).
    Matches each perturbed cell to a random control cell.
    """
    B = batch_size
    # indices for perturbed cells (exclude controls)
    if fixed_label is None:
        pert_idx = np.where(pert_labels != control_label)[0]
    else:
        pert_idx = np.where(pert_labels == fixed_label)[0]
    if len(pert_idx) < B:
        choice = rng.choice(pert_idx, size=B, replace=True)
    else:
        choice = rng.choice(pert_idx, size=B, replace=False)
    # random controls
    ctrl_idx = np.where(pert_labels == control_label)[0]
    rand_ctrl = rng.choice(ctrl_idx, size=B, replace=True)
    bx_ctrl = torch.from_numpy(X_ctrl[rand_ctrl]).float()
    bx_pert = torch.from_numpy(X_pert[choice]).float()
    btargets = pert_labels[choice].tolist()
    return bx_ctrl, bx_pert, btargets

def sample_minibatch_knn(
    X: np.ndarray,
    labels: np.ndarray,
    control_label: str,
    batch_size: int,
    rng: np.random.Generator,
    knn_k: int = 32,
    knn_temp: float = 0.1,
    metric: str = "l2",
    pre_norm_ctrl: np.ndarray | None = None,
    pre_norm_pert: np.ndarray | None = None,
    fixed_label: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Choose B perturbed cells; for each, sample one control from its top-k nearest controls
    with softmax(-dist / temp). Supports L2 or cosine. Small-panel friendly (no index lib).
    """
    # index pools
    ctrl_idx = np.where(labels == control_label)[0]
    if fixed_label is None:
        pert_idx = np.where(labels != control_label)[0]
    else:
        pert_idx = np.where(labels == fixed_label)[0]
    if len(pert_idx) == 0 or len(ctrl_idx) == 0:
        raise ValueError("Need both control and perturbed cells.")
    # choose pert rows
    if len(pert_idx) < batch_size:
        sel_pert = rng.choice(pert_idx, size=batch_size, replace=True)
    else:
        sel_pert = rng.choice(pert_idx, size=batch_size, replace=False)
    Xp = X[sel_pert]  # (B,G)
    Xc = X[ctrl_idx]  # (Nc,G)
    # distances (B,Nc)
    if metric == "cosine":
        # use pre-normalized rows if provided
        P = pre_norm_pert if pre_norm_pert is not None else (Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-8))
        C = pre_norm_ctrl if pre_norm_ctrl is not None else (Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8))
        # cosine distance = 1 - cosine similarity
        dists = 1.0 - (P @ C.T)
    else:  # l2
        # (a-b)^2 = a^2  b^2 - 2ab
        a2 = np.sum(Xp * Xp, axis=1, keepdims=True)        # (B,1)
        b2 = np.sum(Xc * Xc, axis=1, keepdims=True).T      # (1,Nc)
        dists = a2 + b2 - 2.0 * (Xp @ Xc.T)                # (B,Nc)
        dists = np.maximum(dists, 0.0)
    # top-k and softmax sampling
    B = sel_pert.shape[0]
    sel_ctrl = np.empty(B, dtype=int)
    for i in range(B):
        di = dists[i]
        k = min(knn_k, di.shape[0])
        cand = np.argpartition(di, k-1)[:k]      # indices of k smallest
        # softmax over -dist / temp  (smaller distance => higher prob)
        logits = -di[cand] / max(knn_temp, 1e-6)
        logits -= logits.max()                   # stabilize
        p = np.exp(logits)
        p /= p.sum()
        sel_ctrl[i] = rng.choice(cand, p=p)
    # pack tensors
    bx_ctrl = torch.from_numpy(Xc[sel_ctrl]).float()
    bx_pert = torch.from_numpy(Xp).float()
    btargets = labels[sel_pert].tolist()
    return bx_ctrl, bx_pert, btargets

def sample_batch_by_mode(single_pert_batches: bool, pretrain_mode: bool, rng: np.random.Generator, pert_unique: List[str],
                         X: np.ndarray, labels: np.ndarray, control_label: str, ctrl_by_dset: dict | None,
                         ctrl_mean_np: np.ndarray, match_controls: str, knn_k: int, knn_temp: float, knn_metric: str,
                         X_ctrl: np.ndarray, X_pert: np.ndarray, pert_labels: np.ndarray, batch_size: int, adata: ad.AnnData,
                         pre_norm_ctrl: np.ndarray | None = None, pre_norm_pert: np.ndarray | None = None):
    """
    Sample a minibatch of (control, perturbed) pairs according to the specified mode.
    """
    sel_pert = None
    # if requested, choose a single perturbation label for this batch
    fixed_label = None
    if single_pert_batches:
        fixed_label = rng.choice(np.array(pert_unique)) if len(pert_unique) > 0 else None

    if pretrain_mode and (ctrl_by_dset is not None):
        # --- Stage-1 pseudobulk controls: match control by dataset_id, not by sampling ---
        # choose perturbed rows for this batch
        if fixed_label is None:
            pert_idx = np.where(labels != control_label)[0]
        else:
            pert_idx = np.where(labels == fixed_label)[0]
        if len(pert_idx) < batch_size:
            sel_pert = rng.choice(pert_idx, size=batch_size, replace=True)
        else:
            sel_pert = rng.choice(pert_idx, size=batch_size, replace=False)
        bx_pert = torch.from_numpy(X[sel_pert]).float()
        # build per-row control vector from the SAME dataset; fallback to global control mean
        dsets = adata.obs["dataset_id"].astype(str).values
        bx_ctrl_rows = []
        for j in sel_pert:
            dj = dsets[j]
            if (dj in ctrl_by_dset):
                bx_ctrl_rows.append(ctrl_by_dset[dj])
            else:
                bx_ctrl_rows.append(ctrl_mean_np)
        bx_ctrl = torch.from_numpy(np.stack(bx_ctrl_rows, axis=0)).float()
        btargets = labels[sel_pert].tolist()
    elif match_controls == "knn":
        # lazily build cosine norms if requested
        if knn_metric == "cosine":
            if pre_norm_ctrl is None:
                ctrl_idx_all = np.where(labels == control_label)[0]
                pre_norm_ctrl = X[ctrl_idx_all] / (np.linalg.norm(X[ctrl_idx_all], axis=1, keepdims=True) + 1e-8)
                pre_norm_pert = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        bx_ctrl, bx_pert, btargets = sample_minibatch_knn(
            X=X, labels=labels, control_label=control_label,
            batch_size=batch_size, rng=rng,
            knn_k=knn_k, knn_temp=knn_temp, metric=knn_metric,
            pre_norm_ctrl=pre_norm_ctrl, pre_norm_pert=pre_norm_pert,
            fixed_label=fixed_label
        )
    else:
        bx_ctrl, bx_pert, btargets = sample_minibatch(
            X_ctrl=X_ctrl, X_pert=X_pert, pert_labels=pert_labels,
            control_label=control_label, batch_size=batch_size, rng=rng,
            fixed_label=fixed_label
        )
    return bx_ctrl, bx_pert, btargets, sel_pert

def get_dset_indices(sel_pert, pert_rowidx, adata: ad.AnnData, device: str) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    z_d, z_ct = None, None
    # identify the row indices that produced bx_pert
    if 'sel_pert' in locals() and sel_pert is not None:           # pretrain_mode branch (or random sampler)
        idx_rows = np.asarray(sel_pert, dtype=int)
    elif 'pert_rowidx' in locals() and pert_rowidx is not None:
        # KNN sampler should provide these indices (may be a CUDA tensor)
        if torch.is_tensor(pert_rowidx):
            idx_rows = pert_rowidx.detach().cpu().numpy().astype(int)
        else:
            idx_rows = np.asarray(pert_rowidx, dtype=int)
    else:
        idx_rows = None
    if idx_rows is not None:
        if "dataset_id" in adata.obs.columns:
            z_d = torch.tensor(pd.Categorical(adata.obs.iloc[idx_rows]["dataset_id"]).codes,
                                device=device)
        if "cell_type" in adata.obs.columns:
            ct_slice = adata.obs.iloc[idx_rows]["cell_type"]
            # Cast to str before fill/replace to avoid Categorical category errors
            if isinstance(ct_slice.dtype, pd.CategoricalDtype):
                ct_slice = ct_slice.astype(str)
            else:
                ct_slice = ct_slice.astype(str)
            ct_slice = ct_slice.replace({"<NA>": "UNK", "mixed": "UNK"}).fillna("UNK")
            z_ct = torch.tensor(pd.Categorical(ct_slice).codes, device=device)
    return z_d, z_ct

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

def collapse_to_pseudobulk(adata, target_label: str):
    """Return a new AnnData with one row per label (perturbation + control)."""
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    df = pd.DataFrame(X, columns=adata.var_names).groupby(labels).mean()
    ad_bulk = ad.AnnData(df.values.astype(np.float32))
    ad_bulk.var_names = adata.var_names.copy()
    ad_bulk.obs[target_label] = df.index.astype(str)
    return ad_bulk

def make_pretrain_pseudobulk_from_adata(adata: ad.AnnData, target_label: str, control_label: str, dataset_id: str) -> ad.AnnData:
    """One-row-per-pert pseudobulk of the provided AnnData (already normalized/log1p).
    Keeps columns: dataset_id, is_control, target_idx, target_present, cell_type (UNK), tech_batch_id (UNK), lab_id (UNK)."""
    groups = adata.obs[target_label].astype(str).values
    uniq = np.unique(groups)
    X_rows = []
    obs_rows = []
    # map gene -> index once
    gene_to_pos = {g: i for i, g in enumerate(adata.var_names)}
    for p in uniq:
        m = (groups == p)
        X_rows.append(np.asarray(adata[m].X.mean(axis=0)).squeeze())
        if p == control_label:
            pert_type = "control"; t_idx = -1; t_present = False
        else:
            pos = gene_to_pos.get(p, None)
            if pos is None:
                pert_type = "non_gene"; t_idx = -1; t_present = False
            else:
                pert_type = "gene"; t_idx = int(pos); t_present = True
        obs_rows.append({
            "dataset_id": dataset_id,
            "is_control": (p == control_label),
            target_label: p,
            "pert_type": pert_type,
            "target_idx": t_idx,
            "target_present": t_present,
            "cell_type": "UNK",
            "tech_batch_id": "UNK",
            "lab_id": "UNK",
        })
    X_bulk = np.stack(X_rows, axis=0)
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"{dataset_id}::{r[target_label]}" for _, r in obs.iterrows()]
    return ad.AnnData(X=X_bulk, obs=obs, var=adata.var.copy())

def prep_external_data(pb: ad.AnnData, target_label: str, control_label: str, adata_train: ad.AnnData) -> ad.AnnData:
    if 'target_present' in pb.obs.columns:
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
    if args.pretrain_pseudobulk:
        pb_paths.append(args.pretrain_pseudobulk)
    if args.pretrain_pseudobulk_list:
        with open(args.pretrain_pseudobulk_list, "r") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                pb_paths.append(s)

    if len(pb_paths) > 0:
        for p in pb_paths:
            pb_i = ad.read_h5ad(p)
            pb_i = prep_external_data(pb_i, args.target_label, args.control_label, adata_train)
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