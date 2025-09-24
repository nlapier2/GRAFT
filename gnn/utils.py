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
