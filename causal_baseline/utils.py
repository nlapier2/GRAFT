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

def train_test_split(args, adata):
    # ---------------------------
    # Train/Test setup
    # If --test_h5ad is provided, use that file for evaluation and ignore --test_pct_perts.
    # Otherwise, do the leave-perturbations-out split as before.
    # ---------------------------
    if args.test_h5ad:
        print(f"=== Using external TEST set: {args.test_h5ad} (overrides --test_pct_perts) ===")
        adata_train = adata
        adata_test = ad.read_h5ad(args.test_h5ad)
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
        adata_test = adata[adata.obs[args.target_label].isin([args.control_label] + list(test_perts))].copy() if n_test > 0 else None
        print("=== Split summary ===")
        print(f"Total perts (excl. control): {len(all_perts)}  |  Held-out test perts: {len(test_perts)}")
        if n_test > 0:
            print(f"Test perts: {sorted(test_perts)[:10]}{' ...' if len(test_perts) > 10 else ''}")
        print(f"Train cells: {adata_train.n_obs}, Test cells: {adata_test.n_obs if adata_test is not None else 0}")
    return adata_train, adata_test

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
