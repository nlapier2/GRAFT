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

from models import GeneMPNN

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

def get_dset_indices(sel_pert, pert_rowidx, adata: ad.AnnData, device: str, model: Optional[nn.Module] = None) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Prefer model's Stage-1 category->row maps (dset_id2row / ct_id2row) so Stage-2
    indices line up with learned embedding rows. If any row is unmapped, return None
    for that embedding so the forward() zeros-fallback is used. Otherwise, fall back
    to per-batch Categorical codes. Always return torch.long when not None.
    """
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
        # --- DATASET indices ---
        if "dataset_id" in adata.obs.columns and \
           (getattr(model, "dset_E", None) is not None) and \
           isinstance(getattr(model, "dset_E", None), nn.Embedding) and \
           (getattr(model, "dset_id2row", None) is not None):
            vals = adata.obs["dataset_id"].astype(str).values[idx_rows]
            idxs = [model.dset_id2row.get(v, -1) for v in vals]
            if all(i >= 0 for i in idxs):
                z_d = torch.tensor(idxs, device=device, dtype=torch.long)
            else:
                z_d = None  # triggers zeros-fallback in forward()
        elif "dataset_id" in adata.obs.columns:
            # fallback: per-batch categoricals (may not align with Stage-1 ordering)
            z_d = torch.tensor(
                pd.Categorical(adata.obs.iloc[idx_rows]["dataset_id"]).codes,
                device=device, dtype=torch.long
            )

        # --- CELL-TYPE indices ---
        if "cell_type" in adata.obs.columns and \
           (getattr(model, "ct_E", None) is not None) and \
           isinstance(getattr(model, "ct_E", None), nn.Embedding) and \
           (getattr(model, "ct_id2row", None) is not None):
            vals = adata.obs["cell_type"].astype(str).values[idx_rows]
            idxs = [model.ct_id2row.get(v, -1) for v in vals]
            if all(i >= 0 for i in idxs):
                z_ct = torch.tensor(idxs, device=device, dtype=torch.long)
            else:
                z_ct = None
        elif "cell_type" in adata.obs.columns:
            ct_slice = adata.obs.iloc[idx_rows]["cell_type"]
            # robust cast to str then sanitize UNK-like values
            if isinstance(ct_slice.dtype, pd.CategoricalDtype):
                ct_slice = ct_slice.astype(str)
            else:
                ct_slice = ct_slice.astype(str)
            ct_slice = ct_slice.replace({"<NA>": "UNK", "mixed": "UNK"}).fillna("UNK")
            z_ct = torch.tensor(pd.Categorical(ct_slice).codes, device=device, dtype=torch.long)
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

@torch.no_grad()
def precompute_hop_shells(A_base: torch.Tensor, max_hops: int = 4) -> Dict[int, List[torch.Tensor]]:
    """
    Precompute hop 'shells' S_h(t) around each node t:
      S_0(t)={t}, S_1(t)=neighbors(t), S_2(t)=nodes at 2 hops excluding S_0,S_1, ...
    Args:
      A_base: (G,G) adjacency (float or bool); nonzero = edge
      max_hops: compute up to this hop (inclusive if reachable)
    Returns:
      shells: dict t -> list[Tensor], where list[h] are node indices at hop h
    """
    G = A_base.shape[0]
    # treat any nonzero as an edge
    A_bool = (A_base > 0)
    shells: Dict[int, List[torch.Tensor]] = {}
    device = A_base.device
    all_idx = torch.arange(G, device=device)
    for t in range(G):
        S = []
        visited = torch.zeros(G, dtype=torch.bool, device=device)
        # S0
        S0 = torch.tensor([t], device=device, dtype=torch.long)
        S.append(S0)
        visited[t] = True
        frontier = S0
        for _ in range(1, max_hops + 1):
            # neighbors of frontier
            neigh_mask = A_bool[frontier].any(dim=0)
            # remove visited
            new_mask = neigh_mask & (~visited)
            if not new_mask.any():
                break
            Sh = all_idx[new_mask]
            S.append(Sh)
            visited[Sh] = True
            frontier = Sh
        shells[t] = S
    return shells

@torch.no_grad()
def compute_locality_metrics(
    delta: torch.Tensor,
    target_idx: torch.Tensor,
    shells: Dict[int, List[torch.Tensor]],
    ks: List[int] = [1, 2],
) -> Dict[str, float]:
    """
    Compute Expected Hop Distance (EHD) and Loc@K over the batch.
    Args:
      delta: (B,G) predicted delta = yhat - x0
      target_idx: (B,) target indices (>=0 for gene-target perts)
      shells: dict from precompute_hop_shells
      ks: which K to report for Loc@K
    Returns:
      Dict with 'ehd' and 'loc@{K}' averaged over rows with valid targets.
    """
    device = delta.device
    B, G = delta.shape
    abs_delta = delta.abs()
    eps = 1e-12
    ehd_sum = 0.0
    loc_sums = {k: 0.0 for k in ks}
    n = 0
    for i in range(B):
        t = int(target_idx[i].item())
        if t < 0:  # skip non-gene perts
            continue
        S = shells.get(t, None)
        if not S:
            continue
        denom = float(abs_delta[i].sum().item() + eps)
        # mass per hop
        p_h = []
        for h, Sh in enumerate(S):
            if Sh.numel() == 0:
                p_h.append(0.0)
            else:
                p_h.append(float(abs_delta[i, Sh].sum().item() / denom))
        # EHD
        ehd_i = sum(h * ph for h, ph in enumerate(p_h))
        ehd_sum += ehd_i
        # Loc@K
        for k in ks:
            # cap k if shells shorter
            k_eff = min(k, len(S) - 1)
            loc = sum(p_h[: k_eff + 1])
            loc_sums[k] += loc
        n += 1
    if n == 0:
        return {"ehd": 0.0, **{f"loc@{k}": 0.0 for k in ks}}
    out = {"ehd": ehd_sum / n}
    for k in ks:
        out[f"loc@{k}"] = loc_sums[k] / n
    return out


def _merge_id_maps(old_map: dict | None, new_cats: list[str]) -> dict:
    """
    Keep previous indices for known categories; append new ones at the end.
    """
    old_map = old_map or {}
    merged = dict(old_map)  # copy
    next_idx = 0 if not old_map else (max(old_map.values()) + 1)
    for c in new_cats:
        if c not in merged:
            merged[c] = next_idx
            next_idx += 1
    return merged

def _load_state_with_embedding_expansion(
    model,
    state: dict,
    expand_keys=("dset_E.weight", "ct_E.weight", "cond_E.weight"),
):
    """
    Expand checkpoint embedding matrices to the model's size (row-wise), copy
    overlapping rows, and then load once. Leaves new rows randomly initialized.
    Returns (missing, unexpected) like load_state_dict.
    """
    model_state = model.state_dict()
    # Make a shallow copy so we can edit tensors in-place for load
    state_expanded = dict(state)
    for k in expand_keys:
        if (k in state_expanded) and (k in model_state):
            W_old = state_expanded[k]
            W_new = model_state[k]
            if (
                W_old.dim() == 2
                and W_new.dim() == 2
                and W_old.size(1) == W_new.size(1)   # same embed dim
                and W_old.size(0) <= W_new.size(0)   # old rows ≤ new rows
            ):
                # Build an expanded tensor initialized as current model param,
                # then copy old rows into the front slice.
                W_exp = W_new.clone()
                W_exp[: W_old.size(0), :] = W_old.to(W_new.device, dtype=W_new.dtype)
                state_expanded[k] = W_exp
            # else: shapes incompatible → let load_state_dict report it
    missing, unexpected = model.load_state_dict(state_expanded, strict=False)
    return missing, unexpected

def expand_adam_states_for_embeddings(optimizer: torch.optim.Optimizer):
    """
    If an embedding param grew rows (e.g., [2, D] -> [3, D]), expand Adam buffers
    (exp_avg, exp_avg_sq) to match the new shape by zero-padding the extra rows.
    Safe no-op for non-Adam or already-matching shapes.
    """
    for p, st in optimizer.state.items():
        if not st:
            continue
        exp_avg = st.get("exp_avg", None)
        exp_var = st.get("exp_avg_sq", None)
        if exp_avg is None or exp_var is None:
            continue
        # Only handle 2D (embedding-like) tensors where only row count changed
        if p.data.dim() == 2 and exp_avg.dim() == 2 and exp_var.dim() == 2:
            rows_new, dim_new = p.data.size(0), p.data.size(1)
            rows_old, dim_old = exp_avg.size(0), exp_avg.size(1)
            if dim_new == dim_old and rows_old < rows_new:
                dev = p.data.device
                dt  = exp_avg.dtype
                ea  = torch.zeros((rows_new, dim_new), device=dev, dtype=dt)
                ev  = torch.zeros((rows_new, dim_new), device=dev, dtype=dt)
                ea[:rows_old, :] = exp_avg.to(dev)
                ev[:rows_old, :] = exp_var.to(dev)
                st["exp_avg"]    = ea
                st["exp_avg_sq"] = ev

# --- helpers for (re)building and (re)loading ---
def build_model_for_dataset(adata_like, args, load_weights_from: str = ""):
    G = adata_like.n_vars

    # Defaults from args
    dset_dim = int(getattr(args, "dset_embed_dim", 0) or 0)
    ct_dim   = int(getattr(args, "ct_embed_dim",   0) or 0)
    dset_vocab = 0
    ct_vocab   = 0

    # Categories present in *this* adata
    dset_cats = []
    ct_cats   = []
    if ("dataset_id" in adata_like.obs.columns) and (dset_dim > 0):
        dset_cats = list(pd.Categorical(adata_like.obs["dataset_id"].astype(str)).categories)
        dset_vocab = len(dset_cats)
    if ("cell_type" in adata_like.obs.columns) and (ct_dim > 0):
        ct_cats = list(pd.Categorical(adata_like.obs["cell_type"].astype(str)).categories)
        ct_vocab = len(ct_cats)

    ckpt = None
    state = None
    # If loading weights, union vocab with checkpoint (allow expansion)
    if load_weights_from:
        ckpt = torch.load(load_weights_from, map_location=args.device)
        state = ckpt.get("state_dict", ckpt)
        # Embed dims from checkpoint if present
        if "dset_E.weight" in state:
            w = state["dset_E.weight"]; dset_dim_ck, dset_vocab_ck = int(w.size(1)), int(w.size(0))
            dset_dim = dset_dim_ck
            dset_vocab = max(dset_vocab, dset_vocab_ck)
        if "ct_E.weight" in state:
            w = state["ct_E.weight"]; ct_dim_ck, ct_vocab_ck = int(w.size(1)), int(w.size(0))
            ct_dim = ct_dim_ck
            ct_vocab = max(ct_vocab, ct_vocab_ck)
        # Also match node embedding dim if present
        if "node_E.weight" in state:
            args.node_dim = int(state["node_E.weight"].size(1))

    # ---- decide final vocab sizes from *merged maps* (old maps + new categories) ----
    old_dmap = None
    old_ctmap = None
    if isinstance(ckpt, dict):
        meta = ckpt.get("meta", {})
        old_dmap = meta.get("dset_id2row", None)
        old_ctmap = meta.get("ct_id2row", None)

    # Merge id maps: keep existing indices; append new categories at the end
    merged_dmap = _merge_id_maps(old_dmap, dset_cats) if dset_dim > 0 else None
    merged_ctmap = _merge_id_maps(old_ctmap, ct_cats) if ct_dim > 0 else None

    # Final vocabs MUST be at least the size of merged maps (prevents out-of-range indices)
    if merged_dmap is not None:
        dset_vocab = max(dset_vocab, len(merged_dmap))
    if merged_ctmap is not None:
        ct_vocab = max(ct_vocab, len(merged_ctmap))

    # Build the model with the resolved shapes
    model = GeneMPNN(
        G=G,
        hidden=args.hidden,
        T=args.T,
        tau=args.tau,
        node_dim=args.node_dim,
        proj_dim=args.proj_dim,
        dset_vocab=dset_vocab, 
        dset_dim=dset_dim,
        ct_vocab=ct_vocab,     
        ct_dim=ct_dim,
    ).to(args.device)

    # Load weights, allowing embedding expansion
    if load_weights_from:
        missing, unexpected = _load_state_with_embedding_expansion(model, state)
        print(f"[load-weights] {load_weights_from} | missing={len(missing)} unexpected={len(unexpected)}")
        # Set merged maps on the model (so get_dset_indices uses the expanded namespace)
        model.dset_id2row = merged_dmap
        model.ct_id2row   = merged_ctmap

    # If no checkpoint: derive maps from this adata
    if (getattr(model, "dset_E", None) is not None) and (model.dset_id2row is None) and dset_cats:
        model.dset_id2row = {cat: i for i, cat in enumerate(dset_cats)}
    if (getattr(model, "ct_E", None) is not None) and (model.ct_id2row is None) and ct_cats:
        model.ct_id2row   = {cat: i for i, cat in enumerate(ct_cats)}

    return model

def load_full_checkpoint(load_path: str, device: str) -> Tuple[Dict, Dict | None, int, dict]:
    """
    Returns: (state_dict, optimizer_state_dict, start_epoch:int, meta:dict)
    """
    ckpt = torch.load(load_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    opt_state = ckpt.get("optimizer_state_dict", None)
    start_epoch = int(ckpt.get("epoch", 0))
    meta = ckpt.get("meta", {})
    print(f"[load-ckpt] {load_path} | epoch={start_epoch}")
    return state, opt_state, start_epoch, meta

def save_full_checkpoint(path: str, model, optimizer,  extra_meta: dict = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dset_id2row = getattr(model, "dset_id2row", None)
    ct_id2row = getattr(model, "ct_id2row", None)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": {
            "G": getattr(model, "G", None),
            "hidden": getattr(model, "hidden", None),
            "T": getattr(model, "T", None),
            "proj_dim": getattr(model, "proj_dim", None),
            "dset_id2row": dset_id2row,
            "ct_id2row": ct_id2row
        },
    }
    if extra_meta:
        payload["meta"].update(extra_meta)
    torch.save(payload, path)

def load_similarity_npz(
    npz_path: str,
    genes_wanted: Sequence[str],
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load saved CSR similarity and subset/reorder to 'genes_wanted'.

    Returns:
      rowptr:(G'+1,) int64, colind:(E',) int64, values:(E',) float32
      where rows are in the exact order of 'genes_wanted', and columns follow that same order.

    Notes:
      - Edges pointing to genes not in 'genes_wanted' are dropped.
      - Rows with zero remaining neighbors are allowed (degree 0); the GNN layer handles it.
    """
    z = np.load(npz_path, allow_pickle=True)
    indptr = z["indptr"]     # (G+1,)
    indices = z["indices"]   # (E,)
    data = z["data"]         # (E,)
    saved_genes = z["genes"].tolist()  # list[str]
    shape = tuple(z["shape"].tolist())
    assert len(saved_genes) == shape[0] == shape[1], "Saved CSR must be square & gene-aligned."

    # Map requested genes into saved index space
    pos = {g: i for i, g in enumerate(saved_genes)}
    wanted = [g for g in genes_wanted if g in pos]
    Gp = len(wanted)
    if Gp == 0:
        raise ValueError("None of the requested genes are present in the saved similarity.")

    # New order mapping: saved_idx -> new_idx (or -1 if not present)
    saved_to_new = np.full(len(saved_genes), -1, dtype=np.int64)
    for new_i, g in enumerate(wanted):
        saved_to_new[pos[g]] = new_i

    # Build new CSR by scanning each kept row
    new_indptr = np.zeros(Gp + 1, dtype=np.int64)
    new_indices = []
    new_values = []

    edge_count = 0
    for new_i, g in enumerate(wanted):
        old_i = pos[g]
        s, e = indptr[old_i], indptr[old_i + 1]
        cols = indices[s:e]
        vals = data[s:e]
        # keep only columns that are also wanted
        mask = saved_to_new[cols] >= 0
        cols_new = saved_to_new[cols[mask]]
        vals_new = vals[mask]
        new_indices.append(cols_new)
        new_values.append(vals_new)
        edge_count += cols_new.size
        new_indptr[new_i + 1] = edge_count

    # Concatenate
    if edge_count == 0:
        new_indices_arr = np.zeros((0,), dtype=np.int64)
        new_values_arr = np.zeros((0,), dtype=np.float32)
    else:
        new_indices_arr = np.concatenate(new_indices).astype(np.int64, copy=False)
        new_values_arr = np.concatenate(new_values).astype(np.float32, copy=False)

    # → torch (optionally to device)
    rowptr_t = torch.from_numpy(new_indptr)
    colind_t = torch.from_numpy(new_indices_arr)
    values_t = torch.from_numpy(new_values_arr)
    if device:
        rowptr_t = rowptr_t.to(device, non_blocking=True)
        colind_t = colind_t.to(device, non_blocking=True)
        values_t = values_t.to(device, non_blocking=True)
    return rowptr_t, colind_t, values_t