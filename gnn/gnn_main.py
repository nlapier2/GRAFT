#!/usr/bin/env python3
import argparse, math, os
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

from models import GeneMPNN
from losses import compute_distance_loss, mse_loss, target_efficacy_loss, locality_damping, prototype_loss, prior_loss
from utils import to_numpy, build_target_to_gene_index, sample_minibatch, sample_minibatch_knn, make_base_adjacency, collapse_to_pseudobulk, make_pretrain_pseudobulk_from_adata, prep_external_data



# ----------------------------
# Training
# ----------------------------
def train(
    adata: ad.AnnData,
    target_label: str,
    control_label: str,
    hidden: int = 128,
    T: int = 2,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_target: float = 0.2,
    weight_local: float = 0.0,
    weight_mse: float = 0.0,
    weight_proto: float = 0.2,
    seed: int = 0,
    tau: float = 0.0,
    device: str = "cuda",
    match_controls: str = "knn",  # or "random"
    knn_k: int = 32,
    knn_temp: float = 0.1,
    knn_metric: str = "l2",  # or "cosine"
    dist_loss: str = "mmd",  # or "swd" or "energy"
    swd_projections: int = 128,
    weight_dist: float = 1.0,
    single_pert_batches: bool = False,
    W_meta: np.ndarray | None = None,
    init_from_meta: bool = False,
    weight_prior: float = 0.0,
    meta_topk: int = 0,
    model: nn.Module | None = None,
    pretrain_mode: bool = False,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Data arrays
    X = to_numpy(adata.X).astype(np.float32)  # assume normalized/log space already
    labels = adata.obs[target_label].astype(str).values
    G = adata.n_vars

    # Index pools
    ctrl_mask = labels == control_label
    if ctrl_mask.sum() == 0:
        raise ValueError("No control cells found.")
    # perturbed pool includes all non-controls (even if target gene not found)
    pert_mask = ~ctrl_mask
    if pert_mask.sum() == 0:
        raise ValueError("No perturbed cells found.")
    # unique perturbation labels (exclude control)
    pert_unique = sorted({l for l in labels if l != control_label})

    X_ctrl = X  # we’ll pick rows via indices
    X_pert = X
    pert_labels = labels
    # Precompute normalized rows for cosine kNN (one-time)
    pre_norm_ctrl = pre_norm_pert = None

    # Map perturbation label -> gene index (for Step-0); unknown => -1
    t2gi = build_target_to_gene_index(adata, target_label)
    # Precompute a tensor of target indices per cell
    tgt_idx = np.full(adata.n_obs, -1, dtype=np.int64)
    for i, lab in enumerate(labels):
        tgt_idx[i] = t2gi.get(lab, -1)

    # Model
    # Build stable mapping from label -> embedding row
    pert_names_unique = sorted(set(labels.tolist()))
    pert2row = {p: i for i, p in enumerate(pert_names_unique)}

    ctrl_mean_np = X[ctrl_mask].mean(axis=0).astype(np.float32)
    ctrl_mean = torch.from_numpy(ctrl_mean_np).to(device)

    # Dataset / cell-type categorical codes (only if present; safe to ignore otherwise)
    dset_codes = None
    ct_codes = None
    if "dataset_id" in adata.obs.columns:
        dset_codes = pd.Categorical(adata.obs["dataset_id"]).codes.astype(np.int64)
    if "cell_type" in adata.obs.columns:
        ct_col = adata.obs["cell_type"]
        # Casting to str avoids Categorical fillna errors for unseen categories
        if isinstance(ct_col.dtype, pd.CategoricalDtype):
            ct_col = ct_col.astype(str)
        else:
            ct_col = ct_col.astype(str)
        # Normalize missing/mixed to UNK
        ct_vals = ct_col.replace({"<NA>": "UNK", "mixed": "UNK"}).fillna("UNK")
        ct_codes = pd.Categorical(ct_vals).codes.astype(np.int64)

    # Model (reuse if provided)
    if model is None:
        prior_dim = W_meta.shape[0] if W_meta is not None else None
        model = GeneMPNN(G=G, hidden=hidden, T=T, tau=tau, prior_dim=prior_dim).to(device)
        # Optionally initialize node embeddings from prior
        if (W_meta is not None) and init_from_meta:
            Wm_torch = torch.from_numpy(W_meta.astype(np.float32)).to(device)  # (R,G)
            model.init_from_prior(Wm_torch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Base adjacency: either dense (as before) or prior-based top-k cosine graph
    if (W_meta is not None) and (meta_topk > 0):
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
        print(f"[graph] Using prior kNN graph (top-k={k}) from M_meta.")
    else:
        # dense fully-connected adjacency (previous behavior)
        A_base = make_base_adjacency(G, self_loops=True).to(device)

    # Simple schedule
    steps_per_epoch = math.ceil(pert_mask.sum() / batch_size)

    # Stage-1: build per-dataset control pseudobulks (if available)
    ctrl_by_dset = None
    if pretrain_mode and ("dataset_id" in adata.obs.columns):
        ctrl_by_dset = {}
        dsets = adata.obs["dataset_id"].astype(str).values
        for d in sorted(set(dsets)):
            m = (labels == control_label) & (dsets == d)
            if m.any():
                ctrl_by_dset[d] = X[m].mean(axis=0).astype(np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"mse": 0.0, "targ": 0.0, "loc": 0.0, "proto": 0.0, "dist": 0.0, "prior": 0.0, "tot": 0.0}
        for step in range(steps_per_epoch):
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
            # per-sample target index tensor
            tidx = torch.tensor([t2gi.get(t, -1) for t in btargets], dtype=torch.long)

            bx_ctrl = bx_ctrl.to(device)
            bx_pert = bx_pert.to(device)
            tidx = tidx.to(device)

            # dataset / cell-type indices pulled from the SAME rows as bx_pert
            z_d, z_ct = None, None
            # identify the row indices that produced bx_pert
            if 'sel_pert' in locals():           # pretrain_mode branch (or random sampler)
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

            # per-sample perturbation row indices for embedding / FiLM
            pert_rowidx = torch.tensor([pert2row[t] for t in btargets], dtype=torch.long, device=device)
            yhat, x0, alpha_vec = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx, dset_idx=z_d, ct_idx=z_ct)

            # --- Prototype / bulk-delta loss (aligns to PDS) ---
            # Group rows by perturbation label in this batch
            by_lbl = defaultdict(list)
            for i, lbl in enumerate(btargets):
                by_lbl[lbl].append(i)

            # In Stage-1 pretraining, pseudobulk may contain NaNs/-1 for missing genes; skip MSE entirely.
            loss_mse = (yhat - x0).new_tensor(0.0) if pretrain_mode else mse_loss(yhat - x0, bx_pert - x0)
            loss_loc = locality_damping(yhat, x0, tidx, weight=1.0) if weight_local > 0 else yhat.new_tensor(0.0)
            loss_proto = prototype_loss(yhat, x0, bx_pert, bx_ctrl, by_lbl, t2gi, pretrain_mode=pretrain_mode)
            loss_t = target_efficacy_loss(yhat, bx_ctrl, bx_pert, tidx, alpha_vec)
            loss_dist = compute_distance_loss(yhat, x0, bx_pert, btargets, by_lbl, t2gi, dist_loss, pretrain_mode, swd_projections)
            loss_prior = prior_loss(yhat, model, W_meta, weight_prior)

            if weight_mse == 0.0:
                loss_mse = loss_mse * 0.0
            if weight_target == 0.0:
                loss_t = loss_t * 0.0
            if weight_local == 0.0:
                loss_loc = loss_loc * 0.0
            if weight_proto == 0.0:
                loss_proto = loss_proto * 0.0
            if weight_dist == 0.0:
                loss_dist = loss_dist * 0.0
            if weight_prior == 0.0:
                loss_prior = loss_prior * 0.0

            loss = weight_mse * loss_mse \
                 + weight_target * loss_t \
                 + weight_local * loss_loc \
                 + weight_proto * loss_proto \
                 + weight_dist * loss_dist \
                 + weight_prior * loss_prior

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["mse"]  += float(loss_mse.item())
            running["targ"] += float(loss_t.item())
            running["loc"]  += float(loss_loc.item())
            running["proto"] += float(loss_proto.item())
            running["dist"] += float(loss_dist.item())
            running["prior"] += float(loss_prior.item())
            running["tot"]  += float(loss.item())

        denom = max(steps_per_epoch, 1)
        do_print = True
        if pretrain_mode and epoch != 1 and epoch % 20 != 0 and epoch != epochs:
            do_print = False
        if do_print:
            print(f"[epoch {epoch:03d}] "
                f"mse={running['mse']/denom:.5f}  "
                f"targ={running['targ']/denom:.5f}  "
                f"loc={running['loc']/denom:.5f}  "
                f"proto={running['proto']/denom:.5f}  "
                f"dist={running['dist']/denom:.5f}  "
                f"prior={running['prior']/denom:.5f}  "
                f"total={running['tot']/denom:.5f}")

    # estimate mean alpha on training targets (linear KD from pseudobulk)
    with torch.no_grad():
        # quick calc using control pseudobulk + training perts
        X = to_numpy(adata.X).astype(np.float32)
        labels = adata.obs[target_label].astype(str).values
        ctrl_mean = X[labels == control_label].mean(axis=0)
        t2gi = build_target_to_gene_index(adata, target_label)
        train_perts = sorted({l for l in labels if l != control_label})
        alphas = []
        for p in train_perts:
            idx = np.where(labels == p)[0]
            if len(idx)==0: continue
            t = t2gi.get(p, -1)
            if t < 0: continue
            true_t = np.expm1(X[idx, t]).mean()
            ctrl_t = np.expm1(ctrl_mean[t])
            a = np.clip(1.0 - true_t/(ctrl_t + 1e-8), 0.0, 1.0)
            alphas.append(a)
        model.register_buffer("alpha_mean_train", torch.tensor(float(np.mean(alphas) if alphas else 0.8)))

    return model


@torch.no_grad()
def predict_all_perturbations(
    adata: ad.AnnData,
    model: nn.Module,
    target_label: str,
    control_label: str,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 0,
):
    """
    For every perturbed cell, match a random control, run model, and collect predictions.
    Returns:
      pred_mat: (N_pert, G) predicted expressions (aligned to perturbed rows)
      true_mat: (N_pert, G) true perturbed expressions
      pert_names: list[str] of length N_pert (labels for each row)
      ctrl_mean: (G,) global control pseudobulk (mean of all control cells)
    """
    rng = np.random.default_rng(seed)
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    G = adata.n_vars

    # pools
    ctrl_idx = np.where(labels == control_label)[0]
    pert_idx = np.where(labels != control_label)[0]
    if len(ctrl_idx) == 0 or len(pert_idx) == 0:
        raise ValueError("Need both control and perturbed cells for evaluation.")

    # control pseudobulk (global)
    ctrl_mean = X[ctrl_idx].mean(axis=0)

    # target mapping (label -> gene index), -1 if not a gene in panel
    t2gi = build_target_to_gene_index(adata, target_label)

    # adjacency
    A_base = make_base_adjacency(G, self_loops=True).to(device)

    # allocate
    Np = len(pert_idx)
    pred_mat = np.zeros((Np, G), dtype=np.float32)
    true_mat = X[pert_idx]  # (Np,G)
    pert_names = labels[pert_idx].tolist()

    # stable mapping label -> row index (should match training order if same labels set)
    pert_names_unique = sorted(set(labels.tolist()))
    pert2row = {p: i for i, p in enumerate(pert_names_unique)}

    # batched forward with random control matches
    model.eval()
    for start in range(0, Np, batch_size):
        end = min(start + batch_size, Np)
        b_idx = np.arange(start, end)
        # random controls (with replacement)
        rand_ctrl = rng.choice(ctrl_idx, size=len(b_idx), replace=True)

        bx_ctrl = torch.from_numpy(X[rand_ctrl]).float().to(device)
        tidx = torch.tensor([t2gi.get(p, -1) for p in pert_names[start:end]], dtype=torch.long, device=device)
        pert_rowidx = torch.tensor([pert2row[p] for p in pert_names[start:end]], dtype=torch.long, device=device)

        yhat, _, _ = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx)
        pred_mat[b_idx] = yhat.detach().cpu().numpy()

    return pred_mat, true_mat, pert_names, ctrl_mean, pert_idx


def evaluate_model(
    adata: ad.AnnData,
    model: nn.Module,
    target_label: str,
    control_label: str,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 0,
):
    """
    Computes:
      - per-perturbation MAE
      - knockdown efficiency (abs & %) for true vs predicted at the target gene
      - perturbation similarity: mean & min pairwise Pearson corr between predicted mean effect vectors
      - PDS (Perturbation Discrimination Score): mean over perturbations
    Prints a concise report and returns a dict with all metrics.
    """
    pred_mat, true_mat, pert_names, ctrl_mean, pert_idx = predict_all_perturbations(
        adata, model, target_label, control_label, device=device, batch_size=batch_size, seed=seed
    )
    G = adata.n_vars
    df_obs = adata.obs
    labels = df_obs[target_label].astype(str).values

    # group indices by perturbation (excluding control)
    perts = sorted(set(pert_names))
    # target mapping
    t2gi = build_target_to_gene_index(adata, target_label)

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
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--out_pred_h5ad", type=str, default="",
                    help="If set, write an AnnData with predictions for the evaluation split.")
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--T", type=int, default=2, help="Number of message-passing steps.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_target", type=float, default=0.1)
    ap.add_argument("--weight_local", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0, help="Step-0 anchor (e.g., 0.0 for CRISPRi).")
    ap.add_argument("--use_pseudobulk", action="store_true",
                    help="Collapse to one mean row per perturbation (incl. control).")
    ap.add_argument("--match_controls", choices=["random", "knn"], default="knn",
                    help="How to choose a control for each perturbed cell.")
    ap.add_argument("--knn_k", type=int, default=32, help="Top-k controls to sample from.")
    ap.add_argument("--knn_temp", type=float, default=0.1, help="Softmax temperature over distances.")
    ap.add_argument("--knn_metric", choices=["l2", "cosine"], default="l2",
                    help="Distance metric for kNN control matching.")
    ap.add_argument("--test_pct_perts", type=float, default=0.0,
                    help="Fraction of perturbation labels (excluding control) to hold out for testing. 0.0 = no holdout.")
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional path to a separate test AnnData. If set, overrides --test_pct_perts.")
    ap.add_argument("--dist_loss", choices=["none","mmd","swd","energy"], default="mmd",
                    help="Distribution loss between predicted and true deltas per perturbation.")
    ap.add_argument("--weight_dist", type=float, default=1.0, help="Weight for distribution loss.")
    ap.add_argument("--swd_projections", type=int, default=128, help="Num random projections for SWD.")
    ap.add_argument("--single_pert_batches", action="store_true",
                    help="If set, each batch contains cells from a single perturbation label.")
    ap.add_argument("--meta_path", type=str, default="",
                    help="Path to M_meta.npy produced by embed_pathways.py (shape R x G; columns aligned to var_names).")
    ap.add_argument("--init_from_meta", action="store_true",
                    help="If set, initialize node embeddings from the projected pathway prior.")
    ap.add_argument("--weight_prior", type=float, default=0.0,
                    help="L2 proximity loss weight to keep node embeddings near the projected pathway prior (Phase 1).")
    ap.add_argument("--meta_topk", type=int, default=0,
                    help="If >0, build a top-k cosine kNN adjacency from the pathway prior instead of dense A.")
    ap.add_argument("--pretrain_pseudobulk", type=str, default="",
                    help="Path to a pseudobulk .h5ad for Stage-1 pretraining; empty = skip Stage-1")
    ap.add_argument("--pretrain_pseudobulk_list", type=str, default="",
                        help="Text file with one pseudobulk .h5ad path per line; blank/comment lines ignored")
    ap.add_argument("--include_target_pseudobulk", action="store_true",
                        help="Also pseudobulk the target dataset and include it in Stage-1 pretraining")
    ap.add_argument("--pretrain_epochs", type=int, default=10,
                    help="Epochs to run Stage-1 pseudobulk pretraining")
    ap.add_argument("--use_dset_embed", action="store_true",
                    help="Enable dataset_id embeddings (used in FiLM/proto conditioning)")
    ap.add_argument("--use_celltype_embed", action="store_true",
                    help="Enable cell_type embeddings (used in FiLM/proto conditioning)")
    ap.add_argument("--dset_embed_dim", type=int, default=16,
                    help="Dimensionality of dataset embedding (if enabled)")
    ap.add_argument("--ct_embed_dim", type=int, default=16,
                    help="Dimensionality of cell_type embedding (if enabled)")
    ap.add_argument("--missing_gene_fill", type=str, default="nan", choices=["nan", "-1"],
                        help="Placeholder used in pseudobulk for missing genes; masked in Stage-1 losses")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.in_h5ad)
    pb_target = None  # pseudobulked target data for Stage-1 pretraining
    if args.include_target_pseudobulk:
        pb_target = make_pretrain_pseudobulk_from_adata(adata, args.target_label, args.control_label, dataset_id="target_all")
        sc.pp.normalize_total(pb_target, inplace=True)
        sc.pp.log1p(pb_target)
    if args.use_pseudobulk:  # stage 2 pseudobulk
        args.batch_size = 1  # enforce single-row batches
        adata = collapse_to_pseudobulk(adata, args.target_label)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway

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

    # ----- Optional: load pathway prior M_meta.npy (R x G) -----
    W_meta = None
    if args.meta_path:
        print(f"[prior] Loading pathway meta from {args.meta_path}")
        W_meta = np.load(args.meta_path)
        assert W_meta.ndim == 2 and W_meta.shape[1] == adata_train.n_vars, \
            f"M_meta shape mismatch: got {W_meta.shape}, expected (R,{adata_train.n_vars}) aligned to var_names."
        
    # Optional Stage-1: pseudobulk pretraining (reuses the same train() loop)
    model = None
    pb_paths = []
    pbs = []
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
        print(f"=== Stage-1: pretraining on {len(pbs)} pseudobulk sources; total rows: {pb_all.n_obs} ===")

        model = train(
            adata=pb_all,
            target_label=args.target_label,
            control_label=args.control_label,
            hidden=args.hidden,
            T=args.T,
            epochs=args.pretrain_epochs,
            batch_size=1,
            lr=args.lr,
            weight_target=args.weight_target,     # keep α supervision for gene perts
            weight_local=0.0,
            seed=args.seed,
            tau=args.tau,
            device=args.device,
            match_controls="random",              # ignored in pretrain_mode due to per-dataset controls
            knn_k=args.knn_k,
            knn_temp=args.knn_temp,
            knn_metric=args.knn_metric,
            dist_loss="none",                     # no distribution loss in Stage-1
            weight_dist=0.0,
            swd_projections=args.swd_projections,
            single_pert_batches=False,
            W_meta=W_meta,
            init_from_meta=args.init_from_meta,
            weight_prior=args.weight_prior,
            meta_topk=args.meta_topk,
            model=None,
            pretrain_mode=True,
        )

    print(f"=== Stage-2: training on {'train+test' if adata_test is not None else 'train'} set ===")
    model = train(
        adata=adata_train,
        target_label=args.target_label,
        control_label=args.control_label,
        hidden=args.hidden,
        T=args.T,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_target=args.weight_target,
        weight_local=args.weight_local,
        seed=args.seed,
        tau=args.tau,
        device=args.device,
        match_controls=args.match_controls,
        knn_k=args.knn_k,
        knn_temp=args.knn_temp,
        knn_metric=args.knn_metric,
        dist_loss=args.dist_loss,
        weight_dist=args.weight_dist,
        swd_projections=args.swd_projections,
        single_pert_batches=args.single_pert_batches,
        W_meta=W_meta,
        init_from_meta=args.init_from_meta,
        weight_prior=args.weight_prior,
        meta_topk=args.meta_topk,
        model=model,  # continue from Stage-1 if done
        pretrain_mode=False,
    )

    # Evaluate: external test if provided, else held-out split, else train split
    eval_adata = adata_test if adata_test is not None else adata_train
    print("\n=== Evaluation on {} set ===".format("TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)"))
    eval_metrics = evaluate_model(
        adata=eval_adata,
        model=model,
        target_label=args.target_label,
        control_label=args.control_label,
        device=args.device,
        batch_size=512,
        seed=args.seed,
    )

    # Optional: save weights
    out_path = os.path.splitext(args.in_h5ad)[0] + f".mpnn_hidden{args.hidden}_T{args.T}.pt"
    torch.save({"state_dict": model.state_dict(),
                "G": model.G,
                "hidden": model.hidden,
                "T": model.T}, out_path)

    # ---------------------------
    # Optional: write predictions AnnData for the evaluation split
    # ---------------------------
    if args.out_pred_h5ad:
        print(f"\n[write] Generating predictions AnnData → {args.out_pred_h5ad}")
        # run the same batched predictor to get per-pert predictions + their row indices
        pred_mat, _, pert_names_eval, _, pert_idx = predict_all_perturbations(
            eval_adata, model, args.target_label, args.control_label,
            device=args.device, batch_size=512, seed=args.seed
        )
        # start from a copy of eval_adata.X and replace perturbed rows with predictions
        X_eval = to_numpy(eval_adata.X).astype(np.float32, copy=True)
        X_eval[pert_idx, :] = pred_mat  # controls remain unchanged
        ad_pred = ad.AnnData(X_eval, obs=eval_adata.obs.copy(), var=eval_adata.var.copy())
        ad_pred.write_h5ad(args.out_pred_h5ad, compression="lzf")
        eval_adata.write_h5ad(os.path.splitext(args.out_pred_h5ad)[0] + ".true.h5ad", compression="lzf")
        print(f"[done] Wrote {args.out_pred_h5ad} (cells={ad_pred.n_obs}, genes={ad_pred.n_vars})")

    print(f"[done] saved model to {out_path}")

if __name__ == "__main__":
    main()
