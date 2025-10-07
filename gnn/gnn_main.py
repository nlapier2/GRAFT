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
from losses import *
from utils import *


def parse_arguments():
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
    ap.add_argument("--weight_mse", type=float, default=0.0, help="Weight for per-cell MSE loss.")
    ap.add_argument("--weight_proto", type=float, default=0.2, help="Weight for prototype loss.")
    ap.add_argument("--node_dim", type=int, default=128, help="Dimensionality of gene node embeddings.")
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
    ap.add_argument("--dset_embed_dim", type=int, default=0,
                    help="Dimensionality of dataset embedding (if enabled)")
    ap.add_argument("--ct_embed_dim", type=int, default=0,
                    help="Dimensionality of cell_type embedding (if enabled)")
    ap.add_argument("--missing_gene_fill", type=str, default="nan", choices=["nan", "-1"],
                        help="Placeholder used in pseudobulk for missing genes; masked in Stage-1 losses")
    ap.add_argument("--weight_contrast", type=float, default=0.0,
                    help="Weight of InfoNCE contrastive loss (pred-bulk vs obs-bulk).")
    ap.add_argument("--proj_dim", type=int, default=128,
                    help="Projection dim for contrastive embeddings.")
    ap.add_argument("--contrast_tau", type=float, default=0.1,
                    help="InfoNCE temperature.")
    ap.add_argument("--queue_size", type=int, default=64,
                    help="Number of negative keys to keep in memory queue.")
    ap.add_argument("--neg_k", type=int, default=16,
                    help="Sample at most K negatives per step for InfoNCE.")
    ap.add_argument("--neg_cap_per_label", type=int, default=4,
                    help="Keep at most this many keys per pert label in the queue.")
    ap.add_argument("--contrast_query_type", type=str, default="context",
                    choices=["context", "delta"],
                    help="Use 'context' (target+dset+ct+alpha) or 'delta' (predicted pseudobulk)")
    ap.add_argument("--load_model_path", type=str, default="",
                    help="Path to a saved checkpoint (.pt). If set, training will start from these weights.")
    ap.add_argument("--save_model_path", type=str, default="",
                    help="Where to save the trained model (.pt). If empty, an auto name based on --in_h5ad is used.")
    ap.add_argument('--use_sparse_topk', action='store_true', help='Use candidate CSR + Top-K attention')
    ap.add_argument('--topk_keep', type=int, default=12, help='Neighbors kept per node after attention')
    ap.add_argument('--num_tokens', type=int, default=0, help='Global tokens R (0=off)')
    ap.add_argument('--token_dim', type=int, default=0, help='Token dim (0 => hidden)')
    ap.add_argument("--similarity_npz", type=str, default="", help="Precomputed gene-gene similarity CSR .npz")
    ap.add_argument('--weight_edge_l1', type=float, default=0.0,
                     help='L1 weight for learned edge strengths (sparse SpMM path)')
    ap.add_argument('--learn_dense_edges', action='store_true', help='Learn dense edge strengths (default: False)')
    ap.add_argument('--remove_non_gene_perts', action='store_true', help='Remove non-gene perturbation labels')
    ap.add_argument('--eval_on_train', action='store_true', help='Evaluate on training set in addition to test set')
    ap.add_argument('--test_zero_adj', action='store_true', help='For ablation: use zero adjacency during testing')
    ap.add_argument('--write_test', action='store_true', help='Write true test set')
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return args


# --- QUICK TOGGLES (just set True to try) ---
SCRAMBLE_ACROSS_CELLS_PER_GENE = False   # column-wise: for each gene g, shuffle x[:, g] across the batch
SCRAMBLE_WITHIN_EACH_CELL      = False  # row-wise: for each cell b, shuffle x[b, :] across genes
RUN_MGM_PRETRAIN               = False  # run masked gene pretraining before main training (always on if --use_sparse_topk and no similarity_npz)

def scramble_across_cells_per_gene(x: torch.Tensor) -> torch.Tensor:
    """
    For each gene g, randomly permute the values across cells in the batch.
    x: (B,G) or (B,G,C)
    """
    B, G = x.shape[:2]
    # idx has shape (B,G): each column g is a permutation of [0..B-1]
    idx = torch.stack([torch.randperm(B, device=x.device) for _ in range(G)], dim=1)
    if x.dim() == 2:
        return x.gather(0, idx)
    else:
        # expand idx to (B,G,1) to gather the first dim and keep channels intact
        return x.gather(0, idx.unsqueeze(-1).expand(-1, -1, x.size(2)))

def scramble_within_each_cell(x: torch.Tensor) -> torch.Tensor:
    """
    For each cell b, randomly permute its gene positions.
    x: (B,G) or (B,G,C)
    """
    B, G = x.shape[:2]
    # idx has shape (B,G): each row b is a permutation of [0..G-1]
    idx = torch.stack([torch.randperm(G, device=x.device) for _ in range(B)], dim=0)
    if x.dim() == 2:
        return x.gather(1, idx)
    else:
        # expand idx to (B,G,1) to gather along gene dim and keep channels intact
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(2)))


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
    node_dim: int = 128,
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
    weight_contrast: float = 0.0,
    proj_dim: int = 128,
    contrast_tau: float = 0.1,
    queue_size: int = 64,
    neg_k: int = 16,
    neg_cap_per_label: int = 4,
    optimizer_state: Dict | None = None,    
    contrast_query_type: str = "context",
    dset_embed_dim: int = 0,
    ct_embed_dim: int = 0,
    use_sparse_topk: bool = False,
    topk_keep: int = 12,
    num_tokens: int = 0,
    token_dim: int = 0,
    similarity_npz: str = "",
    weight_edge_l1: float = 0.0,
    learn_dense_edges: bool = False
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
    # Precompute normalized rows for cosine kNN (one-time)
    pre_norm_ctrl = pre_norm_pert = None
    pert_rowidx = None

    # Map perturbation label -> gene index (for Step-0); unknown => -1
    t2gi = build_target_to_gene_index(adata, target_label)
    # Map NON-gene perturbation label -> small embedding row (0..K-1), for drugs/OOF genes
    all_perts = sorted({l for l in labels if l != control_label})
    gene_set = set(adata.var_names.tolist())
    extra_labels = [p for p in all_perts if p not in gene_set]
    extra2i = {p: i for i, p in enumerate(extra_labels)}

    # Build stable mapping from label -> embedding row
    pert_names_unique = sorted(set(labels.tolist()))
    pert2row = {p: i for i, p in enumerate(pert_names_unique)}
    ctrl_mean_np = X[ctrl_mask].mean(axis=0).astype(np.float32)
    ctrl_mean = torch.from_numpy(ctrl_mean_np).to(device)

    # Base adjacency: either dense (as before) or prior-based top-k cosine graph
    if (W_meta is not None) and (meta_topk > 0):
        A_base, k = make_adjacency_prior(W_meta, meta_topk, G, device)
        print(f"[graph] Using prior kNN graph (top-k={k}) from M_meta.")
    else:
        # dense fully-connected adjacency (previous behavior)
        A_base = make_base_adjacency(G, self_loops=True).to(device)

    # Create Model (reuse if provided)
    if model is None:
        prior_dim = W_meta.shape[0] if W_meta is not None else None

        # --- infer dataset / cell-type vocab sizes from this adata ---
        # Use embeddings only if columns exist and dims are >0
        dset_vocab = 0
        ct_vocab   = 0
        dset_dim_  = 0
        ct_dim_    = 0
        if ("dataset_id" in adata.obs.columns) and (dset_embed_dim > 0):
            # stable category list (works for strings)
            dset_vocab = int(pd.Categorical(adata.obs["dataset_id"].astype(str)).categories.size)
            dset_dim_  = dset_embed_dim
        if ("cell_type" in adata.obs.columns) and (ct_embed_dim > 0):
            ct_vocab = int(pd.Categorical(adata.obs["cell_type"].astype(str)).categories.size)
            ct_dim_  = ct_embed_dim

        model = GeneMPNN(
            G=G, A_base=A_base, device=device, hidden=hidden, T=T, tau=tau, node_dim=node_dim, prior_dim=prior_dim,
            dset_vocab=dset_vocab, dset_dim=dset_dim_, ct_vocab=ct_vocab, ct_dim=ct_dim_,
            proj_dim=proj_dim, use_sparse_topk=use_sparse_topk, topk_keep=topk_keep,
            num_tokens=num_tokens, token_dim=token_dim, learn_dense_edges=learn_dense_edges,
            num_extra_perts=len(extra2i)
        ).to(device)
        if RUN_MGM_PRETRAIN:
            print("=== MGM pretraining: learning sparse graph from masked-gene recovery ===")
            masked_graph_pretrain(model, adata, device, steps=250, batch_size=min(256, batch_size), mask_p=0.15, K=64, refresh_every=10)

    if use_sparse_topk:
        genes_order = list(map(str, adata.var_names))
        if not RUN_MGM_PRETRAIN:
            rowptr, colind, values = load_similarity_npz(similarity_npz, genes_order, device=device)
            # model.set_candidate_csr(rowptr, colind, values)  # used for attention layers, no longer used
            model.set_sparse_A(rowptr, colind, values)

        # --- store category→row maps for reuse in later stages ---
        if ("dataset_id" in adata.obs.columns) and (getattr(model, "dset_E", None) is not None):
            dcat = pd.Categorical(adata.obs["dataset_id"].astype(str))
            model.dset_id2row = {cat: i for i, cat in enumerate(dcat.categories)}
        if ("cell_type" in adata.obs.columns) and (getattr(model, "ct_E", None) is not None):
            ccat = pd.Categorical(adata.obs["cell_type"].astype(str))
            model.ct_id2row   = {cat: i for i, cat in enumerate(ccat.categories)}
        # Optionally initialize node embeddings from prior
        if (W_meta is not None) and init_from_meta:
            Wm_torch = torch.from_numpy(W_meta.astype(np.float32)).to(device)  # (R,G)
            model.init_from_prior(Wm_torch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if optimizer_state is not None:
        try:
            opt.load_state_dict(optimizer_state)
            expand_adam_states_for_embeddings(opt)
            print("[train] optimizer state loaded")
        except Exception as e:
            print(f"[train] optimizer state load failed: {e}")

    # --- simple FIFO queue for contrastive loss negative keys (L2-normalized embeddings) ---
    neg_bank = torch.empty((0, proj_dim), device=device)
    neg_labels: list[str] = []  # list[str] same length as neg_bank rows
    def _enqueue(k: torch.Tensor, lbl: str):
        nonlocal neg_bank, neg_labels
        # k: (1,D) or (B,D) -> ensure 2D
        if k.ndim == 1:
            k = k.unsqueeze(0)
        if k.size(0) > 1:
            # take first row (we use one pseudobulk per batch)
            k = k[:1]
        # enforce per-label cap
        if neg_labels.count(lbl) >= neg_cap_per_label:
            # find first index for this label, drop it
            idx = next(i for i, L in enumerate(neg_labels) if L == lbl)
            keep_mask = torch.ones(len(neg_labels), dtype=torch.bool, device=neg_bank.device)
            keep_mask[idx] = False
            neg_bank = neg_bank[keep_mask]
            del neg_labels[idx]
        # append
        neg_bank = torch.cat([neg_bank, k.detach()], dim=0)
        neg_labels.append(lbl)
        # trim
        if neg_bank.size(0) > queue_size:
            cut = neg_bank.size(0) - queue_size
            neg_bank = neg_bank[cut:]
            del neg_labels[:cut]

    # EMA centroids for observed pseudobulk deltas per (pert,dset)
    ema_centroid = {}   # dict[(label, dset_id)] -> torch.Tensor (1, G)
    ema_beta = 0.9

    # Simple schedule
    steps_per_epoch = math.ceil(pert_mask.sum() / batch_size)
    print('Steps per epoch:', steps_per_epoch)

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
        running = {"mse": 0.0, "targ": 0.0, "loc": 0.0, "proto": 0.0, "dist": 0.0, "prior": 0.0, "contrast": 0.0, "edge_l1": 0.0, "tot": 0.0}
        for step in range(steps_per_epoch):
            bx_ctrl, bx_pert, btargets, sel_pert = sample_batch_by_mode(
                single_pert_batches, pretrain_mode, rng, pert_unique,
                X, labels, control_label, ctrl_by_dset,
                ctrl_mean_np, match_controls, knn_k, knn_temp, knn_metric,
                X_ctrl, X_pert, labels, batch_size, adata, pre_norm_ctrl, pre_norm_pert
            )

            # per-sample target index tensor
            tidx = torch.tensor([t2gi.get(t, -1) for t in btargets], dtype=torch.long)
            pidx = torch.tensor([extra2i.get(t, -1) for t in btargets], device=device, dtype=torch.long)
            bx_ctrl = bx_ctrl.to(device)
            bx_pert = bx_pert.to(device)
            tidx = tidx.to(device)
            pidx = pidx.to(device)

            if SCRAMBLE_ACROSS_CELLS_PER_GENE:
                bx_ctrl = scramble_across_cells_per_gene(bx_ctrl)
                bx_pert = scramble_across_cells_per_gene(bx_pert)

            if SCRAMBLE_WITHIN_EACH_CELL:
                bx_ctrl = scramble_within_each_cell(bx_ctrl)
                bx_pert = scramble_within_each_cell(bx_pert)

            # per-sample perturbation row indices for embedding / FiLM
            pert_rowidx = torch.tensor([pert2row[t] for t in btargets], dtype=torch.long, device=device)
            # dataset / cell-type indices pulled from the SAME rows as bx_pert
            z_d, z_ct = get_dset_indices(sel_pert, pert_rowidx, adata, device, model=model)
            yhat, x0, alpha_vec = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx, dset_idx=z_d, ct_idx=z_ct, pidx=pidx)

            # --- Prototype / bulk-delta loss (aligns to PDS) ---
            # Group rows by perturbation label in this batch
            by_lbl = defaultdict(list)
            for i, lbl in enumerate(btargets):
                by_lbl[lbl].append(i)

            # In Stage-1 pretraining, pseudobulk may contain NaNs/-1 for missing genes; skip MSE entirely.
            loss_mse = (yhat - x0).new_tensor(0.0) if pretrain_mode else mse_loss(yhat - x0, bx_pert - x0)
            loss_loc = locality_damping(yhat, x0, tidx, weight=1.0) if weight_local > 0 else yhat.new_tensor(0.0)
            loss_proto = prototype_loss(yhat, x0, bx_pert, bx_ctrl, by_lbl, t2gi, pretrain_mode=pretrain_mode)
            # loss_t = target_efficacy_loss(yhat, bx_ctrl, bx_pert, tidx, alpha_vec)
            loss_t = target_efficacy_batch_loss(bx_ctrl, bx_pert, tidx, alpha_vec, btargets)
            loss_dist = compute_distance_loss(yhat, x0, bx_pert, btargets, by_lbl, t2gi, dist_loss, pretrain_mode, swd_projections)
            loss_prior = prior_loss(yhat, model, W_meta, weight_prior)

            # === Contrastive retrieval on pseudobulk deltas (single-pert batches) ===
            if weight_contrast > 0.0:
                # observed batch pseudobulk delta (log1p space)
                delta_obs = (bx_pert - bx_ctrl).mean(dim=0, keepdim=True)      # (1,G)
                # predicted batch pseudobulk delta
                delta_pred = (yhat - x0).mean(dim=0, keepdim=True)             # (1,G)
                # IMPORTANT: drop the target gene column from both deltas to match
                # the model's constraint y[:, t] := x0[:, t] (predicted delta at t is 0).
                t = int(tidx[0].item()) if tidx.numel() > 0 else -1
                if t >= 0:
                    # clone to avoid modifying upstream tensors
                    delta_obs  = delta_obs.clone()
                    delta_pred = delta_pred.clone()
                    delta_obs[0, t]  = 0.0
                    delta_pred[0, t] = 0.0
                # embeddings
                # === EMA centroid key ===
                cur_lbl = str(btargets[0])
                dset_id = int(z_d[0].item()) if z_d is not None else -1
                key_id = (cur_lbl, dset_id)
                with torch.no_grad():
                    if key_id in ema_centroid:
                        ema_centroid[key_id] = ema_beta * ema_centroid[key_id] + (1 - ema_beta) * delta_obs
                    else:
                        ema_centroid[key_id] = delta_obs
                    key_vec = ema_centroid[key_id]
                k_pos = model.project_key(delta_obs)        # (1,D), no grad
                # === NEW: decoupled query from CONTEXT (default) ===
                if contrast_query_type == "context":
                    # Use the *same batch* context: target index, alpha from forward, and optional dataset/celltype ids
                    q = model.project_query_from_context(
                        target_idx=tidx,                 # (B,)
                        alpha=alpha_vec.detach(),        # (B,) use grad if you want InfoNCE to shape alpha; detach to keep alpha head independent
                        dset_idx=z_d if z_d is not None else None,
                        ct_idx=z_ct if z_ct is not None else None
                    ).mean(dim=0, keepdim=True)          # (1,D) use batch-mean context query (stable)
                else:
                    # Old behavior: query from predicted pseudobulk delta
                    q = model.project_query(delta_pred)   # (1,D)
                # mask out negatives with same pert label (avoid trivial collisions)
                cur_lbl = str(btargets[0])
                # mask out same-label negatives
                if len(neg_labels) > 0:
                    keep_mask = torch.tensor([lbl != cur_lbl for lbl in neg_labels],
                                             device=device, dtype=torch.bool)
                    bank = neg_bank[keep_mask]
                    labels_kept = [L for L, m in zip(neg_labels, keep_mask.tolist()) if m]
                    # subsample up to K negatives
                    if bank.size(0) > neg_k:
                        idx = torch.randperm(bank.size(0), device=device)[:neg_k]
                        bank = bank[idx]
                        # labels_kept becomes subset only for logging; not needed below
                else:
                    bank = neg_bank
                loss_contrast = info_nce_loss(q, k_pos, bank, tau=contrast_tau)
            else:
                loss_contrast = torch.tensor(0.0, device=device)

            if use_sparse_topk and (weight_edge_l1 > 0.0):
                loss_edge_l1 = model.edge_l1()
            elif learn_dense_edges and (weight_edge_l1 > 0.0):
                W = torch.softmax(model.dense_edge_logit, dim=1)
                ent = -(W * (W + 1e-12).log()).sum(dim=1).mean()   # mean over rows
                loss_edge_l1 = ent
            else:
                loss_edge_l1 = torch.tensor(0.0, device=device)

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
            if weight_contrast == 0.0:
                loss_contrast = loss_contrast * 0.0

            loss = weight_mse * loss_mse \
                 + weight_target * loss_t \
                 + weight_local * loss_loc \
                 + weight_proto * loss_proto \
                 + weight_dist * loss_dist \
                 + weight_prior * loss_prior \
                 + weight_contrast * loss_contrast \
                 + weight_edge_l1 * loss_edge_l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # enqueue current positive key for future negatives
            if weight_contrast > 0.0:
                _enqueue(k_pos, cur_lbl)

            running["mse"]  += float(loss_mse.item())
            running["targ"] += float(loss_t.item())
            running["loc"]  += float(loss_loc.item())
            running["proto"] += float(loss_proto.item())
            running["dist"] += float(loss_dist.item())
            running["prior"] += float(loss_prior.item())
            running["contrast"] += float(loss_contrast.item())
            running["edge_l1"] += float(loss_edge_l1.item())
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
                f"contrast={running['contrast']/denom:.5f}  "
                f"edge_l1={running['edge_l1']/denom:.5f}  "
                f"total={running['tot']/denom:.5f}")
            print_edge_weight_stats(model, prefix=f"epoch{epoch:03d}")

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

    return model, opt


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

    # Reuse the adjacency the model was built/trained with (dense or prior-kNN).
    # For sparse SpMM, A_base is ignored inside forward, but we pass it for API parity.
    A_base = getattr(model, "A_base", None)
    if A_base is None:
        print('[predict] Warning: model has no A_base attribute; using dense adjacency.')
        A_base = make_base_adjacency(G, self_loops=True).to(device)
    else:
        A_base = A_base.to(device)

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
        z_d, z_ct = get_dset_indices(None, pert_rowidx, adata, device, model=model)

        yhat, _, _ = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx, dset_idx=z_d, ct_idx=z_ct)
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
    args = parse_arguments()

    # ---------------------------
    # Read input data
    # ---------------------------
    adata = ad.read_h5ad(args.in_h5ad)
    adata.obs['dataset_id'] = "target_all"
    adata.obs['cell_type'] = "UNK"
    pb_target = None  # pseudobulked target data for Stage-1 pretraining
    if args.include_target_pseudobulk:
        pb_target = make_pretrain_pseudobulk_from_adata(adata, args.target_label, args.control_label, dataset_id="target_all")
        sc.pp.normalize_total(pb_target, inplace=True)
        sc.pp.log1p(pb_target)
    if args.use_pseudobulk:  # stage 2 pseudobulk
        args.batch_size = 1  # enforce single-row batches
        adata = collapse_to_pseudobulk(adata, args.target_label)
        adata.obs['dataset_id'] = "target_all"
        adata.obs['cell_type'] = "UNK"
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway
    # train/test split
    adata_train, adata_test, pb_target = train_test_split(args, adata, pb_target)
    adata_train.obs['dataset_id'] = "target_all"
    adata_train.obs['cell_type'] = "UNK"
    adata_test.obs['dataset_id'] = "target_all"
    adata_test.obs['cell_type'] = "UNK"

    # ----- Optional: load pathway prior M_meta.npy (R x G) -----
    W_meta = None
    if args.meta_path:
        print(f"[prior] Loading pathway meta from {args.meta_path}")
        W_meta = np.load(args.meta_path)
        assert W_meta.ndim == 2 and W_meta.shape[1] == adata_train.n_vars, \
            f"M_meta shape mismatch: got {W_meta.shape}, expected (R,{adata_train.n_vars}) aligned to var_names."
        
    # ---------------------------
    # Optional Stage-1: pseudobulk pretraining (reuses the same train() loop)
    # ---------------------------
    model = None
    resume_model = None
    resume_opt_state = None
    pb_all, pb_len = prep_pb_all(pb_target, adata_train, args)

    if pb_all is not None:
        print(f"=== Stage-1: pretraining on {pb_len} pseudobulk sources; total rows: {pb_all.n_obs} ===")
        # Optionally resume Stage-1 from a full checkpoint
        if args.load_model_path:
            # Model & weights are already loaded (with embedding expansion) by the builder:
            resume_model = build_model_for_dataset(pb_all, args, load_weights_from=args.load_model_path)
            # Only take optimizer state from the checkpoint; do NOT reload model weights again.
            _, opt_state, _, _ = load_full_checkpoint(args.load_model_path, device=args.device)
            resume_opt_state = opt_state
        model, opt = train(
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
            weight_mse=0.0,                       # no MSE loss in Stage-1
            weight_proto=args.weight_proto,
            node_dim=args.node_dim,
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
            model=resume_model,  # continue from checkpoint if given
            pretrain_mode=True,
            weight_contrast=args.weight_contrast,
            proj_dim=args.proj_dim,
            contrast_tau=args.contrast_tau,
            queue_size=args.queue_size,
            neg_k=args.neg_k,
            neg_cap_per_label=args.neg_cap_per_label,
            optimizer_state=resume_opt_state,
            contrast_query_type=args.contrast_query_type,
            dset_embed_dim=args.dset_embed_dim,
            ct_embed_dim=args.ct_embed_dim,
            use_sparse_topk=args.use_sparse_topk,
            topk_keep=args.topk_keep,
            num_tokens=args.num_tokens,
            token_dim=args.token_dim,
            similarity_npz=args.similarity_npz,
            weight_edge_l1=args.weight_edge_l1,
            learn_dense_edges=args.learn_dense_edges
        )

    # ---------------------------
    # Stage-2: train on target dataset (with or without held-out perts)
    # ---------------------------
    print(f"=== Stage-2: training on {'train+test' if adata_test is not None else 'train'} set ===")
    # If resuming Stage-2
    if (model is None) and args.load_model_path:
        # Model & weights are already loaded (with embedding expansion) by the builder:
        model = build_model_for_dataset(adata_train, args, load_weights_from=args.load_model_path)
        # Only take optimizer state from the checkpoint; do NOT reload model weights again.
        _, opt_state, _, _ = load_full_checkpoint(args.load_model_path, device=args.device)
        resume_opt_state = opt_state
    elif (model is None) and args.load_model_path:
        model = build_model_for_dataset(adata_train, args, load_weights_from=args.load_model_path)
        resume_opt_state = None
    model, opt = train(
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
        weight_mse=args.weight_mse,
        weight_proto=args.weight_proto,
        node_dim=args.node_dim,
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
        weight_contrast=args.weight_contrast,
        proj_dim=args.proj_dim,
        contrast_tau=args.contrast_tau,
        queue_size=args.queue_size,
        neg_k=args.neg_k,
        neg_cap_per_label=args.neg_cap_per_label,
        optimizer_state=resume_opt_state,
        contrast_query_type=args.contrast_query_type,
        dset_embed_dim=args.dset_embed_dim,
        ct_embed_dim=args.ct_embed_dim,
        use_sparse_topk=args.use_sparse_topk,
        topk_keep=args.topk_keep,
        num_tokens=args.num_tokens,
        token_dim=args.token_dim,
        similarity_npz=args.similarity_npz,
        weight_edge_l1=args.weight_edge_l1,
        learn_dense_edges=args.learn_dense_edges
    )

    # ---------------------------
    # Evaluate: external test if provided, else held-out split, else train split
    # ---------------------------
    eval_adata = adata_test if adata_test is not None else adata_train
    print("\n=== Evaluation on {} set ===".format("TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)"))
    eval_metrics = evaluate_model(
        adata=eval_adata,
        model=model,
        target_label=args.target_label,
        control_label=args.control_label,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.test_zero_adj:
        print("\n=== Additional evaluation with ZERO adjacency ===")
        # Temporarily replace model adjacency with identity
        if hasattr(model, "A_base"):
            orig_A = model.A_base
            G = orig_A.size(0)
            model.A_base = orig_A * 0.0 # torch.eye(G, device=orig_A.device) * (1.0 / G)
            _ = evaluate_model(
                adata=eval_adata,
                model=model,
                target_label=args.target_label,
                control_label=args.control_label,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            model.A_base = orig_A  # restore
        else:
            print("[warning] model has no A_base attribute; skipping zero-adj eval.")

    if args.eval_on_train and (adata_test is not None):
        print("\n=== Additional evaluation on TRAIN set ===")
        _ = evaluate_model(
            adata=adata_train,
            model=model,
            target_label=args.target_label,
            control_label=args.control_label,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if args.test_zero_adj:
            print("\n=== Additional evaluation on TRAIN with ZERO adjacency ===")
            # Temporarily replace model adjacency with identity
            if hasattr(model, "A_base"):
                orig_A = model.A_base
                G = orig_A.size(0)
                model.A_base = orig_A * 0.0 # torch.eye(G, device=orig_A.device) * (1.0 / G)
                _ = evaluate_model(
                    adata=adata_train,
                    model=model,
                    target_label=args.target_label,
                    control_label=args.control_label,
                    device=args.device,
                    batch_size=args.batch_size,
                    seed=args.seed,
                )
                model.A_base = orig_A  # restore
            else:
                print("[warning] model has no A_base attribute; skipping zero-adj eval.")

    # Save model weights (user-specified path if provided)
    if args.save_model_path:
        save_full_checkpoint(args.save_model_path, model, opt, extra_meta={
            "dset_id2row": getattr(model, "dset_id2row", None),
            "ct_id2row": getattr(model, "ct_id2row", None),
        })
        print(f"[done] saved model to {args.save_model_path}")


    # ---------------------------
    # Optional: write predictions AnnData for the evaluation split
    # ---------------------------
    if args.out_pred_h5ad:
        print(f"\n[write] Generating predictions AnnData → {args.out_pred_h5ad}")
        # run the same batched predictor to get per-pert predictions + their row indices
        pred_mat, _, pert_names_eval, _, pert_idx = predict_all_perturbations(
            eval_adata, model, args.target_label, args.control_label,
            device=args.device, batch_size=args.batch_size, seed=args.seed
        )
        # start from a copy of eval_adata.X and replace perturbed rows with predictions
        X_eval = to_numpy(eval_adata.X).astype(np.float32, copy=True)
        X_eval[pert_idx, :] = pred_mat  # controls remain unchanged
        ad_pred = ad.AnnData(X_eval, obs=eval_adata.obs.copy(), var=eval_adata.var.copy())
        ad_pred.write_h5ad(args.out_pred_h5ad, compression="lzf")
        eval_adata.write_h5ad(os.path.splitext(args.out_pred_h5ad)[0] + ".true.h5ad", compression="lzf")
        print(f"[done] Wrote {args.out_pred_h5ad} (cells={ad_pred.n_obs}, genes={ad_pred.n_vars})")

        if args.write_test:
            adata_test.write_h5ad(args.out_pred_h5ad + '_true.h5ad', compression="lzf")


if __name__ == "__main__":
    main()
