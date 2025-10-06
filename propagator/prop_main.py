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

from models import *
from losses import *
from utils import *


def parse_arguments():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--external_list", type=str, default="",
                        help="Text file with one pseudobulk .h5ad path per line; blank/comment lines ignored")
    ap.add_argument("--out_pred_h5ad", type=str, default="",
                    help="If set, write an AnnData with predictions for the evaluation split.")
    
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--T", type=int, default=2, help="Number of message-passing steps.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--weight_target", type=float, default=0.1)
    ap.add_argument("--weight_mse", type=float, default=0.0, help="Weight for per-cell MSE loss.")
    ap.add_argument("--weight_proto", type=float, default=0.2, help="Weight for prototype loss.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0, help="Step-0 anchor (e.g., 0.0 for CRISPRi).")

    ap.add_argument("--test_pct_perts", type=float, default=0.0,
                    help="Fraction of perturbation labels (excluding control) to hold out for testing. 0.0 = no holdout.")
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional path to a separate test AnnData. If set, overrides --test_pct_perts.")
    ap.add_argument('--eval_on_train', action='store_true', help='Evaluate on training set in addition to test set')

    ap.add_argument("--dset_embed_dim", type=int, default=16,
                    help="Dimensionality of dataset embedding")
    ap.add_argument("--ct_embed_dim", type=int, default=16,
                    help="Dimensionality of cell_type embedding")
    ap.add_argument("--missing_gene_fill", type=str, default="nan", choices=["nan", "-1"],
                        help="Placeholder used in pseudobulk for missing genes; masked in Stage-1 losses")

    ap.add_argument("--load_model_path", type=str, default="",
                    help="Path to a saved checkpoint (.pt). If set, training will start from these weights.")
    ap.add_argument("--save_model_path", type=str, default="",
                    help="Where to save the trained model (.pt). If empty, an auto name based on --in_h5ad is used.")
    
    ap.add_argument("--topk_keep", type=int, default=64, help="Sparsity level for gene-gene graph.")
    ap.add_argument("--qk_dim", type=int, default=32, help="Dimensionality of Q/K embeddings.")
    ap.add_argument("--qk_temp", type=float, default=1.0, help="Temperature for QK attention.")
    ap.add_argument("--alpha", type=float, default=0.3, help="Alpha for propagation (1-alpha)*I + alpha*W.")
    ap.add_argument("--rebuild_every", type=int, default=1, help="Rebuild sparse graph every N epochs.")

    ap.add_argument("--similarity_npz", type=str, default="", help="Precomputed gene-gene similarity CSR .npz")
    ap.add_argument('--remove_non_gene_perts', action='store_true', help='Remove non-gene perturbation labels')
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return args


# ----------------------------
# Training
# ----------------------------
def train(
    adata: ad.AnnData,
    args
):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    # Map dataset/celltype to contiguous ids from obs
    ds_cat = adata.obs['dataset_id'].astype("category").cat
    ct_cat = adata.obs['cell_type' ].astype("category").cat
    num_genes = adata.n_vars
    num_ds    = len(ds_cat.categories)
    num_ct    = len(ct_cat.categories)

    model = PropSpikeModel(
        num_genes=num_genes,
        num_datasets=num_ds,
        num_celltypes=num_ct,
        gene_emb_dim=getattr(args, "gene_emb_dim", 64),
        qk_dim=getattr(args, "qk_dim", 32),
        topk=getattr(args, "topk_keep", 64),
        alpha=getattr(args, "alpha", 0.3),
        T=getattr(args, "T", 2),
        temperature=getattr(args, "qk_temp", 1.0),
        device=device,
    )

    # Freeze vocabularies on the model for consistent mapping at eval time
    model.ds_vocab = [str(x) for x in ds_cat.categories]
    model.ct_vocab = [str(x) for x in ct_cat.categories]
    model.ds2id = {s:i for i,s in enumerate(model.ds_vocab)}
    model.ct2id = {s:i for i,s in enumerate(model.ct_vocab)}

    # === Data tensors (pseudobulk) ===
    X = to_numpy(adata.X).astype(np.float32)            # assume Δμ in PB already
    labels = adata.obs[args.target_label].astype(str).values
    dset_codes = ds_cat.codes.astype(np.int64).to_numpy()
    ct_codes   = ct_cat.codes.astype(np.int64).to_numpy()
    # keep only pert rows (exclude controls)
    ctrl_mask = (labels == args.control_label)
    pert_idx_all = np.where(~ctrl_mask)[0]
    # map pert name -> target gene index (drop non-gene perts for now)
    p2g = build_target_to_gene_index(adata, args.target_label)
    tgt = np.array([p2g.get(labels[i], -1) for i in pert_idx_all], dtype=np.int64)
    keep = tgt >= 0
    pert_rows = pert_idx_all[keep]
    tgt = tgt[keep]
    ds  = dset_codes[pert_rows]
    ct  = ct_codes[pert_rows]
    Y   = X[pert_rows]                                   # target Δμ (N,G)

    # torch tensors
    t_tgt = torch.from_numpy(tgt).to(device=device, dtype=torch.long)
    t_ds  = torch.from_numpy(ds).to(device=device, dtype=torch.long)
    t_ct  = torch.from_numpy(ct).to(device=device, dtype=torch.long)
    t_Y   = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

    # === Optimizer & loss ===
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = torch.nn.SmoothL1Loss(beta=0.1)  # Huber; tweak beta as desired

    # === Train loop ===
    rng = np.random.default_rng(args.seed)
    model.train()
    for epoch in range(1, args.epochs + 1):
        # rebuild sparse graph so Top-K follows Q/K learning
        if (epoch == 1) or (epoch % args.rebuild_every == 0):
            model.eval()
            model.build_sparse_graph(chunk_rows=getattr(args, "qk_chunk", 2048))
            model.train()

        # minibatch over pert rows
        idx = np.arange(len(pert_rows))
        rng.shuffle(idx)
        bs = args.batch_size
        running = 0.0
        for s in range(0, len(idx), bs):
            b = idx[s:s+bs]
            pt = t_tgt[b]
            dsb = t_ds[b]
            ctb = t_ct[b]
            y_true = t_Y[b]                 # (b,G) Δμ
            y_pred = model(pt, dsb, ctb)    # (b,G) Δμ̂
            loss = huber(y_pred, y_true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item() * len(b)
        print(f"[epoch {epoch:03d}] huber={running/len(idx):.5f}")

    model.eval()
    return model, opt


@torch.no_grad()
def predict_all_perturbations(
    adata: ad.AnnData,
    model: nn.Module,
    args,
):
    """
    For every perturbed cell, match a random control, run model, and collect predictions.
    Returns:
      pred_mat: (N_pert, G) predicted expressions (aligned to perturbed rows)
      true_mat: (N_pert, G) true perturbed expressions
      pert_names: list[str] of length N_pert (labels for each row)
      ctrl_mean: (G,) global control pseudobulk (mean of all control cells)
    """
    rng = np.random.default_rng(args.seed)
    pred_mat, true_mat, pert_names, ctrl_mean = None, None, None, None
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[args.target_label].astype(str).values
    G = adata.n_vars

    # pools
    ctrl_idx = np.where(labels == args.control_label)[0]
    pert_idx = np.where(labels != args.control_label)[0]
    if len(ctrl_idx) == 0 or len(pert_idx) == 0:
        raise ValueError("Need both control and perturbed cells for evaluation.")

    # control pseudobulk (global)
    ctrl_mean = X[ctrl_idx].mean(axis=0)

    # Prepare outputs aligned to perturbed rows
    pert_names = labels[pert_idx].tolist()
    true_mat = X[pert_idx]                                  # (N_pert, G)

    # Build (pert_idx, dataset_idx, celltype_idx) for those rows
    # The PB object must carry dataset & celltype columns (categoricals)
    # Map using the model's frozen vocabularies
    ds_labels = adata.obs['dataset_id'].astype(str).values
    ct_labels = adata.obs['cell_type' ].astype(str).values
    # unseen labels map to 0 by default (or choose a dedicated "other" id if you add one)
    dset_codes = np.array([model.ds2id.get(s, 0) for s in ds_labels], dtype=np.int64)
    ct_codes   = np.array([model.ct2id.get(s, 0) for s in ct_labels], dtype=np.int64)
    pidx = build_target_to_gene_index(adata, args.target_label)  # dict: pert_name -> gene index (or -1)
    # Map pert row names -> target gene index (assume pert name == gene symbol when it's a gene)
    tgt_idx = np.array([pidx.get(name, -1) for name in pert_names], dtype=np.int64)
    # Filter out rows where target gene isn't in panel
    keep = tgt_idx >= 0
    if not np.all(keep):
        pert_names = [n for n, k in zip(pert_names, keep) if k]
        true_mat = true_mat[keep]
    tgt_idx = tgt_idx[keep]
    ds_idx  = dset_codes[pert_idx][keep]
    ct_idx  = ct_codes[pert_idx][keep]

    # Forward through PropSpikeModel in reasonable batches
    model.eval()
    preds = []
    bs = max(512, 1)  # large enough for PB rows; adjust if needed
    with torch.no_grad():
        for s in range(0, len(tgt_idx), bs):
            e = min(len(tgt_idx), s + bs)
            pt = torch.from_numpy(tgt_idx[s:e]).to(model.device)
            ds = torch.from_numpy(ds_idx[s:e]).to(model.device)
            ct = torch.from_numpy(ct_idx[s:e]).to(model.device)
            y = model(pt, ds, ct)                            # (b,G) Δμ̂
            preds.append(y.cpu().numpy().astype(np.float32))
    pred_mat = np.concatenate(preds, axis=0)

    return pred_mat, true_mat, pert_names, ctrl_mean


def evaluate_model(
    adata: ad.AnnData,
    model: nn.Module,
    args,
):
    """
    Computes:
      - per-perturbation MAE
      - knockdown efficiency (abs & %) for true vs predicted at the target gene
      - perturbation similarity: mean & min pairwise Pearson corr between predicted mean effect vectors
      - PDS (Perturbation Discrimination Score): mean over perturbations
    Prints a concise report and returns a dict with all metrics.
    """
    pred_mat, true_mat, pert_names, ctrl_mean = predict_all_perturbations(
        adata, model, args
    )
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


# ----------------------------
# CLI
# ----------------------------
def main():
    args = parse_arguments()

    # ---------------------------
    # Read input data
    # ---------------------------
    adata = ad.read_h5ad(args.in_h5ad)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway
    # train/test split
    adata_train, adata_test = train_test_split(args, adata)
    adata_train.obs['dataset_id'] = "target_all"
    adata_train.obs['cell_type'] = "h1_hESC"
    adata_test.obs['dataset_id'] = "target_all"
    adata_test.obs['cell_type'] = "h1_hESC"
    pb_all, pb_len = prep_pb_all(adata_train, adata_train, args)
        
    # ---------------------------
    # Training loop
    # ---------------------------
    model, opt = train(pb_all, args)

    # ---------------------------
    # Evaluate: external test if provided, else held-out split, else train split
    # ---------------------------
    eval_adata = adata_test if adata_test is not None else adata_train
    print("\n=== Evaluation on {} set ===".format("TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)"))
    eval_metrics = evaluate_model(adata=eval_adata, model=model, args=args) 

    if args.eval_on_train and (adata_test is not None):
        print("\n=== Additional evaluation on TRAIN set ===")
        _ = evaluate_model(adata=adata_train, model=model, args=args)

if __name__ == "__main__":
    main()
