#!/usr/bin/env python3
import argparse, math, os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import pairwise_distances

from utils import *
from losses import *


def parse_arguments():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    # Basic and I/O options
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--external_h5ad", required=True, help="Path to the external pseudobulked AnnData object.")
    ap.add_argument("--out_pred_h5ad", type=str, default="",
                    help="If set, write an AnnData with predictions for the evaluation split.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)

    # Train/test split and eval options
    ap.add_argument("--test_pct_perts", type=float, default=0.0,
                    help="Fraction of perturbation labels (excluding control) to hold out for testing. 0.0 = no holdout.")
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional path to a separate test AnnData. If set, overrides --test_pct_perts.")
    ap.add_argument('--remove_non_gene_perts', action='store_true', help='Remove non-gene perturbation labels')
    ap.add_argument('--eval_on_train', action='store_true', help='Evaluate on training set in addition to test set')
    ap.add_argument('--write_test', action='store_true', help='Write true test set')

    # Pseudobulk and batching options
    ap.add_argument("--use_pseudobulk", action="store_true",
                    help="Collapse to one mean row per perturbation (incl. control).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--single_pert_batches", action="store_true",
                    help="If set, each batch contains cells from a single perturbation label.")

    # Model architecture options
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--T", type=int, default=2, help="Number of message-passing steps.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.0, help="Step-0 anchor (e.g., 0.0 for CRISPRi).")
    ap.add_argument("--node_dim", type=int, default=128, help="Dimensionality of gene node embeddings.")

    # Loss function options
    ap.add_argument("--weight_target", type=float, default=0.1)
    ap.add_argument("--weight_local", type=float, default=0.0)
    ap.add_argument("--weight_mse", type=float, default=0.0, help="Weight for per-cell MSE loss.")
    ap.add_argument("--weight_proto", type=float, default=0.2, help="Weight for prototype loss.")
    ap.add_argument("--weight_dist", type=float, default=1.0, help="Weight for distribution loss.")
    ap.add_argument("--dist_loss", choices=["none","mmd","swd","energy"], default="mmd",
                    help="Distribution loss between predicted and true deltas per perturbation.")
    ap.add_argument("--swd_projections", type=int, default=128, help="Num random projections for SWD.")

    # Pretraining options
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
    args = ap.parse_args()
    return args

def intersect_datasets(adata_source, adata_target, target_label, control_label):
    """
    Subsets two AnnData objects to their common genes and perturbations.

    Args:
        adata_source: The source (external) AnnData object.
        adata_target: The target AnnData object.
        target_label: The obs column containing perturbation labels.
        control_label: The label for control samples.

    Returns:
        A tuple of (subsetted source AnnData, subsetted target AnnData).
    """
    print("Finding intersection of genes and perturbations...")
    # First, get a list of valid genes from the source (not all NaN), then intersect.
    common_genes = np.intersect1d(
        adata_source.var_names[~np.isnan(to_numpy(adata_source.X)).all(axis=0)],
        adata_target.var_names
    )

    source_perts = set(adata_source.obs[target_label].unique())
    target_perts = set(adata_target.obs[target_label].unique())
    common_perts = sorted(list(source_perts.intersection(target_perts)))

    # Ensure the control label is always kept, even if it's not in the intersection
    if control_label not in common_perts:
        if control_label in source_perts and control_label in target_perts:
            common_perts.append(control_label)
    
    print(f"  Found {len(common_genes)} common genes.")
    print(f"  Found {len(common_perts) - 1} common perturbations (plus control).")

    adata_source_sub = adata_source[adata_source.obs[target_label].isin(common_perts), common_genes].copy()
    adata_target_sub = adata_target[adata_target.obs[target_label].isin(common_perts), common_genes].copy()

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

def predict_direct_transfer(source_deltas, target_adata, target_label, control_label):
    """
    Generates predictions by adding source deltas to the target control mean.

    Args:
        source_deltas: A dictionary of {perturbation: delta_vector} from the source.
        target_adata: The target AnnData object.
        target_label: The obs column containing perturbation labels.
        control_label: The label for control samples.

    Returns:
        A dictionary mapping perturbation labels to their predicted expression vectors.
    """
    control_mask = target_adata.obs[target_label] == control_label
    target_control_mean = target_adata[control_mask].X.mean(axis=0)

    predictions = {}
    for pert_label, delta_vec in source_deltas.items():
        predictions[pert_label] = target_control_mean + delta_vec
    
    return predictions


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


# ----------------------------
# CLI
# ----------------------------
def main():
    args = parse_arguments()
    # Your script already assumes pseudobulking for this baseline
    args.use_pseudobulk = True

    # ---------------------------
    # Read input data
    # ---------------------------
    adata_target = ad.read_h5ad(args.in_h5ad)
    adata_source = ad.read_h5ad(args.external_h5ad)

    # --- NEW BASELINE LOGIC START ---

    # 1) Subset both datasets to their intersection of genes and perturbations
    adata_source, adata_target = intersect_datasets(
        adata_source, adata_target, args.target_label, args.control_label
    )

    # 2) Process and pseudobulk the target dataset
    adata_target.obs['dataset_id'] = "target_all"
    adata_target.obs['cell_type'] = "UNK"
    sc.pp.normalize_total(adata_target, inplace=True)
    sc.pp.log1p(adata_target)
    if args.use_pseudobulk:
        adata_target = collapse_to_pseudobulk(adata_target, args.target_label)
    
    # Process the source dataset (assumed to be pseudobulked already)
    sc.pp.normalize_total(adata_source, inplace=True)
    sc.pp.log1p(adata_source)

    # 3) Compute deltas from the source dataset
    source_deltas, _ = compute_deltas(adata_source, args.target_label, args.control_label)

    # 4) Generate predictions for the target dataset via direct transfer
    predictions = predict_direct_transfer(source_deltas, adata_target, args.target_label, args.control_label)

    # --- NEW BASELINE LOGIC END ---

    # train/test split on the (now intersected and pseudobulked) target data
    adata_train, adata_test = train_test_split(args, adata_target)

    # ---------------------------
    # Evaluate: external test if provided, else held-out split, else train split
    # ---------------------------
    eval_adata = adata_test if adata_test is not None else adata_train
    print("\n=== Evaluation on {} set ===".format(
        "TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)")
    )
    
    # --- NEW BASELINE LOGIC START ---

    # 5) Assemble the `pred_bundle` for the evaluation function
    # Get the ground truth from the evaluation set
    true_deltas, target_ctrl_mean = compute_deltas(eval_adata, args.target_label, args.control_label)
    
    # Get the list of perturbations in the evaluation set, in order
    pert_mask = eval_adata.obs[args.target_label] != args.control_label
    eval_perts = eval_adata[pert_mask].obs[args.target_label].tolist()
    
    # Look up the prediction for each evaluation perturbation and stack them
    pred_mat = np.array([predictions[p] for p in eval_perts])
    
    # Get the true expression matrix for these perturbations
    true_mat = eval_adata[pert_mask].X
    
    # Create the bundle
    pred_bundle = (pred_mat, true_mat, eval_perts, target_ctrl_mean)

    # --- NEW BASELINE LOGIC END ---

    _ = evaluate_model(adata=eval_adata, args=args, pred_bundle=pred_bundle)

    if args.eval_on_train and (adata_test is not None):
        print("\n=== Evaluation on TRAIN set (fit on TRAIN) ===")
        # Assemble pred_bundle for the training set
        _, train_target_ctrl_mean = compute_deltas(adata_train, args.target_label, args.control_label)
        train_pert_mask = adata_train.obs[args.target_label] != args.control_label
        train_eval_perts = adata_train[train_pert_mask].obs[args.target_label].tolist()
        train_pred_mat = np.array([predictions[p] for p in train_eval_perts])
        train_true_mat = adata_train[train_pert_mask].X
        train_pred_bundle = (train_pred_mat, train_true_mat, train_eval_perts, train_target_ctrl_mean)
        _ = evaluate_model(adata=adata_train, args=args, pred_bundle=train_pred_bundle)

    if args.out_pred_h5ad:
        if hasattr(eval_adata.X, "toarray"):
            eval_adata.X = eval_adata.X.toarray()
        write_pred_true_h5ads(
            eval_adata=eval_adata,
            pred_bundle=pred_bundle,
            out_pred_h5ad=args.out_pred_h5ad,
            target_label=args.target_label,
            control_label=args.control_label,
        )

if __name__ == "__main__":
    main()
