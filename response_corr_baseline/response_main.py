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


# --- Add this new function to boilerplate_main.py ---

def run_correlation_baseline(
    adata: ad.AnnData,
    args
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Implements a simple baseline where the effect on a response gene is
    predicted as the target gene's knockdown delta multiplied by the
    correlation of the response gene's and target gene's deltas across all perturbations.
    """
    print("=== Running simple correlation baseline ===")

    # Ensure data is a dense numpy array for calculations
    X = to_numpy(adata.X)

    # 1. Separate control and perturbed data
    ctrl_mask = adata.obs[args.target_label] == args.control_label
    pert_mask = ~ctrl_mask

    # Since the data is pseudobulked, there is one control row. Get its expression vector.
    ctrl_mean = X[ctrl_mask, :].mean(axis=0)

    # Get the data for the perturbed samples
    pert_X = X[pert_mask, :]
    pert_names = adata.obs[args.target_label][pert_mask].tolist()
    num_perts = len(pert_names)

    # 2. Compute the true deltas (perturbed - control) and the response correlation matrix
    print("Computing gene-gene response correlation matrix...")
    delta_matrix_true = pert_X - ctrl_mean

    # np.corrcoef with rowvar=False computes correlation between columns (genes)
    response_corr = np.corrcoef(delta_matrix_true, rowvar=False)
    # Handle cases where a gene has zero variance across perturbations, which results in NaN
    response_corr = np.nan_to_num(response_corr)

    # 3. Predict effects for each perturbation based on its target gene
    t2gi = build_target_to_gene_index(adata, args.target_label)
    pred_mat = np.zeros_like(pert_X)

    print("Predicting effects for each perturbation...")
    for i, p_name in enumerate(pert_names):
        target_gene_idx = t2gi.get(p_name, -1)
        predicted_delta_vector = np.zeros(adata.n_vars)

        # Only predict an effect if the perturbation targets a known gene in the panel
        if target_gene_idx >= 0:
            # Get the true observed delta for the target gene itself
            target_knockdown_delta = delta_matrix_true[i, target_gene_idx]

            # Get the correlation of this target gene with all other genes
            corr_vector = response_corr[:, target_gene_idx]

            # Predict deltas for all genes: delta_R = delta_P * corr(G_p, R)
            predicted_delta_vector = target_knockdown_delta * corr_vector

        # Final prediction is the control mean + the predicted delta vector
        pred_mat[i, :] = ctrl_mean + predicted_delta_vector

    # 4. Assemble the prediction bundle for the evaluation function
    # The bundle contains: (predicted_expressions, true_expressions, perturbation_names, control_mean)
    pred_bundle = (pred_mat, pert_X, pert_names, ctrl_mean)

    return pred_bundle


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
    if adata_test is not None:
        adata_test.obs['dataset_id'] = "target_all"
        adata_test.obs['cell_type'] = "UNK"
    eval_adata = adata_test if adata_test is not None else adata_train

    print(f"\nRunning baseline and evaluating on pseudobulked data with {eval_adata.n_obs} rows.")
    pred_bundle = run_correlation_baseline(eval_adata, args)

    # ---------------------------
    # Evaluate: external test if provided, else held-out split, else train split
    # ---------------------------
    print("\n=== Evaluation on {} set ===".format(
        "TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)")
    )
    _ = evaluate_model(adata=eval_adata, args=args, pred_bundle=pred_bundle)

    if args.eval_on_train and (adata_test is not None):
        print("\n=== Evaluation on TRAIN set (fit on TRAIN) ===")
        train_labels = adata_train.obs[args.target_label].astype(str).values
        train_perts  = sorted({lab for lab in train_labels if lab != args.control_label})
        _ = evaluate_model(adata=adata_train, args=args, pred_bundle=None)


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
