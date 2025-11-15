#!/usr/bin/env python

import argparse
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse


def to_numpy(X):
    """Convert dense/sparse matrix to a dense NumPy array."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Oracle cell-level simulator: "
            "use true first/second moments to simulate perturbed cells "
            "by subtracting delta vectors from sampled control cells."
        )
    )
    parser.add_argument("--in_h5ad", required=True, help="Input AnnData file (single-cell, normalized/log1p).")
    parser.add_argument("--out_h5ad", required=True, help="Output AnnData file with synthetic perturbed cells.")
    parser.add_argument(
        "--target_label",
        required=True,
        help="obs column containing perturbation labels (e.g. 'target_gene').",
    )
    parser.add_argument(
        "--control_label",
        required=True,
        help="Label in target_label corresponding to control / non-targeting cells.",
    )
    parser.add_argument(
        "--no_var_sampling",
        action="store_true",
        help="If set, disable variance-based sampling and use only mean delta vectors.",
    )
    parser.add_argument(
        "--var_scale",
        type=float,
        default=1.0,
        help="Scaling factor on the per-gene std dev when sampling deltas (default: 1.0).",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=None,
        help="If set, clip all simulated expression values to be >= this value (e.g. 0.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible control-cell sampling and delta noise.",
    )
    parser.add_argument(
        "--keep_delta_pct",
        type=float,
        default=None,
        help=(
            "If set in (0, 1], for each perturbation keep only this fraction of genes "
            "with the largest |delta_mean|; all other deltas are set to 0."
        ),
    )
    parser.add_argument(
        "--keep_foldchange_pct",
        type=float,
        default=None,
        help=(
            "If set in (0, 1], for each perturbation keep only this fraction of genes "
            "with the largest |log2 fold change| (pert vs control); all other deltas are set to 0. "
            "If both keep_delta_pct and keep_foldchange_pct are set, the intersection is kept."
        ),
    )

    args = parser.parse_args()

    print(f"Reading {args.in_h5ad} ...")
    adata = ad.read_h5ad(args.in_h5ad)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    if args.target_label not in adata.obs.columns:
        raise ValueError(f"Column {args.target_label!r} not found in adata.obs")

    labels = adata.obs[args.target_label].astype(str).values
    ctrl_label = str(args.control_label)

    ctrl_mask = labels == ctrl_label
    if not ctrl_mask.any():
        raise ValueError(f"No control cells found with label {ctrl_label!r} in column {args.target_label!r}")

    X = to_numpy(adata.X).astype(np.float32)
    n_cells, G = X.shape
    print(f"Data shape: {n_cells} cells x {G} genes")

    X_ctrl = X[ctrl_mask]
    n_ctrl = X_ctrl.shape[0]
    print(f"Found {n_ctrl} control cells with label {ctrl_label!r}")

    # Small epsilon to stabilize log fold-change calculations
    eps_fc = 1e-8

    # --- 1) Compute control moments (per gene) ---
    # First moment: mean
    ctrl_mean = X_ctrl.mean(axis=0)
    # Second central moment: variance
    # (ddof=1 -> sample variance; change to ddof=0 if you want MLE)
    ctrl_var = X_ctrl.var(axis=0, ddof=1)

    # --- 2) Compute per-pert moments and delta moments ---
    all_perts = sorted(set(labels.tolist()))
    perts = [p for p in all_perts if p != ctrl_label]

    print(f"Found {len(perts)} non-control perturbations in {args.target_label!r}")

    delta_mean_list = []
    delta_var_list = []
    pert_order = []

    # Pre-allocate dicts if you prefer lookup by name
    mean_delta_by_pert = {}
    var_delta_by_pert = {}

    for p in perts:
        mask_p = labels == p
        n_p = int(mask_p.sum())
        if n_p == 0:
            continue

        X_p = X[mask_p]

        # Pert moments
        mean_p = X_p.mean(axis=0)
        var_p = X_p.var(axis=0, ddof=1)

        # Delta = control - pert
        delta_mean = ctrl_mean - mean_p
        # Var(C - P) = Var(C) + Var(P) assuming independence
        delta_var = ctrl_var + var_p

        # Optional sparsification based on delta magnitude and log2 fold-change
        # Delta = control - pert
        delta_mean = ctrl_mean - mean_p
        # Var(C - P) = Var(C) + Var(P) assuming independence
        delta_var = ctrl_var + var_p

        # Optional sparsification based on delta magnitude and log2 fold-change
        if args.keep_delta_pct is not None or args.keep_foldchange_pct is not None:
            keep_mask = np.zeros(G, dtype=bool)

            # Top genes by |delta_mean|
            if args.keep_delta_pct is not None:
                if not (0.0 < args.keep_delta_pct <= 1.0):
                    raise ValueError("--keep_delta_pct must be in (0, 1].")
                k_delta = max(1, int(np.floor(args.keep_delta_pct * G)))
                idx_delta = np.argpartition(-np.abs(delta_mean), k_delta - 1)[:k_delta]
                mask_delta = np.zeros(G, dtype=bool)
                mask_delta[idx_delta] = True
                # UNION: keep any gene in the top-|delta| set
                keep_mask |= mask_delta

            # Top genes by |log2 fold-change|
            if args.keep_foldchange_pct is not None:
                if not (0.0 < args.keep_foldchange_pct <= 1.0):
                    raise ValueError("--keep_foldchange_pct must be in (0, 1].")
                log2fc = np.log2((mean_p + eps_fc) / (ctrl_mean + eps_fc))
                k_fc = max(1, int(np.floor(args.keep_foldchange_pct * G)))
                idx_fc = np.argpartition(-np.abs(log2fc), k_fc - 1)[:k_fc]
                mask_fc = np.zeros(G, dtype=bool)
                mask_fc[idx_fc] = True
                # UNION: keep any gene in the top-|log2FC| set
                keep_mask |= mask_fc

            # Zero out deltas (and their variance) for genes not in the keep set
            delta_mean = delta_mean.copy()
            delta_var = delta_var.copy()
            delta_mean[~keep_mask] = 0.0
            delta_var[~keep_mask] = 0.0

        mean_delta_by_pert[p] = delta_mean.astype(np.float32)
        var_delta_by_pert[p] = delta_var.astype(np.float32)

        delta_mean_list.append(delta_mean.astype(np.float32))
        delta_var_list.append(delta_var.astype(np.float32))
        pert_order.append(p)

    if len(pert_order) == 0:
        raise ValueError("No non-control perturbations with cells found; nothing to simulate.")

    delta_mean_mat = np.stack(delta_mean_list, axis=0)  # (P, G)
    delta_var_mat = np.stack(delta_var_list, axis=0)    # (P, G)

    print("Computed per-pert first and second moments (deltas).")

    # --- 3) Build synthetic dataset ---
    rng = np.random.default_rng(args.seed)

    X_new = np.zeros_like(X, dtype=np.float32)

    # 3a) Controls: copy as-is
    X_new[ctrl_mask, :] = X_ctrl
    print("Copied control cells into synthetic dataset.")

    # 3b) Perturbed cells: sample control cells and subtract delta (with or without variance sampling)
    for p in perts:
        mask_p = labels == p
        idx_p = np.where(mask_p)[0]
        n_p = idx_p.shape[0]
        if n_p == 0:
            continue

        delta_mean = mean_delta_by_pert[p]  # (G,)
        delta_var = var_delta_by_pert[p]    # (G,)

        # Sample base control cells (with replacement if needed)
        replace = n_p > n_ctrl
        ctrl_idx = rng.choice(n_ctrl, size=n_p, replace=replace)
        X_base = X_ctrl[ctrl_idx, :]  # (n_p, G)

        if args.no_var_sampling:
            # Deterministic: use only mean delta
            delta_samples = np.broadcast_to(delta_mean[None, :], (n_p, G))
        else:
            # Stochastic: sample per-cell deltas from Normal(mean=Δ, var=Var(Δ))
            std_vec = np.sqrt(np.clip(delta_var, 0.0, None)) * float(args.var_scale)
            eps = rng.normal(loc=0.0, scale=1.0, size=(n_p, G)).astype(np.float32)
            delta_samples = delta_mean[None, :] + eps * std_vec[None, :]

        X_p_new = X_base - delta_samples  # (n_p, G)

        if args.clip_min is not None:
            np.maximum(X_p_new, float(args.clip_min), out=X_p_new)

        X_new[idx_p, :] = X_p_new

    print("Finished simulating perturbed cells from control pool + delta moments.")

    # --- 4) Construct output AnnData and stash moments in .uns ---
    adata_out = ad.AnnData(
        X=X_new,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy(),
    )

    # Store oracle moment info for later inspection
    adata_out.uns["oracle_ctrl_mean"] = ctrl_mean.astype(np.float32)
    adata_out.uns["oracle_ctrl_var"] = ctrl_var.astype(np.float32)
    adata_out.uns["oracle_pert_order"] = np.array(pert_order, dtype=object)
    adata_out.uns["oracle_delta_mean"] = delta_mean_mat
    adata_out.uns["oracle_delta_var"] = delta_var_mat
    adata_out.uns["oracle_params"] = {
        "target_label": args.target_label,
        "control_label": ctrl_label,
        "no_var_sampling": bool(args.no_var_sampling),
        "var_scale": float(args.var_scale),
        "clip_min": args.clip_min,
        "seed": int(args.seed),
    }

    print(f"Writing synthetic AnnData to {args.out_h5ad} ...")
    adata_out.write_h5ad(args.out_h5ad)
    print("Done.")


if __name__ == "__main__":
    main()
