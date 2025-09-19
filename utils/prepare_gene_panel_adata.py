#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# Provided helper functions
# -----------------------
def get_mean_expr(adata, target_label='target_id'):
    """
    Compute the mean gene expression vector for each perturbation.

    Parameters:
        adata (AnnData): AnnData object.
        target_label (str): obs column specifying perturbation.

    Returns:
        pd.DataFrame: DataFrame with perturbations as rows and genes as columns.
    """
    pert_names = adata.obs[target_label].unique()
    mean_expr = []
    for pert in pert_names:
        cells = adata.obs[target_label] == pert
        expr = adata[cells].X
        # Handle sparse matrices
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        mean_expr.append(expr.mean(axis=0))
    mean_expr_df = pd.DataFrame(mean_expr, index=pert_names, columns=adata.var_names)
    return mean_expr_df


def get_expr_diff(mean_expr_df, control_label='control'):
    """
    Compute the difference in mean gene expression between each perturbation and control.

    Parameters:
        mean_expr_df (pd.DataFrame): DataFrame with perturbations as rows and genes as columns.
        control_label (str): Row label corresponding to control cells.

    Returns:
        pd.DataFrame: DataFrame of differences (perturbation - control), same shape as input but without control row.
    """
    control_vec = mean_expr_df.loc[control_label]
    expr_diff_df = mean_expr_df.subtract(control_vec, axis=1)
    expr_diff_df = expr_diff_df.drop(index=control_label)
    return expr_diff_df


# -----------------------
# Utility
# -----------------------
def _downsample_idx(idx_array, n_keep, rng):
    if n_keep is None or len(idx_array) <= n_keep:
        return idx_array
    sel = rng.choice(idx_array, size=n_keep, replace=False)
    return np.sort(sel)


def build_gene_panel(
    adata: ad.AnnData,
    target_label: str,
    control_label: str,
    top_n_perts: int = 20,
    top_k_genes_per_pert: int = 20,
    max_ctrl_cells: int = 5000,
    max_cells_per_pert: int = 1000,
    random_seed: int = 0,
    test_h5ad: str = "",
):
    """
    Returns a new AnnData with a small gene panel and capped per-pert/control cell counts.
    """

    rng = np.random.default_rng(random_seed)

    # Basic checks
    if target_label not in adata.obs.columns:
        raise ValueError(f"`target_label` '{target_label}' not found in adata.obs.")

    # Make sure control label exists in obs values
    if control_label not in adata.obs[target_label].values:
        raise ValueError(f"`control_label` '{control_label}' not found in adata.obs['{target_label}'].")

    # 1) Compute mean expression per perturbation (includes control row)
    mean_expr_df = get_mean_expr(adata, target_label=target_label)

    # 2) Compute (pert - control) mean-difference matrix and rank genes by |diff|
    expr_diff_df = get_expr_diff(mean_expr_df, control_label=control_label)  # rows exclude control

    # 3) Choose top-N perturbations by cell count (excluding control)
    vc = adata.obs[target_label].value_counts()
    non_control_counts = vc.drop(labels=[control_label], errors="ignore")
    chosen_perts = list(non_control_counts.head(top_n_perts).index)

    if len(chosen_perts) == 0:
        raise ValueError("No perturbations found after excluding control; cannot build panel.")

    # 4) For each chosen perturbation, select top-K most affected genes by |diff|
    chosen_genes = set()
    missing_perts = []
    for p in chosen_perts:
        if p not in expr_diff_df.index:
            missing_perts.append(p)
            continue
        top_genes = expr_diff_df.loc[p].abs().nlargest(top_k_genes_per_pert).index
        chosen_genes.update(top_genes)

    # If perturbation labels are gene symbols present in var_names, force-include them.
    # (Control label is excluded by chosen_perts selection above.)
    target_genes_to_add = [p for p in chosen_perts if p in adata.var_names]
    if target_genes_to_add:
        chosen_genes.update(target_genes_to_add)

    # --- NEW: force-include all genes targeted in the external TEST set ---
    if test_h5ad:
        print(f"[panel] Loading test AnnData for required genes: {test_h5ad}")
        adata_test = ad.read_h5ad(test_h5ad, backed="r")
        # collect perturbation labels in test (excluding control)
        test_labels = adata_test.obs[target_label].astype(str).values
        test_perts = sorted({p for p in test_labels if p != control_label})
        # interpret perturbation labels as gene symbols; include those that are present in training var_names
        train_gene_universe = set(adata.var_names.astype(str).tolist())
        required_from_test = {g for g in test_perts if g in train_gene_universe}
        missing_from_train = [g for g in test_perts if g not in train_gene_universe]
        if missing_from_train:
            print(f"[panel][warn] {len(missing_from_train)} test target(s) not found in training var_names "
                  f"(first few): {missing_from_train[:10]}")
        # union into chosen_genes
        before = len(chosen_genes)
        chosen_genes |= required_from_test
        added = len(chosen_genes) - before
        print(f"[panel] Force-included {added} test-target gene(s) into training panel "
              f"(now {len(chosen_genes)} genes).")

    if missing_perts:
        # Shouldn't happen often, but just in case of label mismatches
        print(f"[warn] Skipped {len(missing_perts)} perturbations missing in expr_diff_df: {missing_perts}")

    chosen_genes = list(chosen_genes)

    # Ensure genes exist in var_names (they should)
    chosen_genes = [g for g in chosen_genes if g in adata.var_names]
    if len(chosen_genes) == 0:
        raise ValueError("Gene selection yielded an empty set. Check labels/inputs.")

    # 5) Subset cells: keep controls + chosen perturbations
    keep_cell_mask = adata.obs[target_label].isin([control_label] + chosen_perts)
    ad_sub = adata[keep_cell_mask, chosen_genes].copy()  # copy to avoid view pitfalls

    # 6) Downsample cells: up to max_ctrl_cells controls and up to max_cells_per_pert per perturbation
    #    (Always keep all controls if max_ctrl_cells=None)
    obs = ad_sub.obs
    ctrl_idx = np.where(obs[target_label].values == control_label)[0]
    keep_ctrl = _downsample_idx(ctrl_idx, max_ctrl_cells, rng)

    keep_pert = []
    for p in chosen_perts:
        p_idx = np.where(obs[target_label].values == p)[0]
        if len(p_idx) == 0:
            continue
        keep_p = _downsample_idx(p_idx, max_cells_per_pert, rng)
        keep_pert.append(keep_p)

    keep_rows = np.sort(np.concatenate([keep_ctrl] + keep_pert)) if len(keep_pert) > 0 else keep_ctrl
    panel = ad_sub[keep_rows, :].copy()

    # Final small sanity printouts
    n_ctrl_final = (panel.obs[target_label] == control_label).sum()
    perts_final = panel.obs[target_label].value_counts().drop(labels=[control_label], errors="ignore")

    print("=== Gene Panel Summary ===")
    print(f"Genes kept: {panel.n_vars}")
    print(f"Total cells kept: {panel.n_obs}")
    print(f"Control cells kept: {n_ctrl_final}")
    print("Perturbations (top):")
    print(perts_final.head(50))

    return panel


def main():
    ap = argparse.ArgumentParser(description="Build a small gene-panel AnnData from a larger dataset.")
    ap.add_argument("--in_h5ad", required=True, help="Input AnnData .h5ad")
    ap.add_argument("--out_h5ad", required=True, help="Output AnnData .h5ad (panel)")
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional test AnnData. If set, all genes targeted in this file will be force-included in the training panel.")
    ap.add_argument("--target_label", default="target_gene", help="obs column for perturbation label")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells")
    ap.add_argument("--top_n_perts", type=int, default=20, help="Number of most frequent perturbations to keep")
    ap.add_argument("--top_k_genes_per_pert", type=int, default=20, help="Top-K genes per perturbation by |mean diff|")
    ap.add_argument("--max_ctrl_cells", type=int, default=5000, help="Max number of control cells to keep")
    ap.add_argument("--max_cells_per_pert", type=int, default=1000, help="Max cells per perturbation to keep")
    ap.add_argument("--random_seed", type=int, default=0, help="Random seed for downsampling")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.in_h5ad)

    # (Optional) make sure X is CSR for decent slicing behavior
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()

    panel = build_gene_panel(
        adata=adata,
        target_label=args.target_label,
        control_label=args.control_label,
        top_n_perts=args.top_n_perts,
        top_k_genes_per_pert=args.top_k_genes_per_pert,
        max_ctrl_cells=args.max_ctrl_cells,
        max_cells_per_pert=args.max_cells_per_pert,
        random_seed=args.random_seed,
        test_h5ad=args.test_h5ad,
    )

    panel.write_h5ad(args.out_h5ad, compression="lzf")
    print(f"[done] Wrote panel to {args.out_h5ad}")


if __name__ == "__main__":
    main()
