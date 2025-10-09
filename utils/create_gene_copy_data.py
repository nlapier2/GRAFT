#!/usr/bin/env python3
import argparse
import os
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a train/test dataset for a gene copy experiment."
    )
    parser.add_argument(
        "--input_h5ad",
        required=True,
        type=str,
        help="Path to the input AnnData object (.h5ad)."
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default="target_gene",
        help="Column in `adata.obs` containing the perturbation labels."
    )
    parser.add_argument(
        "--control_label",
        type=str,
        default="non-targeting",
        help="Label for control cells in the target_label column."
    )
    parser.add_argument(
        "--copy_prefix",
        type=str,
        default="copy_",
        help="Prefix to add to copied gene names."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Directory to save the output files."
    )
    return parser.parse_args()

def main():
    """Main script execution."""
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load the original data
    print(f"🔬 Reading data from: {args.input_h5ad}")
    adata_orig = ad.read_h5ad(args.input_h5ad)
    print(f"Original data shape: {adata_orig.n_obs} cells x {adata_orig.n_vars} genes")

    # 2. Identify perturbed genes that exist in the gene panel
    all_perts = adata_orig.obs[args.target_label].unique()
    genes_to_copy = sorted([
        p for p in all_perts
        if p != args.control_label and p in adata_orig.var_names
    ])
    if not genes_to_copy:
        raise ValueError("No perturbed genes found in `adata.var_names`. Check your `target_label`.")
    print(f"🧬 Found {len(genes_to_copy)} perturbed genes to copy.")

    # Create the name mapping
    copy_map = {gene: f"{args.copy_prefix}{gene}" for gene in genes_to_copy}

    # --- Part A: Create the training dataset with copied genes ---
    print("\n🛠️  Creating training dataset...")

    # Get the indices and data for the genes to be copied
    original_gene_indices = [adata_orig.var_names.get_loc(g) for g in genes_to_copy]
    X_to_copy = adata_orig.X[:, original_gene_indices]

    # Create new variable (gene) metadata for the copies
    var_orig = adata_orig.var
    var_copies = var_orig.iloc[original_gene_indices].copy()
    var_copies.index = [copy_map[g] for g in genes_to_copy]

    # Combine original and copied metadata
    var_train = pd.concat([var_orig, var_copies])

    # Combine the expression matrices
    if sparse.issparse(adata_orig.X):
        X_train = sparse.hstack([adata_orig.X, X_to_copy], format="csr")
    else:
        X_train = np.hstack([adata_orig.X, X_to_copy])

    # Create the new AnnData object for training
    adata_train = ad.AnnData(X=X_train, obs=adata_orig.obs.copy(), var=var_train)
    print(f"New training data shape: {adata_train.n_obs} cells x {adata_train.n_vars} genes")

    # Write training file
    train_path = os.path.join(args.out_dir, "train_gene_copies.h5ad")
    adata_train.write_h5ad(train_path)
    print(f"✅ Wrote training data to: {train_path}")


    # --- Part B: Create the test dataset with relabeled perturbations ---
    print("\n🧪 Creating test dataset...")

    # Create a copy for the test set; we only modify the `obs` table
    adata_test = adata_train.copy()

    # Replace original perturbation names with their 'copy_' counterparts
    # Unperturbed genes and control labels will not be in the map, so they remain unchanged.
    adata_test.obs[args.target_label] = adata_test.obs[args.target_label].map(
        lambda p: copy_map.get(p, p)
    )
    print("Relabeled perturbations in test data (e.g., 'GENE_X' -> 'copy_GENE_X').")

    # Write test file
    test_path = os.path.join(args.out_dir, "test_gene_copies.h5ad")
    adata_test.write_h5ad(test_path)
    print(f"✅ Wrote test data to: {test_path}")
    print("\n✨ Done!")


if __name__ == "__main__":
    main()