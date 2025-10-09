#!/usr/bin/env python3
import argparse
import os
import anndata as ad
import numpy as np
import pandas as pd

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a synthetic external pseudobulk dataset via affine transformation."
    )
    parser.add_argument(
        "--input_h5ad",
        required=True,
        type=str,
        help="Path to the input pseudobulked AnnData object (.h5ad)."
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
        help="Label for the control sample in the target_label column."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="synth_",
        help="Prefix to add to 'dataset_id' and 'cell_type' columns."
    )
    parser.add_argument(
        "--output_h5ad",
        required=True,
        type=str,
        help="Path to save the output synthetic AnnData object (.h5ad)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

def main():
    """Main script execution."""
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.output_h5ad) or ".", exist_ok=True)

    # 0. Set random seed for reproducibility
    print(f"🌱 Setting random seed to {args.seed}")
    np.random.seed(args.seed)

    # 1. Load the pseudobulked data
    print(f"🔬 Reading pseudobulk data from: {args.input_h5ad}")
    pb_adata = ad.read_h5ad(args.input_h5ad)
    print(f"Original data shape: {pb_adata.n_obs} perturbations x {pb_adata.n_vars} genes")

    # 2. Separate control and perturbed data
    control_mask = pb_adata.obs[args.target_label] == args.control_label
    if not control_mask.any():
        raise ValueError(f"Control label '{args.control_label}' not found in column '{args.target_label}'.")
    if control_mask.sum() > 1:
        print(f"⚠️ Warning: Found {control_mask.sum()} control rows. Averaging them to create a single control vector.")
        control_row_adata = pb_adata[control_mask]
        control_mean_vector = control_row_adata.X.mean(axis=0)
    else:
        control_row_adata = pb_adata[control_mask]
        control_mean_vector = control_row_adata.X.flatten()

    pert_adata = pb_adata[~control_mask].copy()
    print(f"Found 1 control profile and {pert_adata.n_obs} perturbation profiles.")

    # 3. Calculate the delta vectors for each perturbation
    delta_vectors = pert_adata.X - control_mean_vector

    # 4. Generate the per-gene transformation parameters
    n_genes = pb_adata.n_vars
    multipliers = np.random.uniform(0.8, 1.2, size=n_genes)
    shift_factors = np.random.uniform(-0.2, 0.2, size=n_genes)
    print(f"Generated {n_genes} random multipliers and shift factors.")

    # 5. Apply the affine transformation
    print("Applying affine transformation to delta vectors...")
    # Step 1: Multiply the delta vectors
    transformed_deltas = delta_vectors * multipliers

    # Step 2: Add back to control to get "raw" transformed expression
    new_pert_raw = control_mean_vector + transformed_deltas

    # Step 3: Calculate the shift based on the new raw expression
    shift_amounts = new_pert_raw * shift_factors

    # Step 4: Apply the shift
    new_pert_shifted = new_pert_raw + shift_amounts

    # 6. Enforce non-negativity
    final_pert_X = np.maximum(0, new_pert_shifted)
    print("Enforced non-negativity constraint.")

    # 7. Update the AnnData object with the new expression values
    pert_adata.X = final_pert_X

    # 8. Re-combine with the original, unmodified control row(s)
    adata_final = ad.concat([control_row_adata, pert_adata], join="inner")

    # 9. Disguise the dataset identity
    print(f"Disguising metadata with prefix '{args.prefix}'...")
    for col in ["dataset_id", "cell_type"]:
        if col in adata_final.obs.columns:
            # Ensure column is of string type to allow concatenation
            adata_final.obs[col] = adata_final.obs[col].astype(str)
            adata_final.obs[col] = args.prefix + adata_final.obs[col]
            print(f"  - Modified '{col}' column.")
        else:
            print(f"  - Column '{col}' not found, skipping.")


    # 10. Write the final synthetic object
    adata_final.write_h5ad(args.output_h5ad)
    print(f"\n✅ Wrote synthetic external data to: {args.output_h5ad}")
    print("✨ Done!")


if __name__ == "__main__":
    main()
