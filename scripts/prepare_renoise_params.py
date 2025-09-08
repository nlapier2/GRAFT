"""
prepare_renoise_params.py (Refactored)

Phase 1 of the inference pipeline. This script now uses the pre-built scVI
input AnnData object, which simplifies the logic significantly.

Workflow:
1.  Loads the scvi_input_{cell_type}_controls.h5ad file. This file contains
    raw counts for all control cells, pre-filtered and aligned to the
    canonical gene list.
2.  Calculates per-gene Negative Binomial dispersion (theta) from these counts.
3.  Collates the empirical library size distributions for each dataset.
4.  Saves these parameters to a single .npz file for fast loading during inference.
"""
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graft.utils.re_noise import estimate_alpha_from_counts


def main():
    parser = argparse.ArgumentParser(description="Prepare re-noising parameters from a pre-built control AnnData.")
    parser.add_argument(
        "--scvi-input-h5ad",
        required=True,
        help="Path to the scvi_input_{cell_type}_controls.h5ad file."
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Path to save the output .npz file containing re-noising parameters."
    )
    parser.add_argument(
        "--l-ref",
        type=float,
        default=10000.0,
        help="Reference library size for normalization, matching the one used for training."
    )
    args = parser.parse_args()

    # 1. Load the pre-built AnnData of control cells
    print(f"Loading control cell data from {args.scvi_input_h5ad}...")
    try:
        adata = ad.read_h5ad(args.scvi_input_h5ad)
    except Exception as e:
        print(f"[ERROR] Failed to load AnnData file: {e}")
        sys.exit(1)

    if not sparse.isspmatrix_csr(adata.X):
        print("Converting counts matrix to CSR format for efficiency...")
        adata.X = adata.X.tocsr()

    print(f"Loaded {adata.n_obs} control cells across {adata.obs['dataset_id'].nunique()} datasets.")

    # 2. Calculate library sizes directly from the matrix
    print("Calculating library sizes...")
    full_lib_sizes = np.array(adata.X.sum(axis=1)).flatten()
    
    # Group library sizes by dataset using the .obs dataframe
    lib_sizes_df = pd.DataFrame({
        "dataset_id": adata.obs['dataset_id'].astype(str),
        "library_size": full_lib_sizes
    })
    lib_sizes_by_dataset = {
        ds: g['library_size'].values for ds, g in lib_sizes_df.groupby('dataset_id')
    }

    # 3. Estimate dispersion parameters (alpha and theta)
    print("Estimating per-gene dispersion (alpha/theta)...")
    
    # Convert the sparse matrix to a dense numpy array before passing.
    if sparse.issparse(adata.X):
        counts_dense = adata.X.toarray()
    else:
        counts_dense = adata.X
        
    alpha = estimate_alpha_from_counts(counts_dense, full_lib_sizes, L_ref=args.l_ref)
    
    # Clip for numerical stability when inverting
    theta = 1.0 / np.clip(alpha, 1e-8, None)

    # 4. Save parameters to a compressed NPZ file
    print(f"Saving parameters to {args.output_npz}...")
    save_dict = {
        'theta': theta,
        'L_ref': np.array(args.l_ref)
    }
    # Add library size arrays to the dictionary for saving
    for ds_id, libs in lib_sizes_by_dataset.items():
        # Sanitize key for NPZ format (e.g., replace '-' or '.')
        safe_key = f"lib_sizes_{ds_id.replace('-', '_').replace('.', '_')}"
        save_dict[safe_key] = libs

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_npz)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    np.savez_compressed(args.output_npz, **save_dict)
    print("\nDone. Re-noising parameters successfully prepared.")


if __name__ == "__main__":
    main()

