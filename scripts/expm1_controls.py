#!/usr/bin/env python

import anndata as ad
import argparse
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Apply expm1 to CONTROL (non-targeting) cells and merge back."
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the source .h5ad file."
    )
    
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to write the modified .h5ad file."
    )
    
    args = parser.parse_args()

    print(f"--- Loading data from: {args.input_file}")
    try:
        adata = ad.read_h5ad(args.input_file)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"  Loaded data with {adata.n_obs} cells and {adata.n_vars} genes.")

    # --- 0. Make obs names unique ---
    # This prevents errors during splitting and merging
    adata.obs_names_make_unique()
    print("  Made obs_names unique.")

    # --- 1. Identify and split control cells ---
    try:
        control_mask = adata.obs['target_gene'] == 'non-targeting'
    except KeyError:
        print(
            "Error: Column 'target_gene' not found in .obs.", file=sys.stderr
        )
        sys.exit(1)

    n_controls = control_mask.sum()
    n_targets = (~control_mask).sum()
    
    if n_controls == 0:
        print("Warning: No 'non-targeting' (control) cells found. No transformation applied.", file=sys.stderr)
        adata_controls = None
        adata_target = adata.copy() # All cells are non-controls
    elif n_targets == 0:
         print("Warning: Only 'non-targeting' cells found. Applying transform to all cells.", file=sys.stderr)
         adata_controls = adata.copy() # All cells are controls
         adata_target = None
    else:
        print(f"--- Splitting into {n_targets} target cells and {n_controls} control cells.")
        adata_target = adata[~control_mask, :].copy()
        adata_controls = adata[control_mask, :].copy()

    # --- 2. Apply expm1 transform to CONTROL cells ---
    if adata_controls is not None and adata_controls.n_obs > 0:
        print(f"--- Applying expm1() to {adata_controls.n_obs} CONTROL cells...")
        
        # Check if data is sparse or dense and apply appropriate expm1
        from scipy.sparse import issparse
        if issparse(adata_controls.X):
            print("  Data is sparse. Using matrix.expm1().")
            adata_controls.X = adata_controls.X.expm1()
        elif isinstance(adata_controls.X, np.ndarray):
            print("  Data is dense. Using np.expm1().")
            adata_controls.X = np.expm1(adata_controls.X)
        else:
            print(f"  Warning: .X is of unrecognized type {type(adata_controls.X)}. Attempting np.expm1().", file=sys.stderr)
            try:
                adata_controls.X = np.expm1(adata_controls.X)
            except Exception as e:
                print(f"Error applying expm1: {e}", file=sys.stderr)
                print("  Could not transform .X data. Aborting.", file=sys.stderr)
                sys.exit(1)
                
        print("  Transform complete.")
    else:
        print("--- No control cells to transform.")

    # --- 3. Add non-control cells back in ---
    if adata_controls is None:
        final_adata = adata_target
    elif adata_target is None:
        final_adata = adata_controls
    else:
        print("--- Merging original target cells and transformed control cells...")
        # Concatenate the two objects
        final_adata = ad.concat([adata_target, adata_controls], join='outer')
        
        # Restore the original cell order
        print("--- Restoring original cell order...")
        final_adata = final_adata[adata.obs_names, :].copy()

    # --- 4. Write output ---
    print(f"\n--- Writing modified data to: {args.output_file}")
    try:
        final_adata.write_h5ad(args.output_file)
        print("--- Done.")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
