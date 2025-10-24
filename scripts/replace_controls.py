#!/usr/bin/env python

# Used to replace log1p'd controls in VCC validation with original controls

import anndata as ad
import argparse
import sys
import numpy as np

def main():
    """
    Main function to parse arguments and perform control replacement.
    """
    parser = argparse.ArgumentParser(
        description="Replace control cell expression data in a target AnnData object."
    )
    
    parser.add_argument(
        "controls_file",
        type=str,
        help="Path to the .h5ad file containing the source control cell data."
    )
    
    parser.add_argument(
        "target_file",
        type=str,
        help="Path to the .h5ad file to be modified (e.g., 'tmp.h5ad')."
    )
    
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to write the modified .h5ad file."
    )
    
    args = parser.parse_args()

    print(f"--- Loading Controls: {args.controls_file}")
    try:
        controls = ad.read_h5ad(args.controls_file)
    except Exception as e:
        print(f"Error reading controls file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"--- Loading Target: {args.target_file}")
    try:
        target = ad.read_h5ad(args.target_file)
    except Exception as e:
        print(f"Error reading target file: {e}", file=sys.stderr)
        sys.exit(1)

    print("--- Validating data dimensions...")

    # 1. Validate gene count
    if controls.shape[1] != target.shape[1]:
        print(
            f"Error: Gene count mismatch!", file=sys.stderr
        )
        print(
            f"  Controls file has {controls.shape[1]} genes.", file=sys.stderr
        )
        print(
            f"  Target file has {target.shape[1]} genes.", file=sys.stderr
        )
        print("  Cannot proceed as genes are not aligned.", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Gene count matches: {controls.shape[1]}")

    # 2. Validate control cell count
    try:
        # Create a boolean mask for control cells in the target object
        control_mask = target.obs['target_gene'] == 'non-targeting'
    except KeyError:
        print(
            "Error: Column 'target_gene' not found in target.obs.", file=sys.stderr
        )
        sys.exit(1)

    n_target_controls = control_mask.sum()
    n_source_controls = controls.shape[0]

    if n_target_controls != n_source_controls:
        print(
            f"Error: Control cell count mismatch!", file=sys.stderr
        )
        print(
            f"  Source file has {n_source_controls} control cells.", file=sys.stderr
        )
        print(
            f"  Target file has {n_target_controls} cells marked as 'non-targeting'.", file=sys.stderr
        )
        print("  Cannot proceed as cell counts do not match.", file=sys.stderr)
        sys.exit(1)

    print(f"  Found matching number of control cells: {n_source_controls}")

    # --- Perform Replacement ---
    print("\n--- Replacing control cell expression data...")
    
    # This row-slicing assignment works for both dense (numpy) 
    # and sparse (scipy.sparse.csr_matrix/csc_matrix) formats in .X
    try:
        # Convert boolean mask to integer indices for sparse matrix compatibility
        control_indices = np.where(control_mask)[0]
        # Assign using integer indices, which is robust for both sparse and dense
        target.X[control_indices] = controls.X.toarray()
        print("  Replacement complete.")
    except Exception as e:
        print(f"Error during data replacement: {e}", file=sys.stderr)
        print("  This can sometimes happen with mismatched sparse/dense formats.", file=sys.stderr)
        sys.exit(1)


    # --- Write Output ---
    print(f"\n--- Writing modified data to: {args.output_file}")
    try:
        target.write_h5ad(args.output_file)
        print("--- Done.")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
