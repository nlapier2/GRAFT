import anndata as ad
import pandas as pd
import numpy as np
import argparse

def subset_anndata(adata_file, target_label, control_label, n_control, n_per_pert, output_file):
    """
    Subsets an AnnData object to a maximum number of cells for controls
    and for each individual perturbation.
    """
    
    # --- 1. Load Data ---
    print(f"Loading AnnData file: {adata_file}")
    try:
        adata = ad.read_h5ad(adata_file)
    except FileNotFoundError:
        print(f"Error: AnnData file not found at {adata_file}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- 2. Validate Column ---
    if target_label not in adata.obs.columns:
        print(f"Error: Target label '{target_label}' not found in .obs columns.")
        print(f"Available columns are: {list(adata.obs.columns)}")
        return

    # Set seed for reproducible sampling
    np.random.seed(42)

    print(f"Subsetting based on '{target_label}' column.")
    print(f"Control label: '{control_label}'")
    print(f"Max controls: {n_control}")
    print(f"Max cells per perturbation: {n_per_pert}")

    # --- 3. Identify Labels ---
    
    # Get all unique labels. This will include the control_label
    # and all perturbation labels (e.g., "GeneA", "GeneB", "GeneA,GeneB", etc.)
    all_labels = adata.obs[target_label].unique()
    
    # Separate perturbation labels from the control label
    pert_labels = list(set(all_labels) - {control_label})
    
    # This list will hold all the cell barcodes (indices) we want to keep
    indices_to_keep = []

    # --- 4. Subset Controls ---
    control_indices = adata.obs[adata.obs[target_label] == control_label].index
    print(f"\nFound {len(control_indices)} total control cells.")
    
    if len(control_indices) > n_control:
        print(f"  Sampling {n_control} control cells...")
        # Randomly choose n_control cells without replacement
        chosen_controls = np.random.choice(control_indices, size=n_control, replace=False)
        indices_to_keep.append(chosen_controls)
    else:
        print(f"  Keeping all {len(control_indices)} control cells.")
        indices_to_keep.append(control_indices)
        
    # --- 5. Subset Perturbations (Iteratively) ---
    print(f"\nFound {len(pert_labels)} unique perturbations to process...")
    
    for pert in sorted([p for p in pert_labels if not pd.isna(p)]):
        
        # Find all cells matching this specific perturbation label
        pert_indices = adata.obs[adata.obs[target_label] == pert].index
        
        if len(pert_indices) > n_per_pert:
            print(f"  Sampling {n_per_pert} / {len(pert_indices)} cells for '{pert}'...")
            # Randomly choose n_per_pert cells
            chosen_perts = np.random.choice(pert_indices, size=n_per_pert, replace=False)
            indices_to_keep.append(chosen_perts)
        else:
            # Keep all cells if there are fewer than or equal to the max
            print(f"  Keeping all {len(pert_indices)} cells for '{pert}'...")
            indices_to_keep.append(pert_indices)
            
    # --- 6. Concatenate indices and create final subset ---
    if not indices_to_keep:
        print("Error: No cells were selected. Please check your labels.")
        return
        
    final_indices = np.concatenate(indices_to_keep)
    print(f"\nTotal cells in final subset: {len(final_indices)}")
    
    # Create the new AnnData object from the selected indices
    # .copy() is important to create a new object, not a view
    adata_subset = adata[final_indices, :].copy()
    
    # --- 7. Write Output ---
    try:
        adata_subset.write_h5ad(output_file)
        print(f"Successfully wrote subsetted file to: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

def main():
    """
    Main function to parse arguments.
    """
    parser = argparse.ArgumentParser(
        description="Subset an AnnData object by sampling cells from perturbation and control groups."
    )
    
    parser.add_argument(
        "--adata_file",
        type=str,
        required=True,
        help="Path to the input .h5ad file."
    )
    
    parser.add_argument(
        "--target_label",
        type=str,
        required=True,
        help="The .obs column name with perturbation/control labels (e.g., 'target_gene')."
    )
    
    parser.add_argument(
        "--control_label",
        type=str,
        required=True,
        help="The specific label for control cells (e.g., 'non-targeting')."
    )
    
    parser.add_argument(
        "--n_control",
        type=int,
        required=True,
        help="Maximum number of control cells to keep."
    )
    
    parser.add_argument(
        "--n_per_pert",
        type=int,
        required=True,
        help="Maximum number of cells to keep *for each* perturbation."
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the output subsetted .h5ad file."
    )
    
    args = parser.parse_args()
    
    subset_anndata(
        args.adata_file, 
        args.target_label, 
        args.control_label, 
        args.n_control, 
        args.n_per_pert, 
        args.output_file
    )

if __name__ == "__main__":
    main()
