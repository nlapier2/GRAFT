import anndata as ad
import pandas as pd
import argparse

def create_indicator_matrix(adata_file, target_label, control_label, output_file):
    """
    Creates a binary perturbation/control indicator matrix from an AnnData object.

    The input .obs column is assumed to handle multi-perturbations via
    comma-separated strings (e.g., "GeneA,GeneB").
    """
    
    # --- 1. Load Data ---
    print(f"Loading AnnData file from: {adata_file}")
    try:
        adata = ad.read_h5ad(adata_file)
    except FileNotFoundError:
        print(f"Error: AnnData file not found at {adata_file}")
        return
    except Exception as e:
        print(f"Error loading AnnData file: {e}")
        return

    # --- 2. Validate Column ---
    if target_label not in adata.obs.columns:
        print(f"Error: Target label '{target_label}' not found in .obs columns.")
        print(f"Available columns are: {list(adata.obs.columns)}")
        return

    print(f"Using '{target_label}' column to create matrix.")
    
    # Ensure the column is treated as strings, handling potential NaNs
    pert_series = adata.obs[target_label].fillna('').astype(str)

    # --- 3. Create Binary Matrix using get_dummies ---
    # This is the most efficient way to do this.
    # It handles comma-separated values automatically.
    print("Generating binary matrix...")
    try:
        matrix_df = pert_series.str.get_dummies(sep=',')
    except Exception as e:
        print(f"Error during dummy matrix creation: {e}")
        return

    # --- 4. Format Columns ---
    
    # Get all unique perturbations found
    all_labels = set(matrix_df.columns)
    
    if control_label not in all_labels:
        print(f"Warning: Control label '{control_label}' was not found in the data.")
        # Add the control column with all zeros if it wasn't present
        matrix_df[control_label] = 0
    
    # Get all perturbation columns (everything *except* control)
    pert_columns = sorted([label for label in all_labels if label != control_label and label != ''])
    
    # Define final column order: sorted perturbations, then control
    final_columns = pert_columns + [control_label]
    
    # Reindex the DataFrame to match this order
    # fill_value=0 handles any labels we want to discard (like '')
    matrix_df = matrix_df.reindex(columns=final_columns, fill_value=0)

    # --- 5. Prepare for Output ---
    
    # Set the index name, which will become the first column header
    matrix_df.index.name = "Cell_Barcode"
    
    # Move the index (Cell_Barcode) to be the first column
    matrix_df.reset_index(inplace=True)

    # --- 6. Write to File ---
    print(f"Writing matrix to: {output_file}")
    try:
        # Use tab-separation for robustness (it's still whitespace)
        matrix_df.to_csv(output_file, sep='\t', index=False)
    except Exception as e:
        print(f"Error writing output file: {e}")
        return
        
    print("\nProcess complete.")
    print(f"Final matrix shape: {matrix_df.shape}")
    print(f"Columns: {list(matrix_df.columns[:5])}...{list(matrix_df.columns[-2:])}")


def main():
    """
    Main function to parse arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a binary perturbation indicator matrix from an AnnData file."
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
        help="The .obs column name with perturbation labels (e.g., 'target_gene')."
    )
    
    parser.add_argument(
        "--control_label",
        type=str,
        required=True,
        help="The specific label for control cells (e.g., 'non-targeting')."
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the output whitespace-delimited file."
    )
    
    args = parser.parse_args()
    
    create_indicator_matrix(
        args.adata_file, 
        args.target_label, 
        args.control_label, 
        args.output_file
    )

if __name__ == "__main__":
    main()
