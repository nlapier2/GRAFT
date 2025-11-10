# For VCC competition -- generate an anndata object from CSV file specifying number of cells per perturbation
# Controls taken from the training data

import anndata as ad
import pandas as pd
import numpy as np
import argparse
import os

def generate_synthetic_adata(input_csv, input_adata, target_label, control_label, output_file):
    """
    Generates a synthetic AnnData object by combining real control cells
    from a template file with synthetically generated perturbation cells
    based on summary statistics from a CSV.
    """
    
    # --- 1. Load Template AnnData ---
    print(f"Loading template AnnData file: {input_adata}")
    try:
        adata = ad.read_h5ad(input_adata)
    except FileNotFoundError:
        print(f"Error: AnnData file not found at {input_adata}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- 2. Validate Column ---
    if target_label not in adata.obs.columns:
        print(f"Error: Target label '{target_label}' not found in .obs columns.")
        print(f"Available columns are: {list(adata.obs.columns)}")
        return

    # --- 3. Load CSV ---
    print(f"Loading CSV file: {input_csv}")
    try:
        csv_df = pd.read_csv(input_csv)
        # Validate required columns
        required_cols = {'target_gene', 'n_cells', 'median_umi_per_cell'}
        if not required_cols.issubset(csv_df.columns):
            print(f"Error: CSV must contain columns: {required_cols}")
            print(f"Found: {list(csv_df.columns)}")
            return
    except FileNotFoundError:
        print(f"Error: CSV file not found at {input_csv}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- 4. Extract Control Cells ---
    print(f"Extracting control cells ('{control_label}')...")
    control_mask = (adata.obs[target_label] == control_label)
    adata_control = adata[control_mask, :].copy()
    
    if adata_control.n_obs == 0:
        print(f"Warning: No control cells found with label '{control_label}'.")
        # We can still proceed, but the final object will only have synthetic cells.
    
    # Store the original var and obs types
    original_obs_dtypes = adata_control.obs.dtypes
    var_template = adata.var.copy()
    n_total_genes = adata.n_vars
    print(f"Found {adata_control.n_obs} control cells and {n_total_genes} genes.")

    # --- 5. Generate Synthetic Perturbation Data ---
    all_adatas_to_concat = [adata_control]
    print(f"Generating synthetic data for {len(csv_df)} perturbations...")
    
    for _, row in csv_df.iterrows():
        target_gene = row['target_gene']
        n_cells = int(row['n_cells'])
        median_umi = float(row['median_umi_per_cell'])
        
        if n_cells <= 0:
            print(f"  Skipping '{target_gene}': n_cells is {n_cells}")
            continue
            
        # Calculate the uniform per-gene expression value
        per_gene_value = median_umi / n_total_genes
        
        print(f"  Creating {n_cells} cells for '{target_gene}' with value {per_gene_value:.2f}...")
        
        # --- Create .X (Expression Matrix) ---
        # Create a dense matrix of shape (n_cells, n_genes)
        X_synthetic = np.full((n_cells, n_total_genes), per_gene_value, dtype=np.float32)
        
        # --- Create .obs (Observation Metadata) ---
        # Create unique cell barcodes for these new cells
        obs_index = [f"synth_{target_gene}_cell_{i}" for i in range(n_cells)]
        # Create the minimal .obs, just the target_label
        obs_synthetic = pd.DataFrame(
            {target_label: target_gene},
            index=obs_index
        )
        
        # --- Assemble AnnData Object ---
        # We re-use the .var from the template file
        adata_pert = ad.AnnData(
            X=X_synthetic,
            obs=obs_synthetic,
            var=var_template
        )
        
        all_adatas_to_concat.append(adata_pert)

    # Concatenate. 'join='outer'' keeps all .obs columns, filling
    # with NaN where data is missing (which is correct for synthetic cells)
    print("Concatenating all AnnData objects...")
    adata_final = ad.concat(all_adatas_to_concat) #, join='outer')
    
    # --- 7. Final Cleanup (Type Conversion) ---
    # `concat` can sometimes change dtypes (e.g., categorical -> object)
    # Let's restore the original types for columns that existed in the control
    print("Cleaning up final .obs dtypes...")
    for col, dtype in original_obs_dtypes.items():
        if col in adata_final.obs.columns:
            try:
                # If the original was categorical, just make the new combined col categorical
                if pd.api.types.is_categorical_dtype(dtype):
                    adata_final.obs[col] = adata_final.obs[col].astype('category')
                # Otherwise, restore the original type
                else:
                    adata_final.obs[col] = adata_final.obs[col].astype(dtype)
            except Exception as e:
                # This might fail if NaNs were introduced into an int col, etc.
                # In that case, we just leave it as is.
                print(f"  Warning: Could not restore dtype for '{col}': {e}")
                pass

    # --- 8. Write Output File ---
    try:
        print(f"\nWriting final synthetic AnnData object to: {output_file}")
        adata_final.write_h5ad(output_file)
        print("Process complete.")
        print(f"Final object shape: {adata_final.shape}")
        print("Final .obs[target_label] counts:")
        print(adata_final.obs[target_label].value_counts())
    except Exception as e:
        print(f"Error writing output file: {e}")

def main():
    """
    Main function to parse arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic AnnData file from real controls and CSV-based perturbations."
    )
    
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the CSV file with synthetic data rules (target_gene, n_cells, median_umi_per_cell)."
    )
    
    parser.add_argument(
        "--input_adata",
        type=str,
        required=True,
        help="Path to the template .h5ad file to source control cells and gene .var."
    )
    
    parser.add_argument(
        "--target_label",
        type=str,
        required=True,
        help="The .obs column name for perturbation/control labels (e.g., 'target_gene')."
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
        help="Path for the output (final) synthetic .h5ad file."
    )
    
    args = parser.parse_args()
    
    generate_synthetic_adata(
        args.input_csv,
        args.input_adata,
        args.target_label,
        args.control_label,
        args.output_file
    )

if __name__ == "__main__":
    main()
