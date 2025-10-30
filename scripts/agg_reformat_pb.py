import anndata as ad
import numpy as np
import pandas as pd
import os
import argparse
import pickle
from scipy.sparse import issparse

def aggregate_and_reformat(input_dir, mapping_file, template_file, output_file):
    """
    Aggregates pseudobulked AnnData files, maps gene names, reformats .obs,
    and aligns to a template AnnData file.
    """
    
    print("Starting aggregation and reformatting process...")

    # --- 1. Find and Concatenate AnnData files ---
    try:
        filenames = [f for f in os.listdir(input_dir) if f.endswith(".h5ad")]
        if not filenames:
            print(f"Error: No .h5ad files found in {input_dir}")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_dir}")
        return

    print(f"Found {len(filenames)} .h5ad files to concatenate...")
    adata_list = []
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        try:
            adata_list.append(ad.read_h5ad(file_path))
        except Exception as e:
            print(f"  Warning: Could not read {filename}. Error: {e}")

    if not adata_list:
        print("Error: No valid AnnData files were loaded. Exiting.")
        return

    # Concatenate all files. 
    # 'outer' join fills missing genes with NaN for now.
    # 'same' merge assumes .var columns (like 'n_cells') are consistent.
    try:
        adata = ad.concat(adata_list, join='outer', merge='same')
        print(f"Successfully concatenated files. Resulting shape: {adata.shape}")
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return

    # --- 2. Map Gene Names (ENSG -> Human-readable) ---
    print(f"Loading gene mapping file from {mapping_file}...")
    try:
        with open(mapping_file, 'rb') as f:
            gene_map = pickle.load(f)
        if not isinstance(gene_map, dict):
            print("Error: Mapping file did not contain a dictionary.")
            return
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # Map current gene names (in .var.index) to new names
    adata.var['mapped_gene_name'] = adata.var.index.map(gene_map)
    
    # Filter out genes that were not in the mapping dictionary
    genes_to_keep_mask = adata.var['mapped_gene_name'].notna()
    adata = adata[:, genes_to_keep_mask].copy()
    print(f"Filtered genes not in mapping dict. New shape: {adata.shape}")

    # Handle potential duplicate gene names (e.g., multiple ENSGs -> one HUGO)
    # We aggregate by taking the mean expression.
    if adata.var['mapped_gene_name'].duplicated().any():
        print("Duplicate gene names found after mapping. Aggregating by mean...")
        
        # Convert to DataFrame for easy aggregation
        # Use .toarray() to handle sparse or dense
        X_df = pd.DataFrame(
            adata.X.toarray() if issparse(adata.X) else adata.X, 
            index=adata.obs.index, 
            columns=adata.var['mapped_gene_name']
        )
        
        # Group by column name (axis=1) and take the mean
        X_df_agg = X_df.groupby(level=0, axis=1).mean()
        
        # Create new .var DataFrame
        new_var = pd.DataFrame(index=X_df_agg.columns)
        new_var.index.name = 'gene_name'
        
        # Re-create AnnData object
        adata = ad.AnnData(X=X_df_agg.values, obs=adata.obs, var=new_var)
        print(f"Aggregated duplicates. New shape: {adata.shape}")
    else:
        # If no duplicates, just set the new index
        adata.var_names = adata.var['mapped_gene_name']
        adata.var.index.name = 'gene_name'
        adata.var = adata.var.drop(columns=['mapped_gene_name'])

    # --- 3. Reformat .obs ---
    print("Reformatting .obs columns...")
    if 'gene_target' in adata.obs.columns:
        adata.obs.rename(columns={'gene_target': 'target_gene'}, inplace=True)
        print("  Renamed 'gene_target' -> 'target_gene'")
        
        # Check if replacement is needed
        if 'Non-Targeting' in adata.obs['target_gene'].unique():
            print("  Found 'Non-Targeting'. Attempting replacement...")
            
            # Robust replacement: check if dtype is categorical
            is_categorical = pd.api.types.is_categorical_dtype(adata.obs['target_gene'])
            
            if is_categorical:
                # Use pandas' built-in category renaming
                adata.obs['target_gene'] = adata.obs['target_gene'].cat.rename_categories(
                    {'Non-Targeting': 'non-targeting'}
                )
            else:
                # Fallback to string replacement
                adata.obs['target_gene'] = adata.obs['target_gene'].astype(str).replace(
                    'Non-Targeting', 'non-targeting'
                )
            
            # Check if successful
            if 'non-targeting' in adata.obs['target_gene'].unique():
                print("  Successfully relabeled 'Non-Targeting' -> 'non-targeting'")
            else:
                print("  Warning: Replacement may not have been successful.")
        else:
             print("  'Non-Targeting' not found in 'target_gene' column. Skipping relabel.")
    else:
        print("  Warning: 'gene_target' column not found. Skipping rename/relabel.")

    # --- 4. Align with Template File ---
    print(f"Loading template file from {template_file}...")
    try:
        template_adata = ad.read_h5ad(template_file)
    except Exception as e:
        print(f"Error loading template file: {e}")
        return

    template_genes = template_adata.var.index
    current_genes = adata.var.index
    
    print(f"Aligning {current_genes.nunique()} genes to {template_genes.nunique()} template genes.")

    # Find genes present in both
    shared_genes = current_genes.intersection(template_genes)
    
    # Get the list of shared genes, but in the order of the template file
    final_gene_list = template_genes[template_genes.isin(shared_genes)]
    
    print(f"  Found {len(final_gene_list)} shared genes. Subsetting and reordering to match template.")
    
    # Create the final AnnData object
    # 1. Subset and reorder our aggregated data to this final gene list
    final_adata = adata[:, final_gene_list].copy()
    
    # 2. Assign the corresponding .var metadata from the template
    #    This ensures .var is also aligned
    final_adata.var = template_adata.var.loc[final_gene_list].copy()

    # --- 5. Write Output ---
    print(f"\nAlignment complete. Final shape: {final_adata.shape}")
    try:
        final_adata.write_h5ad(output_file)
        print(f"Successfully saved final file to: {output_file}")
    except Exception as e:
        print(f"Error writing final .h5ad file: {e}")

def main():
    """
    Main function to parse arguments and run the aggregation.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate, remap, and reformat pseudobulked AnnData files."
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the pseudobulked .h5ad files from the previous script."
    )
    
    parser.add_argument(
        "--mapping_file",
        type=str,
        required=True,
        help="Path to a .pkl file containing a dict mapping ENSG IDs to gene names."
    )
    
    parser.add_argument(
        "--template_file",
        type=str,
        required=True,
        help="Path to the template .h5ad file to align genes against."
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Full path for the final, processed .h5ad output file."
    )
    
    args = parser.parse_args()
    
    aggregate_and_reformat(
        args.input_dir, 
        args.mapping_file, 
        args.template_file, 
        args.output_file
    )

if __name__ == "__main__":
    main()
