#!/usr/bin/env python3
"""
create_validation_adata.py

Creates a validation AnnData object for VCC, ingestable by GRAFT, by:
1. Copying control cells from training data
2. Creating perturbed cells based on validation template counts
3. Matching the obs structure of the training data
"""

import argparse
import anndata as ad
import pandas as pd
import numpy as np
import uuid
from pathlib import Path


def generate_unique_obs_names(n_obs: int, prefix: str = "vcc_val_cell") -> list:
    """Generate unique observation names."""
    return [f"{prefix}_{i:08d}" for i in range(n_obs)]


def generate_random_guide_ids(n_obs: int) -> list:
    """Generate random guide IDs."""
    return [str(uuid.uuid4())[:8] for _ in range(n_obs)]


def create_validation_adata(train_path: str, val_template_path: str, output_path: str):
    """
    Create validation AnnData object from training data and validation template.
    
    Args:
        train_path: Path to training AnnData file
        val_template_path: Path to validation template AnnData file  
        output_path: Path where to save the new validation AnnData file
    """
    
    print("Loading training data...")
    adata_train = ad.read_h5ad(train_path, backed='r')
    
    print("Loading validation template...")
    val_template = ad.read_h5ad(val_template_path, backed='r')
    
    # Get control cells from training data
    print("Extracting control cells from training data...")
    control_mask = adata_train.obs['target_gene'] == 'non-targeting'
    control_cells = adata_train[control_mask].to_memory()
    
    print(f"Found {control_cells.n_obs} control cells in training data")
    print(f"Control cells X shape: {control_cells.X.shape}")
    
    # Analyze validation template requirements (excluding controls)
    print("Analyzing validation template requirements...")
    val_counts = val_template.obs['target_gene'].value_counts()
    print(f"Validation template cell counts per target:")
    for target, count in val_counts.items():
        print(f"  {target}: {count}")
    
    # Prepare lists to collect data for the new validation object
    all_obs_data = []
    all_X_data = []
    
    # Add ALL control cells from training data (unchanged)
    print(f"\nAdding all {control_cells.n_obs} control cells from training data...")
    # Convert sparse matrix to dense if needed
    control_X = control_cells.X.toarray() if hasattr(control_cells.X, 'toarray') else control_cells.X
    all_X_data.append(control_X)
    all_obs_data.append(control_cells.obs.copy())
    
    # Add perturbed cells for each non-control target
    perturbed_targets = [target for target in val_counts.index if target != 'non-targeting']
    
    if perturbed_targets:
        print(f"\nCreating perturbed cells for {len(perturbed_targets)} targets...")
        
        # Use a representative control cell as template for perturbed cells
        template_control_X = control_X[0:1]  # Keep as 2D array (1, n_genes)
        
        for target in perturbed_targets:
            n_cells = val_counts[target]
            print(f"  Creating {n_cells} cells for {target}")
            
            # Create expression data (copy from template control)
            target_X = np.tile(template_control_X, (n_cells, 1))
            all_X_data.append(target_X)
            
            # Get batch values for this target from validation template
            target_batches = val_template.obs[val_template.obs['target_gene'] == target]['batch_var'].values[:n_cells]
            
            # Create obs data for this target with unique obs names
            target_obs = pd.DataFrame({
                'target_gene': [target] * n_cells,
                'batch': target_batches,
                'guide_id': generate_random_guide_ids(n_cells)
            })
            # Generate unique observation names for perturbed cells only
            target_obs.index = generate_unique_obs_names(n_cells, prefix=f"vcc_val_{target}")
            all_obs_data.append(target_obs)
    
    # Combine all data
    print("\nCombining all data...")
    final_X = np.vstack(all_X_data)
    final_obs = pd.concat(all_obs_data, ignore_index=False)  # Keep original indices for controls
    
    print(f"Final X shape: {final_X.shape}")
    print(f"Final obs shape: {final_obs.shape}")
    
    # Create the new AnnData object
    print("Creating new AnnData object...")
    val_adata = ad.AnnData(
        X=final_X,
        obs=final_obs,
        var=pd.DataFrame(index=adata_train.var.index)  # Keep same gene names, no var metadata
    )
    
    print(f"Created validation AnnData with shape: {val_adata.shape}")
    print(f"Target gene distribution:")
    print(val_adata.obs['target_gene'].value_counts())
    
    # Save the new validation object
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    val_adata.write_h5ad(output_path)
    
    print("Done!")
    

def main():
    parser = argparse.ArgumentParser(description="Create validation AnnData from training data and template")
    parser.add_argument("--train", required=True, help="Path to training AnnData file")
    parser.add_argument("--template", required=True, help="Path to validation template AnnData file")
    parser.add_argument("--output", required=True, help="Path for output validation AnnData file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    create_validation_adata(args.train, args.template, args.output)


if __name__ == "__main__":
    main()
