import anndata as ad
import pandas as pd
import numpy as np
import argparse
import os

def create_perturbation_splits(adata_file, target_label, control_label, n_splits, output_dir):
    """
    Generates N disjoint train/test splits of an AnnData object based on
    perturbation labels. Control cells are included in all splits.
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

    # --- 2. Validate Column & Create Output Dir ---
    if target_label not in adata.obs.columns:
        print(f"Error: Target label '{target_label}' not found in .obs columns.")
        print(f"Available columns are: {list(adata.obs.columns)}")
        return
        
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    # --- 3. Identify & Partition Perturbations ---
    print(f"Identifying perturbations from '{target_label}'...")
    all_labels = adata.obs[target_label].unique()
    
    # Separate perturbation labels from the control label
    pert_labels = list(set(all_labels) - {control_label})
    
    if not pert_labels:
        print("Error: No perturbation labels found (only control). Cannot create splits.")
        return
        
    print(f"Found {len(pert_labels)} unique perturbations and 1 control ('{control_label}').")

    # Set seed for reproducible shuffling
    np.random.seed(42)
    # Shuffle perturbations before splitting
    np.random.shuffle(pert_labels)
    
    # Split the list of perturbation names into N folds
    # np.array_split handles cases where n_splits doesn't divide len(pert_labels)
    pert_folds = np.array_split(pert_labels, n_splits)
    
    print(f"Partitioned perturbations into {n_splits} folds.")
    for i, fold in enumerate(pert_folds):
        print(f"  Fold {i+1} has {len(fold)} perturbations.")

    # --- 4. Get Control Cell Indices ---
    # We always need these.
    control_mask = (adata.obs[target_label] == control_label)
    control_indices = adata.obs_names[control_mask]

    # --- 5. Loop Through Folds to Create and Save Splits ---
    
    for i in range(n_splits):
        print(f"\n--- Processing Fold {i+1}/{n_splits} ---")
        
        # --- A. Identify Train/Test Perturbations for this fold ---
        
        # Test perturbations are the i-th fold
        test_perts = pert_folds[i]
        
        # Train perturbations are all other folds
        train_perts = []
        for j, fold in enumerate(pert_folds):
            if i != j:
                train_perts.extend(fold)
                
        print(f"  Test perturbations: {len(test_perts)}")
        print(f"  Train perturbations: {len(train_perts)}")
        
        # --- B. Get Cell Indices for Train/Test ---
        
        # Get indices for cells matching test perturbations
        test_pert_mask = adata.obs[target_label].isin(test_perts)
        test_pert_indices = adata.obs_names[test_pert_mask]
        
        # Get indices for cells matching train perturbations
        train_pert_mask = adata.obs[target_label].isin(train_perts)
        train_pert_indices = adata.obs_names[train_pert_mask]

        # --- C. Create Test Set (Test Perts + Controls) ---
        # Combine control indices and test perturbation indices
        test_indices = np.concatenate([control_indices, test_pert_indices])
        
        # Create the test AnnData object
        # .copy() is important!
        adata_test = adata[test_indices, :].copy()
        
        # Define and write the output file
        output_test_file = os.path.join(output_dir, f"test_fold_{i+1}.h5ad")
        try:
            adata_test.write_h5ad(output_test_file)
            print(f"  Wrote test set: {output_test_file} ({len(test_indices)} cells)")
        except Exception as e:
            print(f"  Error writing test file: {e}")

        # --- D. Create Train Set (Train Perts + Controls) ---
        # Combine control indices and train perturbation indices
        train_indices = np.concatenate([control_indices, train_pert_indices])
        
        # Create the train AnnData object
        adata_train = adata[train_indices, :].copy()
        
        # Define and write the output file
        output_train_file = os.path.join(output_dir, f"train_fold_{i+1}.h5ad")
        try:
            adata_train.write_h5ad(output_train_file)
            print(f"  Wrote train set: {output_train_file} ({len(train_indices)} cells)")
        except Exception as e:
            print(f"  Error writing train file: {e}")

    print("\nProcess complete.")

def main():
    """
    Main function to parse arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create N-fold cross-validation splits of an AnnData object, splitting by perturbation."
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
        "--n_splits",
        type=int,
        required=True,
        help="The number of folds (N) to create."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory to save the split files."
    )
    
    args = parser.parse_args()
    
    if args.n_splits <= 0:
        print("Error: --n_splits must be a positive integer.")
        return
        
    create_perturbation_splits(
        args.adata_file, 
        args.target_label, 
        args.control_label, 
        args.n_splits, 
        args.output_dir
    )

if __name__ == "__main__":
    main()