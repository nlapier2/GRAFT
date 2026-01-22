import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Baseline Active Learning Simulation for LoF/HBA1 Correlation")
    
    # Input file: Named argument but required
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the TSV file containing gene_name, LoF_gamma, and HBA1_beta."
    )
    
    # Batch size: Named argument, defaults to 100
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=100, 
        help="Number of genes to reveal in each batch (default: 100)."
    )
    
    # Optional threshold, just in case you want to change it later
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.05,
        help="P-value threshold for statistical significance (default: 0.05)."
    )

    return parser.parse_args()

def run_baseline_simulation(args):
    file_path = args.input_file
    batch_size = args.batch_size
    p_value_threshold = args.p_threshold

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading data from {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path, sep='\t')
    
    # 1. Preprocessing: Drop NAs
    # We can only "simulate" discovery on data we actually have ground truth for.
    initial_count = len(df)
    df = df.dropna(subset=['LoF_gamma', 'HBA1_beta'])
    filtered_count = len(df)
    
    print(f"Total genes: {initial_count}")
    print(f"Genes after filtering NAs: {filtered_count}")
    
    if filtered_count < 3:
        print("Error: Not enough data points to compute correlation.")
        return

    # 2. Strategy: Sort by descending absolute LoF_gamma
    # Heuristic: Genes with strong LoF effects on the complex trait are prioritized.
    df['abs_lof'] = df['LoF_gamma'].abs()
    df = df.sort_values(by='abs_lof', ascending=False).reset_index(drop=True)
    
    # Prepare vectors
    all_lof = df['LoF_gamma'].values
    all_hba1_true = df['HBA1_beta'].values
    
    total_batches = math.ceil(filtered_count / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Possible batches: {total_batches}\n")
    
    # Track when we hit significance
    sig_batch_observed = None
    sig_batch_imputed = None
    
    print(f"{'Batch':<10} | {'Genes':<10} | {'Corr (Obs)':<12} | {'P-val (Obs)':<12} | {'Corr (Imp)':<12} | {'P-val (Imp)':<12}")
    print("-" * 85)

    for batch_idx in range(1, total_batches + 1):
        # Determine how many genes are "revealed" in this batch
        n_revealed = min(batch_idx * batch_size, filtered_count)
        
        # --- Criterion 1: Observed Correlation ---
        # Correlation using ONLY the genes revealed so far
        lof_revealed = all_lof[:n_revealed]
        hba1_revealed = all_hba1_true[:n_revealed]
        
        # We need at least 2 points to calculate correlation
        if n_revealed >= 2:
            corr_obs, p_obs = stats.pearsonr(lof_revealed, hba1_revealed)
        else:
            corr_obs, p_obs = 0.0, 1.0
            
        # Check if this is the first time we crossed the threshold
        if sig_batch_observed is None and p_obs < p_value_threshold:
            sig_batch_observed = batch_idx

        # --- Criterion 2: Imputed Correlation (Average Known) ---
        # Correlation using ALL genes, imputing the unknown ones with the mean of the known ones
        
        # 1. Calculate mean of revealed targets
        mean_known = np.mean(hba1_revealed)
        
        # 2. Fill the "unrevealed" portion of the vector with this mean
        n_remaining = filtered_count - n_revealed
        if n_remaining > 0:
            hba1_imputed_tail = np.full(n_remaining, mean_known)
            hba1_hybrid = np.concatenate([hba1_revealed, hba1_imputed_tail])
        else:
            hba1_hybrid = hba1_revealed # All genes revealed
            
        # 3. Correlate full LoF vector with Hybrid HBA1 vector
        # Note: We correlate against all_lof (known for all genes)
        corr_imp, p_imp = stats.pearsonr(all_lof, hba1_hybrid)
        
        # Check if this is the first time we crossed the threshold
        if sig_batch_imputed is None and p_imp < p_value_threshold:
            sig_batch_imputed = batch_idx

        # Print Status
        print(f"{batch_idx:<10} | {n_revealed:<10} | {corr_obs:.4f}       | {p_obs:.4e}   | {corr_imp:.4f}       | {p_imp:.4e}")

    print("-" * 85)
    print("\nResults:")
    
    if sig_batch_observed:
        print(f"Criterion 1 (Observed only): reached significance (p < {p_value_threshold}) at Batch {sig_batch_observed}.")
    else:
        print(f"Criterion 1 (Observed only): DID NOT reach significance.")
        
    if sig_batch_imputed:
        print(f"Criterion 2 (Imputation):    reached significance (p < {p_value_threshold}) at Batch {sig_batch_imputed}.")
    else:
        print(f"Criterion 2 (Imputation):    DID NOT reach significance.")

if __name__ == "__main__":
    args = parse_arguments()
    run_baseline_simulation(args)
