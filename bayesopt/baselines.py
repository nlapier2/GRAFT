import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Active Learning Baselines: Random vs. Magnitude Sampling")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the TSV file containing gene_name, LoF_gamma, and HBA1_beta."
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=100, 
        help="Number of genes to reveal in each batch (default: 100)."
    )
    
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.05,
        help="P-value threshold for statistical significance (default: 0.05)."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    return parser.parse_args()

def run_simulation_strategy(name, df_sorted, total_genes, batch_size, p_threshold):
    """
    Runs the batch simulation for a specific ordering of genes (df_sorted).
    """
    print(f"\nRunning Strategy: {name}")
    print(f"{'Batch':<8} | {'Revealed':<8} | {'Corr (Obs)':<11} | {'P (Obs)':<11} | {'Corr (Imp)':<11} | {'P (Imp)':<11}")
    print("-" * 75)

    # These are the full "Ground Truth" vectors
    all_lof_true = df_sorted['LoF_gamma'].values
    all_hba1_true = df_sorted['HBA1_beta'].values
    
    total_batches = math.ceil(total_genes / batch_size)
    
    sig_batch_obs = None
    sig_batch_imp = None

    for batch_idx in range(1, total_batches + 1):
        n_revealed = min(batch_idx * batch_size, total_genes)
        
        # --- Slice Data (Simulate "Revealing" the batch) ---
        # The genes 0..n_revealed are "Known".
        # The genes n_revealed..end are "Unknown".
        
        lof_known = all_lof_true[:n_revealed]
        hba1_known = all_hba1_true[:n_revealed]
        
        # ==========================================
        # Metric 1: Observed Correlation
        # ==========================================
        # Correlation computed ONLY on the subset we have sampled.
        # We use the real LoF values here just for the calculation.
        if n_revealed >= 2:
            # Handle constant input case to avoid warnings
            if np.std(lof_known) == 0 or np.std(hba1_known) == 0:
                corr_obs, p_obs = 0.0, 1.0
            else:
                corr_obs, p_obs = stats.pearsonr(lof_known, hba1_known)
        else:
            corr_obs, p_obs = 0.0, 1.0
            
        if sig_batch_obs is None and p_obs < p_threshold:
            sig_batch_obs = batch_idx

        # ==========================================
        # Metric 2: Imputed Correlation (Average Known)
        # ==========================================
        # LOF: We use the FULL REAL LoF vector (no imputation, we just use the truth).
        # HBA1: We use the HYBRID vector (Real for knowns, Mean of knowns for unknowns).
        
        n_missing = total_genes - n_revealed
        
        if n_missing > 0:
            # Impute HBA1 only
            if len(hba1_known) > 0:
                mean_hba1 = np.mean(hba1_known)
            else:
                mean_hba1 = 0.0 # Fallback if nothing revealed yet
                
            hba1_imputed_tail = np.full(n_missing, mean_hba1)
            hba1_full_hybrid = np.concatenate([hba1_known, hba1_imputed_tail])
        else:
            hba1_full_hybrid = hba1_known
            
        # Calculate correlation: Full Real LoF vs Full Hybrid HBA1
        if np.std(all_lof_true) == 0 or np.std(hba1_full_hybrid) == 0:
             corr_imp, p_imp = 0.0, 1.0
        else:
            corr_imp, p_imp = stats.pearsonr(all_lof_true, hba1_full_hybrid)

        if sig_batch_imp is None and p_imp < p_threshold:
            sig_batch_imp = batch_idx

        # Print row
        print(f"{batch_idx:<8} | {n_revealed:<8} | {corr_obs:+.4f}     | {p_obs:.2e}    | {corr_imp:+.4f}     | {p_imp:.2e}")

    return sig_batch_obs, sig_batch_imp

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found at {args.input_file}")
        return
        
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file, sep='\t')
    
    # Preprocessing: Drop NAs
    # We remove rows that don't have ground truth, as we can't simulate checking them.
    df = df.dropna(subset=['LoF_gamma', 'HBA1_beta'])
    total_genes = len(df)
    
    print(f"Valid genes for simulation: {total_genes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Significance Threshold: {args.p_threshold}")
    
    if total_genes < 3:
        print("Not enough genes to run simulation.")
        return

    # ---------------------------------------------------------
    # Strategy 1: Magnitude Sampling (LoF Known)
    # ---------------------------------------------------------
    # Sort by absolute LoF_gamma
    df_mag = df.copy()
    df_mag['abs_lof'] = df_mag['LoF_gamma'].abs()
    df_mag = df_mag.sort_values(by='abs_lof', ascending=False)
    
    mag_obs, mag_imp = run_simulation_strategy(
        "Magnitude Sorting (LoF Known)", df_mag, total_genes, args.batch_size, args.p_threshold
    )
    
    # ---------------------------------------------------------
    # Strategy 2: Random Sampling (LoF Unknown)
    # ---------------------------------------------------------
    # Shuffle randomly
    df_rnd = df.copy()
    df_rnd = df_rnd.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    rnd_obs, rnd_imp = run_simulation_strategy(
        "Random Sampling (LoF Unknown)", df_rnd, total_genes, args.batch_size, args.p_threshold
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL SUMMARY: Batches needed for Significance")
    print("="*50)
    print(f"{'Strategy':<30} | {'Observed (Subset)':<18} | {'Imputed (Full)':<18}")
    print("-" * 72)
    
    def fmt(val): return str(val) if val else "> Max Batches"
    
    print(f"{'Magnitude Sorting':<30} | {fmt(mag_obs):<18} | {fmt(mag_imp):<18}")
    print(f"{'Random Sampling':<30} | {fmt(rnd_obs):<18} | {fmt(rnd_imp):<18}")
    print("-" * 72)

if __name__ == "__main__":
    main()