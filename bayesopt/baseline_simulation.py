from email import parser
import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import sys
import argparse
import matplotlib.pyplot as plt
import anndata as ad
import scipy.sparse as sp

from active_gp import ActiveGPLearner
from active_strategies import StaticStrategy, StaticGPStrategy, HighLeverageStrategy, UncertaintyStrategy, DiversityStrategy, PCUncertaintyStrategy, VarianceReductionStrategy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Active Learning Baselines: Random vs. Magnitude Sampling")
    parser.add_argument("--lof", type=str, required=True, help="Path to LoF burden test TSV (Source of Gamma).")
    parser.add_argument("--limma", type=str, required=True, help="Path to Limma perturbation TSV (Source of Betas).")
    parser.add_argument("--mapping_file", type=str, default=None, help="Optional TSV (col1=ENSG, col2=Symbol) for gene mapping.")
    parser.add_argument("--module_file", type=str, required=True, help="Path to the NMF module definitions TSV.")
    parser.add_argument("--module_name", type=str, required=True, help="Name of the module (e.g., 'Factor_1') to target.")
    parser.add_argument("--module_cov_agg", type=str, default="mean", choices=["mean", "max"], help="Aggregation method for module covariance (mean/max).")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of genes to reveal in each batch (default: 100).")
    parser.add_argument("--print_every", type=int, default=10, help="Print progress every N batches (default: 10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the correlation plots.")

    # Arguments for Control Strategy
    parser.add_argument("--control_h5ad", type=str, default=None, help="Path to H5AD file containing control cells for covariance calculation.")
    parser.add_argument("--target_label", type=str, default="target_gene", help="Obs column name identifying perturbation/control status (default: target_gene).")
    parser.add_argument("--control_label", type=str, default="", help="Value in target_label that identifies control cells. If blank, uses all cells (default: '').")

    # Argument for External Perturbation Strategy
    parser.add_argument("--external_h5ad", type=str, default=None, help="Path to 'external' pseudobulked H5AD for perturbation effect calculation.")

    # --- GP Imputation Arguments ---
    parser.add_argument("--external_list", type=str, default="", help="List of external h5ad files for GP kernel.")
    parser.add_argument("--embeddings_yaml", type=str, default="", help="Single YAML file defining pathway/embedding sources.")
    parser.add_argument("--emb_metric", type=str, default="cosine", help="Metric for embedding kernels (cosine/rbf).")
    parser.add_argument("--kernel_weight_gamma", type=float, default=1.0, help="Gamma parameter for kernel alignment weights.")
    parser.add_argument("--kernel_agg", type=str, default="mean", choices=["mean", "wmean"], help="GP Kernel aggregation method.")
    parser.add_argument("--gp_noise_var", type=float, default=0.01, help="GP noise variance (lambda).")
    parser.add_argument("--gp_recompute_freq", type=int, default=5, help="How often to re-weight GP kernels (batches).")
    
    # --- Active Learning Arguments ---
    parser.add_argument("--run_active_leverage", action="store_true", help="Run the Active High Leverage strategy.")
    parser.add_argument("--run_active_uncertainty", action="store_true", help="Run the Active Uncertainty strategy.")
    parser.add_argument("--run_active_diversity", action="store_true", help="Run the Active Diversity strategy.")
    parser.add_argument("--run_active_pca", action="store_true", help="Run the Active PC-Uncertainty strategy.")
    parser.add_argument("--run_active_var_reduction", action="store_true", help="Run the Active Stepwise Variance Reduction (Kriging Believer) strategy.")

    parser.add_argument("--acq_beta", type=float, default=1.0, help="Beta parameter for acquisition (mean vs std trade-off).")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to run (optional limit).")
    parser.add_argument("--pca_recompute_freq", type=int, default=1, help="Batch frequency to recompute PCA for PC-Uncertainty.")
    parser.add_argument("--pca_top_k", type=int, default=50, help="Number of Principal Components to use for uncertainty.")
    parser.add_argument("--stepwise_subset_size", type=int, default=400, help="Size of the 'Working Set' for stepwise variance reduction (speed optimization).")

    # --- Static Strategy Options ---
    parser.add_argument("--imputation_method", type=str, default="mean", choices=["mean", "zero", "noise"], help="Imputation method for static strategies: 'mean' (AverageKnown), 'zero', or 'noise' (Empirical Variance centered at 0).")
    parser.add_argument("--gp_imputation_mode", type=str, default="mean", choices=["mean", "sample"], help="For GP strategies: 'mean' (Posterior Mean) or 'sample' (Sample from Posterior).")
    parser.add_argument("--sampling_strategy", type=str, default="strongest", choices=["strongest", "uniform"], help="Order to pick genes for static strategies: 'strongest' (Magnitude Descending) or 'uniform' (Stratified across range).")
    parser.add_argument("--random_samp_pct", type=float, default=0.0, help="Percentage of batch (0.0-1.0) to select randomly for static strategies.")
    parser.add_argument("--static_only", action="store_true", help="Skip all active learning strategies (GP/Active).")

    return parser.parse_args()


def get_gene_mapping(ids_to_map, mapping_file=None):
    """
    Returns a dictionary mapping ID -> Symbol.
    Only queries MyGene for IDs that start with 'ENSG'.
    """
    mapping = {}
    ensembl_ids = [x for x in ids_to_map if str(x).startswith("ENSG")]
    
    if not ensembl_ids:
        return mapping

    if mapping_file:
        print(f"[info] Loading gene mapping from {mapping_file}...")
        try:
            map_df = pd.read_csv(mapping_file, sep='\t', header=0)
            if map_df.shape[1] < 2:
                map_df = pd.read_csv(mapping_file, sep='\t', header=None)
            mapping = pd.Series(map_df.iloc[:, 1].values, index=map_df.iloc[:, 0].values).to_dict()
        except Exception as e:
            sys.exit(f"Error reading mapping file: {e}")
            
    else:
        print(f"[info] Querying MyGene.info for {len(ensembl_ids)} genes...")
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
            for res in results:
                query = res.get('query')
                symbol = res.get('symbol')
                if query and symbol:
                    mapping[query] = symbol
        except ImportError:
            sys.exit("Error: 'mygene' module not found. Install it (pip install mygene) or provide --mapping_file.")
        except Exception as e:
            sys.exit(f"Error querying MyGene.info: {e}")

    return mapping

def process_series(series, name, mapper):
    """
    Takes a pandas Series (index=IDs/Symbols), maps index to Symbols,
    aggregates duplicates (mean), and returns cleaned Series.
    """
    new_index = series.index.map(lambda x: mapper.get(x, x))
    series.index = new_index
    
    if series.index.duplicated().any():
        series = series.groupby(series.index).mean()
        
    return series

def get_symbol(name, mapper):
    return mapper.get(name, name)


def run_simulation_strategy(strategy, y_true_matrix, total_genes, batch_size, max_batches=None, print_every=10):
    """
    Runs the simulation using a Strategy object on a Module of X genes.
    
    Args:
        strategy: An instance of BaseStrategy.
        y_true_matrix: (N_perturbations x X_targets) matrix containing Truth.
        total_genes: N.
    """
    print(f"\nRunning Strategy: {strategy.name}")
    print(f"{'Batch':<8} | {'Revealed':<8} | {'Avg MSE':<12} | {'Avg Beta Corr':<15}")
    print("-" * 55)
    
    # State
    revealed_mask = np.zeros(total_genes, dtype=bool)
    n_revealed = 0

    history_mse = []
    history_r_beta = []
    
    # Diagnostics storage
    diag_target_mag = []
    diag_conn = []
    
    batch_idx = 0
    X_targets = y_true_matrix.shape[1]
    
    while n_revealed < total_genes:
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached max_batches ({max_batches}). Stopping.")
            break

        batch_idx += 1
        
        # 1. Ask Strategy for next batch indices
        new_indices = strategy.select_next_batch(batch_size, revealed_mask, y_true_matrix[revealed_mask])
        
        if len(new_indices) == 0:
            break
            
        # 2. Reveal Data
        revealed_mask[new_indices] = True
        n_revealed = np.sum(revealed_mask)
        
        y_known = y_true_matrix[revealed_mask]
        
        # 3. Notify Strategy
        strategy.update(new_indices, y_true_matrix[new_indices])

        # ==========================================
        # Diagnostic: Analyze Selected Batch Quality
        # ==========================================
        batch_target_abs = np.mean(np.abs(y_true_matrix[new_indices]))
        diag_target_mag.append(batch_target_abs)
        
        learner = getattr(strategy, 'learner', None)
        if learner is not None and getattr(learner, 'K_fused', None) is not None:
            K_sub = learner.K_fused[new_indices, :]
            batch_conn = np.mean(np.abs(K_sub))
        else:
            batch_conn = np.nan
        diag_conn.append(batch_conn)

        # ==========================================
        # Metric: Imputed Beta Reconstruction
        # ==========================================
        prediction = strategy.predict(revealed_mask, y_known)
        
        if prediction is not None:
            y_full_hybrid = prediction
        else:
            imp_method = getattr(strategy.args, 'imputation_method', 'mean')
            
            if imp_method == 'zero':
                y_full_hybrid = np.copy(y_true_matrix)
                y_full_hybrid[~revealed_mask] = 0.0
                
            elif imp_method == 'noise':
                scale = np.std(y_known, axis=0) if len(y_known) > 1 else np.zeros(X_targets)
                n_missing = np.sum(~revealed_mask)
                # Native numpy broadcasting handles size=(n_missing, X_targets) perfectly
                noise_vec = np.random.normal(loc=0.0, scale=scale, size=(n_missing, X_targets))
                
                y_full_hybrid = np.copy(y_true_matrix)
                y_full_hybrid[~revealed_mask] = noise_vec

            else:
                # 'mean' (AverageKnown)
                mean_val = np.mean(y_known, axis=0) if len(y_known) > 0 else np.zeros(X_targets)
                y_full_hybrid = np.copy(y_true_matrix) 
                y_full_hybrid[~revealed_mask] = mean_val 
            
        # Metric 1: MSE (per target) -> Array of size X
        mse = np.mean((y_true_matrix - y_full_hybrid) ** 2, axis=0)
        history_mse.append(mse)

        # Metric 2: Beta Correlation (per target) -> Array of size X
        r_betas = []
        for i in range(X_targets):
            true_col = y_true_matrix[:, i]
            pred_col = y_full_hybrid[:, i]
            if np.std(true_col) > 0 and np.std(pred_col) > 0:
                r = stats.pearsonr(true_col, pred_col)[0]
            else:
                r = 0.0
            r_betas.append(r)
            
        history_r_beta.append(np.array(r_betas))

        # Print Average over the X targets
        if batch_idx == 1 or batch_idx % print_every == 0 or n_revealed == total_genes:
            avg_mse = np.mean(mse)
            avg_r = np.mean(r_betas)
            print(f"{batch_idx:<8} | {n_revealed:<8} | {avg_mse:.4e}   | {avg_r:+.4f}")

    # Aggregate Diagnostics
    avg_diag = {
        'target_mag': np.mean(diag_target_mag) if diag_target_mag else 0.0,
        'conn': np.nanmean(diag_conn) if not np.all(np.isnan(diag_conn)) else 0.0,
        'final_r_beta_mean': np.mean(history_r_beta[-1]) if history_r_beta else 0.0
    }

    return history_mse, history_r_beta, avg_diag


def plot_pvalue_history(p_values, method_name, output_dir, true_p_val=None):
    """
    Plots the -log10(p-value) over batches for a single approach.
    If true_p_val is provided, it is plotted as the 'Ground Truth' reference line.
    """
    if not p_values:
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batches = range(1, len(p_values) + 1)
    
    # Cap p-values at 1e-20
    min_p = 1e-20
    capped_p_values = [max(p, min_p) for p in p_values]
    
    # Convert to -log10
    nlog10_p = [-np.log10(p) for p in capped_p_values]
    
    # Determine Reference Line (Ground Truth or Final Batch)
    if true_p_val is not None:
        ref_p = max(true_p_val, min_p)
        label_text = "Ground Truth"
    else:
        ref_p = capped_p_values[-1]
        label_text = "Final Batch"

    ref_nlog10 = -np.log10(ref_p)
    
    thresh_p = 0.05
    thresh_nlog10 = -np.log10(thresh_p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(batches, nlog10_p, label='-log10(p-value)', linewidth=2)
    
    # Horizontal lines
    plt.axhline(y=thresh_nlog10, color='r', linestyle='--', alpha=0.7, label=f'Marginal Sig (0.05)')
    plt.axhline(y=ref_nlog10, color='g', linestyle='--', alpha=0.7, label=f'{label_text} P-value')
    
    plt.title(f"Significance Trajectory: {method_name}")
    plt.xlabel("Batches")
    plt.ylabel("-log10(p-value)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = os.path.join(output_dir, f"{safe_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


def plot_mse_comparison(mse_histories, output_dir, baseline_val=None):
    """
    Plots MSE trajectories for module outputs.
    Calculates Mean and Std Error across the X target genes for the shaded region.
    """
    if not mse_histories:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Convert to arrays and extract max means for broken axis logic
    max_values = []
    hist_arrays = {}
    for name, history in mse_histories.items():
        if not history: continue
        arr = np.array(history) # Shape: (n_batches, X_targets)
        hist_arrays[name] = arr
        # Find the max of the mean trajectory
        max_values.append(np.max(np.mean(arr, axis=1)))
        
    if not max_values: 
        return
        
    global_max = max(max_values)
    median_max = np.median(max_values)
    
    # Threshold for broken axis
    use_broken_axis = global_max > (5.0 * median_max)

    if use_broken_axis:
        # --- BROKEN AXIS PLOT ---
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.1) 

        for name, arr in hist_arrays.items():
            batches = range(1, arr.shape[0] + 1)
            mean_vals = np.mean(arr, axis=1)
            std_vals = np.std(arr, axis=1)
            
            # Plot line and capture color
            p = ax1.plot(batches, mean_vals, label=name, linewidth=2, alpha=0.8)
            color = p[0].get_color()
            
            # Shaded Std Dev
            ax1.fill_between(batches, np.maximum(0, mean_vals - std_vals), mean_vals + std_vals, color=color, alpha=0.2)
            
            ax2.plot(batches, mean_vals, label=name, linewidth=2, alpha=0.8, color=color)
            ax2.fill_between(batches, np.maximum(0, mean_vals - std_vals), mean_vals + std_vals, color=color, alpha=0.2)

        # Y-lims
        ax1.set_ylim(median_max * 1.5, global_max * 1.1)
        ax2.set_ylim(0, median_max * 1.2)

        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False) 
        ax2.xaxis.tick_bottom()

        # Diagonal break lines
        d = .5 
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        if baseline_val is not None:
            b_val = np.mean(baseline_val) # Safely handle if it's an array
            ax1.axhline(y=b_val, color='k', linestyle='--', alpha=0.6, label='Zero Baseline')
            ax2.axhline(y=b_val, color='k', linestyle='--', alpha=0.6, label='Zero Baseline')

        ax1.set_title("Module Imputation Error Trajectory (Mean MSE \u00b1 Std)")
        ax2.set_ylabel("Mean Squared Error")
        ax2.set_xlabel("Batches")
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
    else:
        # --- STANDARD PLOT ---
        plt.figure(figsize=(10, 6))
        for name, arr in hist_arrays.items():
            batches = range(1, arr.shape[0] + 1)
            mean_vals = np.mean(arr, axis=1)
            std_vals = np.std(arr, axis=1)
            
            p = plt.plot(batches, mean_vals, label=name, linewidth=2, alpha=0.8)
            color = p[0].get_color()
            plt.fill_between(batches, np.maximum(0, mean_vals - std_vals), mean_vals + std_vals, color=color, alpha=0.2)
        
        if baseline_val is not None:
            b_val = np.mean(baseline_val)
            plt.axhline(y=b_val, color='k', linestyle='--', alpha=0.6, label='Zero Baseline')

        plt.title("Module Imputation Error Trajectory (Mean MSE \u00b1 Std)")
        plt.xlabel("Batches")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "MSE_Comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved MSE comparison plot to {out_path}")


def plot_beta_corr_comparison(corr_histories, output_dir):
    """
    Plots Pearson Correlation trajectories (True vs Imputed Beta) for module outputs.
    Calculates Mean and Std Error across the X target genes.
    """
    if not corr_histories:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    for name, history in corr_histories.items():
        if not history: continue
        arr = np.array(history) # Shape: (n_batches, X_targets)
        batches = range(1, arr.shape[0] + 1)
        
        mean_vals = np.mean(arr, axis=1)
        std_vals = np.std(arr, axis=1)
        
        p = plt.plot(batches, mean_vals, label=name, linewidth=2, alpha=0.8)
        color = p[0].get_color()
        plt.fill_between(batches, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
    
    plt.title("Module Beta Reconstruction Accuracy (Mean Pearson r \u00b1 Std)")
    plt.xlabel("Batches")
    plt.ylabel("Pearson Correlation (True vs Imputed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.2, 1.05) 

    out_path = os.path.join(output_dir, "BetaCorr_Comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved Beta Corr comparison plot to {out_path}")


def compute_control_covariance(h5ad_path, target_genes, obs_label, control_val, agg_method="mean"):
    """
    Loads an AnnData file, filters for control cells, and calculates the covariance 
    between every gene and the module of 'target_genes'.
    Aggregates the absolute covariance across the module using 'agg_method' (mean or max).
    Returns a pandas Series mapping gene_name -> aggregated_score.
    """
    if not os.path.exists(h5ad_path):
        print(f"Warning: Control H5AD not found at {h5ad_path}")
        return None

    print(f"Loading control data from {h5ad_path}...")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Error loading H5AD: {e}")
        return None

    # 1. Filter for control cells
    if control_val is not None and str(control_val).strip() != "":
        if obs_label in adata.obs.columns:
            c_val_lower = str(control_val).lower()
            mask = adata.obs[obs_label].astype(str).str.lower() == c_val_lower
            
            n_total = adata.n_obs
            adata = adata[mask].copy()
            print(f"Filtered control cells: {adata.n_obs} / {n_total} cells (label '{obs_label}' ~= '{control_val}')")
        else:
            print(f"Warning: obs column '{obs_label}' not found. Using all {adata.n_obs} cells.")
    else:
        print(f"Control label is blank. Using all {adata.n_obs} cells as controls.")

    if adata.n_obs < 5:
        print("Error: Too few control cells to compute covariance.")
        return None

    # 2. Check for target genes
    valid_targets = [g for g in target_genes if g in adata.var_names]
    if not valid_targets:
        print("Error: None of the target genes were found in H5AD var_names.")
        return None
        
    print(f"Computing covariance for {len(valid_targets)} module genes (Aggregation: {agg_method})...")

    # 3. Compute Covariance Matrix
    target_indices = [adata.var_names.get_loc(g) for g in valid_targets]
    X = adata.X
    
    if sp.issparse(X):
        try:
            X = X.toarray() 
        except MemoryError:
            print("Error: Control matrix too large to densify for covariance calc.")
            return None
        
    # Get target columns and center them (N_cells x X_targets)
    Y_mat = X[:, target_indices]
    Y_centered = Y_mat - np.mean(Y_mat, axis=0)
    
    # Center all genes (N_cells x N_genes)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean[None, :]
    
    # Calculate Covariance Matrix: (N_genes x X_targets)
    N = adata.n_obs
    covariances = np.dot(X_centered.T, Y_centered) / (N - 1)
    
    # 4. Aggregate Absolute Covariances
    abs_cov = np.abs(covariances)
    
    if agg_method == "max":
        agg_scores = np.max(abs_cov, axis=1)
    else:
        agg_scores = np.mean(abs_cov, axis=1)
    
    # Return as Series
    return pd.Series(agg_scores, index=adata.var_names)


def compute_external_perturbation_effect(h5ad_path, target_genes, obs_label, control_val):
    """
    Loads an external H5AD (pseudobulk or single-cell), finds the control population,
    and calculates the Mean Absolute Difference in module expression between 
    each perturbation and the control.
    Returns: pd.Series mapping perturbation_name -> aggregated_effect_size
    """
    if not os.path.exists(h5ad_path):
        print(f"Warning: External H5AD not found at {h5ad_path}")
        return None

    print(f"Loading external perturbation data from {h5ad_path}...")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Error loading H5AD: {e}")
        return None

    # 1. Verify Columns
    if obs_label not in adata.obs.columns:
        print(f"Error: Target label column '{obs_label}' not found in external H5AD.")
        return None
    
    valid_targets = [g for g in target_genes if g in adata.var_names]
    if not valid_targets:
        print(f"Error: None of the target genes found in external H5AD.")
        return None

    # 2. Extract Data for Target Genes
    target_indices = [adata.var_names.get_loc(g) for g in valid_targets]
    Y_mat = adata.X[:, target_indices]
    
    # Densify if sparse
    if sp.issparse(Y_mat):
        Y_mat = Y_mat.toarray()
    else:
        Y_mat = np.asarray(Y_mat)

    # 3. Identify Control Means
    obs_vals = adata.obs[obs_label].astype(str)
    is_ctrl = obs_vals == str(control_val)
    if not is_ctrl.any():
        is_ctrl = obs_vals.str.lower() == str(control_val).lower()
    
    if not is_ctrl.any():
        print(f"Error: No control cells found with label '{control_val}' in column '{obs_label}'.")
        return None
    
    # Vector of control means (length X)
    ctrl_means = np.mean(Y_mat[is_ctrl], axis=0)
    print(f"External Dataset: Found {is_ctrl.sum()} control observations for {len(valid_targets)} module genes.")

    # 4. Compute Means per Perturbation
    # Create a DataFrame of the perturbation cells
    df_pert = pd.DataFrame(Y_mat[~is_ctrl], columns=valid_targets)
    df_pert['pert'] = obs_vals[~is_ctrl].values
    
    # Group by perturbation and calculate mean across all cells for that perturbation
    # Result is a Matrix: (N_perturbations x X_targets)
    pert_means = df_pert.groupby('pert').mean()
    
    # 5. Calculate Aggregated Absolute Delta
    # Delta Matrix
    deltas = pert_means - ctrl_means
    
    # Aggregate magnitude (Mean Absolute Delta across the X genes)
    abs_deltas = deltas.abs().mean(axis=1)
    
    return abs_deltas


def get_stratified_indices(scores, n_bins=10, seed=42):
    """
    Returns indices sorted such that traversing them samples uniformly 
    across the POPULATION PERCENTILES of 'scores'.
    
    1. Bins scores into equal-frequency intervals (pd.qcut).
       - Top Bin = Top 10% of genes (not just top 10% of value range).
    2. Shuffles indices within each bin.
    3. Interleaves the bins (Round Robin) to create the final queue.
    """
    try:
        # Use qcut for Quantile binning (Equal Frequency)
        # duplicates='drop' merges bins if many genes have identical scores (e.g. 0.0)
        bins = pd.qcut(scores, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Fallback to cut (Equal Width) if qcut fails (e.g. all scores identical)
        try:
            bins = pd.cut(scores, bins=n_bins, labels=False, duplicates='drop')
        except ValueError:
            return scores.index.values
    
    # Determine number of actual bins created (might be < n_bins due to duplicates)
    if hasattr(bins, 'categories'):
        n_actual = len(bins.categories)
    else:
        # If labels=False, bins are integers. max() + 1 gives count.
        n_actual = int(bins.max()) + 1
    
    # Group indices by bin
    bin_indices = [[] for _ in range(n_actual)]
    rng = np.random.default_rng(seed)
    
    # This iteration might be slow for huge DFs, but fast for 20k.
    # A vectorized way:
    df_temp = pd.DataFrame({'score': scores, 'bin': bins})
    
    # Group and shuffle
    # We explicitly iterate 0..n_actual-1 to ensure order (lowest bin to highest bin)
    # Note: qcut assigns 0 to the lowest scores and N to highest.
    # If we want to sample uniformly, order doesn't matter much, 
    # but usually we want to cycle Low -> Med -> High -> Low...
    
    rng = np.random.default_rng(seed)
    queues = []
    max_len = 0
    
    for b in range(n_actual):
        # Extract indices belonging to this bin
        indices = df_temp[df_temp['bin'] == b].index.values
        if len(indices) > 0:
            rng.shuffle(indices)
            queues.append(list(indices))
            if len(indices) > max_len:
                max_len = len(indices)
    
    # Interleave (Round Robin)
    stratified_order = []
    for i in range(max_len):
        for q in queues:
            if i < len(q):
                stratified_order.append(q[i])
                
    return np.array(stratified_order, dtype=int)


def main():
    args = parse_arguments()
    
    if not os.path.exists(args.lof) or not os.path.exists(args.limma):
        sys.exit("Error: LoF or Limma file not found.")
        
    # --- 1. Load LoF Data (Gamma) ---
    print(f"Reading LoF data from {args.lof}...")
    try:
        df_lof = pd.read_csv(args.lof, sep='\t')
        if "ensg" not in df_lof.columns or "post_mean" not in df_lof.columns:
             sys.exit("Error: LoF file missing 'ensg' or 'post_mean' columns.")
        df_lof = df_lof.set_index("ensg")
        s_lof = df_lof["post_mean"].rename("LoF_gamma")
    except Exception as e:
        sys.exit(f"Error reading LoF file: {e}")

    # --- 2. Load Limma Data (Matrix) ---
    print(f"Reading Limma data from {args.limma}...")
    try:
        df_limma = pd.read_csv(args.limma, sep='\t', index_col=0)
    except Exception as e:
        sys.exit(f"Error reading Limma file: {e}")

    # --- 3. Build Global Map & Normalize Indices ---
    print("Building Gene Map & Normalizing Indices...")
    all_indices = set(s_lof.index) | set(df_limma.index)
    gene_map = get_gene_mapping(list(all_indices), args.mapping_file)

    s_lof = process_series(s_lof, "LoF Data", gene_map)
    
    df_limma_mapped = df_limma.copy()
    df_limma_mapped.index = df_limma_mapped.index.map(lambda x: gene_map.get(x, x))
    if df_limma_mapped.index.duplicated().any():
        df_limma_mapped = df_limma_mapped.groupby(df_limma_mapped.index).mean()

    # Map Limma Columns (Perturbations) to match LoF index
    if any(str(x).startswith("ENSG") for x in df_limma_mapped.columns[:5]):
         new_cols = df_limma_mapped.columns.map(lambda x: gene_map.get(x, x))
         df_limma_mapped.columns = new_cols
         if df_limma_mapped.columns.duplicated().any():
             df_limma_mapped = df_limma_mapped.groupby(df_limma_mapped.columns, axis=1).mean()

    # --- 4. Load Module Genes ---
    print(f"Loading module definitions from {args.module_file}...")
    try:
        df_modules = pd.read_csv(args.module_file, sep='\t')
        # Cast both to string to avoid int vs str mismatch
        module_genes_df = df_modules[df_modules['Factor'].astype(str) == str(args.module_name)]
        if module_genes_df.empty:
            sys.exit(f"Error: Module '{args.module_name}' not found in {args.module_file}")
        
        target_symbols = module_genes_df['Symbol'].tolist()
        available_targets = [g for g in target_symbols if g in df_limma_mapped.index]
        print(f"Found {len(available_targets)}/{len(target_symbols)} module genes in Limma matrix.")
        
        if len(available_targets) == 0:
            sys.exit("Error: No module genes found in Limma matrix.")
    except Exception as e:
        sys.exit(f"Error processing module file: {e}")

    # --- 5. Extract Betas & Merge ---
    # df will hold the 'universe' of perturbations
    df = pd.DataFrame(s_lof)
    target_cols = []
    
    for target in available_targets:
        s_beta = df_limma_mapped.loc[target]
        col_name = f"{target}_beta"
        s_beta = s_beta.rename(col_name)
        # Inner join ensures we only keep perturbations with BOTH LoF and Beta data
        df = df.join(s_beta, how='inner')
        target_cols.append(col_name)

    df.index.name = "gene_name"
    df = df.reset_index()
    total_genes = len(df)

    print(f"Valid perturbations for simulation: {total_genes}")
    print(f"Batch size: {args.batch_size}")
    
    if total_genes < 3:
        sys.exit("Not enough genes to run simulation.")

    # Matrix of true target values (N, X)
    y_true_matrix = df[target_cols].values

    # Dictionary to store MSE histories for comparison plot
    all_mse_histories = {}
    all_r_beta_histories = {}

    # Store diagnostics for final table
    all_diagnostics = {}

    # ---------------------------------------------------------
    # Strategy 1: GammaMagnitude Sampling
    # ---------------------------------------------------------
    mag_scores = df['LoF_gamma'].abs()
    
    if args.sampling_strategy == "uniform":
        mag_indices = get_stratified_indices(mag_scores, n_bins=10, seed=args.seed)
        strat_name_mag = "GammaMagnitude (Uniform)"
    else:
        mag_indices = mag_scores.sort_values(ascending=False).index.values
        strat_name_mag = "GammaMagnitude (Strongest)"
    
    strat_mag = StaticStrategy(total_genes, args, mag_indices, name=strat_name_mag)
    
    mag_mse, mag_r_beta, mag_diag = run_simulation_strategy(
        strat_mag, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
    )
    all_mse_histories["GammaMagnitude"] = mag_mse
    all_r_beta_histories["GammaMagnitude"] = mag_r_beta
    all_diagnostics["GammaMagnitude"] = mag_diag

    # ---------------------------------------------------------
    # Strategy 2: Random Sampling
    # ---------------------------------------------------------
    rnd_indices = df.sample(frac=1, random_state=args.seed).index.values
    
    strat_rnd = StaticStrategy(total_genes, args, rnd_indices, name="Random Sampling")
    
    rnd_mse, rnd_r_beta, rnd_diag = run_simulation_strategy(
        strat_rnd, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
    )
    all_mse_histories["Random"] = rnd_mse
    all_r_beta_histories["Random"] = rnd_r_beta
    all_diagnostics["Random"] = rnd_diag

    # ---------------------------------------------------------
    # Strategy 3: Control Covariance (Optional)
    # ---------------------------------------------------------
    cov_indices = None 
    
    if args.control_h5ad:
        cov_series = compute_control_covariance(
            args.control_h5ad, available_targets, args.target_label, args.control_label, args.module_cov_agg
        )

        if cov_series is not None:
            cov_aligned = df['gene_name'].map(cov_series).fillna(0.0)
            cov_scores = cov_aligned.abs()
            
            if args.sampling_strategy == "uniform":
                cov_indices = get_stratified_indices(cov_scores, n_bins=10, seed=args.seed)
                strat_name_cov = "Control Covariance (Uniform)"
            else:
                cov_indices = cov_scores.sort_values(ascending=False).index.values
                strat_name_cov = "Control Covariance (Strongest)"
            
            strat_cov = StaticStrategy(total_genes, args, cov_indices, name=strat_name_cov)

            cov_mse, cov_r_beta, cov_diag = run_simulation_strategy(
                strat_cov, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
            )
            all_mse_histories["ControlCovariance"] = cov_mse
            all_r_beta_histories["ControlCovariance"] = cov_r_beta
            all_diagnostics["ControlCovariance"] = cov_diag

    # ---------------------------------------------------------
    # Strategy 4: External Perturbation Effect (Optional)
    # ---------------------------------------------------------
    if args.external_h5ad:
        ext_series = compute_external_perturbation_effect(
            args.external_h5ad, available_targets, args.target_label, args.control_label
        )

        if ext_series is not None:
            ext_aligned = df['gene_name'].map(ext_series).fillna(0.0)
            ext_scores = ext_aligned.abs()
            
            if args.sampling_strategy == "uniform":
                ext_indices = get_stratified_indices(ext_scores, n_bins=10, seed=args.seed)
                strat_name_ext = "External Perturbation (Uniform)"
            else:
                ext_indices = ext_scores.sort_values(ascending=False).index.values
                strat_name_ext = "External Perturbation (Strongest)"
            
            strat_ext = StaticStrategy(total_genes, args, ext_indices, name=strat_name_ext)

            ext_mse, ext_r_beta, ext_diag = run_simulation_strategy(
                strat_ext, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
            )
            all_mse_histories["ExternalPerturbation"] = ext_mse
            all_r_beta_histories["ExternalPerturbation"] = ext_r_beta
            all_diagnostics["ExternalPerturbation"] = ext_diag

    # ---------------------------------------------------------
    # Shared GP Initialization (Run once if any GP strategy is needed)
    # ---------------------------------------------------------
    shared_learner = None
    has_external_data = (args.external_list or args.external_h5ad or args.embeddings_yaml)
    
    if not args.static_only:
        if ActiveGPLearner is not None and has_external_data:
            print("\nInitializing Shared Active GP Learner...")
            shared_learner = ActiveGPLearner(df['gene_name'].values, args)

    # ---------------------------------------------------------
    # Strategy 5: GP Imputation (Covariance/Random + GP Prediction)
    # ---------------------------------------------------------
    gp_strat_name = "GP Imputation"
    mse_key = None
    
    if shared_learner is not None and not args.static_only:
        if cov_indices is not None:
            gp_indices = cov_indices
            gp_strat_name = "Control Covariance + GP Imputation"
            mse_key = "ControlCovariance_GP"
        else:
            print("Warning: Control Covariance not available. Falling back to Random Sampling for GP.")
            gp_indices = rnd_indices
            gp_strat_name = "Random Sampling + GP Imputation"
            mse_key = "Random_GP"
            
        shared_learner.reset()
        strat_gp = StaticGPStrategy(total_genes, args, gp_indices, shared_learner, name=gp_strat_name)

        gp_mse, gp_r_beta, gp_diag = run_simulation_strategy(
            strat_gp, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
        )
        
        all_mse_histories[mse_key] = gp_mse
        all_r_beta_histories[mse_key] = gp_r_beta
        all_diagnostics[mse_key] = gp_diag

    # ---------------------------------------------------------
    # Strategy 6: Active High Leverage (Optional)
    # ---------------------------------------------------------
    if args.run_active_leverage and not args.static_only:
        if shared_learner is None:
            print("\nError: Active High Leverage requires ActiveGPLearner and external data/embeddings.")
        else:
            print("\nRunning Active High Leverage Strategy...")
            shared_learner.reset()
            
            strat_lev = HighLeverageStrategy(total_genes, args, shared_learner, prior_indices=cov_indices)
            
            lev_mse, lev_r_beta, lev_diag = run_simulation_strategy(
                strat_lev, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
            )
            
            all_mse_histories["Active_HighLeverage"] = lev_mse
            all_r_beta_histories["Active_HighLeverage"] = lev_r_beta
            all_diagnostics["Active_HighLeverage"] = lev_diag

    # ---------------------------------------------------------
    # Strategy 7: Active Uncertainty (Optional)
    # ---------------------------------------------------------
    if args.run_active_uncertainty and not args.static_only:
        if shared_learner is None:
            print("\nError: Active Uncertainty requires ActiveGPLearner and external data.")
        else:
            print("\nRunning Active Uncertainty Strategy...")
            shared_learner.reset()
            
            strat_unc = UncertaintyStrategy(total_genes, args, shared_learner, prior_indices=None)

            unc_mse, unc_r_beta, unc_diag = run_simulation_strategy(
                strat_unc, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
            )
            
            all_mse_histories["Active_Uncertainty"] = unc_mse
            all_r_beta_histories["Active_Uncertainty"] = unc_r_beta
            all_diagnostics["Active_Uncertainty"] = unc_diag

    # ---------------------------------------------------------
    # Strategy 8: Active Diversity (Optional)
    # ---------------------------------------------------------
    if args.run_active_diversity and not args.static_only:
        if shared_learner is None:
            print("\nError: Active Diversity requires ActiveGPLearner and external data.")
        else:
            print("\nRunning Active Diversity Strategy...")
            shared_learner.reset()
            
            strat_div = DiversityStrategy(total_genes, args, shared_learner, prior_indices=cov_indices)
            
            div_mse, div_r_beta, div_diag = run_simulation_strategy(
                strat_div, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
            )
            
            all_mse_histories["Active_Diversity"] = div_mse
            all_r_beta_histories["Active_Diversity"] = div_r_beta
            all_diagnostics["Active_Diversity"] = div_diag

    # ---------------------------------------------------------
    # Strategy 9: Active PC-Uncertainty (Optional)
    # ---------------------------------------------------------
    if args.run_active_pca and not args.static_only:
        if shared_learner is None:
            print("\nError: Active PCA requires ActiveGPLearner and external data.")
        else:
            print("\nRunning Active PC-Uncertainty Strategy...")
            shared_learner.reset()
            
            strat_pca = PCUncertaintyStrategy(total_genes, args, shared_learner, prior_indices=cov_indices)
            
            pca_mse, pca_r_beta, pca_diag = run_simulation_strategy(
                strat_pca, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
            )
            
            all_mse_histories["Active_PCUncertainty"] = pca_mse
            all_r_beta_histories["Active_PCUncertainty"] = pca_r_beta
            all_diagnostics["Active_PCUncertainty"] = pca_diag

    # ---------------------------------------------------------
    # Strategy 10: Active Variance Reduction (Kriging Believer)
    # ---------------------------------------------------------
    if args.run_active_var_reduction and not args.static_only:
        if shared_learner is None:
            print("\nError: Active Variance Reduction requires ActiveGPLearner and external data.")
        else:
            print("\nRunning Active Stepwise Variance Reduction (Kriging Believer)...")
            shared_learner.reset()
            
            strat_svr = VarianceReductionStrategy(total_genes, args, shared_learner, prior_indices=cov_indices)
            
            svr_mse, svr_r_beta, svr_diag = run_simulation_strategy(
                strat_svr, y_true_matrix, total_genes, args.batch_size, args.max_batches, args.print_every
            )
            
            all_mse_histories["Active_VarReduction"] = svr_mse
            all_r_beta_histories["Active_VarReduction"] = svr_r_beta
            all_diagnostics["Active_VarReduction"] = svr_diag

    # ---------------------------------------------------------
    # Final Comparative Plots
    # ---------------------------------------------------------
    if all_mse_histories:
        baseline_mse = np.mean(y_true_matrix**2, axis=0)
        plot_mse_comparison(all_mse_histories, args.output_dir, baseline_val=baseline_mse)

    if all_r_beta_histories:
        plot_beta_corr_comparison(all_r_beta_histories, args.output_dir)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL SUMMARY: Beta Reconstruction Accuracy (Final Batch)")
    print("="*80)
    print(f"{'Strategy':<40} | {'Mean MSE':<15} | {'Mean Pearson r':<15}")
    print("-" * 80)

    def print_res(name, key):
        if key in all_mse_histories:
            final_mse = np.mean(all_mse_histories[key][-1])
            final_r = np.mean(all_r_beta_histories[key][-1])
            print(f"{name:<40} | {final_mse:.4e}      | {final_r:+.4f}")

    print_res(strat_name_mag, "GammaMagnitude")
    print_res("Random Sampling", "Random")
    if "ControlCovariance" in all_mse_histories:
        print_res(strat_name_cov, "ControlCovariance")
    if "ExternalPerturbation" in all_mse_histories:
        print_res(strat_name_ext, "ExternalPerturbation")
    if mse_key and mse_key in all_mse_histories:
        print_res(gp_strat_name, mse_key)
    print_res("Active High Leverage", "Active_HighLeverage")
    print_res("Active Uncertainty", "Active_Uncertainty")
    print_res("Active Diversity", "Active_Diversity")
    print_res("Active PC-Uncertainty", "Active_PCUncertainty")
    print_res("Active Var-Reduction", "Active_VarReduction")
    print("-" * 80)

    print("\n" + "="*80)
    print("DIAGNOSTIC REPORT: Average Quality of Selected Genes")
    print("="*80)
    print(f"{'Strategy':<40} | {'Avg |Target|':<12} | {'Avg Connectivity':<16}")
    print("-" * 80)
    
    def print_diag(name, key):
        if key in all_diagnostics:
            d = all_diagnostics[key]
            conn_str = f"{d['conn']:.4f}" if d['conn'] > 0 else "N/A"
            print(f"{name:<40} | {d.get('target_mag', 0.0):.4f}       | {conn_str:<16}")

    print_diag(strat_name_mag, "GammaMagnitude")
    print_diag("Random Sampling", "Random")
    if "ControlCovariance" in all_diagnostics:
        print_diag(strat_name_cov, "ControlCovariance")
    if "ExternalPerturbation" in all_diagnostics:
        print_diag(strat_name_ext, "ExternalPerturbation")
    if mse_key and mse_key in all_diagnostics:
        print_diag(gp_strat_name, mse_key)
    print_diag("Active High Leverage", "Active_HighLeverage")
    print_diag("Active Uncertainty", "Active_Uncertainty")
    print_diag("Active Diversity", "Active_Diversity")
    print_diag("Active PC-Uncertainty", "Active_PCUncertainty")
    print_diag("Active Var-Reduction", "Active_VarReduction")
    print("-" * 80)

if __name__ == "__main__":
    main()
