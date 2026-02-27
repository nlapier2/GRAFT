#!/usr/bin/env python3
"""
Evaluates Kernel Alignment (CKA) between external H5AD datasets and Limma ground truth.
Compares "Global" kernels (using all genes) vs "Module-Specific" kernels (using only target genes).
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sp

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Kernel Alignment (Global vs Module).")
    parser.add_argument("--limma", type=str, required=True, help="Path to Limma perturbation TSV (Ground Truth).")
    parser.add_argument("--module_file", type=str, required=True, help="Path to NMF modules TSV.")
    parser.add_argument("--external_list", type=str, required=True, help="Text file with paths to external H5ADs.")
    parser.add_argument("--mapping_file", type=str, default=None, help="Optional TSV (col1=ENSG, col2=Symbol) for gene mapping.")
    parser.add_argument("--target_label", type=str, default="target_gene", help="Obs column name for perturbations.")
    parser.add_argument("--control_label", type=str, default="non-targeting", help="Value identifying control cells.")
    parser.add_argument("--max_modules", type=int, default=0, help="Maximum number of modules to evaluate (0 to process all).")
    return parser.parse_args()

def get_gene_mapping(ids_to_map, mapping_file=None):
    mapping = {}
    ensembl_ids = [x for x in ids_to_map if str(x).startswith("ENSG")]
    if not ensembl_ids: return mapping

    if mapping_file and os.path.exists(mapping_file):
        map_df = pd.read_csv(mapping_file, sep='\t', header=0 if pd.read_csv(mapping_file, sep='\t', nrows=1).shape[1] > 1 else None)
        mapping = pd.Series(map_df.iloc[:, 1].values, index=map_df.iloc[:, 0].values).to_dict()
    else:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
            for res in results:
                if res.get('query') and res.get('symbol'):
                    mapping[res.get('query')] = res.get('symbol')
        except:
            pass
    return mapping

def center_kernel(K):
    """Centers a kernel matrix in feature space."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def compute_cka(K1, K2):
    """Computes Centered Kernel Alignment (CKA) between two kernel matrices."""
    K1_c = center_kernel(K1)
    K2_c = center_kernel(K2)
    
    # Frobenius inner product
    inner_prod = np.sum(K1_c * K2_c)
    norm1 = np.sqrt(np.sum(K1_c * K1_c))
    norm2 = np.sqrt(np.sum(K2_c * K2_c))
    
    if norm1 == 0 or norm2 == 0: return 0.0
    return inner_prod / (norm1 * norm2)

def build_correlation_kernel(df_deltas):
    """Builds a row-wise cosine/correlation kernel from a DataFrame of deltas."""
    M = df_deltas.values
    # Row-center
    M = M - M.mean(axis=1, keepdims=True)
    # L2 Normalize
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M = np.divide(M, norms, where=norms > 1e-9)
    # Kernel
    K = M @ M.T
    np.fill_diagonal(K, 1.0)
    return pd.DataFrame(K, index=df_deltas.index, columns=df_deltas.index)

def extract_h5ad_deltas(h5ad_path, target_label, control_val, gene_map):
    """Extracts perturbation deltas (Pert - Control) from an H5AD."""
    try:
        adata = ad.read_h5ad(h5ad_path)
    except:
        return None

    if target_label not in adata.obs.columns: return None
    
    obs_vals = adata.obs[target_label].astype(str)
    is_ctrl = (obs_vals == str(control_val)) | (obs_vals.str.lower() == str(control_val).lower())
    
    if not is_ctrl.any(): return None
    
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    ctrl_mean = np.mean(X[is_ctrl], axis=0)
    
    # Map var_names to symbols
    symbols = [gene_map.get(g, g) for g in adata.var_names]
    
    df_pert = pd.DataFrame(X[~is_ctrl], columns=symbols)
    df_pert['pert'] = obs_vals[~is_ctrl].values
    
    # Handle duplicate gene symbols (take mean)
    df_pert = df_pert.loc[:, ~df_pert.columns.duplicated(keep='first')]
    
    pert_means = df_pert.groupby('pert').mean()
    deltas = pert_means - ctrl_mean
    return deltas

def main():
    args = parse_arguments()

    # 1. Load Limma & Get Universal Gene Map
    print("Loading Limma (Target) matrix...")
    df_limma = pd.read_csv(args.limma, sep='\t', index_col=0)
    all_genes = set(df_limma.index)
    gene_map = get_gene_mapping(list(all_genes), args.mapping_file)
    
    df_limma.index = df_limma.index.map(lambda x: gene_map.get(x, x))
    if df_limma.index.duplicated().any():
        df_limma = df_limma.groupby(df_limma.index).mean()
        
    limma_perts = set(df_limma.columns)

    # 2. Load H5ADs & Compute Deltas
    print("\nLoading external H5ADs...")
    with open(args.external_list, 'r') as f:
        h5ad_paths = [line.strip() for line in f if line.strip()]

    h5ad_deltas = {}
    valid_perts = set(limma_perts)

    for path in h5ad_paths:
        name = os.path.basename(path)
        print(f"  Processing {name}...")
        deltas = extract_h5ad_deltas(path, args.target_label, args.control_label, gene_map)
        if deltas is not None:
            h5ad_deltas[name] = deltas
            valid_perts = valid_perts.intersection(set(deltas.index))
            
    valid_perts = sorted(list(valid_perts))
    print(f"\nFound {len(valid_perts)} overlapping perturbations across Limma and all {len(h5ad_deltas)} H5ADs.")
    
    if len(valid_perts) < 3:
        sys.exit("Not enough overlapping perturbations to compute CKA.")

    # 3. Load Modules
    df_modules = pd.read_csv(args.module_file, sep='\t')
    modules = df_modules.groupby('Factor')

    # --- OPTIMIZATION: Precompute Global Kernels ---
    print("\nPrecomputing Global Kernels...")
    precomputed_global = {}
    for name, deltas in h5ad_deltas.items():
        D = deltas.loc[valid_perts]
        K_global = build_correlation_kernel(D).values
        K_global_c = center_kernel(K_global)
        norm_global = np.sqrt(np.sum(K_global_c * K_global_c))
        
        precomputed_global[name] = {
            'D': D,              # Save sliced dataframe for fast module extraction later
            'K_c': K_global_c,   # Centered kernel
            'norm': norm_global  # Frobenius norm
        }

    # 4. Evaluation Loop
    results = []
    print("\nEvaluating Modules...")
    
    processed_count = 0
    
    for factor, group in modules:
        if args.max_modules > 0 and processed_count >= args.max_modules:
            print(f"Reached max_modules limit ({args.max_modules}). Stopping evaluation.")
            break

        module_genes = [gene_map.get(g, g) for g in group['Ensembl_ID'].tolist()]
        valid_module_genes = [g for g in module_genes if g in df_limma.index]
        
        if not valid_module_genes:
            continue
            
        print(f"  -> Processing Module {factor} ({len(valid_module_genes)} valid genes)...")
        processed_count += 1
            
        # Target Kernel (Ground Truth)
        # Computed and centered ONCE per module
        Y_target = df_limma.loc[valid_module_genes, valid_perts].T.values
        K_target = Y_target @ Y_target.T
        K_target_c = center_kernel(K_target)
        norm_target = np.sqrt(np.sum(K_target_c * K_target_c))
        
        if norm_target == 0: 
            continue
        
        for name, global_data in precomputed_global.items():
            D = global_data['D']
            K_g_c = global_data['K_c']
            norm_g = global_data['norm']
            
            # Global CKA (Fast inner product, no matrix multiplication needed!)
            if norm_g > 0:
                cka_global = np.sum(K_g_c * K_target_c) / (norm_g * norm_target)
            else:
                cka_global = 0.0
            
            # Module Kernel (Still needs to be computed because the feature set changes)
            h5ad_module_genes = [g for g in valid_module_genes if g in D.columns]
            if len(h5ad_module_genes) > 1:
                D_mod = D[h5ad_module_genes]
                K_mod = build_correlation_kernel(D_mod).values
                K_mod_c = center_kernel(K_mod)
                norm_mod = np.sqrt(np.sum(K_mod_c * K_mod_c))
                
                if norm_mod > 0:
                    cka_mod = np.sum(K_mod_c * K_target_c) / (norm_mod * norm_target)
                else:
                    cka_mod = 0.0
            else:
                cka_mod = 0.0
                
            results.append({
                'Factor': factor,
                'Module_Size': len(valid_module_genes),
                'H5AD': name,
                'CKA_Global': cka_global,
                'CKA_Module': cka_mod,
                'Winner': 'Module' if cka_mod > cka_global else 'Global'
            })

    # 5. Report Generation
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*85)
    print("KERNEL RELEVANCE REPORT: Global vs. Module-Specific Kernels")
    print("="*85)
    
    global_wins = 0
    module_wins = 0

    for factor in df_res['Factor'].unique():
        sub = df_res[df_res['Factor'] == factor]
        m_size = sub['Module_Size'].iloc[0]
        
        print(f"\n--- Module: {factor} ({m_size} genes) ---")
        print(f"{'H5AD Dataset':<35} | {'Global CKA':<12} | {'Module CKA':<12} | {'Advantage'}")
        print("-" * 80)
        
        for _, row in sub.iterrows():
            adv = row['CKA_Module'] - row['CKA_Global']
            winner_str = f"Module (+{adv:.4f})" if adv > 0 else f"Global ({adv:.4f})"
            print(f"{row['H5AD']:<35} | {row['CKA_Global']:.4f}       | {row['CKA_Module']:.4f}       | {winner_str}")
            
            if adv > 0: module_wins += 1
            else: global_wins += 1
            
        mean_g = sub['CKA_Global'].mean()
        std_g = sub['CKA_Global'].std()
        mean_m = sub['CKA_Module'].mean()
        std_m = sub['CKA_Module'].std()
        
        print("-" * 80)
        print(f"{'MEAN ± STD':<35} | {mean_g:.4f}±{std_g:.4f} | {mean_m:.4f}±{std_m:.4f} |")

    print("\n" + "="*85)
    print("FINAL SUMMARY")
    print("="*85)
    total = global_wins + module_wins
    print(f"Total Comparisons: {total}")
    print(f"Global Kernel Best:  {global_wins} times ({(global_wins/total)*100:.1f}%)")
    print(f"Module Kernel Best:  {module_wins} times ({(module_wins/total)*100:.1f}%)")
    print("="*85)

if __name__ == "__main__":
    main()
