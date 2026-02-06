#!/usr/bin/env python3
"""
Utility script to create a "Wide" simulation dataset.

Input:
- LoF Burden File: Source of 'LoF_gamma' (Disease Association).
- Limma Beta Matrix: Source of '_beta' columns (Regulatory Effects).

Features:
- "Smart Selection": Only selects targets that actually exist in the Beta file.
- "Correlation Summary": Prints a report of Gamma-Beta correlations (Core Gene strength).

Output:
- A single TSV with columns: [gene_name, LoF_gamma, HBA1_beta, GATA1_beta, ...]
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

# Constants
HBA1_ENSG = "ENSG00000206172"
HBA1_SYMBOL = "HBA1"

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
        print(f"[info] Loading gene mapping from {mapping_file}...", file=sys.stderr)
        try:
            map_df = pd.read_csv(mapping_file, sep='\t', header=0)
            if map_df.shape[1] < 2:
                map_df = pd.read_csv(mapping_file, sep='\t', header=None)
            mapping = pd.Series(map_df.iloc[:, 1].values, index=map_df.iloc[:, 0].values).to_dict()
        except Exception as e:
            sys.exit(f"Error reading mapping file: {e}")
            
    else:
        print(f"[info] Querying MyGene.info for {len(ensembl_ids)} genes...", file=sys.stderr)
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
        # print(f"[warn] Found duplicate symbols in {name} after mapping. Averaging.", file=sys.stderr)
        series = series.groupby(series.index).mean()
        
    return series

def get_symbol(name, mapper):
    return mapper.get(name, name)

def print_correlation_summary(df, beta_cols):
    """
    Calculates and prints correlation summary between LoF_gamma and each beta column.
    """
    print("\n" + "="*60, file=sys.stderr)
    print("CORRELATION SUMMARY: LoF_gamma vs. Target Betas", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    # Bins for -log10(p)
    bins = [0, 1, 2, 4, 6, 999]
    bin_labels = ["0-1", "1-2", "2-4", "4-6", "6+"]
    bin_counts = {l: 0 for l in bin_labels}
    
    results = []
    
    # Ensure LoF exists and has variance
    if "LoF_gamma" not in df.columns or df["LoF_gamma"].std() == 0:
        print("[warn] LoF_gamma column missing or constant. Cannot compute correlations.", file=sys.stderr)
        return

    # Drop global NAs for correlation check
    # (Pearson doesn't like NaNs)
    df_clean = df.dropna(subset=["LoF_gamma"])

    for col in beta_cols:
        target_name = col.replace("_beta", "")
        
        # Extract vectors, drop row-wise NAs
        sub = df_clean[["LoF_gamma", col]].dropna()
        
        if len(sub) < 3:
            results.append((target_name, 0.0, 1.0, 0.0))
            continue
            
        r, p = stats.pearsonr(sub["LoF_gamma"], sub[col])
        
        # Handle -log10 conversion
        nlog10p = -np.log10(p) if p > 1e-300 else 300.0
        
        results.append((target_name, r, p, nlog10p))
        
        # Binning
        for i, upper in enumerate(bins[1:]):
            lower = bins[i]
            if lower <= nlog10p < upper:
                bin_counts[bin_labels[i]] += 1
                break
                
    # Sort by strongest correlation (abs R or lowest p)
    # Using p-value for sorting
    results.sort(key=lambda x: x[2])
    
    # Print Bins
    print(f"{'Significance (-log10 P)':<25} | {'Count':<10}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    for label in bin_labels:
        print(f"{label:<25} | {bin_counts[label]:<10}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    # Print Top 5
    print(f"\nTop 5 Strongest Correlations:", file=sys.stderr)
    print(f"{'Gene':<15} | {'Pearson R':<10} | {'P-value':<10}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    for res in results[:5]:
        print(f"{res[0]:<15} | {res[1]:.4f}     | {res[2]:.2e}", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Create Wide Simulation TSV with LoF Gamma and multiple Beta columns.")
    
    # Inputs
    parser.add_argument("--lof", required=True, help="Path to LoF burden test TSV (Source of Gamma)")
    parser.add_argument("--limma", required=True, help="Path to Limma perturbation TSV (Source of Betas)")
    parser.add_argument("--mapping_file", help="Optional TSV (col1=ENSG, col2=Symbol)")
    
    # Output
    parser.add_argument("--out", required=True, help="Output TSV filename")
    
    # Target Selection
    parser.add_argument("--targets", nargs='+', default=[HBA1_SYMBOL], help="List of specific target genes (default: HBA1)")
    parser.add_argument("--n_top_gamma", type=int, default=0, help="Number of top LoF-Gamma genes to add as targets.")
    parser.add_argument("--n_random", type=int, default=0, help="Number of random genes to add as targets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()

    # --- 1. Load LoF Data (Gamma) ---
    print(f"[info] Reading LoF data from {args.lof}...", file=sys.stderr)
    try:
        df_lof = pd.read_csv(args.lof, sep='\t')
        if "ensg" not in df_lof.columns or "post_mean" not in df_lof.columns:
             sys.exit("Error: LoF file missing 'ensg' or 'post_mean' columns.")
        
        df_lof = df_lof.set_index("ensg")
        s_lof = df_lof["post_mean"].rename("LoF_gamma")
    except Exception as e:
        sys.exit(f"Error reading LoF file: {e}")

    # --- 2. Load Limma Data (Matrix) ---
    print(f"[info] Reading Limma data from {args.limma}...", file=sys.stderr)
    try:
        # Assuming Index = Genes (Targets), Columns = Perturbations
        df_limma = pd.read_csv(args.limma, sep='\t', index_col=0)
    except Exception as e:
        sys.exit(f"Error reading Limma file: {e}")

    # --- 3. Build Global Map & Normalize Indices ---
    print("[info] Building Gene Map & Normalizing Indices...", file=sys.stderr)
    
    # Gather all potential IDs
    all_indices = set(s_lof.index) | set(df_limma.index)
    gene_map = get_gene_mapping(list(all_indices), args.mapping_file)

    # Process LoF (Gamma)
    s_lof = process_series(s_lof, "LoF Data", gene_map)
    
    # Process Limma Index (Targets)
    # We Map Limma Index NOW so we know what targets are available
    df_limma_mapped = df_limma.copy()
    df_limma_mapped.index = df_limma_mapped.index.map(lambda x: gene_map.get(x, x))
    
    # Handle duplicate rows (mean)
    if df_limma_mapped.index.duplicated().any():
        df_limma_mapped = df_limma_mapped.groupby(df_limma_mapped.index).mean()

    # Map Limma Columns (Perturbations) to match LoF index
    if any(str(x).startswith("ENSG") for x in df_limma_mapped.columns[:5]):
         new_cols = df_limma_mapped.columns.map(lambda x: gene_map.get(x, x))
         df_limma_mapped.columns = new_cols
         if df_limma_mapped.columns.duplicated().any():
             df_limma_mapped = df_limma_mapped.groupby(df_limma_mapped.columns, axis=1).mean()

    # Define set of available targets (rows in Limma)
    available_targets = set(df_limma_mapped.index)
    print(f"[info] {len(available_targets)} unique target genes available in Limma file.", file=sys.stderr)

    # --- 4. Select Targets (Checking availability) ---
    target_symbols = set()
    
    # A. Explicit Targets
    for t in args.targets:
        sym = get_symbol(t, gene_map)
        if sym in available_targets:
            target_symbols.add(sym)
        else:
            print(f"[warn] Requested target '{sym}' not found in Limma. Skipping.", file=sys.stderr)

    # B. Top Gamma Targets
    if args.n_top_gamma > 0:
        print(f"[info] Selecting top {args.n_top_gamma} genes by LoF magnitude (checking availability)...", file=sys.stderr)
        
        # Sort LoF by magnitude
        sorted_lof = s_lof.abs().sort_values(ascending=False)
        
        # Filter for availability
        candidates = [g for g in sorted_lof.index if g in available_targets]
        
        # Pick top N
        selected_top = candidates[:args.n_top_gamma]
        target_symbols.update(selected_top)
        
        if len(selected_top) < args.n_top_gamma:
            print(f"[warn] Requested {args.n_top_gamma} top-gamma genes, but only found {len(selected_top)} in Limma.", file=sys.stderr)

    # C. Random Targets
    if args.n_random > 0:
        print(f"[info] Selecting {args.n_random} random genes...", file=sys.stderr)
        rng = np.random.default_rng(args.seed)
        
        # Sample from available targets
        # Convert set to sorted list for reproducibility before sampling
        sorted_available = sorted(list(available_targets))
        
        random_genes = rng.choice(sorted_available, size=args.n_random, replace=False)
        target_symbols.update(random_genes)

    sorted_targets = sorted(list(target_symbols))
    print(f"[info] Final Selection: {len(sorted_targets)} target genes.", file=sys.stderr)

    # --- 5. Extract Betas & Merge ---
    # Master DataFrame starts with LoF
    # We use OUTER join to capture all perturbations, even if LoF is missing for some?
    # No, usually simulation drives off LoF. Let's use LoF as base.
    df_master = pd.DataFrame(s_lof)
    
    extracted_cols = []
    
    for target in sorted_targets:
        # We already checked availability, so this should be safe
        s_beta = df_limma_mapped.loc[target]
        col_name = f"{target}_beta"
        s_beta = s_beta.rename(col_name)
        
        # Join (Left join to keep LoF structure)
        df_master = df_master.join(s_beta, how='left')
        extracted_cols.append(col_name)

    # --- 6. Cleanup & Save ---
    df_master.index.name = "gene_name"
    df_master = df_master.reset_index()
    
    print(f"[info] Saving {len(df_master)} rows and {len(extracted_cols)} beta columns to {args.out}...", file=sys.stderr)
    df_master.to_csv(args.out, sep='\t', index=False, na_rep="NA")
    
    # --- 7. Correlation Summary ---
    print_correlation_summary(df_master, extracted_cols)
    
    print("[OK] Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
