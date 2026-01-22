#!/usr/bin/env python3
"""
Utility script to merge LoF posterior means and Limma perturbation effects.
Fixes: Maps IDs to symbols BEFORE merging to ensure rows align correctly.

- Reads LoF burden test (index=ENSG).
- Reads Limma results (extracts row for HBA1/ENSG00000206172).
- Converts all indices to Gene Symbols.
- Merges on Gene Symbol.
- Removes HBA1 (self-interaction) and handles duplicates.
"""

import argparse
import sys
import pandas as pd
import numpy as np

# Constants
HBA1_ENSG = "ENSG00000206172"
HBA1_SYMBOL = "HBA1"

def get_gene_mapping(ids_to_map, mapping_file=None):
    """
    Returns a dictionary mapping ID -> Symbol.
    Only queries MyGene for IDs that start with 'ENSG'.
    """
    mapping = {}
    
    # Filter for things that actually look like Ensembl IDs
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
    Takes a pandas Series (index=IDs/Symbols), maps the index to Symbols,
    aggregates duplicates (mean), and returns a cleaned Series.
    """
    # Map index using the mapper; if not in mapper, keep original (assumes it's already a Symbol)
    new_index = series.index.map(lambda x: mapper.get(x, x))
    series.index = new_index
    
    # If mapping created duplicates (e.g. multiple ENSGs -> same Symbol), take the mean
    if series.index.duplicated().any():
        print(f"[warn] Found duplicate gene symbols in {name} after mapping. Averaging values.", file=sys.stderr)
        series = series.groupby(series.index).mean()
        
    return series

def main():
    parser = argparse.ArgumentParser(description="Merge LoF and Limma results by Gene Name.")
    parser.add_argument("--lof", required=True, help="Path to LoF burden test TSV")
    parser.add_argument("--limma", required=True, help="Path to Limma perturbation TSV")
    parser.add_argument("--out", required=True, help="Output TSV filename")
    parser.add_argument("--mapping_file", help="Optional TSV (col1=ENSG, col2=Symbol)")
    
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"[info] Reading LoF data from {args.lof}...", file=sys.stderr)
    try:
        df_lof = pd.read_csv(args.lof, sep='\t')
        if "ensg" not in df_lof.columns or "post_mean" not in df_lof.columns:
             sys.exit("Error: LoF file missing 'ensg' or 'post_mean' columns.")
        
        df_lof = df_lof.set_index("ensg")
        s_lof = df_lof["post_mean"].rename("LoF_gamma")
    except Exception as e:
        sys.exit(f"Error reading LoF file: {e}")

    print(f"[info] Reading Limma data from {args.limma}...", file=sys.stderr)
    try:
        df_limma = pd.read_csv(args.limma, sep='\t', index_col=0)
        
        # Check for HBA1 in index (try both ID and Symbol)
        if HBA1_ENSG in df_limma.index:
            s_limma = df_limma.loc[HBA1_ENSG]
        elif HBA1_SYMBOL in df_limma.index:
            s_limma = df_limma.loc[HBA1_SYMBOL]
        else:
            sys.exit(f"Error: HBA1 ({HBA1_ENSG}) not found in Limma rows.")
            
        s_limma = s_limma.rename("HBA1_beta")
    except Exception as e:
        sys.exit(f"Error reading Limma file: {e}")

    # --- 2. Build Mapping Dictionary ---
    # Collect all unique indices from both files
    all_indices = set(s_lof.index) | set(s_limma.index)
    gene_map = get_gene_mapping(list(all_indices), args.mapping_file)

    # --- 3. Apply Mapping & Normalize Indices ---
    s_lof = process_series(s_lof, "LoF Data", gene_map)
    s_limma = process_series(s_limma, "Limma Data", gene_map)

    # --- 4. Merge ---
    print("[info] Merging datasets...", file=sys.stderr)
    merged_df = pd.DataFrame(s_lof).join(s_limma, how='outer')
    
    # Fill index name
    merged_df.index.name = "gene_name"
    merged_df = merged_df.reset_index()

    # --- 5. Filter HBA1 ---
    # Filter out HBA1 (self-interaction)
    # Check for likely HBA1 strings
    mask = ~merged_df["gene_name"].isin([HBA1_SYMBOL, HBA1_ENSG, "ENSG00000206172"])
    merged_df = merged_df[mask]

    # --- 6. Save ---
    # Reorder cols
    merged_df = merged_df[["gene_name", "LoF_gamma", "HBA1_beta"]]
    merged_df = merged_df.sort_values("gene_name")
    
    print(f"[info] Saving {len(merged_df)} rows to {args.out}...", file=sys.stderr)
    merged_df.to_csv(args.out, sep='\t', index=False, na_rep="NA")
    print("[OK] Done.", file=sys.stderr)

if __name__ == "__main__":
    main()