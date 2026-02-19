#!/usr/bin/env python3
"""
Utility script to extract the top X loaded genes per factor from an NMF results matrix.
Rows = Factors, Columns = Genes (Ensembl IDs).
Translates Ensembl IDs to Gene Symbols.
"""

import argparse
import sys
import pandas as pd
import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description="Extract top X loaded genes per factor from an NMF matrix.")
    parser.add_argument("--input", "-i", required=True, help="Path to the NMF matrix TSV (rows=factors, cols=genes).")
    parser.add_argument("--top_x", "-x", type=int, required=True, help="Number of top genes to extract per factor.")
    parser.add_argument("--output", "-o", required=True, help="Path to save the output TSV file.")
    parser.add_argument("--mapping_file", help="Optional TSV (col1=ENSG, col2=Symbol) for manual gene name mapping.")
    
    args = parser.parse_args()

    # 1. Load the NMF matrix
    print(f"[info] Loading NMF matrix from {args.input}...", file=sys.stderr)
    try:
        df = pd.read_csv(args.input, sep='\t', index_col=0)
    except Exception as e:
        sys.exit(f"Error reading input file: {e}")

    # 2. Get Gene Translations
    gene_ids = df.columns.tolist()
    gene_map = get_gene_mapping(gene_ids, args.mapping_file)

    # 3. Extract Top X Genes per Factor
    print(f"[info] Extracting top {args.top_x} genes for {len(df)} factors...", file=sys.stderr)
    
    results = []
    
    # Iterate over each factor (row)
    for factor_name, row in df.iterrows():
        # Get absolute values for ranking
        abs_row = row.abs()
        
        # Sort descending by absolute value and take top X
        top_genes = abs_row.sort_values(ascending=False).head(args.top_x).index
        
        # Compile results
        for rank, gene_id in enumerate(top_genes, start=1):
            symbol = gene_map.get(gene_id, gene_id) # Fallback to Ensembl if symbol not found
            loading = row[gene_id] # Keep the original signed loading
            
            results.append({
                'Factor': factor_name,
                'Rank': rank,
                'Ensembl_ID': gene_id,
                'Symbol': symbol,
                'Loading': loading,
                'Abs_Loading': abs(loading)
            })

    # 4. Save Output
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, sep='\t', index=False)
    print(f"[info] Successfully saved {len(out_df)} records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()