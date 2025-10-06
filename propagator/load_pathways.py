# script to load pathway definitions from various sources into a common format
import pandas as pd
import yaml
import pickle


def load_pathway_sources(cfg_path: str | None) -> dict[str, dict[str, str]]:
    """Read YAML -> dict; fall back to defaults if not provided."""
    if cfg_path is None:
        return {}
    with open(cfg_path, "r") as fh:
        data = yaml.safe_load(fh)
    required = {"file", "gene_col", "pathway_col", "format"}
    for name, meta in data.items():
        missing = required - meta.keys()
        if missing:
            raise ValueError(
                f"Pathway source '{name}' missing keys: {', '.join(missing)}"
            )
    return data


def make_pathway_matrix_from_tsv(
    tsv_file: str,
    gene_col: str,
    pathway_col: str,
    var_names: list[str],
) -> pd.DataFrame:
    """
    Create a pathway matrix (pathways x genes) from a TSV file, aligned to genes in an AnnData object.

    Args:
        tsv_file: Path to the TSV file.
        gene_col: Column name for gene names in the TSV.
        pathway_col: Column name for pathway identifiers in the TSV.
        var_names: List of gene names (var_names) from the AnnData object.

    Returns:
        pd.DataFrame: Pathway matrix (rows=pathways, columns=genes), values are 1 if gene in pathway, else 0.
    """
    df = pd.read_csv(tsv_file, sep="\t")

    # If pathway_col looks like it contains URLs, extract the last segment
    if df[pathway_col].astype(str).str.contains("http").any():
        df[pathway_col] = df[pathway_col].astype(str).str.extract(r'.*/([^/]+)$')[0]

    genes = set(var_names)
    df = df[df[gene_col].isin(genes)]

    pathway_matrix = (
        df.groupby([pathway_col, gene_col])
        .size()
        .unstack(fill_value=0)
        .astype(float)
    )

    # Ensure all genes from AnnData are present as columns
    missing_genes = [g for g in var_names if g not in pathway_matrix.columns]
    if missing_genes:
        filler = pd.DataFrame(0.0, index=pathway_matrix.index, columns=missing_genes)
        pathway_matrix = pd.concat([pathway_matrix, filler], axis=1)

    pathway_matrix = pathway_matrix[var_names]

    return pathway_matrix.T


def make_pathway_matrix_presage(
    pickle_file: str,
    var_names: list[str],
) -> pd.DataFrame:
    """
    Load a pathway matrix from a pickle file (genes as rows, pathways as columns),
    transpose it (so pathways are rows, genes are columns), and align to genes in AnnData.

    Args:
        pickle_file: Path to the pickle file containing the DataFrame.
        var_names: List of gene names (var_names) from the AnnData object.

    Returns:
        pd.DataFrame: Pathway matrix (rows=pathways, columns=genes), values as in the pickle file (or 0 if missing).
    """
    with open(pickle_file, "rb") as f:
        pathway_matrix = pickle.load(f).T

    # Intersect with adata genes and ensure all adata genes are present as columns
    genes = set(var_names)
    pathway_matrix = pathway_matrix.loc[:, pathway_matrix.columns.isin(genes)]

    # Add missing genes as columns of zeros
    missing_genes = [g for g in var_names if g not in pathway_matrix.columns]
    if missing_genes:
        filler = pd.DataFrame(0.0, index=pathway_matrix.index, columns=missing_genes)
        pathway_matrix = pd.concat([pathway_matrix, filler], axis=1)

    # Reorder columns to match var_names
    pathway_matrix = pathway_matrix[var_names]

    return pathway_matrix.T


def make_pathway_matrix(
    file_name: str,
    gene_col: str,
    pathway_col: str,
    format: str,
    var_names: list[str],
) -> pd.DataFrame:
    if format == 'tsv':
        return make_pathway_matrix_from_tsv(file_name, gene_col, pathway_col, var_names)
    elif format == 'presage':
        return make_pathway_matrix_presage(file_name, var_names)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'tsv' and 'presage'.")
