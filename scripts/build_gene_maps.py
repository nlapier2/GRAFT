#!/usr/bin/env python3
# temporary workaround for script visibility
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import pandas as pd

from utils.config import load_datasets_yaml
from utils.normalize import normalize_hgnc


def _read_var_names(h5ad_path: str) -> List[str]:
    """Read var_names from an AnnData H5AD (backed), normalize, deduplicate (preserve order)."""
    A = ad.read_h5ad(h5ad_path, backed="r")
    try:
        raw = [str(x) for x in A.var_names]
    finally:
        # ensure HDF5 handle is released
        try:
            A.file.close()
        except Exception:
            try:
                A._backed.close()
            except Exception:
                pass

    # normalize + drop empties
    genes = [normalize_hgnc(g) for g in raw]
    genes = [g for g in genes if g]

    # deduplicate preserving first occurrence
    seen = set()
    uniq = []
    for g in genes:
        if g not in seen:
            uniq.append(g)
            seen.add(g)
    return uniq


def _apply_filters(genes: List[str],
                   allow_set: Optional[set],
                   deny_regex: Optional[re.Pattern]) -> List[str]:
    if allow_set is not None:
        genes = [g for g in genes if g in allow_set]
    if deny_regex is not None:
        genes = [g for g in genes if not deny_regex.search(g)]
    return genes


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for x in lines:
            f.write(f"{x}\n")


def _make_to_common_idx(dataset_genes: List[str], master_index: Dict[str, int]) -> Dict[str, int]:
    """Map dataset gene position -> master index (only for genes that exist in master)."""
    to_common = {}
    for i, g in enumerate(dataset_genes):
        j = master_index.get(g, -1)
        if j != -1:
            to_common[str(i)] = j
    return to_common


def _present_master_indices(dataset_genes: List[str], master_index: Dict[str, int]) -> List[int]:
    """Return sorted list of master indices that are present in this dataset."""
    present = []
    for g in dataset_genes:
        j = master_index.get(g, -1)
        if j != -1:
            present.append(j)
    # unique + sorted (in case dataset has dupes prior to normalization)
    return sorted(set(present))


def main(
    yaml_path: str,
    out_dir: str = "artifacts",
    min_gene_len: int = 2000,
    target_dataset_id: Optional[str] = None,
    allow_list_tsv: Optional[str] = None,
    deny_regex: Optional[str] = None,
):
    """
    Build the master gene list and per-dataset maps.

    Modes:
      - Default (no target): master = intersection across datasets (sorted).
      - Target-specified: master = ALL genes from target dataset, in that dataset's order.

    Filters:
      - allow_list_tsv: optional TSV with one gene symbol per line to whitelist.
      - deny_regex: optional regex to drop dubious gene symbols.
    """
    cfg = load_datasets_yaml(yaml_path)
    datasets = cfg.get("datasets", {})
    if not datasets:
        raise SystemExit("No datasets in YAML.")

    # Optional filters
    allow_set = None
    if allow_list_tsv:
        allow_set = {normalize_hgnc(x.strip()) for x in Path(allow_list_tsv).read_text().splitlines() if x.strip()}
        allow_set.discard("")  # remove empties if any

    deny_pat = re.compile(deny_regex) if deny_regex else None

    # Read per-dataset gene lists
    gene_lists: Dict[str, List[str]] = {}
    for dataset_id, ds_cfg in datasets.items():
        path = ds_cfg.get("raw_path") or ds_cfg.get("path")
        if path is None or not os.path.exists(path):
            print(f"[WARN] {dataset_id}: raw_path not found → skip")
            continue

        print(f"[INFO] Reading var: {dataset_id}")
        genes = _read_var_names(path)
        genes = _apply_filters(genes, allow_set, deny_pat)
        if not genes:
            print(f"[WARN] {dataset_id}: no genes after filtering → skip")
            continue
        gene_lists[str(dataset_id)] = genes

    if not gene_lists:
        raise SystemExit("No usable datasets found after reading/filters.")

    # Determine master gene list
    master: List[str]
    master_source = "intersection"
    if target_dataset_id:
        td = str(target_dataset_id)
        if td not in gene_lists:
            raise SystemExit(f"target_dataset_id '{td}' not found among loaded datasets.")
        master = list(gene_lists[td])  # preserve original order of the target dataset
        master_source = f"target:{td}"
    else:
        # intersection across datasets (sorted for stability)
        inter = set(next(iter(gene_lists.values())))
        for gl in gene_lists.values():
            inter &= set(gl)
        master = sorted(list(inter))
        if len(master) < min_gene_len:
            print(f"[WARN] Intersection has only {len(master)} genes (<{min_gene_len}). "
                  f"Consider using --target-dataset-id to widen coverage.")

    if not master:
        raise SystemExit("Master gene list is empty after filtering; aborting.")

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Write master gene list
    out_gene_list = os.path.join(out_dir, "gene_list.tsv")
    _write_lines(out_gene_list, master)
    print(f"[OK] Wrote master gene list ({master_source}): {out_gene_list} ({len(master)} genes)")

    # Build per-dataset maps
    maps_dir = os.path.join(out_dir, "gene_map")
    os.makedirs(maps_dir, exist_ok=True)
    master_index = {g: i for i, g in enumerate(master)}

    for dataset_id, genes in gene_lists.items():
        to_common = _make_to_common_idx(genes, master_index)
        present_idx = _present_master_indices(genes, master_index)

        mp = {
            "dataset_id": dataset_id,
            "dataset_gene_count": len(genes),
            "master_gene_count": len(master),
            "mapped_dataset_genes": len(to_common),   # number of dataset genes that map into master
            "present_master_genes": len(present_idx), # number of master genes present in this dataset
            "to_common_idx": to_common,               # str(i_dataset) -> j_master
            "present_master_idx": present_idx,        # sorted list of j_master present in this dataset
        }
        with open(os.path.join(maps_dir, f"{dataset_id}.json"), "w") as f:
            json.dump(mp, f)

    print(f"[OK] Wrote per-dataset gene maps to: {maps_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build master gene list and per-dataset gene maps.")
    ap.add_argument("--yaml", required=True, help="Path to datasets YAML.")
    ap.add_argument("--out-dir", default="artifacts", help="Output directory.")
    ap.add_argument("--min-gene-len", type=int, default=2000, help="Warn if intersection smaller than this.")
    ap.add_argument("--target-dataset-id", default=None, help="If set, master genes = all genes from this dataset (in order).")
    ap.add_argument("--allow-list-tsv", default=None, help="Optional TSV with one gene symbol per line to keep.")
    ap.add_argument("--deny-regex", default=None, help="Optional regex; any matching genes will be dropped.")
    args = ap.parse_args()

    main(
        yaml_path=args.yaml,
        out_dir=args.out_dir,
        min_gene_len=args.min_gene_len,
        target_dataset_id=args.target_dataset_id,
        allow_list_tsv=args.allow_list_tsv,
        deny_regex=args.deny_regex,
    )
