import os, json
import anndata as ad
import pandas as pd
from utils.config import load_datasets_yaml
from utils.normalize import normalize_hgnc
from pathlib import Path

def main(yaml_path: str, out_dir: str = "artifacts", min_gene_len: int = 2000):
    cfg = load_datasets_yaml(yaml_path)
    datasets = cfg["datasets"]
    gene_lists = {}
    for dataset_id, ds_cfg in datasets.items():
        path = ds_cfg.get("raw_path") or ds_cfg.get("path")
        if path is None or not os.path.exists(path):
            print(f"[WARN] {dataset_id}: raw_path not found → skip")
            continue
        print(f"[INFO] Reading var: {dataset_id}")
        adata = ad.read_h5ad(path, backed="r")
        genes = [normalize_hgnc(g) for g in adata.var_names.astype(str)]
        # drop empties
        genes = [g for g in genes if g]
        gene_lists[dataset_id] = genes

    if not gene_lists:
        raise SystemExit("No datasets found")

    # Compute intersection
    inter = set(next(iter(gene_lists.values())))
    for gl in gene_lists.values():
        inter &= set(gl)

    inter = sorted(list(inter))
    if len(inter) < min_gene_len:
        print(f"[WARN] Intersection has only {len(inter)} genes (<{min_gene_len}). Consider restricting datasets.")
    out_gene_list = os.path.join(out_dir, "gene_list.tsv")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_gene_list, "w") as f:
        for g in inter:
            f.write(g + "\n")
    print(f"[OK] Wrote common gene list: {out_gene_list} ({len(inter)} genes)")

    # Per-dataset mapping
    maps_dir = os.path.join(out_dir, "gene_map")
    os.makedirs(maps_dir, exist_ok=True)
    common_index = {g:i for i,g in enumerate(inter)}
    for dataset_id, genes in gene_lists.items():
        to_common = {}
        for i, g in enumerate(genes):
            j = common_index.get(g, -1)
            if j != -1:
                to_common[str(i)] = j
        mp = {
            "dataset_gene_count": len(genes),
            "common_gene_count": len(inter),
            "mapped_genes": len(to_common),
            "to_common_idx": to_common
        }
        with open(os.path.join(maps_dir, f"{dataset_id}.json"), "w") as f:
            json.dump(mp, f)
    print(f"[OK] Wrote per-dataset gene maps to: {maps_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--min-gene-len", type=int, default=2000)
    args = ap.parse_args()
    main(args.yaml, args.out_dir, args.min_gene_len)
