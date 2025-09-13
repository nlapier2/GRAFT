#!/usr/bin/env python3
"""
normalize_with_scvi_pipeline.py

Normalize an AnnData using the exact same streaming/scVI pipeline as predict.py,
then write a new AnnData with .X = scVI normalized expression.

- No GRAFT; no re-noising; no gene unscrambling (assumes genes already match scVI training).
- Uses your config (paths.*) and the same dataset loader.
- Works even if the input AnnData lacks dataset_id/batch_id; the loader handles setup.

Example:
  python normalize_with_scvi_pipeline.py \
    --config configs/graft_vcc_randsplit.yaml \
    --input-h5ad /path/to/input.h5ad \
    --dataset-id VCC \
    --output-h5ad /path/to/output_scvi_norm.h5ad \
    --batch-size 4096 \
    --save-original-to-layer counts
"""

import os
import argparse
import tempfile
import numpy as np
import pandas as pd
import anndata as ad
import yaml
from scipy import sparse

# Import your pipeline bits (same as predict.py)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.build_index import main as build_index_main
from graft.data.samplers import BalancedRoundRobin
from graft.data.dataset import GraftStreamingConfig, GraftStreamingDataset
from graft.utils.chunk_preprocess import load_gene_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Your training config YAML (same one used by predict.py)")
    ap.add_argument("--input-h5ad", required=True, help="AnnData to normalize with scVI")
    ap.add_argument("--dataset-id", required=True, help="Dataset ID key under paths.datasets_yaml to point to this file")
    ap.add_argument("--output-h5ad", required=True, help="Where to write the normalized AnnData")
    ap.add_argument("--batch-size", type=int, default=4096, help="Streaming batch size")
    ap.add_argument("--save-original-to-layer", default=None, help="If set, stash original X in this layer name")
    args = ap.parse_args()

    # --- Load config and paths (same as predict.py) ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    paths, model_cfg, train_cfg = cfg["paths"], cfg["model"], cfg["training"]

    gene_list = load_gene_list(paths["gene_list_tsv"])

    # --- Build a tiny datasets YAML pointing the requested dataset to the input file ---
    with open(paths["datasets_yaml"], "r") as f:
        datasets_yaml_data = yaml.safe_load(f)
    if args.dataset_id not in datasets_yaml_data["datasets"]:
        raise ValueError(f"--dataset-id={args.dataset_id} not found in {paths['datasets_yaml']}")
    target_ds_config = datasets_yaml_data["datasets"][args.dataset_id]

    temp_config = {
        "defaults": datasets_yaml_data.get("defaults", {}),
        "datasets": {args.dataset_id: dict(target_ds_config)},  # shallow copy
    }
    temp_config["datasets"][args.dataset_id]["raw_path"] = args.input_h5ad

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_yaml:
        yaml.dump(temp_config, tmp_yaml)
        temp_yaml_file = tmp_yaml.name

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_idx:
        temp_index_file = tmp_idx.name

    # --- Index the input file (exactly like predict.py) ---
    print("[info] Building index for input AnnData...")
    build_index_main(yaml_path=temp_yaml_file, out_index=temp_index_file, gene_list_path=paths["gene_list_tsv"])

    # --- Load the input AnnData for final write-out scaffolding ---
    A_in = ad.read_h5ad(args.input_h5ad)
    original_obs = A_in.obs.copy()
    original_var = A_in.var.copy()
    original_genes = A_in.var_names.astype(str).tolist()

    # Optionally stash original counts as a layer on the output
    save_layer = args.save_original_to_layer

    # --- Stream with same loader knobs (no filtering; include all rows in query) ---
    # We want normalized expression for ALL cells in the file.
    sampler = BalancedRoundRobin(dataset_ids=[args.dataset_id], steps=None)
    ds_cfg = GraftStreamingConfig(
        datasets_yaml=temp_yaml_file, index_parquet=temp_index_file, #paths["index_parquet"],
        gene_list_tsv=paths["gene_list_tsv"], scvi_model_dir=paths["scvi_model_dir"],
        scvi_input_h5ad=paths["scvi_input_h5ad"], control_index_dir=paths["control_index_dir"],
        control_z_npz=paths["control_z_npz"], control_xbar_npz=paths["control_xbar_npz"],
        batch_size=args.batch_size, chunk_size=train_cfg.get("chunk_size", 50000),
        k_controls=train_cfg["k_controls"], oversample=train_cfg["oversample"],
        match_within=train_cfg.get("match_within", "dataset"), forward_batch_size=train_cfg["forward_batch_size"],
        include_controls_in_query=True,
        filter_by_index=True  # Ok with temp index since we want all cells in the template
    )
    ds = GraftStreamingDataset(ds_cfg, sampler)
    data_iterator = iter(ds)

    # Pre-allocate output in input order
    n_obs, G = A_in.n_obs, len(gene_list)
    X_norm = np.empty((n_obs, G), dtype=np.float32)

    # Map global -> local index to reassemble in input order
    # Global ids are "dataset::local"; local ids are A_in.obs_names (str)
    local_to_pos = {str(lid): i for i, lid in enumerate(A_in.obs_names.astype(str))}
    filled = 0

    print("[info] Streaming and collecting scVI-normalized expression...")
    while filled < n_obs:
        try:
            batch = next(data_iterator)
        except StopIteration:
            break

        cell_ids = batch["cell_ids"]  # list[str] like "DS::local_id"
        # normalized expression for the query rows (what we want)
        # Depending on your dataset version, the key is commonly "xbar_q".
        if "xbar_q" in batch:
            xbar = batch["xbar_q"]
        elif "x_q" in batch:
            xbar = batch["x_q"]
        else:
            raise KeyError("Batch missing normalized expression ('xbar_q' or 'x_q'). Please ensure dataset.py exposes it.")

        # Place rows into output in the exact input order
        # Extract local ids after the "::"
        local_ids = [cid.split("::", 1)[1] for cid in cell_ids]
        for row, lid in enumerate(local_ids):
            pos = local_to_pos.get(lid)
            if pos is None:
                # If a cell in the stream isn't in the input (shouldn't happen with this index), skip
                continue
            X_norm[pos, :] = xbar[row, :]
            filled += 1

    if filled != n_obs:
        print(f"[warn] Filled {filled} / {n_obs} rows from the stream. Missing rows will be left as zeros.")

    # --- Build output AnnData ---
    # Keep obs/var from the input; replace X with normalized expression
    from scipy.sparse import csr_matrix
    A_out = ad.AnnData(
        X=csr_matrix(X_norm),  # sparse to save space; change to dense if you prefer
        obs=original_obs,
        var=original_var,
    )
    if save_layer:
        # store original counts (as CSR) before replacement
        if sparse.issparse(A_in.X):
            A_out.layers[str(save_layer)] = A_in.X.copy()
        else:
            A_out.layers[str(save_layer)] = csr_matrix(A_in.X)

    # --- Write result ---
    outdir = os.path.dirname(args.output_h5ad)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    A_out.var_names = original_genes
    A_out.write_h5ad(args.output_h5ad, compression="lzf")
    print(f"[done] Wrote scVI-normalized AnnData to: {args.output_h5ad}")

if __name__ == "__main__":
    main()
