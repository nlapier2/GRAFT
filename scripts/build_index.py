# scripts/build_index.py

# temporary workaround for script visibility
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import json
import anndata as ad
import pandas as pd
from pathlib import Path
from utils.config import load_datasets_yaml
from utils.normalize import normalize_hgnc
from utils.report import coverage_tables

CANON_COLS = ["dataset_id","cell_id","lab_id","batch_id","cell_type",
              "is_control","pert_type","target_gene","guide_id",
              "dose","time_h","target_id","perturbation"]

def clean_str_series(s: pd.Series) -> pd.Series:
    """
    Coerce a series to clean strings: use pandas 'string' dtype, replace NaN/None with '',
    strip whitespace. Returns a pandas Series of python str objects.
    """
    if s is None:
        return pd.Series([], dtype="string")
    try:
        s2 = s.astype("string")
    except Exception:
        s2 = pd.Series(s, dtype="string")
    return s2.fillna("").astype(str).str.strip()

def resolve_is_control(obs: pd.DataFrame, ds_cfg: dict, defaults: dict) -> pd.Series:
    # 1) explicit boolean column
    is_col = ds_cfg.get("obs_map", {}).get("is_control_col")
    if is_col and is_col in obs.columns:
        return obs[is_col].astype("boolean").fillna(False).astype(bool)

    # control tokens from YAML defaults/dataset; include blanks & nan by default
    tokens = set([str(v).lower() for v in ds_cfg.get("control_values", defaults.get("control_tokens", []))])

    # 2) derive from ANY OF multiple string columns
    any_of = ds_cfg.get("obs_map", {}).get("is_control_any_of")
    if any_of:
        for col in any_of:
            if col in obs.columns:
                s = clean_str_series(obs[col]).str.lower()
                mask = s.isin(tokens) | s.eq("")  # treat empty as control
                # also treat actual missing as control
                mask = mask | obs[col].isna()
                if mask.any():
                    return mask.fillna(False)

    # 3) derive from a single source column
    from_col = ds_cfg.get("obs_map", {}).get("is_control_from")
    if from_col and from_col in obs.columns:
        s = clean_str_series(obs[from_col]).str.lower()
        mask = s.isin(tokens) | s.eq("")
        mask = mask | obs[from_col].isna()
        return mask.fillna(False)

    # 4) numeric rule
    if ds_cfg.get("control_rule") == "equals_zero":
        from_col = ds_cfg.get("obs_map", {}).get("is_control_from")
        if from_col and from_col in obs.columns:
            return (pd.to_numeric(obs[from_col], errors="coerce").fillna(1) == 0)

    # 5) fallback: False
    return pd.Series(False, index=obs.index)

def main(yaml_path: str, out_index: str = "artifacts/cell_index.parquet", gene_list_path: str = "artifacts/gene_list.tsv"):
    cfg = load_datasets_yaml(yaml_path)
    defaults = cfg["defaults"]
    rows = []
    all_genes = None
    gene_list = None
    # if a precomputed gene list exists, read it to set common_gene_set flags
    if os.path.exists(gene_list_path):
        gene_list = [line.strip() for line in open(gene_list_path, "r") if line.strip()]

    for dataset_id, ds_cfg in cfg["datasets"].items():
        path = ds_cfg.get("raw_path") or ds_cfg.get("path")
        if path is None or not os.path.exists(path):
            print(f"[WARN] {dataset_id}: raw_path not found → skip")
            continue
        print(f"[INFO] Reading {dataset_id} (obs/var only): {path}")
        adata = ad.read_h5ad(path, backed="r")

        obs = adata.obs.copy()
        # map columns
        omap = (ds_cfg.get("obs_map") or {})
        out = pd.DataFrame(index=obs.index)

        # dataset_id and cell_id
        out["dataset_id"] = dataset_id
        out["cell_id"] = dataset_id + "::" + obs.index.astype(str)

        # lab_id/batch_id/cell_type/pert_type/target_gene/guide_id/target_id/perturbation/dose/time_h
        lab = ds_cfg.get("lab_id") or dataset_id
        out["lab_id"] = clean_str_series(pd.Series(lab, index=obs.index))

        # batch_id
        bcol = omap.get("batch_id")
        if bcol and bcol in obs:
            out["batch_id"] = clean_str_series(obs[bcol])
        else:
            out["batch_id"] = clean_str_series(pd.Series(ds_cfg.get("fixed", {}).get("batch_id", "batch0"), index=obs.index))

        # cell_type
        ccol = omap.get("cell_type")
        if ccol and ccol in obs:
            out["cell_type"] = clean_str_series(obs[ccol])
        else:
            out["cell_type"] = clean_str_series(pd.Series(ds_cfg.get("fixed", {}).get("cell_type", "UNKNOWN"), index=obs.index))

        # pert_type
        pcol = omap.get("pert_type")
        if pcol and pcol in obs:
            out["pert_type"] = clean_str_series(obs[pcol]).str.lower()
        else:
            out["pert_type"] = clean_str_series(pd.Series(ds_cfg.get("fixed", {}).get("pert_type") or defaults.get("pert_type_guess", "crispri"), index=obs.index)).str.lower()

        # target_gene (normalized)
        tgcol = omap.get("target_gene")
        if tgcol and tgcol in obs:
            out["target_gene"] = clean_str_series(obs[tgcol]).apply(normalize_hgnc)
        else:
            out["target_gene"] = clean_str_series(pd.Series("", index=obs.index))

        # guide_id
        gcol = omap.get("guide_id")
        out["guide_id"] = clean_str_series(obs[gcol]) if gcol and gcol in obs else clean_str_series(pd.Series("", index=obs.index))

        # target_id (keep raw)
        tcol = omap.get("target_id")
        out["target_id"] = clean_str_series(obs[tcol]) if tcol and tcol in obs else clean_str_series(pd.Series("", index=obs.index))

        # perturbation (raw string if present)
        p2 = omap.get("perturbation")
        out["perturbation"] = clean_str_series(obs[p2]) if p2 and p2 in obs else clean_str_series(pd.Series("", index=obs.index))

        # dose/time_h
        dcol = omap.get("dose")
        tcol = omap.get("time_h")
        out["dose"] = pd.to_numeric(obs[dcol], errors="coerce") if dcol and dcol in obs else pd.NA
        out["time_h"] = pd.to_numeric(obs[tcol], errors="coerce") if tcol and tcol in obs else pd.NA

        # is_control
        out["is_control"] = resolve_is_control(obs, ds_cfg, defaults)

        # common_gene_set flag
        if gene_list is not None:
            ad_genes = adata.var_names.astype(str).str.upper().tolist()
            adset = set(ad_genes)
            out["common_gene_set"] = all(g.upper() in adset for g in gene_list)
        else:
            out["common_gene_set"] = True  # will be corrected after gene map step

        rows.append(out[CANON_COLS + ["common_gene_set"]])

    if not rows:
        raise SystemExit("No datasets loaded. Check paths in YAML.")

    idx = pd.concat(rows, axis=0).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    idx.to_parquet(out_index, index=False)
    print(f"[OK] Wrote index: {out_index} ({len(idx):,} rows)")
    # write a coverage report
    coverage_tables(idx, cfg["defaults"].get("env_key","batch_id"), "reports/coverage_step0.md")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="configs/datasets.yaml")
    ap.add_argument("--out", default="artifacts/cell_index.parquet")
    ap.add_argument("--gene-list", default="artifacts/gene_list.tsv")
    args = ap.parse_args()
    main(args.yaml, args.out, args.gene_list)
