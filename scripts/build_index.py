import os
import json
import anndata as ad
import pandas as pd
from pathlib import Path
from utils.config import load_datasets_yaml
from utils.normalize import normalize_hgnc
from utils.report import coverage_tables

# temporary workaround for script visibility
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CANON_COLS = ["dataset_id","cell_id","lab_id","batch_id","cell_type",
              "is_control","pert_type","target_gene","guide_id",
              "dose","time_h","target_id","perturbation"]

def resolve_is_control(obs: pd.DataFrame, ds_cfg: dict, defaults: dict) -> pd.Series:
    # boolean column first
    is_col = ds_cfg.get("obs_map", {}).get("is_control_col")
    if is_col and is_col in obs.columns:
        return obs[is_col].astype(bool).fillna(False)

    # derive from string column(s)
    tokens = set([str(v).lower() for v in ds_cfg.get("control_values", defaults.get("control_tokens", []))])
    any_of = ds_cfg.get("obs_map", {}).get("is_control_any_of")
    if any_of:
        for col in any_of:
            if col in obs.columns:
                s = obs[col].astype(str).str.lower()
                mask = s.isin(tokens)
                if mask.any():
                    return mask.fillna(False)
    from_col = ds_cfg.get("obs_map", {}).get("is_control_from")
    if from_col and from_col in obs.columns:
        s = obs[from_col].astype(str).str.lower()
        mask = s.isin(tokens)
        return mask.fillna(False)

    # numeric rule
    if ds_cfg.get("control_rule") == "equals_zero":
        from_col = ds_cfg.get("obs_map", {}).get("is_control_from")
        if from_col and from_col in obs.columns:
            return (pd.to_numeric(obs[from_col], errors="coerce").fillna(1) == 0)

    # fallback: False
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
        out["lab_id"] = lab
        # batch_id
        bcol = omap.get("batch_id")
        if bcol and bcol in obs:
            out["batch_id"] = obs[bcol].astype(str)
        else:
            out["batch_id"] = ds_cfg.get("fixed", {}).get("batch_id", "batch0")
        # cell_type
        ccol = omap.get("cell_type")
        if ccol and ccol in obs:
            out["cell_type"] = obs[ccol].astype(str)
        else:
            out["cell_type"] = ds_cfg.get("fixed", {}).get("cell_type", "UNKNOWN")
        # pert_type
        pcol = omap.get("pert_type")
        if pcol and pcol in obs:
            out["pert_type"] = obs[pcol].astype(str).str.lower()
        else:
            out["pert_type"] = (ds_cfg.get("fixed", {}).get("pert_type")
                                or defaults.get("pert_type_guess", "crispri"))
        # target_gene (normalized)
        tgcol = omap.get("target_gene")
        if tgcol and tgcol in obs:
            out["target_gene"] = obs[tgcol].apply(normalize_hgnc)
        else:
            out["target_gene"] = ""
        # guide_id
        gcol = omap.get("guide_id")
        out["guide_id"] = obs[gcol].astype(str) if gcol and gcol in obs else ""
        # target_id (keep raw)
        tcol = omap.get("target_id")
        out["target_id"] = obs[tcol].astype(str) if tcol and tcol in obs else ""
        # perturbation (raw string if present)
        p2 = omap.get("perturbation")
        out["perturbation"] = obs[p2].astype(str) if p2 and p2 in obs else ""
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
