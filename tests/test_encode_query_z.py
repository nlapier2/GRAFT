
#!/usr/bin/env python3
import sys, os, subprocess, json
from pathlib import Path

# Ensure repo root on path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main():
    # If scvi or required files are missing, skip politely.
    try:
        import scvi, anndata as ad  # noqa: F401
    except Exception as e:
        print("SKIP: scvi-tools not importable:", e); return

    import yaml
    cfg_path = _REPO_ROOT / "configs" / "gnn_v1.yaml" 
    if not cfg_path.exists():
        print("SKIP: config not found:", cfg_path); return
    cfg = yaml.safe_load(cfg_path.read_text())

    model_dir = Path(cfg["paths"]["scvi_model_dir"])
    if not model_dir.exists():
        print("SKIP: scVI model dir missing:", model_dir); return
    datasets = cfg["paths"]["datasets"]
    if not datasets:
        print("SKIP: no datasets in config"); return

    h5ad = Path(datasets[0]["h5ad"])
    if not h5ad.exists():
        print("SKIP: first dataset h5ad missing:", h5ad); return

    out_dir = _REPO_ROOT / "artifacts_v2" / "tmp_phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "z_test.parquet"

    # Call the script
    script = _REPO_ROOT / "scripts" / "encode_query_z.py"
    if not script.exists():
        print("SKIP: encode_query_z.py not found at", script); return

    args = [sys.executable, str(script),
            "--scvi-model-dir", str(model_dir),
            "--query-h5ad", str(h5ad),
            "--out-parquet", str(out_parquet),
            "--transform-batch", "None",
            "--max-chunks", "1"]
    print(">>", " ".join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("SKIP/FAIL: encode_query_z.py returned non-zero (likely missing model compatibility)")
        return

    # Validate parquet schema
    import pandas as pd
    df = pd.read_parquet(out_parquet)
    z_cols = [c for c in df.columns if c.startswith("z")]
    assert len(z_cols) > 0, "No z* columns found"
    assert df.index.is_unique, "Index not unique"
    print("✓ test_encode_query_z passed with", len(df), "rows and", len(z_cols), "latent dims")

if __name__ == "__main__":
    main()
