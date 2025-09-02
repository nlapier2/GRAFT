
#!/usr/bin/env python3
import sys, os, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root on path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main():
    tmp = _REPO_ROOT / "artifacts_v2" / "tmp_phase1"
    tmp.mkdir(parents=True, exist_ok=True)
    ds_id = "ToyDS"

    # Build a tiny z parquet with obvious neighbors
    # cells: c0..c5; controls = c0,c1,c2; perturbed = c3,c4,c5
    idx = [f"c{i}" for i in range(6)]
    Z = np.array([
        [0.0, 0.0],  # c0 control near perturbed c3
        [5.0, 5.0],  # c1 control near perturbed c4
        [9.0, 9.0],  # c2 control near perturbed c5
        [0.1, 0.1],  # c3 pert -> NN should be c0
        [4.9, 5.1],  # c4 pert -> NN should be c1
        [9.2, 8.9],  # c5 pert -> NN should be c2
    ], dtype=np.float32)
    df_z = pd.DataFrame(Z, index=pd.Index(idx, name="cell_id"), columns=["z0","z1"])
    z_path = tmp / "toy_z.parquet"
    df_z.to_parquet(z_path)

    # Index parquet with control flags and dataset_id
    df_idx = pd.DataFrame({
        "dataset_id": [ds_id]*6,
        "is_control": [True, True, True, False, False, False],
    }, index=df_z.index)
    idx_path = tmp / "toy_index.parquet"
    df_idx.to_parquet(idx_path)

    out_path = tmp / "toy_knn.parquet"
    script = _REPO_ROOT / "scripts" / "knn_matcher.py"
    if not script.exists():
        print("SKIP: knn_matcher.py not found at", script); return

    args = [sys.executable, str(script),
            "--z-parquet", str(z_path),
            "--index-parquet", str(idx_path),
            "--out-parquet", str(out_path),
            "--k", "1",
            "--max-perturbed", "1000",]
    print(">>", " ".join(args))
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("FAIL: knn_matcher.py returned non-zero"); sys.exit(1)

    # Validate outputs
    out = pd.read_parquet(out_path)
    assert "ctrl_id" in out.columns, "Missing ctrl_id in output"
    # Only perturbed should appear
    assert set(out.index) == set(["c3","c4","c5"]), "Output index should be perturbed cells only"
    # Check matches
    assert out.loc["c3","ctrl_id"] == "c0"
    assert out.loc["c4","ctrl_id"] == "c1"
    assert out.loc["c5","ctrl_id"] == "c2"
    print("✓ test_knn_matcher passed")

if __name__ == "__main__":
    main()
