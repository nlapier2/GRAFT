#!/usr/bin/env python3
"""
Phase-0 Sanity Runner (tests/phase0_sanity.py)
==============================================

Run this to verify your environment, paths, and basic module construction.
It is resilient to repo layout by adding the repo root (parent of tests/) to sys.path.

Usage:
  python tests/phase0_sanity.py --config configs/gnn_k562_v1.yaml
"""

from __future__ import annotations
import argparse, os, sys
# temporary workaround for script visibility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

# --- Ensure repo root on sys.path (parent of tests/ directory) ---
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1] if _THIS.name == "phase0_sanity.py" else Path(".").resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/gnn_v1.yaml")
    args = ap.parse_args()

    # 0) Versions / GPU
    print("=== Environment ===")
    import torch, sys as _sys
    print("Python:", _sys.version.split()[0])
    print("Torch :", torch.__version__)
    print("CUDA? :", torch.cuda.is_available(), "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    try:
        import scvi, anndata as ad  # noqa: F401
        print("scvi-tools:", scvi.__version__)
    except Exception as e:
        print("scvi-tools not importable:", e)

    # 1) Ensure artifacts dirs
    print("\n=== Artifacts dirs ===")
    for p in ["artifacts_v2/learned_factor_encoders", "artifacts_v2/gnn_runs"]:
        Path(p).mkdir(parents=True, exist_ok=True)
        print("✓ ensured", p)

    # 2) Validate config / paths
    import yaml
    cfg_path = Path(args.config)
    assert cfg_path.exists(), f"Config file not found: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    print("\n=== Config loaded ===")
    print("Config:", cfg_path)

    def must(path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        return str(p)

    must(cfg["paths"]["scvi_model_dir"])
    must(cfg["paths"]["factor_U"])
    must(cfg["paths"]["index_parquet"])

    # ds_ids = []
    # for d in cfg["paths"]["datasets"]:
    #     ds_ids.append(d["id"])
    #     must(d["z_parquet"])
    #     must(d["h5ad"])
    #     if d.get("knn_parquet"):
    #         print("(optional) knn:", d["knn_parquet"])
    # assert len(ds_ids) == len(set(ds_ids)), f"Duplicate dataset ids: {ds_ids}"
    # print("✓ Dataset ids:", ds_ids)
    # print("✓ All required files present.")

    # 3) Coherence with index parquet
    import pandas as pd
    df_index = pd.read_parquet(cfg["paths"]["index_parquet"])
    print("\n=== Index parquet ===")
    print("Index columns (first 10):", list(df_index.columns)[:10])
    if "dataset_id" in df_index.columns:
        have = set(df_index["dataset_id"].astype(str).unique())
        want = {d["id"] for d in cfg["paths"]["datasets"]}
        print("Datasets in index :", sorted(list(have)))
        print("Datasets in config:", sorted(list(want)))
        missing = want - have
        assert not missing, f"Config refers to dataset ids not in index: {missing}"
        print("✓ Index/config coherence OK.")
    else:
        print("Note: no 'dataset_id' column; train_gnn will rely on per-dataset parquets (OK for v1).")

    # 4) Module init & shape smoke (CPU)
    print("\n=== Module init & shape smoke ===")
    from graft.utils.common import seed_everything
    from graft.models.gnn_core import StatePropagator
    from graft.models.step0 import StepZeroClamp
    from graft.models.heads import MediatedHead, SparseDirectHead

    G = int(cfg["data"]["n_genes"]); F = int(cfg["model"]["mediated"]["F"]); d = int(cfg["model"]["z_dim"])
    seed_everything(13)
    prop = StatePropagator(d,
        hidden=int(cfg["model"]["propagator"].get("hidden", 256)),
        layers=int(cfg["model"]["propagator"].get("layers", 2)),
        steps=int(cfg["model"]["propagator"].get("steps", 2)),
        dropout=float(cfg["model"]["propagator"].get("dropout", 0.0)),
        use_env_film=bool(cfg["model"]["propagator"].get("use_env_film", True)),
        use_target_cond=bool(cfg["model"]["propagator"].get("use_target_cond", True)),
        target_embed_dim=int(cfg["model"]["propagator"].get("target_embed_dim", 32)),
        n_envs=len(cfg["paths"]["datasets"]),
        n_genes=G,
    )
    step0 = StepZeroClamp(d,
        n_labs=len(cfg["paths"]["datasets"]),
        hidden=int(cfg["model"]["step0"].get("hidden", 64)),
        init_eff=float(cfg["model"]["step0"].get("init_eff", 0.9)),
        mode=str(cfg["model"]["step0"].get("mode", "down")),
    )
    head_m = MediatedHead(d, F,
        hidden=int(cfg["model"]["mediated"].get("hidden", 256)),
        use_factor_feats=bool(cfg["model"]["mediated"].get("use_factor_feats", False)),
        a_dim=int(cfg["model"]["mediated"].get("a_dim", 0)),
        nonneg=bool(cfg["model"]["mediated"].get("nonneg", True)),
        dropout=float(cfg["model"]["mediated"].get("dropout", 0.0)),
    )
    head_d = SparseDirectHead(d, G,
        hidden=int(cfg["model"]["direct"].get("hidden", 256)),
        dropout=float(cfg["model"]["direct"].get("dropout", 0.0)),
        bound=None,
    )

    import torch
    B=8
    z=torch.randn(B,d)
    env=torch.zeros(B,dtype=torch.long)
    tgt=torch.full((B,),-1, dtype=torch.long)
    z_ref=prop(z, tgt, env); assert z_ref.shape==(B,d)
    x0=torch.rand(B,G)
    x_clamp, eff = step0(x0, z_ref, env, tgt); assert x_clamp.shape==(B,G) and eff.shape==(B,)
    m = head_m(z_ref); assert m.shape==(B,F)
    dxd = head_d(z_ref); assert dxd.shape==(B,G)
    assert torch.isfinite(z_ref).all() and torch.isfinite(x_clamp).all() and torch.isfinite(dxd).all()
    print("✓ Shapes OK; no NaNs.")

    # 5) Determinism smoke
    print("\n=== Determinism (init/forward under fixed seed) ===")
    seed_everything(42)
    m1 = StatePropagator(d,
        hidden=int(cfg["model"]["propagator"].get("hidden", 256)),
        layers=int(cfg["model"]["propagator"].get("layers", 2)),
        steps=int(cfg["model"]["propagator"].get("steps", 2)),
        dropout=float(cfg["model"]["propagator"].get("dropout", 0.0)),
        use_env_film=bool(cfg["model"]["propagator"].get("use_env_film", True)),
        use_target_cond=bool(cfg["model"]["propagator"].get("use_target_cond", True)),
        target_embed_dim=int(cfg["model"]["propagator"].get("target_embed_dim", 32)),
        n_envs=len(cfg["paths"]["datasets"]),
        n_genes=G,
    )
    z=torch.randn(4,d); env=torch.zeros(4,dtype=torch.long); tgt=torch.full((4,),-1, dtype=torch.long)
    out1 = m1(z, tgt, env).detach().clone()
    seed_everything(42)
    m2 = StatePropagator(d,
        hidden=int(cfg["model"]["propagator"].get("hidden", 256)),
        layers=int(cfg["model"]["propagator"].get("layers", 2)),
        steps=int(cfg["model"]["propagator"].get("steps", 2)),
        dropout=float(cfg["model"]["propagator"].get("dropout", 0.0)),
        use_env_film=bool(cfg["model"]["propagator"].get("use_env_film", True)),
        use_target_cond=bool(cfg["model"]["propagator"].get("use_target_cond", True)),
        target_embed_dim=int(cfg["model"]["propagator"].get("target_embed_dim", 32)),
        n_envs=len(cfg["paths"]["datasets"]),
        n_genes=G,
    )
    out2 = m2(z, tgt, env).detach().clone()
    max_diff = (out1-out2).abs().max().item()
    print("Max abs diff:", max_diff)
    assert torch.allclose(out1, out2), "Non-deterministic init/forward under fixed seed"
    print("✓ Determinism passed.")

    print("\nAll Phase-0 checks passed.")

if __name__ == "__main__":
    main()
