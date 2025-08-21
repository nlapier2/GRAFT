# test holdout environment splits
import json, numpy as np
from utils.split_resolver import load_index, resolve_holdout_env

# temporary workaround for script visibility
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

idx = load_index("artifacts/cell_index.parquet")

spec = json.load(open("splits/holdout_env_spec.json"))
cell_type = spec["cell_types"][0]
env_key = spec["env_key"]

env, tr, te = resolve_holdout_env(
    idx, cell_type=cell_type,
    env_key=env_key, strategy=spec["selection"],
    min_test_cells=spec["min_test_cells"], seed=spec["seed"]
)
import os
os.makedirs("artifacts/splits/holdout_indices", exist_ok=True)
np.save(f"artifacts/splits/holdout_indices/{cell_type}_{env_key}-{env}_train.npy", tr)
np.save(f"artifacts/splits/holdout_indices/{cell_type}_{env_key}-{env}_test.npy", te)
print(f"[OK] {cell_type} holdout {env_key}={env}: train={len(tr)} test={len(te)}")
