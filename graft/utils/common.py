
from __future__ import annotations
import numpy as np
import pandas as pd

def load_z_and_meta(z_path: str, meta_path: str):
    if z_path.endswith(".parquet"):
        z = pd.read_parquet(z_path)
    elif z_path.endswith(".npz"):
        arr = np.load(z_path)
        if "z" in arr:
            z = pd.DataFrame(arr["z"], index=arr["cell_ids"] if "cell_ids" in arr else None)
        else:
            raise ValueError("npz missing 'z' array")
    else:
        raise ValueError("Unsupported z format")
    meta = pd.read_parquet(meta_path)
    return z, meta
