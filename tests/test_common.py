
#!/usr/bin/env python3
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root on path (parent of tests/)
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graft.utils.common import read_parquet_indexed, encode_categories, seed_everything

def test_parquet_roundtrip(tmp_path: Path = None):
    tmp_dir = tmp_path if tmp_path else Path("artifacts_v2/tmp_phase1")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Create toy dataframe with index & mixed dtypes
    idx = pd.Index([f"cell_{i}" for i in range(5)], name="cell_id")
    df = pd.DataFrame({
        "a_float": np.arange(5, dtype=np.float32) * 0.5,
        "b_int":   np.arange(5, dtype=np.int64),
        "c_str":   ["x","y","x","z","x"],
    }, index=idx)
    pq_path = tmp_dir / "roundtrip.parquet"
    df.to_parquet(pq_path)
    df2 = read_parquet_indexed(str(pq_path))
    # Index preserved & equal
    assert df2.index.equals(df.index)
    # Columns match
    assert list(df2.columns) == list(df.columns)
    # Values equal (dtype tolerant)
    pd.testing.assert_frame_equal(df2, df, check_dtype=False)

def test_encode_categories():
    labels = ["A","B","A","C","B","A"]
    codes, mapping = encode_categories(labels)
    # codes should be ints 0..k-1
    assert codes.dtype.kind in "iu"
    assert set(mapping.keys()) == {"A","B","C"}
    # round-trip
    inv = {v:k for k,v in mapping.items()}
    recovered = [inv[int(c)] for c in codes]
    assert recovered == labels

if __name__ == "__main__":
    test_parquet_roundtrip()
    test_encode_categories()
    print("✓ test_common passed")
