# test leave one gene out (logo) splits
import json, numpy as np, pandas as pd
from utils.split_resolver import load_index, logo_select_genes, resolve_logo

# temporary workaround for script visibility
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

idx = load_index("artifacts/cell_index.parquet")

spec = json.load(open("splits/logo_spec.json"))
cell_type = spec["cell_types"][0]
genes, counts = logo_select_genes(
    idx, cell_type=cell_type,
    min_test_cells=spec["min_test_cells"],
    max_genes=spec["max_genes"],
    seed=spec["seed"]
)

# Save the gene list used
pd.Series(genes).to_csv(f"artifacts/splits/logo_genes_{cell_type}.txt", index=False, header=False)

# Materialize a few splits for quick testing (or loop all genes)
import os
os.makedirs("artifacts/splits/logo_indices", exist_ok=True)
for g in genes[:5]:
    tr, te = resolve_logo(idx, cell_type, g)
    np.save(f"artifacts/splits/logo_indices/{cell_type}_{g}_train.npy", tr)
    np.save(f"artifacts/splits/logo_indices/{cell_type}_{g}_test.npy", te)
    print(f"[OK] {cell_type} {g}: train={len(tr)} test={len(te)}")
