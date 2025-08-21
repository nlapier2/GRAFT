
import json, os
import numpy as np
import pandas as pd

def load_index(path="artifacts/cell_index.parquet"):
    return pd.read_parquet(path)

def logo_select_genes(idx: pd.DataFrame, cell_type: str, min_test_cells=100, max_genes=None, seed=13):
    df = idx[(idx["cell_type"] == cell_type) & (~idx["is_control"]) & (idx["target_gene"] != "")]
    counts = df.groupby("target_gene").size().reset_index(name="n").sort_values("n", ascending=False)
    counts = counts[counts["n"] >= min_test_cells]
    genes = counts["target_gene"].tolist()
    if max_genes is not None and len(genes) > max_genes:
        rng = np.random.default_rng(seed)
        genes = list(rng.choice(genes, size=max_genes, replace=False))
    return genes, counts

def resolve_logo(idx: pd.DataFrame, cell_type: str, target_gene: str):
    test_mask = (idx["cell_type"] == cell_type) & (idx["target_gene"] == target_gene)
    train_mask = (idx["cell_type"] == cell_type) & (~test_mask)
    test_idx = np.flatnonzero(test_mask.values)
    train_idx = np.flatnonzero(train_mask.values)
    return train_idx, test_idx

def resolve_holdout_env(idx: pd.DataFrame, cell_type: str, env_key: str = "batch_id", strategy="largest", min_test_cells=200, seed=13):
    sub = idx[idx["cell_type"] == cell_type]
    grp = sub.groupby(env_key).size().reset_index(name="n")
    grp = grp[grp["n"] >= min_test_cells]
    if grp.empty:
        raise ValueError(f"No environment with ≥{min_test_cells} cells for cell_type={cell_type}")
    if strategy == "largest":
        env = grp.sort_values("n", ascending=False)[env_key].iloc[0]
    elif strategy == "random":
        env = grp.sample(1, random_state=seed)[env_key].iloc[0]
    else:
        env = grp[env_key].iloc[0]
    test_mask = (idx["cell_type"] == cell_type) & (idx[env_key] == env)
    train_mask = (idx["cell_type"] == cell_type) & (~test_mask)
    return env, np.flatnonzero(train_mask.values), np.flatnonzero(test_mask.values)
