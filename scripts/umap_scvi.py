# scripts/umap_scvi.py
# UMAP QC for scVI latents: robust loader (parquet/npz/npy), stratified downsample, lab-mixing score.

import os, sys, argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.neighbors import NearestNeighbors

def load_obs_backed(h5ad_path):
    A = ad.read_h5ad(h5ad_path, backed="r")
    cols = [c for c in ["dataset_id","lab_id","batch_id","cell_type","is_control","target_gene"] if c in A.obs.columns]
    obs = A.obs[cols].copy()
    obs.index = A.obs_names.copy()
    del A
    return obs

def load_z_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        # try pyarrow, then fastparquet
        try:
            df = pd.read_parquet(path)  # defaults to pyarrow
        except Exception as e1:
            try:
                df = pd.read_parquet(path, engine="fastparquet")
            except Exception as e2:
                raise RuntimeError(f"Failed to read parquet with both pyarrow and fastparquet.\npyarrow: {e1}\nfastparquet: {e2}")
        # assume index=cell_id
        return df
    elif ext == ".npz":
        d = np.load(path, allow_pickle=False)
        z = d["z"]
        if "cell_ids" in d:
            idx = pd.Index(d["cell_ids"].astype(str))
        else:
            # fallback: synthetic index
            idx = pd.Index([f"cell_{i}" for i in range(z.shape[0])])
        cols = [f"z{i}" for i in range(z.shape[1])]
        return pd.DataFrame(z, index=idx, columns=cols)
    elif ext == ".npy":
        z = np.load(path, allow_pickle=False)
        cols = [f"z{i}" for i in range(z.shape[1])]
        idx = pd.RangeIndex(z.shape[0])
        return pd.DataFrame(z, index=idx, columns=cols)
    else:
        raise ValueError(f"Unsupported z format: {ext}")

def stratified_downsample(df, by=("lab_id","is_control"), max_total=100_000, seed=13):
    rng = np.random.default_rng(seed)
    if max_total is None or len(df) <= max_total:
        return df
    if isinstance(by, (list, tuple)) and len(by) > 0:
        groups = df.groupby(list(by), sort=False)
    else:
        groups = [(None, df)]
    sizes = [len(g) for _, g in groups]
    total = float(sum(sizes))
    targets = [max(1, int(round(max_total * (s/total)))) for s in sizes]
    diff = sum(targets) - max_total
    if diff != 0:
        order = np.argsort(sizes)[::-1]
        for idx in order:
            if diff == 0: break
            if diff > 0 and targets[idx] > 1:
                cut = min(diff, targets[idx] - 1)
                targets[idx] -= cut; diff -= cut
            elif diff < 0:
                add = min(-diff, sizes[idx] - targets[idx])
                targets[idx] += add; diff += add
    parts = []
    if isinstance(by, (list, tuple)) and len(by) > 0:
        for (k, g), t in zip(groups, targets):
            parts.append(g if len(g) <= t else g.sample(t, random_state=int(rng.integers(0, 1<<31))))
    else:
        g = df
        t = targets[0]
        parts.append(g if len(g) <= t else g.sample(t, random_state=int(rng.integers(0, 1<<31))))
    return pd.concat(parts, axis=0)

def knn_label_entropy(emb, labels, n_neighbors=15):
    if len(emb) <= 2: return 0.0
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(emb)-1))
    nn.fit(emb)
    dists, idx = nn.kneighbors(emb, return_distance=True)
    if idx.shape[1] > 0 and np.all(idx[:,0] == np.arange(len(emb))):
        idx = idx[:,1:]
    neigh_labels = labels[idx]
    ent = []
    for row in neigh_labels:
        vals, counts = np.unique(row, return_counts=True)
        p = counts / counts.sum()
        ent.append(-(p * np.log(p + 1e-12)).sum())
    return float(np.mean(ent))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scvi-input", required=True)
    ap.add_argument("--z-path", required=True, help="z file: .parquet or .npz/.npy")
    ap.add_argument("--outdir", default="artifacts/scvi_qc")
    ap.add_argument("--max-cells", type=int, default=100_000)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--random-state", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[load] obs (backed) ...")
    obs = load_obs_backed(args.scvi_input)

    print(f"[load] z from {args.z_path} ...")
    z_df = load_z_any(args.z_path)

    # align indices
    common = z_df.index.intersection(obs.index)
    if len(common) == 0:
        raise SystemExit("No overlapping cell ids between z and h5ad obs_names. If using .npz without cell_ids, you must match ordering.")
    z_df = z_df.loc[common]
    obs = obs.loc[common]

    # Downsample by lab & control (if present)
    by = [c for c in ["lab_id","is_control"] if c in obs.columns]
    sample_obs = stratified_downsample(obs, by=by, max_total=args.max_cells, seed=args.random_state)
    z = z_df.loc[sample_obs.index].values.astype(np.float32)

    print(f"[umap] fit on {len(z)} cells (d={z.shape[1]}) ...")
    um = UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
              metric="euclidean", random_state=args.random_state, verbose=True)
    emb = um.fit_transform(z)

    # lab mixing
    mix_msg = ""
    if "lab_id" in sample_obs.columns:
        mix_score = knn_label_entropy(emb, sample_obs["lab_id"].astype(str).values, n_neighbors=15)
        mix_msg = f"(kNN lab-mixing entropy={mix_score:.3f}, higher=more mixed)"
        print("[qc] " + mix_msg)

    # Save CSV
    csv_path = os.path.join(args.outdir, "umap_embedding.csv")
    out_df = pd.DataFrame({"UMAP1": emb[:,0], "UMAP2": emb[:,1]}, index=sample_obs.index)
    for col in ["dataset_id","lab_id","batch_id","cell_type","is_control","target_gene"]:
        if col in sample_obs.columns:
            out_df[col] = sample_obs[col].values
    out_df.to_csv(csv_path)
    print(f"[save] wrote {csv_path}")

    # Plots
    def scatter_color_by(column, fname, title=None):
        vals = sample_obs[column].astype(str).values
        cats, inv = np.unique(vals, return_inverse=True)
        plt.figure(figsize=(8, 7))
        plt.scatter(emb[:,0], emb[:,1], c=inv, s=1, alpha=0.7, linewidths=0)
        plt.axis("off")
        ttl = title or f"UMAP colored by {column}"
        if mix_msg and column == "lab_id":
            ttl += f"\n{mix_msg}"
        plt.title(ttl, fontsize=12)
        shown = cats[:12]
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, label=c) for c in shown]
        plt.legend(handles=handles, labels=shown.tolist(), loc="best", fontsize=8, frameon=False)
        path = os.path.join(args.outdir, fname)
        plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()
        print(f"[save] {path}")

    if "lab_id" in sample_obs.columns:    scatter_color_by("lab_id", "umap_by_lab.png")
    if "dataset_id" in sample_obs.columns: scatter_color_by("dataset_id", "umap_by_dataset.png")
    if "cell_type" in sample_obs.columns and sample_obs["cell_type"].nunique() > 1:
        scatter_color_by("cell_type", "umap_by_celltype.png")

    print("[OK] UMAP QC done.")

if __name__ == "__main__":
    main()
