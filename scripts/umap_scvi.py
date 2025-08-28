# scripts/umap_scvi.py
# UMAP QC for scVI latents: robust z loader, stratified downsample, mixing metrics, fixed-color legend.

import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from umap import UMAP
from sklearn.neighbors import NearestNeighbors

# ------------------------ I/O helpers ------------------------

def load_obs_backed(h5ad_path: str) -> pd.DataFrame:
    A = ad.read_h5ad(h5ad_path, backed="r")
    cols = [c for c in ["dataset_id","lab_id","batch_id","cell_type","is_control","target_gene"] if c in A.obs.columns]
    obs = A.obs[cols].copy()
    obs.index = A.obs_names.copy()
    del A
    return obs

def load_z_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        # try pyarrow then fastparquet
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_parquet(path, engine="fastparquet")
        return df
    elif ext == ".npz":
        try:
            d = np.load(path, allow_pickle=False)
        except ValueError:
            d = np.load(path, allow_pickle=True)   # fallback for object-dtype cell_ids
        z = d["z"]
        if "cell_ids" in d.files:
            ci = d["cell_ids"]
            # coerce bytes/object → unicode
            if getattr(ci, "dtype", None) is not None and ci.dtype.kind in {"S","O"}:
                idx = pd.Index(np.asarray(ci, dtype="U"))
            else:
                idx = pd.Index(ci.astype("U", copy=False))
        else:
            idx = pd.RangeIndex(z.shape[0])
        cols = [f"z{i}" for i in range(z.shape[1])]
        return pd.DataFrame(z, index=idx, columns=cols)
    elif ext == ".npy":
        z = np.load(path, allow_pickle=False)
        cols = [f"z{i}" for i in range(z.shape[1])]
        return pd.DataFrame(z, index=pd.RangeIndex(z.shape[0]), columns=cols)
    else:
        raise ValueError(f"Unsupported z format: {ext}")

# ------------------------ sampling & metrics ------------------------

def stratified_downsample(df: pd.DataFrame, by=("lab_id","is_control"), max_total=100_000, seed=13) -> pd.DataFrame:
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
    # adjust to exact max_total
    diff = sum(targets) - max_total
    if diff != 0:
        order = np.argsort(sizes)[::-1]
        for i in order:
            if diff == 0: break
            if diff > 0 and targets[i] > 1:
                cut = min(diff, targets[i] - 1)
                targets[i] -= cut; diff -= cut
            elif diff < 0:
                add = min(-diff, sizes[i] - targets[i])
                targets[i] += add; diff += add
    parts = []
    if isinstance(by, (list, tuple)) and len(by) > 0:
        for (_, g), t in zip(groups, targets):
            parts.append(g if len(g) <= t else g.sample(t, random_state=int(rng.integers(0, 1 << 31))))
    else:
        g = df; t = targets[0]
        parts.append(g if len(g) <= t else g.sample(t, random_state=int(rng.integers(0, 1 << 31))))
    return pd.concat(parts, axis=0)

def knn_label_entropy(X: np.ndarray, labels: np.ndarray, n_neighbors: int = 15) -> float:
    """Average entropy of neighbor label distribution (higher = more mixing)."""
    if len(X) <= 2:
        return 0.0
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X)-1))
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    # drop self if present
    if np.all(idx[:, 0] == np.arange(len(X))):
        idx = idx[:, 1:]
    neigh = labels[idx]
    ent = []
    for row in neigh:
        vals, counts = np.unique(row, return_counts=True)
        p = counts / counts.sum()
        ent.append(-(p * np.log(p + 1e-12)).sum())
    return float(np.mean(ent))

# ------------------------ plotting ------------------------

def scatter_with_fixed_legend(emb: np.ndarray,
                              meta: pd.Series,
                              out_png: str,
                              title: str = None,
                              max_legend: int = 16):
    """Scatter using a fixed, repeatable palette and a legend whose marker colors match the plot."""
    # categories & codes
    cats = pd.Categorical(meta.astype(str).values)
    n = len(cats.categories)
    codes = cats.codes

    # palette: repeat tab20 as needed
    base = plt.get_cmap("tab20").colors
    reps = (n + len(base) - 1) // len(base)
    pal = (base * reps)[:n]
    cmap = ListedColormap(pal)

    # plot using integer codes + cmap (fast), ensure range aligns with cmap
    plt.figure(figsize=(8, 7))
    plt.scatter(emb[:, 0], emb[:, 1],
                c=codes, s=1, alpha=0.75,
                cmap=cmap, vmin=-0.5, vmax=n - 0.5,
                linewidths=0)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=12)

    # legend: up to max_legend entries, with matching colors
    shown = cats.categories[:max_legend]
    handles = []
    labels = []
    for i, cat in enumerate(shown):
        handles.append(plt.Line2D([], [], marker="o", linestyle="",
                                  markersize=6, markerfacecolor=pal[i],
                                  markeredgecolor="none"))
        labels.append(str(cat))
    if len(shown) > 0:
        plt.legend(handles=handles, labels=labels, loc="best",
                   fontsize=8, frameon=False, handletextpad=0.4, borderpad=0.2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[save] {out_png}")

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scvi-input", required=True, help="artifacts/scvi_input_<CELL>.h5ad")
    ap.add_argument("--z-path", required=True, help="Latent file (.parquet | .npz | .npy)")
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

    # align
    common = z_df.index.intersection(obs.index)
    if len(common) == 0:
        raise SystemExit("No overlapping cell ids between z and h5ad obs_names.")
    z_df = z_df.loc[common]
    obs = obs.loc[common]

    # stratified downsample (lab & control if present)
    strata_cols = [c for c in ["lab_id", "is_control"] if c in obs.columns]
    sample_obs = stratified_downsample(obs, by=strata_cols, max_total=args.max_cells, seed=args.random_state)
    z = z_df.loc[sample_obs.index].values.astype(np.float32)

    # UMAP
    print(f"[umap] fit on {len(z)} cells (d={z.shape[1]}) ...")
    um = UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
              metric="euclidean", random_state=args.random_state, verbose=True)
    emb = um.fit_transform(z)

    # Mixing metrics
    if "lab_id" in sample_obs.columns:
        lab = sample_obs["lab_id"].astype(str).values
        ent_umap = knn_label_entropy(emb, lab, n_neighbors=15)
        ent_z    = knn_label_entropy(z,   lab, n_neighbors=30)
        print(f"[qc] Lab mixing entropy — UMAP: {ent_umap:.3f}  |  z-space: {ent_z:.3f}")

    # Save CSV of embedding + meta
    csv_path = os.path.join(args.outdir, "umap_embedding.csv")
    out_df = pd.DataFrame({"UMAP1": emb[:, 0], "UMAP2": emb[:, 1]}, index=sample_obs.index)
    for col in ["dataset_id","lab_id","batch_id","cell_type","is_control","target_gene"]:
        if col in sample_obs.columns:
            out_df[col] = sample_obs[col].values
    out_df.to_csv(csv_path)
    print(f"[save] {csv_path}")

    # Plots with fixed legends
    if "lab_id" in sample_obs.columns:
        scatter_with_fixed_legend(emb, sample_obs["lab_id"], os.path.join(args.outdir, "umap_by_lab.png"),
                                  title="UMAP colored by lab_id")
    if "dataset_id" in sample_obs.columns:
        scatter_with_fixed_legend(emb, sample_obs["dataset_id"], os.path.join(args.outdir, "umap_by_dataset.png"),
                                  title="UMAP colored by dataset_id")
    if "cell_type" in sample_obs.columns and sample_obs["cell_type"].nunique() > 1:
        scatter_with_fixed_legend(emb, sample_obs["cell_type"], os.path.join(args.outdir, "umap_by_celltype.png"),
                                  title="UMAP colored by cell_type")

    print("[OK] UMAP QC done.")

if __name__ == "__main__":
    main()
