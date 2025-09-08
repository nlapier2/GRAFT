# scripts/train_scvi.py
# Train scVI and (optionally) dump denoised expression in chunks to avoid OOM.
# Now with robust z writers (parquet/npz/both) and a recode-only mode.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import anndata as ad
import pandas as pd
import numpy as np
import scvi

# --------------------- robust z I/O helpers ---------------------

def _save_z(z: np.ndarray, cell_ids, outdir: str, cell_type: str,
            z_format: str = "both", parquet_name: str = None, npz_name: str = None):
    """
    Save z in the requested format(s).
    - Parquet: write as a table with 'cell_id' column (not index), float32, pyarrow engine.
    - NPZ: write z=float32 and cell_ids as fixed-width unicode (not object).
    """
    os.makedirs(outdir, exist_ok=True)
    if parquet_name is None:
        parquet_name = f"scvi_z_{cell_type}.parquet"
    if npz_name is None:
        npz_name = f"scvi_z_{cell_type}.npz"

    # coerce dtypes
    z = np.asarray(z, dtype=np.float32)
    cell_ids = np.asarray(cell_ids, dtype="U")  # fixed-width unicode, no object dtype

    if z_format in ("parquet", "both"):
        cols = [f"z{i}" for i in range(z.shape[1])]
        df = pd.DataFrame(z, columns=cols)
        df.insert(0, "cell_id", cell_ids)  # explicit column avoids index metadata quirks
        df.to_parquet(os.path.join(outdir, parquet_name), engine="pyarrow", index=False)

    if z_format in ("npz", "both"):
        np.savez(os.path.join(outdir, npz_name), z=z, cell_ids=cell_ids)

def _load_z_any(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load z from parquet/npz/npy robustly and return (z, cell_ids).
    - parquet: expects a 'cell_id' column; if index is present, will fall back to it.
    - npz: expects arrays 'z' and 'cell_ids'; coerces to unicode.
    - npy: returns RangeIndex cell_ids.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            df = pd.read_parquet(path)  # pyarrow
        except Exception:
            df = pd.read_parquet(path, engine="fastparquet")
        if "cell_id" in df.columns:
            cell_ids = df["cell_id"].astype(str).to_numpy()
            Z = df.drop(columns=["cell_id"]).to_numpy(dtype=np.float32, copy=False)
            return Z, cell_ids
        else:
            # fallback: use index if it carries ids
            cell_ids = df.index.astype(str).to_numpy()
            Z = df.to_numpy(dtype=np.float32, copy=False)
            return Z, cell_ids
    elif ext == ".npz":
        try:
            d = np.load(path, allow_pickle=False)
        except ValueError:
            d = np.load(path, allow_pickle=True)  # tolerate older object-dtype dumps
        Z = np.asarray(d["z"], dtype=np.float32)
        if "cell_ids" in d.files:
            ci = d["cell_ids"]
            cell_ids = np.asarray(ci, dtype="U")  # force unicode
        else:
            cell_ids = np.asarray([f"cell_{i}" for i in range(Z.shape[0])], dtype="U")
        return Z, cell_ids
    elif ext == ".npy":
        Z = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        cell_ids = np.asarray([f"cell_{i}" for i in range(Z.shape[0])], dtype="U")
        return Z, cell_ids
    else:
        raise ValueError(f"Unsupported z format: {ext}")

# ---------------- denoised chunk writer (unchanged) ----------------

def _write_denoised_chunks(model, adata, outdir, cell_type, chunk_size=10000, library_size=1e4,
                           transform_batch=None, use_parquet=True):
    """
    Stream get_normalized_expression() in row chunks and write to disk.
    By default writes many parquet parts in: {outdir}/scvi_denoised_{cell_type}_parts/
    """
    os.makedirs(outdir, exist_ok=True)
    parts_dir = os.path.join(outdir, f"scvi_denoised_{cell_type}_parts")
    os.makedirs(parts_dir, exist_ok=True)

    n = adata.n_obs
    genes = adata.var_names.to_list()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx = np.arange(start, end)

        X = model.get_normalized_expression(
            adata=adata,
            indices=idx,
            library_size=library_size,
            transform_batch=transform_batch,
            n_samples=1,
            batch_size=4096,           # GPU forward batch size
            return_numpy=True,         # avoid DataFrame overhead
        ).astype(np.float32, copy=False)

        cell_ids = adata.obs_names[start:end].astype("U")

        if use_parquet:
            df = pd.DataFrame(X, index=cell_ids, columns=genes)
            df.to_parquet(os.path.join(parts_dir, f"part_{start:09d}_{end:09d}.parquet"))
        else:
            np.savez_compressed(
                os.path.join(parts_dir, f"part_{start:09d}_{end:09d}.npz"),
                X=X, cell_ids=np.array(cell_ids), genes=np.array(genes, dtype="U")
            )

        print(f"[denoised] wrote rows [{start}:{end})")

    print(f"[OK] denoised parts in: {parts_dir}")
    return parts_dir

# ------------------------------ main ------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Path to scvi_input_*.h5ad")
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--outdir", default="artifacts", help="Where to save model and outputs")

    # z output options
    ap.add_argument("--z-format", choices=["parquet","npz","both"], default="both",
                    help="Format to save z embeddings (default: both)")
    ap.add_argument("--parquet-name", default=None, help="Optional custom parquet filename for z")
    ap.add_argument("--npz-name", default=None, help="Optional custom npz filename for z")

    # Denoised export options
    ap.add_argument("--save-denoised", action="store_true", help="If set, dump denoised in chunks")
    ap.add_argument("--denoised-chunk-size", type=int, default=10000)
    ap.add_argument("--denoised-format", choices=["parquet","npz"], default="parquet",
                    help="Chunk file format; npz is memory-friendlier")
    ap.add_argument("--library-size", type=float, default=1e4)
    ap.add_argument("--transform-batch", default=None,
                    help="Optional batch name to harmonize across batches")

    # Recode-only mode (no retrain): load an existing z and re-save cleanly
    ap.add_argument("--recode-in", default=None, help="Existing z file (.parquet/.npz/.npy) to recode")
    ap.add_argument("--recode-outdir", default=None, help="Output directory for recoded file(s)")
    ap.add_argument("--recode-celltype", default=None, help="Cell type name for output filenames (e.g., K562)")
    args = ap.parse_args()

    # ---------- Recode-only path ----------
    if args.recode_in is not None:
        if args.recode_outdir is None:
            raise SystemExit("--recode-outdir is required when using --recode-in")
        # Load any existing z and write clean copies
        Z, cell_ids = _load_z_any(args.recode_in)
        ct = args.recode_celltype or "UNKNOWN"
        _save_z(Z, cell_ids, args.recode_outdir, ct, z_format=args.z_format,
                parquet_name=args.parquet_name, npz_name=args.npz_name)
        print(f"[OK] recoded z → {args.recode_outdir} (format={args.z_format})")
        return

    # ---------- Normal training path ----------
    if not args.input:
        raise SystemExit("--input is required unless using --recode-in")

    adata = ad.read_h5ad(args.input)

    # Optional safety: if this input was built as controls-only, keep it that way
    if "is_control" in adata.obs and not bool(adata.obs["is_control"].astype(bool).all()):
        print("[WARN] Non-control cells detected in scVI input; filtering to controls.")
        adata = adata[adata.obs["is_control"].astype(bool)].copy()

    cell_type = str(adata.obs["cell_type"].unique().tolist()[0]) if "cell_type" in adata.obs else "UNKNOWN"
    model_dir = os.path.join(args.outdir, f"scvi_{cell_type}")
    os.makedirs(model_dir, exist_ok=True)

    if "dataset_id" in adata.obs and "batch_id" in adata.obs:
        adata.obs["tech_batch_id"] = (
            adata.obs["dataset_id"].astype(str) + "_" + adata.obs["batch_id"].astype(str)
        ).astype("category")
    else:
        fallback = adata.obs["batch_id"].astype(str) if "batch_id" in adata.obs else pd.Series(["batch0"] * adata.n_obs, index=adata.obs_names)
        adata.obs["tech_batch_id"] = pd.Categorical(fallback)

    covs = [c for c in ["dataset_id", "cell_type", "lab_id"] if c in adata.obs]

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key="tech_batch_id",
        categorical_covariate_keys=covs
    )
    model = scvi.model.SCVI(adata, n_latent=args.n_latent, gene_likelihood="nb", dispersion="gene")
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size,
                early_stopping=True, early_stopping_patience=20, datasplitter_kwargs={"num_workers": 4})

    # Save model
    model.save(model_dir, overwrite=True)

    # Latent
    z_mat = model.get_latent_representation().astype(np.float32, copy=False)
    cell_ids = adata.obs_names.astype("U")
    _save_z(z_mat, cell_ids, args.outdir, cell_type,
            z_format=args.z_format, parquet_name=args.parquet_name, npz_name=args.npz_name)

    # Denoised (optional; chunked)
    if args.save_denoised if hasattr(args, "save_denoised") else args.save_denoised:
        args.save_denoised = args.save_denoised  # no-op
    if args.save_denoised if hasattr(args, "save_denoised") else args.save_denoised:
        pass  # dead guard

    if args.save_denoised:
        use_parquet = (args.denoised_format == "parquet")
        _ = _write_denoised_chunks(
            model=model,
            adata=adata,
            outdir=args.outdir,
            cell_type=cell_type,
            chunk_size=args.denoised_chunk_size,
            library_size=args.library_size,
            transform_batch=args.transform_batch,
            use_parquet=use_parquet,
        )
    else:
        print("[info] Skipping denoised export (use --save-denoised to enable).")

    print(f"[OK] saved model to {model_dir} and z to {args.outdir} (format={args.z_format})")

if __name__ == "__main__":
    main()
