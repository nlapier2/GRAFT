# scripts/train_scvi.py
# Train scVI and (optionally) dump denoised expression in chunks to avoid OOM.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import anndata as ad
import pandas as pd
import numpy as np
import scvi

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
        )
        # Ensure float32 to save RAM/disk
        X = X.astype(np.float32, copy=False)

        # Cell ids for this chunk
        cell_ids = adata.obs_names[start:end]

        if use_parquet:
            # Write each chunk as its own parquet file
            df = pd.DataFrame(X, index=cell_ids, columns=genes)
            df.to_parquet(os.path.join(parts_dir, f"part_{start:09d}_{end:09d}.parquet"))
        else:
            # Write compressed npz (fast and low-overhead)
            np.savez_compressed(
                os.path.join(parts_dir, f"part_{start:09d}_{end:09d}.npz"),
                X=X, cell_ids=np.array(cell_ids), genes=np.array(genes)
            )

        print(f"[denoised] wrote rows [{start}:{end})")

    print(f"[OK] denoised parts in: {parts_dir}")
    return parts_dir

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to scvi_input_*.h5ad")
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--outdir", default="artifacts", help="Where to save model and outputs")

    # Denoised export options
    ap.add_argument("--save-denoised", action="store_true", help="If set, dump denoised in chunks")
    ap.add_argument("--denoised-chunk-size", type=int, default=10000)
    ap.add_argument("--denoised-format", choices=["parquet","npz"], default="parquet",
                    help="Chunk file format; npz is memory-friendlier")
    ap.add_argument("--library-size", type=float, default=1e4)
    ap.add_argument("--transform-batch", default=None,
                    help="Optional batch name to harmonize across batches")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.input)
    # Ensure only controls are used (safety guard)
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
        # Fallback to batch_id if dataset_id is missing (shouldn't happen with current pipeline)
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
                early_stopping=True, early_stopping_patience=20)

    # Save model
    model.save(model_dir, overwrite=True)

    # Latent
    z = model.get_latent_representation()
    pd.DataFrame(z, index=adata.obs_names).to_parquet(os.path.join(args.outdir, f"scvi_z_{cell_type}.parquet"))

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

    print(f"[OK] saved model to {model_dir} and z to {args.outdir}")

if __name__ == "__main__":
    main()
