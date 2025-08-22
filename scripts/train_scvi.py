
# scripts/train_scvi.py
# Train a joint scVI model for a given cell type h5ad produced by make_scvi_input.py.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scvi
import anndata as ad
import pandas as pd

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to scvi_input_*.h5ad")
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--outdir", default="artifacts", help="Where to save model and outputs")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.input)
    cell_type = str(adata.obs["cell_type"].unique().tolist()[0]) if "cell_type" in adata.obs else "UNKNOWN"
    model_dir = os.path.join(args.outdir, f"scvi_{cell_type}")
    os.makedirs(model_dir, exist_ok=True)

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key="batch_id",
        categorical_covariate_keys=[c for c in ["dataset_id","cell_type"] if c in adata.obs]
    )
    model = scvi.model.SCVI(adata, n_latent=args.n_latent, gene_likelihood="nb", dispersion="gene")
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size, early_stopping=True, early_stopping_patience=20)

    # Save model
    model.save(model_dir, overwrite=True)
    # Latent
    z = model.get_latent_representation()
    pd.DataFrame(z, index=adata.obs_names).to_parquet(os.path.join(args.outdir, f"scvi_z_{cell_type}.parquet"))
    # Denoised expression (posterior means)
    den = model.get_normalized_expression(adata=adata, library_size=1e4, transform_batch=None)
    den.to_parquet(os.path.join(args.outdir, f"scvi_denoised_{cell_type}.parquet"))
    print(f"[OK] saved model to {model_dir} and outputs to {args.outdir}")

if __name__ == "__main__":
    main()
