# utils/scvi_stream.py
# Stream denoised expression x̄ from a saved scVI model for given row indices.

from __future__ import annotations
from typing import Optional, Sequence
import os
import numpy as np
import anndata as ad
import scvi

class ScviOnTheFly:
    def __init__(
        self,
        model_dir: str,
        scvi_input_h5ad: str,
        library_size: float = 1e4,
        transform_batch: Optional[str] = None,
        scvi_forward_batch_size: int = 4096,
    ):
        """
        model_dir: path saved by train_scvi.py (model.save(...))
        scvi_input_h5ad: the same h5ad you trained scVI on
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(model_dir)
        if not os.path.exists(scvi_input_h5ad):
            raise FileNotFoundError(scvi_input_h5ad)

        # Load AnnData in-memory (scVI expects an in-memory manager)
        self.adata = ad.read_h5ad(scvi_input_h5ad)
        self.model = scvi.model.SCVI.load(model_dir, adata=self.adata)
        self.model.eval()

        self.library_size = library_size
        self.transform_batch = transform_batch
        self.forward_bs = scvi_forward_batch_size

        # Cache gene names
        self.genes = self.adata.var_names.to_list()
        self.n_obs = self.adata.n_obs
        self.n_vars = self.adata.n_vars

    def get_xbar(self, indices: Sequence[int], return_numpy: bool = False):
        """
        Fetch denoised expression for specified row indices as float32.
        Returns a NumPy array (default) or the same (you can torch.from_numpy() it).
        """
        X = self.model.get_normalized_expression(
            adata=self.adata,
            indices=np.asarray(indices),
            library_size=self.library_size,
            transform_batch=self.transform_batch,
            n_samples=1,
            batch_size=self.forward_bs,
            return_numpy=True,  # avoids pandas overhead
        )
        X = X.astype(np.float32, copy=False)
        if return_numpy:
            return X
        return X  # you can wrap with torch.from_numpy() at call site

    def align_z_index(self, z_index_like) -> np.ndarray:
        """
        Utility: ensure your z parquet index matches self.adata.obs_names order.
        Pass a pandas.Index of z (or anything with .get_indexer).
        Returns an array of positions to reorder z rows.
        """
        return self.adata.obs_names.get_indexer(z_index_like)
