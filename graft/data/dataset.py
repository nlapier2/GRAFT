from __future__ import annotations
from typing import Dict, Optional, Tuple, Iterable

import numpy as np
import pandas as pd


class GraftDataset:
    """
    Container for GRAFT training/validation rows aligned to scVI latents, with
    dataset-centric environment labels for invariance and balanced sampling.

    Expected inputs:
      - df: rows = cells (perturbed + controls). Must contain:
          * 'cell_id' (as a column or be the index)
          * 'dataset_id' (string-like environment label)  [preferred]
            - if missing, we will create a single dummy dataset 'ds0'
          * 'is_control' (bool)  [optional but recommended]
          * 'target_gene' (string gene symbol or None)  [optional]
        You can include other metadata columns; they will be preserved in .df.

      - z: pandas DataFrame of scVI latents with index=cell_id, columns=[z1,...,zd]
      - meta: optional DataFrame indexed by cell_id with extra metadata
      - genes: numpy array of gene symbols in the order used by your model
               (e.g., scVI var_names). Required to build target gene indices.
      - gene_to_idx: optional prebuilt dict[str -> int]. If None, one is built.

    What you get:
      - .df        : dataframe aligned to z (same index order)
      - .z         : float32 array-like (same indexing as .df)
      - .meta      : optional aligned meta
      - .genes     : np.ndarray of gene symbols (length G)
      - .gene_to_idx /.idx_to_gene
      - .datasets  : np.ndarray[str] of dataset labels aligned to rows
      - .dataset_codes : np.ndarray[int] of integer codes (0..E-1)
      - .dataset_to_int / .int_to_dataset : mappings
    """

    def __init__(
        self,
        df: pd.DataFrame,
        z: pd.DataFrame,
        meta: Optional[pd.DataFrame],
        genes: np.ndarray,
        gene_to_idx: Optional[Dict[str, int]] = None,
        dataset_col: str = "dataset_id",
        control_col: str = "is_control",
        target_col: str = "target_gene",
    ):
        # --- Align df to z by cell_id ---
        if "cell_id" in df.columns:
            df = df.set_index("cell_id", drop=True)
        # keep intersection and preserve z's order after intersection
        inter = z.index.intersection(df.index)
        if len(inter) == 0:
            raise ValueError("No overlapping cell_ids between df and z.")
        z = z.loc[inter]
        df = df.loc[inter].copy()
        if meta is not None:
            if not meta.index.is_unique:
                meta = meta[~meta.index.duplicated(keep="first")]
            meta = meta.loc[inter] if set(inter).issubset(set(meta.index)) else meta.reindex(inter)

        # --- Environment: dataset-centric labels ---
        if dataset_col in df.columns:
            datasets = df[dataset_col].astype(str).fillna("dsNA").values
        else:
            # fallback single environment
            datasets = np.array(["ds0"] * len(df), dtype=str)
            df[dataset_col] = datasets  # also store in df for consistency
        uniq_ds = np.unique(datasets)
        ds_to_int = {ds: i for i, ds in enumerate(uniq_ds)}
        ds_codes = np.array([ds_to_int[d] for d in datasets], dtype=np.int64)
        int_to_ds = {v: k for k, v in ds_to_int.items()}

        # --- Controls & targets (optional columns) ---
        if control_col in df.columns:
            is_control = df[control_col].astype(bool).values
        else:
            is_control = np.zeros(len(df), dtype=bool)  # assume perturbed if not provided
            df[control_col] = is_control

        if target_col in df.columns:
            target_series = df[target_col]
        else:
            target_series = pd.Series([None] * len(df), index=df.index, name=target_col)
            df[target_col] = target_series

        # --- Gene mapping ---
        genes = np.asarray(genes)
        if genes.ndim != 1:
            raise ValueError("genes must be a 1D array of gene symbols in model order.")
        if gene_to_idx is None:
            gene_to_idx = {g: i for i, g in enumerate(genes)}
        idx_to_gene = {i: g for g, i in gene_to_idx.items()}

        # build integer target indices (-1 for controls or missing/unknown genes)
        target_idx = self._encode_targets(target_series, gene_to_idx, is_control)

        # --- Store ---
        self.df: pd.DataFrame = df
        self.z: pd.DataFrame = z.astype("float32")
        self.meta: Optional[pd.DataFrame] = meta
        self.genes: np.ndarray = genes
        self.gene_to_idx: Dict[str, int] = gene_to_idx
        self.idx_to_gene: Dict[int, str] = idx_to_gene

        self.datasets: np.ndarray = datasets
        self.dataset_codes: np.ndarray = ds_codes
        self.dataset_to_int: Dict[str, int] = ds_to_int
        self.int_to_dataset: Dict[int, str] = int_to_ds

        self.is_control: np.ndarray = is_control
        self.target_idx: np.ndarray = target_idx  # shape (N,), int, -1 for control

    # ------------------------------
    # Convenience properties & views
    # ------------------------------
    @property
    def cell_ids(self) -> np.ndarray:
        return self.df.index.values

    @property
    def n_cells(self) -> int:
        return self.df.shape[0]

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def n_datasets(self) -> int:
        return len(self.dataset_to_int)

    # ------------------------------
    # Splits & grouping
    # ------------------------------
    def split_by_dataset(self) -> Dict[str, np.ndarray]:
        """
        Return a dict: dataset_id -> row indices (np.ndarray[int]) for that dataset.
        Useful for dataset-balanced samplers and per-dataset losses.
        """
        uniq = np.unique(self.datasets)
        idx = np.arange(self.n_cells, dtype=int)
        return {ds: idx[self.datasets == ds] for ds in uniq}

    # ------------------------------
    # Target encoding
    # ------------------------------
    @staticmethod
    def _encode_targets(
        target_series: pd.Series,
        gene_to_idx: Dict[str, int],
        is_control: np.ndarray,
    ) -> np.ndarray:
        """
        Map target gene symbols to integer indices; controls -> -1.
        Missing/unknown targets are also set to -1.
        """
        out = np.full(len(target_series), -1, dtype=np.int64)
        # nothing to do if all controls
        if (is_control is not None) and is_control.all():
            return out
        # otherwise fill for perturbed rows
        for i, (tg, is_ctrl) in enumerate(zip(target_series.values, is_control)):
            if is_ctrl:
                continue
            if tg is None or (isinstance(tg, float) and np.isnan(tg)):
                continue
            tg_str = str(tg)
            out[i] = gene_to_idx.get(tg_str, -1)
        return out

    # ------------------------------
    # Mini-batch assembly helper
    # ------------------------------
    def batch_dict(self, row_idx: np.ndarray) -> Dict[str, object]:
        """
        Build a minimal batch dict used by the training loop.

        Returns dict with:
          - 'z'            : np.ndarray (B, d)
          - 'dataset_codes': np.ndarray (B,) int
          - 'is_control'   : np.ndarray (B,) bool
          - 'target_idx'   : np.ndarray (B,) int
          - 'cell_ids'     : np.ndarray (B,)
          - 'meta'         : optional DataFrame slice (B, K)
        """
        row_idx = np.asarray(row_idx, dtype=int)
        z_b = self.z.values[row_idx]
        ds_b = self.dataset_codes[row_idx]
        ctrl_b = self.is_control[row_idx]
        t_idx_b = self.target_idx[row_idx]
        cells_b = self.cell_ids[row_idx]
        meta_b = self.meta.iloc[row_idx] if self.meta is not None else None
        return {
            "z": z_b,
            "dataset_codes": ds_b,
            "is_control": ctrl_b,
            "target_idx": t_idx_b,
            "cell_ids": cells_b,
            "meta": meta_b,
        }
