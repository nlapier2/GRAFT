
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

class GraftDataset:
    """
    Minimal container for training/validation rows.

    Expected columns in df:
      - cell_id (index or column)
      - lab_id, dataset_id, tech_batch_id
      - is_control (bool)
      - target_gene (str or None)
      - optional: z_* columns with scVI latents (or we merge with external parquet)
    """
    def __init__(self, df: pd.DataFrame, z: pd.DataFrame, meta: pd.DataFrame, genes: np.ndarray):
        # align to z index
        if "cell_id" in df.columns:
            df = df.set_index("cell_id")
        df = df.loc[df.index.intersection(z.index)].copy()
        z = z.loc[df.index]
        self.df = df
        self.z = z.astype("float32")
        self.meta = meta.loc[self.z.index] if (meta is not None and set(meta.index)==set(self.z.index)) else meta
        self.genes = genes  # gene symbols order for outputs

    @property
    def labs(self):
        return self.df["lab_id"].astype(str).values if "lab_id" in self.df.columns else np.array(["lab0"] * len(self.df))

    def split_by_lab(self) -> Dict[str, np.ndarray]:
        labs = self.labs
        uniq = np.unique(labs)
        idx = np.arange(len(self.df))
        return {lab: idx[labs == lab] for lab in uniq}
