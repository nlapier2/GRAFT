# graft/data/dataset.py
# Streamed GRAFT dataset that materializes chunks from raw H5AD,
# computes scVI z / xbar on the fly, and matches controls via a prebuilt ANN.
from __future__ import annotations

import json
import yaml
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scvi
from scipy import sparse

import time

# Optional ANN backends
_HAS_FAISS = False
_HAS_HNSW = False
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    pass

try:
    import hnswlib  # type: ignore
    _HAS_HNSW = True
except Exception:
    pass

from graft.utils.chunk_preprocess import preprocess_chunk, load_gene_list

# Define the exact columns to keep in the final obs table
FINAL_OBS_COLS = [
        "dataset_id", "cell_id", "lab_id", "batch_id", "cell_type",
        "is_control", "pert_type", "target_gene", "guide_id",
        "target_id", "perturbation"
]


# ----------------------------- Control ANN loader ----------------------------- #

@dataclass
class ControlANN:
    """Wrapper that loads a prebuilt control ANN + metadata and exposes query().
    Requires:
      - control index dir with: knn.index, ctrl_ids.parquet, knn_meta.json
      - control Z npz (same used to build the index) to fetch z_ctrl by label.
      - index_parquet (global) to map control cell_id -> dataset_id (for filters).
    """
    backend: str
    metric: str
    dim: int
    index: object
    ctrl_ids: np.ndarray  # shape (N,)
    z_ctrl: np.ndarray    # shape (N, d)
    ctrl_ds_lookup: Dict[str, str]  # cell_id -> dataset_id

    @staticmethod
    def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    @classmethod
    def load(
        cls,
        control_index_dir: str,
        control_z_npz: str,
        index_parquet: str,
    ) -> "ControlANN":
        """ Load ANN index, control IDs from parquet, and control Z from scVI NPZ """
        meta_path = os.path.join(control_index_dir, "knn_meta.json")
        ids_path = os.path.join(control_index_dir, "ctrl_ids.parquet")
        idx_path = os.path.join(control_index_dir, "knn.index")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        backend = meta.get("backend")
        metric = meta.get("metric", "cosine")
        dim = int(meta.get("dim", 0))

        # Load index
        if backend == "faiss":
            if not _HAS_FAISS:
                raise RuntimeError("FAISS not available but control index backend is faiss.")
            index = faiss.read_index(idx_path)  # type: ignore
        elif backend == "hnswlib":
            if not _HAS_HNSW:
                raise RuntimeError("hnswlib not available but control index backend is hnswlib.")
            import hnswlib  # type: ignore
            meta_n_items = int(meta.get("n_items", 0))
            space = "cosine" if metric == "cosine" else "l2"
            p = hnswlib.Index(space=space, dim=dim)
            p.load_index(idx_path, max_elements=meta_n_items)
            index = p
        else:
            raise ValueError(f"Unknown ANN backend in meta: {backend}")

        # Control IDs order must match the order used to add to the index
        ctrl_ids_df = pd.read_parquet(ids_path)
        if "cell_id" not in ctrl_ids_df.columns:
            raise ValueError("ctrl_ids.parquet must contain a 'cell_id' column.")
        ctrl_ids = ctrl_ids_df["cell_id"].astype(str).values

        # Load z_ctrl from the NPZ produced by train_scvi.py
        zfile = np.load(control_z_npz, allow_pickle=False)
        Z = np.asarray(zfile["z"], dtype=np.float32, order="C")
        if Z.shape[0] != ctrl_ids.shape[0]:
            raise ValueError(f"z rows ({Z.shape[0]}) != ctrl_ids rows ({ctrl_ids.shape[0]}). "
                             "Ensure NPZ is the same used to build the index.")
        if Z.shape[1] != dim:
            # tolerate meta dim mismatch if possible
            dim = Z.shape[1]

        # Lookup control dataset per cell_id for post-filtering
        idx = pd.read_parquet(index_parquet)[["cell_id", "dataset_id"]].copy()
        idx["cell_id"] = idx["cell_id"].astype(str)
        ctrl_ds_lookup = dict(zip(idx["cell_id"].values, idx["dataset_id"].astype(str).values))

        return cls(
            backend=backend,
            metric=metric,
            dim=dim,
            index=index,
            ctrl_ids=ctrl_ids,
            z_ctrl=Z,
            ctrl_ds_lookup=ctrl_ds_lookup,
        )

    def query(
        self,
        z_query: np.ndarray,             # (B, d)
        k: int = 16,
        match_dataset: Optional[str] = None,  # if provided, keep same-dataset controls when possible
        oversample: int = 5,             # pull k*oversample, then filter
        caliper: Optional[float] = None, # optional distance threshold (after normalization if cosine)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given query z for perturbed cells, find k nearest control cells.
        Returns (ctrl_idx, ctrl_ids, ctrl_dist) for each query row.
        ctrl_idx are integer labels into self.z_ctrl / self.ctrl_ids.
        """
        Q = z_query.astype(np.float32, copy=False)
        if self.metric == "cosine":
            Q = self.l2_normalize_rows(Q)

        K0 = max(k * max(1, oversample), k)

        if self.backend == "faiss":
            D, I = self.index.search(Q, K0)  # type: ignore
        else:
            # hnswlib
            labels, distances = self.index.knn_query(Q, k=K0)  # type: ignore
            I, D = labels, distances

        B = I.shape[0]
        out_idx = np.full((B, k), -1, dtype=np.int64)
        out_dist = np.full((B, k), np.inf, dtype=np.float32)

        # Post-filter by dataset (and caliper) if requested
        for i in range(B):
            cand_idx = I[i]
            cand_dist = D[i]
            keep = []
            for j, lab in enumerate(cand_idx):
                if lab < 0:
                    continue
                cid = self.ctrl_ids[lab]
                if (match_dataset is not None) and (self.ctrl_ds_lookup.get(cid) != match_dataset):
                    continue
                if (caliper is not None) and (cand_dist[j] > caliper):
                    continue
                keep.append((lab, cand_dist[j]))
                if len(keep) >= k:
                    break

            # Fallback: if too few, relax dataset constraint
            if len(keep) < k and match_dataset is not None:
                for j, lab in enumerate(cand_idx):
                    if lab < 0:
                        continue
                    cid = self.ctrl_ids[lab]
                    if (caliper is not None) and (D[i, j] > caliper):
                        continue
                    keep.append((lab, D[i, j]))
                    if len(keep) >= k:
                        break

            if keep:
                labs, dists = zip(*keep[:k])
                out_idx[i, :len(labs)] = np.array(labs, dtype=np.int64)
                out_dist[i, :len(dists)] = np.array(dists, dtype=np.float32)

        # Map to ctrl_ids for convenience
        out_ids = self.ctrl_ids[out_idx.clip(min=0)]  # bogus rows will index 0; trainer should mask -1 if needed
        return out_idx, out_ids, out_dist

# ------------------------------ Streaming dataset --------------------------- #

@dataclass
class GraftStreamingConfig:
    datasets_yaml: str
    index_parquet: str
    gene_list_tsv: str
    scvi_model_dir: str
    scvi_input_h5ad: str
    control_index_dir: str
    control_z_npz: str

    batch_size: int = 2048
    chunk_size: int = 50_000
    k_controls: int = 16
    oversample: int = 5
    match_within: str = "dataset"  # "dataset" | "none"
    forward_batch_size: int = 4096
    include_controls_in_query: bool = False  # usually False (query = perturbed only)


class GraftStreamingDataset:
    """Streams mini-batches across datasets with on-the-fly scVI and ANN control matching.

    Usage:
      ds = GraftStreamingDataset(cfg)
      for batch in ds.iter_batches(dataset_order):
          train_step(batch)
    """
    def __init__(self, cfg: GraftStreamingConfig):
        self.cfg = cfg
        # Genes and mapping
        self.gene_list: List[str] = load_gene_list(cfg.gene_list_tsv)
        self.gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(self.gene_list)}

        # Index parquet (authoritative metadata)
        idx_all = pd.read_parquet(cfg.index_parquet)
        idx_all["cell_id"] = idx_all["cell_id"].astype(str)
        idx_all["dataset_id"] = idx_all["dataset_id"].astype(str)

        # Load the reference AnnData used for scVI training first to get valid datasets
        print(f"[info] Loading reference AnnData for scVI: {self.cfg.scvi_input_h5ad}")
        self.scvi_reference_adata = ad.read_h5ad(cfg.scvi_input_h5ad)
        
        # Get the set of datasets actually present in the scVI reference file.
        valid_scvi_datasets = set(self.scvi_reference_adata.obs['dataset_id'].astype(str).unique())
        
        # Filter the main index to include ONLY datasets known to scVI.
        original_count = len(idx_all['dataset_id'].unique())
        idx = idx_all[idx_all["dataset_id"].isin(valid_scvi_datasets)].copy()
        filtered_count = len(idx['dataset_id'].unique())
        
        if filtered_count < original_count:
            print(f"[warn] Filtered index from {original_count} to {filtered_count} datasets "
                  f"to match scvi reference AnnData.")
        if idx.empty:
            raise ValueError("No overlapping datasets found between index_parquet and scvi_reference_adata.")
        
        # Proceed with the filtered index from now on
        self.index = idx

        # Target gene → index mapping (best-effort; absent targets → -1)
        self.target_col = "target_gene" if "target_gene" in idx.columns else ("target" if "target" in idx.columns else None)
        self._target_map_cache: Dict[str, int] = {}

        # Restrict query pool per dataset
        if not cfg.include_controls_in_query and "is_control" in idx.columns:
            self.query_pool = idx[~idx["is_control"].astype(bool)].copy()
        else:
            self.query_pool = idx.copy()

        # Per-dataset tables
        self.by_ds: Dict[str, pd.DataFrame] = {
            dsid: g.copy() for dsid, g in self.query_pool.groupby("dataset_id")
        }

        # Read datasets.yaml (mapping)
        with open(cfg.datasets_yaml, "r") as f:
            y = json.loads(json.dumps(_yaml_safe_load(f.read())))
        ds_map = y.get("datasets", {})
        self.ds_paths: Dict[str, str] = {str(k): str(v["raw_path"]) for k, v in ds_map.items() if isinstance(v, dict) and "raw_path" in v}
        
        # Cache for initialized scVI models, one per dataset
        self.scvi_model_cache: Dict[str, scvi.model.SCVI] = {}

        # Load the reference AnnData for scvi model loading once
        print(f"[info] Loading reference AnnData for scVI: {self.cfg.scvi_input_h5ad}")
        self.scvi_reference_adata = ad.read_h5ad(self.cfg.scvi_input_h5ad)

        # Ensure the reference adata has the tech_batch_id scVI was trained on.
        # This makes the loader robust to older scvi_input files.
        if "tech_batch_id" not in self.scvi_reference_adata.obs.columns:
            print("[warn] 'tech_batch_id' not found in scvi_reference_adata.obs. Creating it on the fly.")
            obs = self.scvi_reference_adata.obs
            if "dataset_id" in obs.columns and "batch_id" in obs.columns:
                obs["tech_batch_id"] = (
                    obs["dataset_id"].astype(str) + "_" + obs["batch_id"].astype(str)
                ).astype("category")
            else:
                raise KeyError("Cannot create 'tech_batch_id'; missing 'dataset_id' or 'batch_id' in reference adata.obs.")

        # Control ANN
        self.ann = ControlANN.load(cfg.control_index_dir, cfg.control_z_npz, cfg.index_parquet)

        # Env coding (dataset → int)
        self.env_codes: Dict[str, int] = {dsid: i for i, dsid in enumerate(sorted(self.by_ds.keys()))}

    def get_dataset_ids(self) -> List[str]:
        return list(self.by_ds.keys())

    # ---- iteration ---- #

    def iter_batches(self, dataset_sequence: Iterable[str]) -> Iterator[Dict[str, np.ndarray]]:
        """Yield batches in the order specified by dataset_sequence (e.g., round-robin)."""
        for dsid in dataset_sequence:
            if dsid not in self.by_ds or dsid not in self.ds_paths:
                continue
            raw_path = self.ds_paths[dsid]
            rows_meta = self.by_ds[dsid]
            if rows_meta.empty:
                continue
            
            # Check cache for an initialized scVI model.
            # The model object is now shared across all datasets.
            if "model" not in self.scvi_model_cache:
                print(f"[info] Initializing shared scVI model.")
                # Load the base scVI model, providing the reference AnnData it was trained on.
                # This is the crucial fix for the ValueError.
                model = scvi.model.SCVI.load(
                    self.cfg.scvi_model_dir, 
                    adata=self.scvi_reference_adata
                )
                model.module.eval()
                # Store the initialized, ready-to-use model in the cache
                self.scvi_model_cache["model"] = model
            
            # Retrieve the cached model
            qmodel = self.scvi_model_cache["model"]

            # Stream the raw H5AD in chunks
            A_b = ad.read_h5ad(raw_path, backed="r")
            n_obs = A_b.n_obs
            cs = int(self.cfg.chunk_size)

            for start in range(0, n_obs, cs):
                t0 = time.time()
                end = min(start + cs, n_obs)
                # Materialize + align + filter this chunk
                A_chunk = preprocess_chunk(
                    A_backed=A_b,
                    row_index=slice(start, end),
                    gene_list=self.gene_list,
                    dataset_id=dsid,
                    index_df=rows_meta,
                    counts_layer=None,
                    keep_cols=FINAL_OBS_COLS,
                )
                if A_chunk.n_obs == 0:
                    continue
                t1 = time.time()
                print(f"[profile] Chunk Preprocessing: {t1 - t0:.4f} sec")
                
                # Use the single cached model to encode/decode the new data chunk
                t2 = time.time()
                z_chunk = qmodel.get_latent_representation(A_chunk, batch_size=self.cfg.forward_batch_size)
                xbar_chunk = qmodel.get_normalized_expression(A_chunk, batch_size=self.cfg.forward_batch_size, n_samples=1, library_size=1e4)
                if not isinstance(xbar_chunk, np.ndarray):
                    xbar_chunk = xbar_chunk.to_numpy()
                t3 = time.time()
                print(f"[profile] Query Cell Decoding (z+xbar): {t3 - t2:.4f} sec")

                # Map per-row targets
                tgt_idx_chunk = self._targets_for_ids(A_chunk.obs_names.astype(str))

                # Match controls for the ENTIRE CHUNK at once
                t4 = time.time()
                match_ds = dsid if (self.cfg.match_within == "dataset") else None
                ctrl_idx_chunk, ctrl_ids_chunk, ctrl_dist_chunk = self.ann.query(
                    z_chunk, k=self.cfg.k_controls, match_dataset=match_ds, oversample=self.cfg.oversample
                )
                z_ctrl_chunk = self.ann.z_ctrl[ctrl_idx_chunk.clip(min=0)]
                t5 = time.time()
                print(f"[profile] ANN Query: {t5 - t4:.4f} sec")

                # Fetch the normalized expression (xbar) for the matched control cells.
                # We need to get unique indices and then map them back to the batch shape.
                t6 = time.time()
                unique_ctrl_indices = np.unique(ctrl_idx_chunk[ctrl_idx_chunk >= 0])
                if len(unique_ctrl_indices) > 0:
                    # Retrieve xbar for all unique controls needed for this chunk.
                    xbar_ctrl_flat = qmodel.get_normalized_expression(
                        indices=unique_ctrl_indices,
                        batch_size=self.cfg.forward_batch_size,
                        n_samples=1,
                        library_size=1e4,
                    ).to_numpy()
                    
                    # Create a mapping from index to expression data
                    index_to_xbar_map = {idx: xbar_ctrl_flat[i] for i, idx in enumerate(unique_ctrl_indices)}
                    
                    # Reconstruct the batch shape (B, k, G)
                    B, k = ctrl_idx_chunk.shape
                    G = xbar_ctrl_flat.shape[1]
                    xbar_ctrl_chunk = np.zeros((B, k, G), dtype=np.float32)
                    for i in range(B):
                        for j in range(k):
                            idx = ctrl_idx_chunk[i, j]
                            if idx in index_to_xbar_map:
                                xbar_ctrl_chunk[i, j] = index_to_xbar_map[idx]
                    t7 = time.time()
                    print(f"[profile] Control Cell Decoding: {t7 - t6:.4f} sec")
                else:
                    print(f"[warn] No valid controls found for dataset {dsid} in this chunk.")
                    xbar_ctrl_chunk = np.zeros((B, k, G), dtype=np.float32) # Fallback if no controls found

                # Break chunk into mini-batches
                B = int(self.cfg.batch_size)
                N = A_chunk.n_obs
                for i0 in range(0, N, B):
                    i1 = min(i0 + B, N)
                    # Slice the pre-computed chunk-level results for this mini-batch
                    z_q = z_chunk[i0:i1]
                    xbar_q = xbar_chunk[i0:i1]
                    ids_q = np.array(A_chunk.obs_names[i0:i1], dtype=str)
                    ctrl_idx = ctrl_idx_chunk[i0:i1]
                    ctrl_ids = ctrl_ids_chunk[i0:i1]
                    z_ctrl = z_ctrl_chunk[i0:i1]
                    x_bar_ctrl = xbar_ctrl_chunk[i0:i1]
                    ctrl_dist = ctrl_dist_chunk[i0:i1]
                    
                    batch = {
                        "z_q": z_q,                       # (b, d)
                        "xbar_q": xbar_q,                 # (b, G)
                        "cell_ids": ids_q,                # (b,)
                        "target_idx": tgt_idx_chunk[i0:i1], # (b,)
                        "env_code": np.full((i1 - i0,), self.env_codes[dsid], dtype=np.int64),
                        # control matches:
                        "ctrl_idx": ctrl_idx,             # (b, k), labels into z_ctrl/ctrl_ids
                        "ctrl_ids": ctrl_ids,             # (b, k)
                        "z_ctrl": z_ctrl,                 # (b, k, d)
                        "xbar_ctrl": x_bar_ctrl,          # (b, k, G)
                        "ctrl_dist": ctrl_dist,           # (b, k)
                        "dataset_id": dsid,               # string for reference
                    }
                    yield batch

    # ---- helpers ---- #

    def _targets_for_ids(self, cell_ids: Iterable[str]) -> np.ndarray:
        if self.target_col is None:
            return np.full((len(list(cell_ids))), -1, dtype=np.int64)
        # build id->target gene and then map to index
        df = self.index[["cell_id", self.target_col]].set_index("cell_id")
        out = np.empty(len(cell_ids), dtype=np.int64)
        for i, cid in enumerate(cell_ids):
            tg = df.at[cid, self.target_col] if cid in df.index else None
            if tg is None or tg == "" or tg not in self.gene_to_idx:
                out[i] = -1
            else:
                out[i] = self.gene_to_idx[tg]
        return out


# ------------------------------ tiny YAML loader ----------------------------- #

def _yaml_safe_load(text: str) -> dict:
    """Very small helper to load YAML as a Python dict (mapping datasets -> config)."""
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError("datasets.yaml must be a mapping.")
    return cfg