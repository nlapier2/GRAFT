# temporary workaround for script visibility
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from graft.data.dataset import (
    GraftStreamingConfig,
    GraftStreamingDataset,
    ControlANN,
    BatchQuery,
)

# ------------------------------ fixtures ------------------------------ #

@pytest.fixture()
def tiny_env(tmp_path: Path, monkeypatch):
    """
    Build a tiny on-disk environment with:
      - 2 datasets (DSA, DSB) as .h5ad (backed-capable)
      - a global index parquet with standardized columns
      - a global gene list TSV
      - a dummy control index dir + z npz
    We monkeypatch ControlANN.load and BatchQuery.encode_and_decode
    to avoid faiss/hnswlib/scVI runtime deps.
    """
    rng = np.random.default_rng(42)

    # Genes and gene list
    genes = ["G1", "G2", "G3", "G4", "G5"]
    gene_list_tsv = tmp_path / "gene_list.tsv"
    gene_list_tsv.write_text("\n".join(genes) + "\n")

    # Build two toy datasets
    def make_ds(name: str, n: int) -> Path:
        X = sp.csr_matrix(rng.poisson(1.5, size=(n, len(genes))).astype(np.float32))
        obs = pd.DataFrame(index=pd.Index([f"{name}_cell{i:03d}" for i in range(n)], name="cell"))
        var = pd.DataFrame(index=pd.Index(genes, name="gene"))
        A = ad.AnnData(X=X, obs=obs, var=var)
        p = tmp_path / f"{name}.h5ad"
        A.write_h5ad(p)
        return p

    dsA_path = make_ds("DSA", 37)
    dsB_path = make_ds("DSB", 29)

    # datasets.yaml (mapping style)
    datasets_yaml = tmp_path / "datasets.yaml"
    datasets_yaml.write_text(
        f"""
defaults: {{}}
datasets:
  DSA:
    raw_path: {dsA_path.as_posix()}
  DSB:
    raw_path: {dsB_path.as_posix()}
""".strip()
    )

    # Build index parquet with global cell ids: "<dataset_id>::<local_cell_id>"
    rows = []
    for name, path, n in [("DSA", dsA_path, 37), ("DSB", dsB_path, 29)]:
        for i in range(n):
            local = f"{name}_cell{i:03d}"
            cid = f"{name}::{local}"
            rows.append(
                {
                    "cell_id": cid,
                    "dataset_id": name,
                    "batch_id": "1",
                    "is_control": False if (i % 3 == 0) else True,  # some perturbed
                    "target_gene": "G2" if (i % 7 == 0) else "",     # sparse targets
                }
            )
    index_df = pd.DataFrame(rows)
    index_parquet = tmp_path / "index.parquet"
    index_df.to_parquet(index_parquet)

    # Dummy control index dir + z npz (we monkeypatch ControlANN.load anyway)
    ctrl_dir = tmp_path / "ctrl_index"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    (ctrl_dir / "knn_meta.json").write_text(
        json.dumps({"backend": "faiss", "metric": "cosine", "dim": 8, "n_items": 4})
    )
    pd.DataFrame(
        {
            "cell_id": [
                "DSA::DSA_cell000",
                "DSA::DSA_cell003",
                "DSB::DSB_cell000",
                "DSB::DSB_cell003",
            ]
        }
    ).to_parquet(ctrl_dir / "ctrl_ids.parquet")
    (ctrl_dir / "knn.index").write_bytes(b"FAKE")  # placeholder

    z_npz = tmp_path / "controls_z.npz"
    np.savez(z_npz, z=np.zeros((4, 8), dtype=np.float32))

    # Monkeypatch ControlANN.load to bypass faiss/hnswlib and file I/O
    class _DummyANN:
        def __init__(self):
            self.backend = "dummy"
            self.metric = "cosine"
            self.dim = 8
            self.ctrl_ids = np.array(
                [
                    "DSA::DSA_cell000",
                    "DSA::DSA_cell003",
                    "DSB::DSB_cell000",
                    "DSB::DSB_cell003",
                ],
                dtype=object,
            )
            self.z_ctrl = np.zeros((4, 8), dtype=np.float32)
            self.ctrl_ds_lookup = {
                "DSA::DSA_cell000": "DSA",
                "DSA::DSA_cell003": "DSA",
                "DSB::DSB_cell000": "DSB",
                "DSB::DSB_cell003": "DSB",
            }

        def query(self, z_query, k=2, match_dataset=None, oversample=5, caliper=None):
            B = z_query.shape[0]
            out_idx = -np.ones((B, k), dtype=np.int64)
            out_ids = np.empty((B, k), dtype=object)
            out_dist = np.zeros((B, k), dtype=np.float32)
            for i in range(B):
                if match_dataset == "DSA":
                    out_idx[i] = np.array([0, 1])
                    out_ids[i] = np.array([self.ctrl_ids[0], self.ctrl_ids[1]], dtype=object)
                elif match_dataset == "DSB":
                    out_idx[i] = np.array([2, 3])
                    out_ids[i] = np.array([self.ctrl_ids[2], self.ctrl_ids[3]], dtype=object)
                else:
                    out_idx[i] = np.array([0, 2])
                    out_ids[i] = np.array([self.ctrl_ids[0], self.ctrl_ids[2]], dtype=object)
            return out_idx, out_ids, out_dist

    monkeypatch.setattr(ControlANN, "load", staticmethod(lambda *args, **kwargs: _DummyANN()))

    # Monkeypatch BatchQuery.encode_and_decode to avoid scVI
    def _fake_encode_and_decode(self, A_chunk: ad.AnnData):
        B = A_chunk.n_obs
        d = 8   # fake z dim
        xbar = (A_chunk.X.toarray() if sp.issparse(A_chunk.X) else np.asarray(A_chunk.X)).astype(np.float32)
        z = np.ones((B, d), dtype=np.float32) * 0.5  # deterministic
        return z, xbar

    monkeypatch.setattr(BatchQuery, "encode_and_decode", _fake_encode_and_decode, raising=True)

    return {
        "datasets_yaml": datasets_yaml,
        "index_parquet": index_parquet,
        "gene_list_tsv": gene_list_tsv,
        "scvi_model_dir": tmp_path / "noop_scvi",  # unused with monkeypatch
        "control_index_dir": ctrl_dir,
        "control_z_npz": z_npz,
    }

# ------------------------------------ tests ----------------------------------- #

def test_streaming_iter_batches_shapes_and_keys(tiny_env):
    cfg = GraftStreamingConfig(
        datasets_yaml=str(tiny_env["datasets_yaml"]),
        index_parquet=str(tiny_env["index_parquet"]),
        gene_list_tsv=str(tiny_env["gene_list_tsv"]),
        scvi_model_dir=str(tiny_env["scvi_model_dir"]),
        control_index_dir=str(tiny_env["control_index_dir"]),
        control_z_npz=str(tiny_env["control_z_npz"]),
        batch_size=16,
        chunk_size=32,
        k_controls=2,
        match_within="dataset",
    )
    ds = GraftStreamingDataset(cfg)

    dsids = sorted(ds.get_dataset_ids())
    assert set(dsids) == {"DSA", "DSB"}

    # Pull a couple of batches from each dataset
    got = 0
    for dsid in ["DSA", "DSB"]:
        for batch in ds.iter_batches([dsid]):
            for k in [
                "z_q", "xbar_q", "cell_ids", "target_idx", "env_code",
                "ctrl_idx", "ctrl_ids", "z_ctrl", "ctrl_dist", "dataset_id",
            ]:
                assert k in batch, f"missing key {k}"

            B, d = batch["z_q"].shape
            G = batch["xbar_q"].shape[1]
            assert d == 8
            assert G == 5
            assert batch["dataset_id"] == dsid
            assert batch["env_code"].shape[0] == B
            assert batch["ctrl_idx"].shape == (B, 2)
            assert batch["z_ctrl"].shape == (B, 2, d)
            got += 1
            if got >= 2:
                break
        if got >= 2:
            break
    assert got >= 2


def test_target_mapping_and_env_codes(tiny_env):
    cfg = GraftStreamingConfig(
        datasets_yaml=str(tiny_env["datasets_yaml"]),
        index_parquet=str(tiny_env["index_parquet"]),
        gene_list_tsv=str(tiny_env["gene_list_tsv"]),
        scvi_model_dir=str(tiny_env["scvi_model_dir"]),
        control_index_dir=str(tiny_env["control_index_dir"]),
        control_z_npz=str(tiny_env["control_z_npz"]),
        batch_size=10,
        chunk_size=20,
        k_controls=2,
        match_within="dataset",
    )
    ds = GraftStreamingDataset(cfg)

    # Grab one batch from DSA
    b = next(iter(ds.iter_batches(["DSA"])))
    ids = b["cell_ids"]
    # env_code must be constant per dataset
    assert (b["env_code"] == b["env_code"][0]).all()

    # target_idx should reflect "target_gene" in the index parquet (G2 for every 7th row)
    gene_to_idx = {g: i for i, g in enumerate(["G1", "G2", "G3", "G4", "G5"])}
    idx_df = pd.read_parquet(tiny_env["index_parquet"]).set_index("cell_id")
    expected = []
    for cid in ids:
        tg = idx_df.at[cid, "target_gene"]
        expected.append(gene_to_idx["G2"] if tg == "G2" else -1)
    assert (b["target_idx"] == np.array(expected, dtype=np.int64)).all()


def test_match_within_dataset_policy(tiny_env, monkeypatch):
    # Spy on the 'match_dataset' argument passed into ANN.query
    calls = {"seen": []}
    real_loader = ControlANN.load

    def _loader_with_spy(*args, **kwargs):
        ann = real_loader(*args, **kwargs)  # already monkeypatched to dummy in fixture
        real_query = ann.query

        def _spy(z_query, k=2, match_dataset=None, oversample=5, caliper=None):
            calls["seen"].append(match_dataset)
            return real_query(z_query, k=k, match_dataset=match_dataset, oversample=oversample, caliper=caliper)

        ann.query = _spy
        return ann

    monkeypatch.setattr(ControlANN, "load", staticmethod(_loader_with_spy))

    cfg = GraftStreamingConfig(
        datasets_yaml=str(tiny_env["datasets_yaml"]),
        index_parquet=str(tiny_env["index_parquet"]),
        gene_list_tsv=str(tiny_env["gene_list_tsv"]),
        scvi_model_dir=str(tiny_env["scvi_model_dir"]),
        control_index_dir=str(tiny_env["control_index_dir"]),
        control_z_npz=str(tiny_env["control_z_npz"]),
        batch_size=8,
        chunk_size=16,
        k_controls=2,
        match_within="dataset",
    )
    ds = GraftStreamingDataset(cfg)

    _ = next(iter(ds.iter_batches(["DSA"])))
    _ = next(iter(ds.iter_batches(["DSB"])))

    assert calls["seen"][0] == "DSA"
    assert calls["seen"][1] == "DSB"
