# temporary workaround for script visibility
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter

import numpy as np
import pytest

from graft.data.samplers import (
    BalancedRoundRobin,
    WeightedDatasetSampler,
    InterleavedGlobalSampler,
    normalize_weights,
    derive_weights_from_sizes,
    make_dataset_chooser,
    estimate_dataset_sizes,
)


def test_normalize_weights_basic():
    w = {"A": 2.0, "B": 1.0, "C": 1.0}
    p = normalize_weights(w)
    assert pytest.approx(sum(p.values()), rel=1e-9, abs=1e-9) == 1.0
    # ratios preserved
    assert pytest.approx(p["A"] / p["B"], rel=1e-6) == 2.0
    # fallback to uniform when all nonpositive
    p2 = normalize_weights({"A": 0.0, "B": 0.0})
    assert pytest.approx(p2["A"], rel=1e-9) == 0.5
    assert pytest.approx(p2["B"], rel=1e-9) == 0.5


def test_derive_weights_modes():
    sizes = {"A": 100, "B": 25, "C": 1}
    # uniform
    w_u = derive_weights_from_sizes(sizes, mode="uniform")
    assert len(set(w_u.values())) == 1
    # count
    w_c = derive_weights_from_sizes(sizes, mode="count")
    assert w_c["A"] > w_c["B"] > w_c["C"]
    # sqrt
    w_s = derive_weights_from_sizes(sizes, mode="sqrt")
    assert w_s["A"] > w_s["B"] > w_s["C"]
    # error on unknown
    with pytest.raises(ValueError):
        derive_weights_from_sizes(sizes, mode="???")


def test_balanced_round_robin_order_no_shuffle():
    ids = ["A", "B", "C", "D"]
    steps = 2 * len(ids)
    it = iter(BalancedRoundRobin(ids, steps=steps, shuffle_each_epoch=False, seed=123))
    out = [next(it) for _ in range(steps)]
    assert out[:4] == ids
    assert out[4:8] == ids  # repeats same order


def test_balanced_round_robin_shuffle_each_epoch():
    ids = ["A", "B", "C", "D"]
    steps = 2 * len(ids)
    it = iter(BalancedRoundRobin(ids, steps=steps, shuffle_each_epoch=True, seed=123))
    out = [next(it) for _ in range(steps)]
    first = out[:4]
    second = out[4:8]
    assert set(first) == set(ids)
    assert set(second) == set(ids)
    # with shuffle, very likely not equal to original order and not equal between epochs
    assert first != ids
    assert first != second


def test_weighted_dataset_sampler_distribution():
    ids = ["A", "B", "C"]
    # weights will produce B >> A > C
    weights = {"A": 1.0, "B": 4.0, "C": 0.5}
    steps = 20000
    rng_seed = 7
    sampler = iter(WeightedDatasetSampler(ids, weights=weights, steps=steps, seed=rng_seed))
    draws = [next(sampler) for _ in range(steps)]
    counts = Counter(draws)
    # normalize
    freqs = {k: counts[k] / steps for k in ids}
    # order should roughly follow weights: B highest, then A, then C
    assert freqs["B"] > freqs["A"] > freqs["C"]
    # sanity: each prob within a reasonable band of normalized weights
    total_w = sum(weights.values())
    for k in ids:
        expected = weights[k] / total_w
        assert abs(freqs[k] - expected) < 0.03  # 3% tolerance for randomness


def test_make_dataset_chooser_weighted_sqrt():
    ids = ["A", "B", "C", "D"]
    sizes = {"A": 100, "B": 25, "C": 9, "D": 4}
    chooser = make_dataset_chooser(
        dataset_ids=ids,
        sizes=sizes,
        policy="weighted",
        weight_mode="sqrt",
        steps=1000,
        seed=123,
    )
    draws = [d for d in chooser]
    assert set(draws).issubset(set(ids))
    # sqrt weighting: A should appear most frequently
    counts = Counter(draws)
    assert counts["A"] >= counts["B"] >= counts["C"] >= counts["D"]


def test_interleaved_global_sampler_positions():
    base = BalancedRoundRobin(["A", "B", "C"], steps=12, shuffle_each_epoch=False, seed=1)
    inter = InterleavedGlobalSampler(base=base, every=3)
    out = [x for x in inter]
    # After every 3 dataset items, a "__GLOBAL__" is inserted
    # Sequence example: A,B,C,__GLOBAL__,A,B,C,__GLOBAL__,A,B,C,__GLOBAL__
    # There should be N//every globals (12//3 = 4), at 0-based positions 3,7,11,15.
    assert out[3] == "__GLOBAL__"
    assert out[7] == "__GLOBAL__"
    assert out[11] == "__GLOBAL__"
    assert out[15] == "__GLOBAL__"
    assert out.count("__GLOBAL__") == 4


def test_estimate_dataset_sizes_without_pandas():
    class DummyDF:
        def __init__(self, n):
            self.shape = (n, 0)
    by_ds = {"A": DummyDF(10), "B": DummyDF(3), "C": DummyDF(0)}
    sizes = estimate_dataset_sizes(by_ds)
    assert sizes == {"A": 10, "B": 3, "C": 0}
