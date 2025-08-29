
from __future__ import annotations
import numpy as np
from typing import Dict, List

class LabBalancedSampler:
    """
    Uniformly picks a lab, then samples indices from that lab's pool.
    """
    def __init__(self, lab_to_idx: Dict[str, np.ndarray], batch_size: int, seed: int = 13):
        self.lab_to_idx = {k: np.array(v, dtype=int) for k, v in lab_to_idx.items() if len(v) > 0}
        self.labs = list(self.lab_to_idx.keys())
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            lab = self.rng.choice(self.labs)
            pool = self.lab_to_idx[lab]
            if len(pool) >= self.batch_size:
                idx = self.rng.choice(pool, size=self.batch_size, replace=False)
            else:
                idx = self.rng.choice(pool, size=self.batch_size, replace=True)
            yield idx
