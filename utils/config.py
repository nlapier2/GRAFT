
import yaml
from typing import Dict, Any

DEFAULTS = {
    "control_tokens": ["control","ntc","non-targeting","neg_control","dmso","vehicle"],
    "target_normalize": "hgnc",
    "env_key": "batch_id",
    "pert_type_guess": "crispri",
}

def load_datasets_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # normalize structure
    defaults = DEFAULTS.copy()
    if isinstance(cfg, dict) and "defaults" in cfg and cfg["defaults"]:
        for k, v in cfg["defaults"].items():
            defaults[k] = v
    datasets = cfg.get("datasets", cfg) if "datasets" in cfg else cfg
    return {"defaults": defaults, "datasets": datasets}
