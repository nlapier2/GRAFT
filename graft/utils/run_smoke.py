#!/usr/bin/env python3
import argparse, tempfile, subprocess, sys
from pathlib import Path
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base config YAML (e.g., configs/gnn_smoke.yaml)")
    ap.add_argument("--steps", type=int, help="Override training.steps_per_epoch")
    ap.add_argument("--epochs", type=int, help="Override training.epochs")
    ap.add_argument("--batch-size", type=int, help="Override training.batch_size")
    ap.add_argument("--chunk-size", type=int, help="Override training.chunk_size")
    ap.add_argument("--lr", type=float, help="Override training.lr")
    ap.add_argument("--outdir", help="Override paths.output_dir")
    args = ap.parse_args()

    with open(args.base, "r") as f:
        cfg = yaml.safe_load(f)

    tr = cfg.setdefault("training", {})
    paths = cfg.setdefault("paths", {})
    if args.steps is not None: tr["steps_per_epoch"] = int(args.steps)
    if args.epochs is not None: tr["epochs"] = int(args.epochs)
    if args.batch_size is not None: tr["batch_size"] = int(args.batch_size)
    if args.chunk_size is not None: tr["chunk_size"] = int(args.chunk_size)
    if args.lr is not None: tr["lr"] = float(args.lr)
    if args.outdir: paths["output_dir"] = args.outdir

    Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name

    print(f"[run] train_gnn.py --config {tmp_path}")
    p = subprocess.run([sys.executable, "train_gnn.py", "--config", tmp_path])
    sys.exit(p.returncode)

if __name__ == "__main__":
    main()
