#!/usr/bin/env python3
"""Merge virtual Piper joint dataset + local real teleop episodes for mixed training.

Does not physically concat parquet files. Instead it writes a mix manifest that
``train/mixed_dataloader.py`` / training configs can consume:

  {
    "virtual": {"root": "...", "weight": 1.0, "sample_ratio": 0.7},
    "local":   {"root": "...", "weight": 1.5, "sample_ratio": 0.3},
    ...
  }

Optional: copy/symlink both roots under an output directory for packaging.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--virtual-root", type=Path, required=True)
    p.add_argument("--local-root", type=Path, required=True)
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--virtual-ratio", type=float, default=0.7)
    p.add_argument("--local-ratio", type=float, default=0.3)
    p.add_argument("--virtual-loss-weight", type=float, default=1.0)
    p.add_argument("--local-loss-weight", type=float, default=1.5)
    p.add_argument("--package-dir", type=Path, default=None, help="Optional dir to symlink datasets into")
    p.add_argument("--task", default="fold the cloth")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if abs(args.virtual_ratio + args.local_ratio - 1.0) > 1e-6:
        raise SystemExit("virtual-ratio + local-ratio must sum to 1.0")

    for root, name in ((args.virtual_root, "virtual"), (args.local_root, "local")):
        if not root.exists():
            raise SystemExit(f"{name} root missing: {root}")

    manifest = {
        "task": args.task,
        "virtual": {
            "root": str(args.virtual_root.resolve()),
            "sample_ratio": args.virtual_ratio,
            "loss_weight": args.virtual_loss_weight,
        },
        "local": {
            "root": str(args.local_root.resolve()),
            "sample_ratio": args.local_ratio,
            "loss_weight": args.local_loss_weight,
        },
        "notes": "7:3 virtual:local mix; local loss weight 1.5x (proposal default).",
    }

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[make_dataset] wrote {args.out_manifest}")

    if args.package_dir is not None:
        args.package_dir.mkdir(parents=True, exist_ok=True)
        for key, root in (("virtual", args.virtual_root), ("local", args.local_root)):
            dst = args.package_dir / key
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(root.resolve())
            print(f"[make_dataset] symlink {dst} -> {root}")
        (args.package_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
