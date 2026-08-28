#!/usr/bin/env python3
"""Stage-2 PEFT: shallow low-LR joint finetune after Soft Prompt warmup.

Loads stage-1 checkpoint, unfreezes last vision blocks (via config flags),
and continues training at a lower learning rate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train_stage2.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER)
    return p.parse_args()


def build_cmd(cfg: dict) -> list[str]:
    t = cfg["training"]
    ds = cfg["dataset"]
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={ds['repo_id']}",
        f"--dataset.root={ds['root']}",
        f"--policy.path={t['policy_path']}",
        f"--output_dir={t['output_dir']}",
        f"--job_name={t['job_name']}",
        f"--policy.device={t.get('device', 'cuda')}",
        f"--steps={t['steps']}",
        f"--batch_size={t['batch_size']}",
        f"--log_freq={t.get('log_freq', 50)}",
        f"--save_freq={t.get('save_freq', 2000)}",
        f"--wandb.enable={str(t.get('wandb_enable', False)).lower()}",
        f"--policy.push_to_hub=false",
        f"--policy.dtype={t.get('dtype', 'bfloat16')}",
        f"--policy.action_mode={t.get('action_mode', 'auto')}",
        f"--policy.max_action_dim={int(t.get('max_action_dim', 20))}",
        f"--policy.chunk_size={int(t.get('chunk_size', 16))}",
        f"--policy.n_action_steps={int(t.get('n_action_steps', 16))}",
        f"--policy.num_denoising_steps={int(t.get('num_denoising_steps', 10))}",
        # Stage-2: allow vision encoder updates (shallow via low LR / short steps)
        f"--policy.freeze_vision_encoder={str(t.get('freeze_vision_encoder', False)).lower()}",
        f"--policy.freeze_language_encoder={str(t.get('freeze_language_encoder', True)).lower()}",
        f"--policy.train_policy_transformer={str(t.get('train_policy_transformer', False)).lower()}",
        f"--policy.train_soft_prompts={str(t.get('train_soft_prompts', True)).lower()}",
        f"--policy.use_proprio={str(t.get('use_proprio', True)).lower()}",
    ]
    if t.get("rename_map"):
        cmd.append(f"--rename_map={json.dumps(t['rename_map'])}")
    if t.get("normalization_mapping"):
        cmd.append(f"--policy.normalization_mapping={json.dumps(t['normalization_mapping'])}")
    return cmd


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cmd = build_cmd(cfg)
    if args.extra:
        extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
        cmd.extend(extra)
    print("[stage2] command:")
    print(" \\\n  ".join(cmd))
    print(
        "[stage2] reminder: monitor source-domain forget rate; early-stop if drop > "
        f"{cfg.get('early_stop_source_drop', 0.05)}"
    )
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
