#!/usr/bin/env python3
"""Real-robot inference helper for Piper joint-space X-VLA policies.

This script focuses on the control-side extras from the proposal:
  - load joint policy
  - optional EMA smoothing
  - package actions as 14-D Piper commands

For the full live stack (cameras + CAN + safety), call into the existing
``piper`` repo ``run_policy_live`` after pointing ``policy_path`` here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from softfold.smoothing import ema_smooth  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoothing-alpha", type=float, default=0.2)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--emit-piper-live-json", type=Path, default=None,
                   help="Write a policy_live overlay JSON for piper scripts")
    p.add_argument("--execute", action="store_true", help="Set execute=true in overlay")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.policy_path.exists():
        raise SystemExit(f"policy path missing: {args.policy_path}")

    # Smoke: load policy if lerobot available
    try:
        from lerobot.common.policies.factory import get_policy_class  # type: ignore
        print(f"[infer] lerobot available; policy_path={args.policy_path}")
    except Exception:
        try:
            from lerobot.policies.factory import get_policy_class  # type: ignore
            print(f"[infer] lerobot available; policy_path={args.policy_path}")
        except Exception as exc:
            print(f"[infer] lerobot not importable ({exc}); writing overlay only")

    if args.emit_piper_live_json is not None:
        overlay = {
            "policy_live": {
                "policy_path": str(args.policy_path.resolve()),
                "device": args.device,
                "inference_dtype": "bfloat16",
                "num_inference_steps": 10,
                "execute": bool(args.execute),
                "fps": args.fps,
                "smoothing_alpha": args.smoothing_alpha,
                "control_speed": 30,
                "max_joint_step_rad": 0.03,
                "max_gripper_step_m": 0.002,
                "notes": "Joint-space Soft-Fold transfer policy (no online IK).",
            }
        }
        args.emit_piper_live_json.parent.mkdir(parents=True, exist_ok=True)
        args.emit_piper_live_json.write_text(json.dumps(overlay, indent=2))
        print(f"[infer] wrote {args.emit_piper_live_json}")

    # Demo EMA on a fake action stream
    prev = np.zeros(14, dtype=np.float64)
    cur = np.linspace(-0.1, 0.1, 14)
    sm = ema_smooth(prev, cur, alpha=args.smoothing_alpha)
    print(f"[infer] ema demo ||sm||={np.linalg.norm(sm):.4f} alpha={args.smoothing_alpha}")
    print("[infer] For live deployment, merge overlay into piper configs and run:")
    print("  bash ../piper/scripts/run_policy_live_xvla.sh <merged_config.json>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
