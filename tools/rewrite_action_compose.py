#!/usr/bin/env python3
"""Rewrite LeRobot actions: joints from observation.state, grippers from action.

Example:
  PYTHONPATH=src python tools/rewrite_action_compose.py \\
    --config configs/cxn/record_pick_cube_cxn_act.json

  PYTHONPATH=src python tools/rewrite_action_compose.py \\
    --source-root data/lerobot/local/cube_v727_cxn \\
    --target-repo-id local/cube_v727_cxn_ojag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from piper_towel_fold.action_compose import (  # noqa: E402
    ACTION_COMPOSE_MODES,
    build_composed_dataset,
    default_compose_repo_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset whose action joints come from observation.state "
        "and grippers from the original action (for ACT training).",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional recording/training JSON; uses repo_id/root and training.action_compose*.",
    )
    parser.add_argument("--source-repo-id", default="")
    parser.add_argument("--source-root", default="", help="Full path to source dataset directory.")
    parser.add_argument("--root", default="data/lerobot", help="LeRobot root when using repo ids.")
    parser.add_argument("--target-repo-id", default="")
    parser.add_argument(
        "--mode",
        default="obs_joints_action_gripper",
        choices=ACTION_COMPOSE_MODES,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of hard-linking (uses more disk).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def main() -> None:
    args = parse_args()
    mode = args.mode
    root = Path(args.root)
    source_repo_id = args.source_repo_id
    target_repo_id = args.target_repo_id
    source_root: Path | None = Path(args.source_root) if args.source_root else None

    if args.config:
        config = load_config(Path(args.config))
        training = config.get("training") or {}
        mode = str(training.get("action_compose") or args.mode)
        source_repo_id = source_repo_id or str(config["repo_id"])
        root = Path(args.root if args.root != "data/lerobot" else config.get("root", "data/lerobot"))
        target_repo_id = target_repo_id or str(
            training.get("action_compose_repo_id") or default_compose_repo_id(source_repo_id, mode)
        )
        if source_root is None:
            source_root = root / source_repo_id

    if source_root is None:
        if not source_repo_id:
            raise SystemExit("Provide --config, or --source-root / --source-repo-id")
        source_root = root / source_repo_id
    else:
        if not source_repo_id:
            source_repo_id = f"local/{source_root.name}"

    if not target_repo_id:
        target_repo_id = default_compose_repo_id(source_repo_id, mode)
    target_root = root / target_repo_id if not target_repo_id.startswith("/") else Path(target_repo_id)
    # when target_repo_id is local/xxx, root/local/xxx is correct
    if "/" in target_repo_id and not Path(target_repo_id).is_absolute():
        target_root = root / target_repo_id

    print("Action compose")
    print(f"  mode: {mode}")
    print(f"  source: {source_root}")
    print(f"  target: {target_root}")
    print(f"  target repo_id: {target_repo_id}")
    if args.dry_run:
        print("dry-run: not writing")
        return

    build_composed_dataset(
        source_root,
        target_root,
        source_repo_id=source_repo_id,
        mode=mode,
        overwrite=args.overwrite,
        hardlink_videos=not args.copy_videos,
    )
    print()
    print("Next: point training.repo_id (or training.action_compose_repo_id + start_training) to:")
    print(f"  {target_repo_id}")


if __name__ == "__main__":
    main()
