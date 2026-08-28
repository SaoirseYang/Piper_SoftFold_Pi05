"""Move Piper arms to episode-0 start pose from a recording/training JSON config.

Reuses the move-to stage from tools/replay_episode.py. Dataset comes from the
config (policy_live.dataset_root, else root/repo_id). Default episode index is 0.

Examples:
  PYTHONPATH=src python tools/move_to_episode_start.py --config configs/fyx.json
  PYTHONPATH=src python tools/move_to_episode_start.py --config configs/fyx.json --dry-run
  bash scripts/run_move_to_episode_start.sh configs/fyx.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from move_to_joints import format_arm, move_to, wait_for_valid_state  # noqa: E402
from replay_episode import (  # noqa: E402
    action_to_deg_targets,
    infer_repo_id,
    load_episode_trajectory,
    print_pose,
)

from piper_towel_fold.config import PiperRobotConfig
from piper_towel_fold.offline_infer import load_dataset
from piper_towel_fold.piper import PiperRobot


def load_json_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def resolve_dataset_from_config(config: dict[str, Any]) -> tuple[str, Path]:
    """Return (repo_id, dataset_root) from recording/training JSON."""
    policy_live = config.get("policy_live") or {}
    if not isinstance(policy_live, dict):
        policy_live = {}

    if policy_live.get("dataset_root"):
        dataset_root = Path(str(policy_live["dataset_root"]))
        repo_id = str(policy_live.get("repo_id") or config.get("repo_id") or "")
        if not repo_id:
            repo_id = infer_repo_id(dataset_root, None)
        return repo_id, dataset_root

    repo_id = str(policy_live.get("repo_id") or config.get("repo_id") or "")
    if not repo_id:
        raise SystemExit("config 缺少 repo_id / policy_live.repo_id")

    root = Path(str(policy_live.get("root") or config.get("root") or "data/lerobot"))
    return repo_id, root / repo_id


def resolve_can_ports(config: dict[str, Any]) -> tuple[str, str]:
    policy_live = config.get("policy_live") or {}
    if not isinstance(policy_live, dict):
        policy_live = {}

    def pick(key: str, default: str) -> str:
        for block in (policy_live, config):
            if key in block and block[key] not in (None, ""):
                return str(block[key])
        return default

    # live 侧常用 follower_*_can；replay 侧用 left/right-can
    left = pick("follower_left_can", pick("left_can", "can2"))
    right = pick("follower_right_can", pick("right_can", "can0"))
    return left, right


def resolve_move_to_pose(
    frames: list[dict[str, float]],
    start_state: dict[str, float] | None,
    move_to_source: str,
    replay_source: str,
) -> tuple[dict[str, float], str]:
    if move_to_source in ("state", "observation"):
        if start_state is None:
            raise SystemExit("数据集无 observation.state，无法使用 --move-to-source observation/state")
        return start_state, "observation.state"
    if move_to_source == "action":
        return frames[0], f"replay[{replay_source}]"
    if start_state is not None:
        return start_state, "observation.state(auto)"
    return frames[0], f"{replay_source}(auto-fallback)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将机械臂移动到配置指向数据集的 episode 初始位姿（默认第 0 条）。",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="录制/训练 JSON，例如 configs/fyx.json",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="episode 序号，默认 0",
    )
    parser.add_argument(
        "--move-to-source",
        choices=("auto", "action", "observation", "state"),
        default="auto",
        help="到位目标：auto=优先 observation.state（与 replay 一致）",
    )
    parser.add_argument(
        "--source",
        default="obs_joints_action_gripper",
        help="预加载轨迹字段（仅用于取首帧；默认与推荐回放一致）",
    )
    parser.add_argument("--left-can", default=None, help="覆盖 config 左臂 CAN")
    parser.add_argument("--right-can", default=None, help="覆盖 config 右臂 CAN")
    parser.add_argument("--control-speed", type=int, default=15)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.1)
    parser.add_argument("--tol-deg", type=float, default=2.5)
    parser.add_argument("--period", type=float, default=0.02)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--state-wait", type=float, default=5.0)
    parser.add_argument("--settle", type=float, default=0.0)
    parser.add_argument(
        "--strict-move-to",
        action="store_true",
        help="到位超时则失败退出；默认只告警",
    )
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(Path(args.config))
    repo_id, dataset_root = resolve_dataset_from_config(config)
    left_can, right_can = resolve_can_ports(config)
    if args.left_can:
        left_can = args.left_can
    if args.right_can:
        right_can = args.right_can

    if not (dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"找不到数据集 meta/info.json：{dataset_root}")

    print(f"config: {args.config}")
    print(f"dataset: repo_id={repo_id} root={dataset_root}")
    print(f"episode: {args.episode_index}")
    print(f"CAN: left={left_can} right={right_can}")

    dataset = load_dataset(repo_id, str(dataset_root), args.video_backend)
    # 只需要首帧；end_frame=1 避免整段预加载
    frames, start_state, _, _, _ = load_episode_trajectory(
        dataset,
        args.episode_index,
        args.source,
        start_frame=0,
        end_frame=1,
    )
    move_to_pose, move_to_label = resolve_move_to_pose(
        frames,
        start_state,
        args.move_to_source,
        args.source,
    )
    print_pose(f"到位目标({move_to_label})", move_to_pose)

    if args.dry_run:
        print("dry-run：未连接机械臂。")
        return

    robot = PiperRobot(
        PiperRobotConfig(
            follower_left_port=left_can,
            follower_right_port=right_can,
            enable_control=True,
            control_speed=args.control_speed,
            max_joint_step_rad=args.max_joint_step_rad,
            cameras={},
        )
    )

    try:
        robot.connect()
        print(f"等待关节反馈（最多 {args.state_wait:.1f}s）…")
        obs = wait_for_valid_state(robot, timeout_s=args.state_wait)
        print("当前位姿：")
        print(format_arm(obs, "left"))
        print(format_arm(obs, "right"))

        left_deg, right_deg, left_g, right_g = action_to_deg_targets(move_to_pose)
        print(f"移动到 episode 初始位姿（{move_to_label}，tol={args.tol_deg}°）…")
        try:
            move_to(
                robot,
                left_deg,
                right_deg,
                left_g,
                right_g,
                tol_deg=args.tol_deg,
                period_s=args.period,
                timeout_s=args.timeout,
            )
        except TimeoutError as exc:
            if args.strict_move_to:
                raise
            print(f"警告：初始到位未完全收敛（{exc}）。")

        if args.settle > 0:
            print(f"静置 {args.settle:.2f}s …")
            time.sleep(args.settle)

        obs = robot.get_observation()
        print("结束位姿：")
        print(format_arm(obs, "left"))
        print(format_arm(obs, "right"))
    except KeyboardInterrupt:
        print("\n已中断。")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
