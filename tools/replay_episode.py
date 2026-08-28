"""Replay a recorded LeRobot episode on Piper dual-arm followers.

Flow:
  1) Load one episode's action (or observation.state) trajectory
  2) Connect arms and move to frame-0 pose from any current state
  3) Stream the remaining frames at dataset fps (optionally scaled)

Examples:
  PYTHONPATH=src python tools/replay_episode.py \\
    --dataset-root data/lerobot/local/cube_v727_cxn --episode-index 0

  PYTHONPATH=src python tools/replay_episode.py \\
    --dataset-root data/lerobot/local/cube_v727_cxn --episode-index 0 --dry-run

  PYTHONPATH=src python tools/replay_episode.py \\
    --dataset-root data/lerobot/local/cube_v727_cxn --episode-index 0 \\
    --source obs_joints_action_gripper

  # 默认回放不软件限速（1:1）；需要限速时加 --rate-limit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

# Allow importing sibling tools/move_to_joints.py when run as a script.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from move_to_joints import (  # noqa: E402
    format_arm,
    move_to,
    wait_for_valid_state,
)

from piper_towel_fold.config import PiperRobotConfig
from piper_towel_fold.offline_infer import (
    action_tensor_to_dict,
    get_episode_bounds,
    load_dataset,
)
from piper_towel_fold.piper import PiperRobot
from piper_towel_fold.recorder import ARM_STATE_KEYS

# 回放轨迹来源
REPLAY_SOURCE_ACTION = "action"
REPLAY_SOURCE_OBSERVATION = "observation"
REPLAY_SOURCE_OBS_JOINTS_ACTION_GRIPPER = "obs_joints_action_gripper"
REPLAY_SOURCES = (
    REPLAY_SOURCE_ACTION,
    REPLAY_SOURCE_OBSERVATION,
    REPLAY_SOURCE_OBS_JOINTS_ACTION_GRIPPER,
    "state",  # observation 别名，兼容旧参数
)

GRIPPER_KEYS = ("left_gripper.pos", "right_gripper.pos")


def infer_repo_id(dataset_root: str | Path, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    root = Path(dataset_root).resolve()
    # Prefer ".../lerobot/<repo_id>" → local/xxx when root ends with local/xxx
    if root.parent.name == "local":
        return f"local/{root.name}"
    return root.name


def normalize_replay_source(source: str) -> str:
    if source == "state":
        return REPLAY_SOURCE_OBSERVATION
    if source not in REPLAY_SOURCES:
        raise ValueError(f"未知 --source: {source}")
    return source


def tensor_or_list_to_dict(values: Any, names: list[str]) -> dict[str, float]:
    return action_tensor_to_dict(values, names)


def _dict_from_observation(sample: dict[str, Any], names: list[str]) -> dict[str, float]:
    state = sample.get("observation.state")
    if state is not None:
        return tensor_or_list_to_dict(state, names)

    out: dict[str, float] = {}
    for name in names:
        if name not in sample:
            raise KeyError(f"sample 缺少 observation.state，且没有键 {name}")
        out[name] = float(sample[name])
    return out


def frame_to_action_dict(sample: dict[str, Any], source: str, names: list[str]) -> dict[str, float]:
    source = normalize_replay_source(source)

    if source == REPLAY_SOURCE_ACTION:
        payload = sample.get("action")
        if payload is None:
            raise KeyError("sample 缺少 action 字段")
        return tensor_or_list_to_dict(payload, names)

    if source == REPLAY_SOURCE_OBSERVATION:
        return _dict_from_observation(sample, names)

    # 关节角用 observation.state，夹爪用 action
    if source == REPLAY_SOURCE_OBS_JOINTS_ACTION_GRIPPER:
        joints = _dict_from_observation(sample, names)
        action_payload = sample.get("action")
        if action_payload is None:
            raise KeyError("sample 缺少 action 字段（混合模式需要夹爪来自 action）")
        action = tensor_or_list_to_dict(action_payload, names)
        merged = dict(joints)
        for key in GRIPPER_KEYS:
            if key in action:
                merged[key] = action[key]
        return merged

    raise ValueError(f"未知 source: {source}")


def action_to_deg_targets(
    action: dict[str, float],
) -> tuple[list[float], list[float], float, float]:
    left = [math.degrees(action[f"left_joint_{i}.pos"]) for i in range(1, 7)]
    right = [math.degrees(action[f"right_joint_{i}.pos"]) for i in range(1, 7)]
    return left, right, float(action["left_gripper.pos"]), float(action["right_gripper.pos"])


def print_pose(label: str, action: dict[str, float]) -> None:
    left, right, lg, rg = action_to_deg_targets(action)
    print(f"{label}（度）：")
    print(
        "  left:  "
        + ", ".join(f"j{i}={v:8.3f}" for i, v in enumerate(left, start=1))
        + f", gripper={lg:.6f}"
    )
    print(
        "  right: "
        + ", ".join(f"j{i}={v:8.3f}" for i, v in enumerate(right, start=1))
        + f", gripper={rg:.6f}"
    )


def load_episode_trajectory(
    dataset: Any,
    episode_index: int,
    source: str,
    *,
    start_frame: int,
    end_frame: int | None,
) -> tuple[list[dict[str, float]], dict[str, float] | None, float, int, int]:
    start, stop = get_episode_bounds(dataset, episode_index)
    episode_len = stop - start
    if start_frame < 0 or start_frame >= episode_len:
        raise IndexError(f"--start-frame={start_frame} 超出 episode 长度 {episode_len}")

    local_end = episode_len if end_frame is None else end_frame
    if local_end <= start_frame or local_end > episode_len:
        raise IndexError(
            f"--end-frame={end_frame} 非法：需满足 {start_frame} < end <= {episode_len}"
        )

    feature_key = (
        "action"
        if source == REPLAY_SOURCE_ACTION
        else "observation.state"
    )
    feature = dataset.meta.features.get(feature_key) or dataset.meta.features["action"]
    names = list(feature.get("names") or ARM_STATE_KEYS)
    if len(names) != len(ARM_STATE_KEYS):
        # 仍按 names 解析；打印提醒
        print(f"警告：特征维数 {len(names)} 与默认 ARM_STATE_KEYS({len(ARM_STATE_KEYS)}) 不一致")

    frames: list[dict[str, float]] = []
    start_state: dict[str, float] | None = None
    abs_start = start + start_frame
    abs_stop = start + local_end
    print(
        f"预加载轨迹：episode={episode_index} frames=[{start_frame}, {local_end}) "
        f"(dataset [{abs_start}, {abs_stop})) source={source} …"
    )
    for index in range(abs_start, abs_stop):
        sample = dataset[index]
        frames.append(frame_to_action_dict(sample, source, names))
        if start_state is None:
            try:
                start_state = frame_to_action_dict(sample, REPLAY_SOURCE_OBSERVATION, names)
            except KeyError:
                start_state = None

    fps = float(getattr(dataset.meta, "fps", 0) or 0)
    if fps <= 0:
        info = getattr(dataset.meta, "info", None)
        if isinstance(info, dict) and info.get("fps"):
            fps = float(info["fps"])
    if fps <= 0:
        fps = 20.0
        print(f"警告：数据集未提供 fps，回退为 {fps}")

    return frames, start_state, fps, start_frame, local_end


def replay_trajectory(
    robot: PiperRobot,
    frames: list[dict[str, float]],
    *,
    period_s: float,
    progress_every: int,
) -> None:
    if not frames:
        raise ValueError("轨迹为空，无法回放")

    t0 = time.monotonic()
    next_tick = t0
    n = len(frames)
    for step, action in enumerate(frames):
        robot.send_action(action)
        if step == 0 or (step + 1) % progress_every == 0 or step + 1 == n:
            elapsed = time.monotonic() - t0
            print(f"replay {step + 1}/{n}  t={elapsed:.2f}s")

        next_tick += period_s
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # 落后则追赶：不累计延迟，从当前时刻重新对齐
            next_tick = time.monotonic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 Piper 从臂上重放一条录制 episode。")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="数据集目录（含 meta/info.json），如 data/lerobot/local/cube_v727_cxn",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo id；默认从 dataset-root 推断为 local/<dirname>",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=REPLAY_SOURCES,
        default=REPLAY_SOURCE_ACTION,
        help=(
            "回放字段："
            "action=录制指令（默认）；"
            "observation=observation.state 从臂反馈（state 为别名）；"
            "obs_joints_action_gripper=关节用 observation、夹爪用 action"
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0, help="episode 内起始帧（含）")
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="episode 内结束帧（不含）；默认到 episode 末尾",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="回放帧率；默认用数据集 meta.fps",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="相对速度倍率（>1 更快），实际 period = 1/(fps*speed)",
    )
    parser.add_argument("--left-can", default="can2")
    parser.add_argument("--right-can", default="can0")
    parser.add_argument(
        "--control-speed",
        type=int,
        default=15,
        help="move-to 阶段 SDK 速度档（0-100）",
    )
    parser.add_argument(
        "--replay-control-speed",
        type=int,
        default=100,
        help="回放阶段 SDK 速度档（0-100）；1:1 回放默认 100",
    )
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.1,
        help="move-to 阶段每 tick 关节限速（弧度）",
    )
    parser.add_argument(
        "--rate-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="回放是否软件限速；默认 --no-rate-limit（1:1 逐帧下发）。需要限速时用 --rate-limit",
    )
    parser.add_argument(
        "--replay-max-joint-step-rad",
        type=float,
        default=0.2,
        help="仅 --rate-limit 时生效：回放每 tick 关节限速（弧度）",
    )
    parser.add_argument(
        "--replay-max-gripper-step-m",
        type=float,
        default=0.002,
        help="仅 --rate-limit 时生效：回放每 tick 夹爪限速（米）",
    )
    parser.add_argument(
        "--tol-deg",
        type=float,
        default=2.5,
        help="初始到位容差（度）；近零位 j2/j3 常有 1~2° 跟不住，默认 2.5",
    )
    parser.add_argument("--period", type=float, default=0.02, help="move-to 控制周期（秒）")
    parser.add_argument("--timeout", type=float, default=60.0, help="move-to 超时（秒）")
    parser.add_argument("--state-wait", type=float, default=5.0)
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="到位后、开始回放前的静置时间（秒）",
    )
    parser.add_argument(
        "--move-to-source",
        choices=("auto", "action", "observation", "state"),
        default="auto",
        help="初始到位目标：auto=优先 observation.state（从臂真实起点），否则用回放首帧",
    )
    parser.add_argument(
        "--skip-move-to",
        action="store_true",
        help="跳过回到首帧，直接从当前位姿开始流式下发（调试用）",
    )
    parser.add_argument(
        "--strict-move-to",
        action="store_true",
        help="move-to 超时则中止；默认只告警并继续回放",
    )
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只加载并打印首/末帧，不连接机械臂",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed 必须 > 0")
    if args.rate_limit and args.replay_max_joint_step_rad <= 0:
        raise SystemExit("--rate-limit 时 --replay-max-joint-step-rad 必须 > 0")
    if args.rate_limit and args.replay_max_gripper_step_m <= 0:
        raise SystemExit("--rate-limit 时 --replay-max-gripper-step-m 必须 > 0")

    dataset_root = Path(args.dataset_root)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"找不到数据集 meta/info.json：{dataset_root}")

    repo_id = infer_repo_id(dataset_root, args.repo_id)
    source = normalize_replay_source(args.source)

    print(f"加载数据集：repo_id={repo_id} root={dataset_root}")
    dataset = load_dataset(repo_id, str(dataset_root), args.video_backend)
    frames, start_state, dataset_fps, start_frame, end_frame = load_episode_trajectory(
        dataset,
        args.episode_index,
        source,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    fps = float(args.fps) if args.fps is not None else dataset_fps
    period_s = 1.0 / (fps * args.speed)

    if args.move_to_source in ("state", "observation"):
        move_to_pose = start_state
        if move_to_pose is None:
            raise SystemExit("数据集无 observation.state，无法使用 --move-to-source observation/state")
        move_to_label = "observation.state"
    elif args.move_to_source == "action":
        # 即使回放是 hybrid/observation，也可用纯 action 首帧做到位
        move_to_pose = frames[0]
        move_to_label = f"replay[{source}]"
    else:
        # auto：优先从臂真实起点，避免 leader action 近零小目标跟不住
        if start_state is not None:
            move_to_pose = start_state
            move_to_label = "observation.state(auto)"
        else:
            move_to_pose = frames[0]
            move_to_label = f"{source}(auto-fallback)"

    print(
        f"episode={args.episode_index}  frames={len(frames)} "
        f"[{start_frame}, {end_frame})  dataset_fps={dataset_fps:.3f} "
        f"replay_fps={fps * args.speed:.3f}  period={period_s:.4f}s"
    )
    print_pose(f"回放首帧({source})", frames[0])
    print_pose(f"到位目标({move_to_label})", move_to_pose)
    if len(frames) > 1:
        print_pose("回放末帧", frames[-1])

    if args.dry_run:
        print("dry-run：未连接机械臂。")
        return

    config = PiperRobotConfig(
        follower_left_port=args.left_can,
        follower_right_port=args.right_can,
        enable_control=True,
        control_speed=args.control_speed,
        max_joint_step_rad=args.max_joint_step_rad,
        cameras={},
    )
    robot = PiperRobot(config)

    try:
        robot.connect()
        print(f"等待关节反馈（最多 {args.state_wait:.1f}s）…")
        obs = wait_for_valid_state(robot, timeout_s=args.state_wait)
        print("当前位姿：")
        print(format_arm(obs, "left"))
        print(format_arm(obs, "right"))

        if not args.skip_move_to:
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
                print(
                    f"警告：初始到位未完全收敛（{exc}）。"
                    "常见于近零位 j2/j3 小角度跟不住；继续回放。"
                )
            if args.settle > 0:
                print(f"静置 {args.settle:.2f}s …")
                time.sleep(args.settle)
        else:
            print("已跳过 move-to，直接回放。")

        # 回放默认不软件限速：按录制 fps 逐帧下发目标（1:1）
        if args.rate_limit:
            robot.config.max_joint_step_rad = args.replay_max_joint_step_rad
            robot.config.max_gripper_step_m = args.replay_max_gripper_step_m
            step_msg = (
                f"rate_limit on  joint_step={args.replay_max_joint_step_rad} "
                f"gripper_step={args.replay_max_gripper_step_m}"
            )
        else:
            robot.config.max_joint_step_rad = float("inf")
            robot.config.max_gripper_step_m = float("inf")
            step_msg = "rate_limit off (1:1 direct command)"

        robot.set_control_speed(args.replay_control_speed)
        print(
            f"开始回放（{step_msg}  control_speed={args.replay_control_speed}  "
            f"period={period_s:.4f}s）… Ctrl+C 可中断。"
        )
        replay_trajectory(
            robot,
            frames,
            period_s=period_s,
            progress_every=max(1, args.progress_every),
        )
        print("回放结束。")
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
