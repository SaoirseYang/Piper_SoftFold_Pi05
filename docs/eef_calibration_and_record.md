# EEF 统一表示：桌面高度标定 + 录制

目标：用指尖 TCP 的绝对位姿（`eef6d`）作为跨数据集统一表示，并用触桌标定桌面高度，便于和 `xvla-soft-fold` 对齐。

## 文件

| 路径 | 作用 |
|------|------|
| `configs/calibration/station.json` | 双臂底座外参、TCP 偏移、桌面高度 |
| `src/piper_towel_fold/kinematics.py` | Piper URDF FK → `eef6d` |
| `tools/calibrate_table_height.py` | 触桌标定桌面高度 |
| `tools/print_eef_state.py` | 实时打印关节 + EEF |
| `tools/record_eef_episode.py` | 录制 `qpos` + `eef6d`（+可选相机） |

## 1. 改工位外参

编辑 `configs/calibration/station.json`：

- `arms.left/right.base_xyz_m`：各臂底座在**工位世界系**下的位置（先填近似双臂间距即可）
- `tcp_offset_*`：默认指尖（相对 `gripper_base`），一般不用改

世界系约定：原点可自行选（例如两底座中点）；**桌面高度**是世界系 Z。

## 2. 标定桌面高度

主从示教打开，夹爪闭合，指尖轻触桌面（换几个 xy 点）：

```bash
./scripts/calibrate_table_height.sh --left-can can2 --right-can can0 --arm both
```

终端里：

- `Enter`：对当前启用的臂各采一次
- `left` / `right`：只采一侧
- `u`：撤销
- `q`：结束并写入 `table_height_m`

建议 ≥3 个触点；`std` 若 >5mm，检查触桌是否一致或底座外参。

离线检查 FK（无硬件）：

```bash
PYTHONPATH=src python tools/calibrate_table_height.py \
  --dry-run-joints 0,1.2,-1.0,0,0.8,0 --arm right
```

## 3. 实时看 EEF

```bash
./scripts/print_eef_state.sh --left-can can2 --right-can can0
```

标定后会多打 `z_above_table`（相对桌面高度）。

## 4. 录制统一 EEF 数据

```bash
# 无相机短录（对齐/标定用）
./scripts/record_eef_episode.sh \
  --left-can can2 --right-can can0 \
  --fps 20 --task "calib fold reach" \
  --repo-name fold_eef_calib_v1

# 带相机
./scripts/record_eef_episode.sh \
  --left-can can2 --right-can can0 \
  --camera-names cam_top,cam_side,cam_right \
  --camera-indices 18,6,16 \
  --fps 20 --repo-name fold_eef_v1
```

输出目录：`data/eef_episodes/<repo>/<episode_*/`：

- `meta.json`：任务、fps、station 快照、特征名
- `frames.jsonl`：每帧
  - `observation.qpos` / `action.qpos`：14 维绝对关节
  - `observation.eef6d` / `action.eef6d`：20 维世界系（左10+右10 = xyz+rot6d+grip）
  - `observation.eef6d_table`：若已标定，则 `z -= table_height_m`（桌面相对高度，利于跨站对齐）
  - `time_stamp`：相对 episode 起点的墙钟秒

`action.eef6d` = 对 `action.qpos` 做 FK（示教时即 leader 指令关节的末端位姿）。

## 5. `eef6d` 布局（与 soft-fold 单臂 10 维一致）

```
[0:3]   xyz
[3:9]   rot6d（旋转矩阵前两列）
[9]     gripper
```

整机：`left(10) | right(10)`。

## 注意

- 当前 FK 来自 URDF，未接 SDK `EndPose`；与真机若有毫米级偏差，可用触桌多点做常数偏置修正。
- 先标定桌面，再录用于和 soft-fold 对比的数据；对比时优先看 `eef6d_table` 的 z 分布。
- 这套录制是 **EEF 对齐专用格式**（jsonl）。转成与 `cube_pi05_v1` / soft-fold 同构的 LeRobot parquet 是后续待办。
