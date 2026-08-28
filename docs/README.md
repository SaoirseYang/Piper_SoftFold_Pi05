# SoftFold

Soft-Fold X-VLA → Piper 跨具身迁移 + **本机录制 / 训练 / 真机推理**。

仓库已从 `cross_embodiment_xvla_transfer` 更名为 **`SoftFold`**。

---

## 录制、训练、推理底层仍是：

| 层 | 依赖 |
|----|------|
| 真机 CAN / 关节读写 | **`piper_sdk`**（pip，不随仓库拷贝） |
| 机器人封装 | 本库 `src/piper_towel_fold`（从原 `piper` 迁入） |
| 数据集 / 训练 / 策略 | **`lerobot`** +（X-VLA）`lerobot/xvla-soft-fold` |

也就是说：**硬件侧继续用 Piper SDK**；本库把原先 `piper` 仓库里的录制/训练/live 入口迁过来，并加上 Soft-Fold 虚拟数据与 PEFT 工具。

请在已安装 `piper_sdk` + `lerobot` 的环境中使用（例如原来的 `conda activate piper`）。

---

## 目录

```
SoftFold/
├── src/
│   ├── softfold/              # IK 虚拟数据 / Soft Prompt / 工作空间映射
│   ├── piper_towel_fold/      # 录制·训练·真机推理（原 piper 封装）
│   └── lerobot_policy_act_piper/  # ACT 插件（可选）
├── scripts/
│   ├── start_recording.sh / start_recording_xvla.sh
│   ├── start_training.sh  / start_training_xvla.sh
│   ├── run_policy_live.sh / run_policy_live_xvla.sh
│   ├── bringup_can.sh / reset_arms.sh / run_move_to_episode_start.sh
│   ├── ik_dry_run.sh / convert_virtual_smoke.sh
│   └── train_stage1.sh / train_stage2.sh
├── tools/                     # move_to_episode_start 等
├── data/                      # EEF→Joint 转换脚本
├── configs/
│   ├── softfold_piper.json    # 本地录制/训练/live 默认配置
│   ├── record_towel_fold_xvla.json
│   ├── workspace_map.json / calibration/station.json
│   └── train_stage1.yaml / train_stage2.yaml
└── docs/PROPOSAL.md
```

---

## 环境

```bash
conda activate piper
cd /home/agilex/code/yjw/SoftFold
export PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH
pip install -e ".[robot]"   # 若尚未装齐依赖
```

---

## 三条主链路（与原先 piper 用法相同）

```bash
# 1) 录制本地真机数据
bash scripts/bringup_can.sh
bash scripts/start_recording.sh configs/softfold_piper.json
# 或 X-VLA 命名相机配置：
bash scripts/start_recording_xvla.sh configs/softfold_piper.json

# 2) 训练（默认会做 *_ojag action 改写）
bash scripts/start_training.sh configs/softfold_piper.json
# 或
bash scripts/start_training_xvla.sh configs/softfold_piper.json --dry-run

# 3) 真机推理（默认先归位到 episode0；execute 默认 false）
bash scripts/run_policy_live.sh configs/softfold_piper.json
# SKIP_MOVE_TO_START=true bash scripts/run_policy_live_xvla.sh configs/softfold_piper.json
```

---

## 远程 GPU（A600）：训练 + 异步推理

GPU 服务器已从 `allinai2` 迁到 **`A600`**（`yangjingwen@allinai1pro`），仓库路径：`/data/yangjingwen/code/SoftFold`。工控机通过 SSH 隧道把 `127.0.0.1:8080` 转到 A600 上的 Policy Server。**checkpoint 留在 A600，不必每次 rsync 回工控机。**

### X-VLA（`softfold_piper.json` / `softfold_piper_xvla_adapt.json`）

```bash
conda activate piper
cd /home/agilex/code/yjw/SoftFold

# 先把含异步脚本的代码同步到 A600（训练完后、推理前至少做一次）
# 适配微调权重用 softfold_piper_xvla_adapt.json；stage1 Soft Prompt 用 softfold_piper.json
bash scripts/sync_code_to_remote.sh configs/softfold_piper_xvla_adapt.json

# 异步推理：三个终端按顺序
# 终端 A（ssh A600 后，或工控机远程拉起）：
bash scripts/start_policy_server_xvla_remote.sh configs/softfold_piper_xvla_adapt.json
# 终端 B：建立推理 channel（保持不关）
bash scripts/ssh_tunnel_policy_server.sh configs/softfold_piper_xvla_adapt.json
# 终端 C：真机 Robot Client（默认 execute=false dry-run）
bash scripts/run_async_policy_client_xvla_remote.sh configs/softfold_piper_xvla_adapt.json
```

确认正常后，把 config 里 `async_inference.execute` 改为 `true` 再控臂。

### pi05（`softfold_piper_pi05.json`）

```bash
conda activate piper
cd /home/agilex/code/yjw/SoftFold

# 1) 同步代码（不含 data/outputs）
bash scripts/sync_code_to_remote.sh configs/softfold_piper_pi05.json

# 2) 上传录制数据集到 A600 相同相对路径
bash scripts/upload_dataset_to_remote.sh configs/softfold_piper_pi05.json

# 3) 远端训练（先同步代码+数据，再 SSH 跑 lerobot-train）
bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json
# DETACHED=true bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json

# 4) 异步推理：必须按顺序，三个终端
#    终端 A（或 ssh A600 后）：启动 Policy Server
bash scripts/start_policy_server_pi05_remote.sh configs/softfold_piper_pi05.json
#    终端 B：建立推理 channel（保持不关）
bash scripts/ssh_tunnel_policy_server.sh configs/softfold_piper_pi05.json
#    终端 C：真机 Robot Client
bash scripts/run_async_policy_client_pi05_remote.sh configs/softfold_piper_pi05.json
```

配置入口：对应 JSON 的 `remote_gpu` / `policy_server` / `async_inference`。
---

## Soft-Fold 虚拟数据（跨具身）

```bash
# IK 可行性（不写盘）
bash scripts/ik_dry_run.sh

# 冒烟生成虚拟关节数据集
bash scripts/convert_virtual_smoke.sh
```

要点：按臂工作空间对齐 + `--orient-mode seed`（见 `configs/workspace_map.json`）。

---

## 建议流程

1. `ik_dry_run` → 小规模 `convert_virtual_smoke`
2. 本机录 15–30 条：`start_recording.sh configs/softfold_piper.json`
3. Soft Prompt 微调：`start_training.sh` 或 `train_stage1.sh`
4. dry-run live → 确认后再把 `policy_live.execute` 改为 `true`
