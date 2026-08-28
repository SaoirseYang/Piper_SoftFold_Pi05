# SoftFold pi05 远程流水线手册（上传 → 训练 → 多实例异步推理）

面向：**工控机本地录制** + **A600 训练/Policy Server** + **工控机 SSH 隧道 + Robot Client**。

仓库默认路径：

| 机器 | 路径 |
|------|------|
| 工控机 | `/home/agilex/code/yjw/SoftFold` |
| A600 | `/data/yangjingwen/code/SoftFold`（见 config 的 `remote_gpu.gpu_repo_root`） |

旧的「三终端」流程仍然可用；本文以新流水线 `scripts/pi05_pipeline.sh` 为主。

---

## 0. 前置条件

```bash
conda activate piper
cd /home/agilex/code/yjw/SoftFold
# SSH 免密到 A600（Host 名需与 config 里 remote_gpu.ssh_host 一致，一般是 A600）
ssh A600 'hostname; nvidia-smi -L'
```

常用配置：

| 文件 | 用途 |
|------|------|
| `configs/softfold_piper_pi05_rgrasp.json` | 录制 / 训练 / 单实例推理基座 |
| `configs/deploy_async_pi05.yaml` | **多实例部署清单**（ckpt + 超参） |
| `configs/deploy_async_pi05_rgrasp.yaml` | rgrasp 示例清单 |

查看 A600 磁盘与 GPU：

```bash
bash scripts/pi05_pipeline.sh resources configs/softfold_piper_pi05_rgrasp.json
```

环境变量（可选）：

| 变量 | 默认 | 含义 |
|------|------|------|
| `DISK_CHECK` | `true` | 上传/部署前检查磁盘 |
| `DISK_MIN_FREE_GIB` | `20` | 远端最少剩余 GiB |
| `GPU_CHECK` | `true` | 训练/部署前检查显存 |
| `GPU_MIN_FREE_MIB` | 训练 `16000` / 部署 `8000` | 最少空闲显存 |
| `MAX_INSTANCES` | `4`（或 YAML `max_instances`） | 多实例上限 |
| `SKIP_RESOURCE_CHECK` | `false` | 强制跳过资源检查 |
| `SYNC_CONFIG` | `true` | 上传时是否同步 config |
| `CUDA_VISIBLE_DEVICES` | YAML 可写 | 远端 server 用哪张卡 |

---

## 1. 录制（工控机）

```bash
bash scripts/bringup_can.sh
bash scripts/start_recording.sh configs/softfold_piper_pi05_rgrasp.json
```

录制完成后，本地应有：

```text
data/lerobot/<repo_id>/meta/info.json
```

`repo_id` / `root` 以你的 JSON config 为准。

---

## 2. 上传数据集 + config

```bash
bash scripts/pi05_pipeline.sh upload configs/softfold_piper_pi05_rgrasp.json
```

会做：

1. 检查 A600 磁盘余量  
2. rsync **当前 JSON config** 到远端同相对路径  
3. rsync 数据集（若有 `*_ojag` 一并上传）

仅上传数据、不同步 config（旧行为）：

```bash
SYNC_CONFIG=false bash scripts/upload_dataset_to_remote.sh configs/softfold_piper_pi05_rgrasp.json
```

---

## 3. 远端训练（本地挂起 tmux）

推荐：

```bash
bash scripts/pi05_pipeline.sh train configs/softfold_piper_pi05_rgrasp.json
```

等价于 `DETACH_MODE=tmux`：在 A600 建 session（名类似 `sf-train-<job_name>`），本地可断开。

查看 / 附着：

```bash
# 脚本打印的 attach 提示，或：
ssh A600 -t 'tmux ls'
ssh A600 -t 'tmux attach -t sf-train-<job_name>'
```

兼容旧方式：

```bash
bash scripts/pi05_pipeline.sh train-fg configs/...      # 前台交互
bash scripts/pi05_pipeline.sh train-nohup configs/...   # nohup 后台
```

训练产物在 **A600**（不必拉回工控机）：

```text
outputs/train/<job>/checkpoints/<step>/pretrained_model
outputs/train/<job>/checkpoints/last/pretrained_model
```

---

## 4. 一键多实例部署（deploy）

### 4.1 编辑 YAML

编辑例如 `configs/deploy_async_pi05.yaml`：

```yaml
base_config: configs/softfold_piper_pi05_rgrasp.json   # 或你的 mapping json
base_port: 8080
max_instances: 4
cuda_visible_devices: "0"   # 同卡多实例注意显存

checkpoints:
  - last
  - 050000

grid:
  num_inference_steps: [10]
  chunk_size_threshold: [0.50]
  smoothing_alpha: [0.20, 0.30]
```

两种选参（**有 `instances` 时优先用它**）：

1. **`checkpoints` + `grid`**：笛卡尔积  
2. **`instances`**：逐条指定 ckpt + 三个参数  

三个参数含义：

| 字段 | 生效位置 |
|------|----------|
| `num_inference_steps` | A600 Policy Server |
| `chunk_size_threshold` | 工控机 Robot Client |
| `smoothing_alpha` | 工控机 Robot Client |

ckpt 写法：`last` / `"050000"`（**步数请加引号**）/ 完整相对路径均可。

> 注意：YAML 1.1 会把未加引号的 `050000` 解析成八进制整数 **20480**，导致去找不存在的 `checkpoints/20480`。请写成 `"050000"`。解析器也会尽量保留 leading-zero 字符串，但仍建议显式加引号。


### 4.2 执行 deploy

```bash
bash scripts/pi05_pipeline.sh deploy --yaml configs/deploy_async_pi05.yaml
```

**这一行跑完之后，系统已经处于「可选择推理」状态**，具体做了：

1. 检查 A600 磁盘 / GPU  
2. 按 YAML 生成派生 config → `runs/async_deploy/<run_id>/cfgs/*.json`  
3. 写 `runs/async_deploy/<run_id>/manifest.json`（并复制一份 `deploy.yaml`）  
4. rsync 派生 config 到 A600 同相对目录  
5. 每个实例：远端 `tmux` 拉起 Policy Server（端口 `base_port + id`）  
6. 每个实例：本地后台 SSH tunnel（`127.0.0.1:<port>` → A600 同端口）  

终端末尾会提示 `run_id` 以及下一步命令。

查看当前部署：

```bash
bash scripts/pi05_pipeline.sh list          # 最新一次
bash scripts/pi05_pipeline.sh list <run_id>
bash scripts/pi05_pipeline.sh status        # list + 远端磁盘/GPU + tunnel 是否活着
```

示例表格字段：`id / port / steps / chunk / alpha / status / preload / srv / tun / ckpt_path`。

`status` 含义（**READY 以远端 preload 日志为准**，不是仅看端口是否通）：

| status | 含义 |
|--------|------|
| `READY` | 远端日志已出现 **`All keys loaded successfully!`**，且本地 tunnel / 端口正常 |
| `DEPLOYING` | Policy Server tmux 在跑，尚未出现上述加载完成日志 |
| `NO_CKPT` | A600 上没有该 checkpoint |
| `CKPT_PARTIAL` | 目录在但缺权重/config |
| `NO_SERVER` | tmux 未运行（尚未启动或已崩溃退出） |
| `FAILED` | 远端日志有 Traceback / OOM 等错误 |
| `NO_TUNNEL` | preload 已完成，但本地 SSH tunnel 未建立 |

表格列 `preload`：`done` = 已见完成日志，`loading` = 部署中，`failed` = 加载失败。

查看远端加载进度：

```bash
ssh A600 'tail -f /data/yangjingwen/code/SoftFold/runs/async_deploy/<run_id>/server_0.log'
```

完成时应出现：`All keys loaded successfully!`

### 4.3 有无 RTC 对比（同一 last）

```bash
bash scripts/pi05_pipeline.sh deploy --yaml configs/deploy_async_pi05_rtc_ab.yaml
bash scripts/pi05_pipeline.sh select   # 0=无 RTC，1=有 RTC
```

| id | name | base_config | RTC | client aggregate |
|----|------|-------------|-----|------------------|
| 0 | `last_no_rtc` | `rgrasp_mapping.json` | off | `weighted_average` |
| 1 | `last_rtc` | `pi05_rtc.json` | on | `latest_only` |

两边共用同一 `checkpoint: last`。`instances` 支持每条自己的 `base_config` / `rtc_enabled` / `aggregate_fn_name`。

---

## 5. deploy 之后：选实例跑真机 Client（你要的下一步）

**同一时刻真机只连一个 Client**；多个 Server+Tunnel 可以常驻，用 `select` 切换。

```bash
# 交互选 id
bash scripts/pi05_pipeline.sh select

# 或指定某次 run
bash scripts/pi05_pipeline.sh select 20260826_110000

# 非交互（脚本/批处理）
SELECT_ID=0 bash scripts/deploy_async_inference.sh select
```

`select` 会：

1. 打印实例表  
2. 让你输入 `id`（或 `q` 退出）  
3. 若对应 tunnel 挂了会尝试重建  
4. 调用 `run_async_policy_client_pi05_remote.sh`（默认会先归位到 episode0）  

跳过归位：

```bash
SKIP_MOVE_TO_START=true SELECT_ID=1 bash scripts/deploy_async_inference.sh select
```

控臂开关在派生 config 的 `async_inference.execute`（继承自 `base_config`）。首次建议先 `false` dry-run，确认链路后再改 base 或派生 json 为 `true`。

换另一组参数：再跑一次 `select`，选别的 `id` 即可（不必重新 deploy，除非改了 YAML）。

---

## 6. 收工：停掉多实例

```bash
# 只停最近一次 deploy（远端 tmux + 本地 tunnel）
bash scripts/pi05_pipeline.sh stop
bash scripts/pi05_pipeline.sh stop <run_id>

# 一次性停掉 A600 上所有 deploy（所有 sf-srv-* tmux）+ 本地全部 deploy tunnel
bash scripts/pi05_pipeline.sh stop-all
```

`stop-all` 会：

- 在 A600 上 `tmux kill-session` 所有名为 `sf-srv-*` 的 Policy Server
- 在工控机上杀掉 `runs/async_deploy/*/pids/tunnel_*.pid` 记录的全部 SSH tunnel

不会停训练 session（`sf-train-*`）。若需手动清远端：

```bash
ssh A600 'tmux ls | grep sf-srv'
```

---

## 7. 推荐完整一天流程（抄这段）

```bash
conda activate piper
cd /home/agilex/code/yjw/SoftFold

# （录制完成后）
bash scripts/pi05_pipeline.sh resources configs/softfold_piper_pi05_rgrasp.json
bash scripts/pi05_pipeline.sh upload    configs/softfold_piper_pi05_rgrasp.json
bash scripts/pi05_pipeline.sh train     configs/softfold_piper_pi05_rgrasp.json
# ... 等训练结束，确认 A600 上有目标 checkpoint ...

# 编辑 configs/deploy_async_pi05.yaml 后：
bash scripts/pi05_pipeline.sh deploy --yaml configs/deploy_async_pi05.yaml
bash scripts/pi05_pipeline.sh status
bash scripts/pi05_pipeline.sh select          # 选 id → 真机跑
# 换参数再 select 另一个 id ...

bash scripts/pi05_pipeline.sh stop            # 收工
```

---

## 8. 旧方式（单实例三终端，仍保留）

```bash
bash scripts/sync_code_to_remote.sh configs/softfold_piper_pi05_rgrasp.json
bash scripts/start_policy_server_pi05_remote.sh configs/softfold_piper_pi05_rgrasp.json   # 终端 A
bash scripts/ssh_tunnel_policy_server.sh configs/softfold_piper_pi05_rgrasp.json         # 终端 B，保持
bash scripts/run_async_policy_client_pi05_remote.sh configs/softfold_piper_pi05_rgrasp.json  # 终端 C
```

改动前的脚本备份：`scripts/backup_20260826/`。

---

## 9. 产物与目录

```text
runs/async_deploy/<run_id>/
  deploy.yaml          # 本次使用的 YAML 副本
  manifest.json        # 实例表（port/tmux/ckpt/超参）
  cfgs/*.json          # 每个实例的派生完整 config
  pids/tunnel_*.pid    # 本地 tunnel pid
```

远端对应：

```text
$gpu_repo_root/runs/async_deploy/<run_id>/cfgs/...
```

`runs/` 已在 `.gitignore` 中。

---

## 10. 常见问题

**Q: deploy 报实例数超限？**  
笛卡尔积过大。缩小 YAML 的 `checkpoints`/`grid`，或提高 `max_instances` / `MAX_INSTANCES`。

**Q: GPU 检查失败、只起了一部分？**  
同卡显存不够。先 `stop`，减少实例数，或改 `cuda_visible_devices` 分卡（需机器确有多卡且逻辑支持）。

**Q: select 连不上 server？**  

```bash
bash scripts/pi05_pipeline.sh status
ssh A600 'tmux ls; tmux capture-pane -pt sf-srv-<...>'
```

确认 server 已 preload 完成，且本地 tunnel 为 `OK`。

**Q: 只想改超参、不改 ckpt？**  
用 YAML `instances` 写多条同一 `checkpoint`、不同 `smoothing_alpha` / `chunk_size_threshold` / `num_inference_steps`，再 `deploy` → `select`。

**Q: `base_config` 指向的 json 不存在？**  
`deploy_async_pi05.yaml` 里的 `base_config` 必须是仓库内真实文件；改对路径后再 deploy。
