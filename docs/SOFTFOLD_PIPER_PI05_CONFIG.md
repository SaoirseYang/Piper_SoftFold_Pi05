# `softfold_piper_pi05_rtc.json` 参数说明

配置路径：`configs/softfold_piper_pi05_rtc.json`。

本文件是 SoftFold **pi05 + 异步推理 + LeRobot RTC（Real-Time Chunking）** 的一体化入口：录制、训练、本机 live、远程 GPU Policy Server、SSH 隧道、Robot Client 共用同一 JSON。相对无 RTC 配置（如 `softfold_piper_pi05_rgrasp_mapping.json`），核心差异是：

| 差异点 | 本配置（RTC） | 无 RTC 典型配置 |
|--------|---------------|-----------------|
| `policy_server.rtc.enabled` | `true` | 无 `rtc` 块或 `false` |
| `async_inference.aggregate_fn_name` | `latest_only` | 常为 `weighted_average` |
| 平滑策略 | **Server 侧 RTC prefix guidance**；Client 侧避免再做 chunk 聚合平滑 | Client 侧用 `weighted_average` 等聚合新旧 chunk |

> RTC **只在 Policy Server 侧生效**。Client 用 `latest_only` 是为避免「RTC 已做前缀衔接 + Client 再加权平均」造成双重平滑。

---

## 1. 顶层：任务 / 数据集 / 录制

| 参数 | 当前值 | 含义 | 不同取值的影响 |
|------|--------|------|----------------|
| `task` | `"fold the cloth"` | 语言任务指令，写入观测并随请求发给策略 | 改文案会改变策略条件；应与训练时 episode 语言一致 |
| `dataset_format` | `"lerobot"` | 数据集格式 | 固定用 LeRobot；其它值未在本流水线验证 |
| `repo_id` | `"local/softfold_piper_v2_rgrasp"` | 数据集 ID | 决定 `root/repo_id` 路径；训练默认还会派生 `*_ojag` |
| `root` | `"data/lerobot"` | 数据集根目录 | 本机与远端应保持相同相对路径，便于 rsync |
| `robot_type` | `"piper"` | 机器人类型注册名 | 须与 `PiperRobotConfig` 子类名一致 |
| `no_videos` | `false` | 录制时是否跳过视频轨 | `true`：省磁盘、无 MP4；回放/可视化弱。`false`：存视频（`video_backend` 可读） |
| `prompt_outcome` | `true` | 每条 episode 结束后是否交互询问成败 | `true`：写 `episode_outcomes.jsonl`，供训练过滤。`false`：用 `episode_outcome` 固定标签 |
| `episode_outcome` | `"skip"` | 非交互时的默认标签 | 常用 `success` / `failure` / `unknown` / `skip`；与 `prompt_outcome` 配合 |
| `output_dir` | `"data/raw"` | 非 LeRobot 原始落盘目录（兼容字段） | lerobot 模式下主数据仍在 `root/repo_id` |
| `fps` | `20.0` | 控制/录制主循环频率（Hz） | 须与 `policy_server.fps` / `async_inference.fps` 对齐；改 fps 会改变时间尺度与 RTC delay 换算 |
| `duration` | `null` | 单次录制最长秒数 | `null`：手动停。数值：到时自动结束 |
| `follower_left_can` / `follower_right_can` | `can2` / `can0` | 从臂 CAN 口 | 接错则左右臂互换或通信失败 |
| `leader_left_can` / `leader_right_can` | `can2` / `can0` | 主臂 CAN 口 | `action_source=leader` 时读主臂作 action |
| `camera_width` / `camera_height` | `640` / `480` | 相机分辨率 | 须与训练数据一致，否则域偏移大 |
| `camera_fps` | `30` | 相机采集帧率 | 可高于控制 fps；过高增带宽与 CPU |
| `image_format` | `"jpg"` | 录制图像编码 | `jpg` 省空间；`png` 无损更大 |
| `image_quality` | `95` | JPEG 质量（1–100） | 越高越清晰、体积越大 |
| `action_source` | `"leader"` | action 来源 | `leader`：遥操主臂；`follower`：从臂状态。pi05 训练前默认还会做 `*_ojag` 改写 |
| `recording_notes` | （说明文字） | 人类备注 | 不参与程序逻辑 |
| `cameras[]` | 三路相机 | `name` + `ref`（OpenCV 索引） | 改名需与策略特征/rename_map 对齐；`ref` 错则画面错位 |
| `preprocessing.enabled` | `false` | 在线帧预处理（CLAHE 等） | `true`：录制/推理都过预处理器；须训推一致 |

---

## 2. `training`：训练（重点）

训练入口：`scripts/start_training.sh` / `start_training_remote.sh` → `piper_towel_fold.start_training` → `lerobot-train`（经 `hf_offline`）。

### 2.1 模型与优化

| 参数 | 当前值 | 含义 | 不同取值的影响 |
|------|--------|------|----------------|
| `policy_type` | `"pi05"` | 策略类型 | 决定 CLI `--policy.type` 与 pi05 专用选项；勿与 xvla/act 混用同一套超参 |
| `job_name` | `"pi05_softfold_piper_v2_rgrasp_ojag_mapping"` | 任务名 / 日志与远端 tmux 名片段 | 仅命名；改名不影响已有 checkpoint 路径除非同步改 `output_dir` |
| `output_dir` | `"outputs/train/..."` | checkpoint 输出根 | 远端训练产物在 A600 此相对路径下 |
| `pretrained_path` | `"lerobot/pi05_base"` | 预训练权重 | Hub ID 或本地路径；换底座会显著改变收敛与表现 |
| `dtype` | `"bfloat16"` | 训练精度 | `bfloat16`：省显存、A600 友好。`float32`：更稳更慢更吃显存 |
| `compile_model` | `true` | `torch.compile` | `true`：首步慢、稳态常更快。`false`：调试更易、可能更慢 |
| `gradient_checkpointing` | `true` | 激活重计算换显存 | `true`：能开更大 `batch_size`，步进略慢。`false`：更快但易 OOM |
| `freeze_vision_encoder` | `false` | 冻结视觉塔 | `true`：只训其余部分，防过拟合、省算力，适应力可能变差。`false`：全量微调视觉 |
| `train_expert_only` | `false` | 只训 expert 子网络 | `true`：参数更少、更快，表达力可能不足。`false`：常规全量（在未冻结部分内） |
| `normalization_mapping` | 见下表 | 各模态归一化方式 | 须与 pi05 预训练习惯一致；乱改会导致尺度错乱 |
| `device` | `"cuda"` | 训练设备 | `cuda` / `cpu`（CPU 仅调试） |
| `steps` | `20000` | 优化步数 | 更大：更拟合、过拟合风险↑。更小：欠拟合 |
| `batch_size` | `16` | 批大小 | 更大：梯度稳、显存↑。OOM 时优先降此项或开 checkpointing |
| `log_freq` | `50` | 日志间隔（步） | 越小日志越密 |
| `save_freq` | `5000` | 存盘间隔（步） | 越小 checkpoint 越多占盘；`last` 仍会维护 |
| `wandb_enable` | `false` | Weights & Biases | `true` 需本机 wandb 登录 |
| `video_backend` | `"pyav"` | 读数据集视频后端 | 与录制一致即可；`torchcodec` 等视环境 |
| `outcome_filter` | `"exclude-failures"` | 按成败筛 episode | 见 §2.3 |
| `include_unknown` | `false` | 过滤时是否纳入 `unknown` | 仅对 `exclude-failures` 有意义 |
| `exclude_unlabeled` | `true` | 是否丢掉无标签 episode | `true`：必须有 `episode_outcomes.jsonl` |
| `push_to_hub` | `false` | 是否推送策略到 Hub | 本地部署保持 `false` |

**`normalization_mapping` 取值：**

| 键 | 当前值 | 含义 |
|----|--------|------|
| `ACTION` | `QUANTILES` | 动作用分位数归一化（pi05 推荐） |
| `STATE` | `QUANTILES` | 本体状态同上 |
| `VISUAL` | `IDENTITY` | 图像不做该层归一化（由模型预处理处理） |

其它可选：`MEAN_STD`、`QUANTILE10`。与预训练不一致时，微调初期 loss 常异常。

### 2.2 默认隐式行为（JSON 未写出但会生效）

| 行为 | 默认 | 说明 |
|------|------|------|
| `action_compose` | `obs_joints_action_gripper` | 训练前自动生成 `*_ojag`：关节监督←`observation.state`，夹爪←原 `action`。设 `"off"` / `null` 可关闭 |
| `action_compose_overwrite` | `false` | `true` 时强制重建 `*_ojag` |
| `report_loss_on_finish` | `true` | 训完根据 train.log 打印收敛摘要 |

### 2.3 `outcome_filter`

| 取值 | 效果 |
|------|------|
| `exclude-failures`（当前） | 只用 `success`；`unknown` 仅当 `include_unknown=true` |
| `success-only` | 仅 success |
| `failures-only` | 仅 failure（少见） |
| `none` / `all` | 不过滤，用全部 episode |

### 2.4 `image_transforms`（配置里有，接线注意）

| 参数 | 当前值 | 含义 | 影响 |
|------|--------|------|------|
| `enable` | `true` | 是否启用图像增广 | 意图：训练时 ColorJitter 提升光照鲁棒 |
| `max_num_transforms` | `1` | 每次最多应用几种 transform | 限制增广强度 |
| `random_order` | `false` | 是否乱序应用 | 一般保持 false |
| `tfs.ColorJitter.*` | brightness/contrast/saturation `0.3`，hue `0.05` | 颜色抖动幅度 | 过大：域偏移；过小：增广弱 |

> **注意：** 当前 SoftFold `start_training.py` 组装的 `lerobot-train` 命令**尚未自动转发** `training.image_transforms`。该块保留在 JSON 中作为约定/待接线项；若要生效，需在训练命令中补上对应 `--dataset.image_transforms.*`（或改 `start_training.py`）。

### 2.5 训练侧调参建议（简表）

| 目标 | 优先动哪些 |
|------|------------|
| OOM | ↓`batch_size`；保持 `gradient_checkpointing=true`；`dtype=bfloat16` |
| 过拟合 / 真机泛化差 | ↑数据量；确认 `outcome_filter`；考虑 `freeze_vision_encoder=true` 做对比 |
| 收敛慢 | 查 `pretrained_path`、normalization、是否误关 `*_ojag` |
| 只要更快出 ckpt 做推理试验 | ↓`steps` / ↑`save_freq` 更密存盘，但勿指望最终性能 |

---

## 3. `policy_live`：本机同步推理（非异步）

用于 `run_policy_live.sh` 等**本机加载权重**路径。异步流水线主要看 §4–§6；但 `policy_path` / `dataset_root` / `num_inference_steps` 会被 Policy Server preload **回退读取**。

| 参数 | 当前值 | 含义 | 不同取值的影响 |
|------|--------|------|----------------|
| `policy_path` | `.../checkpoints/last/pretrained_model` | 权重目录 | Server `preload` 未单独写 `policy_path` 时用此路径 |
| `repo_id` | `..._mapping` | live 用数据集 ID（stats 等） | 可与录制 repo 不同（mapping 变体） |
| `dataset_root` | `data/lerobot/local/softfold_piper_v2_rgrasp` | 数据集根（含 meta/stats） | 错路径会导致归一化/特征不匹配 |
| `video_backend` | `"pyav"` | 读视频后端 | 与训练一致 |
| `device` | `"cuda"` | 本机推理设备 | 本机无 GPU 时异步应走远程 Server |
| `inference_dtype` | `"bfloat16"` | 推理精度 | 与训练 dtype 对齐更稳 |
| `compile_model` | `false` | 推理 compile | 异步 Server 另有独立字段；live 开 compile 首包延迟大 |
| `num_inference_steps` | `10` | 流匹配/扩散去噪步数 | **↑步数：更精细更慢；↓步数：更快、轨迹可能更糙**。异步时以 `policy_server` 为准 |
| `execute` | `true` | 是否真下发关节 | `false`：dry-run 只打印 |
| `duration` | `180.0` | 最长运行秒数 | |
| `fps` | `20.0` | 控制频率 | 与录制一致 |
| `control_speed` | `40` | Piper 底层运动速度档位 | 越大越快越冲；过大不安全 |
| `max_joint_step_rad` | `0.025` | 每控制周期关节最大步进（rad） | **安全限幅**：越小越钝、越不易甩臂 |
| `max_gripper_step_m` | `0` | 夹爪步进上限（米） | `0`：本周期不额外限制夹爪步进（仍受其它逻辑约束） |
| `gripper_effort` | `1000` | 夹爪力矩/力度 | 过大易夹坏物体/堵转 |
| `smoothing_alpha` | `0.20` | EMA：`α*new+(1-α)*old` | **↑α：更跟指令、更抖；↓α：更平滑、更滞后** |
| `pace_by_reach` | `false` | 是否按到位再推进 chunk | `true`：跟踪慢时少积压；可能拖慢任务 |
| `advance_threshold_rad` | `0.08` | 到位判定阈值阈值 | 仅 `pace_by_reach` 时有意义 |
| `max_hold_steps` | `30` | 等待到位最长步数 | |
| `print_every` | `1` | 打印间隔 | |
| `log_jsonl` | `""` | 动作日志路径 | 空=不写 |

---

## 4. `policy_server`：异步 Policy Server（重点）

在 **A600** 上跑：`scripts/start_policy_server_pi05.sh`。负责加载 pi05、（可选）RTC、按观测吐出 action chunk。

| 参数 | 当前值 | 含义 | 不同取值的影响 |
|------|--------|------|----------------|
| `host` | `"0.0.0.0"` | 监听地址 | `0.0.0.0`：允许 SSH 隧道对端连接。勿对公网裸奔 |
| `port` | `8081` | 服务端口 | 须与 `remote_gpu.tunnel_*` / `async_inference.server_address` 一致；多实例时端口递增 |
| `fps` | `20` | Server 侧环境步频假设 | 参与 `inference_delay_steps` 默认换算：`round(latency*fps)` |
| `inference_latency` | `0.2` | 目标推理延迟（秒） | 未显式写 `rtc.inference_delay_steps` 时：`delay≈round(0.2*20)=4`。设大：RTC 认为延迟更长 |
| `obs_queue_timeout` | `15` | 观测队列等待超时（秒） | 过小：网络抖动易断流；过大：卡死时恢复慢 |
| `inference_dtype` | `"bfloat16"` | Server 推理精度 | |
| `compile_model` | `false` | Server 是否 compile | 远程服务建议 false，避免启动极慢/难调试 |
| `num_inference_steps` | `10` | **去噪步数（延迟与质量主旋钮）** | 常见扫 `5/10/20`：步少→chunk 更快、队列更不易空；步多→更准但 `chunk_size_threshold` 要配合 |
| `preload_at_startup` | `true` | 启动时预加载权重 | `true`：READY 前要等 `All keys loaded`；`false`：首连才加载，首包极慢 |

未单独写时，preload 会回退读 `policy_live.policy_path` / `dataset_root`，以及 `async_inference.actions_per_chunk` 等。

---

## 5. `policy_server.rtc`：Real-Time Chunking（重点）

基于 Physical Intelligence RTC：新 chunk 生成时，用**上一 chunk 尚未执行的前缀**做 flow-matching **prefix guidance**，减轻异步换 chunk 时的动作跳变。

查找顺序：`policy_server.rtc` → 否则 `async_inference.rtc`。无 `rtc` 或 `enabled=false` → 行为与旧异步配置相同。

| 参数 | 当前值 | 含义 | 不同取值的影响 |
|------|--------|------|----------------|
| `enabled` | `true` | 总开关 | `false`/缺省：不 guidance。A/B 对比时只改此项并同步改 Client `aggregate_fn_name` |
| `execution_horizon` | `10` | 前缀权重作用的时间范围（步） | **↑**：更长区间与旧轨迹对齐，衔接更硬、新意图更受压制。**↓**：更快采纳新 chunk，可能出现接缝 |
| `max_guidance_weight` | `10.0` | guidance 强度上限 | **↑**：更强制贴旧前缀，可能欠响应。**↓**：引导弱，接近无 RTC。必须 `>0` |
| `prefix_attention_schedule` | `"EXP"` | 前缀权重形状 | 见下表 |
| `inference_delay_steps` | `4` | 视为「推理延迟」的步数，强引导区长度 | **应 ≈ 真实 RTT+推理延迟（步）**。过小：引导不够，接缝仍在；过大：过度约束已过时前缀。不设则用 `round(inference_latency*fps)` |
| `debug` | `false` | RTC debug tracker | `true`：记录中间量，开销↑，仅排查用 |
| `debug_maxlen` | （默认 100） | debug 环形缓冲长度 | 仅 `debug=true` |

**`prefix_attention_schedule`：**

| 取值 | 权重形态 | 典型影响 |
|------|----------|----------|
| `EXP`（当前） | 在 delay→horizon 区间近似指数衰减 | SoftFold/PI 常用；前段贴紧、后段放开 |
| `LINEAR` | 线性从 1→0 | 过渡更匀，引导略弱于 EXP |
| `ONES` | `[0,horizon)` 全 1，其后 0 | 整段强约束，最「粘」旧轨迹 |
| `ZEROS` | 仅前 `inference_delay` 为 1，其余 0 | 几乎只锁延迟窗，引导最弱 |

**与 Client 的配合（重要）：**

| Server RTC | 建议 `async_inference.aggregate_fn_name` | 原因 |
|------------|------------------------------------------|------|
| ON | `latest_only`（本配置） | 避免双重平滑 |
| OFF | `weighted_average` 等 | 用 Client 聚合弥补接缝 |

---

## 6. `remote_gpu`：远端与隧道

| 参数 | 当前值 | 含义 | 影响 |
|------|--------|------|------|
| `ssh_host` | `"A600"` | SSH Host（`~/.ssh/config`） | 所有 remote 脚本读此字段 |
| `gpu_repo_root` | `/data/yangjingwen/code/SoftFold` | 远端仓库根 | 须与 rsync 目标一致 |
| `conda_env` | `"piper"` | 远端 conda 环境名 | |
| `use_ssh_tunnel` | `true` | Client 是否经本机隧道连 Server | 工控机无直连 GPU 网时必须 true |
| `tunnel_local_port` | `8081` | 本机监听端口 | Client 连 `127.0.0.1:此端口` |
| `tunnel_remote_host` | `"127.0.0.1"` | 隧道远端绑定 | Server 在 GPU 本机回环 |
| `tunnel_remote_port` | `8081` | 远端 Server 端口 | 与 `policy_server.port` 一致 |

---

## 7. `async_inference`：Robot Client 异步推理（重点）

工控机：`run_async_policy_client_pi05*.sh`。采图 →（可选 JPEG）→ gRPC 发 Server → 收 chunk → 限幅/平滑 → 下发。

| 参数 | 当前值 | 含义 | 不同取值的影响 |
|------|--------|------|----------------|
| `server_address` | `"127.0.0.1:8081"` | Client 连接地址 | 经隧道时永远是本机；端口跟 tunnel |
| `policy_type` | `"pi05"` | 告知 Server 的策略类型 | 须与权重一致 |
| `policy_device` | `"cuda"` | **Server** 上策略设备（经协议传递） | 不是工控机设备 |
| `actions_per_chunk` | `50` | 每个推理 chunk 含多少步动作 | **↑**：少触发推理、带宽/算力更省，但计划更「旧」。**↓**：更频繁重规划、更跟手、Server 更忙。须 ≤ 模型 chunk 容量 |
| `chunk_size_threshold` | `0.30` | 队列剩余比例 ≤ 此值才发新观测 | **核心异步旋钮**。见下 |
| `aggregate_fn_name` | `"latest_only"` | 新旧 chunk 时间步重叠时的聚合 | 见下表；**RTC 开时用 latest_only** |
| `execute` | `true` | 是否真实控臂 | 首次务必先 `false` dry-run |
| `duration` | `180.0` | Client 最长运行秒 | |
| `network_benchmark_samples` | `5` | 启动前测 RTT 次数 | `0`：跳过；`>0`：打印延迟参考，便于设 RTC delay |
| `log_latency` | `true` | 是否打延迟日志 | |
| `latency_log_jsonl` | `""` | 延迟 JSONL 路径 | 空=不落盘 |
| `fps` | `20.0` | Client 控制环频率 | 与录制/Server 对齐 |
| `control_speed` | `40` | 关节指令速度档 | 同 policy_live |
| `max_joint_step_rad` | `0.025` | 关节步进限幅 | **安全第一旋钮** |
| `max_gripper_step_m` | `0.000` | 夹爪步进限幅 | `0` 表示不按该字段额外限幅 |
| `smoothing_alpha` | `0.30` | 执行侧 EMA | 与 RTC/`aggregate` 叠加；RTC+`latest_only` 时仍可轻微抑抖。**↑更跟手更抖** |
| `obs_image_compression` | `"jpeg"` | 观测图压缩 | `jpeg`：隧道带宽友好。`none`：无损但更慢易堵 |
| `obs_jpeg_quality` | `75` | JPEG 质量 | **↓**：更小更快、细节差。**↑**：更清晰、延迟/带宽↑。常见 60–85 |
| `debug_visualize_queue_size` | `false` | 可视化动作队列长度 | 调 `chunk_size_threshold` 时有用 |

### 7.1 `chunk_size_threshold` 详解

条件（伪代码）：

```text
ready = (queue_size / action_chunk_size) <= chunk_size_threshold
```

| 取值倾向 | 行为 | 适用 |
|----------|------|------|
| 较小（如 `0.2`） | 队列剩得更少才请求 → 推理触发更晚/更少 | Server 慢、想减少重叠；空队列风险↑（idle） |
| 中等（当前 `0.30`） | 平衡 | 默认起点 |
| 较大（如 `0.5`–`0.7`） | 队列还较满就请求 → 更频繁推理、重叠更多 | 低延迟、RTC/聚合负责衔接；Server 负载↑ |

多实例扫参时常与 `num_inference_steps`、`smoothing_alpha` 组成 grid（见 `docs/PI05_PIPELINE.md`）。

### 7.2 `aggregate_fn_name`

重叠时间步上：`new` 与队列中 `old` 的合并方式（LeRobot 内置）：

| 名称 | 公式 | 影响 |
|------|------|------|
| `latest_only`（当前） | `new` | 完全采用新 chunk；**配 RTC** |
| `weighted_average` | `0.3*old + 0.7*new` | 偏新；**无 RTC 默认** |
| `average` | `0.5*old + 0.5*new` | 对称折中 |
| `conservative` | `0.7*old + 0.3*new` | 偏旧，更钝、更少跳变 |

### 7.3 异步侧「延迟–质量」联动

| 若出现… | 可尝试 |
|---------|--------|
| 经常 `Waiting for remote actions` / 队列空 | ↑`chunk_size_threshold`；↓`num_inference_steps`；↓`obs_jpeg_quality` 或确认隧道；↑`actions_per_chunk`（权衡） |
| 换 chunk 时关节突变（无 RTC） | 开 RTC + `latest_only`；或无 RTC 下用 `weighted_average`/`conservative` |
| RTC 开着仍肉眼接缝 | ↑`execution_horizon` / `max_guidance_weight`；校准 `inference_delay_steps`≈实测延迟步数 |
| 动作发黏、不跟场景 | ↓ RTC 强度或 delay；或关 RTC 对比；↓`smoothing_alpha` 的「过平滑」错觉时改为检查 horizon |
| 手臂过冲不安全 | ↓`max_joint_step_rad`、`control_speed`；`execute=false` 先看日志 |

---

## 8. `deployment`：文档型部署说明

| 参数 | 含义 |
|------|------|
| `notes` | 人类可读部署说明（不参与运行） |
| `gpu_server.ssh_host` / `repo_root` / `run_script` | 约定 GPU 侧如何起 Server |
| `robot_ipc.repo_root` / `tunnel_script` / `client_script` | 约定工控机隧道与 Client 脚本 |

实际自动化以 `remote_gpu` + `scripts/pi05_pipeline.sh` / `deploy_async_inference.sh` 为准。

---

## 9. 参数归属速查

| 你在调… | 改哪里 |
|---------|--------|
| 训多久、显存、是否冻视觉 | `training.*` |
| 用哪些成败 episode | `training.outcome_filter` 等 + 录制 `prompt_outcome` |
| 去噪步数 / 预加载 / 端口 | `policy_server.*` |
| RTC 开否与衔接强度 | `policy_server.rtc.*` |
| 何时请求下一块、队列策略 | `async_inference.chunk_size_threshold` / `actions_per_chunk` |
| 新旧 chunk 聚合 | `async_inference.aggregate_fn_name` |
| 真机平滑与安全限幅 | `async_inference.smoothing_alpha` / `max_joint_step_rad` / `control_speed` |
| 图像带宽 | `obs_image_compression` / `obs_jpeg_quality` |
| 本机同步 live（非远程） | `policy_live.*` |

---

## 10. 相关文档与入口

| 文档 / 脚本 | 用途 |
|-------------|------|
| `docs/PI05_PIPELINE.md` | 上传 → 训练 → 多实例 deploy / select |
| `docs/README.md` | SoftFold 总览 |
| `configs/deploy_async_pi05_rtc_ab.yaml` | 同一 ckpt 有无 RTC A/B |
| `bash scripts/pi05_pipeline.sh …` | 推荐一键流水线 |

代码锚点：

- RTC 解析：`src/piper_towel_fold/async_rtc.py`
- Server：`src/piper_towel_fold/start_async_policy_server.py` / `async_policy_server.py`
- Client：`src/piper_towel_fold/start_async_policy_client.py` / `async_robot_client.py`
- 训练：`src/piper_towel_fold/start_training.py`
