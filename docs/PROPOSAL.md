# 跨具身低样本迁移方案书

**——基于 X-VLA Soft-Fold 预训练模型到 Piper 双臂的布料折叠部署（不含 DAgger）**

> 本文件是立项方案原文摘要；实现入口见仓库根目录 `README.md`。

## 1. 项目名称

**Lab-to-Lab：基于 IK 虚拟数据集与 Soft Prompt PEFT 的预训练布料折叠 VLA 跨具身迁移**

## 2. 背景与动机

预训练 VLA 在柔性物体操作上已有能力，但迁移到 Piper（关节空间）时面临：

1. **动作空间失配**：源模型 / Soft-Fold 表征偏 EEF，目标需要 Piper 关节（本实现用 14D = 12 关节 + 2 夹爪）。
2. **小样本过拟合**：本地仅 15~30 条。
3. **环境域偏移**：相机 / 光照 / 台面差异。
4. **视觉-动作一致性**：源机械臂外观与目标关节动作错误关联。

## 3. 目标与非目标

**目标**：IK 虚拟数据集工具、Soft Prompt 两阶段 PEFT、混合采样、真机验证。

**非目标**：DAgger；通用任意机器人框架；完全解决视觉域理论问题。

## 4. 技术方案（实现映射）

| 方案步骤 | 本仓库入口 |
|---------|-----------|
| EEF→Joint + 掩码 | `data/eef_to_joint_converter.py` |
| 按臂工作空间对齐 | `src/softfold/workspace_map.py`, `configs/workspace_map.json` |
| 位置优先姿态 | `--orient-mode seed`（Soft-Fold rot6d 与 Piper 腕部不兼容） |
| 数据集合并清单 | `data/make_dataset.py` |
| Soft Prompt / 改造 | `src/softfold/soft_prompt.py`, `model/xvla_modified.py` |
| 阶段1/2 训练 | `train/train_stage1.py`, `train/train_stage2.py` |
| 混合采样 | `train/mixed_dataloader.py` |
| 真机推理 overlay | `deploy/infer_piper.py` |
| Piper 运动学 | `src/softfold/kinematics.py`, `configs/piper_dh.yaml` |
| 工位标定 | `configs/calibration/station.json` |

## 5. 基线

- **B0**：零样本 + 在线 IK（现有 piper softfold live）
- **B1**：全量微调（混合数据）
- **B2**：仅本地 PEFT
- **B3**：本方案（虚拟 + 本地 + Soft Prompt 两阶段 + 混合采样）

## 6. 时间线（约 4~5 周）

1. 环境 / 数据 / IK 冒烟  
2. 虚拟数据集 + 本地采集  
3. 两阶段训练与消融  
4. 真机评估  
5. 整理开源与短文  

## 7. 风险应对（摘要）

- 视觉-动作不一致 → 优先掩码；否则冻视觉编码器  
- IK 成功率低 → 多种初值 / 放宽阈值 / 过滤 episode  
- Soft Prompt 弱 → 备选 LoRA  
- 本地过少 → 增至 30 条、提高本地权重  
- 遗忘严重 → 限制阶段2 解冻与 LR，早停  

## 8. 局限

不含 DAgger；掩码可能丢失臂姿信息；当前聚焦布料折叠 + Piper。
