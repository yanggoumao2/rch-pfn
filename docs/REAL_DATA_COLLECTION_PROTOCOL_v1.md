# RCH-PFN 多轨迹采集与系统不重叠划分协议 v1

**版本**：v1.0  
**制定日期**：2026-10-28  
**适用对象**：LabUtopia 插管任务、官方 `calibrated_rgb` recorder、RCH-PFN deployable 数据路径  
**运行位置**：`hp-workstation:/home/yxy/data/projects/rch-pfn`  
**参考基准**：`candidate05-45`

## 1. 目的与证据边界

本协议用于建立第一批可训练、可复现、可审计的真实 PFN 数据集，重点回答三件事：

1. 双相机历史是否能支持当前状态、关系变化和未来成功的联合预测；
2. q0→q1 历史重读机制是否对真实接触、滑移、错位和失败历史产生可测变化；
3. 在未见过的物理系统上，模型是否仍能保持状态估计和成功预测能力。

本批数据不直接证明物理闭环控制成功，也不把单条成功轨迹或 importer 通过写成模型效果。所有结论必须区分 deployable 与 recorder_oracle；后者只用于上限和消融。

## 2. 固定采集配置

以 candidate05-45 为成像和时间基准：

| 项目 | 规定 |
|---|---|
| recorder | 官方 `calibrated_rgb` |
| 视角 | `third_person` + `wrist` |
| 输出 | 341 capture，RGB 720×1280×3 |
| 采样 | replay `--sample-every 6`，保留首帧、末帧和 phase boundary |
| 成像 | PathTracing，128 spp，关闭跨姿态 accumulation，固定曝光、灯光、去噪和焦距 |
| 时间 | 两相机 measured timestamp 必须逐帧相同；source timestamp 单独保存 |
| 允许残差 | source-to-camera 最大绝对 residual ≤0.12 s；双相机 measured timestamp 差 ≤1 ms |
| RGB 门槛 | 每帧非空、有限、非全常数；首/中/末三帧人工复核 |
| 校准 | 每帧内参、相机到初始 base 外参、camera_valid 齐全 |
| 归档 | motion/outcome/HDF5 原文件、SHA256、stage/source hash、运行参数、git commit 全部保存 |

0.12 s 是在 candidate05-45 的 0.1000 s 最大残差基础上留出的采集容差，不是把两套时钟当成同一时钟。超过门槛的轨迹只进入 quarantine，不进入 train/validation/test。

## 3. 系统组定义

### 3.1 system_group_id

`system_group_id` 表示一组共享同一物理和成像条件的 episode。以下字段参与系统组哈希：

- 机器人型号、初始关节状态和 base/fixture 几何；
- 插入座、转子、试管、夹爪及其装配关系；
- 物理材料、接触模型、摩擦基准、间隙和目标深度；
- 相机挂载、内参、外参、分辨率、PathTracing/灯光/曝光配置；
- LabUtopia/Isaac Sim、资产、场景和 replay 脚本版本；
- 随机种子及准备阶段 source/stage hash。

episode 级的控制扰动、起始位姿微调和允许的单次操作条件不进入 system_group_id；它们作为 intervention 字段记录。

同一 `system_group_id` 的所有 episode 必须位于同一 split。禁止按帧、按窗口或按 episode 随机切分。任何共享 stage hash、source motion hash 或未经声明的派生副本跨 split 都视为泄漏。

### 3.2 system_group_manifest

每组必须有一条不可变记录：

- `system_group_id`、`system_family_id`、split；
- 组成字段的规范化 JSON 及 SHA256；
- stage/source/asset/calibration hash；
- 该组包含的 episode_id 列表；
- intervention 类型及 outcome 统计；
- recorder、importer、配置和 Git commit；
- quarantine/rejection 原因（如有）。

先生成并冻结 group manifest，再开始训练或调参。验证集和测试集的 outcome 标签可以用于最终评估，但不得用于选择采集参数、模型结构或阈值。

## 4. 第一批采集规模

采用 36 个系统组、每组 10 条轨迹，共 360 条 episode：

| split | 系统组 | episode | 用途 |
|---|---:|---:|---|
| train | 24 | 240 | 模型训练 |
| validation | 6 | 60 | 早停、超参和机制诊断参数冻结 |
| test-system | 3 | 30 | 未见系统、已见干预分布 |
| test-composition | 3 | 30 | 未见系统、组合/边界干预 |

系统组按固定 seed=20261028 的哈希排序后分配，分配结果写入 manifest 并冻结；不能根据结果重新移动组。

第一批正式训练最低门槛是：每个 split 至少保留 90% 的有效 episode，且每个 split 至少有 2 个成功、2 个失败或受控非成功系统组。若失败类无法安全采集，不得伪造标签；应先完成失败安全 recorder，再缩减为仅用于机制分析的受控数据批次。

## 5. 每个系统组的 10 条轨迹模板

每个系统组使用同一 nominal scene 和相同相机配置，按以下模板采集：

| episode | 类型 | 条件 |
|---|---|---|
| N01–N04 | nominal success | 4 次独立重复；只改变允许的执行微扰/随机初态，不改变系统组固定参数 |
| L01 | lateral | 目标或管路横向偏移 +0.25 mm |
| T01 | tilt | 目标姿态绕指定轴偏移 +0.10° |
| F05 | low friction | 全局摩擦系数 0.5 |
| F04 | lower friction | 全局摩擦系数 0.4 |
| C01 | composition | lateral + friction 0.5 的组合；train/validation/test-system 使用此标准组合 |
| C02/R01 | composition/recovery | test-composition 使用 lateral + friction 0.4 的未见组合；其余 split 使用安全恢复或边界条件。若没有经过验证的恢复脚本，暂以第五条 nominal 重复替代，并标记 `recovery_unavailable` |

说明：

- 这些数值来自已有扰动试验，只作为第一批候选范围；正式采集前先做每个条件 1 条 smoke episode。test-composition 的 C02 只在该测试 split 使用，用于检验未见组合，不参与训练和阈值选择。
- 任何会造成不可控高速碰撞、穿模或设备风险的条件直接拒绝，不为了凑失败比例而增加危险扰动。
- `F05/F04` 产生的滑移失败必须保留完整历史，并同时记录滑移起始阶段、最大偏差和最终 task_success。
- nominal 重复用于估计成像、时间和执行噪声；不能把重复轨迹当成独立系统。

## 6. 标签和观测分离

### deployable evidence

允许作为模型输入：

- 两路 RGB、camera_valid、内参和逐帧外参；
- measured camera timestamp、source timestamp、source frame index；
- 测量关节位置/速度、目标/已应用关节命令及 timing；
- 可部署的任务条件。

禁止作为 deployable 输入：

- simulator body transform；
- simulator contact；
- scripted phase；
- high-rate privileged trace；
- future tube pose、future response、future success。

### privileged targets / oracle

可以保存并用于监督或 oracle 消融：

- body transforms；
- contact states；
- scripted phase；
- 480 Hz hand/tube trace；
- strict geometry、task_success、success；
- 未来状态和未来响应。

每条 episode 同时保存原始 JSON、HDF5 和投影后的 NPZ/manifest；任何字段被 mask 时，原始值仍保存在归档中，但不能通过旁路进入 deployable 模型。

## 7. 质量门槛

单条 episode 只有全部通过以下检查才进入正式 split：

1. source motion、outcome、stage 和 HDF5 hash 能互相核对；
2. 341 capture 齐全，双视角逐帧存在且尺寸正确；
3. 两相机 measured timestamp 差 ≤1 ms，timestamp 严格递增；
4. source-to-camera 最大 residual ≤0.12 s；
5. RGB 非空、有限、无黑帧；首/中/末三帧人工确认试管、左右夹指和目标区域可辨识；
6. 关节状态、命令、校准和 camera_valid 字段齐全；
7. outcome 独立于模型输入产生，success/task_success/strict_geometry_pass 分别记录；
8. deployable 和 recorder_oracle import 均通过；deployable 的 privileged mask 仍为 invalid；
9. `validate-data` 通过，且 episode/system identity 与冻结 manifest 一致；
10. 运行日志、命令、Git commit、环境和 SHA256 齐全。

任何失败都写入 quarantine manifest，不能只删除失败文件后重新编号。

## 8. 划分与评估切片

主划分只按 system_group_id：

- **train**：24 组，包含所有基础干预的混合样本；
- **validation**：6 组，包含所有基础干预，冻结选择和早停；
- **test-system**：3 组，评估未见系统上的同分布泛化；
- **test-composition**：3 组，重点评估 lateral+friction、接触歧义和失败/恢复历史。

所有切片额外报告：

- nominal / lateral / tilt / friction / composition / recovery；
- success / task failure / strict geometry failure；
- 接触发生前、接触中、滑移后、恢复段；
- 双视角完整、单视角缺失、遮挡程度分层；
- source-to-camera residual 分层。

每个 episode 只在一个主 split 出现。滑动窗口只能在 episode 内生成，不能跨 episode；同一 episode 的所有窗口必须保留在同一 split。

## 9. q0→q1 机制数据要求

正式训练前，采集流程必须能提供：

- 第一轮 q0、第一轮状态分布；
- delta_q；
- history read bias；
- 原始历史重读后的 q1；
- 第一轮/最终状态分布；
- 重构残差、历史读取权重和有效时间范围；
- episode_id、system_group_id、intervention、阶段标签。

机制干预至少包括：

1. first-pass；
2. no-correction；
3. no-bias；
4. no-residual；
5. random/swapped correction；
6. separate-weights；
7. 删除关键接触阶段；
8. 用同组其他 episode 替换历史。

预期证据是：替换或删除历史后，q0 相对稳定，而 q1、历史读取重点和最终状态分布随相关历史改变；同时联合 NLL、状态误差和成功预测发生可解释变化。仅报告 q0/q1 向量距离不构成机制证据。

## 10. 执行顺序

1. 先采 6 个系统组的 smoke batch：每组 N01、L01、T01、F05 各 1 条；
2. 验证时间、画面可见性、成功/失败 outcome 和 deployable mask；
3. 冻结 system_group_id 生成器、split seed 和 manifest schema；
4. 完成 36 组正式采集；
5. 先只读审计并生成 train/validation/test manifest；
6. 在不读取 test 结果的前提下训练首个 bounded run；
7. 再执行 q0/q1 消融和历史替换干预；
8. 最后评估 test-system 与 test-composition，并保留全部失败和 quarantine 记录。

## 11. 当前工程前置项

当前 `record_labutopia_pose_replay.py` 默认仍只接受 `success/task_success/strict_geometry_pass=true` 的成功 motion；已增加显式 `--capture-mode failure-safe` 路径，使非成功轨迹能够：

- 正常结束并写出官方双相机 HDF5；
- 保存独立 outcome，不因失败而丢弃历史；
- 对危险碰撞、穿模、时间门控和黑帧 fail closed；
- 不改变现有成功轨迹的 recorder 语义；
- 通过同样的 import、mask 和 validate-data 检查。

在该前置项完成前，只能先执行 smoke 的成功轨迹和安全受控变化，不得宣称已经获得完整失败分布。

## 12. 记录格式建议

每个 episode 的 manifest 至少包含：

```json
{
  "system_group_id": "sg-0001",
  "system_family_id": "family-01",
  "episode_id": "sg-0001-F05-01",
  "split": "train",
  "intervention": "low_friction_0.5",
  "outcome": {
    "success": false,
    "task_success": false,
    "strict_geometry_pass": false
  },
  "source_motion_sha256": "...",
  "stage_sha256": "...",
  "calibrated_rgb_sha256": "...",
  "recorder_commit": "...",
  "max_source_camera_residual_s": 0.0,
  "max_cross_camera_timestamp_delta_s": 0.0,
  "quarantine": false
}
```

该 schema 是采集和划分的审计入口；模型训练只能消费通过质量门槛且与冻结 split 一致的 manifest。


