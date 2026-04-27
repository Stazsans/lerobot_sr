# SO101 末端执行器 (EE) 位姿控制示例

本目录提供基于 SO101 机械臂的末端执行器 (End-Effector) 位姿控制完整工作流：
遥操作、数据采集、回放验证、异步推理评估。

## 核心概念

传统关节空间控制直接操作各关节角度，而 EE 位姿控制在笛卡尔空间工作：
- **正运动学 (FK)**：关节角度 → 末端位姿 `(x, y, z, wx, wy, wz, gripper_pos)`
- **逆运动学 (IK)**：末端位姿 → 关节角度

策略在 EE 空间训练和推理，与关节构型解耦，泛化性更强。

## 文件说明

| 文件 | 用途 |
|------|------|
| `teleoperate.py` | 遥操作：leader 关节 → FK → EE 指令 → IK → follower 关节 |
| `argo_teleoperate.py` | Argo 遥操作：leader 关节 → Argo FK → EE 指令 → Argo IK → follower 关节 |
| `move_to_pose.py` | 单目标/多目标末端位姿验证：直接给 EE pose，经 IK 到达 |
| `record.py` | 数据采集：遥操作 + 录制 EE 空间数据集（参照 lerobot-record 流程） |
| `argo_record.py` | Argo 数据采集：使用 Argo FK/IK 遥操作并录制 EE 空间数据集 |
| `replay.py` | 回放验证：读取本地数据集的 EE 动作并通过 IK 执行 |
| `argo_replay.py` | Argo 回放验证：读取 Argo EE 数据集并通过 Argo IK 执行 |
| `evaluate.py` | 异步推理评估：gRPC 策略服务器 + 机器人客户端 |
| `argo_evaluate.py` | Argo 异步推理评估：使用 Argo FK/IK 语义执行 EE 策略 |
| `third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf` | SO101 运动学模型描述文件，以上脚本默认使用该路径 |

## 前置准备

1. **硬件连接**：SO101 follower 臂 + SO101 leader 臂 + USB 摄像头
2. **串口识别**：`lerobot-find-port` 找到 leader/follower 端口
3. **摄像头识别**：`lerobot-find-cameras` 找到摄像头索引
4. **校准**：确保 follower 和 leader 都已校准
5. **URDF 文件**：默认使用仓库内 `third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`，只有在你要替换模型文件时才需要修改 `URDF_PATH`

## 键盘控制

录制和遥操作过程中可使用以下按键：

每个 episode 需要按**两次**方向键：

1. **录制阶段**：按 `→` 结束录制，进入重置阶段
2. **重置阶段**：将环境恢复到初始状态，按 `→` 确认完成，保存并进入下一集

| 按键 | 录制阶段 | 重置阶段 |
|------|---------|---------|
| 右箭头 `→` | 结束录制，进入重置 | 结束重置，保存并进入下一集 |
| 左箭头 `←` | 结束录制并标记重录 | 结束重置，丢弃当前数据并重新录制 |
| `Esc` | 停止整个录制流程 | 停止整个录制流程 |

## 使用流程

### 1. 遥操作测试

验证 FK/IK 管线和硬件连接是否正常：

```bash
python src/lerobot/so101_to_so101_EE/teleoperate.py
```

Argo FK/IK 版本：

```bash
python src/lerobot/so101_to_so101_EE/argo_teleoperate.py
```

添加 `argo_teleoperate.py`
使用和 `teleoperate.py` 相同的在线遥操作流程：连接 leader/follower、循环读取 leader、计算 follower 动作、直接发送到 follower。区别是 FK/IK 换成 Argo：

```text
leader joints -> Argo FK -> EE target
EE target + follower joints -> Argo IK -> follower joint command
follower joint command -> follower.send_action
```

### 2. 数据采集

录制 EE 空间的训练数据，参照 lerobot-record 流程（使用 `record_loop`、`VideoEncodingManager` 等）。
使用前修改脚本顶部的常量配置：

```bash
python src/lerobot/so101_to_so101_EE/record.py
```

Argo FK/IK 版本：

```bash
python src/lerobot/so101_to_so101_EE/argo_record.py
```

添加 `argo_record.py`
使用和 `record.py` 相同的录制流程：`LeRobotDataset.create`、`record_loop`、`VideoEncodingManager`、键盘控制、episode/reset 逻辑。区别是 processor 换成 Argo：

```text
leader joints -> Argo FK -> EE action recorded in dataset
EE action -> Argo IK -> follower joint command
follower joints -> Argo FK -> EE observation recorded in dataset
```

### 2.5 直接给定末端位姿

不依赖 leader 臂，直接给目标 EE pose 做真机 FK/IK 验证：

```bash
python src/lerobot/so101_to_so101_EE/move_to_pose.py
```

`move_to_pose.py` 当前的用户侧位置配置统一使用 `cm`：

- `TARGET_POSE` 里的 `ee.x / ee.y / ee.z` 按厘米填写
- `TARGET_POSE` 里的 `delta.ee.x / delta.ee.y / delta.ee.z` 也按厘米填写
- 运行日志里的位置和位置误差同样按厘米打印
- 脚本内部进入 FK/IK 前会自动换算成米，因此不影响底层运动学实现

### 3. 回放验证

回放数据集中的 EE 动作来验证数据质量：

```bash
python src/lerobot/so101_to_so101_EE/replay.py
```

Argo FK/IK 版本：

```bash
python src/lerobot/so101_to_so101_EE/argo_replay.py
```

添加 `argo_replay.py`
使用和 `replay.py` 相同的回放流程：读取本地 `LeRobotDataset`、选择 episode、逐帧读取 dataset action、按 dataset FPS 下发动作。区别是 EE action 到关节命令的 processor 换成 Argo：

```text
dataset EE action -> Argo IK -> follower joint command
follower observation -> current joint seed for Argo IK
follower joint command -> follower.send_action
```

### 4. 训练

使用标准 lerobot 训练命令训练策略：

```bash
lerobot-train \
    --policy=act \
    --dataset.repo_id=hanoi_ee \
    --dataset.root=./datasets/hanoi_ee
```

### 5. 异步推理评估

评估使用 gRPC 异步推理架构，将策略推理与机器人控制解耦，
推理延迟不阻塞控制循环，提高实时性。

**终端 1 -- 启动策略服务器**（可在 GPU 机器上运行）：

```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
```

**终端 2 -- 启动评估客户端**（在机器人连接的机器上运行）：

```bash
python src/lerobot/so101_to_so101_EE/evaluate.py
```

Argo FK/IK 版本：

```bash
python src/lerobot/so101_to_so101_EE/argo_evaluate.py
```

添加 `argo_evaluate.py`
使用和 `evaluate.py` 相同的异步推理流程：gRPC `PolicyServer`、动作块队列、按需发送观测、控制循环持续执行。区别是客户端侧 FK/IK 换成 Argo：

```text
follower joints -> Argo FK -> EE observation sent to policy server
policy server -> EE action chunk
EE action -> Argo IK -> follower joint command
```

评估流程：
1. 客户端获取关节观测 → FK 转为 EE 观测 → 发送到服务器
2. 服务器执行策略推理 → 返回 EE 动作块
3. 客户端接收 EE 动作 → IK 转为关节动作 → 发送到机器人
4. 动作接收和执行异步并行，机器人以 30 FPS 持续运动

## 配置说明

所有脚本通过顶部常量配置，使用前需修改：

| 常量 | 所在脚本 | 说明 |
|------|---------|------|
| `FOLLOWER_PORT` / `LEADER_PORT` | record, teleoperate | 串口路径 |
| `CAMERA_INDEXES` | record, evaluate | 摄像头索引 |
| `DATASET_NAME` / `DATASET_ROOT` | record, replay | 数据集名称和本地路径 |
| `MODEL_PATH` | evaluate | 训练好的模型路径 |
| `SERVER_ADDRESS` | evaluate | 策略服务器地址 |
| `URDF_PATH` | move_to_pose, record, teleoperate, replay, evaluate | 默认指向仓库内 `third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf` |

推荐先只改串口和相机，再直接跑 `move_to_pose.py` 做最小真机验证；默认 URDF 路径能用时，不要额外改动。

## 数据集存储

数据集默认存储在本地 `./datasets/<DATASET_NAME>/` 目录。
如需推送到 HuggingFace Hub，将 `PUSH_TO_HUB` 常量设为 `True`。

## EE 空间特征

观测和动作使用 7 维 EE 位姿表示：

| 特征 | 含义 |
|------|------|
| `ee.x` | 末端 X 位置 (m) |
| `ee.y` | 末端 Y 位置 (m) |
| `ee.z` | 末端 Z 位置 (m) |
| `ee.wx` | 末端绕 X 轴旋转 (rad) |
| `ee.wy` | 末端绕 Y 轴旋转 (rad) |
| `ee.wz` | 末端绕 Z 轴旋转 (rad) |
| `ee.gripper_pos` | 夹爪开合角度 |

## 安全边界

`EEBoundsAndSafety` 处理器提供运行时安全保护：
- `end_effector_bounds`：EE 位置的工作空间限制
- `max_ee_step_m`：单步最大 EE 位移限制（防止大幅跳变）
- `max_orientation_step_rad`：单步最大 EE 姿态变化限制（防止大幅旋转跳变）

当前 SO-101 真机链路还包含这些保护：

- IK 候选多 seed 搜索与打分选优
- 不安全或末端误差过大的 IK 解回退到当前关节
- `send_action()` 下发前基于电机校准的绝对关节限位裁剪
- gripper 目标显式基于 `gripper.pos` 计算，不依赖关节顺序
