# SO-101 真机复现 FK/IK 操作步骤

本文基于当前仓库状态，给出一套可直接执行的 SO-101 真机正运动学（FK）/逆运动学（IK）复现流程。

推荐顺序：

1. 先做“给定末端位姿直达”的 IK 验证
2. 再做 leader -> FK -> IK -> follower 的遥操作验证

这样更稳，也更容易定位问题。

## 1. 环境准备

在仓库根目录执行：

```bash
cd C:\Users\seer\projects\lerobot_sr
```

安装最少依赖：

```bash
uv sync --extra kinematics
```

如果你还没装电机和其他依赖，建议直接安装更完整的一套：

```bash
uv sync --extra all
```

说明：

- FK/IK 依赖 `placo`
- 当前仓库默认 URDF 路径为 `third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`
- 当前仓库已经包含这个 URDF 文件，通常不需要再改 `URDF_PATH`

## 2. 找串口和相机

先插上 SO-101 follower 臂，然后查串口：

```bash
uv run lerobot-find-port
```

记下 follower 端口。

常见情况：

- Windows: `COM3`、`COM4`
- Linux: `/dev/ttyACM0`、`/dev/ttyACM1`

如果后面要做遥操作，再把 leader 臂也插上，再执行一次：

```bash
uv run lerobot-find-port
```

如果后面要录制或评估，再查相机：

```bash
uv run lerobot-find-cameras opencv
```

## 3. 确认机械臂状态

正式运行前确认：

1. follower 和 leader 都已完成校准
2. 机械臂起始姿态在中间安全位置，不靠近关节极限
3. 周围没有障碍物
4. 手放在急停位置，随时准备断电

## 4. 最小真机 IK 验证

这一阶段不依赖 leader，只验证：

- 当前关节 -> FK -> 当前末端位姿
- 给定目标末端位姿 -> IK -> 真机关节动作

使用脚本：

[move_to_pose.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/move_to_pose.py)

### 4.1 修改脚本配置

打开脚本，至少修改以下内容：

- `FOLLOWER_PORT`
- `TARGET_POSES`

当前脚本里默认的关键参数包括：

- `FPS = 20`
- `INTERPOLATION_STEPS = 60`
- `MAX_EE_STEP_M = 0.015`
- `MAX_JOINT_DELTA_DEG` 为一组保守的单关节跳变阈值

第一次建议使用更保守的目标位姿，例如：

```python
TARGET_POSES = [
    {
        "name": "ready_front",
        "ee.x": 0.18,
        "ee.y": 0.00,
        "ee.z": 0.16,
        "ee.wx": 0.0,
        "ee.wy": 1.57,
        "ee.wz": 0.0,
        "ee.gripper_pos": 30.0,
    },
]
```

### 4.2 运行脚本

```bash
uv run python src/lerobot/so101_to_so101_EE/move_to_pose.py
```

### 4.3 运行时会发生什么

脚本会按这个流程工作：

1. 读取当前关节观测
2. 做 FK，打印当前 EE 位姿
3. 对目标 EE 位姿做插值
4. 每一步都根据当前观测重新做 IK
5. 将关节目标下发到真机
6. 到达后再做一次 FK，打印实际到达 EE 位姿

### 4.4 正常现象

你应该看到类似输出：

- `Current EE: ...`
- `Target EE: ...`
- `Reached EE: ...`

正常情况下：

- 机械臂平滑移动
- 没有突然大跳
- `Reached EE` 和 `Target EE` 比较接近

### 4.5 异常现象

如果出现以下情况，应立即停止：

- 一启动就大幅抽动
- 目标很近，但机械臂绕大圈
- 末端明显朝反方向运动
- 关节持续抖动
- 长时间卡住但仍在尝试发命令

先检查：

1. 串口是否填错
2. 机械臂是否完成校准
3. 目标位姿是否过激
4. URDF 是否与当前机械臂版本匹配
5. 当前起始姿态是否过于靠近边界或奇异位

## 5. 遥操作 FK -> IK 验证

当 `move_to_pose.py` 已稳定后，再做遥操作链路验证。

使用脚本：

[teleoperate.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/teleoperate.py)

这条链路是：

1. leader 关节
2. FK 得到 leader 当前 EE 位姿
3. follower 根据这个 EE 位姿做 IK
4. follower 执行对应关节动作

### 5.1 修改脚本配置

至少修改：

- `FOLLOWER_PORT`
- `LEADER_PORT`

### 5.2 运行脚本

```bash
uv run python src/lerobot/so101_to_so101_EE/teleoperate.py
```

### 5.3 正常现象

正常情况下：

- follower 会基本跟随 leader
- 不应出现明显的随机大跳
- 小范围姿态变化应能稳定映射

## 6. 推荐调试顺序

如果你是第一次在真机上跑，建议严格按下面顺序来：

1. 只接 follower，跑 `move_to_pose.py`
2. 只测一个目标位姿
3. 目标位姿先设在工作空间中央
4. 确认没有大跳、没有乱动
5. 再增加第二个、第三个位姿
6. 之后再接 leader，跑 `teleoperate.py`

## 7. 推荐命令清单

```bash
uv sync --extra kinematics
uv run lerobot-find-port
uv run python src/lerobot/so101_to_so101_EE/move_to_pose.py
uv run python src/lerobot/so101_to_so101_EE/teleoperate.py
```

如果需要相机：

```bash
uv run lerobot-find-cameras opencv
```

## 8. 当前实现的保护机制

当前 SO-101 IK 路径已经有这些保护：

- 末端工作空间边界限制
- 单步 EE 位移限制
- 多 seed IK 求解
- 候选解打分选择
- 不安全或误差过大的解回退到当前关节

因此，相比早期“多解选错后直接乱动”的情况，现在已经稳很多。

但仍然建议：

- 首次测试时小步长、小范围、低速度
- 避免把目标设在机械臂边界和奇异位附近
- 不要一开始就做大姿态旋转

## 9. 相关文件

- [README.md](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/README.md)
- [move_to_pose.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/move_to_pose.py)
- [teleoperate.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/teleoperate.py)
- [robot_kinematic_processor.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py)
