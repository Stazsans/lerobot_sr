# LeRobot FK/IK 与 Argo FK/IK 的区别

两者本质区别是：LeRobot 方案是“LeRobot processor 工作流里的运动学抽象”，而 Argo
方案是“第三方 URDF 运动学库直接求 FK/IK”。它们都能从关节算 EE，也都能从 EE
反解关节，但工程定位、数据流、状态管理和 IK 搜索策略不同。

## LeRobot：RobotKinematics + robot_kinematic_processor

这是 LeRobot 原生 EE 控制管线。`RobotKinematics` 负责底层 FK/IK，
`robot_kinematic_processor.py` 负责把它包装成 LeRobot 的 `RobotProcessorPipeline`
step。

典型链路：

```text
RobotAction / RobotObservation
-> RobotProcessorPipeline
-> ForwardKinematicsJointsToEE
-> EEBoundsAndSafety
-> InverseKinematicsEEToJoints
-> RobotAction
```

它的特点是“和 LeRobot 数据集/训练/评估系统天然对齐”：

- `action_features / observation_features`
- dataset feature transform
- `record_loop`
- replay
- evaluate
- policy input/output

都可以通过 processor 统一处理。比如 `ForwardKinematicsJointsToEE` 不只是算 FK，
它还会把 dataset feature 从：

```text
shoulder_pan.pos, shoulder_lift.pos, ...
```

转换成：

```text
ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos
```

`InverseKinematicsEEToJoints` 也不只是算 IK，它还集成了 LeRobot 侧的策略：

- 当前关节作为 initial guess
- 上一帧 IK 解作为 fallback/seed
- joint preferences
- joint limits
- continuous joints
- `max_joint_delta_deg`
- orientation priority
- `fallback_to_current_joints_on_invalid`
- IK diagnostics

所以 LeRobot 这套更像是：

```text
运动学求解器 + 数据管线适配 + 安全/诊断/特征转换
```

## Argo：URDF_Kinematics / RobotModel

Argo 方案更底层。`URDF_loader` 读 URDF，`RobotModel` 保存机器人模型、关节限位、
base/tool 变换，`URDF_Kinematics` 直接做 FK/IK。

典型链路：

```text
URDF_loader.load(urdf)
RobotModel(loader)
URDF_Kinematics()

q_rad -> forward_kinematics(robot_model, q_rad, target_link_name)
target_pose -> _inverse_kinematics_step_baseTn(...)
```

它不是 LeRobot processor 原生组件，所以它本身不知道：

- `LeRobotDataset`
- `record_loop`
- `action_features`
- `observation_features`
- `SO101Follower` action dict
- `ee.x / ee.wx` feature schema
- camera features
- policy input/output

这些都需要自行包一层适配，比如现在的：

- `argo_teleoperate.py`
- `argo_record.py`
- `argo_replay.py`
- `argo_evaluate.py`

这些脚本做的就是把 Argo 的 `q_rad / pose matrix` 世界翻译成 LeRobot 的：

```text
{"shoulder_pan.pos": deg, ...}
{"ee.x": m, "ee.wx": rad, ...}
LeRobotDataset feature schema
RobotProcessorPipeline
```

## 关键差异

### 1. 抽象层级不同

LeRobot 面向 LeRobot 机器人控制和数据管线的高层 processor。

Argo 面向 URDF 数学模型的底层 FK/IK 库。

### 2. 输入输出语义不同

LeRobot 常用：

- dict action/observation
- joint units: degrees
- gripper: 0-100
- EE pose: `ee.x/y/z/wx/wy/wz`

Argo 常用：

- `np.ndarray q_rad`
- joint units: radians
- pose: 4x4 homogeneous matrix
- gripper: URDF joint radians

所以 Argo 需要大量转换函数：

- `observation_to_argo_q_rad`
- `argo_q_rad_to_robot_action`
- `gripper_percent_to_argo_rad`
- `pose_to_ee_dict`
- `ee_dict_to_pose`

### 3. IK 求解策略不同

LeRobot 的 IK processor 是为了控制链路服务的，它会把求解、限幅、fallback、诊断整合在
一个 processor step 里。

Argo 的 IK 更直接：给它 robot model、当前 q、目标矩阵、target link、gain、iterations，
它返回一个 q。多 seed、排序、限速、重规划这些策略不是 Argo 自动给的，而需要在
`argo_real_ik.py / argo_teleoperate.py` 里自行写。

### 4. 与 dataset 的关系不同

LeRobot processor 可以自然参与：

- `aggregate_pipeline_dataset_features`
- `create_initial_features`
- `record_loop`
- `build_dataset_frame`

Argo 原本不懂这些。为了让 Argo 能录 `LeRobotDataset`，需要在 `argo_record.py` 里写
processor 适配层：

- `ArgoFKJointsToEE`
- `ArgoIKEEToJoints`

它们的作用是把 Argo FK/IK 包装成 LeRobot processor。

### 5. 安全与边界处理不同

LeRobot 有现成：

- `EEBoundsAndSafety`
- `max_ee_step_m`
- `max_orientation_step_rad`
- `max_joint_delta_deg`
- `joint_position_limits_deg`

Argo 原生主要依赖：

- `robot_model.mech_joint_limits_low`
- `robot_model.mech_joint_limits_up`
- `clip_q_to_argo_limits`

额外的伺服限速、重规划暂停、tracking guard，需要在 Argo 脚本里自行实现。

### 6. 可调试性不同

LeRobot 调试更偏 processor 级别：

- 看 transition
- 看 action/observation feature
- 看 IK diagnostics
- 看 processor transform_features

Argo 调试更偏数学/运动学级别：

- 看 `q_rad`
- 看 `target_pose` 4x4
- 看 FK predicted pose
- 看 `pos_err / ori_err`
- 看 seed candidate
- 看 limit violation

## 一句话总结

```text
LeRobot FK/IK = 集成在 LeRobot 数据与控制生态里的高层运动学 processor 管线
Argo FK/IK    = 独立 URDF 运动学求解器，需要手写适配层接入 LeRobot
```

如果目标是快速接 LeRobot 训练/录制/评估，LeRobot 管线更顺。如果目标是验证另一套 IK
是否更适合 SO101 真机，Argo 更灵活，但必须保证 record/replay/evaluate 全链路都使用
同一套 Argo FK/IK 语义。
