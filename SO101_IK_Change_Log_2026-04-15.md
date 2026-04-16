# SO-101 IK 变更记录

日期：2026-04-15

本文整理自 `2026-04-14` 起这条 SO-101 FK/IK 真机复现线上的改动与问题，按三栏归档：

- 已完成
- 已知问题
- 待处理

## 已完成

### 1. SO-101 IK 选解逻辑已增强

文件：

- [robot_kinematic_processor.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py)

已做内容：

- 增加按 `motor_names` 显式取关节值，减少对字典顺序的依赖
- 增加多 seed IK 候选搜索
- 增加候选解打分与选优
- 对不安全或末端误差过大的候选解回退到当前关节

相关锚点：

- [_ordered_joint_positions](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L49)
- [_solve_best_ik_solution](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L184)
- [fallback_to_current_joints_on_invalid](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L235)

### 2. 新增“给定末端位姿直达”真机脚本

文件：

- [move_to_pose.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/move_to_pose.py)

作用：

- 读取当前关节状态
- 做 FK 得到当前末端位姿
- 给定目标 EE pose
- 通过 IK 逐步到达

当前默认配置已经包含：

- 默认仓库内 URDF 路径
- EE 边界限制
- 单步 EE 位移限制
- 单关节最大跳变限制

### 3. SO-101 EE 相关脚本已统一到仓库内默认 URDF

涉及文件：

- [move_to_pose.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/move_to_pose.py)
- [teleoperate.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/teleoperate.py)
- [record.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/record.py)
- [replay.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/replay.py)
- [evaluate.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/evaluate.py)

默认 URDF 路径：

```text
third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

### 4. 默认 URDF 文件已加入仓库工作区

目录：

- `third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf`

说明：

- `2026-04-14` 最初状态下，仓库里没有这个 URDF
- 后续已下载到仓库内相对稳定的位置

### 5. 最小回归测试已补充

文件：

- [test_so_follower_ik.py](/C:/Users/seer/projects/lerobot_sr/tests/test_so_follower_ik.py)

已覆盖内容：

- 按电机顺序取值
- 危险解淘汰
- 无效 IK 回退到当前关节
- RL 路径保留 `IK_solution`
- 绝对关节限位裁剪
- `EEReferenceAndDelta` 的有序关节取值
- EE 姿态单步限幅
- `GripperVelocityToJoint` 不再依赖 gripper 在最后一个位置

相关锚点：

- [test_inverse_kinematics_prefers_safe_solution_and_motor_name_order](/C:/Users/seer/projects/lerobot_sr/tests/test_so_follower_ik.py#L86)
- [test_inverse_kinematics_falls_back_to_current_joints_on_invalid_pose](/C:/Users/seer/projects/lerobot_sr/tests/test_so_follower_ik.py#L108)
- [test_inverse_kinematics_rl_step_keeps_selected_solution_in_complementary_data](/C:/Users/seer/projects/lerobot_sr/tests/test_so_follower_ik.py#L132)

### 6. SO-101 文档已更新

文件：

- [README.md](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/README.md)
- [SO101_real_robot_fk_ik_steps.md](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/SO101_real_robot_fk_ik_steps.md)

已完成内容：

- README 已改为当前真实状态
- 明确默认 URDF 已在仓库内
- 统一从仓库根目录执行脚本的命令格式
- 输出了一份独立的真机 FK/IK 操作手册

### 7. 对话记录已保存

文件：

- [SO101_IK_Conversation_2026-04-14_full.md](/C:/Users/seer/projects/lerobot_sr/SO101_IK_Conversation_2026-04-14_full.md)

说明：

- 保存了 `2026-04-14` 这条 SO-101 IK 相关对话的逐轮原文版

### 8. SO-101 下发前已增加绝对关节限位裁剪

文件：

- [so_follower.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/so_follower.py)

已完成内容：

- 基于当前电机校准中的 `range_min/range_max` 推导绝对关节限位
- 在 `send_action()` 下发前先做绝对关节限位裁剪
- 保留原有 `max_relative_target` 的相对步长保护

相关锚点：

- [_degrees_limits_from_calibration](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/so_follower.py#L38)
- [clip_goal_positions_to_calibration](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/so_follower.py#L49)
- [send_action](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/so_follower.py#L267)

### 9. `EEReferenceAndDelta` 已改为统一按 `motor_names` 取关节

文件：

- [robot_kinematic_processor.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py)

已完成内容：

- 非 `IK_solution` 分支已改为统一使用 `_ordered_joint_positions(...)`
- 不再按 `observation.items()` 过滤拼接关节数组

相关锚点：

- [_ordered_joint_positions](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L49)
- [EEReferenceAndDelta](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L300)

### 10. `EEBoundsAndSafety` 已增加姿态单步限幅

文件：

- [robot_kinematic_processor.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py)

已完成内容：

- 新增 `max_orientation_step_rad`
- 除位置单步限幅外，增加 `ee.wx/wy/wz` 的单步姿态跳变限幅
- `reset()` 会同步清除姿态历史

相关锚点：

- [EEBoundsAndSafety](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L399)
- [max_orientation_step_rad](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L408)

### 11. `GripperVelocityToJoint` 已去除顺序假设

文件：

- [robot_kinematic_processor.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py)

已完成内容：

- 不再通过 `q_raw[-1]` 取 gripper 当前值
- 现在显式依赖 `gripper.pos`

相关锚点：

- [GripperVelocityToJoint](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L570)
- [gripper.pos](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L599)

### 12. RL 路径参考状态已改为“观测优先，IK 解仅在近一致时复用”

文件：

- [robot_kinematic_processor.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py)
- [test_so_follower_ik.py](/C:/Users/seer/projects/lerobot_sr/tests/test_so_follower_ik.py)

已完成内容：

- `EEReferenceAndDelta` 现在总是先读取当前观测关节作为 FK 参考真值
- 当 `use_ik_solution=True` 时，仅在上一拍 `IK_solution` 与当前观测关节偏差足够小的情况下才复用该解
- 若真机未跟上、被扰动或发生漂移，参考状态会自动回退到当前真实观测，不再优先沿用陈旧的 `IK_solution`
- 已补充测试覆盖“陈旧 IK 解应被忽略”和“近一致 IK 解可继续复用”两种情况

## 已知问题

### 1. 当前这批 SO-101 IK 改动里，未再保留“RL 路径优先使用上一拍 `IK_solution`”这个问题

## 待处理

### 1. 做一轮真机验证，确认 RL 增量控制在实际跟随滞后和外力扰动下不再出现参考系累积漂移

建议：

- 改为优先参考当前真机实测关节
- 若保留 `IK_solution`，则仅作为候选参考，不作为唯一优先源

目标：

- 避免真机跟踪落后时参考系持续漂移

### 2. 补充测试

建议增加：

- RL 路径参考当前观测与参考上一拍 `IK_solution` 的行为测试
- 真机配置下的更高层集成测试

## 当前工作区相关文件

已修改：

- [robot_kinematic_processor.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/robots/so_follower/robot_kinematic_processor.py)
- [README.md](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/README.md)
- [evaluate.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/evaluate.py)
- [record.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/record.py)
- [replay.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/replay.py)
- [teleoperate.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/teleoperate.py)

新增未跟踪：

- [move_to_pose.py](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/move_to_pose.py)
- [test_so_follower_ik.py](/C:/Users/seer/projects/lerobot_sr/tests/test_so_follower_ik.py)
- [SO101_real_robot_fk_ik_steps.md](/C:/Users/seer/projects/lerobot_sr/src/lerobot/so101_to_so101_EE/SO101_real_robot_fk_ik_steps.md)
- [SO101_IK_Conversation_2026-04-14_full.md](/C:/Users/seer/projects/lerobot_sr/SO101_IK_Conversation_2026-04-14_full.md)
- [AGENTS.md](/C:/Users/seer/projects/lerobot_sr/AGENTS.md)
- `third_party/`
