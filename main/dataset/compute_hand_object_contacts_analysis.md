# compute_hand_object_contacts.py 距离计算分析

## 计算流程

1. **物体点云采样** (L61-128)
   - 从 URDF 加载 mesh
   - 使用 `trimesh.sample.sample_surface_even` 采样 **1024 个点**
   - **关键操作**：采样后减去物体中心 `object_points - center`，得到**局部坐标系**点云

2. **物体点云变换到世界坐标系** (L187-206, L316-317)
   - 使用物体的 `obj_pose` (4x4 变换矩阵) 将局部点云转换到世界坐标系
   - `transform_object_points` 函数执行变换

3. **手部点获取** (L257-272)
   - 从 `demo_data` 获取 `wrist_pos` 和 `mano_joints`
   - 应用 `mujoco2gym_transf` 转换到世界坐标系
   - 组合成 `hand_points_world` = [wrist + 所有手指关节]

4. **距离计算** (L131-184)
   - 使用 `torch.cdist(hand_points, object_points_world)` 计算所有手部点到所有物体采样点的距离
   - 找出所有小于阈值（默认 5mm）的点对

## 潜在问题

### ⚠️ 问题 1：点云采样密度不足

**问题描述**：
- `base.py` 使用物体的**完整顶点集**（通常是 1000+ 个点，来自 `obj["verts_transf"]`）
- `compute_hand_object_contacts.py` 只采样 **1024 个点**

**影响**：
- 采样点可能无法覆盖所有表面区域，导致最近点计算不准确
- 对于复杂形状的物体，1024 个点可能不够密集

**建议**：
- 增加采样点数（例如 5000-10000）
- 或者直接使用 mesh 的完整顶点集（如果内存允许）

### ⚠️ 问题 2：手部点可能不包含指尖

**问题描述**：
- `hand_points_world` 包含：`wrist + mano_joints`
- 需要确认 `dexhand.body_names` 是否包含所有指尖关节（thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip）

**影响**：
- 如果指尖不在 `hand_points_world` 中，计算的就是其他关节到物体的距离，而不是指尖距离

**验证方法**：
- 检查 `results['hand_point_names']` 是否包含所有 tip 关节

### ⚠️ 问题 3：坐标系转换的一致性

**问题描述**：
- 物体点云先减去中心（L113），再通过 `obj_pose` 变换
- 需要确认 `obj_pose` 是否已经包含了物体中心的位置信息

**影响**：
- 如果 `obj_pose` 的平移部分不是物体中心，会导致点云位置错误

**验证方法**：
- 检查 `obj_traj`（来自 `scene_objects[i]['trajectory']`）的平移部分是否代表物体中心

### ⚠️ 问题 4：与 base.py 的计算方式不一致

**问题描述**：
- `base.py` 使用 `ch_dist`（ChamferDistance）计算点到点云的最近距离
- `compute_hand_object_contacts.py` 使用 `torch.cdist` 计算所有点对距离，然后找最小值

**影响**：
- 两种方法理论上应该等价，但实现细节可能不同
- `ch_dist` 可能使用了更优化的最近邻搜索

### ⚠️ 问题 5：只计算了“接触”距离，没有计算“最近点”距离

**问题描述**：
- 当前代码只返回**小于阈值**的点对（L154: `distances < threshold`）
- 没有计算每个手部点到物体的**真正最近距离**（即使大于阈值）

**影响**：
- 如果手部点距离物体表面 6mm（大于 5mm 阈值），代码不会记录这个距离
- 这可能导致距离统计不完整

**建议**：
- 对于每个手部点，计算到所有物体采样点的最小距离（无论是否小于阈值）
- 这样可以获得完整的距离分布

## 建议的修复方案

1. **增加点云采样密度**：
   ```python
   num_object_points = 5000  # 或使用完整顶点集
   ```

2. **验证手部点包含指尖**：
   ```python
   # 在 analyze_motion_contacts 中添加检查
   tip_names = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
   for tip_name in tip_names:
       assert tip_name in results['hand_point_names'], f"Missing {tip_name}"
   ```

3. **计算所有距离（不仅限于接触）**：
   ```python
   # 在 compute_hand_object_contacts 中添加
   min_distances = distances.min(dim=1)[0]  # [num_hand_points] 每个手部点的最近距离
   ```

4. **与 base.py 保持一致**：
   - 考虑直接使用 `obj["verts_transf"]` 而不是采样点云
   - 或者确保采样点云足够密集
