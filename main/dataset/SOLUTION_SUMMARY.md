# 手-物体接触分析解决方案总结

## 问题说明

你遇到的错误：
```
PyTorch was imported before isaacgym modules. Please import torch after isaacgym modules.
```

这是因为 IsaacGym 要求在导入 torch 之前先导入 isaacgym 相关模块，但项目中有很多依赖（如 `bps_torch`）会自动导入 torch。

## 解决方案

我为你创建了**三个版本**的工具，根据你的环境选择使用：

### 方案 1：独立版本（✅ 推荐）

**文件**：`compute_contacts_standalone.py`

**优点**：
- 不需要 IsaacGym 环境
- 不需要 bps_torch 等复杂依赖
- 只需要：numpy, torch, trimesh

**使用方法**：
```bash
# 直接从 pickle 文件计算接触
python main/dataset/compute_contacts_standalone.py \
    path/to/your_data.pkl \
    --threshold 0.005 \
    --device cuda:0
```

**限制**：需要 pickle 文件包含以下字段：
- `wrist_pos`
- `wrist_rot`
- `mano_joints`
- `scene_objects`

### 方案 2：完整版本

**文件**：`compute_hand_object_contacts.py`

**优点**：
- 可以使用 ManipDataFactory 直接加载数据
- 支持数据索引方式（如 "1906", "baking...244"）

**前提条件**：
1. ✅ 已修复导入顺序问题
2. ❌ 需要安装 bps_torch 和所有依赖
3. ❌ 需要完整的数据加载环境

**使用方法**：
```bash
python main/dataset/compute_hand_object_contacts.py \
    --data_idx "baking_with_spatula_mixing_bowl_and_scooping_to_tray-244" \
    --dexhand "inspire" \
    --side "right"
```

### 方案 3：从 mano2dexhand.py 数据计算

如果你已经运行了 `mano2dexhand.py` 并生成了数据，那些数据缺少物体信息（只有机器人手的优化结果）。

## 当前项目数据结构

### 数据位置

1. **OakInk-v2 原始数据**：`data/OakInk-v2/anno_preview/*.pkl`
   - 格式：包含 `raw_mano`, `obj_transf` 等
   - 需要通过 OakInk2DatasetDexHandRH 加载

2. **Retargeting 结果**：`data/retargeting/*/mano2inspire_*/*.pkl`
   - 格式：只有 `opt_wrist_pos`, `opt_dof_pos` 等
   - ❌ 缺少物体和原始 MANO 数据
   - ❌ 无法直接用于接触计算

3. **Humoto 数据**：`data/humoto/*.pkl`（如果有）

## 建议工作流程

### 选项 A：使用 ManipDataFactory 加载数据然后计算接触

1. 创建临时脚本保存加载的数据：

```python
# save_loaded_data.py
import sys
sys.path.insert(0, '/path/to/ManipTrans')

# 先导入 IsaacGym 相关
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

# 然后导入 torch
import torch
import pickle

data_idx = "your_data_index"
dexhand = DexHandFactory.create_hand("inspire", "right")

dataset_type = ManipDataFactory.dataset_type(data_idx)
demo_d = ManipDataFactory.create_data(
    manipdata_type=dataset_type,
    side="right",
    device="cuda:0",
    mujoco2gym_transf=torch.eye(4, device="cuda:0"),
    dexhand=dexhand,
    verbose=False,
)

data = demo_d[data_idx]

# 保存为标准格式
with open(f'temp_data_{data_idx}.pkl', 'wb') as f:
    pickle.dump(data, f)
```

2. 然后使用独立版本计算接触：

```bash
python main/dataset/compute_contacts_standalone.py temp_data_*.pkl
```

### 选项 B：修改 mano2dexhand.py 集成接触计算

在 `mano2dexhand.py` 的 `fitting` 函数中，添加接触计算逻辑，在优化过程中或优化后计算接触。

### 选项 C：直接在 OakInk-v2 数据上计算 MANO 接触

如果你的目标是分析原始 MANO 手部与物体的接触（而不是机器人手），我可以为你创建一个直接读取 OakInk-v2 annotation 文件的工具。

## 辅助工具

### 1. 检查 pickle 文件格式

```bash
python main/dataset/check_pickle_format.py path/to/file.pkl
```

### 2. 可视化接触结果

```bash
python main/dataset/visualize_contacts.py data/contacts/result.pkl \
    --timeline --statistics
```

### 3. 导出 CSV

```bash
python main/dataset/visualize_contacts.py data/contacts/result.pkl \
    --export_csv output.csv
```

## 下一步建议

**请告诉我你的情况**：

1. **你有哪些数据？**
   - [ ] OakInk-v2 原始数据
   - [ ] Humoto 数据
   - [ ] 已运行 mano2dexhand.py 的结果

2. **你的环境状态？**
   - [ ] 可以使用 ManipDataFactory（所有依赖都装好）
   - [ ] 只能用基本依赖（numpy, torch, trimesh）

3. **你想计算谁的接触？**
   - [ ] 原始 MANO 手与物体
   - [ ] Retarget 后的机器人手与物体

根据你的回答，我可以提供更具体的解决方案！

## 已创建的文件清单

✅ 核心工具：
- `compute_hand_object_contacts.py` - 完整版本（需要完整环境）
- `compute_contacts_standalone.py` - 独立版本（推荐）
- `visualize_contacts.py` - 可视化工具

✅ 辅助工具：
- `check_pickle_format.py` - 检查数据格式
- `test_contacts.py` - 测试依赖

✅ 文档：
- `CONTACTS_README.md` - 完整文档
- `QUICKSTART_CONTACTS.md` - 快速开始
- `SOLUTION_SUMMARY.md` - 本文档

✅ 示例：
- `example_compute_contacts.sh` - 使用示例

