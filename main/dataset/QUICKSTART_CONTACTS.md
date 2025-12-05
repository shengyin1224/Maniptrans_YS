# 快速开始：手-物体接触分析

## 问题说明

由于 IsaacGym 的导入顺序限制和依赖问题，我们提供了两个版本的工具：

### 版本对比

| 特性 | compute_hand_object_contacts.py | compute_contacts_standalone.py |
|------|--------------------------------|-------------------------------|
| 依赖 | 需要完整 ManipDataFactory | ✅ 只需要 numpy, torch, trimesh |
| 数据源 | ManipDataFactory 数据索引 | ✅ 直接读取 pickle 文件 |
| 适用场景 | 完整环境已配置 | ✅ 快速分析、依赖冲突 |

## 推荐：使用独立版本 ⭐

```bash
# 安装基本依赖（如果还没有）
pip install torch numpy trimesh termcolor tqdm

# 直接分析 pickle 数据文件
python main/dataset/compute_contacts_standalone.py \
    path/to/your/data.pkl \
    --threshold 0.005 \
    --num_points 1024 \
    --device cuda:0 \
    --output data/contacts/result.pkl
```

## 数据格式要求

输入的 pickle 文件需要包含以下字段：

```python
{
    'wrist_pos': np.array or torch.Tensor,  # [num_frames, 3]
    'wrist_rot': np.array or torch.Tensor,  # [num_frames, 3] (axis-angle)
    'mano_joints': {                         # Dict 或 Array
        'joint_name': np.array,              # [num_frames, 3]
        ...
    } 或 np.array([num_frames, 20, 3]),
    'scene_objects': [                       # List of objects
        {
            'name': str,
            'urdf': str,                     # URDF 文件路径
            'trajectory': np.array           # [num_frames, 4, 4] or [num_frames, 16]
        },
        ...
    ]
}
```

## 示例数据查找

在你的项目中查找现有的 pickle 数据文件：

```bash
# 查找所有 pickle 文件
find data/ -name "*.pkl" -type f | head -10

# 或查找特定数据集
find data/OakInk-v2/ -name "*.pkl"
find data/retargeting/ -name "*.pkl"
```

## 完整示例

```bash
# 1. 找到一个数据文件
DATA_FILE="data/OakInk-v2/your_sequence.pkl"

# 2. 运行接触分析
python main/dataset/compute_contacts_standalone.py \
    "$DATA_FILE" \
    --threshold 0.005 \
    --device cuda:0 \
    --output "data/contacts/$(basename $DATA_FILE)"

# 3. 可视化结果
python main/dataset/visualize_contacts.py \
    "data/contacts/$(basename $DATA_FILE)" \
    --statistics \
    --timeline
```

## 批量处理

```bash
# 处理目录中的所有 pickle 文件
for pkl_file in data/your_directory/*.pkl; do
    echo "Processing: $pkl_file"
    python main/dataset/compute_contacts_standalone.py \
        "$pkl_file" \
        --threshold 0.005 \
        --device cuda:0 \
        --output "data/contacts/$(basename $pkl_file)"
done
```

## 参数说明

- `pickle_path`: 输入的 pickle 数据文件路径（必需）
- `--threshold`: 接触距离阈值，单位米（默认 0.005 = 5mm）
- `--num_points`: 物体表面采样点数（默认 1024）
- `--device`: 计算设备，cuda:0 或 cpu（默认自动检测）
- `--output`: 输出文件路径（默认在 data/contacts/ 下）

## 可视化结果

分析完成后，使用可视化工具查看结果：

```bash
# 查看统计摘要
python main/dataset/visualize_contacts.py data/contacts/result.pkl

# 绘制图表
python main/dataset/visualize_contacts.py data/contacts/result.pkl \
    --timeline --statistics --save_fig figures/contacts.png

# 导出 CSV 用于进一步分析
python main/dataset/visualize_contacts.py data/contacts/result.pkl \
    --export_csv data/contacts/result.csv

# 查看特定帧的详细信息
python main/dataset/visualize_contacts.py data/contacts/result.pkl \
    --frame_detail 50
```

## 故障排除

### 找不到 pickle 文件

检查你的数据路径：
```bash
ls -lh data/OakInk-v2/
ls -lh data/retargeting/
```

### CUDA 内存不足

使用 CPU 或减少点云密度：
```bash
python main/dataset/compute_contacts_standalone.py \
    your_data.pkl \
    --device cpu \
    --num_points 512
```

### pickle 文件格式不匹配

检查文件内容：
```python
import pickle
with open('your_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
```

需要的关键字段：`wrist_pos`, `mano_joints`, `scene_objects`

## 如果你确实需要完整版本

如果你的环境已经配置好所有依赖（bps_torch等），可以使用：

```bash
python main/dataset/compute_hand_object_contacts.py \
    --data_idx "your_data_index" \
    --dexhand "inspire" \
    --side "right" \
    --threshold 0.005 \
    --output_dir "data/contacts"
```

**注意**：这需要：
1. 正确安装 bps_torch
2. 正确配置 IsaacGym 环境
3. ManipDataFactory 能够访问数据

## 下一步

- 使用 `visualize_contacts.py` 分析接触模式
- 基于接触信息筛选高质量数据
- 将接触信息用于训练或评估

