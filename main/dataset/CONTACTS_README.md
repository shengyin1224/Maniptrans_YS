# 手-物体接触分析工具

这个工具用于分析 MANO 手部运动数据中手与物体的接触情况。

## 功能特点

- 支持多物体场景
- 精确计算手部21个关节点（1个手腕 + 20个手指关节）与物体的接触
- 基于距离阈值检测接触（默认 5mm）
- 详细的接触信息记录（位置、距离、时间）
- 可视化和统计分析工具

## 文件说明

- `compute_hand_object_contacts.py`: 主程序，计算接触
- `visualize_contacts.py`: 可视化和分析工具
- `example_compute_contacts.sh`: 使用示例

## 使用方法

### 1. 计算接触

基本使用：
```bash
python main/dataset/compute_hand_object_contacts.py \
    --data_idx "baking_with_spatula_mixing_bowl_and_scooping_to_tray-244" \
    --dexhand "inspire" \
    --side "right" \
    --threshold 0.007 \
    --output_dir "data/contacts_right"
```

参数说明：
- `--data_idx`: 数据索引（例如 "1906"）
- `--dexhand`: 手部模型类型（例如 "inspire"）
- `--side`: 左手或右手（"left" 或 "right"）
- `--threshold`: 接触距离阈值，单位米（默认 0.005，即 5mm）
- `--num_points`: 物体表面采样点数（默认 1024）
- `--device`: 计算设备（默认 "cuda:0"）
- `--output_dir`: 输出目录

输出结果保存在：`{output_dir}/{dataset_type}/contacts_{data_idx}.pkl`

### 2. 可视化结果

#### 查看接触摘要
```bash
python main/dataset/visualize_contacts.py data/contacts/oakink2/contacts_1906.pkl
```

#### 绘制接触时间线
```bash
python main/dataset/visualize_contacts.py data/contacts/oakink2/contacts_1906.pkl \
    --timeline \
    --object_idx 0 \
    --save_fig "figures/contact_timeline.png"
```

#### 绘制接触统计
```bash
python main/dataset/visualize_contacts.py data/contacts/oakink2/contacts_1906.pkl \
    --statistics \
    --save_fig "figures/contact_stats.png"
```

#### 导出为 CSV
```bash
python main/dataset/visualize_contacts.py data/contacts/oakink2/contacts_1906.pkl \
    --export_csv "data/contacts/contacts_1906.csv"
```

#### 查看特定帧的详细信息
```bash
python main/dataset/visualize_contacts.py data/contacts/oakink2/contacts_1906.pkl \
    --frame_detail 50
```

### 3. 批量处理

```bash
# 处理多个数据
for idx in 1906 1907 1908 1909 1910; do
    python main/dataset/compute_hand_object_contacts.py \
        --data_idx "$idx" \
        --output_dir "data/contacts"
done
```

## 输出数据格式

接触结果以 pickle 格式保存，数据结构如下：

```python
{
    'data_idx': str,                    # 数据索引
    'dataset_type': str,                # 数据集类型
    'num_frames': int,                  # 总帧数
    'num_objects': int,                 # 物体数量
    'num_hand_points': int,             # 手部点数（21）
    'contact_threshold': float,         # 接触阈值（米）
    'hand_point_names': List[str],      # 手部点名称
    'object_names': List[str],          # 物体名称
    'frames': [                         # 每帧的数据
        {
            'frame_idx': int,
            'objects': [                # 每个物体的数据
                {
                    'object_idx': int,
                    'object_name': str,
                    'has_contact': bool,
                    'num_contacts': int,
                    'contacts': [       # 每个接触点的数据
                        {
                            'hand_point_idx': int,
                            'hand_point_name': str,
                            'hand_point_pos': np.array([x, y, z]),
                            'object_contact_pos': np.array([x, y, z]),
                            'distance': float  # 单位：米
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
}
```

## 手部点索引

手部共有 21 个点：
- 索引 0: 手腕 (wrist)
- 索引 1-4: 食指关节 (index: proximal, intermediate, distal, tip)
- 索引 5-8: 中指关节 (middle: proximal, intermediate, distal, tip)
- 索引 9-12: 无名指关节 (ring: proximal, intermediate, distal, tip)
- 索引 13-16: 小指关节 (pinky: proximal, intermediate, distal, tip)
- 索引 17-20: 拇指关节 (thumb: proximal, intermediate, distal, tip)

## 可视化示例

### 接触时间线图
显示每帧中哪些手部点与物体接触，以及接触距离。

### 接触统计图
- 左图：每个手部点的接触频率
- 右图：每个物体被接触的帧数占比

## 注意事项

1. **坐标系转换**：代码会自动处理 MuJoCo 到 IsaacGym 的坐标系转换
2. **数据集支持**：支持 OakInk-v2, Humoto, Favor 等数据集
3. **内存使用**：大规模点云可能占用较多内存，可调整 `--num_points` 参数
4. **计算时间**：取决于帧数和点云密度，使用 GPU 可显著加速

## 应用场景

- 抓取质量评估：分析手部与物体的接触模式
- 接触建模：为模拟器提供接触先验
- 动作分割：基于接触事件分割动作序列
- 数据筛选：过滤接触不良的数据样本

## 扩展建议

如需扩展功能，可以考虑：
- 添加接触力估计
- 区分不同类型的接触（抓取、支撑、滑动等）
- 计算接触面积和接触法向
- 分析接触稳定性

