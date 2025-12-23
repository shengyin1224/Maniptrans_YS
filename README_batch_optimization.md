# 批量优化脚本使用说明

## 概述

`batch_optimization.py` 是一个用于批量运行 mano2dexhand_segmented.py 优化程序的脚本。它支持：

- 多进程并行处理
- 一张GPU对应一个数据
- 每个数据先运行stage 1再运行stage 2
- 自动跳过已完成的数据

## 使用方法

### 基本用法

```bash
# 使用默认设置：所有可用GPU，并行处理4个任务
python batch_optimization.py

# 指定GPU和并行数
python batch_optimization.py --gpu_ids 0,1,2,3 --max_parallel 4

# 只处理指定数据
python batch_optimization.py --data_indices "data1,data2,data3"

# 只运行stage 1
python batch_optimization.py --stage 1

# 只运行stage 2
python batch_optimization.py --stage 2
```

### 参数说明

- `--gpu_ids`: 可用的GPU ID列表，用逗号分隔 (默认: 0,1,2,3,4,5,6,7)
- `--max_parallel`: 最大并行进程数 (默认: 4)
- `--data_indices`: 指定要处理的数据索引，用逗号分隔。如果不指定，则处理所有未完成的数据
- `--stage`: 指定运行的阶段 (1 或 2)。如果不指定，则运行两阶段

## 配置

### 已完成数据列表

在脚本的 `COMPLETED_DATA_INDICES` 列表中添加已完成的数据索引，脚本会自动跳过这些数据：

```python
COMPLETED_DATA_INDICES = [
    "baking_with_spatula_mixing_bowl_and_scooping_to_tray-244",
    "carry_organizer_with_both_hands_at_chest_height-436",
    # 添加更多已完成的数据...
]
```

### 数据列表

`ALL_DATA_INDICES` 包含所有可用的数据索引。如果有新的数据，请添加到这个列表中。

## 工作流程

1. 脚本会检查哪些数据还未完成
2. 为每个未完成的数据分配一个GPU
3. 并行启动多个进程，每个进程处理一个数据
4. 每个数据依次运行 stage 1 和 stage 2（除非指定了特定阶段）
5. 每个数据会为 left 和 right 手分别运行

## 输出

脚本会实时显示处理进度和状态信息：

```
开始处理数据: data_name (GPU 0)
运行命令: python main/dataset/mano2dexhand_segmented.py --data_idx data_name ...
完成: data_name left hand (stage 1)
完成: data_name right hand (stage 1)
Stage 1 完成: data_name
Stage 2 开始: data_name
...
[1/10] data_name: 成功
```

## 注意事项

1. 确保系统有足够的GPU内存
2. 每个数据处理需要较长时间，请耐心等待
3. 如果某个数据处理失败，脚本会继续处理其他数据
4. 可以随时中断脚本（Ctrl+C），已完成的数据不会重新处理
