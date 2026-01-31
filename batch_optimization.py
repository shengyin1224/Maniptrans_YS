#!/usr/bin/env python3
"""
批量运行 mano2dexhand_segmented.py 优化程序
修改版逻辑：
1. 严格顺序：先 Left 后 Right
2. 智能跳过：如果 Stage 1 文件存在则直接 Stage 2
3. 容错增强：捕获 Stage 1 的 Segmentation Fault (-11)，若文件生成则继续
"""

import os
import subprocess
import multiprocessing as mp
from typing import List, Set, Optional
import argparse
import time
from datetime import datetime

# 脚本所在目录作为项目根
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 配置数据 (保持原样) ---
MISSING_FROM_BOTH_15 = [
    "drinking_from_mug1_and_talking-879",
    "getting_off_bed_and_touch_table_lamp_then_grasp_clothes_from_clothes_rack-752",
    "move_clothes_rack_with_both_hands-840",
    "organizing_and_transfering_utensils_to_draw_organizer_tray-489",
    "pour_from_vacuum_flask_and_peel_and_cut_mango-117",
    "sitting_on_working_chair_at_table_working_with_laptop_drinking_coffee_with_mug_in_left_hand-176",
    "transfer_deep_plate_side_plate_and_serving_bowl_to_table_with_both_hands-922",
    "transfer_from_woven_basket_to_wash_tub-105",
    "transfer_mango_and_plastic_bowl_from_side_table_and_basket-017",
    "transfer_mug_and_side_plate_from_side_table_to_utility_cart_and_return-980",
    "turning_working_chair_right_with_right_foot_three_times-209",
    "walk_with_low_chair-308",
    "walking_back_and_forth_while_pushing_the_clothes_rack_with_right_hand_then_left_hand-594",
    "walking_back_and_forth_with_the_draw_organizer_tray_in_right_hand_while_raising_then_switching_hands-867",
    "working_with_laptop_and_drinking_coffee_with_right_hand-870",
]

MISSING_OUTPUT_LOG = os.path.join(BASE_DIR, "batch_optimization_missing_outputs.log")

# ... (ALL_DATA_INDICES 列表保持原样，此处省略以节省空间) ...
ALL_DATA_INDICES = [
    # 请保持你原有的完整列表
    "baking_with_spatula_mixing_bowl_and_scooping_to_tray-244",
    # ...
]

COMPLETED_DATA_INDICES = []

# --- 路径辅助函数 ---

def get_expected_output_paths(data_idx: str):
    lh = os.path.join(BASE_DIR, "data", "retargeting", "Humoto", "mano2inspire_lh", f"{data_idx}.pkl")
    rh = os.path.join(BASE_DIR, "data", "retargeting", "Humoto", "mano2inspire_rh", f"{data_idx}.pkl")
    return lh, rh

def get_stage1_path(data_idx: str, side: str):
    side_abbr = "lh" if side == "left" else "rh"
    return os.path.join(BASE_DIR, "data", "retargeting", "Humoto", f"mano2inspire_{side_abbr}", f"{data_idx}_stage1_nocontact.pkl")

def check_stage1_file_exists(data_idx: str, side: str):
    stage1_path = get_stage1_path(data_idx, side)
    return os.path.isfile(stage1_path), stage1_path

def check_and_log_missing_output(data_idx: str) -> None:
    lh_path, rh_path = get_expected_output_paths(data_idx)
    missing = []
    if not os.path.isfile(lh_path):
        missing.append("lh")
    if not os.path.isfile(rh_path):
        missing.append("rh")
    if not missing:
        return
    line = f"{datetime.now().isoformat()}  data_idx={data_idx}  缺失: {', '.join(missing)}  lh={lh_path}  rh={rh_path}\n"
    with open(MISSING_OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[日志] 未发现输出，已写入: {MISSING_OUTPUT_LOG} -> {data_idx} 缺失: {', '.join(missing)}")

# --- 核心执行逻辑 ---

def execute_subprocess(cmd: List[str], env: dict, desc: str, allow_segfault: bool = False) -> bool:
    """
    执行子进程，处理 Segfault 逻辑。
    返回 True 表示执行成功（或 Segfault 但被允许），False 表示其他错误。
    """
    try:
        subprocess.run(cmd, env=env, check=True, cwd=BASE_DIR)
        return True
    except subprocess.CalledProcessError as e:
        # 检查是否为 Segmentation Fault (Unix 上通常是 -11)
        if e.returncode == -11:
            print(f"⚠️ [警告] {desc} 发生 Segmentation Fault (Code -11)。")
            if allow_segfault:
                print(f"   -> 依据策略，视为非致命错误，将检查产物文件是否生成。")
                return True # 这里的 True 意味着"过程结束了"，后面需要手动检查文件是否存在
            else:
                return False
        else:
            print(f"❌ [错误] {desc} 运行失败 (Code {e.returncode}): {e}")
            return False

def run_single_hand_pipeline(data_idx: str, side: str, gpu_id: int) -> bool:
    """
    处理单个手的完整流程：
    Check S1 -> (Run S1 if missing) -> Run S2
    """
    side_abbr = "lh" if side == "left" else "rh"
    print(f"👉 开始处理: {data_idx} [{side}] (GPU {gpu_id})")

    # 准备环境
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 准备基础命令参数
    contact_data_path = os.path.join(BASE_DIR, "data", f"contacts_{side}", "humoto", f"contacts_{data_idx}.pkl")
    base_cmd = [
        "python", "main/dataset/mano2dexhand_segmented.py",
        "--data_idx", data_idx,
        "--dexhand", "inspire",
        "--iter", "15000",
        "--headless",
        "--draw_all_lines", "0",
        "--side", side,
        "--contact_data", contact_data_path
    ]

    # --- Step 1: 检查或运行 Stage 1 ---
    s1_exists, s1_path = check_stage1_file_exists(data_idx, side)
    
    if s1_exists:
        print(f"   [Step 1] 跳过: {side} 手 Stage 1 文件已存在 -> {s1_path}")
    else:
        print(f"   [Step 1] 运行: {side} 手 Stage 1 ...")
        cmd_s1 = base_cmd + ["--stage", "1"]
        
        # 允许 Segfault，因为用户指出 Stage 1 可能会崩但需要继续
        run_success = execute_subprocess(cmd_s1, env, f"{data_idx} {side} Stage 1", allow_segfault=True)
        
        # 再次检查文件是否生成（无论运行是否报错，只要文件有了就能跑 Stage 2）
        s1_exists, _ = check_stage1_file_exists(data_idx, side)
        if not s1_exists:
            print(f"⛔ [终止] {side} 手 Stage 1 失败或崩溃，且未生成中间文件。跳过该手的 Stage 2。")
            return False
        else:
            print(f"   -> Stage 1 文件确认存在，继续。")

    # --- Step 2: 运行 Stage 2 ---
    print(f"   [Step 2] 运行: {side} 手 Stage 2 ...")
    cmd_s2 = base_cmd + ["--stage", "2"]
    
    # Stage 2 如果崩了，通常就是真失败了，不过我们依然允许 Segfault 以防万一
    s2_success = execute_subprocess(cmd_s2, env, f"{data_idx} {side} Stage 2", allow_segfault=True)

    if s2_success:
        print(f"✅ 完成: {data_idx} [{side}]")
        return True
    else:
        print(f"❌ 失败: {data_idx} [{side}] Stage 2 执行出错")
        return False

def run_two_stage_optimization(data_idx: str, gpu_id: int):
    """
    按顺序处理左右手
    """
    print(f"=== 开始任务: {data_idx} (GPU {gpu_id}) ===")
    
    # 1. 先处理左手
    run_single_hand_pipeline(data_idx, "left", gpu_id)
    
    # 2. 再处理右手 (无论左手成功与否，都尝试右手，除非你想左手失败就全停)
    # 这里保持独立性：左手失败不影响右手尝试
    run_single_hand_pipeline(data_idx, "right", gpu_id)

    print(f"=== 结束任务: {data_idx} ===")
    return True # 总是返回 True 以保持主进程继续，具体缺失由 log 记录

# --- 主程序 (保持原有结构，仅微调调用) ---

def main():
    parser = argparse.ArgumentParser(description="批量运行 mano2dexhand_segmented.py 优化")
    parser.add_argument("--gpu_ids", type=str, default="5", help="GPU ID列表")
    parser.add_argument("--max_parallel", type=int, default=4, help="并发数")
    parser.add_argument("--data_indices", type=str, default=None, help="指定数据索引")
    parser.add_argument("--missing_15", action="store_true", help="跑特定的15个缺失数据")

    args = parser.parse_args()

    # 特殊配置逻辑
    if args.missing_15:
        if args.gpu_ids == "5":
            args.gpu_ids = "5"
        if args.data_indices is None:
            args.data_indices = ",".join(MISSING_FROM_BOTH_15)

    gpu_ids = [int(gpu.strip()) for gpu in args.gpu_ids.split(',')]
    
    # 确定目标数据
    if args.data_indices:
        target_data = [idx.strip() for idx in args.data_indices.split(',')]
    else:
        # 此处简单处理，如果想自动过滤已完成的，需要结合 get_expected_output_paths 判断
        # 这里为了演示逻辑，假设处理 ALL_DATA_INDICES 中不在 COMPLETED_DATA_INDICES 的
        completed_set = set(COMPLETED_DATA_INDICES)
        target_data = [idx for idx in ALL_DATA_INDICES if idx not in completed_set]

    print(f"待处理数据: {len(target_data)} 个")
    print(f"使用 GPU: {gpu_ids}")

    if not target_data:
        return

    # 并行处理
    with mp.Pool(processes=min(args.max_parallel, len(gpu_ids))) as pool:
        tasks = []
        for i, data_idx in enumerate(target_data):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            # 统一入口，不再区分只跑 stage 几，内部逻辑已自动判断
            task = pool.apply_async(run_two_stage_optimization, (data_idx, gpu_id))
            tasks.append((data_idx, task))

        # 结果监控
        completed_count = 0
        for data_idx, task in tasks:
            try:
                task.get(timeout=3600*24) # 这里的返回值不重要了，主要靠内部打印
                completed_count += 1
                # 检查最终输出
                check_and_log_missing_output(data_idx)
            except Exception as e:
                print(f"[{completed_count+1}/{len(tasks)}] {data_idx}: 进程异常 - {e}")
                completed_count += 1

    print("所有任务队列执行完毕。")

if __name__ == "__main__":
    main()