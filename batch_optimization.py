#!/usr/bin/env python3
"""
批量运行 mano2dexhand_segmented.py 优化程序
支持多进程并行，一张卡对应一个数据，每个数据先跑stage 1再跑stage 2
"""

import os
import subprocess
import multiprocessing as mp
from typing import List, Set, Optional
import argparse
import time
from datetime import datetime

# 脚本所在目录作为项目根（Maniptrans_YS）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# humoto 里未同时出现在 mano2inspire_lh 和 mano2inspire_rh 的 15 个动作（用于重新跑）
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

# 跑完后未在 output 路径下发现文件时写入的日志路径
MISSING_OUTPUT_LOG = os.path.join(BASE_DIR, "batch_optimization_missing_outputs.log")

# 所有可用的数据索引列表
ALL_DATA_INDICES = [
    "baking_with_spatula_mixing_bowl_and_scooping_to_tray-244",
    "carry_organizer_with_both_hands_at_chest_height-436",
    "carry_side_table_with_both_hands_walk_around-536",
    "carry_vase_right_hand_transfer_to_left_hand_transfer_to_right_hand_walk_around-942",
    "carry_wok_turner_right_hand_walk_around-968",
    "carrying_cutting_board_with_right_hand-213",
    "carrying_whisk_with_right_hand-037",
    "checking_floor_lamp_with_right_hand-024",
    "checking_organizer_medium_on_table-289",
    "chopping_and_slitting_with_knife-706",
    "cleaning_cutting_board_with_soap-507",
    "cleaning_knife_with_soap_dispenser-in_sink-019",
    "dispensing_soap_on_left_hand_and_washing-517",
    "drag_floor_lamp_walk_back-386",
    "drag_side_table_walk_around-994",
    "dragging_turning__clothes_rack_walking_on_ground-802",
    "drinking_from_mug1_and_talking-879",
    "drinking_from_mug_with_right_hand-815",
    "dropping_blanket_and_blouse_from_clothes_rack_to_woven_basket_on_left-799",
    "eating_from_plastic_bowl_with_spoon-596",
    "examining_cutting_board-141",
    "flipping_pancake_in_frying_pan_with_turner-998",
    "getting_off_bed_and_touch_table_lamp_then_grasp_clothes_from_clothes_rack-752",
    "grasp_and_touch_notebook_with_right_hand_from_side_table-670",
    "grasp_and_transfer_woven_basket_from_top_shelf_to_ground_while_stretch-767",
    "lift_shelf_lie_to_stand-867",
    "lifting_and_checking_basket_90-215",
    "lifting_and_lowering_utility_cart_with_both_hands-397",
    "lifting_and_putting_down_dining_chair-368",
    "lifting_and_tilting_table-304",
    "lifting_dining_chair_from_backrest_and_placing_back_with_right_hand_on_ground-595",
    "lifting_floor_lamp_and_placing_down-346",
    "lifting_frying_pan_and_transfering_eggs_to_side_plate_with_shaky_hands-130",
    "lifting_side_table_and_putting_down-362",
    "lifting_working_chair_and_putting_back-778",
    "move_clothes_rack_with_both_hands-840",
    "moving_low_chair_with_both_hands-723",
    "organizing_and_transfering_utensils_to_draw_organizer_tray-489",
    "peel_mango_with_peeler_by_rotating_mango_in_left_hand_from_left-855",
    "pick_up_fallen_chair_and_place_on_ground-078",
    "picking_wash_tub_and_trash_can_in_woven_basket-678",
    "placing_phone_laptop_on_table_replacing_table_lamp_mug_picking_up_notebook-285",
    "placing_trash_can_down_and_leaving_trash_in_trash_can-255",
    "plugging_usb_in_back_of_laptop_with_right_hand-792",
    "pour_from_vaccum_flask_to_mug_and_drink-097",
    "pour_from_vacuum_flask_and_peel_and_cut_mango-117",
    "pouring_liquid_from_mug_into_trash_can-696",
    "pull_low_chair_from_back_and_front_and_walk_backward-980",
    "pulling_and_pushing_the_dining_chair_from_back_post-203",
    "reading_notebook_under_table_lamp_on_low_chair_in_left_side_of_side_table-164",
    "rearranging_the_gradvis_vase_from_bottom_shelf_to_top_shelf-377",
    "roll_soap_dispenser_head-662",
    "sit_on_dining_chair_and_drink_from_mug_and_shake_mug-654",
    "sitting_on_working_chair_at_table_working_with_laptop_drinking_coffee_with_mug_in_left_hand-176",
    "step_on_step_stool_and_transfer_woven_basket_from_top_shelf_to_ground-267",
    "subject_leaning_against_metalic_clothes_rack_sliding_right_then_left-201",
    "sweeping_and_flipping_blouse_on_hanger_with_lint_roller_on_table_from_left_to_right-614",
    "taking_off_clothes_and_hanging_on_rack-645",
    "transfer_deep_plate_side_plate_and_serving_bowl_to_table_with_both_hands-922",
    "transfer_from_woven_basket_to_wash_tub-105",
    "transfer_mango_and_plastic_bowl_from_side_table_and_basket-017",
    "transfer_mug_and_side_plate_from_side_table_to_utility_cart_and_return-980",
    "transfer_spoon_knife_and_plastic_bowl_from_basket_to_the_table-703",
    "transfer_vase_from_utility_cart_to_shelf_then_to_utility_cart_with_both_hands-798",
    "turning_working_chair_right_with_right_foot_three_times-209",
    "walk_with_low_chair-308",
    "walking_back_and_forth_while_pushing_the_clothes_rack_with_right_hand_then_left_hand-594",
    "walking_back_and_forth_with_the_draw_organizer_tray_in_right_hand_while_raising_then_switching_hands-867",
    "washing_clothes_with_right_hand_in_wash_tub-966",
    "working_with_laptop_and_drinking_coffee_with_right_hand-870",
]

# 已完成的数据列表 - 请在这里修改已完成的数据
COMPLETED_DATA_INDICES = [
    # 在这里添加已完成的数据索引，例如：
    # "baking_with_spatula_mixing_bowl_and_scooping_to_tray-244",
    # "carry_organizer_with_both_hands_at_chest_height-436",
]


def get_expected_output_paths(data_idx: str):
    """返回某 data_idx 跑完后应存在的两个 pkl 路径 (lh, rh)。"""
    lh = os.path.join(BASE_DIR, "data", "retargeting", "Humoto", "mano2inspire_lh", f"{data_idx}.pkl")
    rh = os.path.join(BASE_DIR, "data", "retargeting", "Humoto", "mano2inspire_rh", f"{data_idx}.pkl")
    return lh, rh


def check_and_log_missing_output(data_idx: str) -> None:
    """
    若该动作跑完后对应 output 路径下没有 pkl，则追加写入日志。
    """
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
    print(f"[日志] 未发现输出，已写入: {MISSING_OUTPUT_LOG}  ->  {data_idx} 缺失: {', '.join(missing)}")


def run_optimization_for_data(data_idx: str, gpu_id: int, stage: int = None):
    """
    为指定的数据运行优化

    Args:
        data_idx: 数据索引
        gpu_id: GPU ID
        stage: 阶段 (1 或 2)，如果为None则运行两个阶段
    """
    print(f"开始处理数据: {data_idx} (GPU {gpu_id})")

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    base_cmd = [
        "python", "main/dataset/mano2dexhand_segmented.py",
        "--data_idx", data_idx,
        "--dexhand", "inspire",
        "--iter", "15000",
        "--headless",
        "--draw_all_lines", "0"
    ]

    # 为左右手分别运行
    sides = ["left", "right"]

    for side in sides:
        cmd = base_cmd.copy()
        cmd.extend(["--side", side])

        # 设置contact data路径（相对项目根 BASE_DIR）
        contact_data_path = os.path.join(BASE_DIR, "data", f"contacts_{side}", "humoto", f"contacts_{data_idx}.pkl")
        cmd.extend(["--contact_data", contact_data_path])

        # 如果指定了stage，添加stage参数
        if stage is not None:
            cmd.extend(["--stage", str(stage)])

        print(f"运行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, env=env, check=True, cwd=BASE_DIR)
            print(f"完成: {data_idx} {side} hand (stage {stage if stage else 'all'})")
        except subprocess.CalledProcessError as e:
            print(f"错误: {data_idx} {side} hand 运行失败 (stage {stage if stage else 'all'}): {e}")
            return False

    return True

def run_two_stage_optimization(data_idx: str, gpu_id: int):
    """
    为指定数据运行两阶段优化：先stage 1，再stage 2
    """
    print(f"=== 开始两阶段优化: {data_idx} (GPU {gpu_id}) ===")

    # Stage 1
    print(f"Stage 1 开始: {data_idx}")
    success1 = run_optimization_for_data(data_idx, gpu_id, stage=1)
    if not success1:
        print(f"Stage 1 失败: {data_idx}")
        return False

    print(f"Stage 1 完成: {data_idx}")

    # Stage 2
    print(f"Stage 2 开始: {data_idx}")
    success2 = run_optimization_for_data(data_idx, gpu_id, stage=2)
    if not success2:
        print(f"Stage 2 失败: {data_idx}")
        return False

    print(f"Stage 2 完成: {data_idx}")
    print(f"=== 两阶段优化完成: {data_idx} ===")
    return True

def main():
    parser = argparse.ArgumentParser(description="批量运行 mano2dexhand_segmented.py 优化")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7",
                       help="可用的GPU ID列表，用逗号分隔 (默认: 0,1,2,3,4,5,6,7；使用 --missing_15 时默认 5,7,11)")
    parser.add_argument("--max_parallel", type=int, default=4,
                       help="最大并行进程数 (默认: 4)")
    parser.add_argument("--data_indices", type=str, default=None,
                       help="指定要处理的数据索引，用逗号分隔。如果不指定，则处理所有未完成的数据")
    parser.add_argument("--missing_15", action="store_true",
                       help="只跑 humoto 里未同时出现在 lh/rh 的 15 个动作，并默认使用 GPU 5,7,11")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                       help="指定运行的阶段 (1 或 2)。如果不指定，则运行两阶段")

    args = parser.parse_args()

    # 使用 --missing_15 时：固定 15 个动作 + 默认 5,7,11 卡
    if args.missing_15:
        if args.gpu_ids == "0,1,2,3,4,5,6,7":
            args.gpu_ids = "5,7,11"
        if args.data_indices is None:
            args.data_indices = ",".join(MISSING_FROM_BOTH_15)

    # 解析GPU IDs
    gpu_ids = [int(gpu.strip()) for gpu in args.gpu_ids.split(',')]
    print(f"可用GPU: {gpu_ids}")

    # 确定要处理的数据
    if args.data_indices:
        target_data = [idx.strip() for idx in args.data_indices.split(',')]
    else:
        # 获取未完成的数据
        completed_set = set(COMPLETED_DATA_INDICES)
        target_data = [idx for idx in ALL_DATA_INDICES if idx not in completed_set]

    print(f"需要处理的数据数量: {len(target_data)}")
    print(f"数据列表: {target_data}")

    if not target_data:
        print("没有需要处理的数据")
        return

    # 创建进程池
    with mp.Pool(processes=min(args.max_parallel, len(gpu_ids))) as pool:
        # 为每个数据分配GPU并提交任务
        tasks = []
        for i, data_idx in enumerate(target_data):
            gpu_id = gpu_ids[i % len(gpu_ids)]

            if args.stage is not None:
                # 只运行指定阶段
                task = pool.apply_async(run_optimization_for_data, (data_idx, gpu_id, args.stage))
            else:
                # 运行两阶段优化
                task = pool.apply_async(run_two_stage_optimization, (data_idx, gpu_id))

            tasks.append((data_idx, task))

        # 等待所有任务完成
        completed_count = 0
        for data_idx, task in tasks:
            try:
                success = task.get(timeout=3600*24)  # 24小时超时
                completed_count += 1
                status = "成功" if success else "失败"
                print(f"[{completed_count}/{len(tasks)}] {data_idx}: {status}")
                # 仅在 stage 2 也跑完（任务成功）时再检查 output；stage 1 失败时不会产生最终 pkl，不记缺失
                if success:
                    check_and_log_missing_output(data_idx)
            except Exception as e:
                print(f"[{completed_count+1}/{len(tasks)}] {data_idx}: 异常 - {e}")
                completed_count += 1

    print("所有任务处理完成")

if __name__ == "__main__":
    main()


