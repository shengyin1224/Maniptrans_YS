#!/usr/bin/env python
"""
快速可视化手-物体接触点

用法示例:
    python main/dataset/quick_visualize_contacts.py \
        --data_idx "baking_with_spatula_mixing_bowl_and_scooping_to_tray-244" \
        --dexhand "inspire" \
        --side "left" \
        --threshold 0.02
"""

import os
import sys
import argparse
import subprocess
from termcolor import cprint

def main():
    parser = argparse.ArgumentParser(description="快速可视化手-物体接触点")
    parser.add_argument("--data_idx", type=str, required=True, help="数据索引")
    parser.add_argument("--dexhand", type=str, default="inspire", help="手部模型类型")
    parser.add_argument("--side", type=str, default="left", help="手部侧边 (left/right)")
    parser.add_argument("--threshold", type=float, default=0.02, help="接触距离阈值（米）")
    parser.add_argument("--skip_compute", action="store_true", help="跳过接触计算，直接使用已有数据")
    
    args = parser.parse_args()
    
    # 打印配置信息
    cprint("=" * 60, "cyan")
    cprint("手-物体接触点可视化", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")
    cprint(f"数据索引: {args.data_idx}", "white")
    cprint(f"手部模型: {args.dexhand}", "white")
    cprint(f"手部侧边: {args.side}", "white")
    cprint(f"接触阈值: {args.threshold}m ({args.threshold*1000:.1f}mm)", "white")
    cprint("=" * 60, "cyan")
    print()
    
    # 确定数据集类型和文件路径
    if args.data_idx.isdigit():
        # 数字索引，假设是 OakInk-v2
        dataset_type = "oakink2"
        contact_file = f"data/contacts/oakink2/contacts_{args.data_idx}.pkl"
    else:
        # 字符串索引，假设是 Humoto
        dataset_type = "humoto"
        contact_file = f"data/contacts/humoto/contacts_{args.data_idx}.pkl"
    
    # 步骤1: 计算接触数据（如果需要）
    if not args.skip_compute or not os.path.exists(contact_file):
        cprint("[步骤 1/2] 计算接触数据...", "yellow", attrs=['bold'])
        
        cmd = [
            "python", "main/dataset/compute_hand_object_contacts.py",
            "--data_idx", args.data_idx,
            "--dexhand", args.dexhand,
            "--side", args.side,
            "--threshold", str(args.threshold),
            "--output_dir", "data/contacts"
        ]
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            cprint("\n错误: 接触数据计算失败", "red", attrs=['bold'])
            sys.exit(1)
        
        cprint(f"\n✓ 接触数据已保存到: {contact_file}", "green")
    else:
        cprint(f"[跳过计算] 使用已有接触数据: {contact_file}", "yellow")
    
    # 检查文件是否存在
    if not os.path.exists(contact_file):
        cprint(f"\n错误: 接触数据文件不存在: {contact_file}", "red", attrs=['bold'])
        sys.exit(1)
    
    # 步骤2: 运行可视化
    print()
    cprint("[步骤 2/2] 启动可视化...", "yellow", attrs=['bold'])
    cprint("提示: 红色球体表示手-物体接触点", "cyan")
    print()
    
    cmd = [
        "python", "main/dataset/mano2dexhand.py",
        "--data_idx", args.data_idx,
        "--dexhand", args.dexhand,
        "--side", args.side,
        "--headless", "False",
        "--contact_data", contact_file
    ]
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        cprint("\n错误: 可视化启动失败", "red", attrs=['bold'])
        sys.exit(1)
    
    print()
    cprint("✓ 可视化已完成", "green", attrs=['bold'])

if __name__ == "__main__":
    main()

