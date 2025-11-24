import os
import subprocess
import glob
import argparse # NEW: Import the argparse library

# --- 配置路径和参数 ---

# 脚本和数据目录的根路径 (相对于你运行此脚本的位置)
# 假设你的目录结构是:
# ./process_all_objects.py
# ./maniptrans_envs/
# ./data/
SCRIPT_PATH = "maniptrans_envs/lib/utils/coacd_process.py"
INPUT_DIR_ROOT = "data/OakInk-v2/object_raw/align_ds"
OUTPUT_DIR_ROOT = "data/OakInk-v2/coacd_object_preview/align_ds"

# coacd_process.py 的命令行参数
# 注意：输入(-i)和输出(-o)会在循环中自动设置
COACD_ARGS = [
    '--max-convex-hull', '32',
    '--seed', '1',
    '-mi', '2000',
    '-md', '5',
    '-t', '0.07'
]

# --- 主逻辑 ---

def main():
    """
    遍历所有或指定的对象目录，执行近似凸分解。
    """
    # --- NEW: 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Process 3D object files using coacd_process.py.")
    parser.add_argument(
        '-d', '--directory',
        type=str,
        help="Specify a single subdirectory name (e.g., C10001) to process. If not provided, all directories will be processed."
    )
    args = parser.parse_args()

    # 检查必要的路径是否存在，避免脚本中途出错
    if not os.path.exists(SCRIPT_PATH):
        print(f"错误: 脚本路径不存在 '{SCRIPT_PATH}'")
        return
    if not os.path.isdir(INPUT_DIR_ROOT):
        print(f"错误: 输入数据目录不存在 '{INPUT_DIR_ROOT}'")
        return

    # --- NEW: 2. 根据命令行参数确定要处理的目录 ---
    subdirs = []
    if args.directory:
        # 如果用户指定了一个目录
        target_dir = os.path.join(INPUT_DIR_ROOT, args.directory)
        if os.path.isdir(target_dir):
            subdirs.append(args.directory)
            print(f"指定模式: 将只处理目录 '{args.directory}'。")
        else:
            print(f"错误: 指定的目录不存在 '{target_dir}'")
            return
    else:
        # 如果用户没有指定目录，则获取所有目录 (原始行为)
        try:
            subdirs = [d for d in os.listdir(INPUT_DIR_ROOT) if os.path.isdir(os.path.join(INPUT_DIR_ROOT, d))]
            print(f"批量模式: 成功找到 {len(subdirs)} 个对象目录。")
        except FileNotFoundError:
            print(f"错误: 无法列出目录 '{INPUT_DIR_ROOT}'。请检查路径是否正确。")
            return
    
    if not subdirs:
        print("没有找到需要处理的目录。")
        return
        
    total_dirs = len(subdirs)
    processed_count = 0

    # 遍历每一个子目录 (现在 subdirs 列表可能是所有目录，也可能只有一个)
    for i, dirname in enumerate(subdirs):
        input_subdir = os.path.join(INPUT_DIR_ROOT, dirname)
        output_subdir = os.path.join(OUTPUT_DIR_ROOT, dirname)

        # 查找输入文件 (优先 .obj, 其次 .ply)
        # 使用 glob 可以轻松处理文件名不一定和目录名相同的情况
        input_file = None
        
        # 1. 查找 .obj 文件
        obj_files = glob.glob(os.path.join(input_subdir, '*.obj'))
        if obj_files:
            input_file = obj_files[0] # 使用找到的第一个 .obj 文件
        else:
            # 2. 如果没有 .obj, 查找 .ply 文件
            ply_files = glob.glob(os.path.join(input_subdir, '*.ply'))
            if ply_files:
                input_file = ply_files[0] # 使用找到的第一个 .ply 文件

        # 如果在该目录下既没有 .obj 也没有 .ply, 则跳过
        if not input_file:
            print(f"[{i+1}/{total_dirs}] 警告: 在 '{input_subdir}' 中未找到 .obj 或 .ply 文件，已跳过。")
            continue
            
        # 确保输出目录存在
        os.makedirs(output_subdir, exist_ok=True)
        
        # 构建输出文件路径
        base_filename = os.path.basename(input_file)
        output_file = os.path.join(output_subdir, base_filename)

        # 构建完整的命令行
        command = [
            'python',
            SCRIPT_PATH,
            '-i', input_file,
            '-o', output_file
        ] + COACD_ARGS

        print(f"--- [{i+1}/{total_dirs}] 正在处理: {input_file} ---")
        
        # 执行命令
        try:
            # 使用 subprocess.run 等待命令执行完成
            # text=True 将输出解码为字符串, capture_output=True 捕获标准输出和错误
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print(f"成功: 输出已保存至 -> {output_file}")
            # 如果想看详细的输出日志，可以取消下面这行的注释
            # if result.stdout:
            #     print(result.stdout)
            processed_count += 1
        except subprocess.CalledProcessError as e:
            # 如果命令执行失败 (返回非0状态码)
            print(f"!!!!!! 处理失败: {input_file} !!!!!!")
            print("命令:", ' '.join(e.cmd))
            print("错误信息:")
            print(e.stderr) # 打印标准错误输出，通常包含错误原因
        except FileNotFoundError:
            print("错误: 'python' 命令未找到。请确保 Python 已安装并在系统 PATH 中。")
            break

    print("\n--- 处理完成 ---")
    print(f"总计: {total_dirs} 个目录, 成功处理: {processed_count} 个。")


if __name__ == "__main__":
    main()