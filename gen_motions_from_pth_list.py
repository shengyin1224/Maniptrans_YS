#!/usr/bin/env python3
"""
根据项目根目录下的 `pth_files_list.txt` 生成一个可被
`run_all_motions_parallel.py` 使用的 motion 列表 txt。

注意：
- 目前只自动支持 **第二部分**（在 `Maniptrans_YS/runs/` 下的 pth），
  因为 `run_all_motions_parallel.py` 要求子目录必须在 `runs/` 下面。
- 第一部分那些不在 `runs/` 里的旧实验目录（例如 `0119_nerv/...`），
  如果强行写进列表，脚本会在对应子目录下找不到 `nn/`，直接报 `no_nn_dir`，
  因此这里直接跳过，不写入生成的 txt。

生成规则：
- 对于每一条第二部分的条目：
  - `motion_key` = `DataIndice:` 后面的完整字符串
    （例如：`placing_trash_can_down_and_leaving_trash_in_trash_can-255`）
  - `subfolder` = `runs/` 与 `/nn/` 之间的子目录名
    （例如：`0127_placing_trash_can_down_and_leaving_trash_in_trash_can-255_1024__01-31-17-39-09`）
  - 输出行为：`数字  motion_key  ->  subfolder/`

运行方式（在 `Maniptrans_YS/` 目录下）：
    python gen_motions_from_pth_list.py

生成文件：
    runs/data_motions_from_pth_files_list.txt

然后可用：
    python run_all_motions_parallel.py --list-file runs/data_motions_from_pth_files_list.txt
"""

import os
import re


def parse_pth_files_list(pth_list_path, project_root):
    """
    解析 pth_files_list.txt，返回可供 run_all_motions_parallel.py 使用的条目：
        [(motion_key, subfolder), ...]

    兼容两种情况：
    - 第一部分：相对路径，例如 `0119_nerv/washing_hand/nn/xxx.pth`
      - 实际目录在 `Maniptrans_YS/0119_nerv/...` 下
      - 这里生成的 subfolder 形如 `../0119_nerv/washing_hand`
      - 原脚本中的 RUNS_ROOT=Maniptrans_YS/runs，folder_path=RUNS_ROOT/subfolder
        即解析为 Maniptrans_YS/runs/../0119_nerv/washing_hand
    - 第二部分：绝对路径、且位于 Maniptrans_YS/runs 下
      - 生成的 subfolder 为 runs 下的一层子目录名，例如 `0127_xxx_...`
    """
    results = []

    runs_root = os.path.join(project_root, "Maniptrans_YS", "runs")

    with open(pth_list_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    current_pth_line = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 记录上一行的 pth 路径（可以是绝对路径，也可以是相对路径）
        if stripped.endswith(".pth"):
            current_pth_line = stripped
            continue

        # 找 DataIndice 行，并且需要有当前 pth
        if stripped.startswith("DataIndice:") and current_pth_line is not None:
            # 提取 motion_key：去掉 "DataIndice:"，并砍掉注释部分
            di_part = stripped[len("DataIndice:") :].strip()
            if "#" in di_part:
                di_part = di_part.split("#", 1)[0].strip()
            motion_key = di_part

            p = current_pth_line

            # 情况 1：绝对路径，且在 Maniptrans_YS/runs 下（第二部分）
            marker = os.path.join("Maniptrans_YS", "runs") + os.sep
            subfolder = None
            if os.path.isabs(p) and marker in p:
                idx = p.index(marker)
                after_runs = p[idx + len(marker) :]  # 形如：0127_xxx/nn/last_xxx.pth
                parts = after_runs.split(os.sep)
                if len(parts) >= 2:
                    subfolder = parts[0].strip().rstrip("/")

            # 情况 2：相对路径（第一部分），例如 0119_nerv/washing_hand/nn/xxx.pth
            if subfolder is None and not os.path.isabs(p):
                rel = p
                # 只取 /nn/ 之前的部分作为目录
                if "/nn/" in rel:
                    prefix = rel.split("/nn/", 1)[0].rstrip("/")
                    # 通过 ../ 跳出 runs 目录，落到 Maniptrans_YS/<prefix>
                    subfolder = os.path.join("..", prefix)

            if not subfolder:
                current_pth_line = None
                continue

            # 做一层检查：对应子目录下有 nn 目录即可（config.yaml 缺失也允许跑，只是会用默认 num_envs/dataIndices 逻辑）
            folder_path = os.path.join(runs_root, subfolder)
            nn_dir = os.path.join(folder_path, "nn")
            if not os.path.isdir(nn_dir):
                current_pth_line = None
                continue

            results.append((motion_key, subfolder))
            current_pth_line = None

    return results


def write_motion_list_txt(motions, out_path):
    """
    按 run_all_motions_parallel.parse_motion_list 期望的格式写出：
        数字  motion_key  ->  subfolder/
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (motion_key, subfolder) in enumerate(motions, start=1):
            line = f"{idx:03d}  {motion_key}  ->  {subfolder}/\n"
            f.write(line)


def main():
    # 本脚本所在目录 = Maniptrans_YS/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    pth_list_path = os.path.join(project_root, "pth_files_list.txt")
    if not os.path.isfile(pth_list_path):
        raise FileNotFoundError(f"未找到 pth_files_list.txt: {pth_list_path}")

    motions = parse_pth_files_list(pth_list_path, project_root)
    if not motions:
        print("没有解析到任何位于 Maniptrans_YS/runs 下的条目，未生成列表文件。")
        return

    out_path = os.path.join(base_dir, "runs", "data_motions_from_pth_files_list.txt")
    write_motion_list_txt(motions, out_path)

    print(f"共解析到 {len(motions)} 个 data motion")
    print(f"已生成列表文件:\n  {out_path}")
    print("你可以用下面的命令调用 run_all_motions_parallel.py：")
    print("  cd Maniptrans_YS")
    print("  python run_all_motions_parallel.py --list-file runs/data_motions_from_pth_files_list.txt")


if __name__ == "__main__":
    main()

