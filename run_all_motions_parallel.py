#!/usr/bin/env python3
"""
对 runs/data_motions_nn_over10.txt 中每个子文件夹：
- 用 nn 下最新的一个 .pth 跑 run_all_motions 的两阶段（FIXED_INIT + RANDOM_INIT）
- 每张卡跑一个 data motion，跑完或 segmentation 后自动跑下一个
- 结果写入 txt，包含 status_tag（S_x_F_y / SUCCESS_x）
"""
import os
import re
import sys
import yaml
import h5py
import argparse
import threading
import time
import shutil
from datetime import datetime
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# 项目根目录（脚本所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(BASE_DIR, "runs")
DUMPS_ROOT = os.path.join(BASE_DIR, "dumps")
MOTION_LIST_FILE = os.path.join(RUNS_ROOT, "data_motions_nn_over10.txt")
# 结果输出（默认：<项目根>/runs/run_all_motions_results_YYYYMMDD_HHMMSS.txt）
RESULTS_DIR = RUNS_ROOT
RESULTS_FILE_PREFIX = "run_all_motions_results"
# Physics NaN 错误关键词，用于检测并触发 num_envs 重试（最多 3 次）
NAN_ERROR_MARKER = "Physics simulation generated NaN values."
MAX_NAN_ATTEMPTS = 10
NUM_ENVS_RETRIES = [16, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # 三次尝试依次使用的 num_envs

_RUN_START_TS = None  # 在 main() 中设置，用于只清理本次运行期间产生的 dumps
_TRACK_LOCK = threading.Lock()
_SEEN_EXPERIMENTS = set()  # 记录本次运行中出现过的 experiment 名称（fixed/random 各自算一个）
_KEEP_DUMP_DIR = {}  # exp_name -> dump_dir（仅保留本次运行里“最终要留”的一个）
_CHECKPOINT_OVERRIDE = {}  # key: f"{motion_key}|||{subfolder}" -> 绝对路径 checkpoint.pth


def _list_dump_dirs_for_experiment(exp_name: str):
    """返回 dumps 下所有匹配该 experiment 的 dump 目录（full path）"""
    if not os.path.isdir(DUMPS_ROOT):
        return []
    prefix = f"dump_{exp_name}__"
    out = []
    try:
        for d in os.listdir(DUMPS_ROOT):
            if d.startswith(prefix):
                full = os.path.join(DUMPS_ROOT, d)
                if os.path.isdir(full):
                    out.append(full)
    except Exception:
        return []
    return out


def _is_created_in_this_run(path: str) -> bool:
    """只清理本次运行期间产生的 dump，避免误删历史结果"""
    if _RUN_START_TS is None:
        return False
    try:
        return os.path.getmtime(path) >= (_RUN_START_TS - 1.0)
    except Exception:
        return False


def _cleanup_experiment_dumps(exp_name: str):
    """
    对某个 experiment：
    - 仅在“本次运行期间新产生”的 dump 中，保留 _KEEP_DUMP_DIR[exp_name]（若存在）
    - 其余全部删除（包括 NaN/重试产生的重复 dump）
    """
    keep = None
    with _TRACK_LOCK:
        keep = _KEEP_DUMP_DIR.get(exp_name)

    candidates = [p for p in _list_dump_dirs_for_experiment(exp_name) if _is_created_in_this_run(p)]
    if not candidates:
        return 0

    deleted = 0
    for p in candidates:
        if keep and os.path.abspath(p) == os.path.abspath(keep):
            continue
        try:
            shutil.rmtree(p, ignore_errors=True)
            deleted += 1
        except Exception:
            pass
    return deleted


def get_rollout_counts(dump_dir):
    """读取 HDF5 并返回 (成功条数, 失败条数)"""
    hdf5_path = os.path.join(dump_dir, "rollouts.hdf5")
    if not os.path.exists(hdf5_path):
        return 0, 0
    s_count, f_count = 0, 0
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "rollouts" in f:
                if "successful" in f["rollouts"]:
                    s_count = len(f["rollouts"]["successful"].keys())
                if "failed" in f["rollouts"]:
                    f_count = len(f["rollouts"]["failed"].keys())
    except Exception as e:
        print(f"!!! 读取 HDF5 失败: {e}")
    return s_count, f_count


def find_latest_dump_dir(experiment_name):
    """根据 experiment 名字找到 dumps 下最新的对应文件夹"""
    prefix = f"dump_{experiment_name}__"
    try:
        if not os.path.isdir(DUMPS_ROOT):
            return None
        candidates = [d for d in os.listdir(DUMPS_ROOT) if d.startswith(prefix)]
        if not candidates:
            return None
        candidates.sort(
            key=lambda x: os.path.getmtime(os.path.join(DUMPS_ROOT, x)),
            reverse=True,
        )
        return os.path.join(DUMPS_ROOT, candidates[0])
    except Exception:
        return None


def parse_motion_list(path):
    """
    解析 data_motions_nn_over10.txt，返回 [(motion_key, subfolder), ...]
    跳过注释和空行；每行格式: 数字  motion_key  ->  subfolder/
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                continue
            left, right = line.split("->", 1)
            # 右侧可能带额外字段（例如 "SR=0.6136"），只取第一个 token 作为子目录名
            right = right.strip()
            right = right.split()[0] if right else ""
            right = right.rstrip("/").strip()
            if not right:
                continue
            # motion_key: 左侧去掉开头的数字，取第一个空格后的整段（或整段去掉首数字）
            parts = left.strip().split(None, 1)
            motion_key = parts[1].strip() if len(parts) > 1 else right
            out.append((motion_key, right))
    return out


def parse_pth_files_list(path):
    """
    解析 pth_files_list.txt，返回 [(motion_key, subfolder), ...]
    - 支持两部分：
      1) 相对路径形式：0119_nerv/.../nn/xxx.pth
      2) 绝对路径形式：/home/.../runs/.../nn/xxx.pth
    - 每个 pth 后面紧跟一行: DataIndice: <motion_key>[  ...注释]
    - subfolder 统一转换成相对 RUNS_ROOT 的路径，供 run_one_motion 使用：
      os.path.join(RUNS_ROOT, subfolder) == 该 pth 所在子任务目录（nn 的父目录）
    """
    out = []
    current_pth_dir = None  # nn 目录的父目录（task 子目录）
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # pth 路径行
            if line.endswith(".pth"):
                pth_path = line
                if not os.path.isabs(pth_path):
                    pth_path = os.path.join(BASE_DIR, pth_path)
                # 既支持 .../<task_dir>/nn/xxx.pth 也支持 .../<task_dir>/xxx.pth
                pth_dir = os.path.dirname(pth_path)
                if os.path.basename(pth_dir) == "nn":
                    task_dir = os.path.dirname(pth_dir)
                else:
                    task_dir = pth_dir
                current_pth_dir = (task_dir, pth_path)
                continue
            # DataIndice 行
            if line.lower().startswith("dataindice"):
                if current_pth_dir is None:
                    continue
                try:
                    _, rest = line.split(":", 1)
                    # 去掉注释，只取第一个 token 作为 motion_key
                    rest = rest.strip()
                    if "#" in rest:
                        rest = rest.split("#", 1)[0].strip()
                    motion_key = rest.split()[0] if rest else ""
                    if not motion_key:
                        continue
                    task_dir, pth_path_abs = current_pth_dir
                    # subfolder: task_dir 相对 RUNS_ROOT 的路径（可能带 ../）
                    subfolder = os.path.relpath(task_dir, RUNS_ROOT)
                    out.append((motion_key, subfolder))
                    # 记录该 motion 的 checkpoint 绝对路径，run_one_motion 会优先使用
                    key = f"{motion_key}|||{subfolder}"
                    _CHECKPOINT_OVERRIDE[key] = pth_path_abs
                except Exception:
                    # 某一行解析失败就跳过，不影响其它条目
                    pass
                finally:
                    current_pth_dir = None
                continue
    return out


def get_latest_pth(nn_dir):
    """
    选择 checkpoint 规则：
    - 若存在多个文件名包含 'last' 的 .pth：按 mtime 从新到旧排序，取“第二新”的那个
    - 若不存在 'last'：对全部 .pth 按 mtime 从新到旧排序，取“第二新”的那个
    - 若候选不足 2 个：退化为取最新（第 1 个）
    """
    if not os.path.isdir(nn_dir):
        return None
    pth_files = [f for f in os.listdir(nn_dir) if f.endswith(".pth")]
    if not pth_files:
        return None
    # 优先 last
    last_candidates = [f for f in pth_files if "last" in f.lower()]
    candidates = last_candidates if len(last_candidates) >= 2 else pth_files
    candidates.sort(key=lambda x: os.path.getmtime(os.path.join(nn_dir, x)), reverse=True)
    return candidates[1] if len(candidates) >= 2 else candidates[0]


def parse_ep_from_pth_filename(pth_basename):
    """
    从 pth 文件名解析 ep 值，例如：
    last_0201_walk_with_low_chair-308_1024_ep_10800_rew_... -> 10800
    匹配 _ep_ 后面的数字。
    """
    m = re.search(r"_ep_(\d+)", pth_basename)
    return int(m.group(1)) if m else None


def get_num_envs(config_path):
    """从子文件夹 config.yaml 读取 num_envs，默认 128。"""
    if not os.path.isfile(config_path):
        return 128
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        n = cfg.get("num_envs")
        if n is not None:
            return int(n)
        task_env = (cfg.get("task") or {}).get("env") or {}
        n = task_env.get("numEnvs")
        return int(n) if n is not None else 128
    except Exception:
        return 128


def get_data_indices(config_path, folder_name):
    """从 config.yaml 读取 dataIndices，逻辑与 run_all_motions 一致。"""
    if not os.path.isfile(config_path):
        return [folder_name.replace("NO_0105_", "")]
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        data_indices = cfg.get("dataIndices")
        if not data_indices or (
            isinstance(data_indices, str) and "$" in data_indices
        ):
            data_indices = (
                cfg.get("task") or {}
            ).get("env", {}) or {}
            data_indices = data_indices.get("dataIndices")
        if not isinstance(data_indices, list) or (
            len(data_indices) > 0
            and isinstance(data_indices[0], str)
            and "$" in (data_indices[0] or "")
        ):
            data_indices = [folder_name.replace("NO_0105_", "")]
        return list(data_indices) if data_indices else [folder_name]
    except Exception:
        return [folder_name.replace("NO_0105_", "")]


def run_one_motion(motion_key, subfolder, gpu_id, result_lines, result_lock):
    """
    对单个 data motion 跑两阶段，使用指定 GPU。
    将结果追加到 result_lines（带 result_lock），并写入结果文件。
    """
    folder_path = os.path.join(RUNS_ROOT, subfolder)
    config_path = os.path.join(folder_path, "config.yaml")

    # 若在 pth_files_list.txt 中为该 motion 显式指定了 checkpoint，则优先使用该路径
    override_key = f"{motion_key}|||{subfolder}"
    checkpoint_override = _CHECKPOINT_OVERRIDE.get(override_key)

    checkpoint_path = None
    checkpoint_name = None

    if checkpoint_override and os.path.isfile(checkpoint_override):
        checkpoint_path = checkpoint_override
        checkpoint_name = os.path.basename(checkpoint_path)
    else:
        nn_dir = os.path.join(folder_path, "nn")
        if not os.path.isdir(nn_dir):
            with result_lock:
                line = f"{datetime.now().isoformat()}\t{motion_key}\t{subfolder}\tGPU{gpu_id}\tno_nn_dir\t-\t-\t-\t-\t-\n"
                result_lines.append(line)
            return

        checkpoint_name = get_latest_pth(nn_dir)
        if not checkpoint_name:
            with result_lock:
                line = f"{datetime.now().isoformat()}\t{motion_key}\t{subfolder}\tGPU{gpu_id}\tno_pth\t-\t-\t-\t-\t-\n"
                result_lines.append(line)
            return

        checkpoint_path = os.path.join(nn_dir, checkpoint_name)
    # FLAG: ep >= 140*30*1024/num_envs，num_envs 从该子文件夹 config 读
    num_envs = get_num_envs(config_path)
    threshold = (140 * 30 * 1024) / num_envs
    ep_value = parse_ep_from_pth_filename(checkpoint_name)
    if ep_value is not None:
        FLAG = 1 if ep_value >= threshold else 0
        ep_str, thresh_str, flag_str = str(ep_value), f"{threshold:.1f}", str(FLAG)
    else:
        ep_str, thresh_str, flag_str = "-", f"{threshold:.1f}", "-"
    data_indices = get_data_indices(config_path, subfolder)
    data_indices_str = "[" + ",".join(str(x) for x in data_indices) + "]"
    motion_name_base = motion_key  # 与 run_all_motions 一致用 motion key 作 experiment 前缀

    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES", "0")
    devices = [x.strip() for x in visible.split(",") if x.strip()]
    if gpu_id < len(devices):
        env["CUDA_VISIBLE_DEVICES"] = devices[gpu_id]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    exit_kind = "ok"
    fixed_tag = "-"
    random_tag = "-"
    num_envs_used = None

    def _run_phase(phase_name: str, num_envs_cur: int):
        """
        Run one phase once.
        Returns (ok: bool, is_nan: bool, status_tag: str|None, fail_kind: str|None).
        - ok=True  => status_tag is set (or "no_dump")
        - is_nan=True => Physics NaN detected
        - ok=False and is_nan=False => fail_kind in {"segfault","error"}
        """
        base_cmd = [
            "python",
            "main/rl/train.py",
            "task=ResDexHand",
            "dexhand=inspire",
            "side=BiH",
            f"num_envs={num_envs_cur}",
            "learning_rate=2e-4",
            "rh_base_model_checkpoint=assets/imitator_rh_inspire.pth",
            "lh_base_model_checkpoint=assets/imitator_lh_inspire.pth",
            "actionsMovingAverage=0.4",
            "test=True",
            "save_rollouts=True",
            "min_episode_length=200",
            f"dataIndices={data_indices_str}",
            f"checkpoint={checkpoint_path}",
            "headless=True",
        ]

        if phase_name == "FIXED_INIT":
            cmd = base_cmd + [
                "randomStateInit=False",
                "save_successful_rollouts_only=True",
                "num_rollouts_to_save=3",
                "num_rollouts_to_run=300",
                f"experiment={motion_name_base}_fixed",
            ]
        else:
            cmd = base_cmd + [
                "randomStateInit=True",
                "save_successful_rollouts_only=False",
                "num_rollouts_to_save=20",
                "num_rollouts_to_run=200",
                f"experiment={motion_name_base}_random",
            ]

        exp_name = next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("experiment="))
        with _TRACK_LOCK:
            _SEEN_EXPERIMENTS.add(exp_name)

        try:
            # 既要能检测 NaN（需要读取输出），又要让输出继续在终端可见
            # 这里通过 PIPE 捕获 stdout+stderr，一边打印一边检查关键字
            saw_nan = False
            p = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # 记录运行前已有的 dump，用来推断本次 phase 是否新产生了 dump
            before = set(_list_dump_dirs_for_experiment(exp_name))

            if p.stdout is not None:
                for line in p.stdout:
                    # passthrough to terminal
                    print(line, end="")
                    if NAN_ERROR_MARKER in line:
                        saw_nan = True
            p.wait()

            after = set(_list_dump_dirs_for_experiment(exp_name))
            # 对于 NaN 情况，dump 目录往往只在失败时出现；先记录，统一在所有任务结束后清理
            if p.returncode != 0 and saw_nan:
                # 只记录“本次 phase 新产生”的 dump，避免碰到历史目录
                new_dirs = [d for d in (after - before) if _is_created_in_this_run(d)]
                # 如果无法精确 diff（例如重跑覆盖 mtime），就保守把本次运行期间的都记录下来
                if not new_dirs:
                    new_dirs = [d for d in after if _is_created_in_this_run(d)]
                with _TRACK_LOCK:
                    # NaN 下不设置 keep，最终会删掉本次运行产生的所有 dump
                    for d in new_dirs:
                        # 仅用于统计/排查（真正删除在 main() 收尾统一做）
                        pass
                return False, True, None, None
            if p.returncode in (139, -11, 11) or (p.returncode < 0 and p.returncode != 0):
                return False, False, None, "segfault"
            if p.returncode != 0:
                return False, False, None, "error"
        except Exception:
            return False, False, None, "error"

        # phase ok -> collect status tag from newest dump
        dump_dir = find_latest_dump_dir(exp_name)
        if dump_dir:
            # 本次运行：该 experiment 只保留一个 dump（最新的那个）
            with _TRACK_LOCK:
                _KEEP_DUMP_DIR[exp_name] = dump_dir
            s_count, f_count = get_rollout_counts(dump_dir)
            status_tag = f"S_{s_count}_F_{f_count}" if f_count > 0 else f"SUCCESS_{s_count}"
            count_file = os.path.join(dump_dir, f"COUNT_{status_tag}.txt")
            try:
                with open(count_file, "w") as cf:
                    cf.write(f"Timestamp: {datetime.now()}\n")
                    cf.write(f"Successful rollouts: {s_count}\n")
                    cf.write(f"Failed rollouts: {f_count}\n")
                    cf.write(f"Total: {s_count + f_count}\n")
            except Exception:
                pass
            return True, False, status_tag, None

        # dump not found
        return True, False, "no_dump", None

    # 1) FIXED_INIT: NaN 重试只针对 FIXED_INIT 本身
    fixed_ok = False
    for attempt in range(MAX_NAN_ATTEMPTS):
        num_envs_cur = NUM_ENVS_RETRIES[attempt] if attempt < len(NUM_ENVS_RETRIES) else NUM_ENVS_RETRIES[-1]
        num_envs_used = num_envs_cur
        ok, is_nan, status_tag, fail_kind = _run_phase("FIXED_INIT", num_envs_cur)
        if is_nan:
            if attempt < MAX_NAN_ATTEMPTS - 1:
                nxt = NUM_ENVS_RETRIES[attempt + 1] if attempt + 1 < len(NUM_ENVS_RETRIES) else NUM_ENVS_RETRIES[-1]
                print(f"[GPU{gpu_id}] {motion_key} FIXED_INIT NaN (num_envs={num_envs_cur})，第 {attempt + 1}/{MAX_NAN_ATTEMPTS} 次，将用 num_envs={nxt} 重试")
                time.sleep(10)
                continue
            exit_kind = "nan_skip"
            fixed_tag = "-"
            random_tag = "-"
            break
        if not ok:
            exit_kind = fail_kind or "error"
            fixed_tag = f"exit_{exit_kind}"
            random_tag = "-"
            break
        fixed_tag = status_tag
        fixed_ok = True
        break

    # 2) RANDOM_INIT: 只有在 FIXED_INIT 成功后才跑；NaN 重试只重跑 RANDOM_INIT
    if exit_kind == "ok" and fixed_ok:
        for attempt in range(MAX_NAN_ATTEMPTS):
            num_envs_cur = NUM_ENVS_RETRIES[attempt] if attempt < len(NUM_ENVS_RETRIES) else NUM_ENVS_RETRIES[-1]
            num_envs_used = num_envs_cur
            ok, is_nan, status_tag, fail_kind = _run_phase("RANDOM_INIT", num_envs_cur)
            if is_nan:
                if attempt < MAX_NAN_ATTEMPTS - 1:
                    nxt = NUM_ENVS_RETRIES[attempt + 1] if attempt + 1 < len(NUM_ENVS_RETRIES) else NUM_ENVS_RETRIES[-1]
                    print(f"[GPU{gpu_id}] {motion_key} RANDOM_INIT NaN (num_envs={num_envs_cur})，第 {attempt + 1}/{MAX_NAN_ATTEMPTS} 次，将用 num_envs={nxt} 重试（不重跑 FIXED_INIT）")
                    time.sleep(10)
                    continue
                exit_kind = "nan_skip"
                # 注意：FIXED 已经跑完，保留 fixed_tag
                random_tag = "-"
                break
            if not ok:
                exit_kind = fail_kind or "error"
                random_tag = f"exit_{exit_kind}"
                break
            random_tag = status_tag
            break

    with result_lock:
        line = f"{datetime.now().isoformat()}\t{motion_key}\t{subfolder}\tGPU{gpu_id}\t{exit_kind}\t{fixed_tag}\t{random_tag}\t{ep_str}\t{thresh_str}\t{flag_str}\n"
        result_lines.append(line)
        # 实时追加到结果文件
        results_path = os.path.join(RESULTS_DIR, _results_filename)
        with open(results_path, "a", encoding="utf-8") as rf:
            rf.write(line)


_results_filename = None


def main():
    global _results_filename
    global _RUN_START_TS

    parser = argparse.ArgumentParser(description="多卡并行跑 data motions（nn 最新 pth）")
    parser.add_argument(
        "--list-file",
        default=MOTION_LIST_FILE,
        help="data motion 列表文件路径",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="使用的 GPU 数量，默认用 CUDA_VISIBLE_DEVICES 或 1",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="结果 txt 所在目录",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.list_file):
        print(f"列表文件不存在: {args.list_file}")
        sys.exit(1)

    # 1) 优先按老的 data_motions_*.txt 格式解析（带 '->'）
    motions = parse_motion_list(args.list_file)
    # 2) 若解析不到，则尝试按 pth_files_list.txt 格式解析
    if not motions:
        motions = parse_pth_files_list(args.list_file)
    if not motions:
        print("未解析到任何 data motion（既不是 data_motions_*.txt 也不是 pth_files_list.txt 格式）")
        sys.exit(0)

    num_gpus = args.num_gpus
    if num_gpus is None:
        cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_dev:
            num_gpus = len([x for x in cuda_dev.split(",") if x.strip()])
        else:
            num_gpus = 1
    num_gpus = max(1, num_gpus)

    _results_filename = (
        f"{RESULTS_FILE_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    results_path = os.path.join(args.results_dir, _results_filename)
    os.makedirs(args.results_dir, exist_ok=True)

    # 表头（FLAG=1 表示 ep >= 140*30*1024/num_envs）
    header = (
        "timestamp\tmotion_key\tsubfolder\tgpu\texit_kind\t"
        "FIXED_status_tag\tRANDOM_status_tag\tep_value\tthreshold\tFLAG\n"
    )
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(header)
    print(f"结果 txt 写入: {results_path}")
    print(f"  （默认目录: 项目根/runs/，文件名: {RESULTS_FILE_PREFIX}_YYYYMMDD_HHMMSS.txt）")
    print(f"共 {len(motions)} 个 data motion，{num_gpus} 张卡并行")
    print("FIXED_status_tag / RANDOM_status_tag 即 run_all_motions 中的 S_x_F_y 或 SUCCESS_x")
    print("FLAG=1 表示 最新 pth 文件名中的 ep 值 >= 140*30*1024/num_envs（num_envs 来自该子文件夹 config.yaml）")
    print("若出现 Physics NaN，将自动用 num_envs=128→64→32 重试，最多 3 次；仍报错则记为 nan_skip 并跳过\n")

    # 标记本次脚本启动时间：后续只清理本次运行期间新生成的 dump 目录
    _RUN_START_TS = time.time()

    result_lines = []
    result_lock = threading.Lock()
    task_queue = Queue()
    for m in motions:
        task_queue.put(m)

    def worker(gpu_id):
        while True:
            try:
                motion_key, subfolder = task_queue.get_nowait()
            except Empty:
                break
            try:
                run_one_motion(
                    motion_key, subfolder, gpu_id, result_lines, result_lock
                )
            except Exception as e:
                # 线程内异常不能让整个任务“看起来像 0 秒结束”
                with result_lock:
                    line = (
                        f"{datetime.now().isoformat()}\t{motion_key}\t{subfolder}\tGPU{gpu_id}\t"
                        f"thread_error\t{str(e)[:80]}\t-\t-\t-\t-\n"
                    )
                    result_lines.append(line)
                    results_path = os.path.join(RESULTS_DIR, _results_filename)
                    with open(results_path, "a", encoding="utf-8") as rf:
                        rf.write(line)
            finally:
                task_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_id,))
        for gpu_id in range(num_gpus)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 收尾清理：每个 motion 的每个 phase（fixed/random）最终只保留一个 dump，
    # 把本次运行里由于 NaN/重试产生的重复 dump 全部删掉，避免下次混淆/重复。
    deleted_total = 0
    with _TRACK_LOCK:
        exps = list(_SEEN_EXPERIMENTS)
    for exp_name in exps:
        deleted_total += _cleanup_experiment_dumps(exp_name)
    if deleted_total > 0:
        print(f"\n清理完成：本次运行删除重复/NaN dump 目录 {deleted_total} 个（每个 motion 仅保留 1 个 fixed + 1 个 random）")

    print(f"\n全部完成，结果已写入: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
