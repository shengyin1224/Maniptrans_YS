import os
import subprocess
import yaml
import h5py
from datetime import datetime

# 配置路径
BASE_DIR = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS"
MOTIONS_ROOT = os.path.join(BASE_DIR, "0113_nerv")
DUMPS_ROOT = os.path.join(BASE_DIR, "dumps")

def get_rollout_counts(dump_dir):
    """读取 HDF5 文件并返回 (成功条数, 失败条数)"""
    hdf5_path = os.path.join(dump_dir, "rollouts.hdf5")
    if not os.path.exists(hdf5_path):
        return 0, 0
    s_count, f_count = 0, 0
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'rollouts' in f:
                if 'successful' in f['rollouts']:
                    s_count = len(f['rollouts']['successful'].keys())
                if 'failed' in f['rollouts']:
                    f_count = len(f['rollouts']['failed'].keys())
    except Exception as e:
        print(f"!!! 读取 HDF5 失败: {e}")
    return s_count, f_count

def find_latest_dump_dir(experiment_name):
    """根据 experiment 名字找到 dumps 下最新的对应文件夹"""
    prefix = f"dump_{experiment_name}__"
    try:
        candidates = [d for d in os.listdir(DUMPS_ROOT) if d.startswith(prefix)]
        if not candidates:
            return None
        # 按修改时间排序，取最新的
        candidates.sort(key=lambda x: os.path.getmtime(os.path.join(DUMPS_ROOT, x)), reverse=True)
        return os.path.join(DUMPS_ROOT, candidates[0])
    except Exception:
        return None

def run_tests():
    # 获取 0113_nerv 下的所有文件夹
    motion_folders = [f for f in os.listdir(MOTIONS_ROOT) if os.path.isdir(os.path.join(MOTIONS_ROOT, f))]
    motion_folders.sort()
    
    for folder in motion_folders:
        folder_path = os.path.join(MOTIONS_ROOT, folder)
        config_path = os.path.join(folder_path, "config.yaml")
        nn_dir = os.path.join(folder_path, "nn")
        
        if not os.path.exists(config_path) or not os.path.exists(nn_dir):
            continue
            
        with open(config_path, 'r') as f:
            try:
                cfg = yaml.safe_load(f)
                data_indices = cfg.get('dataIndices')
                if not data_indices or (isinstance(data_indices, str) and "$" in data_indices):
                    data_indices = cfg.get('task', {}).get('env', {}).get('dataIndices')
                if not isinstance(data_indices, list) or (len(data_indices) > 0 and isinstance(data_indices[0], str) and "$" in data_indices[0]):
                    data_indices = [folder.replace("NO_0105_", "")]
            except Exception:
                continue

        pth_files = [f for f in os.listdir(nn_dir) if f.endswith(".pth")]
        if not pth_files: continue
        checkpoint_name = next((f for f in pth_files if 'last' in f), pth_files[0])
        checkpoint_path = os.path.join(nn_dir, checkpoint_name)

        motion_name_base = folder.replace("NO_0105_", "")
        data_indices_str = "[" + ",".join(data_indices) + "]"
        
        base_cmd = [
            "python", "main/rl/train.py",
            "task=ResDexHand",
            "dexhand=inspire",
            "side=BiH",
            "num_envs=4",
            "learning_rate=2e-4",
            "rh_base_model_checkpoint=assets/imitator_rh_inspire.pth",
            "lh_base_model_checkpoint=assets/imitator_lh_inspire.pth",
            "actionsMovingAverage=0.4",
            "test=True",
            "save_rollouts=True",
            "min_episode_length=300",
            f"dataIndices={data_indices_str}",
            f"checkpoint={checkpoint_path}",
            "headless=True"
        ]

        # 两个运行阶段
        phases = [
            ("FIXED_INIT", base_cmd + [
                "randomStateInit=False", 
                "save_successful_rollouts_only=True", 
                "num_rollouts_to_save=10", 
                "num_rollouts_to_run=100", 
                f"experiment={motion_name_base}_fixed"
            ]),
            ("RANDOM_INIT", base_cmd + [
                "randomStateInit=True", 
                "save_successful_rollouts_only=False", 
                "num_rollouts_to_save=20", 
                "num_rollouts_to_run=300", 
                f"experiment={motion_name_base}_random"
            ])
        ]

        for phase_name, cmd in phases:
            exp_name = next(arg.split('=')[1] for arg in cmd if arg.startswith('experiment='))
            
            print(f"\n" + "="*80)
            print(f">>> 正在处理: {motion_name_base} | 阶段: {phase_name}")
            print(f">>> 命令: {' '.join(cmd)}")
            print("="*80 + "\n")
            
            try:
                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = "4"
                subprocess.run(cmd, cwd=BASE_DIR, check=True, env=my_env)
            except subprocess.CalledProcessError as e:
                print(f"!!! 阶段 {phase_name} 退出 (代号: {e.returncode})，尝试进行统计...")
            except KeyboardInterrupt:
                print("\n用户中断任务。")
                return

            # --- 统计逻辑更新 ---
            dump_dir = find_latest_dump_dir(exp_name)
            if dump_dir:
                s_count, f_count = get_rollout_counts(dump_dir)
                
                # 构建更直观的文件名
                if f_count > 0:
                    # 混合存储
                    status_tag = f"S_{s_count}_F_{f_count}"
                else:
                    # 纯成功存储
                    status_tag = f"SUCCESS_{s_count}"
                
                count_file = os.path.join(dump_dir, f"COUNT_{status_tag}.txt")
                with open(count_file, 'w') as f:
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Successful rollouts: {s_count}\n")
                    f.write(f"Failed rollouts: {f_count}\n")
                    f.write(f"Total: {s_count + f_count}\n")
                
                print(f">>> 统计完成: [{status_tag}] 详情已记录在 {os.path.basename(dump_dir)}")
            else:
                print(f"!!! 未能找到该任务的存储目录: {exp_name}")

if __name__ == "__main__":
    run_tests()
