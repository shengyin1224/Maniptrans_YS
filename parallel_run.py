import os
import subprocess
from multiprocessing import Pool, Manager
import queue

# 配置参数
DATA_DIR = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data/humoto"
BASE_DIR = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS"
# 用户提供的路径可能是 /home/zxlei/data/humanoid/Maniptrans_YS/data/
# 但在当前环境下路径为 /home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data/
# 这里使用现有的路径，如果需要可以修改
CONTACT_BASE_DIR = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/data"
NUM_GPUS = 9
GPU_IDS = range(3, 12)

def run_task(task, gpu_id):
    data_name, side = task
    
    contact_data_path = os.path.join(CONTACT_BASE_DIR, f"contacts_{side}", "humoto", f"contacts_{data_name}.pkl")
    
    # 构建命令模版
    # 使用分号连接 stage 1 和 stage 2，确保 stage 1 运行完后（即使崩溃）继续运行 stage 2
    cmd_stage1 = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python main/dataset/mano2dexhand_segmented.py "
        f"--data_idx {data_name} --side {side} --dexhand inspire --iter 15000 --headless "
        f"--contact_data {contact_data_path} --draw_all_lines 0 --stage 1"
    )
    
    cmd_stage2 = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python main/dataset/mano2dexhand_segmented.py "
        f"--data_idx {data_name} --side {side} --dexhand inspire --iter 15000 --headless "
        f"--contact_data {contact_data_path} --draw_all_lines 0 --stage 2"
    )
    
    full_cmd = f"{cmd_stage1}; {cmd_stage2}"
    
    print(f"[GPU {gpu_id}] Starting: {data_name} ({side})")
    
    try:
        # 在项目根目录下执行
        process = subprocess.run(full_cmd, shell=True, cwd=BASE_DIR)
        print(f"[GPU {gpu_id}] Finished: {data_name} ({side}) with exit code {process.returncode}")
    except Exception as e:
        print(f"[GPU {gpu_id}] Exception occurred for {data_name} ({side}): {e}")

def worker(task_queue, gpu_id):
    while True:
        try:
            # 从队列中获取任务
            task = task_queue.get_nowait()
        except queue.Empty:
            break
        
        run_task(task, gpu_id)
        task_queue.task_done()

def main():
    # 获取所有数据名称
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    data_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    sides = ["left", "right"]
    tasks = []
    for data_name in data_names:
        for side in sides:
            tasks.append((data_name, side))
    
    print(f"Total tasks to process: {len(tasks)}")
    print(f"Using GPUs: {GPU_IDS}")

    # 使用 Manager 队列实现多进程任务分发
    manager = Manager()
    task_queue = manager.Queue()
    for task in tasks:
        task_queue.put(task)
    
    # 创建 9 个 worker 进程，每个 worker 绑定一个 GPU ID
    with Pool(processes=NUM_GPUS) as pool:
        pool.starmap(worker, [(task_queue, gpu_id) for gpu_id in GPU_IDS])
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()

