"""
检查 pickle 文件格式
"""
import pickle
import sys
import numpy as np
import torch
from termcolor import cprint


def check_pickle_file(pickle_path):
    """检查 pickle 文件内容和格式"""
    cprint(f"\n{'='*80}", "cyan")
    cprint(f"Checking: {pickle_path}", "cyan", attrs=['bold'])
    cprint(f"{'='*80}\n", "cyan")
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        cprint(f"✗ Failed to load pickle file: {e}", "red")
        return False
    
    cprint("✓ Successfully loaded pickle file", "green")
    
    # 检查数据类型
    if isinstance(data, dict):
        cprint(f"✓ Data is a dictionary with {len(data)} keys", "green")
    else:
        cprint(f"⚠ Data is not a dictionary, type: {type(data)}", "yellow")
    
    # 列出所有键
    if isinstance(data, dict):
        cprint("\nAvailable keys:", "yellow")
        for key in data.keys():
            value = data[key]
            if isinstance(value, (np.ndarray, torch.Tensor)):
                cprint(f"  • {key:30s} : {type(value).__name__:15s} shape={getattr(value, 'shape', 'N/A')}", "white")
            elif isinstance(value, list):
                cprint(f"  • {key:30s} : list, length={len(value)}", "white")
            elif isinstance(value, dict):
                cprint(f"  • {key:30s} : dict with {len(value)} keys", "white")
            else:
                cprint(f"  • {key:30s} : {type(value).__name__}", "white")
    
    # 检查必需字段
    cprint("\nChecking required fields:", "yellow")
    
    required_fields = ['wrist_pos', 'wrist_rot', 'mano_joints', 'scene_objects']
    all_present = True
    
    for field in required_fields:
        if field in data:
            value = data[field]
            if isinstance(value, (np.ndarray, torch.Tensor)):
                cprint(f"  ✓ {field:20s} : shape={value.shape}", "green")
            elif isinstance(value, list):
                cprint(f"  ✓ {field:20s} : list, length={len(value)}", "green")
                if field == 'scene_objects' and len(value) > 0:
                    cprint(f"    First object keys: {list(value[0].keys())}", "white")
            elif isinstance(value, dict):
                cprint(f"  ✓ {field:20s} : dict with {len(value)} keys", "green")
                if field == 'mano_joints':
                    cprint(f"    Joint names: {list(value.keys())[:5]}...", "white")
            else:
                cprint(f"  ✓ {field:20s} : {type(value).__name__}", "green")
        else:
            cprint(f"  ✗ {field:20s} : MISSING", "red")
            all_present = False
    
    # 检查 scene_objects 详情
    if 'scene_objects' in data and isinstance(data['scene_objects'], list) and len(data['scene_objects']) > 0:
        cprint("\nScene objects details:", "yellow")
        for i, obj in enumerate(data['scene_objects']):
            if isinstance(obj, dict):
                obj_name = obj.get('name', f'object_{i}')
                has_urdf = 'urdf' in obj
                has_traj = 'trajectory' in obj
                traj_shape = obj['trajectory'].shape if has_traj and hasattr(obj['trajectory'], 'shape') else 'N/A'
                
                cprint(f"  Object {i} ({obj_name}):", "white")
                cprint(f"    URDF: {'✓' if has_urdf else '✗'} {obj.get('urdf', 'N/A')}", "white")
                cprint(f"    Trajectory: {'✓' if has_traj else '✗'} shape={traj_shape}", "white")
    
    # 推断帧数
    cprint("\nInferred metadata:", "yellow")
    if 'wrist_pos' in data and hasattr(data['wrist_pos'], 'shape'):
        num_frames = data['wrist_pos'].shape[0]
        cprint(f"  Number of frames: {num_frames}", "white")
    
    if 'scene_objects' in data and isinstance(data['scene_objects'], list):
        num_objects = len(data['scene_objects'])
        cprint(f"  Number of objects: {num_objects}", "white")
    
    # 总结
    cprint("\n" + "="*80, "cyan")
    if all_present:
        cprint("✓ This file is COMPATIBLE with compute_contacts_standalone.py", "green", attrs=['bold'])
    else:
        cprint("✗ This file is MISSING required fields", "red", attrs=['bold'])
    cprint("="*80 + "\n", "cyan")
    
    return all_present


if __name__ == "__main__":
    if len(sys.argv) < 2:
        cprint("Usage: python check_pickle_format.py <pickle_file>", "yellow")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    check_pickle_file(pickle_path)

