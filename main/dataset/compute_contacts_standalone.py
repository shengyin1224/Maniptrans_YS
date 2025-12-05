"""
独立的手-物体接触计算工具（不需要完整的 IsaacGym 环境）
只需要数据加载功能
"""
import os
import sys
import pickle
import xml.etree.ElementTree as ET

# 先导入不依赖 torch 的模块
import numpy as np
from termcolor import cprint
from tqdm import tqdm

# 延迟导入 torch 和其他依赖
import torch
import trimesh


def aa_to_rotmat(axis_angle):
    """轴角转旋转矩阵（从 transform.py 复制）"""
    if isinstance(axis_angle, np.ndarray):
        axis_angle = torch.from_numpy(axis_angle).float()
    
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one_minus_cos = 1.0 - cos
    
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    
    rot_mat = torch.zeros((*axis_angle.shape[:-1], 3, 3), dtype=axis_angle.dtype, device=axis_angle.device)
    
    rot_mat[..., 0, 0] = cos + x * x * one_minus_cos
    rot_mat[..., 0, 1] = x * y * one_minus_cos - z * sin
    rot_mat[..., 0, 2] = x * z * one_minus_cos + y * sin
    
    rot_mat[..., 1, 0] = y * x * one_minus_cos + z * sin
    rot_mat[..., 1, 1] = cos + y * y * one_minus_cos
    rot_mat[..., 1, 2] = y * z * one_minus_cos - x * sin
    
    rot_mat[..., 2, 0] = z * x * one_minus_cos - y * sin
    rot_mat[..., 2, 1] = z * y * one_minus_cos + x * sin
    rot_mat[..., 2, 2] = cos + z * z * one_minus_cos
    
    return rot_mat


def to_torch(x, device="cpu"):
    """Convert numpy array to torch tensor"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device, dtype=torch.float32)


def load_pickle_data(pickle_path):
    """直接加载 pickle 数据文件"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_hand_joint_names():
    """手部关节点名称（MANO 标准）"""
    return [
        'wrist',
        'index_proximal', 'index_intermediate', 'index_distal', 'index_tip',
        'middle_proximal', 'middle_intermediate', 'middle_distal', 'middle_tip',
        'ring_proximal', 'ring_intermediate', 'ring_distal', 'ring_tip',
        'pinky_proximal', 'pinky_intermediate', 'pinky_distal', 'pinky_tip',
        'thumb_proximal', 'thumb_intermediate', 'thumb_distal', 'thumb_tip'
    ]


def load_object_point_clouds(scene_objects, num_points=1024, device="cpu"):
    """加载场景中所有物体的点云"""
    object_points_list = []
    
    for obj_info in scene_objects:
        urdf_path = obj_info['urdf']
        urdf_dir = os.path.dirname(urdf_path)
        
        # 解析 URDF 获取 mesh 文件和缩放比例
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        mesh_elem = root.find(".//mesh")
        
        if mesh_elem is None:
            cprint(f"Warning: No mesh found in {urdf_path}, using empty point cloud", "yellow")
            object_points_list.append(torch.zeros(num_points, 3, device=device))
            continue
        
        # 获取 mesh 文件路径
        mesh_filename = mesh_elem.attrib.get("filename", "")
        if mesh_filename.startswith("package://"):
            mesh_filename = mesh_filename.replace("package://", "")
        
        mesh_path = os.path.join(urdf_dir, mesh_filename)
        
        # 获取缩放比例
        scale_str = mesh_elem.attrib.get("scale", "1 1 1")
        scale = float(scale_str.strip().split()[0])
        
        # 加载 mesh
        if not os.path.exists(mesh_path):
            cprint(f"Warning: Mesh file not found: {mesh_path}, using empty point cloud", "yellow")
            object_points_list.append(torch.zeros(num_points, 3, device=device))
            continue
        
        try:
            mesh_obj = trimesh.load(mesh_path, force='mesh')
            mesh_obj.apply_scale(scale)
            
            # 采样点云（相对于物体中心）
            center = np.mean(mesh_obj.vertices, 0)
            object_points, _ = trimesh.sample.sample_surface_even(mesh_obj, count=num_points, seed=2024)
            object_points = to_torch(object_points - center, device=device)
            
            # 如果采样点不足，重复填充
            while object_points.shape[0] < num_points:
                object_points = torch.cat(
                    [object_points, object_points[:num_points - object_points.shape[0]]], 
                    dim=0
                )
            
            object_points_list.append(object_points)
            
        except Exception as e:
            cprint(f"Error loading mesh {mesh_path}: {e}, using empty point cloud", "red")
            object_points_list.append(torch.zeros(num_points, 3, device=device))
    
    return object_points_list


def compute_hand_object_contacts(hand_points, object_points_world, threshold=0.005):
    """计算手部点与物体点云的接触"""
    distances = torch.cdist(hand_points, object_points_world)
    min_distances, min_indices = torch.min(distances, dim=1)
    
    contact_mask = min_distances < threshold
    contact_hand_indices = torch.where(contact_mask)[0].cpu().tolist()
    
    contacts = {
        'has_contact': len(contact_hand_indices) > 0,
        'contact_hand_indices': contact_hand_indices,
        'contact_object_points': [object_points_world[min_indices[i]].cpu().numpy() for i in contact_hand_indices],
        'contact_distances': [min_distances[i].item() for i in contact_hand_indices],
        'contact_hand_points': [hand_points[i].cpu().numpy() for i in contact_hand_indices],
    }
    
    return contacts


def transform_object_points(object_points_local, object_pose):
    """将物体局部坐标系的点云转换到世界坐标系"""
    num_points = object_points_local.shape[0]
    ones = torch.ones(num_points, 1, device=object_points_local.device, dtype=object_points_local.dtype)
    points_homo = torch.cat([object_points_local, ones], dim=1)
    points_transformed = (object_pose @ points_homo.T).T
    return points_transformed[:, :3]


def analyze_motion_contacts_from_pickle(pickle_path, contact_threshold=0.005, num_object_points=1024, device="cpu"):
    """
    直接从 pickle 文件分析接触
    
    Args:
        pickle_path: 数据 pickle 文件路径
        contact_threshold: 接触距离阈值（米）
        num_object_points: 物体点云采样数量
        device: 计算设备
    
    Returns:
        results: Dict，包含所有帧的接触信息
    """
    cprint(f"Loading data from: {pickle_path}", "cyan")
    data = load_pickle_data(pickle_path)
    
    # 提取手部数据
    wrist_pos = to_torch(data.get('wrist_pos', data.get('hand_pose', np.zeros((100, 3)))), device=device)
    wrist_rot = to_torch(data.get('wrist_rot', data.get('hand_rot', np.zeros((100, 3)))), device=device)
    
    # 处理 mano_joints
    if 'mano_joints' in data:
        mano_joints_data = data['mano_joints']
        if isinstance(mano_joints_data, dict):
            # 字典格式：按关节名称组织
            hand_joint_names = get_hand_joint_names()[1:]  # 排除 wrist
            mano_joints = []
            for joint_name in hand_joint_names:
                if joint_name in mano_joints_data:
                    mano_joints.append(to_torch(mano_joints_data[joint_name], device=device))
            if len(mano_joints) > 0:
                mano_joints = torch.stack(mano_joints, dim=1)
            else:
                cprint("Warning: No joint data found in dict format", "yellow")
                num_frames = wrist_pos.shape[0]
                mano_joints = torch.zeros((num_frames, 20, 3), device=device)
        else:
            # 数组格式
            mano_joints = to_torch(mano_joints_data, device=device)
            if mano_joints.dim() == 2:
                mano_joints = mano_joints.view(-1, 20, 3)
    else:
        cprint("Warning: No mano_joints found, using zeros", "yellow")
        num_frames = wrist_pos.shape[0]
        mano_joints = torch.zeros((num_frames, 20, 3), device=device)
    
    num_frames = wrist_pos.shape[0]
    cprint(f"Number of frames: {num_frames}", "green")
    
    # 坐标转换
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(torch.tensor([0, 0, -np.pi / 2])) @ aa_to_rotmat(
        torch.tensor([np.pi / 2, 0, 0])
    )
    table_surface_z = 0.4 + 0.015
    mujoco2gym_transf[:3, 3] = np.array([0, 0, table_surface_z])
    mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=device, dtype=torch.float32)
    
    # 转换手部位置
    wrist_pos_world = (mujoco2gym_transf[:3, :3] @ wrist_pos.T).T + mujoco2gym_transf[:3, 3]
    mano_joints_flat = mano_joints.view(-1, 3)
    mano_joints_world = (mujoco2gym_transf[:3, :3] @ mano_joints_flat.T).T + mujoco2gym_transf[:3, 3]
    mano_joints_world = mano_joints_world.view(num_frames, -1, 3)
    
    # 组合手部所有点
    hand_points_world = torch.cat([wrist_pos_world.unsqueeze(1), mano_joints_world], dim=1)
    num_hand_points = hand_points_world.shape[1]
    
    cprint(f"Number of hand points: {num_hand_points}", "green")
    
    # 加载物体信息
    scene_objects = data.get('scene_objects', [])
    num_objects = len(scene_objects)
    cprint(f"Number of objects: {num_objects}", "green")
    
    if num_objects == 0:
        cprint("Warning: No scene objects found!", "yellow")
        return None
    
    # 加载物体点云
    object_points_list = load_object_point_clouds(scene_objects, num_object_points, device)
    
    # 转换物体轨迹
    processed_trajs = []
    for obj_info in scene_objects:
        traj = to_torch(obj_info['trajectory'], device=device)
        if traj.dim() == 2:
            traj = traj.view(-1, 4, 4)
        traj = mujoco2gym_transf @ traj
        processed_trajs.append(traj)
    
    # 结果结构
    hand_point_names = get_hand_joint_names()
    results = {
        'pickle_path': pickle_path,
        'num_frames': num_frames,
        'num_objects': num_objects,
        'num_hand_points': num_hand_points,
        'contact_threshold': contact_threshold,
        'hand_point_names': hand_point_names,
        'object_names': [obj_info.get('name', f'object_{i}') for i, obj_info in enumerate(scene_objects)],
        'frames': []
    }
    
    cprint(f"Computing contacts for {num_frames} frames...", "cyan")
    
    for frame_idx in tqdm(range(num_frames)):
        frame_result = {
            'frame_idx': frame_idx,
            'objects': []
        }
        
        hand_points_frame = hand_points_world[frame_idx]
        
        for obj_idx, (obj_points_local, obj_traj) in enumerate(zip(object_points_list, processed_trajs)):
            obj_pose = obj_traj[frame_idx]
            obj_points_world = transform_object_points(obj_points_local, obj_pose)
            
            contacts = compute_hand_object_contacts(hand_points_frame, obj_points_world, contact_threshold)
            
            obj_result = {
                'object_idx': obj_idx,
                'object_name': results['object_names'][obj_idx],
                'has_contact': contacts['has_contact'],
                'num_contacts': len(contacts['contact_hand_indices']),
                'contacts': []
            }
            
            for i, hand_idx in enumerate(contacts['contact_hand_indices']):
                contact_info = {
                    'hand_point_idx': hand_idx,
                    'hand_point_name': results['hand_point_names'][hand_idx],
                    'hand_point_pos': contacts['contact_hand_points'][i],
                    'object_contact_pos': contacts['contact_object_points'][i],
                    'distance': contacts['contact_distances'][i]
                }
                obj_result['contacts'].append(contact_info)
            
            frame_result['objects'].append(obj_result)
        
        results['frames'].append(frame_result)
    
    return results


def print_contact_summary(results):
    """打印接触统计摘要"""
    num_frames = results['num_frames']
    num_objects = results['num_objects']
    
    cprint("\n" + "="*80, "cyan")
    cprint("CONTACT ANALYSIS SUMMARY", "cyan", attrs=['bold'])
    cprint("="*80, "cyan")
    
    cprint(f"\nData: {os.path.basename(results['pickle_path'])}", "white", attrs=['bold'])
    cprint(f"Total frames: {num_frames}", "white")
    cprint(f"Contact threshold: {results['contact_threshold']*1000:.1f} mm", "white")
    cprint(f"Number of hand points: {results['num_hand_points']}", "white")
    cprint(f"Number of objects: {num_objects}", "white")
    
    for obj_idx in range(num_objects):
        obj_name = results['object_names'][obj_idx]
        frames_with_contact = 0
        total_contacts = 0
        
        for frame in results['frames']:
            obj_data = frame['objects'][obj_idx]
            if obj_data['has_contact']:
                frames_with_contact += 1
                total_contacts += obj_data['num_contacts']
        
        cprint(f"\n{obj_name}:", "yellow", attrs=['bold'])
        cprint(f"  Frames with contact: {frames_with_contact}/{num_frames} ({100*frames_with_contact/num_frames:.1f}%)", "white")
        cprint(f"  Total contact points: {total_contacts}", "white")
        if frames_with_contact > 0:
            cprint(f"  Avg contacts per frame: {total_contacts/frames_with_contact:.2f}", "white")
    
    hand_contact_counts = {i: 0 for i in range(results['num_hand_points'])}
    
    for frame in results['frames']:
        for obj_data in frame['objects']:
            for contact in obj_data['contacts']:
                hand_contact_counts[contact['hand_point_idx']] += 1
    
    cprint("\nMost frequently contacting hand points:", "yellow", attrs=['bold'])
    sorted_counts = sorted(hand_contact_counts.items(), key=lambda x: x[1], reverse=True)
    for hand_idx, count in sorted_counts[:10]:
        if count > 0:
            hand_name = results['hand_point_names'][hand_idx]
            cprint(f"  {hand_name:30s}: {count:4d} contacts ({100*count/(num_frames*num_objects):.1f}%)", "white")
    
    cprint("\n" + "="*80 + "\n", "cyan")


def save_results(results, output_path):
    """保存结果到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    cprint(f"Results saved to: {output_path}", "green")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute hand-object contacts from pickle file")
    parser.add_argument("pickle_path", type=str, help="Path to input pickle file")
    parser.add_argument("--threshold", type=float, default=0.005, help="Contact distance threshold (meters)")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample on object surface")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output pickle file path")
    
    args = parser.parse_args()
    
    # 分析接触
    results = analyze_motion_contacts_from_pickle(
        pickle_path=args.pickle_path,
        contact_threshold=args.threshold,
        num_object_points=args.num_points,
        device=args.device
    )
    
    if results is None:
        cprint("Failed to analyze contacts!", "red")
        sys.exit(1)
    
    # 打印摘要
    print_contact_summary(results)
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        basename = os.path.splitext(os.path.basename(args.pickle_path))[0]
        output_path = f"data/contacts/contacts_{basename}.pkl"
    
    save_results(results, output_path)
    
    cprint("\nDone!", "green", attrs=['bold'])

