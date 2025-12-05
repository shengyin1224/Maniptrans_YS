import os
import pickle
import xml.etree.ElementTree as ET

# ===== 重要：必须先导入 IsaacGym 相关模块，再导入 torch =====
from isaacgym import gymapi, gymtorch, gymutil
import logging

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_rotmat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

# 现在可以安全导入 torch 和其他依赖
import numpy as np
import torch
import trimesh
from termcolor import cprint
from tqdm import tqdm


def to_torch(x, device="cpu"):
    """Convert numpy array to torch tensor"""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device, dtype=torch.float32)


def pack_data(data, dexhand):
    """从原代码复制的数据打包函数"""
    packed_data = {}
    for k in data[0].keys():
        if k == "mano_joints":
            mano_joints = []
            for d in data:
                mano_joints.append(
                    torch.concat(
                        [
                            d[k][dexhand.to_hand(j_name)[0]]
                            for j_name in dexhand.body_names
                            if dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                )
            packed_data[k] = torch.stack(mano_joints).squeeze()
        elif k == "scene_objects":
            packed_data[k] = data[0][k]
        elif type(data[0][k]) == torch.Tensor:
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif type(data[0][k]) == np.ndarray:
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def load_object_point_clouds(scene_objects, num_points=1024, device="cpu"):
    """
    加载场景中所有物体的点云
    
    Args:
        scene_objects: 物体信息列表
        num_points: 每个物体采样的点云数量
        device: torch device
    
    Returns:
        object_points: List[torch.Tensor], 每个物体的点云 [num_points, 3]
    """
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
    """
    计算手部点与物体点云的接触
    
    Args:
        hand_points: [num_hand_points, 3] 手部关节点位置（世界坐标系）
        object_points_world: [num_object_points, 3] 物体点云位置（世界坐标系）
        threshold: 接触距离阈值（米）
    
    Returns:
        contacts: Dict，包含接触信息
            - has_contact: bool, 是否有接触
            - contact_hand_indices: List[int], 发生接触的手部点索引
            - contact_object_points: List[torch.Tensor], 对应的物体接触点位置
            - contact_distances: List[float], 接触距离
    """
    # 计算每个手部点到所有物体点的距离
    # hand_points: [num_hand_points, 3]
    # object_points_world: [num_object_points, 3]
    # distances: [num_hand_points, num_object_points]
    distances = torch.cdist(hand_points, object_points_world)  # [num_hand_points, num_obj_points]
    
    # 找到每个手部点的最近物体点
    min_distances, min_indices = torch.min(distances, dim=1)  # [num_hand_points]
    
    # 找出距离小于阈值的接触
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
    """
    将物体局部坐标系的点云转换到世界坐标系
    
    Args:
        object_points_local: [num_points, 3] 物体局部坐标系点云
        object_pose: [4, 4] 物体位姿变换矩阵
    
    Returns:
        object_points_world: [num_points, 3] 世界坐标系点云
    """
    # 转换为齐次坐标
    num_points = object_points_local.shape[0]
    ones = torch.ones(num_points, 1, device=object_points_local.device, dtype=object_points_local.dtype)
    points_homo = torch.cat([object_points_local, ones], dim=1)  # [num_points, 4]
    
    # 应用变换
    points_transformed = (object_pose @ points_homo.T).T  # [num_points, 4]
    
    return points_transformed[:, :3]


def analyze_motion_contacts(data_idx, dexhand, device="cuda:0", contact_threshold=0.005, num_object_points=1024):
    """
    分析一个motion的所有帧的手-物体接触
    
    Args:
        data_idx: 数据索引
        dexhand: 手部模型
        device: 计算设备
        contact_threshold: 接触距离阈值（米）
        num_object_points: 物体点云采样数量
    
    Returns:
        results: Dict，包含所有帧的接触信息
    """
    # 加载数据
    dataset_type = ManipDataFactory.dataset_type(data_idx)
    cprint(f"Loading {dataset_type} data: {data_idx}", "cyan")
    
    demo_d = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side="right",  # 可以根据需要修改
        device=device,
        mujoco2gym_transf=torch.eye(4, device=device),
        dexhand=dexhand,
        verbose=False,
    )
    
    demo_data = pack_data([demo_d[data_idx]], dexhand)
    num_frames = demo_data["mano_joints"].shape[0]
    
    cprint(f"Number of frames: {num_frames}", "green")
    
    # 设置坐标转换（从 mano2dexhand.py 复制）
    mujoco2gym_transf = np.eye(4)
    mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
        np.array([np.pi / 2, 0, 0])
    )
    
    # 根据数据集类型设置平移
    if dataset_type == "humoto":
        mujoco2gym_transf[:3, 3] = np.array([0, 0, 0])
    else:
        table_surface_z = 0.4 + 0.015  # table_pos.z + table_half_height
        mujoco2gym_transf[:3, 3] = np.array([0, 0, table_surface_z])
    
    mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=device, dtype=torch.float32)
    
    # 转换手部位置到世界坐标系
    wrist_pos = demo_data["wrist_pos"]  # [num_frames, 3]
    wrist_rot = demo_data["wrist_rot"]  # [num_frames, 3] (axis-angle)
    mano_joints = demo_data["mano_joints"].view(num_frames, -1, 3)  # [num_frames, num_joints, 3]
    
    # 应用坐标转换
    wrist_pos_world = (mujoco2gym_transf[:3, :3] @ wrist_pos.T).T + mujoco2gym_transf[:3, 3]
    mano_joints_flat = mano_joints.view(-1, 3)
    mano_joints_world = (mujoco2gym_transf[:3, :3] @ mano_joints_flat.T).T + mujoco2gym_transf[:3, 3]
    mano_joints_world = mano_joints_world.view(num_frames, -1, 3)
    
    # 组合手部所有点：手腕 + 手指关节
    hand_points_world = torch.cat([wrist_pos_world.unsqueeze(1), mano_joints_world], dim=1)  # [num_frames, 21, 3]
    num_hand_points = hand_points_world.shape[1]
    
    cprint(f"Number of hand points: {num_hand_points} (1 wrist + {num_hand_points-1} finger joints)", "green")
    
    # 加载物体信息
    scene_objects = demo_data["scene_objects"]
    num_objects = len(scene_objects)
    cprint(f"Number of objects: {num_objects}", "green")
    
    # 加载物体点云
    object_points_list = load_object_point_clouds(scene_objects, num_object_points, device)
    
    # 转换物体轨迹到世界坐标系
    processed_trajs = []
    for obj_info in scene_objects:
        traj = obj_info['trajectory']  # [num_frames, 4, 4]
        traj = mujoco2gym_transf.to(traj.device) @ traj
        processed_trajs.append(traj)
    
    # 对每一帧计算接触
    results = {
        'data_idx': data_idx,
        'dataset_type': dataset_type,
        'num_frames': num_frames,
        'num_objects': num_objects,
        'num_hand_points': num_hand_points,
        'contact_threshold': contact_threshold,
        'hand_point_names': ['wrist'] + [dexhand.to_hand(j_name)[0] for j_name in dexhand.body_names if dexhand.to_hand(j_name)[0] != "wrist"],
        'object_names': [obj_info.get('name', f'object_{i}') for i, obj_info in enumerate(scene_objects)],
        'frames': []
    }
    
    cprint(f"Computing contacts for {num_frames} frames...", "cyan")
    
    for frame_idx in tqdm(range(num_frames)):
        frame_result = {
            'frame_idx': frame_idx,
            'objects': []
        }
        
        hand_points_frame = hand_points_world[frame_idx]  # [num_hand_points, 3]
        
        # 对每个物体计算接触
        for obj_idx, (obj_points_local, obj_traj) in enumerate(zip(object_points_list, processed_trajs)):
            # 将物体点云转换到世界坐标系
            obj_pose = obj_traj[frame_idx]  # [4, 4]
            obj_points_world = transform_object_points(obj_points_local, obj_pose)
            
            # 计算接触
            contacts = compute_hand_object_contacts(hand_points_frame, obj_points_world, contact_threshold)
            
            obj_result = {
                'object_idx': obj_idx,
                'object_name': results['object_names'][obj_idx],
                'has_contact': contacts['has_contact'],
                'num_contacts': len(contacts['contact_hand_indices']),
                'contacts': []
            }
            
            # 记录每个接触点的详细信息
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
    
    cprint(f"\nData: {results['data_idx']} ({results['dataset_type']})", "white", attrs=['bold'])
    cprint(f"Total frames: {num_frames}", "white")
    cprint(f"Contact threshold: {results['contact_threshold']*1000:.1f} mm", "white")
    cprint(f"Number of hand points: {results['num_hand_points']}", "white")
    cprint(f"Number of objects: {num_objects}", "white")
    
    # 统计每个物体的接触
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
    
    # 统计哪些手部点最常接触
    hand_contact_counts = {i: 0 for i in range(results['num_hand_points'])}
    
    for frame in results['frames']:
        for obj_data in frame['objects']:
            for contact in obj_data['contacts']:
                hand_contact_counts[contact['hand_point_idx']] += 1
    
    cprint("\nMost frequently contacting hand points:", "yellow", attrs=['bold'])
    sorted_counts = sorted(hand_contact_counts.items(), key=lambda x: x[1], reverse=True)
    for hand_idx, count in sorted_counts[:10]:  # Top 10
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
    
    parser = argparse.ArgumentParser(description="Compute hand-object contacts")
    parser.add_argument("--data_idx", type=str, default="1906", help="Data index")
    parser.add_argument("--dexhand", type=str, default="inspire", help="Dexhand type")
    parser.add_argument("--side", type=str, default="right", help="Hand side (left/right)")
    parser.add_argument("--threshold", type=float, default=0.005, help="Contact distance threshold (meters)")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample on object surface")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--output_dir", type=str, default="data/contacts", help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    # 创建 dexhand 模型（用于获取手部关节名称）
    dexhand = DexHandFactory.create_hand(args.dexhand, args.side)
    
    # 分析接触
    results = analyze_motion_contacts(
        data_idx=args.data_idx,
        dexhand=dexhand,
        device=args.device,
        contact_threshold=args.threshold,
        num_object_points=args.num_points
    )
    
    # 打印摘要
    print_contact_summary(results)
    
    # 保存结果
    dataset_type = results['dataset_type']
    output_path = os.path.join(args.output_dir, dataset_type, f"contacts_{args.data_idx}.pkl")
    save_results(results, output_path)
    
    cprint("\nDone!", "green", attrs=['bold'])

