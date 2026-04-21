import os
import pickle
import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from .base import ManipData
from .decorators import register_manipdata

# 尝试导入官方 HumanModel utils
try:
    from .utils.load_humoto import load_one_humoto_sequence
    from .utils.rotation_helper import quaternion_to_matrix
    from .utils.np_torch_conversion import dict_to_torch
    from .human_model.human_model import HumanModelDifferentiable
except ImportError:
    # 本地调试用的 fallback
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "human_model"))
    from human_model.human_model import HumanModelDifferentiable

# === 1. 定义左右手的骨骼映射 ===
BONE_MAP_RH = {
    "wrist": "mixamorig:RightHand",
    "thumb_proximal": "mixamorig:RightHandThumb1",
    "thumb_intermediate": "mixamorig:RightHandThumb2",
    "thumb_distal": "mixamorig:RightHandThumb3",
    "thumb_tip": "mixamorig:RightHandThumb4",
    "index_proximal": "mixamorig:RightHandIndex1",
    "index_intermediate": "mixamorig:RightHandIndex2",
    "index_distal": "mixamorig:RightHandIndex3",
    "index_tip": "mixamorig:RightHandIndex4",
    "middle_proximal": "mixamorig:RightHandMiddle1",
    "middle_intermediate": "mixamorig:RightHandMiddle2",
    "middle_distal": "mixamorig:RightHandMiddle3",
    "middle_tip": "mixamorig:RightHandMiddle4",
    "ring_proximal": "mixamorig:RightHandRing1",
    "ring_intermediate": "mixamorig:RightHandRing2",
    "ring_distal": "mixamorig:RightHandRing3",
    "ring_tip": "mixamorig:RightHandRing4",
    "pinky_proximal": "mixamorig:RightHandPinky1",
    "pinky_intermediate": "mixamorig:RightHandPinky2",
    "pinky_distal": "mixamorig:RightHandPinky3",
    "pinky_tip": "mixamorig:RightHandPinky4",
}

# 左手映射
BONE_MAP_LH = {
    "wrist": "mixamorig:LeftHand",
    "thumb_proximal": "mixamorig:LeftHandThumb1",
    "thumb_intermediate": "mixamorig:LeftHandThumb2",
    "thumb_distal": "mixamorig:LeftHandThumb3",
    "thumb_tip": "mixamorig:LeftHandThumb4",
    "index_proximal": "mixamorig:LeftHandIndex1",
    "index_intermediate": "mixamorig:LeftHandIndex2",
    "index_distal": "mixamorig:LeftHandIndex3",
    "index_tip": "mixamorig:LeftHandIndex4",
    "middle_proximal": "mixamorig:LeftHandMiddle1",
    "middle_intermediate": "mixamorig:LeftHandMiddle2",
    "middle_distal": "mixamorig:LeftHandMiddle3",
    "middle_tip": "mixamorig:LeftHandMiddle4",
    "ring_proximal": "mixamorig:LeftHandRing1",
    "ring_intermediate": "mixamorig:LeftHandRing2",
    "ring_distal": "mixamorig:LeftHandRing3",
    "ring_tip": "mixamorig:LeftHandRing4",
    "pinky_proximal": "mixamorig:LeftHandPinky1",
    "pinky_intermediate": "mixamorig:LeftHandPinky2",
    "pinky_distal": "mixamorig:LeftHandPinky3",
    "pinky_tip": "mixamorig:LeftHandPinky4",
}

# === 2. 定义基类 (包含所有逻辑) ===

class HumotoDatasetBase(ManipData):
    def __init__(
        self,
        bone_map: dict,
        data_dir: str = "data/humoto", 
        split: str = "all",
        device="cuda:0",
        human_model_json: str = "main/dataset/human_model/human_model_take000_mixamorig_yup.json", 
        fix_orientation: bool = False, 
        fix_coordinate_system: bool = False,
        target_fps: int = 60,
        source_fps: int = 30,
        embodiment: str = None, 
        side: str = "right",
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            device=device,
            **kwargs,
        )
        
        self.bone_map = bone_map
        self.target_fps = target_fps
        self.source_fps = source_fps
        self.fix_orientation = fix_orientation
        self.fix_coordinate_system = fix_coordinate_system

        self.embodiment = embodiment
        self.side = side

        if not os.path.exists(human_model_json):
             raise FileNotFoundError(f"Human Model JSON not found at {human_model_json}")
             
        self.human_model = HumanModelDifferentiable(
            character_data_path=human_model_json, 
            device=self.device
        )
        
        self.data_pathes = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".pkl"):
                    self.data_pathes.append(os.path.join(root, file))
        self.data_pathes.sort()
        
        self.indices = {os.path.basename(p).replace(".pkl", ""): i for i, p in enumerate(self.data_pathes)}

        # 预计算修正矩阵
        self.global_fix_matrix = torch.eye(4, device=self.device)
        
        if self.fix_coordinate_system:
            r_coord = R.from_euler('x', 270, degrees=True).as_matrix()
            mat_coord = torch.eye(4, device=self.device)
            mat_coord[:3, :3] = torch.tensor(r_coord, dtype=torch.float32, device=self.device)
            self.global_fix_matrix = mat_coord @ self.global_fix_matrix

        if self.fix_orientation:
            r_ori = R.from_euler('z', 180, degrees=True).as_matrix()
            mat_ori = torch.eye(4, device=self.device)
            mat_ori[:3, :3] = torch.tensor(r_ori, dtype=torch.float32, device=self.device)
            self.global_fix_matrix = mat_ori @ self.global_fix_matrix

    def _interpolate_pose_params(self, pose_params_dict):
        """插值 30Hz -> 60Hz"""
        interp_dict = {}
        first_key = list(pose_params_dict.keys())[0]
        N = pose_params_dict[first_key].shape[0]
        
        if N < 2: return pose_params_dict

        original_times = np.linspace(0, N - 1, N)
        new_N = int(N * (self.target_fps / self.source_fps))
        target_times = np.linspace(0, N - 1, new_N)

        for bone_name, data in pose_params_dict.items():
            quat = data[:, :4]
            pos = data[:, 4:]

            xyzw_quats = quat[:, [1, 2, 3, 0]]
            rotations = R.from_quat(xyzw_quats)
            slerp = Slerp(original_times, rotations)
            new_rotations = slerp(target_times)
            new_quats_xyzw = new_rotations.as_quat()
            new_quats = new_quats_xyzw[:, [3, 0, 1, 2]]

            lerp = interp1d(original_times, pos, axis=0, kind='linear')
            new_pos = lerp(target_times)
            
            interp_dict[bone_name] = np.concatenate([new_quats, new_pos], axis=1).astype(np.float32)
            
        return interp_dict

    def _quat_pos_to_mat_tensor(self, quat, pos):
        T = quat.shape[0]
        # 确保输入是 Tensor 且在正确设备
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, device=self.device)
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, device=self.device)
            
        # 使用官方 utils 进行转换 (支持 [T, 4] -> [T, 3, 3])
        rot_mats = quaternion_to_matrix(quat) 
        
        mats = torch.eye(4, device=self.device)[None].repeat(T, 1, 1)
        mats[:, :3, :3] = rot_mats
        mats[:, :3, 3] = pos
        return mats

    def __getitem__(self, index):
        if isinstance(index, str):
             idx = self.indices.get(index, 0)
        else:
             idx = index
        
        pkl_path = self.data_pathes[idx]
        seq_folder_path = os.path.dirname(pkl_path)
        
        # 加载数据
        raw_data = load_one_humoto_sequence(
            data_path=seq_folder_path,
            include_text=False,
            y_up=True, 
            object_model=True,
            object_modality=['mesh'],
            pose_params=True
        )

        # === 1. 修改后的物体处理逻辑：加载所有物体 ===
        
        # 获取所有物体名称，不再过滤 table, tray 等
        all_obj_names = list(raw_data['object_pose_params'].keys())
        if not all_obj_names:
             pass

        scene_objects_info = []
        
        for obj_name in all_obj_names:
             # 1.1 处理每个物体的轨迹插值
             obj_pose_dict = {obj_name: raw_data['object_pose_params'][obj_name]}
             interp_obj_dict = self._interpolate_pose_params(obj_pose_dict)
             obj_data = interp_obj_dict[obj_name]
             
             # 1.2 转换为矩阵并应用全局修正
             traj_tensor = self._quat_pos_to_mat_tensor(obj_data[:, :4], obj_data[:, 4:])
             traj_tensor = torch.matmul(self.global_fix_matrix, traj_tensor)
             traj_tensor = traj_tensor[::self.skip]

             # 1.3 获取 Mesh 并采样点云
             if obj_name in raw_data['object_models']:
                 verts_np, faces_np = raw_data['object_models'][obj_name]['mesh']
                 mesh_torch = Meshes(
                     verts=torch.tensor(verts_np, dtype=torch.float32, device=self.device)[None],
                     faces=torch.tensor(faces_np, dtype=torch.float32, device=self.device)[None]
                 )
                 obj_verts = self.random_sampling_pc(mesh_torch)
             else:
                 obj_verts = torch.zeros((1000, 3), device=self.device)

             # 1.4 获取 URDF 路径
             obj_urdf_path = os.path.join(seq_folder_path, f"{obj_name}.urdf")
             
             # 1.5 存入列表
             scene_objects_info.append({
                 "name": obj_name,
                 "urdf": obj_urdf_path,
                 "trajectory": traj_tensor, # [T, 4, 4]
                 "verts": obj_verts,        # [1000, 3]
                 "is_dynamic": True 
             })
        
        # 2. 骨骼处理 (FK)
        # [已移除] 之前检查 root_trans 的代码块
        
        armature_params = raw_data['armature_pose_params']
        interp_armature_dict = self._interpolate_pose_params(armature_params)
        
        pose_params_matrix = {}
        for b_name, b_data in interp_armature_dict.items():
            mat = self._quat_pos_to_mat_tensor(b_data[:, :4], b_data[:, 4:])
            
            if b_name == "mixamorig:Hips":
                mat = torch.matmul(self.global_fix_matrix, mat)
                
            pose_params_matrix[b_name] = mat

        # [已移除] 之前计算临时 FK 检查高度差的代码块

        with torch.no_grad():
            joint_positions_dict = self.human_model.compute_joint_positions(pose_params_matrix)
            bone_transforms_dict = self.human_model.compute_bone_transforms(pose_params_matrix)

        # 3. 提取数据 (使用 self.bone_map)
        mano_joints = {}
        
        # Wrist
        wrist_name = self.bone_map["wrist"]
        wrist_pos = joint_positions_dict[wrist_name]

        # 计算世界坐标系下的完整变换矩阵: World_Transform = Deformation_Matrix @ Rest_Matrix
        # bone_transforms_dict 存储的是从 rest 到 posed 的变形矩阵 (Deformation Matrix)
        wrist_rest_mat = self.human_model.bone_rest_matrices[wrist_name]
        wrist_world_mat = torch.matmul(bone_transforms_dict[wrist_name], wrist_rest_mat[None])
        wrist_rot = wrist_world_mat[:, :3, :3]

        if self.side == 'right':
            # === 右手 (RH) 完美方案 ===
            # 测试结果误差: ~20度 (Correct)
            # 逻辑: Mixamo Y->X, X->Y, Z->-Z (Fix Flip)
            align_matrix = torch.tensor([
                [ 0.0,  1.0,  0.0],
                [ 1.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0]
            ], dtype=torch.float32, device=self.device)
        else:
            # === 左手 (LH) 推导方案 ===
            # 逻辑: 基于 RH 方案绕骨骼轴(New X)旋转 180 度以适配镜像对称
            # 变换: Mixamo Y->X, X->-Y, Z->Z
            align_matrix = torch.tensor([
                [ 0.0, -1.0,  0.0],
                [ 1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0]
            ], dtype=torch.float32, device=self.device)

        # 必须是右乘 (Right Multiplication)！
        wrist_rot = torch.matmul(wrist_rot, align_matrix)

        # Fingers
        for mano_k, humoto_k in self.bone_map.items():
            if mano_k == "wrist": continue
            if humoto_k in joint_positions_dict:
                mano_joints[mano_k] = joint_positions_dict[humoto_k]
            else:
                mano_joints[mano_k] = torch.zeros_like(wrist_pos)

        # 4. 构造返回字典 (统一下采样到 60fps)
        data = {
            "data_path": pkl_path,
            # 兼容字段 (指向第一个物体)
            "obj_id": scene_objects_info[0]["name"] if scene_objects_info else "none",
            "obj_verts": scene_objects_info[0]["verts"] if scene_objects_info else torch.zeros((1000, 3), device=self.device),
            "obj_urdf_path": scene_objects_info[0]["urdf"] if scene_objects_info else "",
            "obj_trajectory": scene_objects_info[0]["trajectory"] if scene_objects_info else torch.eye(4, device=self.device)[None],
            # 新增字段：包含所有物体
            "scene_objects": scene_objects_info,
            "wrist_rot9": wrist_rot[::self.skip],
            "wrist_pos": wrist_pos[::self.skip],
            "wrist_rot": wrist_rot[::self.skip],
            "mano_joints": {k: v[::self.skip] for k, v in mano_joints.items()},
        }

        # 加载 Retargeting 数据
        if self.embodiment is not None:
            # 1. 构造路径
            hand_suffix = "_rh" if self.side == "right" else "_lh"
            dexhand_name = f"{self.embodiment}{hand_suffix}"
            
            filename = os.path.basename(pkl_path)
            retarget_path = os.path.join(
                "data/retargeting/Humoto", 
                f"mano2{dexhand_name}", 
                filename
            )
            
            # 2. 直接调用基类方法
            self.load_retargeted_data(data, retarget_path)

        # process_data 传入第一个物体的 verts
        self.process_data(data, idx, data["obj_verts"])
        return data

# === 3. 注册子类 ===

@register_manipdata("humoto_rh")
class HumotoDatasetRH(HumotoDatasetBase):
    def __init__(self, **kwargs):
        # 初始化时传入右手的骨骼映射
        super().__init__(bone_map=BONE_MAP_RH, side='right', **kwargs)

@register_manipdata("humoto_lh")
class HumotoDatasetLH(HumotoDatasetBase):
    def __init__(self, **kwargs):
        # 初始化时传入左手的骨骼映射
        super().__init__(bone_map=BONE_MAP_LH, side='left',**kwargs)
