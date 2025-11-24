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
        # [修改点]: 给 data_dir 设置默认值！
        # 请将其修改为你存放 humoto 数据集的真实根目录
        data_dir: str = "data/humoto", 
        split: str = "all",
        device="cuda:0",
        # 请确保 JSON 路径也正确，如果变了也要改
        human_model_json: str = "/home/ubuntu/DATA3/shengyin/ManipTrans/main/dataset/human_model/human_model_up_bone_yup.json", 
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

        # 1. 物体处理
        obj_names = list(raw_data['object_pose_params'].keys())
        valid_objs = [o for o in obj_names if o not in ['table', 'tray', 'floor', 'ground', 'room', 'ground_plane']]
        obj_id = valid_objs[0] if valid_objs else obj_names[0]

        obj_pose_dict = {obj_id: raw_data['object_pose_params'][obj_id]}
        interp_obj_dict = self._interpolate_pose_params(obj_pose_dict)
        
        obj_data = interp_obj_dict[obj_id]
        obj_trajectory = self._quat_pos_to_mat_tensor(obj_data[:, :4], obj_data[:, 4:])
        
        # 应用全局修正
        obj_trajectory = torch.matmul(self.global_fix_matrix, obj_trajectory)

        # Mesh 采样
        verts_np, faces_np = raw_data['object_models'][obj_id]['mesh']
        mesh_torch = Meshes(
            verts=torch.tensor(verts_np, dtype=torch.float32, device=self.device)[None],
            faces=torch.tensor(faces_np, dtype=torch.float32, device=self.device)[None]
        )
        obj_verts = self.random_sampling_pc(mesh_torch)
        
        obj_urdf_path = os.path.join(seq_folder_path, f"{obj_id}.urdf")

        # 2. 骨骼处理 (FK)
        armature_params = raw_data['armature_pose_params']
        interp_armature_dict = self._interpolate_pose_params(armature_params)
        
        pose_params_matrix = {}
        for b_name, b_data in interp_armature_dict.items():
            mat = self._quat_pos_to_mat_tensor(b_data[:, :4], b_data[:, 4:])
            
            if b_name == "mixamorig:Hips":
                mat = torch.matmul(self.global_fix_matrix, mat)
                
            pose_params_matrix[b_name] = mat

        with torch.no_grad():
            joint_positions_dict = self.human_model.compute_joint_positions(pose_params_matrix)
            bone_transforms_dict = self.human_model.compute_bone_transforms(pose_params_matrix)

        # 3. 提取数据 (使用 self.bone_map)
        mano_joints = {}
        
        # Wrist
        wrist_name = self.bone_map["wrist"]
        wrist_pos = joint_positions_dict[wrist_name]
        wrist_rot = bone_transforms_dict[wrist_name][:, :3, :3]

        # Fingers
        for mano_k, humoto_k in self.bone_map.items():
            if mano_k == "wrist": continue
            if humoto_k in joint_positions_dict:
                mano_joints[mano_k] = joint_positions_dict[humoto_k]
            else:
                # 如果缺少关节，补0防止报错 (通常不会发生)
                mano_joints[mano_k] = torch.zeros_like(wrist_pos)

        data = {
            "data_path": pkl_path,
            "obj_id": obj_id,
            "obj_verts": obj_verts,
            "obj_urdf_path": obj_urdf_path,
            "obj_trajectory": obj_trajectory,
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        # === [修改] 使用基类的高级方法加载 Retargeting 数据 ===
        # 这一步会替换掉我们刚才手动写的 pickle.load 和 velocity 补全
        
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
            
            # 2. 直接调用基类方法！
            # 这个方法会自动加载 .pkl，并帮你算出 velocity (opt_dof_velocity)
            self.load_retargeted_data(data, retarget_path)
        # =====================================================

        self.process_data(data, idx, obj_verts)
        return data

# === 3. 注册子类 ===

@register_manipdata("humoto_rh")
class HumotoDatasetRH(HumotoDatasetBase):
    def __init__(self, **kwargs):
        # 初始化时传入右手的骨骼映射
        super().__init__(bone_map=BONE_MAP_RH, **kwargs)

@register_manipdata("humoto_lh")
class HumotoDatasetLH(HumotoDatasetBase):
    def __init__(self, **kwargs):
        # 初始化时传入左手的骨骼映射
        super().__init__(bone_map=BONE_MAP_LH, **kwargs)