import math
import os
import pickle
import imageio
import xml.etree.ElementTree as ET
from isaacgym import gymapi, gymtorch, gymutil
import logging

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def pack_data(data, dexhand):
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
        # 特殊处理 scene_objects 列表
        elif k == "scene_objects":
            packed_data[k] = data[0][k]
        elif type(data[0][k]) == torch.Tensor:
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif type(data[0][k]) == np.ndarray:
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class Mano2Dexhand:
    def __init__(self, args, dexhand, scene_objects, dataset_type="oakink2", contact_data=None):
        """
        args: 参数
        dexhand: 机器人手模型
        scene_objects: 包含所有物体信息的列表
        dataset_type: 数据集类型，用于判断坐标转换逻辑
        contact_data: 可选的接触数据字典（从 compute_hand_object_contacts.py 生成的 pkl 文件加载）
        """
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand
        self.scene_objects = scene_objects
        self.contact_data = contact_data 
        # 可选的接触奖励项配置
        self.enable_contact_reward = bool(getattr(args, "enable_contact_reward", 0))
        self.contact_reward_scale = float(getattr(args, "contact_reward_scale", 0.1))
        self.contact_reward_match_scale = float(getattr(args, "contact_reward_match_scale", 0.2))
        self.contact_reward_sigma = float(getattr(args, "contact_reward_sigma", 0.01))
        # 手腕收敛门槛，未满足则只优化手腕，不优化其余关节
        self.wrist_pos_tol = float(getattr(args, "wrist_pos_tol", 0.01))   # meters
        wrist_rot_tol_deg = float(getattr(args, "wrist_rot_tol", 5.0))     # degrees
        self.wrist_rot_tol = math.radians(wrist_rot_tol_deg)               # radians
        self.draw_all_lines = getattr(args, "draw_all_lines", 0) == 1
        self._object_points_local_cache = None

        # 记录手部关节在 target_mano_joints 中的顺序，便于后续取特定关节
        self.hand_joint_order = [
            self.dexhand.to_hand(j)[0] for j in self.dexhand.body_names if self.dexhand.to_hand(j)[0] != "wrist"
        ]
        self._hand_joint_indices = {}
        for _idx, _name in enumerate(self.hand_joint_order):
            # 保留首次出现的索引，避免重复映射覆盖
            if _name not in self._hand_joint_indices:
                self._hand_joint_indices[_name] = _idx

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        self.graphics_device_id = -1 if self.headless else args.graphics_device_id

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu

        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        # In headless mode Isaac Gym should not create a graphics context.
        self.sim = self.gym.create_sim(
            args.compute_device_id, self.graphics_device_id, args.physics_engine, self.sim_params
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        # === 加载机器人 Asset ===
        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file)).read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        # 配置机器人关节参数
        dexhand_dof_stiffness = torch.tensor(
            [10] * self.dexhand.n_dofs, dtype=torch.float, device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [1] * self.dexhand.n_dofs, dtype=torch.float, device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 0.0001
            element.rolling_friction = 0.0001
            element.torsion_friction = 0.0001
        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_rh_props_asset)

        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]
            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)
        
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        default_dof_state["pos"][8] = 0.8
        default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        # === 配置坐标转换 ===
        table_width_offset = 0.2
        mujoco2gym_transf = np.eye(4)
        # 1. 旋转修正 (Y-up to Z-up)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        self._table_surface_z = table_pos.z + table_half_height

        # === [核心修改] 根据数据集类型决定平移策略 ===
        if dataset_type == "humoto":
            print("[INFO] Detected Humoto dataset: Using Scheme 1 (Rotation fix only, No Z-offset).")
            # 方案一：仅旋转，不平移，保留原始高度
            mujoco2gym_transf[:3, 3] = np.array([0, 0, 0])
        else:
            # 原有逻辑：旋转 + 平移到桌面高度
            mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        # ===========================================

        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # 预处理接触点，用于奖励与可视化
        self._contact_reward_frames = None
        # === 预计算接触点数量（用于创建 spheres） ===
        if self.contact_data is not None:
            max_contacts = 0
            # 仅考虑“每帧 num_contacts 最大的那个物体”的接触点数（已经是点对数）
            for frame in self.contact_data['frames']:
                if len(frame['objects']) == 0:
                    continue
                frame_max_contacts = max(obj.get('num_contacts', 0) for obj in frame['objects'])
                max_contacts = max(max_contacts, frame_max_contacts)
            # 为了避免创建过多可视化小球导致 PhysX 内存/对数超限，做可视化上限
            MAX_CONTACT_VIS = 5  # 如需更多可调高，但可能再次触顶
            if max_contacts > MAX_CONTACT_VIS:
                cprint(
                    f"[CONTACT VIS][WARN] Capping visualized contacts from {max_contacts} to {MAX_CONTACT_VIS} to avoid GPU broadphase issues.",
                    "yellow",
                )
            self.max_contacts_per_frame = max(min(max_contacts, MAX_CONTACT_VIS), 1)  # 至少创建1个以避免空列表
            self.contact_sphere_actors = []
            cprint(f"[CONTACT VIS] Will create {self.max_contacts_per_frame} contact point spheres per environment", "cyan")
            # 打印每一帧接触最多的物体需要标注的点数
            print("[CONTACT VIS] Per-frame max contact points (most-contacted object per frame):")
            for fi, frame in enumerate(self.contact_data['frames']):
                if len(frame['objects']) == 0:
                    frame_max = 0
                else:
                    frame_max = max(obj.get('num_contacts', 0) for obj in frame['objects'])
                print(f"  Frame {fi}: {frame_max}")
            # 仅在启用接触奖励时预处理接触点（避免每步重复构造 tensor）
            if self.enable_contact_reward:
                self._contact_reward_frames = []
                for frame in self.contact_data["frames"]:
                    frame_contacts = []
                    for obj in frame["objects"]:
                        for contact in obj.get("contacts", []):
                            frame_contacts.append(
                                {
                                    "pos": torch.tensor(contact["object_contact_pos"], device=self.sim_device, dtype=torch.float32),
                                    "hand_idx": contact.get("hand_point_idx", None),
                                }
                            )
                    self._contact_reward_frames.append(frame_contacts)
        else:
            self.max_contacts_per_frame = 0

        # 预判哪些物体在参考轨迹中保持静止（用于关闭碰撞）
        self.static_object_mask = []
        for obj_info in self.scene_objects:
            traj = obj_info.get("trajectory", None)
            self.static_object_mask.append(self._is_static_trajectory(traj))

        # === 加载所有场景物体的 Assets ===
        self.obj_assets = []
        for obj_info in self.scene_objects:
            asset_options = gymapi.AssetOptions()
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.fix_base_link = True  # 按需求保持固定
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.disable_gravity = True
            asset_options.density = 200
            
            current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_info['urdf']), asset_options)
            
            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = 1.0
                element.rolling_friction = 0.01
                element.torsion_friction = 0.01
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            
            self.obj_assets.append(current_asset)

        self.envs = []
        self.hand_idxs = []
        self.obj_actor_handles = [] 

        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
            # Create DexHand
            # collision_filter 说明：
            # - 两个物体碰撞的条件：它们的 collision_filter 按位与结果不为 0
            # - self.dexhand.self_collision 控制手部内部自碰撞：
            #   * True: 使用 collision_filter=1，允许手部内部碰撞，同时也能与物体碰撞（物体也用1）
            #   * False: 使用 collision_filter=1，禁止手部内部碰撞，但仍能与物体碰撞（物体也用1）
            # 注意：collision_filter=0 会导致无法与任何 collision_filter=0 的物体碰撞
            dexhand_collision_filter = 1  # 使用1而不是0，确保能与物体碰撞
            dexhand_actor = self.gym.create_actor(
                env,
                dexhand_asset,
                self.dexhand_default_pose,
                "dexhand",
                i,
                dexhand_collision_filter,
            )

            self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

            # === 创建所有物体的 Actors ===
            env_obj_handles = []
            for k, asset in enumerate(self.obj_assets):
                obj_name = self.scene_objects[k]['name']
                # collision_filter 说明：
                # - 静态物体：使用 0xFFFF（全1）忽略所有碰撞
                # - 非静态物体：使用 1，与手的 collision_filter=1 按位与结果为 1，会发生碰撞
                # 注意：collision_filter=0 会导致无法与任何 collision_filter=0 的物体碰撞
                collision_filter = 1 if (k < len(self.static_object_mask) and self.static_object_mask[k]) else 0
                handle = self.gym.create_actor(
                    env, asset, gymapi.Transform(), f"{obj_name}_{i}", i, collision_filter
                )
                env_obj_handles.append(handle)
            self.obj_actor_handles.append(env_obj_handles)
            
            if len(env_obj_handles) > 0:
                self.obj_actor = env_obj_handles[0] 
            # ==============================

            # Visualization Spheres
            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)
            
            # === 接触点可视化 Spheres（为当前环境创建） ===
            if self.contact_data is not None and self.max_contacts_per_frame > 0:
                env_contact_actors = []
                env_contact_actor_env_indices = []
                for contact_id in range(self.max_contacts_per_frame):
                    contact_sphere = self.gym.create_sphere(self.sim, 0.008, scene_asset_options)  # 稍大一点便于观察
                    # 初始位置设在远处（地下），后续更新时会移到正确位置
                    init_transform = gymapi.Transform()
                    init_transform.p = gymapi.Vec3(100, 100, -100)
                    init_transform.r = gymapi.Quat(0, 0, 0, 1)
                    contact_actor = self.gym.create_actor(
                        env, contact_sphere, init_transform, 
                        f"contact_point_{contact_id}", self.num_envs + 2, 0b1
                    )
                    # 记录在当前环境内的 actor 序号（用于 tensor API 写入）
                    env_idx_local = self.gym.get_actor_index(env, contact_actor, gymapi.DOMAIN_ENV)
                    env_contact_actor_env_indices.append(env_idx_local)
                    # 使用醒目的红色标记接触点
                    contact_color = gymapi.Vec3(1.0, 0.0, 0.0)  # 红色
                    self.gym.set_rigid_body_color(env, contact_actor, 0, gymapi.MESH_VISUAL, contact_color)
                    env_contact_actors.append(contact_actor)
                self.contact_sphere_actors.append(env_contact_actors)
                if not hasattr(self, "contact_actor_env_indices"):
                    self.contact_actor_env_indices = []
                self.contact_actor_env_indices.append(env_contact_actor_env_indices)

        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_dof_handles = {
            k: self.gym.find_actor_dof_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.dof_names
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.q = self._dof_state[..., 0]
        self.qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        self.isaac2chain_order = [
            self.gym.get_actor_dof_names(env_ptr, dexhand_handle).index(j)
            for j in self.chain.get_joint_parameter_names()
        ]

        self.mano_joint_points = [
            self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
            for i in range(len(self.dexhand.body_names))
        ]

        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

    def _is_static_trajectory(self, trajectory, pos_tol=1e-5, rot_tol=1e-5):
        """判断物体轨迹是否静止（位置与旋转基本不变）"""
        if trajectory is None:
            return False
        try:
            if isinstance(trajectory, torch.Tensor):
                pos = trajectory[:, :3, 3]
                rot = trajectory[:, :3, :3]
                pos_diff = (pos - pos[0]).abs().max().item()
                rot_diff = (rot - rot[0]).abs().max().item()
            else:
                pos = trajectory[:, :3, 3]
                rot = trajectory[:, :3, :3]
                pos_diff = np.max(np.abs(pos - pos[0]))
                rot_diff = np.max(np.abs(rot - rot[0]))
        except Exception:
            return False
        return pos_diff < pos_tol and rot_diff < rot_tol

    def _compute_palm_direction(self, wrist_pos, mano_joints):
        """基于参考关节位置估计手心法线方向"""
        idx_index = self._hand_joint_indices.get("index_proximal", None)
        idx_pinky = self._hand_joint_indices.get("pinky_proximal", None)
        if idx_index is None or idx_pinky is None:
            return None, None
        index_base = mano_joints[:, idx_index]
        pinky_base = mano_joints[:, idx_pinky]
        normal = torch.cross(index_base - wrist_pos, pinky_base - wrist_pos, dim=-1)
        norm = torch.norm(normal, dim=-1, keepdim=True)
        valid_mask = norm.squeeze(-1) > 1e-6
        palm_dir = torch.where(valid_mask.unsqueeze(-1), normal / norm.clamp(min=1e-6), torch.zeros_like(normal))
        return palm_dir, valid_mask

    def _load_object_point_clouds(self, num_points=1024):
        """从 URDF 采样物体局部点云并缓存"""
        if self._object_points_local_cache is not None:
            return self._object_points_local_cache

        object_points = []
        for obj in self.scene_objects:
            urdf_path = obj.get("urdf", None)
            if urdf_path is None or not os.path.exists(urdf_path):
                cprint(f"[PALM_DIR] urdf missing: {urdf_path}", "yellow")
                object_points.append(None)
                continue
            try:
                tree = ET.parse(urdf_path)
                root = tree.getroot()
                mesh_elem = root.find(".//mesh")
                if mesh_elem is None:
                    cprint(f"[PALM_DIR] mesh tag missing in {urdf_path}", "yellow")
                    object_points.append(None)
                    continue
                mesh_filename = mesh_elem.attrib.get("filename", "")
                if mesh_filename.startswith("package://"):
                    mesh_filename = mesh_filename.replace("package://", "")
                mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_filename)
                scale_str = mesh_elem.attrib.get("scale", "1 1 1")
                scale = float(scale_str.strip().split()[0])
                if not os.path.exists(mesh_path):
                    cprint(f"[PALM_DIR] mesh file missing: {mesh_path}", "yellow")
                    object_points.append(None)
                    continue
                mesh_obj = trimesh.load(mesh_path, force="mesh")
                mesh_obj.apply_scale(scale)
                center = np.mean(mesh_obj.vertices, axis=0)
                points, _ = trimesh.sample.sample_surface_even(mesh_obj, count=num_points, seed=2024)
                points = points - center
                points = torch.tensor(points, device=self.sim_device, dtype=torch.float32)
                if points.shape[0] < num_points:
                    points = torch.cat([points, points[: num_points - points.shape[0]]], dim=0)
                object_points.append(points)
            except Exception as e:
                cprint(f"[PALM_DIR] failed to load point cloud from {urdf_path}: {e}", "yellow")
                object_points.append(None)

        self._object_points_local_cache = object_points
        return object_points

    def _compute_palm_direction_from_objects(self, palm_pos, processed_trajs, num_points=1024):
        """
        使用距离手心最近的物体点云点（取其到手心的反方向）计算 palm_dir
        palm_pos: [N, 3] 手心位置（此处用 target_wrist_pos）
        processed_trajs: List[Tensor]，每个物体的 [N, 4, 4] 轨迹（已转到 gym 坐标系）
        """
        if len(self.scene_objects) == 0 or len(processed_trajs) == 0:
            zeros = torch.zeros_like(palm_pos)
            return zeros, torch.zeros(palm_pos.shape[0], device=self.sim_device, dtype=torch.bool), zeros

        object_points_local = self._load_object_point_clouds(num_points=num_points)
        closest_points = torch.zeros_like(palm_pos)
        min_dist = torch.full((palm_pos.shape[0],), float("inf"), device=self.sim_device, dtype=palm_pos.dtype)

        for obj_pts, traj in zip(object_points_local, processed_trajs):
            if obj_pts is None:
                continue
            # traj: [N, 4, 4], obj_pts: [P, 3]
            R = traj[:, :3, :3]
            t = traj[:, :3, 3]
            # 先旋转再平移，得到 [N, P, 3]
            obj_world = torch.einsum("nij,pj->npi", R, obj_pts) + t[:, None, :]
            dist = torch.norm(obj_world - palm_pos[:, None, :], dim=-1)  # [N, P]
            min_d, min_idx = torch.min(dist, dim=1)
            update_mask = min_d < min_dist
            min_dist = torch.where(update_mask, min_d, min_dist)
            chosen = obj_world[torch.arange(palm_pos.shape[0], device=self.sim_device), min_idx]
            closest_points = torch.where(update_mask.unsqueeze(-1), chosen, closest_points)

        dir_vec = closest_points - palm_pos  # 手心 -> 最近点
        norm = torch.norm(dir_vec, dim=-1, keepdim=True)
        valid_mask = (min_dist < float("inf")) & (norm.squeeze(-1) > 1e-6)
        palm_dir = torch.where(valid_mask.unsqueeze(-1), dir_vec / norm.clamp(min=1e-6), torch.zeros_like(dir_vec))
        return palm_dir, valid_mask, closest_points

    def _update_contact_spheres(self):
        """更新所有环境中的接触点sphere位置
        
        注意：接触点spheres作为独立actors，需要通过gym API直接设置位置
        """
        if self.contact_data is None or not hasattr(self, 'contact_sphere_actors'):
            return
        
        # 遍历每个环境（每个环境对应一帧）
        for env_idx in range(self.num_envs):
            if env_idx >= len(self.contact_data['frames']):
                continue
                
            frame_data = self.contact_data['frames'][env_idx]
            contact_idx = 0
            env = self.envs[env_idx]
            
            # 只可视化 num_contacts 最大的那个物体；若点数超过上限则随机采样
            if len(frame_data['objects']) > 0:
                best_obj = max(frame_data['objects'], key=lambda o: o.get('num_contacts', 0))
                contacts_iter = best_obj.get('contacts', [])
                if len(contacts_iter) > self.max_contacts_per_frame:
                    idx = np.random.choice(len(contacts_iter), self.max_contacts_per_frame, replace=False)
                    contacts_iter = [contacts_iter[i] for i in idx]
            else:
                contacts_iter = []
            
            for contact in contacts_iter:
                if contact_idx >= self.max_contacts_per_frame:
                    break
                
                # 获取接触点的世界坐标
                contact_pos = contact['object_contact_pos']  # numpy array [3]
                actor_env_idx = self.contact_actor_env_indices[env_idx][contact_idx]
                
                # 直接写入 root_state tensor（GPU pipeline 安全）
                self._root_state[env_idx, actor_env_idx, 0:3] = torch.tensor(
                    [contact_pos[0], contact_pos[1], contact_pos[2]],
                    device=self._root_state.device,
                    dtype=self._root_state.dtype,
                )
                self._root_state[env_idx, actor_env_idx, 3:7] = torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    device=self._root_state.device,
                    dtype=self._root_state.dtype,
                )
                self._root_state[env_idx, actor_env_idx, 7:] = 0
                
                contact_idx += 1
            
            # 将未使用的 sphere 移到远处（不可见）
            for unused_idx in range(contact_idx, self.max_contacts_per_frame):
                actor_env_idx = self.contact_actor_env_indices[env_idx][unused_idx]
                self._root_state[env_idx, actor_env_idx, 0:3] = torch.tensor(
                    [100.0, 100.0, -100.0],
                    device=self._root_state.device,
                    dtype=self._root_state.dtype,
                )
                self._root_state[env_idx, actor_env_idx, 3:7] = torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    device=self._root_state.device,
                    dtype=self._root_state.dtype,
                )
                self._root_state[env_idx, actor_env_idx, 7:] = 0
    
    def fitting(self, max_iter, target_wrist_pos, target_wrist_rot, target_mano_joints):
        assert target_mano_joints.shape[0] == self.num_envs
        
        # 转换 Target 到 Gym 坐标系
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        # 处理多物体轨迹，并计算 Offset
        processed_trajs = []
        for obj_info in self.scene_objects:
            traj = obj_info['trajectory']
            # 转换轨迹坐标系
            traj = self.mujoco2gym_transf.to(traj.device) @ traj
            processed_trajs.append(traj)

        # === [打印] 第0个环境的所有物体的初始位置 ===
        print(f"\n[MANO2DEXHAND] Env 0 - All Objects Initial Positions:")
        for k, (obj_info, traj) in enumerate(zip(self.scene_objects, processed_trajs)):
            obj_name = obj_info.get('name', f'obj_{k}')
            # 获取第0个环境、第0帧的位置
            obj_pos = traj[0, :3, 3].cpu().numpy()  # [3]
            print(f"  Object {k} ({obj_name}): pos=({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})")
        print()

        # === [新方案] 基于手掌朝向 + 接触锚点的初始位姿 ===
        # 步骤1: 计算锚点 - 使用所有接触点的几何中心
        anchor_points = []
        palm_valid = torch.ones(self.num_envs, device=self.sim_device, dtype=torch.bool)

        if self.contact_data is not None:
            cprint("[PALM_DIR] Using contact-based anchor calculation", "cyan")
            for env_idx in range(self.num_envs):
                if env_idx >= len(self.contact_data["frames"]):
                    anchor_points.append(target_wrist_pos[env_idx])
                    continue

                frame_data = self.contact_data["frames"][env_idx]
                all_contacts = []
                for obj in frame_data["objects"]:
                    for contact in obj.get("contacts", []):
                        all_contacts.append(contact["object_contact_pos"])

                if len(all_contacts) > 0:
                    anchor = np.mean(all_contacts, axis=0)
                    anchor_points.append(torch.tensor(anchor, device=self.sim_device, dtype=torch.float32))
                    cprint(
                        f"[PALM_DIR] Frame {env_idx}: {len(all_contacts)} contacts, anchor=({anchor[0]:.4f}, {anchor[1]:.4f}, {anchor[2]:.4f})",
                        "yellow",
                    )
                else:
                    anchor_points.append(target_wrist_pos[env_idx])
                    cprint(f"[PALM_DIR] Frame {env_idx}: No contacts, using wrist pos", "yellow")
        else:
            cprint("[PALM_DIR] No contact data, using wrist positions as anchors", "yellow")
            for env_idx in range(self.num_envs):
                anchor_points.append(target_wrist_pos[env_idx])

        P_anchor = torch.stack(anchor_points)  # [N, 3]

        # 步骤2: 计算手掌法线 (朝向手心)
        idx_index_prox = self._hand_joint_indices.get("index_proximal", None)
        idx_pinky_prox = self._hand_joint_indices.get("pinky_proximal", None)

        if idx_index_prox is None or idx_pinky_prox is None:
            cprint("[PALM_DIR] Cannot find index/pinky proximal joints, using default offset", "red")
            palm_dir = torch.zeros_like(target_wrist_pos)
            palm_dir[:, 2] = 1.0  # 默认指向手心（+Z）
            palm_valid = torch.zeros(self.num_envs, device=self.sim_device, dtype=torch.bool)
        else:
            P_index_prox = target_mano_joints[:, idx_index_prox]  # [N, 3]
            P_pinky_prox = target_mano_joints[:, idx_pinky_prox]  # [N, 3]

            V1 = P_index_prox - target_wrist_pos
            V2 = P_pinky_prox - target_wrist_pos
            N_palm = torch.cross(V1, V2, dim=-1)

            norm = torch.norm(N_palm, dim=-1, keepdim=True)
            palm_valid = norm.squeeze(-1) > 1e-6
            palm_dir = torch.where(palm_valid.unsqueeze(-1), N_palm / norm.clamp(min=1e-6), torch.zeros_like(N_palm))

            # 通过锚点方向修正法线朝向，确保指向手心
            anchor_dir = P_anchor - target_wrist_pos
            flip_mask = (torch.sum(palm_dir * anchor_dir, dim=-1, keepdim=True) < 0) & palm_valid.unsqueeze(-1)
            palm_dir = torch.where(flip_mask, -palm_dir, palm_dir)

            cprint(f"[PALM_DIR] Computed palm normal for {palm_valid.sum().item()}/{self.num_envs} frames", "green")

        closest_points = P_anchor  # 用于可视化
        palm_pos = P_anchor

        # 步骤3: 最终位置 - 沿手掌法线反向（手背方向）回退
        # P_init = P_anchor - d × N_palm
        retreat_distance = 0.05
        offset = torch.where(
            palm_valid.unsqueeze(-1),
            - retreat_distance * palm_dir,
            torch.zeros_like(target_wrist_pos),
        )

        opt_wrist_pos = torch.tensor(
            target_wrist_pos + offset,
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_wrist_rot = torch.tensor(
            rotmat_to_rot6d(target_wrist_rot), device=self.sim_device, dtype=torch.float32, requires_grad=True
        )
        opt_dof_pos = torch.tensor(
            self.dexhand_default_dof_pos["pos"][None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opti = torch.optim.Adam(
            [{"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.0008}, {"params": [opt_dof_pos], "lr": 0.0004}]
        )

        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "tip" in k:
                if "index" in k: weight.append(20)
                elif "middle" in k: weight.append(10)
                elif "ring" in k: weight.append(17)
                elif "pinky" in k: weight.append(15)
                elif "thumb" in k: weight.append(1)
                else: raise ValueError
            elif "proximal" in k: weight.append(5)
            elif "intermediate" in k: weight.append(5)
            else: weight.append(1)
        weight[0] = 25.0
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        
        iter = 0
        past_loss = 1e10
        current_frame = 0  # 跟踪当前帧索引（用于接触点可视化）
        
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1
            # 更新当前帧索引（每个环境对应一帧）
            # 因为 num_envs == num_frames，环境 i 对应帧 i
            
            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])

            # 更新所有物体的位置
            for k, traj in enumerate(processed_trajs):
                pos = traj[:, :3, 3]
                quat = rotmat_to_quat(traj[:, :3, :3])[:, [1, 2, 3, 0]]
                
                actor_idx = 1 + k
                self._root_state[:, actor_idx, :3] = pos
                self._root_state[:, actor_idx, 3:7] = quat


            opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

            # === 更新接触点位置 ===
            if self.contact_data is not None:
                self._update_contact_spheres()
            
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(opt_dof_pos_clamped))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self.headless:
                self.gym.step_graphics(self.sim)

            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            isaac_joints = torch.stack(
                [self._rigid_body_state[:, self.dexhand_handles[k], :3] for k in self.dexhand.body_names],
                dim=1,
            )

            ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
            pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
            pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(
                -1, -2
            ) + opt_wrist_pos[:, None]

            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
            # 可视化参考轨迹连线（绿色）
            if (not self.headless) and hasattr(self.dexhand, "bone_links") and len(self.dexhand.bone_links) > 0:
                env_ids = range(self.num_envs) if self.draw_all_lines else [0]
                for env_idx in env_ids:
                    env_ptr = self.envs[env_idx]
                    joints_np = target_joints[env_idx].detach().cpu().numpy().astype(np.float32)  # [num_joints,3]
                    lines = np.array([[joints_np[a], joints_np[b]] for a, b in self.dexhand.bone_links], dtype=np.float32)
                    verts = lines.reshape(-1, 3).astype(np.float32)  # [num_lines*2, 3]
                    colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (verts.shape[0], 1))
                    self.gym.add_lines(
                        self.viewer,
                        env_ptr,
                        lines.shape[0],
                        verts,
                        colors,
                    )
                    # 显示“最近点 -> 手心”连线（红色），便于检查方向
                    if 'closest_points' in locals():
                        cp = closest_points[env_idx].detach().cpu().numpy().astype(np.float32)
                        pp = palm_pos[env_idx].detach().cpu().numpy().astype(np.float32)
                        line_cp = np.array([[pp, cp]], dtype=np.float32)  # 手 -> 最近点
                        verts_cp = line_cp.reshape(-1, 3).astype(np.float32)
                        colors_cp = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float32), (verts_cp.shape[0], 1))
                        self.gym.add_lines(
                            self.viewer,
                            env_ptr,
                            line_cp.shape[0],
                            verts_cp,
                            colors_cp,
                        )
            # 计算每个关节的偏差
            joint_err = torch.norm(pk_joints - target_joints, dim=-1)  # [B, J]

            # Wrist 收敛门控：仅当手腕位置/姿态都在阈值内时，才对其它关节施加损失
            wrist_pos_err = torch.norm(opt_wrist_pos - target_wrist_pos, dim=-1)  # [B]
            wrist_rot_mat = rot6d_to_rotmat(opt_wrist_rot)                       # [B,3,3]
            target_wrist_rot_mat = target_wrist_rot                              # [B,3,3]
            rot_rel = torch.matmul(wrist_rot_mat.transpose(-1, -2), target_wrist_rot_mat)
            cos_theta = (rot_rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            wrist_rot_err = torch.acos(cos_theta)                                # [B]

            wrist_ready = (wrist_pos_err <= self.wrist_pos_tol) & (wrist_rot_err <= self.wrist_rot_tol)
            # 构造逐样本权重：手腕始终保留，其他关节需 wrist_ready 才生效
            weight_per_env = weight[None, :].repeat(pk_joints.shape[0], 1)
            weight_per_env[:, 1:] = weight_per_env[:, 1:] * wrist_ready[:, None]

            loss = torch.mean(joint_err * weight_per_env)
            # === 可选接触奖励：鼓励手关节贴近接触点，匹配指定关节给予更大奖励 ===
            if self._contact_reward_frames is not None and len(self._contact_reward_frames) > 0:
                contact_terms = []
                sigma = max(self.contact_reward_sigma, 1e-6)
                max_env = min(len(self._contact_reward_frames), pk_joints.shape[0])
                for env_idx in range(max_env):
                    contacts = self._contact_reward_frames[env_idx]
                    if not contacts:
                        continue
                    joints_pos = pk_joints[env_idx]  # [num_joints,3]
                    for contact in contacts:
                        contact_pos = contact["pos"]
                        dists = torch.norm(joints_pos - contact_pos[None, :], dim=-1)
                        min_dist = torch.min(dists)
                        target_dist = min_dist
                        h_idx = contact.get("hand_idx", None)
                        if h_idx is not None and h_idx < joints_pos.shape[0]:
                            target_dist = dists[h_idx]
                        # 奖励采用 exp(-dist/sigma)，距离越近奖励越大
                        bonus = self.contact_reward_scale * torch.exp(-min_dist / sigma)
                        bonus += self.contact_reward_match_scale * torch.exp(-target_dist / sigma)
                        # 奖励以负号加入 loss，使其距离越近总损失越小
                        contact_terms.append(-bonus)
                if len(contact_terms) > 0 and past_loss - loss.item() < 5 * 1e-2:
                    contact_loss = torch.stack(contact_terms).mean()
                    loss = loss + contact_loss
            opti.zero_grad()
            loss.backward()
            opti.step()

            if iter % 100 == 0:
                cprint(f"{iter} {loss.item()}", "green")
                if iter > 1 and past_loss - loss.item() < 1e-5:
                    break
                past_loss = loss.item()

        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="Mano to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 4000,
            },
            {
                "name": "--data_idx",
                "type": str,
                "default": "1906",
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
            },
            {
                "name": "--contact_data",
                "type": str,
                "default": None,
                "help": "Path to contact data pkl file (optional, for visualization)",
            },
            {
                "name": "--enable_contact_reward",
                "type": int,
                "default": 0,
                "help": "1 to add contact reward term in loss, 0 to disable",
            },
            {
                "name": "--contact_reward_scale",
                "type": float,
                "default": 0.1,
                "help": "Reward weight for any joint touching a contact point",
            },
            {
                "name": "--contact_reward_match_scale",
                "type": float,
                "default": 0.2,
                "help": "Extra reward weight when the annotated joint touches the contact point",
            },
            {
                "name": "--contact_reward_sigma",
                "type": float,
                "default": 0.01,
                "help": "Distance falloff (m) for contact reward exp(-d/sigma)",
            },
            {
                "name": "--wrist_pos_tol",
                "type": float,
                "default": 0.01,
                "help": "Start optimizing other joints only when wrist position error <= this (m)",
            },
            {
                "name": "--wrist_rot_tol",
                "type": float,
                "default": 5.0,
                "help": "Start optimizing other joints only when wrist rotation error <= this (degrees)",
            },
            {
                "name": "--max_frames",
                "type": int,
                "default": -1,
                "help": "Optimize only the first N frames; -1 means use all frames",
            },
            {
                "name": "--draw_all_lines",
                "type": int,
                "default": 0,
                "help": "1 to draw reference lines for all envs (frames). 0 to draw only env0",
            },
            {
                "name": "--render_pkl",
                "type": str,
                "default": None,
                "help": "If set, visualize a saved pkl with red lines (skip optimization)",
            },
            {
                "name": "--render_output_dir",
                "type": str,
                "default": "render_outputs",
                "help": "Directory to save rendered frames when using --render_pkl in headless mode",
            },
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def visualize_pkl(pkl_path, dexhand, headless=False, output_dir="render_outputs"):
        """加载 pkl 结果并用红线逐帧展示（支持所有帧），headless 时保存图片。"""
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"pkl not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if "opt_joints_pos" not in data:
            raise RuntimeError("pkl missing opt_joints_pos; unable to visualize")
        joints = torch.tensor(data["opt_joints_pos"], dtype=torch.float32)  # [T, J, 3]
        T = joints.shape[0]

        # headless 无显示时强制使用 EGL，避免缺少 X11/GLFW 导致崩溃
        if headless:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        graphics_device_id = 0  # 使用 GPU 图形设备；EGL 场景下不需要物理显示
        sim = gym.create_sim(0, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            raise RuntimeError(
                "Failed to create sim. Headless 模式需要有效的 EGL/驱动环境，"
                "请确认已安装并可用（或使用 xvfb-run 提供虚拟显示）。"
            )
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane_params)
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
        # headless 环境下避免创建 viewer（需要 GLFW/显示环境，会导致崩溃）
        viewer = None if headless else gym.create_viewer(sim, gymapi.CameraProperties())

        # 相机传感器（headless 用于离线渲染）
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 480
        cam_props.enable_tensors = True
        cam_props.use_collision_geometry = False
        camera_handle = gym.create_camera_sensor(env, cam_props)
        cam_pos = gymapi.Vec3(1.0, 0.0, 0.8)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.4)
        gym.set_camera_location(camera_handle, env, cam_pos, cam_target)

        if headless:
            os.makedirs(output_dir, exist_ok=True)

        if len(dexhand.bone_links) == 0:
            cprint("[WARN] dexhand.bone_links not found; cannot draw lines", "yellow")
        else:
            for frame_idx in range(T):
                if viewer is not None:
                    gym.clear_lines(viewer)
                    j_np = joints[frame_idx].cpu().numpy().astype(np.float32)
                    lines = np.array([[j_np[a], j_np[b]] for a, b in dexhand.bone_links], dtype=np.float32)
                    verts = lines.reshape(-1, 3).astype(np.float32)
                    colors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (verts.shape[0], 1))
                    gym.add_lines(viewer, env, lines.shape[0], verts, colors)

                gym.step_graphics(sim)
                if viewer is not None:
                    gym.draw_viewer(viewer, sim, False)
                    gym.sync_frame_time(sim)
                gym.render_all_camera_sensors(sim)
                img = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
                if headless:
                    out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                    imageio.imwrite(out_path, img)
                if viewer is not None and gym.query_viewer_has_closed(viewer):
                    break
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

    def run(parser, idx):
        dataset_type = ManipDataFactory.dataset_type(idx)
        demo_d = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=parser.side,
            device="cuda:0",
            mujoco2gym_transf=torch.eye(4, device="cuda:0"),
            dexhand=dexhand,
            verbose=False,
        )

        demo_data = pack_data([demo_d[idx]], dexhand)

        total_frames = demo_data["mano_joints"].shape[0]
        if parser.max_frames is not None and parser.max_frames > 0:
            use_frames = min(parser.max_frames, total_frames)
            # 截断到指定帧数
            demo_data["wrist_pos"] = demo_data["wrist_pos"][:use_frames]
            demo_data["wrist_rot"] = demo_data["wrist_rot"][:use_frames]
            demo_data["mano_joints"] = demo_data["mano_joints"][:use_frames]
            # 场景物体轨迹截断
            for obj in demo_data["scene_objects"]:
                if "trajectory" in obj:
                    obj["trajectory"] = obj["trajectory"][:use_frames]
                if "velocity" in obj:
                    obj["velocity"] = obj["velocity"][:use_frames]
                if "angular_velocity" in obj:
                    obj["angular_velocity"] = obj["angular_velocity"][:use_frames]
            parser.num_envs = use_frames
            cprint(f"[INFO] Limiting optimization to first {use_frames} frames (total {total_frames})", "cyan")
        else:
            parser.num_envs = total_frames
        scene_objects = demo_data["scene_objects"]

        # === 加载接触数据（如果提供） ===
        contact_data = None
        if parser.contact_data is not None and os.path.exists(parser.contact_data):
            cprint(f"[CONTACT VIS] Loading contact data from: {parser.contact_data}", "cyan")
            with open(parser.contact_data, 'rb') as f:
                contact_data = pickle.load(f)
            cprint(f"[CONTACT VIS] Loaded {contact_data['num_frames']} frames with contact data", "green")
        
        # === [修改] 传入 dataset_type 和 contact_data ===
        mano2inspire = Mano2Dexhand(parser, dexhand, scene_objects, dataset_type=dataset_type, contact_data=contact_data)
        # ==============================

        to_dump = mano2inspire.fitting(
            parser.iter,
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            demo_data["mano_joints"].view(parser.num_envs, -1, 3),
        )

        if dataset_type == "oakink2":
            dump_path = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "humoto":
            filename = os.path.basename(demo_data['data_path'][0])
            dump_path = f"data/retargeting/Humoto/mano2{str(dexhand)}/{filename}"
        elif dataset_type == "favor":
            dump_path = (
                f"data/retargeting/favor_pass1/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
            )
        elif dataset_type == "grabdemo":
            dump_path = f"data/retargeting/grab_demo/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.npy', '.pkl')}"
        elif dataset_type == "oakink2_mirrored":
            dump_path = f"data/retargeting/OakInk-v2-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "favor_mirrored":
            dump_path = f"data/retargeting/favor_pass1-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
        else:
            raise ValueError("Unsupported dataset type")

        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)

    if _parser.render_pkl is not None:
        visualize_pkl(_parser.render_pkl, dexhand, headless=_parser.headless, output_dir=_parser.render_output_dir)
    else:
        run(_parser, _parser.data_idx)
