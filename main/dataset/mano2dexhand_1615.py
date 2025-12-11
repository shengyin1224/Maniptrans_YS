import math
import os
import pickle
from isaacgym import gymapi, gymtorch, gymutil
import logging

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import numpy as np
import pytorch_kinematics as pk
import torch
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
        self.draw_all_lines = getattr(args, "draw_all_lines", 0) == 1

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        if self.headless:
            self.graphics_device_id = -1

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu

        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params
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
            MAX_CONTACT_VIS = 32  # 如需更多可调高，但可能再次触顶
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
        else:
            self.max_contacts_per_frame = 0

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
            dexhand_actor = self.gym.create_actor(
                env,
                dexhand_asset,
                self.dexhand_default_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),
            )

            self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

            # === 创建所有物体的 Actors ===
            env_obj_handles = []
            for k, asset in enumerate(self.obj_assets):
                obj_name = self.scene_objects[k]['name']
                handle = self.gym.create_actor(env, asset, gymapi.Transform(), f"{obj_name}_{i}", i, 0)
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
            
            # 只可视化 num_contacts 最大的那个物体
            if len(frame_data['objects']) > 0:
                best_obj = max(frame_data['objects'], key=lambda o: o.get('num_contacts', 0))
                contacts_iter = best_obj.get('contacts', [])
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

        # 计算 Offset：优先用该帧的接触点中心；无接触时用最近物体方向
        middle_pos = (target_mano_joints[:, 3] + target_wrist_pos) / 2
        
        # 预取每帧接触点中心（若有），并打印每帧各物体接触点数与是否使用 contact 方向
        has_contacts = torch.zeros(self.num_envs, device=self.sim_device, dtype=torch.bool)
        contact_centers = torch.zeros(self.num_envs, 3, device=self.sim_device, dtype=middle_pos.dtype)
        if self.contact_data is not None:
            print("[CONTACT/OFFSET] Per-frame contact counts and offset source:")
        if self.contact_data is not None:
            for env_idx in range(self.num_envs):
                if env_idx >= len(self.contact_data['frames']):
                    continue
                frame = self.contact_data['frames'][env_idx]
                obj_contact_counts = []
                pts = []
                for obj in frame.get('objects', []):
                    obj_contact_counts.append((obj.get('object_name', 'unknown'), obj.get('num_contacts', 0)))
                    for c in obj.get('contacts', []):
                        if 'object_contact_pos' in c:
                            p = torch.tensor(c['object_contact_pos'], device=self.sim_device, dtype=middle_pos.dtype)
                            pts.append(p)
                if len(pts) > 0:
                    has_contacts[env_idx] = True
                    contact_centers[env_idx] = torch.stack(pts, dim=0).mean(dim=0)
                if self.contact_data is not None:
                    counts_str = ", ".join([f"{name}:{cnt}" for name, cnt in obj_contact_counts])
                    source = "contact" if has_contacts[env_idx] else "nearest_obj"
                    print(f"  Frame {env_idx}: {counts_str} | offset_source={source}")
        
        # 找最近物体（仅无接触时使用）
        min_dist = torch.full((self.num_envs,), float('inf'), device=middle_pos.device, dtype=middle_pos.dtype)
        closest_obj_pos = None
        
        for obj_traj in processed_trajs:
            obj_pos = obj_traj[:, :3, 3]  # [N, 3]
            dist = torch.norm(middle_pos - obj_pos, dim=-1)  # [N]
            
            # 只考虑距离<0.2的物体
            mask = dist < 0.2
            # 更新最小距离和对应的物体位置
            update_mask = mask & (dist < min_dist)
            min_dist = torch.where(update_mask, dist, min_dist)
            if closest_obj_pos is None:
                closest_obj_pos = obj_pos.clone()
            closest_obj_pos = torch.where(update_mask.unsqueeze(-1), obj_pos, closest_obj_pos)
        
        has_close_obj = min_dist < 0.2
        if closest_obj_pos is None:
            closest_obj_pos = torch.zeros_like(middle_pos)
        
        # 选择用于方向计算的目标：有接触则用接触中心，否则用最近物体
        target_anchor = torch.where(has_contacts.unsqueeze(-1), contact_centers, closest_obj_pos)
        
        # 计算 offset
        offset_vec = middle_pos - target_anchor
        offset_norm = torch.norm(offset_vec, dim=-1, keepdim=True).clamp(min=1e-6)
        offset = offset_vec / offset_norm * 0.2  # 调整为 1m
        # 无近物体且无接触的环境，offset 置零
        offset = torch.where((has_contacts | has_close_obj).unsqueeze(-1), offset, torch.zeros_like(offset))

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
                elif "ring" in k: weight.append(7)
                elif "pinky" in k: weight.append(5)
                elif "thumb" in k: weight.append(25)
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
            loss = torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])
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
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

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

    run(_parser, _parser.data_idx)