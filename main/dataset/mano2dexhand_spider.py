"""
SPIDER-inspired sampling-based optimization for hand retargeting
基于 SPIDER 论文的采样优化方法，使用虚拟接触引导
"""
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
        elif k == "scene_objects":
            packed_data[k] = data[0][k]
        elif type(data[0][k]) == torch.Tensor:
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif type(data[0][k]) == np.ndarray:
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


class ContactPair:
    """记录一个接触对：哪个手指关节和哪个物体的接触"""
    def __init__(self, finger_idx, object_idx, relative_pos, start_frame):
        self.finger_idx = finger_idx  # 手指关节索引
        self.object_idx = object_idx  # 物体索引
        self.relative_pos = relative_pos  # 手指相对物体的位置 (3,)
        self.start_frame = start_frame  # 接触开始帧
        self.positions = []  # 记录接触点在物体表面的轨迹
        self.active = True  # 是否仍然活跃


class Mano2DexhandSPIDER:
    def __init__(self, args, dexhand, scene_objects, dataset_type="oakink2", contact_data=None):
        """
        SPIDER风格的采样优化
        
        args: 参数
        dexhand: 机器人手模型
        scene_objects: 包含所有物体信息的列表
        dataset_type: 数据集类型
        contact_data: 可选的接触数据字典
        """
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand
        self.scene_objects = scene_objects
        self.contact_data = contact_data

        # SPIDER 参数
        self.num_samples = args.num_samples  # 每次迭代采样数量
        self.num_elite = args.num_elite  # 精英样本数量
        self.contact_threshold = 0.02  # 接触判定距离阈值 (m)
        self.min_contact_duration = 3  # 最小接触持续帧数
        self.max_contact_slip = 0.05  # 最大滑移距离 (m)
        self.virtual_force_init = args.virtual_force_init  # 初始虚拟力强度
        self.virtual_force_decay = args.virtual_force_decay  # 虚拟力衰减率
        self.annealing_schedule = args.annealing_schedule  # 退火温度调度

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

        # 加载机器人 Asset
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
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]
            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        default_dof_state["pos"][8] = 0.8
        default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        # 配置坐标转换
        table_width_offset = 0.2
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        table_half_height = 0.015
        self._table_surface_z = table_pos.z + table_half_height

        if dataset_type == "humoto":
            print("[INFO] Detected Humoto dataset: Using Scheme 1 (Rotation fix only, No Z-offset).")
            mujoco2gym_transf[:3, 3] = np.array([0, 0, 0])
        else:
            mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])

        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        # 创建多个环境用于并行采样（每个环境测试一个样本）
        self.num_envs = self.num_samples
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # 加载所有场景物体的 Assets
        self.obj_assets = []
        for obj_info in self.scene_objects:
            asset_options = gymapi.AssetOptions()
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.fix_base_link = True
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
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
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

            env_obj_handles = []
            for k, asset in enumerate(self.obj_assets):
                obj_name = self.scene_objects[k]['name']
                handle = self.gym.create_actor(env, asset, gymapi.Transform(), f"{obj_name}_{i}", i, 0)
                env_obj_handles.append(handle)
            self.obj_actor_handles.append(env_obj_handles)

        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }

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

        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)
        
        cprint(f"[SPIDER] Initialized with {self.num_samples} parallel samples, {self.num_elite} elite", "cyan")

    def identify_contact_pairs(self, ref_trajectory, object_trajectories):
        """
        步骤1: 识别接触对
        从参考轨迹中识别哪些手指关节和哪些物体有接触意图
        
        ref_trajectory: [T, num_joints, 3] 参考手指关节轨迹
        object_trajectories: List of [T, 4, 4] 物体轨迹列表
        
        Returns: List[ContactPair]
        """
        T, num_joints, _ = ref_trajectory.shape
        contact_pairs = []
        
        cprint(f"[SPIDER] Identifying contact pairs from {T} frames...", "yellow")
        
        # 对每个手指关节和每个物体的组合进行检测
        for finger_idx in range(num_joints):
            for obj_idx, obj_traj in enumerate(object_trajectories):
                # 提取物体位置 [T, 3]
                obj_pos = obj_traj[:, :3, 3]
                finger_pos = ref_trajectory[:, finger_idx, :]  # [T, 3]
                
                # 计算距离 [T]
                distances = torch.norm(finger_pos - obj_pos, dim=-1)
                
                # 识别接触段（距离小于阈值）
                is_contact = distances < self.contact_threshold
                
                # 查找连续接触段
                contact_start = None
                for t in range(T):
                    if is_contact[t] and contact_start is None:
                        contact_start = t
                    elif not is_contact[t] and contact_start is not None:
                        # 接触段结束
                        duration = t - contact_start
                        if duration >= self.min_contact_duration:
                            # 计算接触段的平均相对位置
                            contact_segment = slice(contact_start, t)
                            relative_pos = (finger_pos[contact_segment] - obj_pos[contact_segment]).mean(dim=0)
                            
                            # 检查滑移（接触段内相对位置变化）
                            relative_positions = finger_pos[contact_segment] - obj_pos[contact_segment]
                            max_slip = torch.max(torch.norm(
                                relative_positions - relative_positions.mean(dim=0, keepdim=True),
                                dim=-1
                            )).item()
                            
                            if max_slip < self.max_contact_slip:
                                contact_pair = ContactPair(
                                    finger_idx=finger_idx,
                                    object_idx=obj_idx,
                                    relative_pos=relative_pos,
                                    start_frame=contact_start
                                )
                                contact_pairs.append(contact_pair)
                                cprint(
                                    f"  Found contact: finger {finger_idx} <-> object {obj_idx}, "
                                    f"frames {contact_start}-{t}, slip={max_slip:.4f}m",
                                    "green"
                                )
                        contact_start = None
                
                # 处理最后一个接触段
                if contact_start is not None:
                    duration = T - contact_start
                    if duration >= self.min_contact_duration:
                        contact_segment = slice(contact_start, T)
                        relative_pos = (finger_pos[contact_segment] - obj_pos[contact_segment]).mean(dim=0)
                        relative_positions = finger_pos[contact_segment] - obj_pos[contact_segment]
                        max_slip = torch.max(torch.norm(
                            relative_positions - relative_positions.mean(dim=0, keepdim=True),
                            dim=-1
                        )).item()
                        
                        if max_slip < self.max_contact_slip:
                            contact_pair = ContactPair(
                                finger_idx=finger_idx,
                                object_idx=obj_idx,
                                relative_pos=relative_pos,
                                start_frame=contact_start
                            )
                            contact_pairs.append(contact_pair)
                            cprint(
                                f"  Found contact: finger {finger_idx} <-> object {obj_idx}, "
                                f"frames {contact_start}-{T}, slip={max_slip:.4f}m",
                                "green"
                            )
        
        cprint(f"[SPIDER] Total {len(contact_pairs)} contact pairs identified", "cyan")
        return contact_pairs

    def compute_virtual_forces(self, current_finger_pos, object_positions, contact_pairs, virtual_force_strength):
        """
        步骤2: 计算虚拟接触力
        在手指和物体之间施加虚拟力（类似弹簧力）
        
        current_finger_pos: [num_samples, num_joints, 3] 当前手指位置
        object_positions: List of [num_samples, 3] 当前物体位置
        contact_pairs: List[ContactPair]
        virtual_force_strength: float 虚拟力强度系数
        
        Returns: [num_samples, num_joints, 3] 虚拟力
        """
        num_samples, num_joints, _ = current_finger_pos.shape
        virtual_forces = torch.zeros_like(current_finger_pos)
        
        for contact_pair in contact_pairs:
            finger_idx = contact_pair.finger_idx
            obj_idx = contact_pair.object_idx
            
            # 目标位置：物体位置 + 相对位置
            target_pos = object_positions[obj_idx] + contact_pair.relative_pos.to(object_positions[obj_idx].device)  # [num_samples, 3]
            
            # 当前手指位置
            current_pos = current_finger_pos[:, finger_idx, :]  # [num_samples, 3]
            
            # 弹簧力：F = k * (target - current)
            force = virtual_force_strength * (target_pos - current_pos)  # [num_samples, 3]
            
            virtual_forces[:, finger_idx, :] += force
        
        return virtual_forces

    def sample_trajectories(self, mean_params, std_params, temperature):
        """
        步骤3: 退火采样
        从高斯分布采样轨迹参数，温度控制探索范围
        
        mean_params: [num_dofs] 均值
        std_params: [num_dofs] 标准差
        temperature: float 退火温度
        
        Returns: [num_samples, num_dofs]
        """
        noise = torch.randn(self.num_samples, len(mean_params), device=self.sim_device)
        samples = mean_params[None, :] + temperature * std_params[None, :] * noise
        
        # 限制在关节范围内
        samples = torch.clamp(samples, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)
        
        return samples

    def evaluate_samples(self, samples, target_joints, virtual_forces):
        """
        步骤4: 评估样本质量
        计算每个样本的损失（模仿误差 + 物理约束）
        
        samples: [num_samples, num_dofs]
        target_joints: [num_joints, 3] 目标关节位置
        virtual_forces: [num_samples, num_joints, 3]
        
        Returns: [num_samples] 损失
        """
        # 正向运动学计算当前关节位置
        ret = self.chain.forward_kinematics(samples[:, self.isaac2chain_order])
        current_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)  # [num_samples, num_joints, 3]
        
        # 模仿误差
        imitation_error = torch.norm(current_joints - target_joints[None, :, :], dim=-1).mean(dim=-1)  # [num_samples]
        
        # 虚拟力惩罚（希望最终不依赖虚拟力）
        force_penalty = torch.norm(virtual_forces, dim=-1).mean(dim=-1)  # [num_samples]
        
        # 碰撞检测（通过接触力）
        collision_forces = torch.norm(self._net_cf, dim=-1).max(dim=-1)[0]  # [num_samples]
        collision_penalty = torch.clamp(collision_forces - 0.1, min=0.0)  # 超过0.1N的力视为碰撞
        
        # 总损失
        total_loss = imitation_error + 0.1 * force_penalty + 10.0 * collision_penalty
        
        return total_loss

    def fitting(self, max_iter, target_wrist_pos, target_wrist_rot, target_mano_joints):
        """
        SPIDER主循环：采样优化 + 虚拟接触引导
        """
        assert target_mano_joints.shape[0] == 1, "SPIDER processes one frame at a time"
        
        # 转换到 Gym 坐标系
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(1, -1, 3)

        # 处理物体轨迹
        processed_trajs = []
        for obj_info in self.scene_objects:
            traj = obj_info['trajectory']
            traj = self.mujoco2gym_transf.to(traj.device) @ traj
            processed_trajs.append(traj)

        # 步骤1: 识别接触对
        ref_trajectory = torch.cat([target_wrist_pos[:, None], target_mano_joints[0]], dim=1)  # [1, num_joints, 3]
        contact_pairs = self.identify_contact_pairs(ref_trajectory, processed_trajs)

        # 初始化采样分布（以默认姿态为起点）
        mean_dof = torch.tensor(self.dexhand_default_dof_pos["pos"], device=self.sim_device, dtype=torch.float32)
        std_dof = torch.ones_like(mean_dof) * 0.1  # 初始标准差

        # 目标关节位置
        target_joints = torch.cat([target_wrist_pos, target_mano_joints[0]], dim=0)  # [num_joints, 3]

        # 退火调度
        temperature = 1.0
        virtual_force_strength = self.virtual_force_init

        cprint(f"[SPIDER] Starting optimization with {max_iter} iterations", "cyan")

        for iter in range(max_iter):
            # 步骤2: 退火采样
            samples = self.sample_trajectories(mean_dof, std_dof, temperature)  # [num_samples, num_dofs]

            # 设置所有环境的DOF
            self.q[:, :] = samples
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(samples))

            # 更新物体位置（所有环境使用相同物体位置）
            for k, traj in enumerate(processed_trajs):
                pos = traj[0, :3, 3].repeat(self.num_envs, 1)
                quat = rotmat_to_quat(traj[0, :3, :3])[None, [1, 2, 3, 0]].repeat(self.num_envs, 1)
                actor_idx = 1 + k
                self._root_state[:, actor_idx, :3] = pos
                self._root_state[:, actor_idx, 3:7] = quat

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            # 物理仿真
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            # 获取当前手指位置
            current_finger_pos = torch.stack(
                [self._rigid_body_state[:, self.dexhand_handles[k], :3] for k in self.dexhand.body_names],
                dim=1,
            )  # [num_samples, num_joints, 3]

            # 获取当前物体位置
            object_positions = [self._root_state[:, 1 + k, :3] for k in range(len(processed_trajs))]

            # 步骤3: 计算虚拟力
            virtual_forces = self.compute_virtual_forces(
                current_finger_pos, object_positions, contact_pairs, virtual_force_strength
            )

            # 步骤4: 评估样本
            losses = self.evaluate_samples(samples, target_joints, virtual_forces)

            # 步骤5: 选择精英样本并更新分布
            elite_indices = torch.argsort(losses)[:self.num_elite]
            elite_samples = samples[elite_indices]

            mean_dof = elite_samples.mean(dim=0)
            std_dof = elite_samples.std(dim=0).clamp(min=0.01)  # 防止方差过小

            # 退火：逐渐降低温度和虚拟力
            temperature *= self.annealing_schedule
            virtual_force_strength *= self.virtual_force_decay

            if iter % 10 == 0:
                best_loss = losses[elite_indices[0]].item()
                cprint(
                    f"Iter {iter}: best_loss={best_loss:.4f}, temp={temperature:.4f}, "
                    f"vf_strength={virtual_force_strength:.4f}",
                    "green"
                )

                if not self.headless:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, False)
                    self.gym.sync_frame_time(self.sim)

        # 返回最优解
        best_sample = mean_dof.detach().cpu().numpy()
        to_dump = {
            "opt_wrist_pos": target_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rotmat_to_aa(target_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": best_sample,
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="SPIDER-inspired Mano to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 100,
                "help": "Number of optimization iterations",
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
                "name": "--num_samples",
                "type": int,
                "default": 512,
                "help": "Number of parallel samples per iteration",
            },
            {
                "name": "--num_elite",
                "type": int,
                "default": 50,
                "help": "Number of elite samples for distribution update",
            },
            {
                "name": "--virtual_force_init",
                "type": float,
                "default": 10.0,
                "help": "Initial virtual force strength",
            },
            {
                "name": "--virtual_force_decay",
                "type": float,
                "default": 0.95,
                "help": "Virtual force decay rate per iteration",
            },
            {
                "name": "--annealing_schedule",
                "type": float,
                "default": 0.98,
                "help": "Temperature annealing rate",
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
        scene_objects = demo_data["scene_objects"]

        # 只处理第一帧作为演示
        frame_idx = 0
        single_frame_data = {
            "wrist_pos": demo_data["wrist_pos"][frame_idx:frame_idx+1],
            "wrist_rot": demo_data["wrist_rot"][frame_idx:frame_idx+1],
            "mano_joints": demo_data["mano_joints"][frame_idx:frame_idx+1],
        }
        
        # 物体轨迹也只取第一帧
        for obj in scene_objects:
            if "trajectory" in obj:
                obj["trajectory"] = obj["trajectory"][frame_idx:frame_idx+1]

        mano2spider = Mano2DexhandSPIDER(parser, dexhand, scene_objects, dataset_type=dataset_type)

        to_dump = mano2spider.fitting(
            parser.iter,
            single_frame_data["wrist_pos"],
            single_frame_data["wrist_rot"],
            single_frame_data["mano_joints"].view(1, -1, 3),
        )

        # 保存结果
        if dataset_type == "humoto":
            filename = os.path.basename(demo_data['data_path'][0])
            dump_path = f"data/retargeting/Humoto-SPIDER/mano2{str(dexhand)}/{filename}"
        else:
            dump_path = f"data/retargeting/SPIDER/mano2{str(dexhand)}/{idx}.pkl"

        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)
        
        cprint(f"[SPIDER] Results saved to {dump_path}", "green")

    run(_parser, _parser.data_idx)

