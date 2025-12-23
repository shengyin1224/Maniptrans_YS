from __future__ import annotations

import os
import random
from collections import deque
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from copy import deepcopy
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

# from main.dataset.favor_dataset_dexhand import FavorDatasetDexHand
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class DexHandManipBiHEnv(VecTask):

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        # self.dexhand_rh_dof_noise = self.cfg["env"]["dexhand_rDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]
        self.dexhand_rh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "right")
        self.dexhand_lh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "left")

        self.use_pid_control = self.cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            self.Kp_rot = self.dexhand_rh.Kp_rot
            self.Ki_rot = self.dexhand_rh.Ki_rot
            self.Kd_rot = self.dexhand_rh.Kd_rot

            self.Kp_pos = self.dexhand_rh.Kp_pos
            self.Ki_pos = self.dexhand_rh.Ki_pos
            self.Kd_pos = self.dexhand_rh.Kd_pos

        self.cfg["env"]["numActions"] = (
            (1 + 6 + self.dexhand_lh.n_dofs) if use_quat_rot else (6 + self.dexhand_lh.n_dofs)
        ) * (2 if self.cfg["env"]["bimanual_mode"] == "united" else 1)
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        # Values to be filled in at runtime
        self.rh_states = {}
        self.lh_states = {}
        self.dexhand_rh_handles = {}  # will be dict mapping names to relevant sim handles
        self.dexhand_lh_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        self.dataIndices = self.cfg["env"]["dataIndices"]
        # self.dataIndices = [tuple([int(i) for i in idx.split("@")]) for idx in self.dataIndices]
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]

        # === [新增] 动态难度调整相关配置（基于epoch） ===
        if self.tighten_method == "adaptive":
            # 配置参数（所有窗口大小都是基于epoch，而非episode）
            self.adaptive_success_window = self.cfg["env"].get("adaptiveSuccessWindow", 5)  # 连续N个epoch检查成功率
            self.adaptive_success_threshold = self.cfg["env"].get("adaptiveSuccessThreshold", 0.10)  # 成功率阈值（20%）
            self.adaptive_step_window = self.cfg["env"].get("adaptiveStepWindow", 5)  # 连续N个epoch检查平均step
            self.adaptive_step_threshold = self.cfg["env"].get("adaptiveStepThreshold", 20)  # 平均step阈值
            self.adaptive_no_improvement_window = self.cfg["env"].get("adaptiveNoImprovementWindow", 10)  # 连续N个epoch没有提升就降低难度
            self.adaptive_scale_factor_min = self.cfg["env"].get("adaptiveScaleFactorMin", 0.7)  # scale_factor最小值
            self.adaptive_scale_factor_max = self.cfg["env"].get("adaptiveScaleFactorMax", 1.0)  # scale_factor最大值
            self.adaptive_scale_step = self.cfg["env"].get("adaptiveScaleStep", 0.05)  # 每次调整的步长
            
            # 初始scale_factor从tighten_factor开始，如果没有则从1.0开始
            # tighten_factor如果存在且有效，作为初始值；否则从1.0（最宽松）开始
            if self.tighten_factor is not None and 0 < self.tighten_factor <= 1.0:
                initial_scale = self.tighten_factor
            else:
                initial_scale = 1.0
            
            # 全局难度调整（使用所有环境的平均表现，基于epoch而非episode）
            # 由于多环境并行，我们使用全局的scale_factor，基于所有环境的平均表现进行调整
            self.adaptive_global_scale_factor = initial_scale
            self.adaptive_epoch_success_rate_history = deque(maxlen=self.adaptive_success_window)  # 每个epoch的成功率历史
            self.adaptive_epoch_steps_history = deque(maxlen=self.adaptive_step_window)  # 每个epoch的平均step历史
            
            # 用于tensorboard记录的当前scale_factor（在compute_reward中更新）
            self.current_scale_factor = initial_scale
            
            # 当前epoch的累积数据
            self.adaptive_current_epoch = -1  # 当前epoch编号
            self.adaptive_current_epoch_success_count = 0  # 当前epoch的成功次数
            self.adaptive_current_epoch_reset_count = 0  # 当前epoch内的reset次数（用于计算成功率）
            self.adaptive_current_epoch_step_sum = 0.0  # 当前epoch的累计step
            self.adaptive_current_epoch_reward_sum = 0.0  # 当前epoch的累计reward（保留用于兼容）
            
            # 跟踪难度提升历史
            self.adaptive_last_difficulty_increase_epoch = -1  # 上次难度提升（scale_factor降低）时的epoch
            
            # 用于tensorboard记录的当前epoch统计信息
            self.current_epoch_success_rate = 0.0  # 当前epoch的成功率
            self.current_epoch_avg_steps = 0.0  # 当前epoch的平均step
            
            print(f"[INFO] Adaptive difficulty adjustment enabled (based on EPOCH):")
            print(f"  - Success rate window: {self.adaptive_success_window} epochs")
            print(f"  - Success rate threshold: {self.adaptive_success_threshold*100:.1f}%")
            print(f"  - Step check window: {self.adaptive_step_window} epochs")
            print(f"  - Step threshold: {self.adaptive_step_threshold}")
            print(f"  - No improvement window: {self.adaptive_no_improvement_window} epochs (auto-decrease if no improvement)")
            print(f"  - Scale factor range: [{self.adaptive_scale_factor_min}, {self.adaptive_scale_factor_max}]")
            print(f"  - Scale adjustment step: {self.adaptive_scale_step}")
            print(f"  - Initial scale factor: {initial_scale}")
        else:
            self.adaptive_success_window = None
            self.adaptive_success_threshold = None
            self.adaptive_step_window = None
            self.adaptive_step_threshold = None
            self.adaptive_no_improvement_window = None
            self.adaptive_scale_factor_min = None
            self.adaptive_scale_factor_max = None
            self.adaptive_scale_step = None
        
        # 用于tensorboard记录的当前scale_factor（所有模式都使用）
        if not hasattr(self, 'current_scale_factor'):
            # 如果不是adaptive模式，初始化为1.0或tighten_factor
            initial_scale_non_adaptive = self.tighten_factor if (self.tighten_factor is not None and 0 < self.tighten_factor <= 1.0) else 1.0
            self.current_scale_factor = initial_scale_non_adaptive
        
        # 初始化tensorboard统计信息（非adaptive模式也需要，设置为默认值）
        if not hasattr(self, 'current_epoch_success_rate'):
            self.current_epoch_success_rate = 0.0
            self.current_epoch_avg_steps = 0.0

        self.rollout_len = self.cfg["env"].get("rolloutLen", None)
        self.rollout_begin = self.cfg["env"].get("rolloutBegin", None)

        assert len(self.dataIndices) == 1 or self.rollout_len is None, "rolloutLen only works with one data"
        assert len(self.dataIndices) == 1 or self.rollout_begin is None, "rolloutBegin only works with one data"

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._dexhand_rh_effort_limits = None  # Actuator effort limits for dexhand_r
        self._dexhand_rh_dof_speed_limits = None  # Actuator speed limits for dexhand_r
        self._global_dexhand_rh_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device(sim_device)
        
        # === [新增] 初始化失败日志保存路径 ===
        self._init_failure_log_path()
        
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        # dexhand_r defaults
        # TODO hack here
        # default_pose = self.cfg["env"].get("dexhand_rDefaultDofPos", None)
        default_pose = torch.ones(self.dexhand_rh.n_dofs, device=self.device) * np.pi / 12
        if self.cfg["env"]["dexhand"] == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_rh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)
        self.dexhand_lh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)  # ? TODO check this
        # self.dexhand_rh_default_dof_pos = torch.tensor([-3.5322e-01,  -0.100e-01,  3.2278e-01, -2.51e+00,  1.6036e-01,
        #   2.564e+00, 0.5,  0.10,  0.10], device=self.sim_device)

        # load BPS model
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts_rh = self.demo_data_rh["obj_verts"]
        self.obj_bps_rh = self.bps_layer.encode(obj_verts_rh, feature_type=self.bps_feat_type)[self.bps_feat_type]
        obj_verts_lh = self.demo_data_lh["obj_verts"]
        self.obj_bps_lh = self.bps_layer.encode(obj_verts_lh, feature_type=self.bps_feat_type)[self.bps_feat_type]

        self._configure_target_obs_space()

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()
    
    def _init_failure_log_path(self):
        """初始化失败日志保存路径"""
        # 尝试从配置中获取实验目录
        experiment_dir = None
        
        # 方法1: 从 cfg 中获取 full_experiment_name（可能在 env 或顶层配置中）
        if hasattr(self, 'cfg') and self.cfg is not None:
            # 检查顶层配置
            full_experiment_name = self.cfg.get("full_experiment_name", None)
            # 如果顶层没有，检查 env 配置
            if not full_experiment_name and "env" in self.cfg:
                full_experiment_name = self.cfg["env"].get("full_experiment_name", None)
            
            if full_experiment_name:
                # 尝试在 runs/ 和 dumps/ 目录下查找
                for base_dir in ["runs", "dumps"]:
                    potential_dir = os.path.join(base_dir, full_experiment_name)
                    if os.path.exists(potential_dir):
                        experiment_dir = potential_dir
                        break
        
        # 方法2: 尝试从当前工作目录查找最近的 runs/ 或 dumps/ 目录
        if experiment_dir is None:
            # 查找当前目录下是否有 runs/ 或 dumps/ 目录
            for base_dir in ["runs", "dumps"]:
                if os.path.exists(base_dir) and os.path.isdir(base_dir):
                    # 获取最新的子目录
                    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                    if subdirs:
                        # 按修改时间排序，取最新的
                        subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
                        experiment_dir = os.path.join(base_dir, subdirs[0])
                        break
        
        # 方法3: 如果找不到，使用统一的文件夹
        if experiment_dir is None:
            experiment_dir = "failure_logs"
            os.makedirs(experiment_dir, exist_ok=True)
        else:
            # 确保目录存在
            os.makedirs(experiment_dir, exist_ok=True)
        
        # 创建失败日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.failure_log_file = os.path.join(experiment_dir, f"failure_diagnostics_{timestamp}.txt")
        
        # 初始化日志文件（写入头部信息）
        try:
            with open(self.failure_log_file, "w", encoding="utf-8") as f:
                f.write(f"Failure Diagnostics Log\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log file: {self.failure_log_file}\n")
                f.write(f"Experiment dir: {experiment_dir}\n")
                f.write("=" * 80 + "\n\n")
            print(f"[INFO] Failure diagnostics will be saved to: {self.failure_log_file}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize failure log file: {e}")
            self.failure_log_file = None

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # === [修改 1] 提前检测数据集类型 ===
        # 注意：此时数据还未加载，我们需要先临时判断一下，或者将 dataset_list 的逻辑提上来
        # 这里为了安全，我们先遍历 dataIndices 获取类型
        dataset_types = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))
        is_humoto = "humoto" in dataset_types

        # === [修改 2] 桌子配置 ===
        table_width_offset = 0.2
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        table_half_height = 0.015
        table_half_width = 0.4
        self._table_surface_z = table_pos.z + table_half_height
        
        # 只有非 Humoto 才需要桌子 Asset
        table_asset = None
        if not is_humoto:
            table_asset_options = gymapi.AssetOptions()
            table_asset_options.fix_base_link = True
            table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        # ... (保留 dexhand_pose 设置) ...
        self.dexhand_rh_pose = gymapi.Transform()
        self.dexhand_rh_pose.p = gymapi.Vec3(-table_half_width, 0, self._table_surface_z + ROBOT_HEIGHT)
        self.dexhand_rh_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)
        self.dexhand_lh_pose = deepcopy(self.dexhand_rh_pose)

        # === [修改 3] 坐标转换矩阵 (Z轴归零逻辑) ===
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
            )

        if is_humoto:
            print("[INFO] Humoto Dataset: Using Raw Z (No Table Offset).")
            mujoco2gym_transf[:3, 3] = np.array([0, 0, 0])
        else:
            print("[INFO] Other Dataset: Using Table Offset.")
            mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])

        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        self.demo_dataset_lh_dict = {}
        self.demo_dataset_rh_dict = {}

        for dataset_type in dataset_list:
            self.demo_dataset_lh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="left",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_lh,
                embodiment=self.cfg["env"]["dexhand"],
            )
            self.demo_dataset_rh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="right",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand_rh,
                embodiment=self.cfg["env"]["dexhand"],
            )

        print(f"Start loading unique data for {len(self.dataIndices)} tasks...")
        unique_data_lh_list = []
        unique_data_rh_list = []
        for data_idx in self.dataIndices:
            dtype = ManipDataFactory.dataset_type(data_idx)
            unique_data_lh_list.append(self.demo_dataset_lh_dict[dtype][data_idx])
            unique_data_rh_list.append(self.demo_dataset_rh_dict[dtype][data_idx])

        print("Packing unique data to GPU...")
        packed_unique_lh = self.pack_data(unique_data_lh_list, side="lh")
        packed_unique_rh = self.pack_data(unique_data_rh_list, side="rh")

        env_to_data_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % len(self.dataIndices)

        print(f"Broadcasting data to {self.num_envs} environments (Vectorized)...")
        def broadcast_dict(packed_source, indices_tensor):
            new_data = {}
            indices_cpu = indices_tensor.cpu().numpy().tolist()
            for k, v in packed_source.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[indices_tensor].clone()
                elif k == "scene_objects":
                    # scene_objects 是 list，这里简单的浅拷贝 list 引用
                    new_data[k] = [v[i] for i in indices_cpu]
                elif isinstance(v, list):
                    new_data[k] = [v[i] for i in indices_cpu]
                else:
                    new_data[k] = [v[i] for i in indices_cpu]
            return new_data

        self.demo_data_lh = broadcast_dict(packed_unique_lh, env_to_data_indices)
        self.demo_data_rh = broadcast_dict(packed_unique_rh, env_to_data_indices)
        
        # === [新增] 将多物体轨迹预处理为 Tensor 以加速 Reward 计算 ===
        # 目标: 生成 self.rh_multi_obj_traj [NumEnvs, MaxObjs, T, 4, 4]
        def prepare_multi_obj_tensor(packed_unique, indices):
            # 1. 从 unique data 中提取轨迹 Tensor
            # packed_unique['scene_objects'] 是一个 list (length=UniqueBatch)
            # 每个元素是一个 list of dicts (objects)
            
            unique_trajs = [] # [UniqueBatch, NumObjs, T, 4, 4]
            
            scene_objs_batch = packed_unique["scene_objects"]
            # 假设同一个 Batch 内物体数量一致，取最大值或第一个
            max_objs = len(scene_objs_batch[0]) if len(scene_objs_batch) > 0 else 0
            
            for scene_objs in scene_objs_batch:
                objs_traj = []
                for i in range(max_objs):
                    if i < len(scene_objs):
                        # 取出轨迹 [T, 4, 4]（已在 Gym 坐标系，因为 process_data 中已转换）
                        t = scene_objs[i]["trajectory"]
                        # 注意：不需要再次转换，因为 ManipDataFactory.create_data 时传入了 mujoco2gym_transf
                    else:
                        # Padding
                        t = torch.eye(4, device=self.device).unsqueeze(0).repeat(packed_unique["seq_len"].max(), 1, 1)
                    objs_traj.append(t)
                
                if objs_traj:
                    unique_trajs.append(torch.stack(objs_traj))
                else:
                    # 空数据处理
                    dummy = torch.eye(4, device=self.device).view(1, 1, 4, 4)
                    unique_trajs.append(dummy)

            unique_trajs_stack = torch.stack(unique_trajs) # [UniqueBatch, NumObjs, T, 4, 4]
            
            # 2. Broadcast 到 NumEnvs
            return unique_trajs_stack[indices].clone()

        self.rh_multi_obj_traj = prepare_multi_obj_tensor(packed_unique_rh, env_to_data_indices)
        self.lh_multi_obj_traj = prepare_multi_obj_tensor(packed_unique_lh, env_to_data_indices)
        self.num_objs_per_env = self.rh_multi_obj_traj.shape[1] # 记录物体数量

        print("Data loading finished.")

        dexhand_rh_asset_file = self.dexhand_rh.urdf_path
        dexhand_lh_asset_file = self.dexhand_lh.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_rh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_rh_asset_file), asset_options)
        dexhand_lh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_lh_asset_file), asset_options)
        dexhand_rh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_rh_dof_damping = torch.tensor(
            [30] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_damping = torch.tensor(
            [30] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        asset_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }
        self.limit_info["lh"] = {
            "lower": np.asarray(asset_lh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_lh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_rh_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_rh_asset, rigid_shape_rh_props_asset)

        rigid_shape_lh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_lh_asset)
        for element in rigid_shape_lh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_lh_asset, rigid_shape_lh_props_asset)

        self.num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        self.num_dexhand_rh_dofs = self.gym.get_asset_dof_count(dexhand_rh_asset)
        self.num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        self.num_dexhand_lh_dofs = self.gym.get_asset_dof_count(dexhand_lh_asset)

        print(f"Num dexhand_r Bodies: {self.num_dexhand_rh_bodies}")
        print(f"Num dexhand_r DOFs: {self.num_dexhand_rh_dofs}")
        print(f"Num dexhand_l Bodies: {self.num_dexhand_lh_bodies}")
        print(f"Num dexhand_l DOFs: {self.num_dexhand_lh_dofs}")

        dexhand_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        self.dexhand_rh_dof_lower_limits = []
        self.dexhand_rh_dof_upper_limits = []
        self._dexhand_rh_effort_limits = []
        self._dexhand_rh_dof_speed_limits = []
        for i in range(self.num_dexhand_rh_dofs):
            dexhand_rh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_rh_dof_props["stiffness"][i] = dexhand_rh_dof_stiffness[i]
            dexhand_rh_dof_props["damping"][i] = dexhand_rh_dof_damping[i]

            self.dexhand_rh_dof_lower_limits.append(dexhand_rh_dof_props["lower"][i])
            self.dexhand_rh_dof_upper_limits.append(dexhand_rh_dof_props["upper"][i])
            self._dexhand_rh_effort_limits.append(dexhand_rh_dof_props["effort"][i])
            self._dexhand_rh_dof_speed_limits.append(dexhand_rh_dof_props["velocity"][i])

        self.dexhand_rh_dof_lower_limits = torch.tensor(self.dexhand_rh_dof_lower_limits, device=self.sim_device)
        self.dexhand_rh_dof_upper_limits = torch.tensor(self.dexhand_rh_dof_upper_limits, device=self.sim_device)
        self._dexhand_rh_effort_limits = torch.tensor(self._dexhand_rh_effort_limits, device=self.sim_device)
        self._dexhand_rh_dof_speed_limits = torch.tensor(self._dexhand_rh_dof_speed_limits, device=self.sim_device)

        # set dexhand_l dof properties
        dexhand_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.dexhand_lh_dof_lower_limits = []
        self.dexhand_lh_dof_upper_limits = []
        self._dexhand_lh_effort_limits = []
        self._dexhand_lh_dof_speed_limits = []
        for i in range(self.num_dexhand_lh_dofs):
            dexhand_lh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_lh_dof_props["stiffness"][i] = dexhand_lh_dof_stiffness[i]
            dexhand_lh_dof_props["damping"][i] = dexhand_lh_dof_damping[i]

            self.dexhand_lh_dof_lower_limits.append(dexhand_lh_dof_props["lower"][i])
            self.dexhand_lh_dof_upper_limits.append(dexhand_lh_dof_props["upper"][i])
            self._dexhand_lh_effort_limits.append(dexhand_lh_dof_props["effort"][i])
            self._dexhand_lh_dof_speed_limits.append(dexhand_lh_dof_props["velocity"][i])

        self.dexhand_lh_dof_lower_limits = torch.tensor(self.dexhand_lh_dof_lower_limits, device=self.sim_device)
        self.dexhand_lh_dof_upper_limits = torch.tensor(self.dexhand_lh_dof_upper_limits, device=self.sim_device)
        self._dexhand_lh_effort_limits = torch.tensor(self._dexhand_lh_effort_limits, device=self.sim_device)
        self._dexhand_lh_dof_speed_limits = torch.tensor(self._dexhand_lh_dof_speed_limits, device=self.sim_device)

        # compute aggregate size
        num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        num_dexhand_rh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_rh_asset)
        num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        num_dexhand_lh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_lh_asset)

        # 初始化容器
        self.dexhand_rs = []
        self.dexhand_ls = []
        self.envs = []
        self.objs_handles_rh = []
        self.objs_handles_lh = []
        
        # 缓存 Asset (防止重复加载)
        self.objs_assets = {} 

        # [新增] 用于收集 Mass 和 CoM 的临时列表 (List of Lists)
        raw_rh_masses = []
        raw_rh_coms = []
        raw_lh_masses = []
        raw_lh_coms = []
        # [新增] 用于收集静态物体信息的临时列表
        raw_rh_is_static = []
        raw_lh_is_static = []

        num_per_row = int(np.sqrt(self.num_envs))

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)

            # [迁移] 如果需要 Camera，在这里添加
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(env=env_ptr, isaac_gym=self.gym)
                )

            # Create Robot Actors
            dexhand_rh_actor = self.gym.create_actor(
                env_ptr, dexhand_rh_asset, self.dexhand_rh_pose, "dexhand_r", i,
                (1 if self.dexhand_rh.self_collision else 0),
            )
            dexhand_lh_actor = self.gym.create_actor(
                env_ptr, dexhand_lh_asset, self.dexhand_lh_pose, "dexhand_l", i,
                (1 if self.dexhand_lh.self_collision else 0),
            )
            # Set Props
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_rh_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_lh_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_rh_actor, dexhand_rh_dof_props) # 请取消注释并确保变量存在
            self.gym.set_actor_dof_properties(env_ptr, dexhand_lh_actor, dexhand_lh_dof_props) # 请取消注释并确保变量存在

            # === [修改] 创建桌子 (仅非 Humoto) ===
            if not is_humoto:
                table_pose = gymapi.Transform()
                table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
                table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
                table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
                table_props[0].friction = 0.1
                self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
                self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            # === [修改] 创建多物体 (Multi-Object Creation) ===
            # RH
            env_objs_rh, env_mass_rh, env_com_rh, env_is_static_rh = self._create_scene_objects(env_ptr, i, side="rh")
            self.objs_handles_rh.append(env_objs_rh)
            raw_rh_masses.append(env_mass_rh)
            raw_rh_coms.append(env_com_rh)
            raw_rh_is_static.append(env_is_static_rh)

            # LH
            if self._scene_objects_shared(i):
                env_objs_lh, env_mass_lh, env_com_lh, env_is_static_lh = env_objs_rh, env_mass_rh, env_com_rh, env_is_static_rh
            else:
                env_objs_lh, env_mass_lh, env_com_lh, env_is_static_lh = self._create_scene_objects(env_ptr, i, side="lh")
            self.objs_handles_lh.append(env_objs_lh)
            raw_lh_masses.append(env_mass_lh)
            raw_lh_coms.append(env_com_lh)
            raw_lh_is_static.append(env_is_static_lh)

            if self.aggregate_mode > 0:
                # 简单处理，暂不 end_aggregate 以防数量计算错误
                pass

            self.dexhand_rs.append(dexhand_rh_actor)
            self.dexhand_ls.append(dexhand_lh_actor)

            # ... (保留 Visualization Spheres 代码) ...

        # =================== DEBUG 代码开始 ===================
        import sys
        print(f"\n{'='*20} [DEBUG] Data Profiling {'='*20}")
        
        # 1. 模拟取出第一个样本 (模拟 segment_data 的行为)
        try:
            debug_idx = self.dataIndices[0]
            debug_dataset_type = ManipDataFactory.dataset_type(debug_idx)
            # 获取原始样本数据
            sample_data = self.demo_dataset_lh_dict[debug_dataset_type][debug_idx]
            
            print(f"Inspecting Sample ID: {debug_idx}")
            print(f"{'Key':<30} | {'Type':<15} | {'Device':<10} | {'Shape/Len':<20} | {'Size (MB)':<10}")
            print("-" * 100)

            total_mb = 0
            for k, v in sample_data.items():
                obj_type = type(v).__name__
                device_str = "CPU"
                shape_str = "N/A"
                size_mb = 0.0
                
                # 分析 Tensor
                if isinstance(v, torch.Tensor):
                    device_str = str(v.device)
                    shape_str = str(list(v.shape))
                    size_mb = v.element_size() * v.nelement() / (1024 * 1024)
                
                # 分析 Numpy
                elif isinstance(v, np.ndarray):
                    obj_type = "Numpy"
                    shape_str = str(v.shape)
                    size_mb = v.nbytes / (1024 * 1024)
                
                # 分析 List
                elif isinstance(v, list):
                    shape_str = f"len={len(v)}"
                    # 粗略估算 list 大小
                    size_mb = sys.getsizeof(v) / (1024 * 1024) 
                    if len(v) > 0:
                        if isinstance(v[0], torch.Tensor):
                            obj_type = "List[Tensor]"
                            device_str = str(v[0].device) # 检查 list 里的 tensor 在哪
                        elif isinstance(v[0], str):
                            obj_type = "List[Str]"

                # 其他
                else:
                    size_mb = sys.getsizeof(v) / (1024 * 1024)

                total_mb += size_mb
                
                # 高亮潜在问题
                highlight = ""
                if "cpu" in device_str.lower() and isinstance(v, torch.Tensor):
                    highlight = " <--- 慢! CPU Tensor"
                if "List" in obj_type:
                    highlight = " <--- 慢! Python List"

                print(f"{k:<30} | {obj_type:<15} | {device_str:<10} | {shape_str:<20} | {size_mb:<10.4f}{highlight}")

            print("-" * 100)
            print(f"单个样本大小: {total_mb:.4f} MB")
            print(f"如果循环复制 {self.num_envs} 次，总数据量: {(total_mb * self.num_envs):.2f} MB")
            print(f"如果是广播 (Broadcasting)，总数据量: {total_mb:.2f} MB (几乎不增加显存)")
            print("="*60 + "\n")
        except Exception as e:
            print(f"[DEBUG] Profiling failed: {e}")
        # =================== DEBUG 代码结束 ===================

        def pad_and_stack(data_list, pad_value=0.0):
            # 找出最大物体数量
            max_len = max([len(x) for x in data_list]) if data_list else 0
            padded_list = []
            for item in data_list:
                # 还要补多少个
                pad_len = max_len - len(item)
                if isinstance(item[0] if item else 0, list): # 如果是 CoM (list of list)
                    # 补 [0,0,0]
                    padded_item = item + [[pad_value]*3] * pad_len
                else: # 如果是 Mass (list of float) 或 is_static (list of bool)
                    padded_item = item + [pad_value] * pad_len
                padded_list.append(padded_item)
            return torch.tensor(padded_list, device=self.device, dtype=torch.float32)
        
        def pad_and_stack_bool(data_list, pad_value=False):
            # 专门处理布尔值列表
            max_len = max([len(x) for x in data_list]) if data_list else 0
            padded_list = []
            for item in data_list:
                pad_len = max_len - len(item)
                padded_item = item + [pad_value] * pad_len
                padded_list.append(padded_item)
            return torch.tensor(padded_list, device=self.device, dtype=torch.bool)

        self.manip_obj_rh_mass = pad_and_stack(raw_rh_masses)      # Shape: [NumEnvs, MaxObjs]
        self.manip_obj_rh_com = pad_and_stack(raw_rh_coms)         # Shape: [NumEnvs, MaxObjs, 3]
        self.manip_obj_rh_is_static = pad_and_stack_bool(raw_rh_is_static)  # Shape: [NumEnvs, MaxObjs]
        
        self.manip_obj_lh_mass = pad_and_stack(raw_lh_masses)
        self.manip_obj_lh_com = pad_and_stack(raw_lh_coms)
        self.manip_obj_lh_is_static = pad_and_stack_bool(raw_lh_is_static)  # Shape: [NumEnvs, MaxObjs]

        print(f"Object Mass Shape: {self.manip_obj_rh_mass.shape}") # Debug 确认形状

        # Setup data
        self.init_data()


    def _create_scene_objects(self, env_ptr, i, side="rh"):
        """
        为指定环境创建所有场景物体
        返回: 
            handles: list[int] (Actor handles)
            masses: list[float] (每个物体的质量)
            coms: list[list[float]] (每个物体的质心 [x,y,z])
            is_static_list: list[bool] (每个物体是否静态)
        """
        demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        scene_objs_list = demo_data["scene_objects"][i]
        
        handles = []
        masses = []  
        coms = []
        is_static_list = []  # 新增：记录静态物体信息
        
        for k, obj_info in enumerate(scene_objs_list):
            obj_name = obj_info['name']
            urdf_path = obj_info['urdf']
            
            # 0. 检查物体是否静态（轨迹中所有帧的pose是否相同）
            traj = obj_info["trajectory"]  # [T, 4, 4]
            if isinstance(traj, np.ndarray):
                traj_tensor = torch.tensor(traj, device=self.device, dtype=torch.float32)
            elif isinstance(traj, torch.Tensor):
                traj_tensor = traj.to(self.device).to(torch.float32)
            else:
                traj_tensor = torch.tensor(traj, device=self.device, dtype=torch.float32)
            
            # 判断物体是否静态：检查所有帧的变换矩阵是否相同（允许小的数值误差）
            is_static = False
            if traj_tensor.shape[0] > 1:
                first_frame = traj_tensor[0:1]  # [1, 4, 4]
                all_frames = traj_tensor  # [T, 4, 4]
                # 计算所有帧与第一帧的差异
                diff = torch.abs(all_frames - first_frame)  # [T, 4, 4]
                max_diff = torch.max(diff).item()
                # 如果最大差异小于阈值（1e-4，考虑浮点数误差），认为是静态物体
                is_static = max_diff < 1e-4
            else:
                # 只有一帧，认为是静态
                is_static = True
            
            # 打印调试信息（仅在第一个环境打印）
            if i == 0:
                print(f"[INFO] Object '{obj_name}' (env {i}, side={side}): is_static={is_static}")
            
            # 1. 加载 Asset (带缓存)
            # 注意：如果同一个urdf既有静态又有动态实例，缓存可能会有问题
            # 这里假设同一个urdf的所有实例都是同一种类型（静态或动态）
            asset_cache_key = f"{urdf_path}_static_{is_static}"
            if asset_cache_key in self.objs_assets:
                asset = self.objs_assets[asset_cache_key]
            else:
                asset_options = gymapi.AssetOptions()
                asset_options.override_com = True
                asset_options.override_inertia = True
                asset_options.vhacd_enabled = True
                asset_options.vhacd_params = gymapi.VhacdParams()
                asset_options.vhacd_params.resolution = 200000
                asset_options.density = 200
                asset_options.fix_base_link = is_static  # 静态物体固定base link
                # asset_options.disable_gravity = True 
                
                asset = self.gym.load_asset(self.sim, *os.path.split(urdf_path), asset_options)
                self.objs_assets[asset_cache_key] = asset
            
            # 2. 设置初始 Pose（轨迹已是 Gym 坐标系）
            # 注意：traj_tensor 已经在上面计算过了
            traj_first_frame = traj_tensor[0]
            if isinstance(traj_first_frame, np.ndarray):
                traj_first_frame = torch.tensor(traj_first_frame, device=self.device, dtype=torch.float32)
            elif isinstance(traj_first_frame, torch.Tensor):
                traj_first_frame = traj_first_frame.to(self.device).to(torch.float32)
            init_transf = traj_first_frame.cpu().numpy()
            # init_transf = (self.mujoco2gym_transf @ traj_first_frame).cpu().numpy()

            

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(init_transf[0, 3], init_transf[1, 3], init_transf[2, 3])
            # rotmat_to_quat 返回 [w, x, y, z]，但 gymapi.Quat 期望 (x, y, z, w)
            q = rotmat_to_quat(torch.tensor(init_transf[:3, :3]).unsqueeze(0))[0]
            q_xyzw = q[[1, 2, 3, 0]]  # 转换为 [x, y, z, w] 格式
            pose.r = gymapi.Quat(q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3])

            # 3. 创建 Actor
            handle = self.gym.create_actor(env_ptr, asset, pose, f"{obj_name}_{side}_{k}", i, 0)
            
            # === [修改重点] 4. 分别设置质量(Body)和摩擦力(Shape) ===
            
            # 4.1 设置质量 (Rigid Body Properties)
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
            
            original_mass = body_props[0].mass
            new_mass = max(0.1, min(0.5, original_mass)) # 限制质量范围
            
            body_props[0].mass = new_mass
            # 注意：friction 不能在这里设置！
            self.gym.set_actor_rigid_body_properties(env_ptr, handle, body_props)
            
            # 4.2 设置摩擦力 (Rigid Shape Properties)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            # 通常一个 actor 可能有多个 shape，这里假设统一设置
            for shape_prop in shape_props:
                shape_prop.friction = 2.0
                shape_prop.rolling_friction = 0.05 # 推荐加一点防止无限滚动
                shape_prop.torsion_friction = 0.05 
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, shape_props)
            
            # 5. 记录数据
            handles.append(handle)
            masses.append(new_mass)
            # 记录质心 (使用 body_props)
            coms.append([body_props[0].com.x, body_props[0].com.y, body_props[0].com.z])
            is_static_list.append(is_static)  # 新增：记录静态物体信息
            
        return handles, masses, coms, is_static_list

    def _scene_objects_shared(self, env_idx: int) -> bool:
        """Return True if LH scene definition matches RH for given env."""
        if "scene_objects" not in self.demo_data_rh or "scene_objects" not in self.demo_data_lh:
            return False

        rh_scene = self.demo_data_rh["scene_objects"][env_idx]
        lh_scene = self.demo_data_lh["scene_objects"][env_idx]

        if len(rh_scene) != len(lh_scene):
            return False

        for rh_obj, lh_obj in zip(rh_scene, lh_scene):
            if rh_obj.get("name") != lh_obj.get("name"):
                return False
            if rh_obj.get("urdf") != lh_obj.get("urdf"):
                return False
        return True
    

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_rh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_r")
        dexhand_lh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_l")
        self.dexhand_rh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_rh_handle, k) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_lh_handle, k) for k in self.dexhand_lh.body_names
        }
        self.dexhand_rh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_lh.body_names
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._rh_base_state = self._root_state[:, 0, :]
        self._lh_base_state = self._root_state[:, 1, :]

        # ? >>> for visualization
        if not self.headless:

            self.mano_joint_rh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"rh_mano_joint_{i}"), :]
                for i in range(self.dexhand_rh.n_bodies)
            ]
            self.mano_joint_lh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"lh_mano_joint_{i}"), :]
                for i in range(self.dexhand_lh.n_bodies)
            ]
        # ? <<<

        self._manip_obj_rh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_rh")
        self._manip_obj_rh_root_state = self._root_state[:, self._manip_obj_rh_handle, :]
        self._manip_obj_lh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_lh")
        self._manip_obj_lh_root_state = self._root_state[:, self._manip_obj_lh_handle, :]
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_rh_handle, "base"
        )
        self._manip_obj_lh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_lh_handle, "base"
        )
        self._manip_obj_rh_cf = self.net_cf[:, self._manip_obj_rh_rigid_body_handle, :]
        self._manip_obj_lh_cf = self.net_cf[:, self._manip_obj_lh_rigid_body_handle, :]

        self.dexhand_rh_root_state = self._root_state[:, dexhand_rh_handle, :]
        self.dexhand_lh_root_state = self._root_state[:, dexhand_lh_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.rh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_dexhand_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_r", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_dexhand_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_l", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        # === [修改部分开始] 获取多物体的全局索引 ===
        # 使用 self.objs_handles_rh/lh (在 _create_envs 中填充的 list of lists)
        
        def get_actor_indices(handles_list_of_lists):
            indices = []
            # 找出最大物体数量（用于 padding）
            max_objs = max([len(handles) for handles in handles_list_of_lists]) if handles_list_of_lists else 0
            
            for i in range(self.num_envs):
                env_indices = []
                # 遍历当前环境的所有物体 handle
                for handle in handles_list_of_lists[i]:
                    # 获取该 handle 在 Simulation 中的全局索引
                    idx = self.gym.get_actor_index(self.envs[i], handle, gymapi.DOMAIN_SIM)
                    env_indices.append(idx)
                
                # Padding: 如果该环境的物体数量不足，用 -1 填充（无效索引）
                while len(env_indices) < max_objs:
                    env_indices.append(-1)
                
                indices.append(env_indices)
            return torch.tensor(indices, dtype=torch.int32, device=self.sim_device)

        # 这里的 shape 将会是 [NumEnvs, NumObjs]
        self._global_manip_obj_rh_indices = get_actor_indices(self.objs_handles_rh)
        self._global_manip_obj_lh_indices = get_actor_indices(self.objs_handles_lh)
        # === [修改部分结束] ===

        CONTACT_HISTORY_LEN = 3
        self.rh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()
        self.lh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()

    def pack_data(self, data, side="rh"):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        
        # ... (保留 fill_data 内部函数不变) ...
        def fill_data(stack_data):
            # ... (保留原有的 fill_data 逻辑，记得去掉 squeeze) ...
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat([
                        stack_data[i],
                        stack_data[i][-1].unsqueeze(0).repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]])
                    ], dim=0)
            return torch.stack(stack_data)

        for k in data[0].keys():

            # ←←←← 新增开始：统一处理所有手部/腕部状态（解决你现在的报错） ←←←←
            hand_keys = [
                "wrist_pos", "wrist_rot", "wrist_velocity", "wrist_angular_velocity",
                "opt_dof_pos", "opt_dof_velocity", "tips_distance",
                "obj_velocity", "obj_angular_velocity",
                "opt_wrist_pos", "opt_wrist_rot",
                "opt_wrist_velocity", "opt_wrist_angular_velocity"
            ]
            if k in hand_keys:
                stack_data = [d[k] for d in data]
                packed_data[k] = fill_data(stack_data)
                continue
            # ←←←← 新增结束

            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    if side == "rh":
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_rh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_rh.body_names
                                    if self.dexhand_rh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                    else:
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_lh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_lh.body_names
                                    if self.dexhand_lh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                packed_data[k] = fill_data(mano_joints)
            
            # === [核心修改] ===
            elif k == "scene_objects":
                # 直接保存列表，不做 tensor 转换
                packed_data[k] = [d[k] for d in data]
            # ================
            
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data) # 记得去掉 squeeze
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]
        return packed_data

    def allocate_buffers(self):
        # ensure privileged observation space matches dynamic object count before allocation
        self._resize_privileged_obs_space()
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _resize_privileged_obs_space(self):
        """Dynamically adjust privileged obs dimension based on runtime object count."""
        if len(self._privileged_obs_keys) == 0:
            return

        if not hasattr(self, "num_objs_per_env") or self.num_objs_per_env is None:
            return

        new_dim = self._infer_privileged_obs_dim()
        if new_dim == self.privileged_obs_dim:
            return

        self.privileged_obs_dim = new_dim
        self.cfg["env"]["privilegedObsDim"] = new_dim

        if isinstance(self.obs_space, spaces.Dict) and "privileged" in self.obs_space.spaces:
            self.obs_space.spaces["privileged"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(new_dim,),
            )

    def _infer_privileged_obs_dim(self):
        """Compute total privileged observation dimension across both hands."""
        dim_rh = self._compute_side_privileged_dim("rh")
        dim_lh = self._compute_side_privileged_dim("lh")
        return dim_rh + dim_lh

    def _compute_side_privileged_dim(self, side: str) -> int:
        """Compute privileged obs size contributed by one hand."""
        dim = 0
        num_objs = int(self.num_objs_per_env) if self.num_objs_per_env is not None else 0
        num_dofs = self.num_dexhand_rh_dofs if side == "rh" else self.num_dexhand_lh_dofs
        num_tips = len(self.dexhand_rh.contact_body_names if side == "rh" else self.dexhand_lh.contact_body_names)

        for key in self._privileged_obs_keys:
            if key == "dq":
                dim += num_dofs
            elif key == "manip_obj_pos":
                dim += num_objs * 3
            elif key == "manip_obj_quat":
                dim += num_objs * 4
            elif key == "manip_obj_vel":
                dim += num_objs * 3
            elif key == "manip_obj_ang_vel":
                dim += num_objs * 3
            elif key == "manip_obj_com":
                dim += num_objs * 3
            elif key == "manip_obj_weight":
                dim += num_objs
            elif key == "tip_force":
                dim += num_tips * 4  # xyz + magnitude
            else:
                # fallback to current config for keys that do not depend on object count
                # assumes their contribution is unchanged from the existing total
                continue

        return dim

    def _configure_target_obs_space(self):
        """Align target observation space with the current multi-object layout."""
        if not hasattr(self, "num_objs_per_env") or self.num_objs_per_env is None:
            return

        target_dim_per_side = self._compute_target_obs_dim_per_side()
        self.target_obs_dim = target_dim = target_dim_per_side * 2

        self.obs_dict["target"] = torch.zeros((self.num_envs, target_dim), device=self.device)

        obs_space = dict(self.obs_space.spaces)
        obs_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(target_dim,))
        self.obs_space = spaces.Dict(obs_space)

    def _compute_target_obs_dim_per_side(self) -> int:
        """Compute per-side target observation length."""
        future = self.obs_future_length
        num_objs = int(self.num_objs_per_env)
        num_joints = self.dexhand_rh.n_bodies - 1
        num_bodies = self.dexhand_rh.n_bodies
        tips_dim = self.demo_data_rh["tips_distance"].shape[-1]
        bps_dim = self.obj_bps_rh.shape[-1]

        wrist_terms = 3 + 3 + 3 + 4 + 4 + 3 + 3
        joint_terms = num_joints * 3
        obj_vec3 = num_objs * 3
        obj_vec4 = num_objs * 4
        obj_to_joint = num_objs * num_bodies

        per_future = (
            wrist_terms
            + joint_terms * 3  # delta, vel, delta_vel
            + obj_vec3 * 5  # pos/vel/delta_vel + ang vel/delta_ang_vel
            + obj_vec4 * 2  # quats and deltas
            + obj_to_joint
            + tips_dim
        )

        return bps_dim + future * per_future

    def _update_states(self):
        self.rh_states.update(
            {
                "q": self._q[:, : self.num_dexhand_rh_dofs],
                "cos_q": torch.cos(self._q[:, : self.num_dexhand_rh_dofs]),
                "sin_q": torch.sin(self._q[:, : self.num_dexhand_rh_dofs]),
                "dq": self._qd[:, : self.num_dexhand_rh_dofs],
                "base_state": self._rh_base_state[:, :],
            }
        )

        self.rh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_rh_handles[k], :][:, :10] for k in self.dexhand_rh.body_names],
            dim=1,
        )

        # === [修改 2] 抓取多物体的状态 ===
        # 目标: [NumEnvs, NumObjs, 3] 或 [NumEnvs, NumObjs, 4]
        
        # 1. 获取所有物体的 Root State [NumEnvs, NumObjs, 13]
        # self._root_state 是一维的 [TotalActors, 13] 或者 [NumEnvs, MaxActors, 13]
        # 这里最稳妥的是利用 indices 进行 gather
        
        # flatten indices 用于索引: [NumEnvs * NumObjs]
        flat_rh_indices = self._global_manip_obj_rh_indices.flatten().long()
        flat_lh_indices = self._global_manip_obj_lh_indices.flatten().long()
        
        # 从全局 _root_state (view成一维处理比较方便) 中取出
        # 注意: self._root_state 在 init_data 里被 view 成了 [NumEnvs, -1, 13]
        # 但它的原始 tensor 是 flat 的。为避免 shape 问题，我们用 gymtorch 的索引操作或者直接 gather
        # 假设 self._root_state 已经是 [NumEnvs, TotalActorsPerEnv, 13]，那 indices 必须是局部的。
        # 但我们上面获取的是 global indices。
        
        # 更简单的方法：直接用 gym 提供的 API 或者直接从 flat tensor 索引
        # 这里我们假设 self._root_state 包含了所有数据，我们重新 reshape 一下临时变量来索引
        root_state_flat = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state_flat = gymtorch.wrap_tensor(root_state_flat) # [TotalActors, 13]

        rh_obj_states = root_state_flat[flat_rh_indices].view(self.num_envs, self.num_objs_per_env, 13)
        lh_obj_states = root_state_flat[flat_lh_indices].view(self.num_envs, self.num_objs_per_env, 13)


        self.rh_states.update(
            {
                "manip_obj_pos": rh_obj_states[..., :3],      # [N, K, 3]
                "manip_obj_quat": rh_obj_states[..., 3:7],    # [N, K, 4]
                "manip_obj_vel": rh_obj_states[..., 7:10],    # [N, K, 3]
                "manip_obj_ang_vel": rh_obj_states[..., 10:], # [N, K, 3]
            }
        )

        self.lh_states.update(
            {
                "q": self._q[:, self.num_dexhand_rh_dofs :],
                "cos_q": torch.cos(self._q[:, self.num_dexhand_rh_dofs :]),
                "sin_q": torch.sin(self._q[:, self.num_dexhand_rh_dofs :]),
                "dq": self._qd[:, self.num_dexhand_rh_dofs :],
                "base_state": self._lh_base_state[:, :],
            }
        )
        self.lh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_lh_handles[k], :][:, :10] for k in self.dexhand_lh.body_names],
            dim=1,
        )
        self.lh_states.update(
            {
                "manip_obj_pos": lh_obj_states[..., :3],
                "manip_obj_quat": lh_obj_states[..., 3:7],
                "manip_obj_vel": lh_obj_states[..., 7:10],
                "manip_obj_ang_vel": lh_obj_states[..., 10:],
            }
        )

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        lh_rew_buf, lh_reset_buf, lh_success_buf, lh_failure_buf, lh_reward_dict, lh_error_buf = (
            self.compute_reward_side(actions, side="lh")
        )
        rh_rew_buf, rh_reset_buf, rh_success_buf, rh_failure_buf, rh_reward_dict, rh_error_buf = (
            self.compute_reward_side(actions, side="rh")
        )
        self.rew_buf = rh_rew_buf + lh_rew_buf
        self.reset_buf = rh_reset_buf | lh_reset_buf
        self.success_buf = rh_success_buf & lh_success_buf
        self.failure_buf = rh_failure_buf | lh_failure_buf
        self.error_buf = rh_error_buf | lh_error_buf
        self.reward_dict = {
            **{"rh_" + k: v for k, v in rh_reward_dict.items()},
            **{"lh_" + k: v for k, v in lh_reward_dict.items()},
        }


    def compute_reward_side(self, actions, side="rh"):
        side_demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        # 使用我们预处理好的多物体轨迹 Tensor
        multi_obj_traj = self.rh_multi_obj_traj if side == "rh" else self.lh_multi_obj_traj
        
        target_state = {}
        max_length = torch.clip(side_demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf

        # --- [修复核心] 机器人相关 Reward 数据提取：强制指定形状 ---
        
        # 1. Wrist Position: 强制转为 (N, 3)
        cur_wrist_pos = side_demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_pos"] = cur_wrist_pos.view(self.num_envs, 3) 

        # 2. Wrist Rotation -> Quaternion: 确保输入 aa_to_quat 是 (N, 3)，输出是 (N, 4)
        cur_wrist_rot = side_demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot.view(self.num_envs, 3))[:, [1, 2, 3, 0]].view(self.num_envs, 4)

        # 3. Wrist Velocity: 强制转为 (N, 3)
        target_state["wrist_vel"] = side_demo_data["wrist_velocity"][torch.arange(self.num_envs), cur_idx].view(self.num_envs, 3)
        
        # 4. Wrist Angular Velocity: 强制转为 (N, 3)
        target_state["wrist_ang_vel"] = side_demo_data["wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx].view(self.num_envs, 3)
        
        # --- 下面的代码保持原样 ---
        
        target_state["tips_distance"] = side_demo_data["tips_distance"][torch.arange(self.num_envs), cur_idx]
        cur_joints_pos = side_demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = side_demo_data["mano_joints_velocity"][torch.arange(self.num_envs), cur_idx].reshape(self.num_envs, -1, 3)

        # --- [修改] 多物体 Reward 计算 ---
        # 1. 获取所有物体的目标状态 [NumEnvs, NumObjs, 4, 4]
        # cur_idx 需要 unsqueeze 才能 gather
        # multi_obj_traj: [N, K, T, 4, 4]
        # 我们需要取 T=cur_idx
        
        # 构造索引 [N, K]
        batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).repeat(1, self.num_objs_per_env)
        obj_idx = torch.arange(self.num_objs_per_env, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        time_idx = cur_idx.unsqueeze(1).repeat(1, self.num_objs_per_env)
        
        # 取出当前帧所有物体的 Target Pose [N, K, 4, 4]
        target_objs_pose = multi_obj_traj[batch_idx, obj_idx, time_idx]
        target_objs_pos = target_objs_pose[..., :3, 3] # [N, K, 3]
        
        # 2. 获取所有物体的当前状态 [N, K, 3]
        # 使用 _update_states 中已经正确获取的状态（通过 _global_manip_obj_rh_indices/lh_indices）
        side_states_temp = getattr(self, f"{side}_states")
        # 注意：这里需要先调用 _refresh 确保状态是最新的，但 _refresh 已经在 compute_reward 之前被调用了
        # 直接从 side_states 中获取已经正确索引的物体状态
        current_objs_pos = side_states_temp["manip_obj_pos"]  # [N, K, 3]
        current_objs_quat = side_states_temp["manip_obj_quat"]  # [N, K, 4]
        current_objs_vel = side_states_temp["manip_obj_vel"]  # [N, K, 3]
        current_objs_ang_vel = side_states_temp["manip_obj_ang_vel"]  # [N, K, 3]

        # 3. 计算误差 (平均误差)
        diff_objs = target_objs_pos - current_objs_pos
        dist_objs = torch.norm(diff_objs, dim=-1) # [N, K]
        mean_dist_objs = dist_objs.mean(dim=-1) # [N] -> 每个环境的平均物体误差

        # --- [Hack] 构造虚拟单物体状态传给 JIT ---
        # 我们把 "计算好的误差" 伪装成 "target=0, current=error" 传进去
        # 这样 JIT 里的 diff = 0 - error = -error, norm(diff) = error
        # 从而复用 JIT 的 exp(-k * dist) 公式
        
        target_state["manip_obj_pos"] = target_objs_pos # [N, K, 3]
        
        # 从 target_objs_pose 提取 quaternion [N, K, 4]
        target_objs_rotmat = target_objs_pose[..., :3, :3]  # [N, K, 3, 3]
        target_objs_quat = rotmat_to_quat(target_objs_rotmat.reshape(-1, 3, 3))  # [N*K, 4]
        target_objs_quat = target_objs_quat[:, [1, 2, 3, 0]]  # [w, x, y, z] -> [x, y, z, w]
        target_state["manip_obj_quat"] = target_objs_quat.reshape(self.num_envs, self.num_objs_per_env, 4)
        
        # 从 scene_objects 中读取每个物体的速度和角速度
        target_objs_vel_list = []
        target_objs_ang_vel_list = []
        for env_id in range(self.num_envs):
            env_vels = []
            env_ang_vels = []
            scene_objs = side_demo_data["scene_objects"][env_id]
            for k in range(self.num_objs_per_env):
                if k < len(scene_objs):
                    scene_obj = scene_objs[k]
                    # 获取速度
                    if 'velocity' in scene_obj:
                        vel = scene_obj['velocity']
                        if isinstance(vel, torch.Tensor):
                            env_vels.append(vel[cur_idx[env_id]])  # [3]
                        else:
                            env_vels.append(torch.tensor(vel[cur_idx[env_id]], device=self.device, dtype=torch.float32))
                    else:
                        # 向后兼容：使用第一个物体的速度
                        env_vels.append(side_demo_data["obj_velocity"][env_id, cur_idx[env_id]])
                    
                    # 获取角速度
                    if 'angular_velocity' in scene_obj:
                        ang_vel = scene_obj['angular_velocity']
                        if isinstance(ang_vel, torch.Tensor):
                            env_ang_vels.append(ang_vel[cur_idx[env_id]])  # [3]
                        else:
                            env_ang_vels.append(torch.tensor(ang_vel[cur_idx[env_id]], device=self.device, dtype=torch.float32))
                    else:
                        # 向后兼容：使用第一个物体的角速度
                        env_ang_vels.append(side_demo_data["obj_angular_velocity"][env_id, cur_idx[env_id]])
                else:
                    # Padding: 如果物体数量不足，使用零
                    env_vels.append(torch.zeros(3, device=self.device, dtype=torch.float32))
                    env_ang_vels.append(torch.zeros(3, device=self.device, dtype=torch.float32))
            target_objs_vel_list.append(torch.stack(env_vels))  # [K, 3]
            target_objs_ang_vel_list.append(torch.stack(env_ang_vels))  # [K, 3]
        
        target_state["manip_obj_vel"] = torch.stack(target_objs_vel_list)  # [N, K, 3]
        target_state["manip_obj_ang_vel"] = torch.stack(target_objs_ang_vel_list)  # [N, K, 3]
        
        # hack_current_vel = current_objs_vel.mean(dim=1)
        # hack_current_ang_vel = current_objs_ang_vel.mean(dim=1)

        # 4. 其他接触力等 (保持不变)
        target_state["tip_force"] = torch.stack(
            [
                self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                for k in (self.dexhand_rh.contact_body_names if side == "rh" else self.dexhand_lh.contact_body_names)
            ],
            axis=1,
        )
        setattr(
            self,
            f"{side}_tips_contact_history",
            torch.concat(
                [
                    getattr(self, f"{side}_tips_contact_history")[:, 1:],
                    (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
                ],
                dim=1,
            ),
        )
        target_state["tip_contact_state"] = getattr(self, f"{side}_tips_contact_history")

        side_states = getattr(self, f"{side}_states")
        # [Hack] 临时替换 side_states 里的物体位置，欺骗 JIT 函数
        # 注意：这里只影响 JIT 计算，不影响物理引擎
        original_obj_pos = side_states["manip_obj_pos"]
        original_obj_quat = side_states["manip_obj_quat"]
        original_obj_vel = side_states["manip_obj_vel"]
        original_obj_ang_vel = side_states["manip_obj_ang_vel"]
        
        # 使用真实的多物体状态（已经在上面从 side_states_temp 中获取了正确的状态）
        side_states["manip_obj_pos"] = current_objs_pos # [N, K, 3]
        side_states["manip_obj_quat"] = current_objs_quat # [N, K, 4]
        side_states["manip_obj_vel"] = current_objs_vel # [N, K, 3]
        side_states["manip_obj_ang_vel"] = current_objs_ang_vel # [N, K, 3]

        if side == "rh":
            power = torch.abs(torch.multiply(self.dof_force[:, : self.dexhand_rh.n_dofs], side_states["dq"])).sum(
                dim=-1
            )
        else:
            power = torch.abs(torch.multiply(self.dof_force[:, self.dexhand_rh.n_dofs :], side_states["dq"])).sum(
                dim=-1
            )
        target_state["power"] = power

        base_handle = getattr(self, f"dexhand_{side}_handles")[
            self.dexhand_rh.to_dex("wrist")[0] if side == "rh" else self.dexhand_lh.to_dex("wrist")[0]
        ]

        wrist_power = torch.abs(
            torch.sum(
                self.apply_forces[:, base_handle, :] * side_states["base_state"][:, 7:10],
                dim=-1,
            )
        ) 
        wrist_power += torch.abs(
            torch.sum(
                self.apply_torque[:, base_handle, :] * side_states["base_state"][:, 10:],
                dim=-1,
            )
        )
        target_state["wrist_power"] = wrist_power

        if self.training:
            last_step = self.gym.get_frame_count(self.sim)
            # === [新增] 估算当前 epoch 数 ===
            # 每个 epoch 的 frames = horizon_length * num_envs
            # 从配置中获取 horizon_length（默认32，如果无法获取则使用估算值）
            horizon_length = getattr(self, 'horizon_length', 32)  # 默认32，从ResDexHandPPO.yaml
            frames_per_epoch = horizon_length * self.num_envs
            # 估算 epoch（向下取整）
            estimated_epoch = int(last_step // frames_per_epoch) if frames_per_epoch > 0 else 0
            # 如果 total_train_env_frames 可用，使用它来更准确地估算
            if hasattr(self, 'total_train_env_frames') and self.total_train_env_frames is not None:
                estimated_epoch = int(self.total_train_env_frames // frames_per_epoch) if frames_per_epoch > 0 else 0
            
            if self.tighten_method == "None":
                scale_factor = 1.0
            elif self.tighten_method == "const":
                scale_factor = self.tighten_factor
            elif self.tighten_method == "adaptive":
                # === [新增] 自适应难度模式：使用动态调整的scale_factor ===
                scale_factor = self.adaptive_global_scale_factor
            elif self.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                scale_factor = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                raise NotImplementedError(f"Unknown tighten_method: {self.tighten_method}")
            
            # === [新增] 更新当前scale_factor用于tensorboard记录 ===
            # 只在第一次调用时更新（因为每个side都会调用，我们只需要记录一次）
            if side == "rh":  # 只在一个side更新，避免重复
                self.current_scale_factor = scale_factor
        else:
            scale_factor = 1.0
            estimated_epoch = 0
            frames_per_epoch = 0
            last_step = 0
            # 非训练模式，保持current_scale_factor不变或设置为1.0
            if not hasattr(self, 'current_scale_factor'):
                self.current_scale_factor = 1.0

        # assert not self.headless or isinstance(compute_imitation_reward, torch.jit.ScriptFunction)

        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)

        # 获取静态物体信息
        obj_is_static = getattr(self, f"manip_obj_{side}_is_static")  # [N, K]
        
        rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            side_states,
            target_state,
            max_length,
            scale_factor,
            (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx,
            obj_is_static,  # 新增：静态物体信息
        )
        
        # [Hack 结束] 恢复原始状态，以免影响其他逻辑
        side_states["manip_obj_pos"] = original_obj_pos
        side_states["manip_obj_quat"] = original_obj_quat
        side_states["manip_obj_vel"] = original_obj_vel
        side_states["manip_obj_ang_vel"] = original_obj_ang_vel
        
        # === [新增] 打印失败原因诊断并保存到文件 ===
        if failure_buf.any():
            failed_env_ids = failure_buf.nonzero(as_tuple=False).flatten().cpu().numpy()
            for env_id in failed_env_ids:
                env_id_item = int(env_id)
                cur_step = self.progress_buf[env_id_item].item()
                running_steps = self.running_progress_buf[env_id_item].item()
                
                # 重新计算各个失败条件（用于诊断）
                # 获取静态物体信息
                obj_is_static_env = getattr(self, f"manip_obj_{side}_is_static")[env_id_item]  # [K]
                
                # 1. 物体位置误差（排除静态物体）
                dist_objs_env = dist_objs[env_id_item]  # [K]
                if dist_objs_env.dim() > 0:
                    # 排除静态物体
                    dynamic_mask = ~obj_is_static_env
                    if dynamic_mask.any():
                        obj_pos_err = dist_objs_env[dynamic_mask].max().item()
                    else:
                        obj_pos_err = 0.0  # 所有物体都是静态的
                else:
                    obj_pos_err = dist_objs_env.item()
                obj_pos_threshold = 0.12 / 0.343 * scale_factor**3  # 调整：0.08 → 0.15（基于实际误差分析：平均0.148m，95%分位数0.22m）
                obj_pos_failed = obj_pos_err > obj_pos_threshold
                
                # 2. 物体旋转误差（修复180°问题：使用 quat_to_angle_axis，与 JIT 函数一致，排除静态物体）
                # 注意：quat_mul 和 quat_conjugate 期望 (x,y,z,w) 格式，quat_to_angle_axis 也期望 (x,y,z,w) 格式
                target_obj_quat_env = target_state["manip_obj_quat"][env_id_item]  # [K, 4] (x,y,z,w)
                current_obj_quat_env = current_objs_quat[env_id_item]  # [K, 4] (x,y,z,w)
                # 计算旋转差（与 JIT 函数中的逻辑完全一致）
                diff_rot_quat = quat_mul(target_obj_quat_env, quat_conjugate(current_obj_quat_env))  # [K, 4] (x,y,z,w)
                # 使用 quat_to_angle_axis 计算角度（与 JIT 函数一致，返回角度已归一化到 [-pi, pi]）
                diff_rot_angle, _ = quat_to_angle_axis(diff_rot_quat)  # [K] (已归一化到 [-pi, pi])
                # 取绝对值并转换为度数
                diff_rot_angle_deg = torch.abs(diff_rot_angle) / np.pi * 180  # [K]
                # 排除静态物体
                if diff_rot_angle_deg.numel() > 0:
                    dynamic_mask = ~obj_is_static_env
                    if dynamic_mask.any():
                        obj_rot_err = diff_rot_angle_deg[dynamic_mask].max().item()
                    else:
                        obj_rot_err = 0.0  # 所有物体都是静态的
                else:
                    obj_rot_err = 0.0
                # 上调阈值：90° → 180°（基于实际误差分析：平均151°，95%分位数178°）
                obj_rot_threshold = 180 / 0.343 * scale_factor**3
                obj_rot_failed = obj_rot_err > obj_rot_threshold
                
                # 3. 手指位置误差
                joints_pos = side_states["joints_state"][env_id_item, 1:, :3]  # [J, 3]
                target_joints_pos = target_state["joints_pos"][env_id_item]  # [J, 3]
                diff_joints_pos = target_joints_pos - joints_pos
                diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)
                
                dexhand = self.dexhand_rh if side == "rh" else self.dexhand_lh
                thumb_tip_dist = diff_joints_pos_dist[[k - 1 for k in dexhand.weight_idx["thumb_tip"]]].mean().item()
                index_tip_dist = diff_joints_pos_dist[[k - 1 for k in dexhand.weight_idx["index_tip"]]].mean().item()
                middle_tip_dist = diff_joints_pos_dist[[k - 1 for k in dexhand.weight_idx["middle_tip"]]].mean().item()
                pinky_tip_dist = diff_joints_pos_dist[[k - 1 for k in dexhand.weight_idx["pinky_tip"]]].mean().item()
                ring_tip_dist = diff_joints_pos_dist[[k - 1 for k in dexhand.weight_idx["ring_tip"]]].mean().item()
                level_1_dist = diff_joints_pos_dist[[k - 1 for k in dexhand.weight_idx["level_1_joints"]]].mean().item()
                level_2_dist = diff_joints_pos_dist[[k - 1 for k in dexhand.weight_idx["level_2_joints"]]].mean().item()
                
                # 手指位置误差阈值：基于实际误差分析，提升50-60%（95%分位数）
                thumb_threshold = 0.18 / 0.7 * scale_factor  # 0.1125 → 0.18（提升60%，95%分位数0.183m）
                index_threshold = 0.20 / 0.7 * scale_factor  # 0.12375 → 0.20（提升62%，95%分位数0.200m）
                middle_threshold = 0.18 / 0.7 * scale_factor  # 0.1125 → 0.18（提升60%，95%分位数0.179m）
                pinky_threshold = 0.23 / 0.7 * scale_factor  # 0.1575 → 0.23（提升46%，95%分位数0.233m）
                ring_threshold = 0.22 / 0.7 * scale_factor  # 0.135 → 0.22（提升63%，95%分位数0.215m）
                level_1_threshold = 0.22 / 0.7 * scale_factor  # 0.1575 → 0.22（提升40%，95%分位数0.220m）
                level_2_threshold = 0.25 / 0.7 * scale_factor  # 0.18 → 0.25（提升39%，基于Level 1的调整）
                
                thumb_failed = thumb_tip_dist > thumb_threshold
                index_failed = index_tip_dist > index_threshold
                middle_failed = middle_tip_dist > middle_threshold
                pinky_failed = pinky_tip_dist > pinky_threshold
                ring_failed = ring_tip_dist > ring_threshold
                level_1_failed = level_1_dist > level_1_threshold
                level_2_failed = level_2_dist > level_2_threshold
                
                # 4. 接触违规（完全按照 JIT 函数中的逻辑实现）
                # JIT 函数中：tip_contact_state 是 [N, 3, 5] 或 [N, 5]，finger_tip_distance 是 [N, 5]
                # 这里对于单个环境：tip_contact_state[env_id_item] 是 [3, 5]，finger_tip_distance[env_id_item] 是 [5]
                finger_tip_distance = target_state["tips_distance"][env_id_item]  # [5]
                tip_contact_state = target_state["tip_contact_state"][env_id_item]  # [3, 5] (CONTACT_HISTORY_LEN=3, 5 fingers)
                tip_contact_state_bool = tip_contact_state.to(torch.bool)
                
                # 完全按照 JIT 函数的逻辑（第2919-2927行）：
                # 对于 [3, 5]，相当于 JIT 中的 dim >= 3 情况，但这里是单个环境，所以用 dim=0 代替 dim=1
                if tip_contact_state_bool.dim() >= 2:
                    # [3, 5] -> [5] (对历史维度做 any，相当于 JIT 中的 dim=1，但这里是 dim=0)
                    tip_contact_active = torch.any(tip_contact_state_bool, dim=0)
                elif tip_contact_state_bool.dim() == 1:
                    # [5] 直接使用（相当于 JIT 中的 dim == 2 情况）
                    tip_contact_active = tip_contact_state_bool
                else:
                    # 其他情况（相当于 JIT 中的 else 分支）
                    tip_contact_active = tip_contact_state_bool.unsqueeze(-1)
                
                # 确保维度匹配（相当于 JIT 函数第2926-2927行的逻辑）
                if tip_contact_active.shape[-1] == 1 and finger_tip_distance.shape[-1] != 1:
                    tip_contact_active = tip_contact_active.repeat(finger_tip_distance.shape[-1])
                
                # 计算接触违规（相当于 JIT 函数第2928-2930行，但这里是单个环境，所以直接 .item()）
                contact_violation = torch.any(
                    (finger_tip_distance < 0.005) & ~tip_contact_active
                ).item()
                
                # 5. 速度异常
                eef_vel_norm = torch.norm(side_states["base_state"][env_id_item, 7:10]).item()
                eef_ang_vel_norm = torch.norm(side_states["base_state"][env_id_item, 10:13]).item()
                joints_vel_norm = torch.norm(side_states["joints_state"][env_id_item, 1:, 7:10], dim=-1).mean().item()
                dof_vel_norm = torch.abs(side_states["dq"][env_id_item]).mean().item()
                obj_vel_norm = torch.norm(current_objs_vel[env_id_item], dim=-1).max().item() if current_objs_vel.dim() > 1 else torch.norm(current_objs_vel[env_id_item]).item()
                obj_ang_vel_norm = torch.norm(current_objs_ang_vel[env_id_item], dim=-1).max().item() if current_objs_ang_vel.dim() > 1 else torch.norm(current_objs_ang_vel[env_id_item]).item()
                
                error_eef_vel = eef_vel_norm > 100
                error_eef_ang_vel = eef_ang_vel_norm > 200
                error_joints_vel = joints_vel_norm > 100
                error_dof_vel = dof_vel_norm > 200
                error_obj_vel = obj_vel_norm > 100
                error_obj_ang_vel = obj_ang_vel_norm > 200
                
                # 收集失败原因
                failure_reasons = []
                
                if obj_pos_failed:
                    failure_reasons.append(f"  - 物体位置误差过大: {obj_pos_err:.4f} > {obj_pos_threshold:.4f} m")
                if obj_rot_failed:
                    failure_reasons.append(f"  - 物体旋转误差过大: {obj_rot_err:.2f}° > {obj_rot_threshold:.2f}°")
                if thumb_failed:
                    failure_reasons.append(f"  - 拇指位置误差过大: {thumb_tip_dist:.4f} > {thumb_threshold:.4f} m")
                if index_failed:
                    failure_reasons.append(f"  - 食指位置误差过大: {index_tip_dist:.4f} > {index_threshold:.4f} m")
                if middle_failed:
                    failure_reasons.append(f"  - 中指位置误差过大: {middle_tip_dist:.4f} > {middle_threshold:.4f} m")
                if pinky_failed:
                    failure_reasons.append(f"  - 小指位置误差过大: {pinky_tip_dist:.4f} > {pinky_threshold:.4f} m")
                if ring_failed:
                    failure_reasons.append(f"  - 无名指位置误差过大: {ring_tip_dist:.4f} > {ring_threshold:.4f} m")
                if level_1_failed:
                    failure_reasons.append(f"  - Level 1 关节位置误差过大: {level_1_dist:.4f} > {level_1_threshold:.4f} m")
                if level_2_failed:
                    failure_reasons.append(f"  - Level 2 关节位置误差过大: {level_2_dist:.4f} > {level_2_threshold:.4f} m")
                if contact_violation:
                    failure_reasons.append(f"  - 接触惩罚: 手指距离过近但未检测到接触")
                if error_eef_vel:
                    failure_reasons.append(f"  - 手腕线速度异常: {eef_vel_norm:.2f} > 100 m/s")
                if error_eef_ang_vel:
                    failure_reasons.append(f"  - 手腕角速度异常: {eef_ang_vel_norm:.2f} > 200 rad/s")
                if error_joints_vel:
                    failure_reasons.append(f"  - 关节速度异常: {joints_vel_norm:.2f} > 100 m/s")
                if error_dof_vel:
                    failure_reasons.append(f"  - DOF速度异常: {dof_vel_norm:.2f} > 200 rad/s")
                if error_obj_vel:
                    failure_reasons.append(f"  - 物体线速度异常: {obj_vel_norm:.2f} > 100 m/s")
                if error_obj_ang_vel:
                    failure_reasons.append(f"  - 物体角速度异常: {obj_ang_vel_norm:.2f} > 200 rad/s")
                
                # 打印到控制台
                # print(f"\n[FAILURE DIAGNOSIS] Env {env_id_item} ({side.upper()} side), Step {cur_step}, Running {running_steps} steps:")
                # if failure_reasons:
                #     for reason in failure_reasons:
                #         print(reason)
                # else:
                #     print(f"  - 未知原因 (可能是 running_progress_buf < 8)")
                # print(f"  Scale Factor: {scale_factor:.4f}")
                
                # 保存到文件
                if hasattr(self, 'failure_log_file') and self.failure_log_file is not None:
                    try:
                        with open(self.failure_log_file, "a", encoding="utf-8") as f:
                            f.write(f"\n[FAILURE DIAGNOSIS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Env {env_id_item} ({side.upper()} side), Step {cur_step}, Running {running_steps} steps\n")
                            f.write(f"  Estimated Epoch: {estimated_epoch} (Frame: {last_step}, Frames/Epoch: {frames_per_epoch})\n")  # 新增：epoch信息
                            if failure_reasons:
                                for reason in failure_reasons:
                                    f.write(reason + "\n")
                            else:
                                f.write(f"  - 未知原因 (可能是 running_progress_buf < 8)\n")
                            f.write(f"  Scale Factor: {scale_factor:.4f}\n")
                            f.write("-" * 80 + "\n")
                    except Exception as e:
                        print(f"[WARNING] Failed to write to failure log file: {e}")
        
        self.total_rew_buf += rew_buf
        return rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf


    def compute_observations(self):
        self._refresh()
        obs_rh = self.compute_observations_side("rh")
        obs_lh = self.compute_observations_side("lh")
        for k in obs_rh.keys():
            self.obs_dict[k] = torch.cat([obs_rh[k], obs_lh[k]], dim=-1)

    def compute_observations_side(self, side="rh"):
        # obs_keys: q, cos_q, sin_q, base_state
        side_states = getattr(self, f"{side}_states")
        side_demo_data = getattr(self, f"demo_data_{side}")

        obs_dict = {}

        obs_values = []
        for ob in self._obs_keys:
            if ob == "base_state":
                obs_values.append(
                    torch.cat([torch.zeros_like(side_states[ob][:, :3]), side_states[ob][:, 3:]], dim=-1)
                )  # ! ignore base position
            else:
                obs_values.append(side_states[ob])
        obs_dict["proprioception"] = torch.cat(obs_values, dim=-1)


        # privileged_obs_keys: dq, manip_obj_pos, manip_obj_quat, manip_obj_vel, manip_obj_ang_vel
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    # pos: [N, K, 3], base: [N, 3] -> [N, 1, 3]
                    # 结果: [N, K, 3] -> Flatten -> [N, K*3]
                    rel_pos = side_states[ob] - side_states["base_state"][:, :3].unsqueeze(1)
                    pri_obs_values.append(rel_pos.reshape(self.num_envs, -1))
                    
                elif ob == "manip_obj_com":
                    # === [重点修改] 多物体 CoM 计算 ===
                    # Quat: [N, K, 4] -> Rot: [N, K, 3, 3]
                    # CoM_local: [N, K, 3]
                    
                    # 1. 准备数据并 Reshape 方便计算
                    batch_size = self.num_envs
                    num_objs = self.num_objs_per_env
                    
                    # 将 Batch 和 Obj 维度合并，因为 quat_to_rotmat 通常处理 [..., 4]
                    flat_quat = side_states["manip_obj_quat"].reshape(-1, 4) # [N*K, 4]
                    flat_rot = quat_to_rotmat(flat_quat[:, [1, 2, 3, 0]])    # [N*K, 3, 3]
                    
                    flat_com_local = getattr(self, f"manip_obj_{side}_com").reshape(-1, 3) # [N*K, 3]
                    
                    # 2. 旋转 CoM (Batch Matrix Multiply)
                    # [N*K, 3, 3] @ [N*K, 3, 1] -> [N*K, 3, 1]
                    flat_com_world_offset = torch.bmm(flat_rot, flat_com_local.unsqueeze(-1)).squeeze(-1)
                    
                    # 3. 加上物体位置
                    flat_obj_pos = side_states["manip_obj_pos"].reshape(-1, 3) # [N*K, 3]
                    flat_com_world = flat_com_world_offset + flat_obj_pos
                    
                    # 4. 减去手腕位置 (变为相对坐标)
                    # 手腕位置只有 N 个，需要 repeat 到 N*K
                    base_pos = side_states["base_state"][:, :3].repeat_interleave(num_objs, dim=0) # [N*K, 3]
                    
                    flat_rel_com = flat_com_world - base_pos
                    
                    # 5. 变回 [N, K*3] 添加到观测
                    pri_obs_values.append(flat_rel_com.reshape(self.num_envs, -1))
                    
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    # Mass: [N, K] -> [N, K*1]
                    weights = getattr(self, f"manip_obj_{side}_mass") * -1 * prop.gravity.z
                    pri_obs_values.append(weights.reshape(self.num_envs, -1))
                    
                elif ob == "tip_force":
                    # Tip force 处理保持不变，因为它是关于手指的，和物体数量无关
                    tip_force = torch.stack(
                        [
                            self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                            for k in (
                                self.dexhand_rh.contact_body_names
                                if side == "rh"
                                else self.dexhand_lh.contact_body_names
                            )
                        ],
                        axis=1,
                    )
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )  # add force magnitude
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                else:
                    # 对于其他属性 (如 manip_obj_vel, manip_obj_quat)，只要它们是 [N, K, ...]
                    # 我们都统一 Flatten 处理
                    val = side_states[ob]
                    if val.dim() > 2: # 如果是多维 [N, K, D]
                        val = val.reshape(self.num_envs, -1)
                    pri_obs_values.append(val)
                    
            obs_dict["privileged"] = torch.cat(pri_obs_values, dim=-1)

        next_target_state = {}

        cur_idx = self.progress_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(side_demo_data["seq_len"]), side_demo_data["seq_len"] - 1)

        cur_idx = torch.stack(
            [cur_idx + t for t in range(self.obs_future_length)], dim=-1
        )  # [B, K], K = obs_future_length
        nE, nT = side_demo_data["wrist_pos"].shape[:2]
        nF = self.obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        target_wrist_pos = indicing(side_demo_data["wrist_pos"], cur_idx)  # [B, K, 3]
        cur_wrist_pos = side_states["base_state"][:, :3]  # [B, 3]
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(side_demo_data["wrist_velocity"], cur_idx)
        cur_wrist_vel = side_states["base_state"][:, 7:10]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(nE, -1)
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(side_demo_data["wrist_rot"], cur_idx)
        cur_wrist_rot = side_states["base_state"][:, 3:7]

        next_target_state["wrist_quat"] = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(nE, -1)
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(nE, -1)

        target_wrist_ang_vel = indicing(side_demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = side_states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(nE, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        target_joints_pos = indicing(side_demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_pos = side_states["joints_state"][:, 1:, :3]  # skip the base joint
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(side_demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_vel = side_states["joints_state"][:, 1:, 7:10]  # skip the base joint
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)


        ##################################

        # 1. 准备多物体轨迹数据 (N, K, T, 4, 4)
        raw_multi_traj = self.rh_multi_obj_traj if side == "rh" else self.lh_multi_obj_traj
        # Transpose: [N, K, T, 4, 4] -> [N, T, K, 4, 4] 以适配 indicing
        multi_traj_time_first = raw_multi_traj.transpose(1, 2)
        
        # 2. 获取未来帧的目标位姿 [N, F, K, 4, 4]
        target_obj_transf = indicing(multi_traj_time_first, cur_idx)
        
        # 3. 计算位置 Delta
        target_pos = target_obj_transf[..., :3, 3] # [N, F, K, 3]
        # Current: [N, K, 3] -> [N, 1, K, 3] (广播)
        current_pos = side_states["manip_obj_pos"][:, None, :, :]
        next_target_state["delta_manip_obj_pos"] = (target_pos - current_pos).reshape(nE, -1)

        # 4. 计算速度 Delta - 从 scene_objects 中读取每个物体的速度
        # 构造目标速度 tensor [N, F, K, 3]
        target_obj_vel_list = []
        for env_id in range(nE):
            env_vels_future = []
            scene_objs = side_demo_data["scene_objects"][env_id]
            env_cur_idx = cur_idx[env_id]  # [F]
            for k in range(self.num_objs_per_env):
                if k < len(scene_objs):
                    scene_obj = scene_objs[k]
                    if 'velocity' in scene_obj:
                        vel = scene_obj['velocity']
                        if isinstance(vel, torch.Tensor):
                            # vel: [T, 3], env_cur_idx: [F] -> [F, 3]
                            vel_future = vel[env_cur_idx]  # [F, 3]
                        else:
                            vel_tensor = torch.tensor(vel, device=self.device, dtype=torch.float32)
                            vel_future = vel_tensor[env_cur_idx]
                    else:
                        # 向后兼容：使用第一个物体的速度
                        vel_future = side_demo_data["obj_velocity"][env_id, env_cur_idx]  # [F, 3]
                else:
                    # Padding: 如果物体数量不足，使用零
                    vel_future = torch.zeros((nF, 3), device=self.device, dtype=torch.float32)
                env_vels_future.append(vel_future)  # [F, 3]
            target_obj_vel_list.append(torch.stack(env_vels_future, dim=1))  # [F, K, 3]
        target_obj_vel = torch.stack(target_obj_vel_list)  # [N, F, K, 3]
            
        current_vel = side_states["manip_obj_vel"][:, None, :, :]
        next_target_state["manip_obj_vel"] = target_obj_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_vel"] = (target_obj_vel - current_vel).reshape(nE, -1)

        # 5. 计算旋转 Delta (多物体)
        target_rotmat = target_obj_transf[..., :3, :3] # [N, F, K, 3, 3]
        _shape_prefix = target_rotmat.shape[:-2] 
        target_quat = rotmat_to_quat(target_rotmat.reshape(-1, 3, 3)) 
        target_quat = target_quat[:, [1, 2, 3, 0]].reshape(*_shape_prefix, 4) # [N, F, K, 4]
        next_target_state["manip_obj_quat"] = target_quat.reshape(nE, -1)
        
        # Current Quat: [N, K, 4] -> [N, 1, K, 4] -> repeat -> [N, F, K, 4]
        current_quat = side_states["manip_obj_quat"][:, None, :, :].repeat(1, nF, 1, 1)
        
        # Flatten 计算 quaternion multiplication
        flat_target_quat = target_quat.reshape(-1, 4)
        flat_current_quat = current_quat.reshape(-1, 4)
        flat_delta_quat = quat_mul(flat_current_quat, quat_conjugate(flat_target_quat))
        
        next_target_state["delta_manip_obj_quat"] = flat_delta_quat.reshape(nE, -1)

        # 6. 计算角速度 Delta - 从 scene_objects 中读取每个物体的角速度
        # 构造目标角速度 tensor [N, F, K, 3]
        target_obj_ang_vel_list = []
        for env_id in range(nE):
            env_ang_vels_future = []
            scene_objs = side_demo_data["scene_objects"][env_id]
            env_cur_idx = cur_idx[env_id]  # [F]
            for k in range(self.num_objs_per_env):
                if k < len(scene_objs):
                    scene_obj = scene_objs[k]
                    if 'angular_velocity' in scene_obj:
                        ang_vel = scene_obj['angular_velocity']
                        if isinstance(ang_vel, torch.Tensor):
                            # ang_vel: [T, 3], env_cur_idx: [F] -> [F, 3]
                            ang_vel_future = ang_vel[env_cur_idx]  # [F, 3]
                        else:
                            ang_vel_tensor = torch.tensor(ang_vel, device=self.device, dtype=torch.float32)
                            ang_vel_future = ang_vel_tensor[env_cur_idx]
                    else:
                        # 向后兼容：使用第一个物体的角速度
                        ang_vel_future = side_demo_data["obj_angular_velocity"][env_id, env_cur_idx]  # [F, 3]
                else:
                    # Padding: 如果物体数量不足，使用零
                    ang_vel_future = torch.zeros((nF, 3), device=self.device, dtype=torch.float32)
                env_ang_vels_future.append(ang_vel_future)  # [F, 3]
            target_obj_ang_vel_list.append(torch.stack(env_ang_vels_future, dim=1))  # [F, K, 3]
        target_obj_ang_vel = torch.stack(target_obj_ang_vel_list)  # [N, F, K, 3]

        current_ang_vel = side_states["manip_obj_ang_vel"][:, None, :, :]
        next_target_state["manip_obj_ang_vel"] = target_obj_ang_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_ang_vel"] = (target_obj_ang_vel - current_ang_vel).reshape(nE, -1)

        # 7. 物体到关节的距离 (All Objs to All Joints)
        # Joints: [N, 1, J, 3] - Objs: [N, K, 1, 3] -> [N, K, J, 3] -> Norm -> [N, K*J]
        next_target_state["obj_to_joints"] = torch.norm(
            side_states["manip_obj_pos"][:, :, None, :] - side_states["joints_state"][:, None, :, :3], 
            dim=-1
        ).reshape(self.num_envs, -1)

        #############################

        next_target_state["gt_tips_distance"] = indicing(side_demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        next_target_state["bps"] = getattr(self, f"obj_bps_{side}")
        obs_dict["target"] = torch.cat(
            [
                next_target_state[ob]
                for ob in [  # ! must be in the same order as the following
                    "delta_wrist_pos",
                    "wrist_vel",
                    "delta_wrist_vel",
                    "wrist_quat",
                    "delta_wrist_quat",
                    "wrist_ang_vel",
                    "delta_wrist_ang_vel",
                    "delta_joints_pos",
                    "joints_vel",
                    "delta_joints_vel",
                    "delta_manip_obj_pos",
                    "manip_obj_vel",
                    "delta_manip_obj_vel",
                    "manip_obj_quat",
                    "delta_manip_obj_quat",
                    "manip_obj_ang_vel",
                    "delta_manip_obj_ang_vel",
                    "obj_to_joints",
                    "gt_tips_distance",
                    "bps",
                ]
            ],
            dim=-1,
        )

        if not self.training:
            manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")
            dexhand_handles = getattr(self, f"dexhand_{side}_handles")
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_manip_obj_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "state_manip_obj_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "joint_state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_rh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "joint_state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_lh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "q_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "q_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "dq_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "dq_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "tip_force_rh" and side == "rh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_rh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "tip_force_lh" and side == "lh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_lh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "reward":
                    self.dump_fileds[prop_name][:] = self.rew_buf.reshape(self.num_envs, -1).detach()
                else:
                    pass
        return obs_dict

    def _reset_default(self, env_ids):
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.98 * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data_rh["seq_len"][env_ids] * 0.98).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data_rh["seq_len"][env_ids]
                    * 0.98
                    * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                ).long()
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data_rh["seq_len"][env_ids].long())
            else:
                seq_idx = torch.zeros_like(self.demo_data_rh["seq_len"][env_ids].long())

        self._reset_default_side(env_ids, seq_idx, side="lh")
        self._reset_default_side(env_ids, seq_idx, side="rh")

        # === [打印] 所有物体的初始位置 ===
        # 获取所有物体的位置信息
        demo_data_rh = self.demo_data_rh
        demo_data_lh = self.demo_data_lh

        dexhand_multi_env_ids_int32 = torch.concat(
            [
                self._global_dexhand_rh_indices[env_ids].flatten(),
                self._global_dexhand_lh_indices[env_ids].flatten(),
            ]
        )
        manip_obj_multi_env_ids_int32 = torch.concat(
            [self._global_manip_obj_rh_indices[env_ids].flatten(), self._global_manip_obj_lh_indices[env_ids].flatten()]
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
            len(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        # 刷新状态以获取模拟器中的实际位置
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 获取扁平化的 root state（全局索引）
        root_state_flat = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state_flat = gymtorch.wrap_tensor(root_state_flat)  # [TotalActors, 13]

        # === [打印] 从模拟器 API 获取所有物体的实际位置和方向 ===
        demo_data_rh = self.demo_data_rh
        demo_data_lh = self.demo_data_lh
        
        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0

        if self.use_pid_control:
            self.rh_prev_pos_error[env_ids] = 0
            self.rh_prev_rot_error[env_ids] = 0
            self.rh_pos_error_integral[env_ids] = 0
            self.rh_rot_error_integral[env_ids] = 0
            self.lh_prev_pos_error[env_ids] = 0
            self.lh_prev_rot_error[env_ids] = 0
            self.lh_pos_error_integral[env_ids] = 0
            self.lh_rot_error_integral[env_ids] = 0

        self.lh_tips_contact_history[env_ids] = torch.ones_like(self.lh_tips_contact_history[env_ids]).bool()
        self.rh_tips_contact_history[env_ids] = torch.ones_like(self.rh_tips_contact_history[env_ids]).bool()

    def _reset_default_side(self, env_ids, seq_idx, side="rh"):

        side_demo_data = getattr(self, f"demo_data_{side}")

        dof_pos = side_demo_data["opt_dof_pos"][env_ids, seq_idx]
        dof_pos = torch_jit_utils.tensor_clamp(
            dof_pos,
            getattr(self, f"dexhand_{side}_dof_lower_limits").unsqueeze(0),
            getattr(self, f"dexhand_{side}_dof_upper_limits").unsqueeze(0),
        )
        dof_vel = side_demo_data["opt_dof_velocity"][env_ids, seq_idx]
        dof_vel = torch_jit_utils.tensor_clamp(
            dof_vel,
            -1 * getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
            getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
        )

        opt_wrist_pos = side_demo_data["opt_wrist_pos"][env_ids, seq_idx]
        opt_wrist_rot = aa_to_quat(side_demo_data["opt_wrist_rot"][env_ids, seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = side_demo_data["opt_wrist_velocity"][env_ids, seq_idx]
        opt_wrist_ang_vel = side_demo_data["opt_wrist_angular_velocity"][env_ids, seq_idx]

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)

        getattr(self, f"_{side}_base_state")[env_ids, :] = opt_hand_pose_vel

        if side == "rh":
            self._q[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
            self._qd[env_ids, : self.num_dexhand_rh_dofs] = dof_vel
            self._pos_control[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
        else:
            self._q[env_ids, self.num_dexhand_rh_dofs :] = dof_pos
            self._qd[env_ids, self.num_dexhand_rh_dofs :] = dof_vel
            self._pos_control[env_ids, self.num_dexhand_rh_dofs :] = dof_pos

        # === [修改] 多物体位置重置 ===
        # 参考 origin_dexhandmanip_bih.py 的方式，但适配多物体
        demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        
        # 获取物体数量（使用统一的 num_objs_per_env，已考虑 padding）
        num_objs = self.num_objs_per_env
        
        # 批量处理所有物体
        for k in range(num_objs):
            # 1. 收集这一批环境 (env_ids) 中第 k 个物体在当前帧 (seq_idx) 的轨迹
            obj_trajs = []
            obj_vels = []
            obj_ang_vels = []
            
            for idx, (env_id, s_idx) in enumerate(zip(env_ids, seq_idx)):
                env_id_item = env_id.item()
                s_idx_item = s_idx.item()
                
                # 从 scene_objects 中获取第 k 个物体的数据
                # 检查边界：如果该环境的物体数量不足，跳过（使用 padding 的零值）
                if env_id_item >= len(demo_data["scene_objects"]) or k >= len(demo_data["scene_objects"][env_id_item]):
                    # Padding: 使用零值
                    obj_trajs.append(torch.eye(4, device=self.device, dtype=torch.float32))
                    obj_vels.append(torch.zeros(3, device=self.device, dtype=torch.float32))
                    obj_ang_vels.append(torch.zeros(3, device=self.device, dtype=torch.float32))
                    continue
                
                scene_obj = demo_data["scene_objects"][env_id_item][k]
                
                # 获取轨迹 [T, 4, 4] - 已经是转换后的坐标系
                traj = scene_obj['trajectory']
                if isinstance(traj, torch.Tensor):
                    frame_pose = traj[s_idx_item]  # [4, 4]
                else:
                    frame_pose = torch.tensor(traj[s_idx_item], device=self.device, dtype=torch.float32)
                
                obj_trajs.append(frame_pose)
                
                # 获取速度 - 从 scene_objects 中读取（在 base.py 中已计算）
                if 'velocity' in scene_obj:
                    vel = scene_obj['velocity']
                    if isinstance(vel, torch.Tensor):
                        obj_vel = vel[s_idx_item]  # [3]
                    else:
                        obj_vel = torch.tensor(vel[s_idx_item], device=self.device, dtype=torch.float32)
                else:
                    # 如果没有 velocity，使用第一个物体的速度（向后兼容）
                    obj_vel = side_demo_data["obj_velocity"][env_id_item, s_idx_item]
                
                obj_vels.append(obj_vel)
                
                # 获取角速度
                if 'angular_velocity' in scene_obj:
                    ang_vel = scene_obj['angular_velocity']
                    if isinstance(ang_vel, torch.Tensor):
                        obj_ang_vel = ang_vel[s_idx_item]  # [3]
                    else:
                        obj_ang_vel = torch.tensor(ang_vel[s_idx_item], device=self.device, dtype=torch.float32)
                else:
                    # 如果没有 angular_velocity，使用第一个物体的角速度（向后兼容）
                    obj_ang_vel = side_demo_data["obj_angular_velocity"][env_id_item, s_idx_item]
                
                obj_ang_vels.append(obj_ang_vel)
            
            # 2. 堆叠成 tensor
            obj_trajs = torch.stack(obj_trajs).to(self.device)  # [N_reset, 4, 4]
            obj_vels = torch.stack(obj_vels).to(self.device)  # [N_reset, 3]
            obj_ang_vels = torch.stack(obj_ang_vels).to(self.device)  # [N_reset, 3]
            
            # 3. 提取位置和旋转（轨迹已经是转换后的坐标系）
            obj_pos_init = obj_trajs[:, :3, 3]  # [N_reset, 3]
            obj_rot_init = obj_trajs[:, :3, :3]  # [N_reset, 3, 3]
            obj_rot_init = rotmat_to_quat(obj_rot_init)
            # [w, x, y, z] to [x, y, z, w]
            obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]  # [N_reset, 4]
            
            # 4. 批量更新 root_state
            # 使用 _global_manip_obj_rh_indices 或 _global_manip_obj_lh_indices 获取正确的局部索引
            # 注意：_root_state 的第二个维度是环境内的局部索引
            # 需要将全局索引转换为局部索引
            num_actors_per_env = self._root_state.shape[1]
            
            # 获取每个环境对应的物体局部索引
            for idx, env_id in enumerate(env_ids):
                env_id_item = env_id.item()
                # 获取该环境中第 k 个物体的全局索引
                if side == "rh":
                    obj_global_idx = self._global_manip_obj_rh_indices[env_id_item, k].item()
                else:
                    obj_global_idx = self._global_manip_obj_lh_indices[env_id_item, k].item()
                
                # 检查索引是否有效（-1 表示该环境没有第 k 个物体）
                if obj_global_idx < 0:
                    continue  # 跳过无效索引
                
                # 将全局索引转换为环境内的局部索引
                actor_local_idx = obj_global_idx % num_actors_per_env
                
                # 更新该环境中的物体状态
                self._root_state[env_id_item, actor_local_idx, :3] = obj_pos_init[idx]
                self._root_state[env_id_item, actor_local_idx, 3:7] = obj_rot_init[idx]
                self._root_state[env_id_item, actor_local_idx, 7:10] = obj_vels[idx]
                self._root_state[env_id_item, actor_local_idx, 10:13] = obj_ang_vels[idx]

    def _update_adaptive_difficulty(self, env_ids):
        """
        更新自适应难度：基于epoch记录表现并评估是否需要调整难度
        
        难度调整规则（基于epoch而非episode）：
        1. 如果连续N个epoch（默认5个）成功率超过阈值（默认20%）-> 提升难度（降低scale_factor，使阈值更严格）
        2. 如果连续M个epoch（默认5个）平均step低于阈值（默认20步）-> 降低难度（提升scale_factor，使阈值更宽松）
        3. 如果连续K个epoch（默认10个）没有提升难度 -> 降低难度（提升scale_factor，防止难度过高）
        
        注意：在每个epoch内累积所有reset的表现数据，当epoch变化时记录上一个epoch的平均值。
        
        Args:
            env_ids: 需要重置的环境ID列表（Tensor）
        """
        if self.tighten_method != "adaptive" or not self.training:
            return
        
        # 计算当前epoch
        last_step = self.gym.get_frame_count(self.sim)
        horizon_length = getattr(self, 'horizon_length', 32)
        frames_per_epoch = horizon_length * self.num_envs
        current_epoch = int(last_step // frames_per_epoch) if frames_per_epoch > 0 else 0
        
        # 如果total_train_env_frames可用，使用它来更准确地计算epoch
        if hasattr(self, 'total_train_env_frames') and self.total_train_env_frames is not None:
            current_epoch = int(self.total_train_env_frames // frames_per_epoch) if frames_per_epoch > 0 else 0
        
        # 初始化当前epoch（第一次调用）
        if self.adaptive_current_epoch < 0:
            self.adaptive_current_epoch = current_epoch
        
        # 检测epoch变化
        if current_epoch != self.adaptive_current_epoch:
            # Epoch发生了变化，记录上一个epoch的表现
            if self.adaptive_current_epoch_reset_count > 0:
                # 计算上一个epoch的成功率
                success_rate_last_epoch  = self.adaptive_current_epoch_success_count / self.adaptive_current_epoch_reset_count
                avg_steps_last_epoch = self.adaptive_current_epoch_step_sum / self.adaptive_current_epoch_reset_count
                
                # 更新用于tensorboard的当前值
                self.current_epoch_success_rate = success_rate_last_epoch
                self.current_epoch_avg_steps = avg_steps_last_epoch
                
                # 记录到历史
                self.adaptive_epoch_success_rate_history.append(success_rate_last_epoch)
                self.adaptive_epoch_steps_history.append(avg_steps_last_epoch)
                
                # 评估并调整难度
                old_scale = self.adaptive_global_scale_factor
                scale_changed = False
                
                # 检查1：连续N个epoch成功率是否超过阈值 -> 提升难度
                if len(self.adaptive_epoch_success_rate_history) >= self.adaptive_success_window:
                    recent_success_rates = list(self.adaptive_epoch_success_rate_history)
                    
                    # 检查最近N个epoch的成功率是否都超过阈值
                    if all(sr >= self.adaptive_success_threshold for sr in recent_success_rates):
                        # 提升难度：降低scale_factor（降低阈值，更严格）
                        new_scale = max(
                            self.adaptive_scale_factor_min,
                            self.adaptive_global_scale_factor - self.adaptive_scale_step
                        )
                        if new_scale < self.adaptive_global_scale_factor:
                            self.adaptive_global_scale_factor = new_scale
                            scale_changed = True
                            # 更新上次难度提升的epoch
                            self.adaptive_last_difficulty_increase_epoch = self.adaptive_current_epoch
                            avg_success_rate = sum(recent_success_rates) / len(recent_success_rates)
                            print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Increasing difficulty: "
                                  f"scale_factor {old_scale:.4f} -> {new_scale:.4f} "
                                  f"(success rate {avg_success_rate*100:.1f}% >= {self.adaptive_success_threshold*100:.1f}% for {len(recent_success_rates)} epochs)")
                
                # 检查2：连续M个epoch平均step是否低于阈值 -> 降低难度
                # if len(self.adaptive_epoch_steps_history) >= self.adaptive_step_window:
                #     recent_steps = list(self.adaptive_epoch_steps_history)
                #     avg_recent_steps = sum(recent_steps) / len(recent_steps)
                    
                #     if avg_recent_steps < self.adaptive_step_threshold:
                #         # 降低难度：提升scale_factor（提高阈值，更宽松）
                #         new_scale = min(
                #             self.adaptive_scale_factor_max,
                #             self.adaptive_global_scale_factor + self.adaptive_scale_step
                #         )
                #         if new_scale > self.adaptive_global_scale_factor:
                #             self.adaptive_global_scale_factor = new_scale
                #             scale_changed = True
                #             # 重新开始计时：降低难度后，从当前epoch开始重新计时
                #             self.adaptive_last_difficulty_increase_epoch = self.adaptive_current_epoch
                #             print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Decreasing difficulty: "
                #                   f"scale_factor {old_scale:.4f} -> {new_scale:.4f} "
                #                   f"(avg steps too low: {avg_recent_steps:.2f} < {self.adaptive_step_threshold})")
                
                # 检查3：连续N个epoch没有提升难度 -> 降低难度
                if not scale_changed:
                    if self.adaptive_last_difficulty_increase_epoch >= 0:
                        # 曾经提升过难度，检查是否超过N个epoch没有再次提升
                        epochs_since_last_increase = self.adaptive_current_epoch - self.adaptive_last_difficulty_increase_epoch
                        if epochs_since_last_increase >= self.adaptive_no_improvement_window:
                            # 降低难度：提升scale_factor（提高阈值，更宽松）
                            new_scale = min(
                                self.adaptive_scale_factor_max,
                                self.adaptive_global_scale_factor + self.adaptive_scale_step
                            )
                            if new_scale > self.adaptive_global_scale_factor:
                                self.adaptive_global_scale_factor = new_scale
                                scale_changed = True
                                print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Decreasing difficulty (no improvement): "
                                      f"scale_factor {old_scale:.4f} -> {new_scale:.4f} "
                                      f"(no difficulty increase for {epochs_since_last_increase} epochs)")
                                # 重新开始计时：从当前epoch开始，看看是否能在N个epoch内提升难度
                                self.adaptive_last_difficulty_increase_epoch = self.adaptive_current_epoch
                    else:
                        # 从未提升过难度，检查是否已经过了N个epoch（初始难度可能太高）
                        if self.adaptive_current_epoch >= self.adaptive_no_improvement_window:
                            # 降低难度：提升scale_factor（提高阈值，更宽松）
                            new_scale = min(
                                self.adaptive_scale_factor_max,
                                self.adaptive_global_scale_factor + self.adaptive_scale_step
                            )
                            if new_scale > self.adaptive_global_scale_factor:
                                self.adaptive_global_scale_factor = new_scale
                                scale_changed = True
                                print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Decreasing difficulty (no improvement from start): "
                                      f"scale_factor {old_scale:.4f} -> {new_scale:.4f} "
                                      f"(no difficulty increase after {self.adaptive_current_epoch} epochs)")
                                # 重新开始计时：从当前epoch开始，看看是否能在N个epoch内提升难度
                                self.adaptive_last_difficulty_increase_epoch = self.adaptive_current_epoch
                
                # 如果难度改变了，清空历史记录（避免频繁调整）
                if scale_changed:
                    self.adaptive_epoch_success_rate_history.clear()
                    self.adaptive_epoch_steps_history.clear()
            
            # 重置当前epoch的累积数据
            self.adaptive_current_epoch = current_epoch
            self.adaptive_current_epoch_success_count = 0
            self.adaptive_current_epoch_reset_count = 0
            self.adaptive_current_epoch_step_sum = 0.0
            self.adaptive_current_epoch_reward_sum = 0.0  # 保留用于兼容
            
            # 重置tensorboard统计信息（新epoch开始时，成功率暂时未知）
            if hasattr(self, 'current_epoch_success_rate'):
                self.current_epoch_success_rate = 0.0
                self.current_epoch_avg_steps = 0.0
        
        # 累积当前epoch的表现数据
        if len(env_ids) > 0:
            # 获取这些环境的success状态、reward和step
            episode_success = self.success_buf[env_ids].cpu().numpy()  # 是否成功
            episode_rewards = self.total_rew_buf[env_ids].cpu().numpy()  # 累计reward
            episode_steps = self.running_progress_buf[env_ids].cpu().numpy()  # 运行步数
            
            # 累积到当前epoch
            self.adaptive_current_epoch_success_count += int(episode_success.sum())  # 成功次数
            self.adaptive_current_epoch_reset_count += len(env_ids)  # reset的环境数量
            self.adaptive_current_epoch_step_sum += float(episode_steps.mean())
            self.adaptive_current_epoch_reward_sum += float(episode_rewards.mean())  # 保留用于兼容

    def reset_idx(self, env_ids):
        self._refresh()
        
        # === [新增] 在重置前记录表现并更新自适应难度 ===
        if self.tighten_method == "adaptive" and self.training:
            self._update_adaptive_difficulty(env_ids)
        
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        last_step = self.gym.get_frame_count(self.sim)
        if self.training and len(self.dataIndices) == 1 and last_step >= self.tighten_steps:
            running_steps = self.running_progress_buf[env_ids] - 1
            max_running_steps, max_running_idx = running_steps.max(dim=0)
            max_running_env_id = env_ids[max_running_idx]
            if max_running_steps > self.best_rollout_len:
                self.best_rollout_len = max_running_steps
                self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps

        self._reset_default(env_ids)

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["reward_dict"] = self.reward_dict
        info["total_rewards"] = self.total_rew_buf
        info["total_steps"] = self.progress_buf
        # === [新增] 添加难度参数和统计信息到info，方便tensorboard记录 ===
        # 所有模式都添加当前的scale_factor（便于统一查看）
        if hasattr(self, 'current_scale_factor'):
            info["adaptive_difficulty/scale_factor"] = float(self.current_scale_factor)
        
        # 如果是自适应模式，添加更多统计信息
        if self.tighten_method == "adaptive" and hasattr(self, 'current_epoch_success_rate'):
            info["adaptive_difficulty/current_epoch_success_rate"] = float(self.current_epoch_success_rate)
            info["adaptive_difficulty/current_epoch_avg_steps"] = float(self.current_epoch_avg_steps)
            if hasattr(self, 'adaptive_current_epoch') and self.adaptive_current_epoch >= 0:
                info["adaptive_difficulty/current_epoch"] = float(self.adaptive_current_epoch)
        
        return obs, rew, done, info

    def pre_physics_step(self, actions):

        # ? >>> for visualization
        if not self.headless:

            cur_idx = self.progress_buf

            self.gym.clear_lines(self.viewer)

            def set_side_joint(cur_idx, side="rh"):
                cur_wrist_pos = getattr(self, f"demo_data_{side}")["wrist_pos"][torch.arange(self.num_envs), cur_idx]
                cur_mano_joint_pos = getattr(self, f"demo_data_{side}")["mano_joints"][
                    torch.arange(self.num_envs), cur_idx
                ].reshape(self.num_envs, -1, 3)
                cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
                for k in range(len(getattr(self, f"mano_joint_{side}_points"))):
                    getattr(self, f"mano_joint_{side}_points")[k][:, :3] = cur_mano_joint_pos[:, k]
                for env_id, env_ptr in enumerate(self.envs):
                    for rh_k, k in zip(
                        self.dexhand_rh.body_names,
                        (self.dexhand_rh.body_names if side == "rh" else self.dexhand_lh.body_names),
                    ):
                        self.set_force_vis(
                            env_ptr,
                            rh_k,
                            torch.norm(self.net_cf[env_id, getattr(self, f"dexhand_{side}_handles")[k]], dim=-1) != 0,
                            side,
                        )

                    def add_lines(viewer, env_ptr, hand_joints, color):
                        assert hand_joints.shape[0] == self.dexhand_rh.n_bodies and hand_joints.shape[1] == 3
                        hand_joints = hand_joints.cpu().numpy()
                        lines = np.array([[hand_joints[b[0]], hand_joints[b[1]]] for b in self.dexhand_rh.bone_links])
                        for line in lines:
                            self.gym.add_lines(viewer, env_ptr, 1, line, color)

                    color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                    add_lines(self.viewer, env_ptr, cur_mano_joint_pos[env_id].cpu(), color)

            set_side_joint(cur_idx, "lh")
            set_side_joint(cur_idx, "rh")

        # ? <<< for visualization

        root_control_dim = 9 if self.use_pid_control else 6
        res_split_idx = (
            actions.shape[1] // 2
            if not self.use_pid_control
            else ((actions.shape[1] - 2 * (root_control_dim - 6)) // 2) + 2 * (root_control_dim - 6)
        )

        base_action = actions[:, :res_split_idx]  # ? in the range of [-1, 1]
        residual_action = actions[:, res_split_idx:] * 2  # ? the delta action is theoritically in the range of [-2, 2]

        rh_dof_pos = (
            1.0 * base_action[:, root_control_dim : root_control_dim + self.num_dexhand_rh_dofs]
            + residual_action[:, 6 : 6 + self.num_dexhand_rh_dofs]
        )
        rh_dof_pos = torch.clamp(rh_dof_pos, -1, 1)

        lh_dof_pos = (
            1.0 * base_action[:, root_control_dim + root_control_dim + self.num_dexhand_rh_dofs :]
            + residual_action[:, 6 + 6 + self.num_dexhand_rh_dofs :]
        )
        lh_dof_pos = torch.clamp(lh_dof_pos, -1, 1)

        curr_act_moving_average = self.act_moving_average

        self.rh_curr_targets = torch_jit_utils.scale(
            rh_dof_pos,  # ! actions must in [-1, 1]
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.rh_curr_targets = (
            curr_act_moving_average * self.rh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, : self.num_dexhand_rh_dofs]
        )
        self.rh_curr_targets = torch_jit_utils.tensor_clamp(
            self.rh_curr_targets,
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.prev_targets[:, : self.num_dexhand_rh_dofs] = self.rh_curr_targets[:]

        self.lh_curr_targets = torch_jit_utils.scale(
            lh_dof_pos,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.lh_curr_targets = (
            curr_act_moving_average * self.lh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, self.num_dexhand_rh_dofs :]
        )
        self.lh_curr_targets = torch_jit_utils.tensor_clamp(
            self.lh_curr_targets,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.prev_targets[:, self.num_dexhand_rh_dofs :] = self.lh_curr_targets[:]

        if self.use_pid_control:
            rh_position_error = base_action[:, 0:3]
            self.rh_pos_error_integral += rh_position_error * self.dt
            self.rh_pos_error_integral = torch.clamp(self.rh_pos_error_integral, -1, 1)
            rh_pos_derivative = (rh_position_error - self.rh_prev_pos_error) / self.dt
            rh_force = (
                self.Kp_pos * rh_position_error
                + self.Ki_pos * self.rh_pos_error_integral
                + self.Kd_pos * rh_pos_derivative
            )
            self.rh_prev_pos_error = rh_position_error

            rh_force = rh_force + residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_position_error = base_action[
                :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
            ]
            self.lh_pos_error_integral += lh_position_error * self.dt
            self.lh_pos_error_integral = torch.clamp(self.lh_pos_error_integral, -1, 1)
            lh_pos_derivative = (lh_position_error - self.lh_prev_pos_error) / self.dt
            lh_force = (
                self.Kp_pos * lh_position_error
                + self.Ki_pos * self.lh_pos_error_integral
                + self.Kd_pos * lh_pos_derivative
            )
            self.lh_prev_pos_error = lh_position_error

            lh_force = (
                lh_force
                + residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

            rh_rotation_error = base_action[:, 3:root_control_dim]
            rh_rotation_error = rot6d_to_aa(rh_rotation_error)
            self.rh_rot_error_integral += rh_rotation_error * self.dt
            self.rh_rot_error_integral = torch.clamp(self.rh_rot_error_integral, -1, 1)
            rh_rot_derivative = (rh_rotation_error - self.rh_prev_rot_error) / self.dt
            rh_torque = (
                self.Kp_rot * rh_rotation_error
                + self.Ki_rot * self.rh_rot_error_integral
                + self.Kd_rot * rh_rot_derivative
            )
            self.rh_prev_rot_error = rh_rotation_error

            rh_torque = rh_torque + residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_rotation_error = base_action[
                :,
                root_control_dim
                + self.num_dexhand_rh_dofs
                + 3 : root_control_dim
                + self.num_dexhand_rh_dofs
                + root_control_dim,
            ]
            lh_rotation_error = rot6d_to_aa(lh_rotation_error)
            self.lh_rot_error_integral += lh_rotation_error * self.dt
            self.lh_rot_error_integral = torch.clamp(self.lh_rot_error_integral, -1, 1)
            lh_rot_derivative = (lh_rotation_error - self.lh_prev_rot_error) / self.dt
            lh_torque = (
                self.Kp_rot * lh_rotation_error
                + self.Ki_rot * self.lh_rot_error_integral
                + self.Kd_rot * lh_rot_derivative
            )
            self.lh_prev_rot_error = lh_rotation_error

            lh_torque = (
                lh_torque
                + residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
        else:
            rh_force = 1.0 * (base_action[:, 0:3] * self.dt * self.translation_scale * 500) + (
                residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            )
            rh_torque = 1.0 * (base_action[:, 3:6] * self.dt * self.orientation_scale * 200) + (
                residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            )
            lh_force = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
                ]
                * self.dt
                * self.translation_scale
                * 500
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            lh_torque = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs + 3 : root_control_dim + self.num_dexhand_rh_dofs + 6
                ]
                * self.dt
                * self.orientation_scale
                * 200
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )

            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self._pos_control[:] = self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):

        self.compute_observations()
        self.compute_reward(self.actions)

        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def set_force_vis(self, env_ptr, part_k, has_force, side):
        self.gym.set_rigid_body_color(
            env_ptr,
            self.gym.find_actor_handle(env_ptr, "dexhand_l" if side == "lh" else "dexhand_r"),
            getattr(self, f"dexhand_rh_handles")[part_k],  # tricks here, because the handle is the same
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


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def ensure_multi_object(tensor: Tensor, num_envs: int, feature_dim: int) -> Tensor:
    """Ensure tensor shape is [num_envs, num_objs, feature_dim]."""
    if tensor.dim() == 0:
        return tensor.view(num_envs, 1, feature_dim)
    if tensor.shape[-1] != feature_dim:
        return tensor.reshape(num_envs, -1, feature_dim)
    if tensor.dim() == 1:
        return tensor.view(num_envs, 1, feature_dim)
    if tensor.dim() == 2:
        if tensor.shape[0] == num_envs:
            return tensor.unsqueeze(1)
        return tensor.reshape(num_envs, -1, feature_dim)
    if tensor.shape[0] != num_envs:
        tensor = tensor.reshape(num_envs, -1, feature_dim)
    return tensor


def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
    obj_is_static: Tensor,  # [N, K] 新增：静态物体信息
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, float,  Dict[str, List[int]], Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints
    # assert diff_joints_pos_dist.shape[1] == 17  # ignore the base joint
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-10 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-0.8 * (diff_eef_rot_angle).abs())

    num_envs = current_eef_pos.shape[0]

    # object pose reward
    current_obj_pos = ensure_multi_object(states["manip_obj_pos"], num_envs, 3)
    current_obj_quat = ensure_multi_object(states["manip_obj_quat"], num_envs, 4)

    target_obj_pos = ensure_multi_object(target_states["manip_obj_pos"], num_envs, 3)
    target_obj_quat = ensure_multi_object(target_states["manip_obj_quat"], num_envs, 4)
    current_obj_vel = ensure_multi_object(states["manip_obj_vel"], num_envs, 3)
    target_obj_vel = ensure_multi_object(target_states["manip_obj_vel"], num_envs, 3)
    current_obj_ang_vel = ensure_multi_object(states["manip_obj_ang_vel"], num_envs, 3)
    target_obj_ang_vel = ensure_multi_object(target_states["manip_obj_ang_vel"], num_envs, 3)

    num_objs = current_obj_pos.shape[1]
    obj_is_static = obj_is_static.to(torch.bool)  # [N, K]
    dynamic_mask = ~obj_is_static
    dynamic_count = dynamic_mask.sum(dim=-1).clamp_min(1)  # 避免除0
    
    # 处理多物体维度 [N, K, 3]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)
    # 屏蔽静态物体，避免静态占位的 NaN/无效值污染奖励
    diff_obj_pos_dist = torch.where(dynamic_mask, diff_obj_pos_dist, torch.zeros_like(diff_obj_pos_dist))
    
    # 如果是多物体 [N, K]，我们对每个物体计算 Reward 然后 Sum
    reward_obj_pos = torch.exp(-30 * diff_obj_pos_dist)
    if reward_obj_pos.dim() > 1:
        reward_obj_pos = (reward_obj_pos * dynamic_mask).sum(dim=-1) / dynamic_count

    # Rotation
    # Flatten to [N*K, 4] for quat functions
    flat_curr = current_obj_quat.reshape(-1, 4)
    flat_targ = target_obj_quat.reshape(-1, 4)
    
    diff_obj_rot = quat_mul(flat_targ, quat_conjugate(flat_curr))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0].view(num_envs, num_objs)
    diff_obj_rot_angle = torch.where(dynamic_mask, diff_obj_rot_angle, torch.zeros_like(diff_obj_rot_angle))
    
    reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())
    if reward_obj_rot.dim() > 1:
        reward_obj_rot = (reward_obj_rot * dynamic_mask).sum(dim=-1) / dynamic_count
    
    diff_obj_vel = target_obj_vel - current_obj_vel
    diff_obj_vel = torch.where(dynamic_mask.unsqueeze(-1), diff_obj_vel, torch.zeros_like(diff_obj_vel))
    # [N, K, 3] -> norm -> [N, K] -> mean over D
    # diff_obj_vel.abs().mean(dim=-1) 是原代码逻辑 (L1 Norm per dim)
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))
    if reward_obj_vel.dim() > 1:
        reward_obj_vel = (reward_obj_vel * dynamic_mask).sum(dim=-1) / dynamic_count

    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    diff_obj_ang_vel = torch.where(dynamic_mask.unsqueeze(-1), diff_obj_ang_vel, torch.zeros_like(diff_obj_ang_vel))
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))
    if reward_obj_ang_vel.dim() > 1:
        reward_obj_ang_vel = (reward_obj_ang_vel * dynamic_mask).sum(dim=-1) / dynamic_count


    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    finger_tip_force = target_states["tip_force"]
    finger_tip_distance = target_states["tips_distance"]
    contact_range = [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

    # 更平滑的指尖力奖励，随力线性增长并tanh压缩
    force_sum = torch.norm(finger_tip_force_masked, dim=-1).sum(-1)
    reward_finger_tip_force = torch.tanh(0.5 * force_sum)

    # obj_vel shape: [N, K, 3] -> norm -> [N, K]
    # We need to check if ANY object exceeds limit -> [N]
    # So using .norm().max(dim=-1)[0] or .any() logic
    
    # Make sure obj_vel_norm is [N]
    obj_vel_norm = torch.norm(current_obj_vel, dim=-1)
    if obj_vel_norm.dim() > 1:
        obj_vel_norm = obj_vel_norm.max(dim=-1)[0]
        
    # Make sure obj_ang_vel_norm is [N]
    obj_ang_vel_norm = torch.norm(current_obj_ang_vel, dim=-1)
    if obj_ang_vel_norm.dim() > 1:
        obj_ang_vel_norm = obj_ang_vel_norm.max(dim=-1)[0]

    # [DEBUG] Check shapes before bitwise OR
    # print(f"DEBUG Shapes: eef_vel={torch.norm(current_eef_vel, dim=-1).shape}, joints_vel={torch.norm(joints_vel, dim=-1).mean(-1).shape}, obj_vel={obj_vel_norm.shape}")

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (obj_vel_norm > 100)
        | (obj_ang_vel_norm > 200)
    )  # sanity check

    # For failed_execute logic involving multi-object distances
    # diff_obj_pos_dist is [N, K], we probably want "if any object is too far"
    # or "mean distance too far". Original code was single object.
    # Let's assume "any object deviates too much" => failure
    # 排除静态物体：只计算动态物体的误差
    # 确保 obj_is_static 的形状与 diff_obj_pos_dist 匹配
    if obj_is_static.dim() >= 2 and obj_is_static.shape[0] == diff_obj_pos_dist.shape[0]:
        # obj_is_static: [N, K], 需要确保 K 维度匹配
        if obj_is_static.shape[1] == diff_obj_pos_dist.shape[1]:
            # 将静态物体的误差设置为负无穷，这样 max 操作会忽略它们
            dynamic_mask = ~obj_is_static  # [N, K]
            # 对于静态物体，设置误差为负无穷，这样 max 操作会忽略它们
            obj_pos_err = torch.where(dynamic_mask, diff_obj_pos_dist, torch.full_like(diff_obj_pos_dist, float('-inf')))
            obj_rot_err_deg = diff_obj_rot_angle.abs() / np.pi * 180  # [N, K]
            obj_rot_err = torch.where(dynamic_mask, obj_rot_err_deg, torch.full_like(obj_rot_err_deg, float('-inf')))
        else:
            # 如果形状不匹配，使用原始逻辑
            obj_pos_err = diff_obj_pos_dist
            obj_rot_err = diff_obj_rot_angle.abs() / np.pi * 180
    else:
        # 如果 obj_is_static 形状不匹配或为空，使用原始逻辑
        obj_pos_err = diff_obj_pos_dist
        obj_rot_err = diff_obj_rot_angle.abs() / np.pi * 180
    
    if obj_pos_err.dim() > 1:
        obj_pos_err = obj_pos_err.max(dim=-1)[0] # Max error among K objects (excluding static)
        # 如果所有物体都是静态的，max 会返回 -inf，需要处理为 0
        obj_pos_err = torch.clamp(obj_pos_err, min=0.0)
    elif obj_pos_err.dim() == 1:
        # 如果已经是 [N]，直接使用
        pass
    else:
        # 如果是标量，保持不变
        pass
        
    if obj_rot_err.dim() > 1:
        obj_rot_err = obj_rot_err.max(dim=-1)[0]
        # 如果所有物体都是静态的，max 会返回 -inf，需要处理为 0
        obj_rot_err = torch.clamp(obj_rot_err, min=0.0)
    elif obj_rot_err.dim() == 1:
        # 如果已经是 [N]，直接使用
        pass
    else:
        # 如果是标量，保持不变
        pass

    tip_contact_state = target_states["tip_contact_state"].to(torch.bool)
    if tip_contact_state.dim() >= 3:
        tip_contact_active = torch.any(tip_contact_state, dim=1)
    elif tip_contact_state.dim() == 2:
        tip_contact_active = tip_contact_state
    else:
        tip_contact_active = tip_contact_state.unsqueeze(-1)
    if tip_contact_active.shape[-1] == 1 and finger_tip_distance.shape[-1] != 1:
        tip_contact_active = tip_contact_active.repeat(1, finger_tip_distance.shape[-1])
    contact_violation = torch.any(
        (finger_tip_distance < 0.005) & ~tip_contact_active, dim=-1
    )

    failed_execute = (
        (
            (obj_pos_err > 0.12 / 0.343 * scale_factor**2)  # 调整：0.08 → 0.15（基于实际误差分析：平均0.148m，95%分位数0.22m）
            | (diff_thumb_tip_pos_dist > 0.18 / 0.7 * scale_factor)  # 0.1125 → 0.18（提升60%，95%分位数0.183m）
            | (diff_index_tip_pos_dist > 0.20 / 0.7 * scale_factor)  # 0.12375 → 0.20（提升62%，95%分位数0.200m）
            | (diff_middle_tip_pos_dist > 0.18 / 0.7 * scale_factor)  # 0.1125 → 0.18（提升60%，95%分位数0.179m）
            | (diff_pinky_tip_pos_dist > 0.23 / 0.7 * scale_factor)  # 0.1575 → 0.23（提升46%，95%分位数0.233m）
            | (diff_ring_tip_pos_dist > 0.22 / 0.7 * scale_factor)  # 0.135 → 0.22（提升63%，95%分位数0.215m）
            | (diff_level_1_pos_dist > 0.22 / 0.7 * scale_factor)  # 0.1575 → 0.22（提升40%，95%分位数0.220m）
            | (diff_level_2_pos_dist > 0.25 / 0.7 * scale_factor)  # 0.18 → 0.25（提升39%，基于Level 1的调整）
            | (obj_rot_err > 180 / 0.343 * scale_factor**2)  # 上调：90° → 180°（基于实际误差分析：平均151°，95%分位数178°）
        )
        & (running_progress_buf >= 8)
    ) | error_buf
    # contact_violation penalty: -1 if violation occurs, 0 otherwise (scale=1)
    reward_contact_violation = torch.where(contact_violation, -1.0, 0.0)

    reward_execute = (
        1 * reward_eef_pos  # 从0.1增加到0.3，提高wrist位置跟踪的权重
        + 1 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.75 * reward_middle_tip_pos
        + 0.6 * reward_pinky_tip_pos
        + 0.6 * reward_ring_tip_pos
        + 0.5 * reward_level_1_pos
        + 0.3 * reward_level_2_pos
        + 10.0 * reward_obj_pos
        + 5.0 * reward_obj_rot
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.1 * reward_obj_vel
        + 0.1 * reward_obj_ang_vel
        + 4.0 * reward_finger_tip_force
        + 0.05 * reward_power
        + 0.05 * reward_wrist_power
        + 1.0 * reward_contact_violation
    )

    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        "reward_finger_tip_force": reward_finger_tip_force,
        "reward_contact_violation": reward_contact_violation,
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf
