from __future__ import annotations

import os
import shutil
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


def _urdf_path_for_isaac(urdf_path):
    """
    Isaac Gym 只识别扩展名为 .urdf 的文件；对 name.000.urdf / name.001.urdf 会报
    "Unrecognized extension '.000.urdf'"。将此类路径转为同目录下仅含一个点的文件名
    (如 name_000.urdf)，复制后返回 (asset_root, asset_file) 供 load_asset 使用。
    同品类多实例用 .000/.001 区分时，必须经过此转换才能被 Isaac 加载。
    """
    root, filename = os.path.split(urdf_path)
    base, ext = os.path.splitext(filename)  # e.g. ("clothes_hanger.000", ".urdf")
    if ext.lower() != ".urdf":
        return root, filename
    # 若 base 中还包含点（如 clothes_hanger.000），Isaac 会误判扩展名
    if "." in base:
        # 改为下划线：clothes_hanger.000 -> clothes_hanger_000，保证扩展名仅为 .urdf
        safe_base = base.replace(".", "_")
        safe_filename = safe_base + ".urdf"
        safe_path = os.path.join(root, safe_filename)
        if safe_path != urdf_path:
            shutil.copy2(urdf_path, safe_path)
        return root, safe_filename
    return root, filename


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
        self.aggregate_mode = 0
        # self.aggregate_mode = self.cfg["env"]["aggregateMode"]
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

        self.reward_interact_scale = self.cfg["env"].get("rewardInteractScale", 2.0)
        self.reward_obj_pos_scale = self.cfg["env"].get("rewardObjPosScale", 6.0)
        self.reward_obj_rot_scale = self.cfg["env"].get("rewardObjRotScale", 5.0)
        self.reward_finger_tip_force_scale = self.cfg["env"].get("rewardFingerTipForceScale", 15.0)
        self.terminate_on_eef = self.cfg["env"].get("terminateOnEEF", False)
        self.terminate_obj_pos_threshold = self.cfg["env"].get("terminateObjPosThreshold", 0.2)
        self.terminate_obj_rot_threshold = self.cfg["env"].get("terminateObjRotThreshold", 70.0)
        self.terminate_obj_pos_final = self.cfg["env"].get("terminateObjPosFinal", 0.03)
        self.terminate_obj_rot_final = self.cfg["env"].get("terminateObjRotFinal", 30.0)
        self.terminate_thumb_threshold = self.cfg["env"].get("terminateThumbThreshold", 0.18)
        self.terminate_index_threshold = self.cfg["env"].get("terminateIndexThreshold", 0.20)
        self.terminate_middle_threshold = self.cfg["env"].get("terminateMiddleThreshold", 0.18)
        self.terminate_pinky_threshold = self.cfg["env"].get("terminatePinkyThreshold", 0.23)
        self.terminate_ring_threshold = self.cfg["env"].get("terminateRingThreshold", 0.22)
        self.terminate_level_1_threshold = self.cfg["env"].get("terminateLevel1Threshold", 0.22)
        self.terminate_level_2_threshold = self.cfg["env"].get("terminateLevel2Threshold", 0.25)
        self.eef_vel_limit = self.cfg["env"].get("eefVelLimit", 100.0)
        self.eef_ang_vel_limit = self.cfg["env"].get("eefAngVelLimit", 200.0)
        self.joints_vel_limit = self.cfg["env"].get("jointsVelLimit", 100.0)
        self.dof_vel_limit = self.cfg["env"].get("dofVelLimit", 200.0)
        self.obj_vel_limit = self.cfg["env"].get("objVelLimit", 100.0)
        self.obj_ang_vel_limit = self.cfg["env"].get("objAngVelLimit", 200.0)
        support_force_cfg = self.cfg.get("env", {}).get("support_force", {})
        self.support_force_decay_start_factor = support_force_cfg.get("decay_start_factor", 0.8)

        # === [新增] 自适应采样相关配置 ===
        # 时间片段(bin)数量，将根据动作长度动态确定
        self.adaptive_sampling_bins = self.cfg["env"].get("adaptiveSamplingBins", 12)
        
        if self.random_state_init:
            # 平滑核大小，用于对失败率进行平滑处理
            self.adaptive_sampling_kernel_size = self.cfg["env"].get("adaptiveSamplingKernelSize", 3)
            # 平滑衰减因子，越接近1变化越慢
            self.adaptive_sampling_lambda = self.cfg["env"].get("adaptiveSamplingLambda", 0.8)
            # 均匀采样比例，避免某些bin完全不被采样
            self.adaptive_sampling_uniform_ratio = self.cfg["env"].get("adaptiveSamplingUniformRatio", 0.4)
            # 失败率更新时的衰减因子，越接近1变化越慢
            self.adaptive_sampling_alpha = self.cfg["env"].get("adaptiveSamplingAlpha", 0.2)
            # 所有bin成功率超过此阈值才允许提升难度
            self.adaptive_sampling_all_bins_threshold = self.cfg["env"].get("adaptiveSamplingAllBinsThreshold", 0.40)
        else:
            # 非自适应采样模式下，依然保持 bin 数量以便记录成功率统计
            pass

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]
        self.target_epoch = self.cfg["env"].get("targetEpoch", 800)

        # === [新增] Curriculum Learning 配置 ===
        self.terminate_on_contact = self.cfg["env"].get("terminateOnContact", False)
        self.no_regression_threshold = self.cfg["env"].get("noRegressionThreshold", 1.5)
        # 初始化旋转难度系数，初始与全局系数同步
        self.rot_scale_factor = self.tighten_factor if (self.tighten_factor is not None and 0 < self.tighten_factor <= 1.0) else 1.0
        self.obj_pos_fail_count = 0
        self.obj_rot_fail_count = 0
        self.total_fail_count = 0
        self.last_difficulty_update_epoch = -1
        self.stuck_epoch_counter = 0

        # === [新增] 动态难度调整相关配置（基于epoch） ===
        # 初始化基础统计变量，不论什么模式都可用
        self.adaptive_current_epoch = -1  # 当前epoch编号
        self.adaptive_current_epoch_success_count = 0  # 当前epoch的成功次数
        self.adaptive_current_epoch_reset_count = 0  # 当前epoch内的reset次数（用于计算成功率）
        self.adaptive_current_epoch_step_sum = 0.0  # 当前epoch的累计step
        self.adaptive_current_epoch_reward_sum = 0.0  # 当前epoch的累计reward（保留用于兼容）
        
        self.current_epoch_success_rate = 0.0  # 当前epoch的成功率
        self.current_epoch_avg_steps = 0.0  # 当前epoch的平均step
        self.current_epoch_bin_rates = None # 当前epoch各bin的成功率
        
        if self.tighten_method in ["adaptive_dual", "adaptive_real"]:
            # 配置参数（所有窗口大小都是基于epoch，而非episode）
            self.adaptive_success_window = self.cfg["env"].get("adaptiveSuccessWindow", 5)  # 连续N个epoch检查成功率
            self.adaptive_success_threshold = self.cfg["env"].get("adaptiveSuccessThreshold", 0.10)  # 成功率阈值（20%）
            self.adaptive_step_window = self.cfg["env"].get("adaptiveStepWindow", 5)  # 连续N个epoch检查平均step
            self.adaptive_step_threshold = self.cfg["env"].get("adaptiveStepThreshold", 20)  # 平均step阈值
            self.adaptive_no_improvement_window = self.cfg["env"].get("adaptiveNoImprovementWindow", 10)  # 连续N个epoch没有提升就降低难度
            self.adaptive_stuck_threshold = self.cfg["env"].get("adaptiveStuckThreshold", 100) # 当一个难度持续N个epoch没下降难度就立刻下降
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
            self.all_bins_high_success_history = deque(maxlen=5)  # 记录是否所有bin都高于60%
            self.low_success_history = deque(maxlen=30) # 记录最近 30 个 epoch 是否出现过 <20% 的情况
            
            # 用于tensorboard记录的当前scale_factor（在compute_reward中更新）
            self.current_scale_factor = initial_scale
            
            # 跟踪难度提升历史
            self.adaptive_last_difficulty_increase_epoch = -1  # 上次难度提升（scale_factor降低）时的epoch
            
            self.difficulty_increase_timer = 0 # 难度增加冷却计时器
            self.epochs_at_current_scale = 0 # 当前难度下的 epoch 计数
            self.adaptive_difficulty_is_warming_up = True # 是否处于难度切换后的 30 epoch 训练期
            self.bin_success_sum_at_current_scale = None # 当前难度下各 bin 成功率累加
            self.scale_increase_counts = {} # [新增] 记录每个难度系数下，上升难度系数的次数，每个难度系数只能上升一次
            
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
            self.adaptive_global_scale_factor = self.tighten_factor if (self.tighten_factor is not None and 0 < self.tighten_factor <= 1.0) else 1.0
            self.adaptive_epoch_success_rate_history = deque(maxlen=5)
            self.adaptive_epoch_steps_history = deque(maxlen=5)
        
        # 用于tensorboard记录的当前scale_factor（所有模式都使用）
        if not hasattr(self, 'current_scale_factor'):
            # 如果不是adaptive模式，初始化为1.0或tighten_factor
            initial_scale_non_adaptive = self.tighten_factor if (self.tighten_factor is not None and 0 < self.tighten_factor <= 1.0) else 1.0
            self.current_scale_factor = initial_scale_non_adaptive
        
        # 初始化tensorboard统计信息（非adaptive模式也需要，设置为默认值）
        if not hasattr(self, 'current_epoch_success_rate'):
            self.current_epoch_success_rate = 0.0
            self.current_epoch_avg_steps = 0.0

        # 初始化重力和摩擦力统计信息
        if not hasattr(self, 'current_epoch_gravity'):
            self.current_epoch_gravity = -9.8  # 默认重力值
            self.current_epoch_avg_friction = 3.0  # 默认摩擦力值

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
        # self._init_failure_log_path() # 将在 super().__init__ 之后，数据加载完成后调用
        self.failure_log_count = 0
        self.last_failure_log_epoch = -1

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )

        # === [修改] 根据动作长度动态确定 bin 的个数 ===
        # 已经在 super().__init__ -> create_sim -> _create_envs 中加载了数据
        if hasattr(self, 'demo_data_rh') and "seq_len" in self.demo_data_rh:
            max_seq_len = self.demo_data_rh["seq_len"].float().max().item()
            # 根据总帧数 / 60来确定，然后四舍五入
            self.adaptive_sampling_bins = max(1, int(round(max_seq_len / 60.0)))
            print(f"[ADAPTIVE SAMPLING] Dynamically set num_bins to {self.adaptive_sampling_bins} based on max motion length {max_seq_len:.1f}")
        
        if self.random_state_init:
            print(f"[ADAPTIVE SAMPLING] Enabled with {self.adaptive_sampling_bins} bins")
            print(f"  - Kernel size: {self.adaptive_sampling_kernel_size}, Lambda: {self.adaptive_sampling_lambda}")
            print(f"  - Uniform ratio: {self.adaptive_sampling_uniform_ratio}")
            print(f"  - All bins success threshold: {self.adaptive_sampling_all_bins_threshold}")
        else:
            print(f"[BIN STATS] Enabled with {self.adaptive_sampling_bins} bins for logging")

        # 现在可以安全地初始化失败日志路径了
        self._init_failure_log_path()

        self.env_start_bin_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # [新增] 初始化从最开始开始 reset 的比例 (默认为 10%)
        self.start_from_beginning_ratio = 0.10
        # [新增] 记录每个环境是否发生了特定的失败
        self.env_obj_pos_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.env_obj_rot_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


        # === [新增] 初始化自适应采样 tensors (在父类初始化之后，因为需要 self.device) ===
        if self.random_state_init and self.adaptive_sampling_bins is not None:
            # 初始化采样统计（记录成功和失败次数）
            self.bin_success_count = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self.bin_total_count = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self.bin_ema_success_rates = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self.bin_success_sum_at_current_scale = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self._current_bin_success = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self._current_bin_total = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)

            # 初始化每个环境启动时的bin索引，用于统计从启动到结束跨越的所有bin
            

            # 初始化bin重置计数统计
            self.bin_reset_count = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self._current_bin_reset = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)

            # [新增] 初始化 obj-pos 和 obj-rot 通过率统计
            self._current_bin_obj_pos_pass = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self._current_bin_obj_rot_pass = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self.bin_obj_pos_pass_rate = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)
            self.bin_obj_rot_pass_rate = torch.zeros(self.adaptive_sampling_bins, dtype=torch.float, device=self.device)

            
            # 创建平滑核
            self.adaptive_sampling_kernel = torch.tensor(
                [self.adaptive_sampling_lambda**i for i in range(self.adaptive_sampling_kernel_size)], device=self.device
            )

        num_bins = self.adaptive_sampling_bins
        self.bin_accumulated_epochs_since_reset = torch.zeros((num_bins,), dtype=torch.long, device=self.device)

        # [新增] 维持最小稳定 scale 的逻辑所需的缓冲区
        # 记录过去 30 个 epoch 的通过率是否 > 70% (1 或 0)
        self.bin_pos_pass_history = torch.zeros((num_bins, 30), dtype=torch.bool, device=self.device)
        self.bin_rot_pass_history = torch.zeros((num_bins, 30), dtype=torch.bool, device=self.device)
        # 记录过去 50 个 epoch 的综合成功率 (float)
        self.bin_success_history_50 = torch.zeros((num_bins, 50), dtype=torch.float, device=self.device)
        # 最小稳定 scale：初始设为较大的 1.0
        self.bin_pos_stable_best = torch.full((num_bins,), 1.0, device=self.device)
        self.bin_rot_stable_best = torch.full((num_bins,), 1.0, device=self.device)
        # 当前历史记录的写入指针
        self.bin_history_ptr_30 = 0
        self.bin_history_ptr_50 = 0
        
        self.bin_next_trial_starter = torch.zeros((num_bins,), dtype=torch.long, device=self.device) # 0: pos, 1: rot
        # === [新增] 初始化 Per-bin independent scale factors and states for adaptive_dual ===
        # 必须在父类初始化之后，因为需要 self.device
        if self.tighten_method == "adaptive_dual" and self.adaptive_sampling_bins is not None:
            
            # 重新获取 initial_scale
            if self.tighten_factor is not None and 0 < self.tighten_factor <= 1.0:
                initial_scale = self.tighten_factor
            else:
                initial_scale = 1.0
                
            self.bin_pos_scale = torch.full((num_bins,), initial_scale, device=self.device)
            self.bin_rot_scale = torch.full((num_bins,), initial_scale, device=self.device)
            # bin_state: [num_bins, 2] where 2 is (pos, rot). 
            # 0: Normal, 1: TrialB, 2: Aggressive, 3: Frozen, 4: Terminal, 5: Pending
            self.bin_state = torch.zeros((num_bins, 2), dtype=torch.long, device=self.device)
            self.bin_epochs_at_scale = torch.zeros((num_bins, 2), dtype=torch.long, device=self.device)
            self.bin_trial_start_epoch = torch.zeros((num_bins, 2), dtype=torch.long, device=self.device)
            self.bin_phase_a_rate = torch.zeros((num_bins, 2), device=self.device)
            self.bin_phase_b_acc = torch.zeros((num_bins, 2), device=self.device)
            self.bin_best_scales = torch.full((num_bins, 2), initial_scale, device=self.device)
            self.bin_last_stable_scale = torch.full((num_bins, 2), initial_scale, device=self.device)
            self.bin_last_aggressive_rate = torch.zeros((num_bins, 2), device=self.device)

            # [新增] 用于新版 adaptive_dual 逻辑的变量
            self.bin_ema_at_start = torch.zeros((num_bins, 2), device=self.device)
            self.bin_initial_scale_before_trial = torch.full((num_bins, 2), initial_scale, device=self.device)
            self.bin_first_mover = torch.full((num_bins,), -1, dtype=torch.long, device=self.device) # -1: none, 0: pos, 1: rot
            self.bin_aggressive_turn = torch.full((num_bins,), -1, dtype=torch.long, device=self.device) # -1: none, 0: pos, 1: rot

            
            
            # 记录回升历史，每层每个组件只能回升一次
            self.bin_scale_increase_history = [{"pos": {}, "rot": {}} for _ in range(num_bins)]
            self.adaptive_sampling_kernel = self.adaptive_sampling_kernel / self.adaptive_sampling_kernel.sum()
            
            # [新增] 用于冻结恢复限制和条件的变量
            self.bin_unfreeze_history = [{"pos": {}, "rot": {}} for _ in range(num_bins)]
            self.bin_other_scale_at_freeze = torch.full((num_bins, 2), 1.0, device=self.device)

            print(f"  - Aggressive sampling kernel: {self.adaptive_sampling_kernel.tolist()}, Lambda: {self.adaptive_sampling_lambda}")

        # === [新增] 初始化 Per-bin independent scale factors and states for adaptive_real ===
        if self.tighten_method == "adaptive_real" and self.adaptive_sampling_bins is not None:
            num_bins = self.adaptive_sampling_bins
            # 重新获取 initial_scale
            if self.tighten_factor is not None and 0 < self.tighten_factor <= 1.0:
                initial_scale = self.tighten_factor
            else:
                initial_scale = 1.0
                
            # Per-bin pos 和 rot scale
            self.bin_pos_scale = torch.full((num_bins,), initial_scale, device=self.device)
            self.bin_rot_scale = torch.full((num_bins,), initial_scale, device=self.device)
            
            # 每个 bin 的状态：
            # 0: 前900轮线性下降阶段
            # 1: 触底后，正在调整阶段（需要统计失败率）
            # 2: 已经成功（永不改变）
            # 3: 连续30个epoch没有失败但也没有成功（可以继续提升）
            # 4: 降低难度后反而失败（永不改变）
            self.bin_status = torch.zeros(num_bins, dtype=torch.long, device=self.device)
            
            # 每个 bin 在当前 scale 下停留的 epoch 数
            self.bin_epochs_at_current_scale = torch.zeros((num_bins, 2), dtype=torch.long, device=self.device)  # [num_bins, 2] for pos and rot
            
            # 统计失败率（从 scale 降到 0.7 开始统计）
            # obj_pos 失败数和 obj_rot 失败数
            self.bin_pos_fail_count = torch.zeros(num_bins, dtype=torch.float, device=self.device)
            self.bin_rot_fail_count = torch.zeros(num_bins, dtype=torch.float, device=self.device)
            self.bin_total_pass_count = torch.zeros(num_bins, dtype=torch.float, device=self.device)
            
            # 记录过去30个epoch的成功率，用于判断是否成功或失败
            self.bin_success_rate_history_30 = torch.zeros((num_bins, 30), dtype=torch.float, device=self.device)
            self.bin_history_ptr_30 = 0  # 循环指针
            
            # 连续没有失败的 epoch 数
            self.bin_no_fail_epochs = torch.zeros(num_bins, dtype=torch.long, device=self.device)
            
            # 记录降低难度前的 scale（用于回退）
            self.bin_pos_scale_before_decrease = torch.full((num_bins,), initial_scale, device=self.device)
            self.bin_rot_scale_before_decrease = torch.full((num_bins,), initial_scale, device=self.device)
            
            # 记录是否已经尝试过降低难度
            self.bin_tried_decrease = torch.zeros((num_bins, 2), dtype=torch.bool, device=self.device)  # [num_bins, 2] for pos and rot
            
            # 用于判断失败率的标志（是否已经触底）
            self.bin_reached_bottom = torch.zeros(num_bins, dtype=torch.bool, device=self.device)
            
            # [新增] 锁定标志：如果 pos/rot 在成功期间稳定超过阈值，则不允许再上调
            self.bin_pos_locked = torch.zeros(num_bins, dtype=torch.bool, device=self.device)  # 锁定 pos 上调
            self.bin_rot_locked = torch.zeros(num_bins, dtype=torch.bool, device=self.device)  # 锁定 rot 上调
            
            # [新增] 触底后稳定计数：只看 scale 是否保持不变（与“是否成功”无关）
            # 连续稳定达到阈值后，锁定对应的 pos/rot，禁止再上调
            self.bin_pos_stable_epochs = torch.zeros(num_bins, dtype=torch.long, device=self.device)
            self.bin_rot_stable_epochs = torch.zeros(num_bins, dtype=torch.long, device=self.device)
            # 使用 -1.0 作为未初始化的标记值
            self.bin_pos_scale_last = torch.full((num_bins,), -1.0, device=self.device)
            self.bin_rot_scale_last = torch.full((num_bins,), -1.0, device=self.device)
            
            print(f"[ADAPTIVE REAL] Initialized with {num_bins} bins, initial scale: {initial_scale:.4f}")
            print(f"  - Phase 1: Linear decay for 900 epochs (adjusted for num_envs={self.num_envs})")
            print(f"  - Phase 2: Adaptive adjustment based on failure rates when scale reaches 0.70")

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
                f.write("Failure Diagnostics Log\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log file: {self.failure_log_file}\n")
                f.write(f"Experiment dir: {experiment_dir}\n")
                
                # 新增：打印 Bin 信息
                if hasattr(self, 'adaptive_sampling_bins') and self.adaptive_sampling_bins is not None:
                    f.write(f"\nAdaptive Sampling Bins Configuration ({self.adaptive_sampling_bins} bins):\n")
                    f.write("Note: Bins are segments of the total motion length (0% to 100%).\n")
                    f.write(f"Calculation: Bin = floor((current_step / total_motion_length) * {self.adaptive_sampling_bins})\n")
                    f.write("Example (for a motion of length 1000):\n")
                    ref_len = 1000
                    for i in range(self.adaptive_sampling_bins):
                        start_pct = (i / self.adaptive_sampling_bins) * 100
                        end_pct = ((i + 1) / self.adaptive_sampling_bins) * 100
                        start_step = int(i * ref_len / self.adaptive_sampling_bins)
                        end_step = int((i + 1) * ref_len / self.adaptive_sampling_bins) - 1
                        f.write(f"  Bin {i:02d}: {start_pct:5.1f}% - {end_pct:5.1f}% (Steps {start_step:3d} - {end_step:3d})\n")
                
                f.write("=" * 80 + "\n\n")
            self._failure_log_buffer = []  # 累积失败日志，每 20 个 epoch 写一次文件
            print(f"[INFO] Failure diagnostics will be saved to: {self.failure_log_file}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize failure log file: {e}")
            self.failure_log_file = None
            self._failure_log_buffer = []

    def _reset_bin_histories(self, bin_idx):
        """当某个 bin 的 scale 或状态发生重大改变时，重置其通过率历史统计"""
        if self.adaptive_sampling_bins is not None:
            self.bin_pos_pass_history[bin_idx].fill_(False)
            self.bin_rot_pass_history[bin_idx].fill_(False)
            self.bin_ema_success_rates[bin_idx] = 0
            self.bin_success_history_50[bin_idx].fill_(0)
            self.bin_accumulated_epochs_since_reset[bin_idx] = 0
            # 注意：不重置 bin_history_ptr_30，让它在下次写入时继续滑动覆盖，
            # 但把内容清空可以确保短期内不会触发“稳定”判定。
            # 也可以选择重置指针，但指针是全局的。

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8

        # === [新增] 打印初始重力设置 ===
        print(f"[GRAVITY MONITOR] Initial gravity_z = {self.sim_params.gravity.z:.2f} m/s²")
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        # [第 5 点] 初始化重力缓存，供 compute_observations 中 manip_obj_weight 使用
        self._gravity_z = float(self.sim_params.gravity.z)
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

        def _tensor_len_or_neg1(x):
            if isinstance(x, torch.Tensor):
                return int(x.shape[0])
            return -1

        def _mano_len_or_neg1(sample):
            mano = sample.get("mano_joints", None)
            if isinstance(mano, dict) and len(mano) > 0:
                first_joint = next(iter(mano.values()))
                return _tensor_len_or_neg1(first_joint)
            return -1

        for data_idx in self.dataIndices:
            dtype = ManipDataFactory.dataset_type(data_idx)
            lh_item = self.demo_dataset_lh_dict[dtype][data_idx]
            rh_item = self.demo_dataset_rh_dict[dtype][data_idx]
            unique_data_lh_list.append(lh_item)
            unique_data_rh_list.append(rh_item)

            lh_wrist_len = _tensor_len_or_neg1(lh_item.get("wrist_pos", None))
            rh_wrist_len = _tensor_len_or_neg1(rh_item.get("wrist_pos", None))
            lh_mano_len = _mano_len_or_neg1(lh_item)
            rh_mano_len = _mano_len_or_neg1(rh_item)
            lh_obj_len = _tensor_len_or_neg1(lh_item.get("obj_trajectory", None))
            rh_obj_len = _tensor_len_or_neg1(rh_item.get("obj_trajectory", None))
            print(
                f"[DATA LEN] idx={data_idx} | "
                f"LH(wrist={lh_wrist_len}, mano={lh_mano_len}, obj={lh_obj_len}) | "
                f"RH(wrist={rh_wrist_len}, mano={rh_mano_len}, obj={rh_obj_len})"
            )

        print("Packing unique data to GPU...")
        packed_unique_lh = self.pack_data(unique_data_lh_list, side="lh")
        packed_unique_rh = self.pack_data(unique_data_rh_list, side="rh")
        if "seq_len" in packed_unique_lh and "seq_len" in packed_unique_rh:
            print(
                f"[DATA LEN][PACKED] LH seq_len={packed_unique_lh['seq_len'].detach().cpu().tolist()} | "
                f"RH seq_len={packed_unique_rh['seq_len'].detach().cpu().tolist()}"
            )

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

        # [打印功能] 直接打印第177帧到186帧的 wrist-rot 和 opt-wrist-rot (取第一个环境)
        # print("\n" + "="*50)
        # print("[DEBUG] Printing frames 177 to 186 from loaded demo data (Env 0):")
        # for frame_idx in range(177, 187):
        #     print(f"--- Frame {frame_idx} ---")
        #     for side, data_dict in [("RH", self.demo_data_rh), ("LH", self.demo_data_lh)]:
        #         if "wrist_rot" in data_dict:
        #             val = data_dict["wrist_rot"][0, frame_idx].cpu().numpy()
        #             print(f"  [{side}] wrist_rot:     {val}")
        #         if "opt_wrist_rot" in data_dict:
        #             val = data_dict["opt_wrist_rot"][0, frame_idx].cpu().numpy()
        #             print(f"  [{side}] opt_wrist_rot: {val}")
        # print("="*50 + "\n")
        # import pdb; pdb.set_trace()
        
        # === [新增] 将多物体轨迹预处理为 Tensor 以加速 Reward 计算 ===
        # 目标: 生成 self.rh_multi_obj_traj [NumEnvs, MaxObjs, T, 4, 4]
        def prepare_multi_obj_tensor(packed_unique, indices):
            # 1. 从 unique data 中提取轨迹 Tensor
            
            unique_trajs = [] # [UniqueBatch, NumObjs, T, 4, 4]
            unique_vels = []  # [UniqueBatch, NumObjs, T, 3]
            unique_ang_vels = [] # [UniqueBatch, NumObjs, T, 3] # [新增]
            
            scene_objs_batch = packed_unique["scene_objects"]
            # 假设同一个 Batch 内物体数量一致，取最大值或第一个
            max_objs = len(scene_objs_batch[0]) if len(scene_objs_batch) > 0 else 0
            
            for scene_objs in scene_objs_batch:
                objs_traj = []
                objs_vel = []
                objs_ang_vel = []
                for i in range(max_objs):
                    if i < len(scene_objs):
                        # 取出轨迹 [T, 4, 4]
                        t = scene_objs[i]["trajectory"]
                        
                        # 取出速度 [T, 3]
                        v = scene_objs[i].get("velocity", None)
                        if v is None:
                            v = torch.zeros((t.shape[0], 3), device=self.device)
                        elif not isinstance(v, torch.Tensor):
                            v = torch.tensor(v, device=self.device, dtype=torch.float32)
                        
                        # [新增] 取出角速度 [T, 3]
                        av = scene_objs[i].get("angular_velocity", None)
                        if av is None:
                            av = torch.zeros((t.shape[0], 3), device=self.device)
                        elif not isinstance(av, torch.Tensor):
                            av = torch.tensor(av, device=self.device, dtype=torch.float32)
                    else:
                        # Padding
                        t = torch.eye(4, device=self.device).unsqueeze(0).repeat(packed_unique["seq_len"].max(), 1, 1)
                        v = torch.zeros((t.shape[0], 3), device=self.device)
                        av = torch.zeros((t.shape[0], 3), device=self.device)
                    objs_traj.append(t)
                    objs_vel.append(v)
                    objs_ang_vel.append(av)
                
                if objs_traj:
                    unique_trajs.append(torch.stack(objs_traj))
                    unique_vels.append(torch.stack(objs_vel))
                    unique_ang_vels.append(torch.stack(objs_ang_vel))
                else:
                    # 空数据处理
                    unique_trajs.append(torch.eye(4, device=self.device).view(1, 1, 4, 4))
                    unique_vels.append(torch.zeros((1, 1, 3), device=self.device))
                    unique_ang_vels.append(torch.zeros((1, 1, 3), device=self.device))

            unique_trajs_stack = torch.stack(unique_trajs) # [UniqueBatch, NumObjs, T, 4, 4]
            unique_vels_stack = torch.stack(unique_vels)   # [UniqueBatch, NumObjs, T, 3]
            unique_ang_vels_stack = torch.stack(unique_ang_vels) # [UniqueBatch, NumObjs, T, 3]
            
            # 2. Broadcast 到 NumEnvs
            return unique_trajs_stack[indices].clone(), unique_vels_stack[indices].clone(), unique_ang_vels_stack[indices].clone()

        self.rh_multi_obj_traj, self.rh_multi_obj_vel, self.rh_multi_obj_ang_vel = prepare_multi_obj_tensor(packed_unique_rh, env_to_data_indices)
        self.lh_multi_obj_traj, self.lh_multi_obj_vel, self.lh_multi_obj_ang_vel = prepare_multi_obj_tensor(packed_unique_lh, env_to_data_indices)
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

        # === [新增] 计算聚合组大小 (Aggregate Group Size) ===
        # 聚合组大小 = 机器人 + 桌子 + 所有物体的 Body/Shape 总数
        # 预估最大物体数量（双手的物体，每个物体预留 5 个 Shape/Body 冗余）
        max_objs = self.num_objs_per_env
        # 动态计算 Asset 真实的 Body 和 Shape 数量
        max_body_per_obj = 0
        max_shape_per_obj = 0
        for asset in self.objs_assets.values():
            max_body_per_obj = max(max_body_per_obj, self.gym.get_asset_rigid_body_count(asset))
            max_shape_per_obj = max(max_shape_per_obj, self.gym.get_asset_rigid_shape_count(asset))

        # 重新计算聚合组配额，并预留 20 个单位的冗余空间
        max_agg_bodies = (num_dexhand_rh_bodies + num_dexhand_lh_bodies + 2 + 
                        self.num_objs_per_env * max_body_per_obj + 20)
        max_agg_shapes = (num_dexhand_rh_shapes + num_dexhand_lh_shapes + 2 + 
                        self.num_objs_per_env * max_shape_per_obj + 20)
        self.max_agg_bodies = max_agg_bodies
        self.max_agg_shapes = max_agg_shapes
        # max_agg_bodies = num_dexhand_rh_bodies + num_dexhand_lh_bodies + 1 + 2 * max_objs * 5 
        # max_agg_shapes = num_dexhand_rh_shapes + num_dexhand_lh_shapes + 1 + 2 * max_objs * 5

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

        # num_per_row = int(np.sqrt(self.num_envs))
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))

        # === [新增] 隐藏手部控制参数 ===
        self.hide_hands = self.cfg["env"].get("hide_hands", False)
        if self.hide_hands:
            print("[INFO] Hands are HIDDEN (will be transparent and non-colliding).")

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)

            # === [新增] 开始聚合 (Begin Aggregate) ===
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # [迁移] 如果需要 Camera，在这里添加
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(env=env_ptr, isaac_gym=self.gym)
                )

            # Create Robot Actors
            # 如果隐藏手部，设置碰撞掩码为 -1 (不与任何物体碰撞)
            hand_collision_mask = 1 if self.hide_hands else (1 if self.dexhand_rh.self_collision else 0)

            
            dexhand_rh_actor = self.gym.create_actor(
                env_ptr, dexhand_rh_asset, self.dexhand_rh_pose, "dexhand_r", i,
                hand_collision_mask,
            )
            dexhand_lh_actor = self.gym.create_actor(
                env_ptr, dexhand_lh_asset, self.dexhand_lh_pose, "dexhand_l", i,
                hand_collision_mask,
            )

            # 如果隐藏手部，将手设为全透明
            if self.hide_hands:
                transparent_color = gymapi.Vec3(0, 0, 0) # 实际上由 set_rigid_body_color 的 alpha 决定更准，但这里设为黑色
                # 遍历手的所有 rigid bodies
                for b_idx in range(self.gym.get_actor_rigid_body_count(env_ptr, dexhand_rh_actor)):
                    self.gym.set_rigid_body_color(env_ptr, dexhand_rh_actor, b_idx, gymapi.MESH_VISUAL, transparent_color)
                for b_idx in range(self.gym.get_actor_rigid_body_count(env_ptr, dexhand_lh_actor)):
                    self.gym.set_rigid_body_color(env_ptr, dexhand_lh_actor, b_idx, gymapi.MESH_VISUAL, transparent_color)

            # Set Props
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_rh_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_lh_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_rh_actor, dexhand_rh_dof_props) 
            self.gym.set_actor_dof_properties(env_ptr, dexhand_lh_actor, dexhand_lh_dof_props) 

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

            # === [新增] 结束聚合 (End Aggregate) ===
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

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
                asset_options.density = 10
                asset_options.fix_base_link = is_static  # 静态物体固定base link
                # asset_options.disable_gravity = True 
                
                # 检查并处理 URDF 路径（使用与 mano2dexhand_segmented.py 相同的处理方式）
                if not os.path.exists(urdf_path):
                    raise FileNotFoundError(f"URDF file not found: {urdf_path}")
                
                # 使用 _urdf_path_for_isaac 处理 .000.urdf 等特殊扩展名
                asset_root, asset_file = _urdf_path_for_isaac(urdf_path)
                
                try:
                    asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
                except Exception as e:
                    raise RuntimeError(f"Failed to load asset from {urdf_path} (processed as {asset_file}): {str(e)}")
                
                if asset is None:
                    raise RuntimeError(f"Asset loading returned None for {urdf_path} (processed as {asset_file})")
                
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

            # if obj_name == "vase":
            #     handle = self.gym.create_actor(env_ptr, asset, pose, f"{obj_name}_{side}_{k}", i, 1)
            #     import pdb; pdb.set_trace()
            # else:
            handle = self.gym.create_actor(env_ptr, asset, pose, f"{obj_name}_{side}_{k}", i, 0)
            # 3. 创建 Actor
            #handle = self.gym.create_actor(env_ptr, asset, pose, f"{obj_name}_{side}_{k}", i, 0)
            
            # === [修改重点] 4. 分别设置质量(Body)和摩擦力(Shape) ===
            
            # 4.1 设置质量 (Rigid Body Properties)
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
            
            # 检查 body_props 是否为空
            if len(body_props) == 0:
                raise RuntimeError(
                    f"Actor '{obj_name}_{side}_{k}' (handle={handle}) has no rigid bodies. "
                    f"This usually means the asset failed to load or the URDF is invalid. "
                    f"URDF path: {urdf_path}"
                )
            
            original_mass = body_props[0].mass
            # import pdb; pdb.set_trace()
            new_mass = max(0.1, min(0.5, original_mass)) # 限制质量范围
            
            body_props[0].mass = new_mass
            # 注意：friction 不能在这里设置！
            self.gym.set_actor_rigid_body_properties(env_ptr, handle, body_props)
            
            # 4.2 设置摩擦力 (Rigid Shape Properties)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            # 通常一个 actor 可能有多个 shape，这里假设统一设置
            if len(shape_props) == 0:
                if i == 0:
                    print(f"[WARNING] Actor '{obj_name}_{side}_{k}' has no rigid shapes")
            else:
                for shape_prop in shape_props:
                    shape_prop.friction = 3.0  # 初始摩擦力，VecTask 的自适应逻辑会在第一次 randomization 时根据配置覆盖此值
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

    def _adaptive_sampling(self, env_ids):
        """
        自适应采样：根据失败率优先采样困难的片段

        Args:
            env_ids: 需要重置的环境ID列表（Tensor）

        Returns:
            sampled_progress_ratios: 为每个环境采样到的进度比例 (0.0-1.0)
        """
        if not self.random_state_init or self.adaptive_sampling_bins is None:
            # 如果没有启用自适应采样，返回均匀随机采样
            return torch.rand(len(env_ids), device=self.device)

        # 计算采样概率：基于成功率，越难的片段（成功率越低）采样概率越高
        # 避免除零：对于没有尝试过的bin，假设成功率0%以增加采样权重
        bin_success_rates = torch.where(
            self.bin_total_count > 0,
            self.bin_success_count / self.bin_total_count,
            torch.zeros_like(self.bin_success_count)  # 没有数据的bin假设0%成功率
        )

        # 难度权重 = 1 / (成功率 + epsilon)，成功率越低权重越高
        epsilon = 1e-6
        difficulty_weights = 1.0 / (bin_success_rates + epsilon)

        # 归一化难度权重
        difficulty_weights = difficulty_weights / difficulty_weights.sum()

        # 添加均匀分布作为baseline
        sampling_probabilities = difficulty_weights + self.adaptive_sampling_uniform_ratio / float(self.adaptive_sampling_bins)

        # 使用卷积进行平滑处理
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_sampling_kernel_size - 1),  # 非因果核
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities,
            self.adaptive_sampling_kernel.view(1, 1, -1)
        ).view(-1)

        # 归一化概率
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # 采样bin索引
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # 将bin索引转换为进度比例 (0.0-1.0)
        # 在每个bin内添加随机偏移
        bin_width = 1.0 / self.adaptive_sampling_bins
        bin_starts = sampled_bins.float() / self.adaptive_sampling_bins
        # 在bin内均匀随机
        random_offsets = torch.rand(len(env_ids), device=self.device) * bin_width
        sampled_progress_ratios = bin_starts + random_offsets

        # 确保在有效范围内
        sampled_progress_ratios = torch.clamp(sampled_progress_ratios, 0.0, 0.98)  # 留出2%余量

        return sampled_progress_ratios, sampled_bins

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
        # [第 3 点] 预计算 joints_state 的 body 索引，避免 _update_states 中每步 Python 循环 + torch.stack
        self._rh_joints_state_body_indices = torch.tensor(
            [self.dexhand_rh_handles[k] for k in self.dexhand_rh.body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._lh_joints_state_body_indices = torch.tensor(
            [self.dexhand_lh_handles[k] for k in self.dexhand_lh.body_names],
            dtype=torch.long,
            device=self.device,
        )
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
        
        # === [新增] 获取多物体的 Rigid Body 索引 (用于 apply_forces) ===
        def get_rb_indices(handles_list_of_lists):
            indices = []
            max_objs = max([len(handles) for handles in handles_list_of_lists]) if handles_list_of_lists else 0
            for i in range(self.num_envs):
                env_ptr = self.envs[i]
                env_indices = []
                for handle in handles_list_of_lists[i]:
                    # 获取该 actor 的第一个 rigid body handle (在 env 内的索引)
                    # 假设物体只有一个 body "base" 或第一个 body 是主要的
                    rb_idx = self.gym.get_actor_rigid_body_index(env_ptr, handle, 0, gymapi.DOMAIN_ENV)
                    env_indices.append(rb_idx)
                while len(env_indices) < max_objs:
                    env_indices.append(-1)
                indices.append(env_indices)
            return torch.tensor(indices, dtype=torch.long, device=self.device)

        self._manip_obj_rh_rb_indices = get_rb_indices(self.objs_handles_rh)
        self._manip_obj_lh_rb_indices = get_rb_indices(self.objs_handles_lh)
        
        # 检查左右侧物体是否完全相同（全局）
        self.is_scene_objects_shared = torch.all(self._manip_obj_rh_rb_indices == self._manip_obj_lh_rb_indices).item()
        if self.is_scene_objects_shared:
            print("[INFO] Scene objects are shared between RH and LH. Support forces will be applied once.")
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

        # [第 3 点] 使用预计算索引，单次索引替代 Python 循环 + torch.stack
        self.rh_states["joints_state"] = self._rigid_body_state[:, self._rh_joints_state_body_indices, :][:, :, :10]

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

        # === 调试断点 1：检查物理仿真状态 ===
        if torch.isnan(rh_obj_states).any() or torch.isnan(lh_obj_states).any():
            print("\n[!!!] 检测到物理仿真产生 NaN !")
            if torch.isnan(rh_obj_states).any():
                nan_envs = torch.where(torch.isnan(rh_obj_states).any(dim=-1).any(dim=-1))[0]
                print(f"右侧故障环境 IDs: {nan_envs.tolist()}")
            if torch.isnan(lh_obj_states).any():
                nan_envs = torch.where(torch.isnan(lh_obj_states).any(dim=-1).any(dim=-1))[0]
                print(f"左侧故障环境 IDs: {nan_envs.tolist()}")

            if self.training:
                import pdb; pdb.set_trace()
            else:
                raise RuntimeError("Physics simulation generated NaN values.")

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
        # [第 3 点] 使用预计算索引，单次索引替代 Python 循环 + torch.stack
        self.lh_states["joints_state"] = self._rigid_body_state[:, self._lh_joints_state_body_indices, :][:, :, :10]
        self.lh_states.update(
            {
                "manip_obj_pos": lh_obj_states[..., :3],
                "manip_obj_quat": lh_obj_states[..., 3:7],
                "manip_obj_vel": lh_obj_states[..., 7:10],
                "manip_obj_ang_vel": lh_obj_states[..., 10:],
            }
        )

    def _refresh(self):
        # [第 4 点优化] 同帧内避免重复刷新；Reset 后通过 _last_refresh_step = -1 强制下次必刷新，保证观测正确
        if not hasattr(self, "_last_refresh_step"):
            self._last_refresh_step = -1
        current_frame = self.gym.get_frame_count(self.sim)
        if current_frame == self._last_refresh_step:
            return

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()
        self._last_refresh_step = current_frame

    def compute_reward(self, actions):
        # [新增] 打印物体重量并设置断点
        if not hasattr(self, "_printed_weights"):
            self._printed_weights = True
            print("\n" + "="*50)
            print(f"[DEBUG] Current Motion Total Frames (Env 0): {self.demo_data_rh['seq_len'][0].item()}")
            print("[DEBUG] Current Object Weights (Env 0):")
            prop = self.gym.get_sim_params(self.sim)
            gravity_z = prop.gravity.z
            for side in ["rh", "lh"]:
                print(f"  Side {side.upper()}:")
                # 获取第一个环境 (env 0) 的物体 handles
                handles = getattr(self, f"objs_handles_{side}")[0]
                env_ptr = self.envs[0]
                
                for i, handle in enumerate(handles):
                    # 通过 IsaacGym 函数直接获取 Rigid Body 属性
                    body_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
                    # 获取第一个 body 的质量 (物体通常只有一个 body)
                    mass = body_props[0].mass
                    weight = mass * -1 * gravity_z
                    print(f"    Object {i}: {weight:.4f} N (Mass: {mass:.4f} kg)")
            print("="*50 + "\n")
            # import pdb; pdb.set_trace()

        lh_rew_buf, lh_reset_buf, lh_success_buf, lh_failure_buf, lh_reward_dict, lh_error_buf, lh_failure_reasons = (
            self.compute_reward_side(actions, side="lh")
        )
        rh_rew_buf, rh_reset_buf, rh_success_buf, rh_failure_buf, rh_reward_dict, rh_error_buf, rh_failure_reasons = (
            self.compute_reward_side(actions, side="rh")
        )
        self.rew_buf = rh_rew_buf + lh_rew_buf
        self.reset_buf = rh_reset_buf | lh_reset_buf
        self.success_buf = rh_success_buf & lh_success_buf
        self.failure_buf = rh_failure_buf | lh_failure_buf
        self.error_buf = rh_error_buf | lh_error_buf
        
        # [新增] 记录每个环境是否发生了特定类型的失败
        self.env_obj_pos_failed |= lh_failure_reasons["obj_pos_failed"] | rh_failure_reasons["obj_pos_failed"]
        self.env_obj_rot_failed |= lh_failure_reasons["obj_rot_failed"] | rh_failure_reasons["obj_rot_failed"]

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
        # [修复] 增加对 cur_idx 的 clamp，防止 progress_buf 超过 demo 长度导致的 indexing 报错
        cur_idx = torch.clamp(self.progress_buf, torch.zeros_like(side_demo_data["seq_len"]), side_demo_data["seq_len"] - 1)

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
        
        # [优化] 使用预处理好的 multi_obj_vel / multi_obj_ang_vel 做向量化索引，替代 Python 双层循环
        multi_obj_vel = self.rh_multi_obj_vel if side == "rh" else self.lh_multi_obj_vel  # [N, K, T, 3]
        multi_obj_ang_vel = self.rh_multi_obj_ang_vel if side == "rh" else self.lh_multi_obj_ang_vel  # [N, K, T, 3]
        target_state["manip_obj_vel"] = multi_obj_vel[batch_idx, obj_idx, time_idx]  # [N, K, 3]
        target_state["manip_obj_ang_vel"] = multi_obj_ang_vel[batch_idx, obj_idx, time_idx]  # [N, K, 3]
        
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

        # 手指接触力模长 [N, 5]，供 tensorboard 记录
        finger_force = torch.norm(target_state["tip_force"], dim=-1)  # [N, 5]
        setattr(self, f"_{side}_finger_force", finger_force)

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

        # [新增] 交互关键点数据
        target_state["tips_closest_obj_idx"] = side_demo_data["tips_closest_obj_idx"][torch.arange(self.num_envs), cur_idx]
        target_state["tips_closest_pt_local"] = side_demo_data["tips_closest_pt_local"][torch.arange(self.num_envs), cur_idx]
        target_state["tips_closest_pt_world"] = side_demo_data["tips_closest_pt_world"][torch.arange(self.num_envs), cur_idx]

        # === [新增] 计算每个环境当前的 bin 索引 ===
        if self.adaptive_sampling_bins is not None:
            seq_lens = side_demo_data["seq_len"].float()
            progress_ratio = self.progress_buf.float() / (seq_lens * 0.98 + 1e-6)
            progress_ratio = torch.clamp(progress_ratio, 0.0, 1.0)
            env_bin_indices = (progress_ratio * self.adaptive_sampling_bins).long()
            env_bin_indices = torch.clamp(env_bin_indices, 0, self.adaptive_sampling_bins - 1)
        else:
            env_bin_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # === [新增] 估算当前 epoch 数 ===
        last_step = self.gym.get_frame_count(self.sim)
        horizon_length = getattr(self, 'horizon_length', 32)
        frames_per_epoch = horizon_length * self.num_envs
        estimated_epoch = int(last_step // frames_per_epoch) if frames_per_epoch > 0 else 0
        if hasattr(self, 'total_train_env_frames') and self.total_train_env_frames is not None:
            estimated_epoch = int(self.total_train_env_frames // frames_per_epoch) if frames_per_epoch > 0 else 0

        if self.training:
            if self.tighten_method == "None":
                scale_factor_val = 1.0
                rot_scale_factor_val = 1.0
            elif self.tighten_method == "const":
                scale_factor_val = self.tighten_factor
                rot_scale_factor_val = self.rot_scale_factor
            elif self.tighten_method == "adaptive_dual":
                # [核心逻辑] adaptive_dual 模式下使用 per-bin 的 scale，不受全局线性衰减的强制约束
                # 这样可以确保每个 bin 的终止判定精度（Termination）是独立自适应的
                scale_factor = self.bin_pos_scale[env_bin_indices]
                current_rot_scale_factor = self.bin_rot_scale[env_bin_indices]
                
                # 全局系数（用于重力补偿、支撑力等）依然遵循线性衰减计划
                scale_factor_val = self.adaptive_global_scale_factor
                rot_scale_factor_val = self.rot_scale_factor
            elif self.tighten_method == "adaptive_real":
                # [核心逻辑] adaptive_real 模式下使用 per-bin 的 scale
                scale_factor = self.bin_pos_scale[env_bin_indices]
                current_rot_scale_factor = self.bin_rot_scale[env_bin_indices]
                
                # 全局系数（用于重力补偿、支撑力等）依然遵循线性衰减计划
                scale_factor_val = self.adaptive_global_scale_factor
                rot_scale_factor_val = self.rot_scale_factor
            elif self.tighten_method == "linear_decay":
                scale_factor_val = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
                rot_scale_factor_val = scale_factor_val
            elif self.tighten_method == "exp_decay":
                scale_factor_val = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
                rot_scale_factor_val = scale_factor_val
            elif self.tighten_method == "cos":
                scale_factor_val = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
                rot_scale_factor_val = scale_factor_val
            else:
                raise NotImplementedError(f"Unknown tighten_method: {self.tighten_method}")
            
            # 如果不是 adaptive_dual 或 adaptive_real，将单值转为 tensor
            if self.tighten_method not in ["adaptive_dual", "adaptive_real"]:
                scale_factor = torch.full((self.num_envs,), scale_factor_val, device=self.device)
                current_rot_scale_factor = torch.full((self.num_envs,), rot_scale_factor_val, device=self.device)
                self.rot_scale_factor = rot_scale_factor_val # 更新全局变量

            if side == "rh":
                self.current_scale_factor = scale_factor_val
        else:
            scale_factor = torch.ones(self.num_envs, device=self.device)
            current_rot_scale_factor = torch.ones(self.num_envs, device=self.device)
            if not hasattr(self, 'current_scale_factor'):
                self.current_scale_factor = 1.0
            
            self.support_force_kp = 0.0
            self.support_force_kd = 0.0
            self.support_force_kp_rot = 0.0
            self.support_force_kd_rot = 0.0

        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)

        # 获取静态物体信息
        obj_is_static = getattr(self, f"manip_obj_{side}_is_static")  # [N, K]
        
        # [新增] 计算每帧固定的进度奖励单价 (100.0 * num_bins / 原始总全长)
        # side_demo_data["seq_len"] 已经是广播到 num_envs 维度的 Tensor 了
        full_motion_length = side_demo_data["seq_len"].to(self.device).float()
        progress_reward_unit = (100.0 * self.adaptive_sampling_bins) / full_motion_length

        rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf, failure_reasons, failure_values = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            side_states,
            target_state,
            max_length,
            scale_factor,
            (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx,
            obj_is_static,
            progress_reward_unit, # 传入计算好的 unit
            self.reward_interact_scale,
            self.terminate_on_eef,
            self.reward_obj_pos_scale,
            self.reward_obj_rot_scale,
            self.reward_finger_tip_force_scale,
            self.terminate_obj_pos_threshold,
            self.terminate_obj_rot_threshold,
            current_rot_scale_factor,
            self.terminate_on_contact,
            self.terminate_obj_pos_final,
            self.terminate_obj_rot_final,
            self.terminate_thumb_threshold,
            self.terminate_index_threshold,
            self.terminate_middle_threshold,
            self.terminate_pinky_threshold,
            self.terminate_ring_threshold,
            self.terminate_level_1_threshold,
            self.terminate_level_2_threshold,
            self.eef_vel_limit,
            self.eef_ang_vel_limit,
            self.joints_vel_limit,
            self.dof_vel_limit,
            self.obj_vel_limit,
            self.obj_ang_vel_limit,
        )

        # [Hack 结束] 恢复原始状态，以免影响其他逻辑
        side_states["manip_obj_pos"] = original_obj_pos
        side_states["manip_obj_quat"] = original_obj_quat
        side_states["manip_obj_vel"] = original_obj_vel
        side_states["manip_obj_ang_vel"] = original_obj_ang_vel
        
        # === [新增] Mode 4: 统计失败原因 ===
        if self.tighten_method == "adaptive_dual" and self.training and failure_buf.any():
            self.obj_pos_fail_count += int(failure_reasons["obj_pos_failed"].sum().item())
            self.obj_rot_fail_count += int(failure_reasons["obj_rot_failed"].sum().item())
            self.total_fail_count += int(failure_buf.sum().item())

        # === [新增] adaptive_real: 统计每个 bin 的失败原因 ===
        if self.tighten_method == "adaptive_real" and self.training and self.adaptive_sampling_bins is not None:
            # 统计每个 bin 的失败率（只在 scale 降到 0.7 之后统计）
            for i in range(self.num_envs):
                if failure_buf[i]:
                    bin_idx = env_bin_indices[i].item()
                    # 只有在已经触底的 bin 才统计失败率
                    if self.bin_reached_bottom[bin_idx]:
                        if failure_reasons["obj_pos_failed"][i]:
                            self.bin_pos_fail_count[bin_idx] += 1
                        if failure_reasons["obj_rot_failed"][i]:
                            self.bin_rot_fail_count[bin_idx] += 1
                
                # 统计所有经过这个 bin 的环境数（用于计算失败率）
                if self.bin_reached_bottom[env_bin_indices[i]]:
                    self.bin_total_pass_count[env_bin_indices[i]] += 1

        # === [新增] 打印失败原因诊断并保存到文件 ===
        if failure_buf.any():
            if estimated_epoch != self.last_failure_log_epoch:
                self.last_failure_log_epoch = estimated_epoch
                self.failure_log_count = 0
            
            if self.failure_log_count < 5:
                failed_env_ids = failure_buf.nonzero(as_tuple=False).flatten().cpu().numpy()
                for env_id in failed_env_ids:
                    if self.failure_log_count >= 5:
                        break
                    
                    env_id_item = int(env_id)
                    cur_step = self.progress_buf[env_id_item].item()
                    running_steps = self.running_progress_buf[env_id_item].item()
                    
                    failure_reasons_list = []
                    
                    # 映射失败原因到展示文本
                    reason_map = [
                        ("obj_pos_failed", "物体位置误差过大", "obj_pos_err", "obj_pos_threshold", "m"),
                        ("obj_rot_failed", "物体旋转误差过大", "obj_rot_err", "obj_rot_threshold", "°"),
                        ("thumb_failed", "拇指位置误差过大", "thumb_tip_dist", "thumb_threshold", "m"),
                        ("index_failed", "食指位置误差过大", "index_tip_dist", "index_threshold", "m"),
                        ("middle_failed", "中指位置误差过大", "middle_tip_dist", "middle_threshold", "m"),
                        ("pinky_failed", "小指位置误差过大", "pinky_tip_dist", "pinky_threshold", "m"),
                        ("ring_failed", "无名指位置误差过大", "ring_tip_dist", "ring_threshold", "m"),
                        ("level_1_failed", "Level 1 关节位置误差过大", "level_1_dist", "level_1_threshold", "m"),
                        ("level_2_failed", "Level 2 关节位置误差过大", "level_2_dist", "level_2_threshold", "m"),
                        ("eef_pos_failed", "[EEF Term] 手腕位置误差过大", "eef_pos_err", "eef_pos_threshold", "m"),
                        ("eef_rot_failed", "[EEF Term] 手腕旋转误差过大", "eef_rot_err_deg", "eef_rot_threshold_deg", "°"),
                        ("eef_vel_failed", "[EEF Term] 手腕线速度误差过大", "eef_vel_err", "eef_vel_threshold", "m/s"),
                        ("eef_ang_vel_failed", "[EEF Term] 手腕角速度误差过大", "eef_ang_vel_err", "eef_ang_vel_threshold", "rad/s"),
                        ("error_eef_vel", "[EEF Abnormal] 手腕线速度异常", "eef_vel_norm", None, "m/s", self.eef_vel_limit),
                        ("error_eef_ang_vel", "[EEF Abnormal] 手腕角速度异常", "eef_ang_vel_norm", None, "rad/s", self.eef_ang_vel_limit),
                        ("error_joints_vel", "关节速度异常", "joints_vel_norm", None, "m/s", self.joints_vel_limit),
                        ("error_dof_vel", "DOF速度异常", "dof_vel_norm", None, "rad/s", self.dof_vel_limit),
                        ("error_obj_vel", "物体线速度异常", "obj_vel_norm", None, "m/s", self.obj_vel_limit),
                        ("error_obj_ang_vel", "物体角速度异常", "obj_ang_vel_norm", None, "rad/s", self.obj_ang_vel_limit),
                    ]

                    for key, label, val_key, threshold_key, unit, *extra in reason_map:
                        if failure_reasons[key][env_id_item]:
                            val = failure_values[val_key][env_id_item].item()
                            if threshold_key:
                                threshold = failure_values[threshold_key][env_id_item].item()
                            else:
                                # 确保 threshold 是 float 而非 Tensor
                                threshold = float(extra[0])
                            failure_reasons_list.append(f"  - {label}: {val:.4f} > {threshold:.4f} {unit}")
                    
                    if failure_reasons["contact_violation"][env_id_item]:
                        failure_reasons_list.append("  - 接触惩罚: 手指距离过近但未检测到接触")

                    # 累积到 buffer，每 20 个 epoch 统一写入文件
                    if hasattr(self, 'failure_log_file') and self.failure_log_file is not None and hasattr(self, '_failure_log_buffer'):
                        try:
                            cur_bin_str = ""
                            if self.adaptive_sampling_bins is not None:
                                cur_max_len = max_length[env_id_item].item()
                                if cur_max_len > 0:
                                    cur_bin = min(int((cur_step / cur_max_len) * self.adaptive_sampling_bins), self.adaptive_sampling_bins - 1)
                                    cur_bin_str = f" (Bin {cur_bin})"
                                else:
                                    cur_bin_str = " (Bin N/A)"
                            env_scale = scale_factor[env_id_item].item()
                            env_rot_scale = current_rot_scale_factor[env_id_item].item()
                            lines = [
                                f"\n[FAILURE DIAGNOSIS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                                f"Env {env_id_item} ({side.upper()} side), Step {cur_step}{cur_bin_str}, Running {running_steps} steps\n",
                                f"  Estimated Epoch: {estimated_epoch} (Frame: {last_step}, Frames/Epoch: {frames_per_epoch})\n",
                            ]
                            if failure_reasons_list:
                                lines.extend([reason + "\n" for reason in failure_reasons_list])
                            else:
                                lines.append(f"  - 未知原因 (可能是 running_progress_buf < 8)\n")
                            lines.append(f"  Scale Factor (Pos): {env_scale:.4f}, (Rot): {env_rot_scale:.4f}\n")
                            lines.append("-" * 80 + "\n")
                            self._failure_log_buffer.extend(lines)
                        except Exception as e:
                            print(f"[WARNING] Failed to append to failure log buffer: {e}")
                    
                    self.failure_log_count += 1

        self.total_rew_buf += rew_buf
        return rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf, failure_reasons


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
                    # [第 5 点] 使用缓存的重力，避免每步 get_sim_params；在 _update_adaptive_difficulty 中会更新 _gravity_z
                    if getattr(self, "_gravity_z", None) is None:
                        self._gravity_z = float(self.gym.get_sim_params(self.sim).gravity.z)
                    weights = getattr(self, f"manip_obj_{side}_mass") * (-self._gravity_z)
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

        # 4. 计算速度 Delta - [优化] 使用预处理好的 multi_obj_vel 做 indicing，替代 Python 双层循环
        raw_multi_vel = self.rh_multi_obj_vel if side == "rh" else self.lh_multi_obj_vel  # [N, K, T, 3]
        multi_vel_time_first = raw_multi_vel.transpose(1, 2)  # [N, T, K, 3] 以适配 indicing
        target_obj_vel = indicing(multi_vel_time_first, cur_idx)  # [N, F, K, 3]
            
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

        # 6. 计算角速度 Delta - [优化] 使用预处理好的 multi_obj_ang_vel 做 indicing，替代 Python 双层循环
        raw_multi_ang_vel = self.rh_multi_obj_ang_vel if side == "rh" else self.lh_multi_obj_ang_vel  # [N, K, T, 3]
        multi_ang_vel_time_first = raw_multi_ang_vel.transpose(1, 2)  # [N, T, K, 3] 以适配 indicing
        target_obj_ang_vel = indicing(multi_ang_vel_time_first, cur_idx)  # [N, F, K, 3]

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
            if self.adaptive_sampling_bins is not None:
                # === [修改] 使用自适应采样 ===
                # 为每个环境采样进度比例
                sampled_progress_ratios, sampled_bins = self._adaptive_sampling(env_ids)

                # [新增] 根据比例强制部分环境从最开始 (Bin 0) 开始 reset
                if self.start_from_beginning_ratio > 0:
                    num_to_reset = len(env_ids)
                    reset_rand = torch.rand(num_to_reset, device=self.device)
                    from_beginning_mask = reset_rand < self.start_from_beginning_ratio
                    if from_beginning_mask.any():
                        sampled_progress_ratios[from_beginning_mask] = 0.0
                        sampled_bins[from_beginning_mask] = 0

                # 记录采样的bin索引，用于后续统计从起点到终点跨越的所有bin
                self.env_start_bin_index[env_ids] = sampled_bins

                # 根据序列长度计算实际的序列索引
                if self.rollout_begin is not None:
                    # 如果设置了rollout_begin，需要在这个基础上偏移
                    max_progress = self.rollout_len * 0.98
                    seq_idx = (
                        torch.floor(max_progress * sampled_progress_ratios).long()
                        + self.rollout_begin
                    )
                    seq_idx = torch.clamp(
                        seq_idx,
                        torch.zeros(1, device=self.device).long(),
                        torch.floor(self.demo_data_rh["seq_len"][env_ids] * 0.98).long(),
                    )
                else:
                    # 直接根据序列长度采样
                    seq_idx = torch.floor(
                        self.demo_data_rh["seq_len"][env_ids]
                        * 0.98
                        * sampled_progress_ratios
                    ).long()
            else:
                # === [原有逻辑] 均匀随机采样 ===
                if self.rollout_begin is not None:
                    max_progress_ratios = torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                    seq_idx = (
                        torch.floor(
                            self.rollout_len * 0.98 * max_progress_ratios
                        ).long()
                        + self.rollout_begin
                    )
                    seq_idx = torch.clamp(
                        seq_idx,
                        torch.zeros(1, device=self.device).long(),
                        torch.floor(self.demo_data_rh["seq_len"][env_ids] * 0.98).long(),
                    )
                    # 如果启用了 bin 统计，记录起始 bin
                    if self.adaptive_sampling_bins is not None:
                        self.env_start_bin_index[env_ids] = torch.floor(max_progress_ratios * self.adaptive_sampling_bins).long().clamp(0, self.adaptive_sampling_bins - 1)
                else:
                    max_progress_ratios = torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                    seq_idx = torch.floor(
                        self.demo_data_rh["seq_len"][env_ids]
                        * 0.98
                        * max_progress_ratios
                    ).long()
                    # 如果启用了 bin 统计，记录起始 bin
                    if self.adaptive_sampling_bins is not None:
                        self.env_start_bin_index[env_ids] = torch.floor(max_progress_ratios * self.adaptive_sampling_bins).long().clamp(0, self.adaptive_sampling_bins - 1)
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data_rh["seq_len"][env_ids].long())
                # 计算对应的 bin 索引
                if self.adaptive_sampling_bins is not None:
                    # 对于 rollout_begin，需要计算它相对于总长度的比例
                    # 这里简化处理，直接计算 bin
                    for i, env_id in enumerate(env_ids):
                        seq_len = self.demo_data_rh["seq_len"][env_id].item()
                        if seq_len > 0:
                            ratio = self.rollout_begin / (seq_len * 0.98)
                            bin_idx = min(int(ratio * self.adaptive_sampling_bins), self.adaptive_sampling_bins - 1)
                            self.env_start_bin_index[env_id] = bin_idx
            else:
                seq_idx = torch.zeros_like(self.demo_data_rh["seq_len"][env_ids].long())
                if self.adaptive_sampling_bins is not None:
                    self.env_start_bin_index[env_ids] = 0

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
        if not self.training:
            return

        # === [新增] 更新自适应采样统计 ===
        if self.adaptive_sampling_bins is not None:
            # 为所有重置的环境记录采样统计
            for env_id in env_ids:
                env_id_item = env_id.item()
                # 获取该环境的进度和序列长度
                current_progress = self.progress_buf[env_id_item].item()
                seq_length = self.demo_data_rh["seq_len"][env_id_item].item()

                if seq_length > 0:
                    # 计算进度比例 (0.0 到 1.0)
                    # 注意：reset时使用了 0.98 的缩放 (seq_idx = seq_length * 0.98 * ratio)
                    # 为了保证起点 bin 索引一致，这里也使用同样的缩放
                    progress_ratio = current_progress / (seq_length * 0.98)
                    progress_ratio = min(max(progress_ratio, 0.0), 1.0)
                    
                    # 映射到当前bin索引（失败或成功的位置）
                    end_bin = min(int(progress_ratio * self.adaptive_sampling_bins), self.adaptive_sampling_bins - 1)
                    # 获取起始bin索引
                    start_bin = self.env_start_bin_index[env_id_item].item()

                    # 记录重置次数（记录在起始bin上）
                    self._current_bin_reset[start_bin] += 1

                    # 记录成功或失败的统计逻辑：
                    # 1. 从起始bin到失败bin的前一个bin，都视为成功通过
                    # 2. 如果是失败，则失败bin只增加总数不增加成功数
                    # 3. 如果是成功，则失败bin（此时是结束位置）也增加成功数
                    is_success = self.success_buf[env_id_item].item() > 0
                    obj_pos_failed = self.env_obj_pos_failed[env_id_item].item()
                    obj_rot_failed = self.env_obj_rot_failed[env_id_item].item()
                    
                    # 确保 start_bin <= end_bin
                    actual_start = min(start_bin, end_bin)
                    
                    if is_success:
                        # 成功：从起始到结束所有bin都累加成功
                        for i in range(actual_start, end_bin + 1):
                            self._current_bin_success[i] += 1
                            self._current_bin_total[i] += 1
                            self._current_bin_obj_pos_pass[i] += 1
                            self._current_bin_obj_rot_pass[i] += 1
                    else:
                        # 失败：起始到失败前一个bin累加成功
                        for i in range(actual_start, end_bin):
                            self._current_bin_success[i] += 1
                            self._current_bin_total[i] += 1
                            self._current_bin_obj_pos_pass[i] += 1
                            self._current_bin_obj_rot_pass[i] += 1
                        # 当前bin累加一次失败（只加total）
                        self._current_bin_total[end_bin] += 1
                        # 如果当前bin失败不是因为 obj_pos/obj_rot，则计入通过
                        if not obj_pos_failed:
                            self._current_bin_obj_pos_pass[end_bin] += 1
                        if not obj_rot_failed:
                            self._current_bin_obj_rot_pass[end_bin] += 1
        
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
                
                # 获取当前的通过率判定系数用于打印
                pos_scale_val = self.bin_pos_scale.mean().item() if self.tighten_method in ["adaptive_dual", "adaptive_real"] else self.adaptive_global_scale_factor
                rot_scale_val = self.bin_rot_scale.mean().item() if self.tighten_method in ["adaptive_dual", "adaptive_real"] else self.rot_scale_factor

                # 每 20 个 epoch 打印一次，并刷新失败日志到文件
                if self.adaptive_current_epoch % 20 == 0:
                    if getattr(self, '_failure_log_buffer', None) is not None and hasattr(self, 'failure_log_file') and self.failure_log_file is not None and len(self._failure_log_buffer) > 0:
                        try:
                            with open(self.failure_log_file, "a", encoding="utf-8") as f:
                                for line in self._failure_log_buffer:
                                    f.write(line)
                            self._failure_log_buffer.clear()
                        except Exception as e:
                            print(f"[WARNING] Failed to flush failure log buffer: {e}")
                    print(f"[EPOCH STATS] Epoch {self.adaptive_current_epoch}: "
                          f"Success Rate: {success_rate_last_epoch:.4f}, "
                          f"Avg Steps: {avg_steps_last_epoch:.2f}, "
                          f"Global Scale: {self.adaptive_global_scale_factor:.4f}, "
                          f"Obj-Pos Scale: {pos_scale_val:.4f}, "
                          f"Obj-Rot Scale: {rot_scale_val:.4f}")
                
                # === [新增] 更新各 bin 成功率统计，不论什么模式都执行 ===
                if self.adaptive_sampling_bins is not None:
                    # 累积 epoch 计数
                    self.bin_accumulated_epochs_since_reset += 1

                    # 1. 计算当前 epoch 的各 bin 成功率
                    current_epoch_bin_rates = torch.where(
                        self._current_bin_total > 0,
                        self._current_bin_success / self._current_bin_total,
                        self.bin_ema_success_rates 
                    )
                    self.current_epoch_bin_rates = current_epoch_bin_rates.clone() 
                    
                    # [新增] 如果 global_scale 降至 0.7 之后，且 support force 已衰减到 0，根据所有 bin 的成功率动态增加从头开始的比例
                    if self.adaptive_global_scale_factor <= 0.7001:
                        end_decay_epoch_sf = 30 * (1024 / self.num_envs) * 100  # 与 set_adaptive_scale_factor 中 support force 归零的 epoch 一致
                        if (current_epoch >= end_decay_epoch_sf) and torch.all(current_epoch_bin_rates > 0.60):
                            self.start_from_beginning_ratio = min(1.0, self.start_from_beginning_ratio + 0.005)
                            print(f"[ADAPTIVE RESET] Support force reached 0, all bin rates > 0.60, increasing start_from_beginning_ratio to {self.start_from_beginning_ratio:.4f}")
                    
                    # [新增] 计算当前 epoch 的各 bin obj-pos 和 obj-rot 通过率
                    self.bin_obj_pos_pass_rate = torch.where(
                        self._current_bin_total > 0,
                        self._current_bin_obj_pos_pass / self._current_bin_total,
                        torch.zeros_like(self.bin_obj_pos_pass_rate)
                    )
                    self.bin_obj_rot_pass_rate = torch.where(
                        self._current_bin_total > 0,
                        self._current_bin_obj_rot_pass / self._current_bin_total,
                        torch.zeros_like(self.bin_obj_rot_pass_rate)
                    )
                    
                    # 2. 更新 EMA 成功率
                    update_mask = self._current_bin_total > 0
                    if torch.any(update_mask):
                        self.bin_ema_success_rates[update_mask] = (
                            0.3 * (self._current_bin_success[update_mask] / self._current_bin_total[update_mask]) + 
                            0.7 * self.bin_ema_success_rates[update_mask]
                        )
                    
                    # 记录整体 bin 统计历史
                    self.bin_success_count += self._current_bin_success
                    self.bin_total_count += self._current_bin_total
                    self.bin_reset_count += self._current_bin_reset

                # === [新增] 更新历史记录和最小稳定 Scale 变量 ===
                if self.adaptive_sampling_bins is not None:
                    # 更新 30 epoch 的通过率历史
                    self.bin_pos_pass_history[:, self.bin_history_ptr_30] = (self.bin_obj_pos_pass_rate > 0.60)
                    self.bin_rot_pass_history[:, self.bin_history_ptr_30] = (self.bin_obj_rot_pass_rate > 0.60)
                    self.bin_history_ptr_30 = (self.bin_history_ptr_30 + 1) % 30
                    
                    # 更新 50 epoch 的成功率历史
                    self.bin_success_history_50[:, self.bin_history_ptr_50] = self.current_epoch_bin_rates
                    self.bin_history_ptr_50 = (self.bin_history_ptr_50 + 1) % 50

                    # [修改] 计算 30 epoch 内的平均成功率 (取最近 30 个)
                    indices = torch.tensor([(self.bin_history_ptr_50 - 1 - k) % 50 for k in range(30)], device=self.device)
                    bin_success_avg_30 = self.bin_success_history_50.index_select(1, indices).mean(dim=1)

                    # [修改] Best Scale 判定条件：(累积满 30 epoch 且 30 epoch 平均成功率 > 45%) 或 EMA > 50%
                    avg_30_ok = (self.bin_accumulated_epochs_since_reset >= 30) & (bin_success_avg_30 > 0.45)
                    pos_stable_mask = avg_30_ok | (self.bin_ema_success_rates > 0.50)
                    rot_stable_mask = pos_stable_mask

                    # pos
                    current_pos_scales = self.bin_pos_scale
                    # [修改] 更新条件：除了满足稳定条件外，如果当前 epoch 表现极好 (>50%) 且比历史最好更小，也允许更新 best_stable
                    # 这样可以防止在未满 35 epoch 时发生回退导致跳回 1.0 的问题
                    update_pos_mask = (pos_stable_mask | (self.current_epoch_bin_rates > 0.55)) & (current_pos_scales < self.bin_pos_stable_best)
                    self.bin_pos_stable_best[update_pos_mask] = current_pos_scales[update_pos_mask]
                    
                    # rot
                    current_rot_scales = self.bin_rot_scale
                    update_rot_mask = (rot_stable_mask | (self.current_epoch_bin_rates > 0.55)) & (current_rot_scales < self.bin_rot_stable_best)
                    self.bin_rot_stable_best[update_rot_mask] = current_rot_scales[update_rot_mask]

                # 评估并调整难度
                scale_changed = False
                
                # === [新增] 处理系数更新 ===
                if self.tighten_method in ["adaptive_dual", "adaptive_real"]:
                    # [修改] 如果是 adaptive_dual 或 adaptive_real，线性降到 0.7
                    # 使用 900 * (1024 / num_envs) 作为 target_epoch for adaptive_real (例如 128 envs 时为 7200)
                    # 使用 1800 * (1024 / num_envs) 作为 target_epoch for adaptive_dual
                    if self.tighten_method == "adaptive_real":
                        cur_target_epoch = 900 * (1024 / self.num_envs)
                        # [adaptive_real] 触底后，global 不再变化
                        all_bins_reached_bottom = hasattr(self, 'bin_reached_bottom') and torch.all(self.bin_reached_bottom)
                        if not all_bins_reached_bottom:
                            # [修改] 改为阶梯下降逻辑，与 bin scale 同步
                            epochs_per_update = int(30 * (1024 / self.num_envs))
                            num_updates = current_epoch // epochs_per_update
                            new_scale = max(0.7, 1.0 - 0.01 * num_updates)
                        else:
                            # 触底后不再改变 global scale
                            new_scale = self.adaptive_global_scale_factor
                    else: # adaptive_dual
                        cur_target_epoch = 1800 * (1024 / self.num_envs)
                        progress = current_epoch / cur_target_epoch
                        progress = max(0.0, min(1.0, progress))
                        new_scale = 1.0 - (1.0 - 0.7) * progress
                    
                    if abs(new_scale - self.adaptive_global_scale_factor) > 1e-5:
                        # [同步修改] 计算增量并应用到旋转系数
                        delta = new_scale - self.adaptive_global_scale_factor
                        self.adaptive_global_scale_factor = new_scale
                        self.rot_scale_factor = max(0.7, min(1.0, self.rot_scale_factor + delta))
                        
                        scale_changed = True
                        
                        # [adaptive_real] 同时更新所有 bin 的 pos/rot scale（在前900轮）
                        if self.tighten_method == "adaptive_real" and self.adaptive_sampling_bins is not None:
                            # 既然已经进入阶梯更新（new_scale 发生了变化），直接同步到 bin scale
                            if current_epoch <= cur_target_epoch:
                                # 按照阶梯下降规则，同步到 new_scale
                                self.bin_pos_scale.fill_(new_scale)
                                self.bin_rot_scale.fill_(new_scale)
                                print(f"[ADAPTIVE REAL] Epoch {current_epoch}: All bin pos/rot scales updated to {new_scale:.4f} (Step-wise)")
                            
                            # 检查是否所有 bin 都已经触底
                            if torch.all(self.bin_pos_scale <= 0.7001) and torch.all(self.bin_rot_scale <= 0.7001) and not torch.all(self.bin_reached_bottom):
                                self.bin_reached_bottom[:] = True
                                print(f"[ADAPTIVE REAL] Epoch {current_epoch}: All bins reached bottom (0.70). Starting failure rate tracking.")
                        
                        if self.tighten_method == "adaptive_dual":
                            print(f"[ADAPTIVE DUAL] Epoch {current_epoch}: Global Scale decayed to {new_scale:.4f}")
                        elif self.tighten_method == "adaptive_real" and not all_bins_reached_bottom:
                            print(f"[ADAPTIVE REAL] Epoch {current_epoch}: Global Scale decayed to {new_scale:.4f}")

                # === [修改逻辑] 难度调整规则 ===
                # Mode 4 Combined Logic
                if self.tighten_method == "adaptive_dual" and self.random_state_init and self.adaptive_sampling_bins is not None:
                    ema_bin_rates = self.bin_ema_success_rates
                    self.epochs_at_current_scale += 1
                    
                    # [新增] Warmup 逻辑：前 35 个 epoch 不调整难度，且在之后重新从 0 开始统计
                    if self.adaptive_difficulty_is_warming_up:
                        if self.epochs_at_current_scale <= 35:
                            pos_scale_val = self.bin_pos_scale.mean().item() if self.tighten_method in ["adaptive_dual", "adaptive_real"] else self.adaptive_global_scale_factor
                            rot_scale_val = self.bin_rot_scale.mean().item() if self.tighten_method in ["adaptive_dual", "adaptive_real"] else self.rot_scale_factor
                            print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Warmup phase ({self.epochs_at_current_scale}/35). Global Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Pos Scale: {pos_scale_val:.4f}, Obj-Rot Scale: {rot_scale_val:.4f}. Stats recorded for logging but no adjustment.")
                        else:
                            # 35 epoch warmup 结束，进入正式评估期
                            self.adaptive_difficulty_is_warming_up = False
                            self.epochs_at_current_scale = 1
                            self.bin_success_sum_at_current_scale.zero_()
                            self.low_success_history.clear()
                            self.stuck_epoch_counter = 0
                            self.adaptive_epoch_success_rate_history.clear()
                            self.adaptive_epoch_steps_history.clear()
                            pos_scale_val = self.bin_pos_scale.mean().item() if self.tighten_method in ["adaptive_dual", "adaptive_real"] else self.adaptive_global_scale_factor
                            rot_scale_val = self.bin_rot_scale.mean().item() if self.tighten_method in ["adaptive_dual", "adaptive_real"] else self.rot_scale_factor
                            print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Warmup finished. Starting evaluation from zero. Global Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Pos Scale: {pos_scale_val:.4f}, Obj-Rot Scale: {rot_scale_val:.4f}")

                    if not self.adaptive_difficulty_is_warming_up:
                        self.bin_success_sum_at_current_scale += current_epoch_bin_rates
                        
                    # 3. 检查难度回退条件 (Make Easier)
                    any_bin_low = torch.any(current_epoch_bin_rates < 0.20).item()
                    if not self.adaptive_difficulty_is_warming_up:
                        self.low_success_history.append(any_bin_low)
                    
                    low_count = sum(self.low_success_history)
                    
                    # Mode 3: No Regression Check
                    can_make_easier = True
                    if self.no_regression_threshold > 0:
                        limit_scale = 0.7 * (self.no_regression_threshold ** (1.0/3.0))
                        if self.adaptive_global_scale_factor <= limit_scale:
                            can_make_easier = False

                    # 修改后的回退条件：warmup过后的30个epoch内，如果累计出现10次以上某个bin低于20%成功率
                    if not self.adaptive_difficulty_is_warming_up and self.epochs_at_current_scale <= 30 and low_count >= 10 and self.difficulty_increase_timer <= 0:
                        if can_make_easier:
                            # [新增] 检查该难度系数是否已经回升过
                            current_scale_key = round(float(self.adaptive_global_scale_factor), 4)
                            if self.scale_increase_counts.get(current_scale_key, 0) < 1:
                                # [同步修改] 难度回退时，两个系数同步增加
                                if self.tighten_method != "adaptive_dual":
                                    self.adaptive_global_scale_factor = min(self.adaptive_scale_factor_max, self.adaptive_global_scale_factor + 0.01)
                                    self.rot_scale_factor = min(1.0, self.rot_scale_factor + 0.01)
                                
                                scale_changed = True
                                self.difficulty_increase_timer = 5
                                self.scale_increase_counts[current_scale_key] = self.scale_increase_counts.get(current_scale_key, 0) + 1
                                print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Coefficient Rises (+0.01) due to low success ({low_count}/30 in evaluation). Global Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Pos Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Rot Scale: {self.rot_scale_factor:.4f}")
                                print(f"  -> Scale {current_scale_key:.4f} increase count: {self.scale_increase_counts[current_scale_key]}")
                            else:
                                print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Prevented Coefficient Rise. Scale {current_scale_key:.4f} has already been increased {self.scale_increase_counts[current_scale_key]} times.")
                        else:
                            print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Prevented Coefficient Rise (No Regression Mode). Global Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Pos Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Rot Scale: {self.rot_scale_factor:.4f} <= Limit: {limit_scale:.4f}")

                    if not scale_changed and self.difficulty_increase_timer > 0:
                        self.difficulty_increase_timer -= 1
                    
                    # 4. 检查难度提升条件 (Make Harder - Standard)
                    all_bins_ema_high = torch.all(ema_bin_rates > self.adaptive_sampling_all_bins_threshold).item()
                    avg_rates_at_current_scale = self.bin_success_sum_at_current_scale / self.epochs_at_current_scale
                    min_avg_rate = torch.min(avg_rates_at_current_scale).item()
                    fallback_trigger = (self.epochs_at_current_scale >= 50 and min_avg_rate >= 0.40)
                    stuck_trigger = (self.epochs_at_current_scale >= self.adaptive_stuck_threshold)

                    if not scale_changed:
                        if all_bins_ema_high or fallback_trigger or stuck_trigger:
                            # [同步修改] 难度提升时，两个系数同步减小
                            if self.tighten_method != "adaptive_dual":
                                self.adaptive_global_scale_factor = max(self.adaptive_scale_factor_min, self.adaptive_global_scale_factor - 0.01)
                                self.rot_scale_factor = max(0.7, self.rot_scale_factor - 0.01)
                            
                            scale_changed = True
                            
                            reason = "all bins EMA high" if all_bins_ema_high else ("fallback trigger" if fallback_trigger else "stuck trigger")
                            print(f"[ADAPTIVE] Epoch {self.adaptive_current_epoch}: Coefficient Falls (-0.01) due to {reason}. Global Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Pos Scale: {self.adaptive_global_scale_factor:.4f}, Obj-Rot Scale: {self.rot_scale_factor:.4f}")

                    # 5. Mode 4: Specialized Per-Bin Independent Adaptive Dual Logic
                    if self.tighten_method == "adaptive_dual":
                        # [Per-bin State Machine Engine]
                        num_bins = self.adaptive_sampling_bins
                        for i in range(num_bins):
                            # 获取 Bin 的 EMA 成功率
                            bin_ema = self.bin_ema_success_rates[i].item()
                            
                            # 获取 pos 和 rot 的状态
                            pos_state = self.bin_state[i, 0].item()
                            rot_state = self.bin_state[i, 1].item()
                            
                            # 提前计算恢复条件 [修改后的阈值]
                            # [修改] 使用 Avg_30 替代 Avg_50，且要求必须满 30 epoch
                            avg_30_recover_ok = (self.bin_accumulated_epochs_since_reset[i] >= 30) and (bin_success_avg_30[i] >= 0.55)
                            can_recover_base = avg_30_recover_ok

                            # 0. Terminal/Frozen 恢复检测
                            for c_idx in range(2):
                                state = self.bin_state[i, c_idx].item()
                                if state == 3 or state == 4:
                                    component = "pos" if c_idx == 0 else "rot"
                                    other_idx = 1 - c_idx
                                    
                                    # 当前自己的 scale
                                    current_scale = self.bin_pos_scale[i].item() if c_idx == 0 else self.bin_rot_scale[i].item()
                                    scale_key = round(float(current_scale), 4)
                                    
                                    other_scale = self.bin_pos_scale[i].item() if other_idx == 0 else self.bin_rot_scale[i].item()
                                    scale_at_freeze = self.bin_other_scale_at_freeze[i, c_idx].item()
                                    
                                    # 检查解冻条件：阈值满足 + 另一个变量 scale 降低 + 未超过该 scale 的解冻次数限制
                                    unfreeze_count = self.bin_unfreeze_history[i][component].get(scale_key, 0)
                                    scale_dropped = other_scale < scale_at_freeze - 1e-5 # 容差处理
                                    count_ok = unfreeze_count < 2
                                    
                                    if can_recover_base and scale_dropped and count_ok:
                                        self.bin_state[i, c_idx] = 0 # 恢复 Normal
                                        self.bin_epochs_at_scale[i, c_idx] = 0 # [修复] 重置在该 scale 下的停留计时
                                        self.bin_unfreeze_history[i][component][scale_key] = unfreeze_count + 1
                                        print(f"[ADAPTIVE DUAL] Bin {i} {component} naturally recovered at scale {scale_key:.4f} (Count {unfreeze_count+1}/2). "
                                              f"Other scale: {other_scale:.4f} < {scale_at_freeze:.4f}. Phase -> Normal.")
                                        self._reset_bin_histories(i)

                            # 刷新局部状态变量
                            pos_state = self.bin_state[i, 0].item()
                            rot_state = self.bin_state[i, 1].item()

                            # [重构] 提前计算解冻判定条件
                            bin_any_scale_high = (self.bin_pos_scale[i] > 0.9) or (self.bin_rot_scale[i] > 0.9)
                            bin_auto_unfreeze = (self.adaptive_global_scale_factor > 0.9) or bin_any_scale_high

                            # 1. 终局锁死判定 (只要二者同时处于 Frozen，不论通过率直接进入 Terminal)
                            if pos_state == 3 and rot_state == 3:
                                if bin_auto_unfreeze:
                                    self.bin_state[i, 0] = 0 # 恢复 Normal
                                    self.bin_state[i, 1] = 0 # 恢复 Normal
                                    self.bin_epochs_at_scale[i, 0] = 0
                                    self.bin_epochs_at_scale[i, 1] = 0
                                    print(f"[ADAPTIVE DUAL] Bin {i} both Frozen but auto unfreeze triggered. Phase -> Normal.")
                                    self._reset_bin_histories(i)
                                    # 刷新本地变量以便后续逻辑
                                    pos_state, rot_state = 0, 0
                                else:
                                    # 进入 Terminal 时记录对手当前的 scale (虽然 Frozen 时理论上已记录，这里作为二次确认)
                                    if self.bin_state[i, 0] != 4:
                                        self.bin_other_scale_at_freeze[i, 0] = self.bin_rot_scale[i]
                                    if self.bin_state[i, 1] != 4:
                                        self.bin_other_scale_at_freeze[i, 1] = self.bin_pos_scale[i]
                                        
                                    self.bin_state[i, 0] = 4 # Terminal
                                    self.bin_state[i, 1] = 4 # Terminal

                                    print(f"[ADAPTIVE DUAL] Bin {i} entered TERMINAL (Both Frozen). Locked to BEST STABLE.")
                                    self._reset_bin_histories(i)
                                    continue # 跳过该 Bin 后续处理

                            # 2. 状态机逻辑循环 (处理 pos 和 rot)
                            components = ["pos", "rot"]
                            starter = self.bin_next_trial_starter[i].item()
                            order = [starter, 1 - starter]

                            for c_idx in order:
                                component = components[c_idx]
                                other_idx = 1 - c_idx
                                state = self.bin_state[i, c_idx].item()
                                other_state = self.bin_state[i, other_idx].item()
                                
                                raw_scale = self.bin_pos_scale[i].item() if c_idx == 0 else self.bin_rot_scale[i].item()
                                global_scale = self.adaptive_global_scale_factor if c_idx == 0 else self.rot_scale_factor
                                
                                # [修改] 自动解冻逻辑：满足全局或任一局部 scale > 0.9 时解冻
                                if state == 3 and bin_auto_unfreeze:
                                    self.bin_state[i, c_idx] = 0 # 恢复 Normal
                                    self.bin_epochs_at_scale[i, c_idx] = 0
                                    print(f"[ADAPTIVE DUAL] Bin {i} {component} auto unfreeze. Phase -> Normal.")
                                    self._reset_bin_histories(i)
                                    state = 0 # 更新本地状态变量以便后续逻辑使用

                                # === [阶段 0: Normal] ===
                                if state == 0:
                                    # 对齐逻辑 (渐进)
                                    # [修改] 增加 35 epoch 缓冲区，确保每个 scale 都有足够的评估时间
                                    if global_scale < raw_scale and can_recover_base and self.bin_epochs_at_scale[i, c_idx] >= 35:
                                        new_raw_scale = max(global_scale, raw_scale - 0.01)
                                        if c_idx == 0:
                                            self.bin_pos_scale[i] = new_raw_scale
                                        else:
                                            self.bin_rot_scale[i] = new_raw_scale
                                        self.bin_epochs_at_scale[i, c_idx] = 0 # [修复] Scale 改变，重置计时
                                        print(f"[ADAPTIVE DUAL] Bin {i} {component} Normal gradual descent to {new_raw_scale:.4f}")
                                        self._reset_bin_histories(i) # [恢复] 每次 Scale 改变都重置历史，确保新 Scale 下有准确评估数据
                                    
                                    # 触发尝试期检查
                                    if other_state in [0, 3, 4]:
                                        self.bin_epochs_at_scale[i, c_idx] += 1
                                        is_stuck = self.bin_epochs_at_scale[i, c_idx] >= self.adaptive_stuck_threshold
                                        
                                        any_bin_low_in_epoch = torch.any(current_epoch_bin_rates < 0.20).item()
                                        low_count_global = sum(self.low_success_history)
                                        expect_rise = (any_bin_low_in_epoch and self.epochs_at_current_scale <= 30 and low_count_global >= 10)
                                        
                                        current_scale_key = round(float(raw_scale), 4)
                                        rise_already_done = self.bin_scale_increase_history[i][component].get(current_scale_key, 0) >= 1
                                        rise_rejected = expect_rise and rise_already_done
                                        
                                        # [修改] 如果 global_scale 为 1 或者已经跑过的 epoch 数还未超过 200，那么就忽略 stuck 条件
                                        ignore_stuck = (abs(global_scale - 1.0) < 1e-6) or (self.adaptive_current_epoch < 200)
                                        trial_trigger = rise_rejected if ignore_stuck else (is_stuck or rise_rejected)

                                        if trial_trigger and bin_ema < 0.3 and self.bin_epochs_at_scale[i, c_idx] >= 35:
                                            # 轮流制：设置当前为 first_mover，并切换下一次的 starter
                                            self.bin_state[i, c_idx] = 1 # TrialB
                                            # [修改] 保护 Frozen 状态：如果对手是 Frozen/Terminal，保持不变；否则改为 Pending
                                            if other_state not in [3, 4]:
                                                self.bin_state[i, other_idx] = 5 # Pending

                                            self.bin_first_mover[i] = c_idx
                                            self.bin_next_trial_starter[i] = 1 - c_idx # 切换下一次启动者
                                            self.bin_ema_at_start[i, c_idx] = bin_ema
                                            self.bin_initial_scale_before_trial[i, c_idx] = raw_scale
                                            self.bin_trial_start_epoch[i, c_idx] = self.adaptive_current_epoch
                                            
                                            new_scale = max(0.7, raw_scale - 0.04)
                                            if c_idx == 0:
                                                self.bin_pos_scale[i] = new_scale
                                            else:
                                                self.bin_rot_scale[i] = new_scale
                                            
                                            self.bin_epochs_at_scale[i, c_idx] = 0
                                            self._reset_bin_histories(i)
                                            msg_other = f"Other {components[other_idx]} remains {['Normal','TrialB','Aggressive','Frozen','Terminal','Pending'][other_state]}." if other_state in [3, 4] else f"Other {components[other_idx]} set to Pending."
                                            print(f"[ADAPTIVE DUAL] Bin {i} {component} initiated Trial. {msg_other} Scale Jump -0.04.")
                                            break

                                        elif expect_rise and not rise_already_done and self.bin_epochs_at_scale[i, c_idx] >= 35:
                                            new_scale = min(self.adaptive_scale_factor_max, raw_scale + 0.01)
                                            if c_idx == 0:
                                                self.bin_pos_scale[i] = new_scale
                                            else:
                                                self.bin_rot_scale[i] = new_scale
                                            self.bin_scale_increase_history[i][component][current_scale_key] = 1
                                            self.bin_epochs_at_scale[i, c_idx] = 0
                                            self._reset_bin_histories(i) # [恢复] 每次 Scale 改变都重置历史
                                            print(f"[ADAPTIVE DUAL] Bin {i} {component} scale rise +0.01.")

                                # === [阶段 1: TrialB] ===
                                elif state == 1:
                                    trial_duration = self.adaptive_current_epoch - self.bin_trial_start_epoch[i, c_idx]
                                    if trial_duration >= 35:
                                        start_ema = self.bin_ema_at_start[i, c_idx].item()
                                        if bin_ema >= start_ema:
                                            self.bin_state[i, c_idx] = 2 # Aggressive
                                            self.bin_ema_at_start[i, c_idx] = bin_ema
                                            self.bin_epochs_at_scale[i, c_idx] = 0
                                            
                                            # [新增] 如果自己是后行者，且对方在 Pending，则让先行者也进入 Aggressive
                                            first_mover = self.bin_first_mover[i].item()
                                            if c_idx != first_mover and other_state == 5:
                                                self.bin_state[i, other_idx] = 2 # 先行者进入 Aggressive
                                                self.bin_ema_at_start[i, other_idx] = bin_ema 
                                                self.bin_epochs_at_scale[i, other_idx] = 0
                                                self.bin_aggressive_turn[i] = 0 # 默认从 pos 开始交替
                                                print(f"[ADAPTIVE DUAL] Bin {i} Second Mover {component} Trial improved. Both entering Aggressive.")
                                            else:
                                                print(f"[ADAPTIVE DUAL] Bin {i} {component} Trial improved. Phase: TrialB -> Aggressive.")
                                        else:
                                            first_mover = self.bin_first_mover[i].item()
                                            if c_idx == first_mover:
                                                # [修改] 如果对手是 Frozen/Terminal，自己直接 Frozen 且不强制对手 Trial
                                                if other_state in [3, 4]:
                                                    self.bin_state[i, c_idx] = 3 # Frozen
                                                    if c_idx == 0:
                                                        self.bin_pos_scale[i] = self.bin_pos_stable_best[i]
                                                    else:
                                                        self.bin_rot_scale[i] = self.bin_rot_stable_best[i]
                                                    print(f"[ADAPTIVE DUAL] Bin {i} First Mover {component} failed Trial. Other is Frozen, so First Mover becomes Frozen as well.")
                                                else:
                                                    self.bin_state[i, c_idx] = 5 # Pending
                                                    self.bin_state[i, other_idx] = 1 # TrialB
                                                    other_raw_scale = self.bin_pos_scale[i].item() if other_idx == 0 else self.bin_rot_scale[i].item()
                                                    self.bin_initial_scale_before_trial[i, other_idx] = other_raw_scale
                                                    self.bin_ema_at_start[i, other_idx] = bin_ema
                                                    self.bin_trial_start_epoch[i, other_idx] = self.adaptive_current_epoch
                                                    new_other_scale = max(0.7, other_raw_scale - 0.04)
                                                    if other_idx == 0:
                                                        self.bin_pos_scale[i] = new_other_scale
                                                    else:
                                                        self.bin_rot_scale[i] = new_other_scale
                                                    print(f"[ADAPTIVE DUAL] Bin {i} First Mover {component} failed Trial. Now {components[other_idx]} starts Trial.")
                                                self._reset_bin_histories(i)
                                            else:
                                                # 记录此时对手(c_idx)将被还原到的 scale
                                                initial_scale_second = self.bin_initial_scale_before_trial[i, c_idx].item()
                                                self.bin_other_scale_at_freeze[i, other_idx] = initial_scale_second

                                                self.bin_state[i, other_idx] = 3 # Frozen (First mover)
                                                self.bin_state[i, c_idx] = 0 # Normal (Second mover)
                                                self.bin_epochs_at_scale[i, c_idx] = 0 # [修复] 重置停留计时
                                                if other_idx == 0:
                                                    self.bin_pos_scale[i] = self.bin_pos_stable_best[i]
                                                else:
                                                    self.bin_rot_scale[i] = self.bin_rot_stable_best[i]
                                                initial_scale_second = self.bin_initial_scale_before_trial[i, c_idx].item()
                                                if c_idx == 0:
                                                    self.bin_pos_scale[i] = initial_scale_second
                                                else:
                                                    self.bin_rot_scale[i] = initial_scale_second
                                                print(f"[ADAPTIVE DUAL] Bin {i} Second Mover {component} failed. Reverting both.")
                                                self._reset_bin_histories(i)
                                    break

                                # === [阶段 2: Aggressive] ===
                                elif state == 2:
                                    # [修改] 成功判定标准：EMA >= 0.5 或 Avg_30 >= 0.45 (要求满 30 epoch)
                                    avg_30_target_ok = (self.bin_accumulated_epochs_since_reset[i] >= 30) and (bin_success_avg_30[i] >= 0.45)
                                    if avg_30_target_ok:
                                        if other_state == 5:
                                            # 记录对手(other_idx)当前的 scale
                                            other_scale = self.bin_pos_scale[i].item() if other_idx == 0 else self.bin_rot_scale[i].item()
                                            self.bin_other_scale_at_freeze[i, c_idx] = other_scale

                                            self.bin_state[i, c_idx] = 3
                                            self.bin_state[i, other_idx] = 0
                                            self.bin_epochs_at_scale[i, other_idx] = 0 # [修复] 重置停留计时
                                            print(f"[ADAPTIVE DUAL] Bin {i} {component} reached target. Phase: Aggressive -> Frozen, Other -> Normal.")
                                        elif other_state in [3, 4]:
                                            # [新增] 如果对手已经是锁定状态，自己成功后只把自己冻结，完全不触碰对手的状态和 scale
                                            self.bin_other_scale_at_freeze[i, c_idx] = self.bin_pos_scale[i] if other_idx == 0 else self.bin_rot_scale[i]
                                            self.bin_state[i, c_idx] = 3
                                            print(f"[ADAPTIVE DUAL] Bin {i} {component} reached target. Other is already {['Frozen','Terminal'][other_state-3]}, keeping its state and scale.")
                                        else:
                                            # 记录双方对手当前的 scale (针对对手在 Normal, TrialB, Aggressive 的情况)
                                            self.bin_other_scale_at_freeze[i, c_idx] = self.bin_pos_scale[i] if other_idx == 0 else self.bin_rot_scale[i]
                                            self.bin_other_scale_at_freeze[i, other_idx] = self.bin_pos_scale[i] if c_idx == 0 else self.bin_rot_scale[i]

                                            self.bin_state[i, c_idx] = 3
                                            self.bin_state[i, other_idx] = 3
                                            print(f"[ADAPTIVE DUAL] Bin {i} joint target reached. Both -> Frozen.")
                                        self._reset_bin_histories(i)
                                    else:
                                        self.bin_epochs_at_scale[i, c_idx] += 1
                                        if self.bin_epochs_at_scale[i, c_idx] >= 50:
                                            start_ema = self.bin_ema_at_start[i, c_idx].item()
                                            if bin_ema >= start_ema:
                                                self.bin_ema_at_start[i, c_idx] = bin_ema
                                                if other_state == 2:
                                                    turn = self.bin_aggressive_turn[i].item()
                                                    if turn == -1: turn = 0
                                                    if turn == c_idx:
                                                        new_scale = max(0.7, raw_scale - 0.01)
                                                        if c_idx == 0:
                                                            self.bin_pos_scale[i] = new_scale
                                                        else:
                                                            self.bin_rot_scale[i] = new_scale
                                                        self.bin_aggressive_turn[i] = 1 - c_idx
                                                        print(f"[ADAPTIVE DUAL] Bin {i} joint Aggressive: {component} decreased to {new_scale:.4f}")
                                                        self.bin_epochs_at_scale[i, c_idx] = 0
                                                        self._reset_bin_histories(i) # [恢复] 每次 Scale 改变都重置历史
                                                else:
                                                    # 只有自己激进
                                                    new_scale = max(0.7, raw_scale - 0.01)
                                                    if c_idx == 0:
                                                        self.bin_pos_scale[i] = new_scale
                                                    else:
                                                        self.bin_rot_scale[i] = new_scale
                                                    print(f"[ADAPTIVE DUAL] Bin {i} {component} Aggressive decrease to {new_scale:.4f}")
                                                    self.bin_epochs_at_scale[i, c_idx] = 0
                                                    self._reset_bin_histories(i) # [恢复] 每次 Scale 改变都重置历史
                                            else:
                                                first_mover = self.bin_first_mover[i].item()
                                                if other_state == 5:
                                                    self.bin_state[i, c_idx] = 5
                                                    self.bin_state[i, other_idx] = 1
                                                    other_raw_scale = self.bin_pos_scale[i].item() if other_idx == 0 else self.bin_rot_scale[i].item()
                                                    self.bin_initial_scale_before_trial[i, other_idx] = other_raw_scale
                                                    self.bin_ema_at_start[i, other_idx] = bin_ema
                                                    self.bin_trial_start_epoch[i, other_idx] = self.adaptive_current_epoch
                                                    new_other_scale = max(0.7, other_raw_scale - 0.04)
                                                    if other_idx == 0:
                                                        self.bin_pos_scale[i] = new_other_scale
                                                    else:
                                                        self.bin_rot_scale[i] = new_other_scale
                                                    print(f"[ADAPTIVE DUAL] Bin {i} First Mover {component} worsened. Now {components[other_idx]} starts Trial.")
                                                    self._reset_bin_histories(i)
                                                elif other_state in [3, 4]:
                                                    # [修改] 阶段 2 (Aggressive) 失败保护：如果对手是 Frozen/Terminal，自己直接 Frozen
                                                    self.bin_state[i, c_idx] = 3 # Frozen
                                                    if c_idx == 0:
                                                        self.bin_pos_scale[i] = self.bin_pos_stable_best[i]
                                                    else:
                                                        self.bin_rot_scale[i] = self.bin_rot_stable_best[i]
                                                    print(f"[ADAPTIVE DUAL] Bin {i} Aggressive First Mover {component} worsened. Other is Frozen, so First Mover becomes Frozen.")
                                                    self._reset_bin_histories(i)
                                                else:
                                                    # 记录对手(1-first_mover)将被还原到的 scale
                                                    initial_scale_other = self.bin_initial_scale_before_trial[i, 1-first_mover].item()
                                                    self.bin_other_scale_at_freeze[i, first_mover] = initial_scale_other

                                                    self.bin_state[i, first_mover] = 3
                                                    self.bin_state[i, 1-first_mover] = 0
                                                    self.bin_epochs_at_scale[i, 1-first_mover] = 0 # [修复] 重置停留计时
                                                    if first_mover == 0:
                                                        self.bin_pos_scale[i] = self.bin_pos_stable_best[i]
                                                    else:
                                                        self.bin_rot_scale[i] = self.bin_rot_stable_best[i]
                                                    initial_scale_second = self.bin_initial_scale_before_trial[i, 1-first_mover].item()
                                                    if 1-first_mover == 0:
                                                        self.bin_pos_scale[i] = initial_scale_second
                                                    else:
                                                        self.bin_rot_scale[i] = initial_scale_second
                                                    print(f"[ADAPTIVE DUAL] Bin {i} joint Aggressive worsened. Reverting both.")
                                                    self._reset_bin_histories(i)
                                    break

                    # 6. Reset general stats if difficulty changed
                    if scale_changed:
                        self.epochs_at_current_scale = 0
                        self.adaptive_difficulty_is_warming_up = True
                        self.bin_success_sum_at_current_scale.zero_()
                        self.difficulty_increase_timer = 0
                        self.stuck_epoch_counter = 0
                        self.adaptive_epoch_success_rate_history.clear()
                        self.adaptive_epoch_steps_history.clear()
                        # [修改] global_scale 改变时不再清空 EMA，仅在 local bin scale 变化时清空
                        self.low_success_history.clear()

                # === [新增] adaptive_real 模式的自适应调整逻辑 ===
                if self.tighten_method == "adaptive_real" and self.random_state_init and self.adaptive_sampling_bins is not None:
                    # 更新成功率历史（用于计算 avg-30）
                    self.bin_success_rate_history_30[:, self.bin_history_ptr_30] = current_epoch_bin_rates
                    self.bin_history_ptr_30 = (self.bin_history_ptr_30 + 1) % 30
                    
                    num_bins = self.adaptive_sampling_bins
                    # 仅当 support force 已衰减到 0 后，才允许将 bin 的 status 设为 2 或 4
                    end_decay_epoch_sf = 30 * (1024 / self.num_envs) * 100
                    support_force_reached_zero = (current_epoch >= end_decay_epoch_sf)
                    
                    # [新增] 计算稳定阈值：30 * 4 * (num_envs / 1024)
                    stability_threshold = int(30 * 4 * ( 1024 / self.num_envs))
                    
                    for i in range(num_bins):
                        # 只处理已经触底的 bin
                        if not self.bin_reached_bottom[i]:
                            continue
                        
                        status = self.bin_status[i].item()
                        
                        # 状态 2: 已经成功，永不改变
                        # 状态 4: 降低难度后反而失败，永不改变
                        if status in [2, 4]:
                            continue

                        # === 新需求：触底后开始统计稳定（不检测是否成功过） ===
                        # 只要 pos 或 rot 的 scale 连续 stability_threshold 个 epoch 不变，就锁定对应项，禁止再上调。
                        # pos
                        last_pos = self.bin_pos_scale_last[i].item()
                        cur_pos = self.bin_pos_scale[i].item()
                        if last_pos < 0:
                            self.bin_pos_scale_last[i] = cur_pos
                            self.bin_pos_stable_epochs[i] = 0
                        else:
                            if abs(cur_pos - last_pos) <= 1e-6:
                                self.bin_pos_stable_epochs[i] += 1
                            else:
                                self.bin_pos_scale_last[i] = cur_pos
                                self.bin_pos_stable_epochs[i] = 0

                        # rot
                        last_rot = self.bin_rot_scale_last[i].item()
                        cur_rot = self.bin_rot_scale[i].item()
                        if last_rot < 0:
                            self.bin_rot_scale_last[i] = cur_rot
                            self.bin_rot_stable_epochs[i] = 0
                        else:
                            if abs(cur_rot - last_rot) <= 1e-6:
                                self.bin_rot_stable_epochs[i] += 1
                            else:
                                self.bin_rot_scale_last[i] = cur_rot
                                self.bin_rot_stable_epochs[i] = 0

                        if (self.bin_pos_stable_epochs[i] >= stability_threshold) and (not self.bin_pos_locked[i]):
                            self.bin_pos_locked[i] = True
                            print(
                                f"[ADAPTIVE REAL] Bin {i} pos scale locked! "
                                f"Stable for {self.bin_pos_stable_epochs[i].item()} epochs after bottom. "
                                f"pos={self.bin_pos_scale[i].item():.4f}"
                            )

                        if (self.bin_rot_stable_epochs[i] >= stability_threshold) and (not self.bin_rot_locked[i]):
                            self.bin_rot_locked[i] = True
                            print(
                                f"[ADAPTIVE REAL] Bin {i} rot scale locked! "
                                f"Stable for {self.bin_rot_stable_epochs[i].item()} epochs after bottom. "
                                f"rot={self.bin_rot_scale[i].item():.4f}"
                            )
                        
                        # 更新在当前 scale 下停留的 epoch 数
                        min_epochs = min(self.bin_epochs_at_current_scale[i, 0].item(), self.bin_epochs_at_current_scale[i, 1].item())
                        self.bin_epochs_at_current_scale[i, 0] += 1  # pos
                        self.bin_epochs_at_current_scale[i, 1] += 1  # rot
                        
                        # 只有在至少跑了 30 个 epoch 后才开始判断
                        if min_epochs < 30:
                            continue
                        
                        # 计算 avg-30 成功率
                        bin_avg_30 = self.bin_success_rate_history_30[i].mean().item()
                        bin_latest = current_epoch_bin_rates[i].item()
                        
                        # 计算 obj-pos 和 obj-rot 的失败率
                        if self.bin_total_pass_count[i] > 0:
                            pos_fail_rate = self.bin_pos_fail_count[i] / self.bin_total_pass_count[i]
                            rot_fail_rate = self.bin_rot_fail_count[i] / self.bin_total_pass_count[i]
                        else:
                            pos_fail_rate = 0.0
                            rot_fail_rate = 0.0
                        
                        # 判断是否成功（avg-30 和最新 epoch 成功率都超过 50%）；仅当 support force 归零后才可设为 2
                        if bin_avg_30 >= 0.50 and bin_latest >= 0.50:
                            if support_force_reached_zero:
                                self.bin_status[i] = 2  # 成功状态
                                print(f"[ADAPTIVE REAL] Bin {i} succeeded! avg-30: {bin_avg_30:.2%}, latest: {bin_latest:.2%}. Freezing pos/rot scales at pos={self.bin_pos_scale[i]:.4f}, rot={self.bin_rot_scale[i]:.4f}")
                            continue
                        
                        # 判断是否失败（avg-30 和最新 epoch 成功率都低于 30%）
                        if bin_avg_30 < 0.30 and bin_latest < 0.30:
                            # 分支 A：回滚锁定 (Rollback & Lock)
                            # 只有当这个 Bin 之前已经因为“停滞缓解”而降低过难度，现在却依然失败时，才回滚
                            if self.bin_tried_decrease[i, 0] or self.bin_tried_decrease[i, 1]:
                                self.bin_pos_scale[i] = self.bin_pos_scale_before_decrease[i]
                                self.bin_rot_scale[i] = self.bin_rot_scale_before_decrease[i]
                                # scale 改变则重置稳定计数
                                self.bin_pos_scale_last[i] = self.bin_pos_scale[i].item()
                                self.bin_rot_scale_last[i] = self.bin_rot_scale[i].item()
                                self.bin_pos_stable_epochs[i] = 0
                                self.bin_rot_stable_epochs[i] = 0
                                if support_force_reached_zero:
                                    self.bin_status[i] = 4  # 永不改变状态（仅 support force 归零后才可设置）
                                print(f"[ADAPTIVE REAL] Bin {i} failed after stagnation relief! Rolling back and freezing. pos={self.bin_pos_scale[i]:.4f}, rot={self.bin_rot_scale[i]:.4f}")
                                continue
                            
                            # 分支 B：正常救赎降低难度 (Decrease Difficulty)
                            # 此时不设置 bin_tried_decrease，允许继续救赎
                            if pos_fail_rate > rot_fail_rate:
                                # [新增] 检查 pos 是否被锁定
                                if self.bin_pos_locked[i]:
                                    print(f"[ADAPTIVE REAL] Bin {i} Failure Recovery: pos scale is locked, skipping pos adjustment. Trying rot instead.")
                                    # 如果 pos 被锁定，尝试调整 rot（如果 rot 未被锁定）
                                    if not self.bin_rot_locked[i]:
                                        old_scale = self.bin_rot_scale[i].item()
                                        self.bin_rot_scale_before_decrease[i] = old_scale # 记录回滚点
                                        self.bin_rot_scale[i] = min(1.0, old_scale + 0.01)
                                        # scale 改变则重置稳定计数
                                        self.bin_rot_scale_last[i] = self.bin_rot_scale[i].item()
                                        self.bin_rot_stable_epochs[i] = 0
                                        print(f"[ADAPTIVE REAL] Bin {i} Failure Recovery: Increasing rot scale: {old_scale:.4f} -> {self.bin_rot_scale[i]:.4f}")
                                        self.bin_epochs_at_current_scale[i, 1] = 0
                                else:
                                    old_scale = self.bin_pos_scale[i].item()
                                    self.bin_pos_scale_before_decrease[i] = old_scale # 记录回滚点
                                    self.bin_pos_scale[i] = min(1.0, old_scale + 0.01)
                                    # scale 改变则重置稳定计数
                                    self.bin_pos_scale_last[i] = self.bin_pos_scale[i].item()
                                    self.bin_pos_stable_epochs[i] = 0
                                    print(f"[ADAPTIVE REAL] Bin {i} Failure Recovery: Increasing pos scale: {old_scale:.4f} -> {self.bin_pos_scale[i]:.4f}")
                                    self.bin_epochs_at_current_scale[i, 0] = 0
                            else:
                                # [新增] 检查 rot 是否被锁定
                                if self.bin_rot_locked[i]:
                                    print(f"[ADAPTIVE REAL] Bin {i} Failure Recovery: rot scale is locked, skipping rot adjustment. Trying pos instead.")
                                    # 如果 rot 被锁定，尝试调整 pos（如果 pos 未被锁定）
                                    if not self.bin_pos_locked[i]:
                                        old_scale = self.bin_pos_scale[i].item()
                                        self.bin_pos_scale_before_decrease[i] = old_scale # 记录回滚点
                                        self.bin_pos_scale[i] = min(1.0, old_scale + 0.01)
                                        # scale 改变则重置稳定计数
                                        self.bin_pos_scale_last[i] = self.bin_pos_scale[i].item()
                                        self.bin_pos_stable_epochs[i] = 0
                                        print(f"[ADAPTIVE REAL] Bin {i} Failure Recovery: Increasing pos scale: {old_scale:.4f} -> {self.bin_pos_scale[i]:.4f}")
                                        self.bin_epochs_at_current_scale[i, 0] = 0
                                else:
                                    old_scale = self.bin_rot_scale[i].item()
                                    self.bin_rot_scale_before_decrease[i] = old_scale # 记录回滚点
                                    self.bin_rot_scale[i] = min(1.0, old_scale + 0.01)
                                    # scale 改变则重置稳定计数
                                    self.bin_rot_scale_last[i] = self.bin_rot_scale[i].item()
                                    self.bin_rot_stable_epochs[i] = 0
                                    print(f"[ADAPTIVE REAL] Bin {i} Failure Recovery: Increasing rot scale: {old_scale:.4f} -> {self.bin_rot_scale[i]:.4f}")
                                    self.bin_epochs_at_current_scale[i, 1] = 0
                            
                            # 重置统计，准备下一轮
                            self.bin_pos_fail_count[i] = 0
                            self.bin_rot_fail_count[i] = 0
                            self.bin_total_pass_count[i] = 0
                            self.bin_no_fail_epochs[i] = 0
                            continue
                        
                        # 判断停滞缓解 (Stagnation Relief)
                        # 条件: 连续 30 个 Epoch 成功率在 [30%, 50%)，且未达到成功标准
                        if 0.30 <= bin_avg_30 < 0.50 and 0.30 <= bin_latest < 0.50:
                            self.bin_no_fail_epochs[i] += 1
                            
                            if self.bin_no_fail_epochs[i] >= 30:
                                # 模型卡住了，虽然不死，但也学不会 (成功率在 30%-50% 震荡)
                                if pos_fail_rate > rot_fail_rate and pos_fail_rate > 0:
                                    # [新增] 检查 pos 是否被锁定
                                    if self.bin_pos_locked[i]:
                                        print(f"[ADAPTIVE REAL] Bin {i} STAGNANT Relief: pos scale is locked, skipping pos adjustment.")
                                        # 如果 pos 被锁定且 rot 未被锁定，尝试调整 rot
                                        if not self.bin_rot_locked[i] and rot_fail_rate > 0:
                                            old_scale = self.bin_rot_scale[i].item()
                                            self.bin_rot_scale_before_decrease[i] = old_scale
                                            self.bin_rot_scale[i] = min(1.0, old_scale + 0.01)
                                            # scale 改变则重置稳定计数
                                            self.bin_rot_scale_last[i] = self.bin_rot_scale[i].item()
                                            self.bin_rot_stable_epochs[i] = 0
                                            self.bin_tried_decrease[i, 1] = True # 只有在这里（停滞缓解）才设为 True
                                            print(f"[ADAPTIVE REAL] Bin {i} STAGNANT Relief: Success rate stuck at {bin_avg_30:.2%}. Increasing rot scale to {self.bin_rot_scale[i]:.4f}. Set tried_decrease=True.")
                                            self.bin_epochs_at_current_scale[i, 1] = 0
                                    else:
                                        old_scale = self.bin_pos_scale[i].item()
                                        self.bin_pos_scale_before_decrease[i] = old_scale
                                        self.bin_pos_scale[i] = min(1.0, old_scale + 0.01)
                                        # scale 改变则重置稳定计数
                                        self.bin_pos_scale_last[i] = self.bin_pos_scale[i].item()
                                        self.bin_pos_stable_epochs[i] = 0
                                        self.bin_tried_decrease[i, 0] = True # 只有在这里（停滞缓解）才设为 True
                                        print(f"[ADAPTIVE REAL] Bin {i} STAGNANT Relief: Success rate stuck at {bin_avg_30:.2%}. Increasing pos scale to {self.bin_pos_scale[i]:.4f}. Set tried_decrease=True.")
                                        self.bin_epochs_at_current_scale[i, 0] = 0
                                elif rot_fail_rate > 0:
                                    # [新增] 检查 rot 是否被锁定
                                    if self.bin_rot_locked[i]:
                                        print(f"[ADAPTIVE REAL] Bin {i} STAGNANT Relief: rot scale is locked, skipping rot adjustment.")
                                        # 如果 rot 被锁定且 pos 未被锁定，尝试调整 pos
                                        if not self.bin_pos_locked[i] and pos_fail_rate > 0:
                                            old_scale = self.bin_pos_scale[i].item()
                                            self.bin_pos_scale_before_decrease[i] = old_scale
                                            self.bin_pos_scale[i] = min(1.0, old_scale + 0.01)
                                            # scale 改变则重置稳定计数
                                            self.bin_pos_scale_last[i] = self.bin_pos_scale[i].item()
                                            self.bin_pos_stable_epochs[i] = 0
                                            self.bin_tried_decrease[i, 0] = True # 只有在这里（停滞缓解）才设为 True
                                            print(f"[ADAPTIVE REAL] Bin {i} STAGNANT Relief: Success rate stuck at {bin_avg_30:.2%}. Increasing pos scale to {self.bin_pos_scale[i]:.4f}. Set tried_decrease=True.")
                                            self.bin_epochs_at_current_scale[i, 0] = 0
                                    else:
                                        old_scale = self.bin_rot_scale[i].item()
                                        self.bin_rot_scale_before_decrease[i] = old_scale
                                        self.bin_rot_scale[i] = min(1.0, old_scale + 0.01)
                                        # scale 改变则重置稳定计数
                                        self.bin_rot_scale_last[i] = self.bin_rot_scale[i].item()
                                        self.bin_rot_stable_epochs[i] = 0
                                        self.bin_tried_decrease[i, 1] = True # 只有在这里（停滞缓解）才设为 True
                                        print(f"[ADAPTIVE REAL] Bin {i} STAGNANT Relief: Success rate stuck at {bin_avg_30:.2%}. Increasing rot scale to {self.bin_rot_scale[i]:.4f}. Set tried_decrease=True.")
                                        self.bin_epochs_at_current_scale[i, 1] = 0
                                
                                # 重置统计
                                self.bin_pos_fail_count[i] = 0
                                self.bin_rot_fail_count[i] = 0
                                self.bin_total_pass_count[i] = 0
                                self.bin_no_fail_epochs[i] = 0
                        else:
                            # 出现了严重的失败 ( < 30%) 或者已经接近成功 ( >= 50%)，重置停滞计数
                            self.bin_no_fail_epochs[i] = 0

                # === [原有逻辑移除] ===
                # ... (原来的 Check 1, 2, 3 逻辑已被上面的新逻辑替换)
                
                # === [新增] 记录当前epoch的重力和摩擦力信息到tensorboard ===
                # 获取当前模拟器实际使用的重力，并更新缓存供 compute_observations 使用 [第 5 点]
                current_sim_params = self.gym.get_sim_params(self.sim)
                current_gravity = current_sim_params.gravity.z
                self._gravity_z = float(current_gravity)

                # 获取摩擦力的平均值（遍历所有环境的非hand actor）
                total_friction = 0.0
                friction_count = 0

                # 遍历所有环境来获取摩擦力信息
                for env_id in range(self.num_envs):
                    env = self.envs[env_id]
                    num_actors = self.gym.get_actor_count(env)

                    for actor_idx in range(num_actors):
                        handle = self.gym.get_actor_handle(env, actor_idx)
                        actor_name = self.gym.get_actor_name(env, handle)

                        # 只统计非hand的物体actor
                        if "hand" not in actor_name:
                            try:
                                prop = self.gym.get_actor_rigid_shape_properties(env, handle)
                                for shape_prop in prop:
                                    total_friction += shape_prop.friction
                                    friction_count += 1
                            except Exception:
                                # 如果获取失败，跳过
                                continue

                # 计算平均摩擦力
                avg_friction = total_friction / friction_count if friction_count > 0 else 0.0

                # 保存到类变量中，供tensorboard记录使用
                self.current_epoch_gravity = current_gravity
                self.current_epoch_avg_friction = avg_friction

                # === [新增] 更新自适应采样的统计信息 ===
                if self.random_state_init and self.adaptive_sampling_bins is not None:
                    # 使用指数移动平均更新统计信息
                    self.bin_success_count = (
                        self.adaptive_sampling_alpha * self._current_bin_success +
                        (1 - self.adaptive_sampling_alpha) * self.bin_success_count
                    )
                    self.bin_total_count = (
                        self.adaptive_sampling_alpha * self._current_bin_total +
                        (1 - self.adaptive_sampling_alpha) * self.bin_total_count
                    )

                    # 每 20 个 epoch 打印一次 bin 统计信息
                    if self.adaptive_current_epoch % 20 == 0:
                        bin_reset_counts = self._current_bin_reset.int().tolist()
                        current_rates = current_epoch_bin_rates.tolist()
                        ema_rates = self.bin_ema_success_rates.tolist()
                        print(f"[ADAPTIVE STATS] Epoch {self.adaptive_current_epoch} (Global Scale: {self.adaptive_global_scale_factor:.4f}, Start Ratio: {self.start_from_beginning_ratio:.4f}, Obj-Pos Scale: {self.bin_pos_scale.mean().item():.4f}, Obj-Rot Scale: {self.bin_rot_scale.mean().item():.4f}):")
                        print(f"  - Bin Reset Counts: {bin_reset_counts}")
                        print(f"  - Bin Success Rates (Current): {[f'{r:.2f}' for r in current_rates]}")
                        print(f"  - Bin Success Rates (EMA):     {[f'{r:.2f}' for r in ema_rates]}")
                        print(f"  - Bin Obj-Pos Pass Rates:      {[f'{r:.2f}' for r in self.bin_obj_pos_pass_rate.tolist()]}")
                        print(f"  - Bin Obj-Rot Pass Rates:      {[f'{r:.2f}' for r in self.bin_obj_rot_pass_rate.tolist()]}")

                    # 重置当前epoch的统计
                    self._current_bin_success.zero_()
                    self._current_bin_total.zero_()
                    self._current_bin_reset.zero_()
                    self._current_bin_obj_pos_pass.zero_()
                    self._current_bin_obj_rot_pass.zero_()

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
        if self.training:
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
        
        # [新增] 重置特定类型的失败记录
        if hasattr(self, "env_obj_pos_failed"):
            self.env_obj_pos_failed[env_ids] = False
            self.env_obj_rot_failed[env_ids] = False

        # [第 4 点] 瞬移后标记“数据已过期”，强制下一帧取 obs 时必做一次 _refresh，避免观测到 reset 前的旧状态
        self._last_refresh_step = -1

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
        
        # 添加支撑力 Kp 和 Kd 到 TensorBoard
        if hasattr(self, 'support_force_kp'):
            info["adaptive_difficulty/support_force_kp"] = float(self.support_force_kp)
        if hasattr(self, 'support_force_kd'):
            info["adaptive_difficulty/support_force_kd"] = float(self.support_force_kd)
        if hasattr(self, 'support_force_kp_rot'):
            info["adaptive_difficulty/support_force_kp_rot"] = float(self.support_force_kp_rot)
        if hasattr(self, 'support_force_kd_rot'):
            info["adaptive_difficulty/support_force_kd_rot"] = float(self.support_force_kd_rot)
        if hasattr(self, 'start_from_beginning_ratio'):
            info["adaptive_difficulty/start_from_beginning_ratio"] = float(self.start_from_beginning_ratio)

        # === [重构] 统一难度和采样统计信息填充 (仅按 Iteration 逻辑记录) ===
        if self.adaptive_sampling_bins is not None:
            # 1. 基础统计 (EMA 成功率和通过率)
            if hasattr(self, 'current_epoch_bin_rates') and self.current_epoch_bin_rates is not None:
                for i in range(self.adaptive_sampling_bins):
                    info[f"adaptive_sampling/bin_{i}_success_rate"] = float(self.current_epoch_bin_rates[i].item())
            
            if hasattr(self, 'bin_obj_pos_pass_rate') and self.bin_obj_pos_pass_rate is not None:
                for i in range(self.adaptive_sampling_bins):
                    info[f"adaptive_sampling/bin_{i}_obj_pos_pass_rate"] = float(self.bin_obj_pos_pass_rate[i].item())
            if hasattr(self, 'bin_obj_rot_pass_rate') and self.bin_obj_rot_pass_rate is not None:
                for i in range(self.adaptive_sampling_bins):
                    info[f"adaptive_sampling/bin_{i}_obj_rot_pass_rate"] = float(self.bin_obj_rot_pass_rate[i].item())

            # 2. Adaptive Dual 专有数据 (Scale 和 State)
            if self.tighten_method == "adaptive_dual":
                for i in range(self.adaptive_sampling_bins):
                    if hasattr(self, 'bin_pos_scale'):
                        info[f"adaptive_difficulty/bin_{i}_pos_scale"] = float(self.bin_pos_scale[i].item())
                    if hasattr(self, 'bin_rot_scale'):
                        info[f"adaptive_difficulty/bin_{i}_rot_scale"] = float(self.bin_rot_scale[i].item())
                    if hasattr(self, 'bin_state'):
                        info[f"adaptive_difficulty/bin_{i}_pos_state"] = float(self.bin_state[i, 0].item())
                        info[f"adaptive_difficulty/bin_{i}_rot_state"] = float(self.bin_state[i, 1].item())
                    
                    # [新增] 记录每个 bin 的历史最佳稳定 scale
                    if hasattr(self, 'bin_pos_stable_best'):
                        info[f"adaptive_difficulty/bin_{i}_pos_stable_best"] = float(self.bin_pos_stable_best[i].item())
                    if hasattr(self, 'bin_rot_stable_best'):
                        info[f"adaptive_difficulty/bin_{i}_rot_stable_best"] = float(self.bin_rot_stable_best[i].item())
            
            # [新增] Adaptive Real 专有数据 (Scale 和 Status)
            if self.tighten_method == "adaptive_real":
                for i in range(self.adaptive_sampling_bins):
                    if hasattr(self, 'bin_pos_scale'):
                        info[f"adaptive_difficulty/bin_{i}_pos_scale"] = float(self.bin_pos_scale[i].item())
                    if hasattr(self, 'bin_rot_scale'):
                        info[f"adaptive_difficulty/bin_{i}_rot_scale"] = float(self.bin_rot_scale[i].item())
                    if hasattr(self, 'bin_status'):
                        info[f"adaptive_difficulty/bin_{i}_status"] = float(self.bin_status[i].item())
                    if hasattr(self, 'bin_pos_fail_count') and hasattr(self, 'bin_total_pass_count'):
                        if self.bin_total_pass_count[i] > 0:
                            info[f"adaptive_difficulty/bin_{i}_pos_fail_rate"] = float(self.bin_pos_fail_count[i] / self.bin_total_pass_count[i])
                            info[f"adaptive_difficulty/bin_{i}_rot_fail_rate"] = float(self.bin_rot_fail_count[i] / self.bin_total_pass_count[i])
                        else:
                            info[f"adaptive_difficulty/bin_{i}_pos_fail_rate"] = 0.0
                            info[f"adaptive_difficulty/bin_{i}_rot_fail_rate"] = 0.0
                    if hasattr(self, 'bin_no_fail_epochs'):
                        info[f"adaptive_difficulty/bin_{i}_no_fail_epochs"] = float(self.bin_no_fail_epochs[i].item())

            # 3. 总体成功率
            if hasattr(self, 'current_epoch_success_rate'):
                info["adaptive_sampling/overall_success_rate"] = float(self.current_epoch_success_rate)

        # 4. 强制 Iteration 轴对齐
        if hasattr(self, 'adaptive_current_epoch') and self.adaptive_current_epoch >= 0:
            current_iter = int(self.adaptive_current_epoch)
            info["iteration"] = current_iter
            info["adaptive_difficulty/current_iteration"] = current_iter

        # 5. 辅助信息 (重力、摩擦力等)
        if hasattr(self, 'current_epoch_gravity'):
            info["adaptive_difficulty/current_epoch_gravity"] = float(self.current_epoch_gravity)
        if hasattr(self, 'current_epoch_avg_friction'):
            info["adaptive_difficulty/current_epoch_avg_friction"] = float(self.current_epoch_avg_friction)

        # 6. 手指接触力 finger_force = norm(tip_force) [N, 5]，写入 tensorboard
        if hasattr(self, "_rh_finger_force") and hasattr(self, "_lh_finger_force"):
            info["finger_force/rh_mean"] = float(self._rh_finger_force.mean().item())
            info["finger_force/lh_mean"] = float(self._lh_finger_force.mean().item())
            for i in range(self._rh_finger_force.shape[1]):
                info[f"finger_force/rh_{i}"] = float(self._rh_finger_force[:, i].mean().item())
                info[f"finger_force/lh_{i}"] = float(self._lh_finger_force[:, i].mean().item())

        # === [物理引擎设置] ===
        if hasattr(self, "current_scale_factor"):
            self.set_adaptive_scale_factor(self.current_scale_factor)
        elif hasattr(self, "adaptive_global_scale_factor"):
            self.set_adaptive_scale_factor(self.adaptive_global_scale_factor)

        return obs, rew, done, info

    def pre_physics_step(self, actions):

        # ? >>> for visualization
        if not self.headless:
            # [修复] 增加对 cur_idx 的 clamp，防止 progress_buf 超过 demo 长度导致的 indexing 报错
            cur_idx = torch.clamp(self.progress_buf, torch.zeros_like(self.demo_data_rh["seq_len"]), self.demo_data_rh["seq_len"] - 1)

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

        # === [新增] 施加物体 PD 辅助力 ===
        if self.support_force_kp > 0 or self.support_force_kd > 0:
            self._apply_support_forces()

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self._pos_control[:] = self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def _apply_support_forces(self):
        """
        [新增] 动态 PD 控制器：在物理模拟前给动态物体施加辅助力 F_support 和扭矩 Tau_support
        F_support = Kp * (pos_target - pos_current) + Kd * (vel_target - vel_current)
        Tau_support = Kp_rot * Error(q_tar, q_cur) + Kd_rot * (omega_tar - omega_cur)
        """
        cur_idx = self.progress_buf
        
        # 准备索引 [N, K]
        batch_idx = torch.arange(self.num_envs, device=self.device).view(-1, 1).expand(-1, self.num_objs_per_env)
        obj_idx = torch.arange(self.num_objs_per_env, device=self.device).view(1, -1).expand(self.num_envs, -1)
        
        # 如果左右物体完全相同，只计算并施加一次（通常使用右手的目标轨迹）
        sides = ["rh"] if getattr(self, "is_scene_objects_shared", False) else ["rh", "lh"]

        # 获取旋转相关的 Kp/Kd (如果未定义则默认为位置 Kp 的 0.1 倍，因为旋转单位是弧度，扭矩较敏感)
        kp_rot = getattr(self, "support_force_kp_rot", self.support_force_kp * 0.1)
        kd_rot = getattr(self, "support_force_kd_rot", self.support_force_kd * 0.1)
        
        for side in sides:
            # 1. 获取目标状态 (N, K, ...)
            multi_obj_traj = getattr(self, f"{side}_multi_obj_traj")
            multi_obj_vel_traj = getattr(self, f"{side}_multi_obj_vel")
            multi_obj_ang_vel_traj = getattr(self, f"{side}_multi_obj_ang_vel")
            
            # time_idx 限制在序列长度内
            max_T = multi_obj_traj.shape[2]
            time_idx = torch.clamp(cur_idx, 0, max_T - 1).view(-1, 1).expand(-1, self.num_objs_per_env)
            
            # 取出当前帧的目标位置、旋转矩阵、速度、角速度
            target_pos = multi_obj_traj[batch_idx, obj_idx, time_idx][..., :3, 3]
            target_rot_mat = multi_obj_traj[batch_idx, obj_idx, time_idx][..., :3, :3]
            
            # 使用 rotmat_to_quat (来自 transform.py)，它返回 (w, x, y, z)
            # 然后统一转换为 (x, y, z, w)
            target_quat_wxyz = rotmat_to_quat(target_rot_mat.reshape(-1, 3, 3))
            target_quat = target_quat_wxyz[:, [1, 2, 3, 0]].reshape(self.num_envs, self.num_objs_per_env, 4)
            
            target_vel = multi_obj_vel_traj[batch_idx, obj_idx, time_idx]
            target_ang_vel = multi_obj_ang_vel_traj[batch_idx, obj_idx, time_idx]
            
            # 2. 获取当前状态 (N, K, ...)
            side_states = getattr(self, f"{side}_states")
            current_pos = side_states["manip_obj_pos"]
            current_quat = side_states["manip_obj_quat"]
            current_vel = side_states["manip_obj_vel"]
            current_ang_vel = side_states["manip_obj_ang_vel"]
            
            # 3. 计算 PD 辅助力
            force = self.support_force_kp * (target_pos - current_pos) + \
                    self.support_force_kd * (target_vel - current_vel)
            
            # 4. 计算 PD 辅助扭矩
            # 4.1 四元数误差 Error(q_tar, q_cur)
            # q_err = q_tar * inv(q_cur)
            q_cur_inv = quat_conjugate(current_quat)
            q_err = quat_mul(target_quat, q_cur_inv)
            
            # 4.2 确保最短路径 (Shortest Path)
            # if w < 0, q_err = -q_err
            w_mask = (q_err[..., 3] < 0)
            q_err[w_mask] = -q_err[w_mask]
            
            # 4.3 旋转向量误差 (Axis-Angle Error)
            # 使用 atan2 计算精确的旋转向量误差，相比 2*q_xyz 在大角度下更准确
            q_xyz = q_err[..., :3]
            q_w = q_err[..., 3].unsqueeze(-1)
            norm_xyz = torch.norm(q_xyz, dim=-1, keepdim=True)
            angle = 2.0 * torch.atan2(norm_xyz, q_w)
            # 防止除以极小值，当角度很小时退化为线性近似 2.0 * q_xyz
            rot_err = torch.where(norm_xyz > 1e-6, (q_xyz / norm_xyz) * angle, 2.0 * q_xyz)
            
            # 4.4 扭矩公式
            torque = kp_rot * rot_err + kd_rot * (- current_ang_vel)
            max_torque = 0.1
            torque = torch.clamp(torque, -max_torque, max_torque)
            
            # 5. 掩码：只对动态物体施加
            is_static = getattr(self, f"manip_obj_{side}_is_static")
            force = force * (~is_static).unsqueeze(-1).float()
            torque = torque * (~is_static).unsqueeze(-1).float()
            
            # 6. [新增] 强化版辅助力和扭矩可视化
            if not self.headless:
                for env_id in range(min(self.num_envs, 4)): 
                    env_ptr = self.envs[env_id]
                    for obj_k in range(self.num_objs_per_env):
                        if not is_static[env_id, obj_k]:
                            p_curr = current_pos[env_id, obj_k].cpu().numpy()
                            p_target = target_pos[env_id, obj_k].cpu().numpy()
                            f_val = force[env_id, obj_k].cpu().numpy()
                            t_val = torque[env_id, obj_k].cpu().numpy()

                            # 1. 绘制力矢量 (黄色)
                            p_force_end = p_curr + f_val * 1.0 
                            yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
                            self.gym.add_lines(self.viewer, env_ptr, 1, np.concatenate([p_curr, p_force_end]).astype(np.float32), yellow)

                            # 2. 绘制扭矩矢量 (紫色，表示旋转轴和强度)
                            p_torque_end = p_curr + t_val * 2.0 
                            purple = np.array([1.0, 0.0, 1.0], dtype=np.float32)
                            self.gym.add_lines(self.viewer, env_ptr, 1, np.concatenate([p_curr, p_torque_end]).astype(np.float32), purple)

                            # 3. 绘制指向目标的连线 (青色)
                            cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)
                            self.gym.add_lines(self.viewer, env_ptr, 1, np.concatenate([p_curr, p_target]).astype(np.float32), cyan)

            # 7. 写入 apply_forces 和 apply_torque
            rb_indices = getattr(self, f"_manip_obj_{side}_rb_indices")
            valid_mask = (rb_indices != -1)
            
            flat_env_idx = batch_idx[valid_mask]
            flat_rb_idx = rb_indices[valid_mask]
            
            self.apply_forces[flat_env_idx, flat_rb_idx] = force[valid_mask]
            self.apply_torque[flat_env_idx, flat_rb_idx] = torque[valid_mask]

    def post_physics_step(self):

        self.compute_observations()
        self.compute_reward(self.actions)

        # [新增] 在测试模式下输出 wrist 的线速度和角速度，以及与 reference 的误差 (仅对环境 0)
        # if not self.training:
        #     # 1. 获取当前索引
        #     cur_idx = torch.clamp(self.progress_buf, torch.zeros_like(self.demo_data_rh["seq_len"]), self.demo_data_rh["seq_len"] - 1)
        #     e0 = 0 # Env 0

        #     # 2. 获取当前状态 (Env 0)
        #     # base_state 包含: pos(3), quat(4), vel(3), ang_vel(3)
        #     rh_pos = self.rh_states["base_state"][e0, :3]
        #     rh_quat = self.rh_states["base_state"][e0, 3:7]
        #     rh_vel = self.rh_states["base_state"][e0, 7:10]
        #     rh_ang_vel = self.rh_states["base_state"][e0, 10:13]

        #     lh_pos = self.lh_states["base_state"][e0, :3]
        #     lh_quat = self.lh_states["base_state"][e0, 3:7]
        #     lh_vel = self.lh_states["base_state"][e0, 7:10]
        #     lh_ang_vel = self.lh_states["base_state"][e0, 10:13]

        #     # 3. 获取目标状态 (Env 0)
        #     rh_target_pos = self.demo_data_rh["wrist_pos"][e0, cur_idx[e0]]
        #     rh_target_aa = self.demo_data_rh["wrist_rot"][e0, cur_idx[e0]]
        #     rh_target_quat = aa_to_quat(rh_target_aa.view(1, 3))[:, [1, 2, 3, 0]].view(4)
        #     rh_target_vel = self.demo_data_rh["wrist_velocity"][e0, cur_idx[e0]]
        #     rh_target_ang_vel = self.demo_data_rh["wrist_angular_velocity"][e0, cur_idx[e0]]

        #     lh_target_pos = self.demo_data_lh["wrist_pos"][e0, cur_idx[e0]]
        #     lh_target_aa = self.demo_data_lh["wrist_rot"][e0, cur_idx[e0]]
        #     lh_target_quat = aa_to_quat(lh_target_aa.view(1, 3))[:, [1, 2, 3, 0]].view(4)
        #     lh_target_vel = self.demo_data_lh["wrist_velocity"][e0, cur_idx[e0]]
        #     lh_target_ang_vel = self.demo_data_lh["wrist_angular_velocity"][e0, cur_idx[e0]]

        #     # 4. 计算误差 (与 compute_imitation_reward 逻辑一致)
        #     # Position Error (Norm)
        #     rh_pos_err = torch.norm(rh_target_pos - rh_pos).item()
        #     lh_pos_err = torch.norm(lh_target_pos - lh_pos).item()

        #     # Rotation Error (Angle in degrees)
        #     rh_rot_diff = quat_mul(rh_target_quat.view(1, 4), quat_conjugate(rh_quat.view(1, 4)))
        #     rh_rot_err = (quat_to_angle_axis(rh_rot_diff)[0].abs() / np.pi * 180).item()
        #     lh_rot_diff = quat_mul(lh_target_quat.view(1, 4), quat_conjugate(lh_quat.view(1, 4)))
        #     lh_rot_err = (quat_to_angle_axis(lh_rot_diff)[0].abs() / np.pi * 180).item()

        #     # Velocity Error (Mean Absolute Error)
        #     rh_vel_err = (rh_target_vel - rh_vel).abs().mean().item()
        #     lh_vel_err = (lh_target_vel - lh_vel).abs().mean().item()

        #     # Angular Velocity Error (Mean Absolute Error)
        #     rh_ang_vel_err = (rh_target_ang_vel - rh_ang_vel).abs().mean().item()
        #     lh_ang_vel_err = (lh_target_ang_vel - lh_ang_vel).abs().mean().item()

        #     print(f"Step {self.progress_buf[0].item()}:")
        #     print(f"  [RH Wrist] LinVel: {rh_vel.cpu().numpy()}, AngVel: {rh_ang_vel.cpu().numpy()}")
        #     print(f"             ERR: pos={rh_pos_err:.4f}m, rot={rh_rot_err:.2f}deg, vel={rh_vel_err:.4f}, ang_vel={rh_ang_vel_err:.4f}")
        #     print(f"  [LH Wrist] LinVel: {lh_vel.cpu().numpy()}, AngVel: {lh_ang_vel.cpu().numpy()}")∂
        #     print(f"             ERR: pos={lh_pos_err:.4f}m, rot={lh_rot_err:.2f}deg, vel={lh_vel_err:.4f}, ang_vel={lh_ang_vel_err:.4f}")

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

    def set_adaptive_scale_factor(self, scale_factor):
        """覆盖基类的难度调整逻辑，支持自定义的支撑力衰减逻辑"""
        # [新增] 调用基类方法，以触发基类中定义的重力、摩擦力调整逻辑
        if hasattr(super(), "set_adaptive_scale_factor"):
            super().set_adaptive_scale_factor(scale_factor)

        self.current_adaptive_scale_factor = scale_factor

        # === [核心修改] 支撑力衰减逻辑 ===
        # 根据 epoch 进行衰减
        # 0 - 30 * 1024 / num_envs * 100 epoch 是不变的
        # 从 30 * 1024 / num_envs * 100 epoch - 30 * 1024 / num_envs * 140 epoch 线性衰减到 0
        
        last_step = self.gym.get_frame_count(self.sim)
        horizon_length = getattr(self, 'horizon_length', 32)
        frames_per_epoch = horizon_length * self.num_envs
        current_epoch = int(last_step // frames_per_epoch) if frames_per_epoch > 0 else 0
        if hasattr(self, 'total_train_env_frames') and self.total_train_env_frames is not None:
            current_epoch = int(self.total_train_env_frames // frames_per_epoch) if frames_per_epoch > 0 else 0
            
        start_decay_epoch = 30 * (1024 / self.num_envs) * 100
        end_decay_epoch = 30 * (1024 / self.num_envs) * 140
        
        if current_epoch < start_decay_epoch:
            decay_factor = 1.0
        elif current_epoch < end_decay_epoch:
            decay_factor = 1.0 - (current_epoch - start_decay_epoch) / (end_decay_epoch - start_decay_epoch)
        else:
            decay_factor = 0.0

        self.support_force_kp = self.support_force_kp_start * decay_factor
        self.support_force_kd = self.support_force_kd_start * decay_factor
        self.support_force_kp_rot = self.support_force_kp_rot_start * decay_factor
        self.support_force_kd_rot = self.support_force_kd_rot_start * decay_factor

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
    max_length: Tensor,
    scale_factor: Tensor, # [N] Changed to Tensor
    dexhand_weight_idx: Dict[str, List[int]],
    obj_is_static: Tensor,  # [N, K] 新增：静态物体信息
    progress_reward_unit: Tensor, # [新增] 每帧固定的奖励额度 (基于全长计算)
    reward_interact_scale: float = 2.0, # 新增
    terminate_on_eef: bool = True, # 新增
    reward_obj_pos_scale: float = 6.0,
    reward_obj_rot_scale: float = 5.0,
    reward_finger_tip_force_scale: float = 15.0,
    terminate_obj_pos_threshold: float = 0.2,
    terminate_obj_rot_threshold: float = 70.0,
    rot_scale_factor: Tensor = None, # [N] Changed to Tensor
    terminate_on_contact: bool = False,
    terminate_obj_pos_final: float = 0.03,
    terminate_obj_rot_final: float = 30.0,
    terminate_thumb_threshold: float = 0.18,
    terminate_index_threshold: float = 0.20,
    terminate_middle_threshold: float = 0.18,
    terminate_pinky_threshold: float = 0.23,
    terminate_ring_threshold: float = 0.22,
    terminate_level_1_threshold: float = 0.22,
    terminate_level_2_threshold: float = 0.25,
    eef_vel_limit: float = 100.0,
    eef_ang_vel_limit: float = 200.0,
    joints_vel_limit: float = 100.0,
    dof_vel_limit: float = 200.0,
    obj_vel_limit: float = 100.0,
    obj_ang_vel_limit: float = 200.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Dict[str, Tensor], Dict[str, Tensor]]:

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

    # === 调试断点 2：检查奖励计算输入 ===
    if torch.isnan(states["manip_obj_pos"]).any() or torch.isnan(target_states["manip_obj_pos"]).any():
        print("\n[!!!] 奖励计算输入包含 NaN !")
        print(f"States Pos NaN: {torch.isnan(states['manip_obj_pos']).any()}")
        print(f"Target Pos NaN: {torch.isnan(target_states['manip_obj_pos']).any()}")
        # import pdb; pdb.set_trace()

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

    # === [新增] 计算 Alpha (基于 Reference 中的抓握力) ===
    # 规则: 统计 reference 中指尖距离 < 0.004 的手指数量 N
    # alpha = 0.2 + (归一化力分数之和 * 0.8 / N)
    # 归一化分数: force < 0.5 为 0, force > 2.0 为 1
    ref_finger_dist = target_states["tips_distance"] # [N, 5]
    ref_finger_force = torch.norm(target_states["tip_force"], dim=-1) # [N, 5]
    
    contact_mask = ref_finger_dist < 0.03 # [N, 5]
    N_contacts = contact_mask.sum(dim=-1, keepdim=True).float() # [N, 1]
    
    # 计算归一化力分数 [N, 5]
    force_score = torch.clamp((ref_finger_force - 0.1) / (1.5 - 0.1), 0.0, 1.0)
    
    # 计算每个手指的贡献权重
    force_contribution = (force_score * 1.0) / (N_contacts + 1e-6)
    alpha_sum = (force_contribution * contact_mask.float()).sum(dim=-1, keepdim=True) # [N, 1]
    
    alpha = 0.4 + 2.6 * alpha_sum
    # 如果没有手指接触，alpha 设为 1.0
    alpha = torch.where(N_contacts > 0.5, alpha, torch.ones_like(alpha)).squeeze(-1) # [N]

    # 将 alpha 应用到位姿奖励
    reward_obj_pos = reward_obj_pos * alpha
    reward_obj_rot = reward_obj_rot * alpha

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

    error_eef_vel = torch.norm(current_eef_vel, dim=-1) > eef_vel_limit
    error_eef_ang_vel = torch.norm(current_eef_ang_vel, dim=-1) > eef_ang_vel_limit
    error_joints_vel = torch.norm(joints_vel, dim=-1).mean(-1) > joints_vel_limit
    error_dof_vel = torch.abs(current_dof_vel).mean(-1) > dof_vel_limit
    error_obj_vel = obj_vel_norm > obj_vel_limit
    error_obj_ang_vel = obj_ang_vel_norm > obj_ang_vel_limit

    error_buf = (
        error_eef_vel
        | error_eef_ang_vel
        | error_joints_vel
        | error_dof_vel
        | error_obj_vel
        | error_obj_ang_vel
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

    # 1. 计算各个失败原因
    obj_pos_threshold_final = terminate_obj_pos_final / 0.343 * scale_factor**3
    obj_rot_threshold_final = terminate_obj_rot_final / 0.343 * rot_scale_factor**3

    obj_pos_failed = obj_pos_err > obj_pos_threshold_final
    obj_rot_failed = obj_rot_err > obj_rot_threshold_final
    thumb_failed = diff_thumb_tip_pos_dist > 2 * 1.5 * terminate_thumb_threshold / 0.7 * (scale_factor * scale_factor)
    index_failed = diff_index_tip_pos_dist > 2 * 1.5 * terminate_index_threshold / 0.7 * (scale_factor * scale_factor)
    middle_failed = diff_middle_tip_pos_dist > 2 * 1.5 * terminate_middle_threshold / 0.7 * (scale_factor * scale_factor)
    pinky_failed = diff_pinky_tip_pos_dist > 2 * 1.5 * terminate_pinky_threshold / 0.7 * (scale_factor * scale_factor)
    ring_failed = diff_ring_tip_pos_dist > 2 * 1.5 * terminate_ring_threshold / 0.7 * (scale_factor * scale_factor)
    level_1_failed = diff_level_1_pos_dist > 2 * 1.5 * terminate_level_1_threshold / 0.7 * (scale_factor * scale_factor)
    level_2_failed = diff_level_2_pos_dist > 2 * 1.5 * terminate_level_2_threshold / 0.7 * (scale_factor * scale_factor)

    # 手腕误差
    eef_pos_err = diff_eef_pos_dist
    diff_eef_rot_angle_deg = diff_eef_rot_angle.abs() / np.pi * 180
    eef_vel_err = diff_eef_vel.abs().mean(dim=-1)
    eef_ang_vel_err = diff_eef_ang_vel.abs().mean(dim=-1)

    eef_pos_threshold = 0.08 / 0.7 * scale_factor
    eef_rot_threshold_deg = 45 / 0.7 * scale_factor
    eef_vel_threshold = 1.0 / 0.7 * scale_factor
    eef_ang_vel_threshold = 14.0 / 0.7 * scale_factor

    eef_pos_failed = eef_pos_err > eef_pos_threshold
    eef_rot_failed = diff_eef_rot_angle_deg > eef_rot_threshold_deg
    eef_vel_failed = eef_vel_err > eef_vel_threshold
    eef_ang_vel_failed = eef_ang_vel_err > eef_ang_vel_threshold

    failed_execute_eef = (
        eef_pos_failed
        | eef_rot_failed
    ) if terminate_on_eef else torch.zeros_like(obj_pos_err, dtype=torch.bool)

    failed_execute = (
        (
            obj_pos_failed
            | thumb_failed
            | index_failed
            | middle_failed
            | pinky_failed
            | ring_failed
            | level_1_failed
            | level_2_failed
            | obj_rot_failed
            | failed_execute_eef
            | (contact_violation if terminate_on_contact else torch.zeros_like(contact_violation, dtype=torch.bool))
        )
        & (running_progress_buf >= 8)
    ) | error_buf

    failure_reasons = {
        "obj_pos_failed": obj_pos_failed,
        "obj_rot_failed": obj_rot_failed,
        "thumb_failed": thumb_failed,
        "index_failed": index_failed,
        "middle_failed": middle_failed,
        "pinky_failed": pinky_failed,
        "ring_failed": ring_failed,
        "level_1_failed": level_1_failed,
        "level_2_failed": level_2_failed,
        "contact_violation": contact_violation,
        "eef_pos_failed": eef_pos_failed,
        "eef_rot_failed": eef_rot_failed,
        "eef_vel_failed": eef_vel_failed,
        "eef_ang_vel_failed": eef_ang_vel_failed,
    }

    failure_values = {
        "obj_pos_err": obj_pos_err,
        "obj_rot_err": obj_rot_err,
        "thumb_tip_dist": diff_thumb_tip_pos_dist,
        "index_tip_dist": diff_index_tip_pos_dist,
        "middle_tip_dist": diff_middle_tip_pos_dist,
        "pinky_tip_dist": diff_pinky_tip_pos_dist,
        "ring_tip_dist": diff_ring_tip_pos_dist,
        "level_1_dist": diff_level_1_pos_dist,
        "level_2_dist": diff_level_2_pos_dist,
        "eef_pos_err": eef_pos_err,
        "eef_rot_err_deg": diff_eef_rot_angle_deg,
        "eef_vel_err": eef_vel_err,
        "eef_ang_vel_err": eef_ang_vel_err,
        "eef_vel_norm": torch.norm(current_eef_vel, dim=-1),
        "eef_ang_vel_norm": torch.norm(current_eef_ang_vel, dim=-1),
        "joints_vel_norm": torch.norm(joints_vel, dim=-1).mean(-1),
        "dof_vel_norm": torch.abs(current_dof_vel).mean(-1),
        "obj_vel_norm": obj_vel_norm,
        "obj_ang_vel_norm": obj_ang_vel_norm,
        "obj_pos_threshold": obj_pos_threshold_final,
        "obj_rot_threshold": obj_rot_threshold_final,
        "thumb_threshold": 2 * 1.5 * terminate_thumb_threshold / 0.7 * (scale_factor * scale_factor),
        "index_threshold": 2 * 1.5 * terminate_index_threshold / 0.7 * (scale_factor * scale_factor),
        "middle_threshold": 2 * 1.5 * terminate_middle_threshold / 0.7 * (scale_factor * scale_factor),
        "pinky_threshold": 2 * 1.5 * terminate_pinky_threshold / 0.7 * (scale_factor * scale_factor),
        "ring_threshold": 2 * 1.5 * terminate_ring_threshold / 0.7 * (scale_factor * scale_factor),
        "level_1_threshold": 2 * 1.5 * terminate_level_1_threshold / 0.7 * (scale_factor * scale_factor),
        "level_2_threshold": 2 * 1.5 * terminate_level_2_threshold / 0.7 * (scale_factor * scale_factor),
        "eef_pos_threshold": eef_pos_threshold,
        "eef_rot_threshold_deg": eef_rot_threshold_deg,
        "eef_vel_threshold": eef_vel_threshold,
        "eef_ang_vel_threshold": eef_ang_vel_threshold,
    }

    # sanity check flags
    failure_reasons["error_eef_vel"] = error_eef_vel
    failure_reasons["error_eef_ang_vel"] = error_eef_ang_vel
    failure_reasons["error_joints_vel"] = error_joints_vel
    failure_reasons["error_dof_vel"] = error_dof_vel
    failure_reasons["error_obj_vel"] = error_obj_vel
    failure_reasons["error_obj_ang_vel"] = error_obj_ang_vel
    # contact_violation penalty: -1 if violation occurs, 0 otherwise (scale=1)
    reward_contact_violation = torch.where(contact_violation, -1.0, 0.0)

    # === [新增] 手-物交互向量一致性奖励 (r_interact) ===
    fingertip_indices = [
        dexhand_weight_idx["thumb_tip"][0] - 1,
        dexhand_weight_idx["index_tip"][0] - 1,
        dexhand_weight_idx["middle_tip"][0] - 1,
        dexhand_weight_idx["ring_tip"][0] - 1,
        dexhand_weight_idx["pinky_tip"][0] - 1,
    ]
    
    # 1. 获取参考和仿真的指尖位置 [N, 5, 3]
    p_hand_ref = target_states["joints_pos"][:, fingertip_indices, :]
    p_hand_sim = states["joints_state"][:, [i+1 for i in fingertip_indices], :3] # joints_state 包含 wrist(idx 0)
    
    # 2. 获取参考和仿真的物体关键点位置 [N, 5, 3]
    p_obj_ref = target_states["tips_closest_pt_world"] # [N, 5, 3]
    
    # 仿真关键点: 使用参考中的局部坐标，结合仿真中物体的当前位姿
    closest_obj_idx = target_states["tips_closest_obj_idx"] # [N]
    p_obj_local = target_states["tips_closest_pt_local"] # [N, 5, 3]
    
    # 提取当前被选中的物体在仿真的位姿
    # states["manip_obj_pos"]: [N, K, 3], states["manip_obj_quat"]: [N, K, 4] (x,y,z,w)
    N = states["base_state"].shape[0]
    curr_obj_pos = torch.gather(states["manip_obj_pos"], 1, closest_obj_idx.view(N, 1, 1).expand(-1, 1, 3)).squeeze(1) # [N, 3]
    curr_obj_quat = torch.gather(states["manip_obj_quat"], 1, closest_obj_idx.view(N, 1, 1).expand(-1, 1, 4)).squeeze(1) # [N, 4]
    
    # 将局部坐标转为仿真世界坐标
    curr_obj_rot = quat_to_rotmat(curr_obj_quat[:, [3, 0, 1, 2]]) # [N, 3, 3]

    # === 调试断点 3：检查交互奖励计算中间值 ===
    if torch.isnan(curr_obj_rot).any():
        print("\n[!!!] quat_to_rotmat 产生 NaN ! 可能是四元数全零或未归一化")
        nan_mask = torch.isnan(curr_obj_rot).any(dim=-1).any(dim=-1)
        print(f"非法四元数样例 (第一个故障环境): {curr_obj_quat[nan_mask][0]}")
        # import pdb; pdb.set_trace()

    p_obj_sim = torch.bmm(curr_obj_rot, p_obj_local.transpose(-1, -2)).transpose(-1, -2) + curr_obj_pos.unsqueeze(1)
    
    # 3. 构建相对位置向量 [N, 5, 3]
    v_ref = p_obj_ref - p_hand_ref
    v_sim = p_obj_sim - p_hand_sim
    
    # 4. 计算误差 E_interact (MSE)
    e_interact = torch.mean(torch.norm(v_sim - v_ref, dim=-1)**2, dim=-1) # [N]
    
    # 5. 动态权重 lambda_dynamic
    ref_dist = target_states["tips_distance"].mean(dim=-1) # [N]
    lambda_dynamic = 50.0 + 100.0 * torch.exp(-20.0 * ref_dist)
    
    # 6. 计算最终奖励
    reward_interact = torch.exp(-lambda_dynamic * e_interact)

    # [新增] 接触奖励奖励：如果 reference 中指尖离物体最近点的距离小于5mm，
    # 并且模拟器中指尖也小于5mm且力 > 0.1N，增加额外奖励
    dist_ref = torch.norm(v_ref, dim=-1) # [N, 5]
    dist_sim = torch.norm(v_sim, dim=-1) # [N, 5]
    force_sim = torch.norm(target_states["tip_force"], dim=-1) # [N, 5]
    
    ref_contact_mask = dist_ref < 0.03
    # sim_contact_mask = dist_sim < 0.003

    force_score = torch.clamp((force_sim - 0.1) / (1.5 - 0.1), 0.0, 1.0)
    
    # 只有在位置正确（mask为True）时，力越大奖励越高
    # 基础奖励 4.0 + 额外力奖励 6.0 * force_score
    current_finger_reward = (0.1 + 3.0 * force_score)

    contact_bonus = ((ref_contact_mask).float() * current_finger_reward).sum(dim=-1)
    reward_interact = reward_interact + contact_bonus

    # === [新增] 基于 Bin 的进度奖励 ===
    # 使用外部传入的、基于全长计算的每帧固定奖励
    reward_progress = progress_reward_unit

    # [新增] 通过 bin 的奖励：如果跨越了 bin 边界，给 50 奖励
    bin_now = (progress_buf.float() * progress_reward_unit / 100.0).long()
    bin_prev = ((progress_buf.float() - 1.0).clamp(min=0.0) * progress_reward_unit / 100.0).long()
    reward_bin_pass = torch.where(bin_now > bin_prev, torch.ones_like(reward_progress) * 50.0, torch.zeros_like(reward_progress))

    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps

    # 成功额外奖励 200
    reward_success_bonus = torch.where(succeeded, torch.ones_like(reward_progress) * 200.0, torch.zeros_like(reward_progress))

    reward_execute = (
        1.5 * reward_eef_pos  # 从0.1增加到0.3，提高wrist位置跟踪的权重
        + 2 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.75 * reward_middle_tip_pos
        + 0.6 * reward_pinky_tip_pos
        + 0.6 * reward_ring_tip_pos
        + 0.1 * reward_level_1_pos
        + 0.1 * reward_level_2_pos
        + reward_obj_pos_scale * reward_obj_pos
        + reward_obj_rot_scale * reward_obj_rot
        + 0.6 * reward_eef_vel
        + 0.8 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.1 * reward_obj_vel
        + 0.1 * reward_obj_ang_vel
        + reward_finger_tip_force_scale * reward_finger_tip_force
        + 0.5 * reward_power
        + 0.5 * reward_wrist_power
        + 0.0 * reward_contact_violation
        + reward_interact_scale * reward_interact # 新增
        + 1.0 * (reward_progress + reward_success_bonus + reward_bin_pass) # 新增进度、成功和跨 bin 奖励
    )

    # [恢复] 失败立刻重置，或到达轨迹终点/成功时重置
    reset_buf = torch.where(
        (progress_buf + 1 + 3 >= max_length) | succeeded | failed_execute,
        torch.ones_like(reset_buf),
        torch.zeros_like(reset_buf),
    )
    # [DEBUG] 如果用户确实想在失败时不重置（为了观察），由于我们上面已经加了 cur_idx clamp，
    # 其实不重置也不会报错了。但通常为了训练逻辑正确，到结尾还是应该重置。
    # 这里我们优先保证不报错。

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
        "reward_obj_alpha": alpha, # 新增记录 alpha
        "reward_interact": reward_interact,
        "reward_progress": reward_progress,         # 新增
        "reward_success_bonus": reward_success_bonus, # 新增
        "reward_bin_pass": reward_bin_pass,           # 新增
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

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf, failure_reasons, failure_values
