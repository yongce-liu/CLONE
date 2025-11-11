import os
import argparse
from teleop.torch_utils import *
from teleop.crc import CRC

import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    LowState,
    LowCmd,
    MotorCmd,
)
import time
from collections import deque
from multiprocessing import Process, shared_memory, Array

from vision_wrapper import VisionWrapper
from teleop.local2word import fk_dof
from multiprocessing import Process, shared_memory
from teleop.image_server.image_client import ImageClient
from teleop.gamepad import Gamepad, parse_remote_data
import transforms3d as t3d
from teleop.utils.mat_tool import rotate_quaternion
from pytorch_kinematics import matrix_to_quaternion, quaternion_to_matrix
from threading import Thread
import socket
import pytorch_kinematics as pk
import numpy 
import torch 
import localization
from dex_server import start_dex_server
import datetime
from config import VISION_WRAPPER_BACKEND, VISION_PRO_IP, VISION_PRO_DELTA_H, USE_DEX_HANDS

import mujoco
import mujoco.viewer

import multiprocessing

HW_DOF = 29

WALK_STRAIGHT = False
LOG_DATA = False
MAX_LOG_LEN = 50
LOG_STEP_LEN = 128
HISTROY_LEN = 25

USE_DIFF = True
CLIP_GOAL = True
CLIP_GOAL_DIST = 1.0
USE_GRIPPPER = False
NO_MOTOR = False
NO_ROS = False
FIX_POS = False
VP_POS_DH = VISION_PRO_DELTA_H
LOCK_WAIST = False
NO_REF_VEL = False
DT_BIAS = 0.001
OBS_VEL_SCALE = 0.005
ZERO_GLOBAL = 1 # 0 to disable global
USE_DEX = USE_DEX_HANDS
COLLECT_SID_DATA = False
COLLECT_MOTION_DATA = False
PLAY_MOTION_DATA = None

imu_correction = torch.tensor([0., 0.0, 0., 1.])
imu_correction /= imu_correction.norm()

AVP_IP = VISION_PRO_IP
POLICY_PATH = 'models/g1_student_moel.pt'

crc = CRC()

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

class G1():
    def __init__(self, task='stand'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = task

        self.num_envs = 1 
        self.num_observations = 3328 + (HISTROY_LEN - 25) * 128
        self.num_actions = 29
        self.num_privileged_obs = None
        self.obs_context_len = HISTROY_LEN
        
        self.scale_lin_vel = 2.0
        self.scale_ang_vel = 0.25
        self.scale_orn = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_action = 0.25
        
        # prepare gait commands
        self.cycle_time = 0.64
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        

        # prepare action deployment joint positions offsets and PD gains
        # hip_pgain = 80.
        # hip_dgain = 2.
        # hip_pitch_pgain = 80.
        # hip_pitch_dgain = 2.
        # knee_pgain = 160.
        # knee_dgain = 4.
        # ankle_pgain = 20.
        # ankle_dgain = 0.5
        # waist_pgain = 200.
        # waist_dgain = 5.
        # shoulder_pgain = 40.
        # shoulder_dgain = 1
        # elbow_pgain = 40.
        # elbow_dgain = 1
        # wrist_roll_pgain = 40.
        # wrist_roll_dgain = 0.5
        # wrist_pitch_pgain = 40.
        # wrist_pitch_dgain = 0.5
        # wrist_yaw_pgain = 40.
        # wrist_yaw_dgain = 0.5

        # self.p_gains = np.array([hip_pitch_pgain,hip_pgain,hip_pgain,knee_pgain,ankle_pgain,ankle_pgain,hip_pitch_pgain,hip_pgain,hip_pgain,knee_pgain,ankle_pgain,ankle_pgain,waist_pgain,waist_pgain,waist_pgain,shoulder_pgain,shoulder_pgain,shoulder_pgain,elbow_pgain,wrist_roll_pgain,wrist_pitch_pgain,wrist_yaw_pgain,shoulder_pgain,shoulder_pgain,shoulder_pgain,elbow_pgain,wrist_roll_pgain,wrist_pitch_pgain,wrist_yaw_pgain])
        # self.d_gains = np.array([hip_pitch_dgain,hip_dgain,hip_dgain,knee_dgain,ankle_dgain,ankle_dgain,hip_pitch_dgain,hip_dgain,hip_dgain,knee_dgain,ankle_dgain,ankle_dgain,waist_dgain,waist_dgain,waist_dgain,shoulder_dgain,shoulder_dgain,shoulder_dgain,elbow_dgain,wrist_roll_dgain,wrist_pitch_dgain,wrist_yaw_dgain,shoulder_dgain,shoulder_dgain,shoulder_dgain,elbow_dgain,wrist_roll_dgain,wrist_pitch_dgain,wrist_yaw_dgain])
        
        hip_pgain = 80.
        hip_dgain = 2.
        hip_roll_pgain = 120.
        hip_roll_dgain = 3.
        knee_pgain = 160.
        knee_dgain = 4.
        ankle_pgain = 40.
        ankle_dgain = 1.
        waist_yaw_pgain = 200.
        waist_yaw_dgain = 5.
        waist_pgain = 200.
        waist_dgain = 5.
        shoulder_pgain = 40.
        shoulder_dgain = 1
        elbow_pgain = 40.
        elbow_dgain = 1
        wrist_roll_pgain = 20.
        wrist_roll_dgain = 0.5
        wrist_pitch_pgain = 20.
        wrist_pitch_dgain = 0.5
        wrist_yaw_pgain = 20.
        wrist_yaw_dgain = 0.5

        self.p_gains = np.array([hip_pgain,hip_roll_pgain,hip_pgain,knee_pgain,ankle_pgain,ankle_pgain,hip_pgain,hip_roll_pgain,hip_pgain,knee_pgain,ankle_pgain,ankle_pgain,waist_yaw_pgain,waist_pgain,waist_pgain,shoulder_pgain,shoulder_pgain,shoulder_pgain,elbow_pgain,wrist_roll_pgain,wrist_pitch_pgain,wrist_yaw_pgain,shoulder_pgain,shoulder_pgain,shoulder_pgain,elbow_pgain,wrist_roll_pgain,wrist_pitch_pgain,wrist_yaw_pgain])
        # self.p_gains[27] = 0.
        self.d_gains = np.array([hip_dgain,hip_roll_dgain,hip_dgain,knee_dgain,ankle_dgain,ankle_dgain,hip_dgain,hip_roll_dgain,hip_dgain,knee_dgain,ankle_dgain,ankle_dgain,waist_yaw_dgain,waist_dgain,waist_dgain,shoulder_dgain,shoulder_dgain,shoulder_dgain,elbow_dgain,wrist_roll_dgain,wrist_pitch_dgain,wrist_yaw_dgain,shoulder_dgain,shoulder_dgain,shoulder_dgain,elbow_dgain,wrist_roll_dgain,wrist_pitch_dgain,wrist_yaw_dgain])
        

        self.joint_limit_lo_ = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -100., -0.2618, -2.5307,-2.9671,-2.7576,-0.087267,-100.,-0.2618,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558])
        self.joint_limit_hi_ = np.array([2.8798, 2.9671, 2.7576, 2.8798, 100., 0.2618, 2.8798, 0.5236, 2.7576, 2.8798, 100., 0.2618, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558])

        # self.joint_limit_lo_ = [-2.5307, -0.5236, -2.7576, -0.087267, -100, -100, -2.5307,-2.9671,-2.7576,-0.087267,-100,-100,-2.5307,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]
        # self.joint_limit_hi_ = [2.8798, 2.9671, 2.7576, 2.8798, 100, 100, 2.8798, 0.5236, 2.7576, 2.8798, 100, 100, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]

        # self.joint_limit_lo = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -100, -100, -2.5307,-2.9671,-2.7576,-0.087267,-100,-100,-2.5307,-100.,-100.,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558])
        # self.joint_limit_hi = np.array([2.8798, 2.9671, 2.7576, 2.8798, 100, 100, 2.8798, 0.5236, 2.7576, 2.8798, 100, 100, 2.618, 100., 100.,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558])

        self.joint_limit_lo = np.array([-2.5307, -0.5236, -2.7576, -100., -100, -100, -2.5307,-2.9671,-2.7576, -100.,-100,-100,-2.5307,-100.,-100.,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-100.,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558])
        self.joint_limit_hi = np.array([2.8798, 2.9671, 2.7576, 2.8798, 100, 100, 2.8798, 0.5236, 2.7576, 2.8798, 100, 100, 2.618, 100., 100.,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,100.,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558])

        free_limit_joints = [0, 6, 19, 20, 21, 26, 27, 28, 3, 9]
        # self.joint_limit_lo_[free_limit_joints], self.joint_limit_hi_[free_limit_joints] = -100., 100.
        self.joint_limit_lo[free_limit_joints], self.joint_limit_hi[free_limit_joints] = -100., 100.

        self.action_mask = np.ones(29)
        if LOCK_WAIST:
            self.action_mask[12:15] = 0.

        self.torque_limit = np.array([ 88., 139.,  88., 139.,  50.,  1000.,  88., 139.,  88., 139.,  50.,  1000.,
                                    88.,  50.,  50.,  25.,  25.,  25.,  25.,  25.,   4.,   4.,  25.,  25.,
                                    25.,  25.,  25.,   4.,   4.])
        self.max_delta_angles = self.torque_limit / self.p_gains
        self.max_delta_angle_factor = np.zeros_like(self.max_delta_angles)

        self.soft_dof_pos_limit = 0.98
        self.soft_torque_limit = 0.99

        self.dof_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']

        for i in range(len(self.joint_limit_lo)):
            # soft limits
            m = (self.joint_limit_lo[i] + self.joint_limit_hi[i]) / 2
            r = self.joint_limit_hi[i] - self.joint_limit_lo[i]
            # self.joint_limit_lo[i] = m - 0.5 * r * self.soft_dof_pos_limit
            # self.joint_limit_hi[i] = m + 0.5 * r * self.soft_dof_pos_limit
            self.joint_limit_lo[i] *= self.soft_dof_pos_limit
            self.joint_limit_hi[i] *= self.soft_dof_pos_limit
        
        self.lower_body_mask = np.ones(29)
        self.lower_body_mask[:12] = 0.

        self.default_dof_pos_np = np.zeros(29)
        
        self.default_dof_pos_np = np.array([
                                            -0.1, #left hip pitch
                                            0.0, #left hip roll
                                            0.0, #left hip pitch
                                            0.3, #left knee
                                            -0.2, #left ankle pitch 
                                            0, #left ankle roll 
                                            -0.1, #right hip pitch
                                            0.0, #right hip roll
                                            0.0, #right hip pitch
                                            0.3, #right knee
                                            -0.2, #right ankle pitch 
                                            0, #right ankle roll 
                                            0, #waist
                                            0, #waist
                                            0, #waist
                                            0.0,
                                            0.0,
                                            0.,
                                            0.0,
                                            0.,
                                            0.,
                                            0.,
                                            0.0,
                                            -0.0,
                                            0.,
                                            0.0,
                                            0.,
                                            0.,
                                            0.,
                                            ])
        
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)

        print(f"default_dof_pos.shape: {self.default_dof_pos.shape}")

        # prepare osbervations buffer
        self.obs_buf = torch.zeros(1, self.num_observations, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_history = deque(maxlen=self.obs_context_len)
        for _ in range(self.obs_context_len):
            self.obs_history.append(torch.zeros(
                1, self.num_observations, dtype=torch.float, device=self.device))
            
        self.trajectories = np.zeros((MAX_LOG_LEN * LOG_STEP_LEN))
        self.init_mujoco_viewer()


    def init_mujoco_viewer(self):

        device = torch.device("cpu")
        curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
        humanoid_xml = "resources/g1.xml"

        self.mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = dt
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)


        for _ in range(2):
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0.5, 1.0, 0.5, 0.5]))
        add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0.5, 1.0, 1.0, 0.5]))
        add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1.0, 1.0, 1.0, 0.5]))

            

    def record_trajectory(self, dof_pos, dof_vel, base_ang_vel, projected_gravity, actions, task_obs):
        current_obs_a = torch.cat((dof_pos, dof_vel, base_ang_vel, projected_gravity, actions, task_obs), dim=0).numpy()
        self.trajectories[1 * LOG_STEP_LEN:] = self.trajectories[:-1 * LOG_STEP_LEN].copy()
        self.trajectories[0 * LOG_STEP_LEN: 1 * LOG_STEP_LEN] = current_obs_a
            

class DeployNode(Node):

    class WirelessButtons:
        R1 =            0b00000001 # 1
        L1 =            0b00000010 # 2
        start =         0b00000100 # 4
        select =        0b00001000 # 8
        R2 =            0b00010000 # 16
        L2 =            0b00100000 # 32
        F1 =            0b01000000 # 64
        F2 =            0b10000000 # 128
        A =             0b100000000 # 256
        B =             0b1000000000 # 512
        X =             0b10000000000 # 1024
        Y =             0b100000000000 # 2048
        up =            0b1000000000000 # 4096
        right =         0b10000000000000 # 8192
        down =          0b100000000000000 # 16384
        left =          0b1000000000000000 # 32768

    def __init__(self, task='stand'):
        super().__init__("deploy_node")  # type: ignore
        
        # init subcribers & publishers
        # self.joy_stick_sub = self.create_subscription(WirelessController, "wirelesscontroller", self._joy_stick_callback, 1)
        # self.joy_stick_sub  # prevent unused variable warning
        self.lowlevel_state_sub = self.create_subscription(LowState, "lowstate", self.lowlevel_state_cb, 1)  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        self.low_state = LowState()
        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)
        self.joint_tau = np.zeros(HW_DOF)
        self.real_q = np.zeros(HW_DOF)
        self.last_action = np.zeros(HW_DOF)
        self.projected_gravity = np.zeros(3)

        self.motor_pub = self.create_publisher(LowCmd, "lowcmd_buffer", 1)
        self.motor_pub_freq = 50
        self.control_dt = 1/self.motor_pub_freq - DT_BIAS
        self.vp_freq = 50
        self.vp_dt = 1/self.vp_freq

        self.vp_head_seq = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self.vp_lhand_seq = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self.vp_rhand_seq = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self.vp_d_seqlen = 5

        self.cmd_msg = LowCmd()

        self.cmd_msg.mode_pr = 0
        self.cmd_msg.mode_machine = 5

        # init motor command
        self.motor_cmd = []
        for id in range(HW_DOF):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=1, reserve=0)
            self.motor_cmd.append(cmd)
        for id in range(HW_DOF, 35):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0, reserve=0)
            self.motor_cmd.append(cmd)
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # init policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.start_policy = False

        # standing up
        self.get_logger().info("Standing up")
        self.stand_up = False
        self.stand_up = True

        self.last_lock_time = time.monotonic()
        self.last_record_time = time.monotonic()
        self.recording = False
        self.rec_delta_pos = torch.zeros(3)
        self.motion_data = {'meta': None, 'seq': []}

        self.playing = False
        self.play_initial_pos = torch.zeros(3)
        self.last_play_time = time.monotonic()
        self.motion_buffer = None
        self.motion_idx = 0

        # commands 
        self.lin_vel_deadband = 0.1
        self.ang_vel_deadband = 0.1
        self.move_by_wireless_remote = True
        self.cmd_px_range = [0.1, 0.5]
        self.cmd_nx_range = [0.1, 0.5]
        self.cmd_py_range = [0.1, 0.4]
        self.cmd_ny_range = [0.1, 0.4]
        self.cmd_pyaw_range = [0.2, 0.6]
        self.cmd_nyaw_range = [0.2, 0.6]

        # start
        self.start_time = time.monotonic()
        self.get_logger().info("Press L1 to start policy")
        self.get_logger().info("Press L2 for emergent stop")
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_pos_hist = []
        self.dof_vel_hist = []
        self.imu_hist = []
        self.ang_vel_hist = []
        self.foot_contact_hist = []
        self.tau_hist = []
        self.obs_hist = []

        if COLLECT_SID_DATA:
            self.sid_data = {'kp': self.env.p_gains, 'kd': self.env.d_gains, 'data_points': []}

        self.over_limit = np.zeros(HW_DOF).astype(bool)

        ##########################################
        self.head_vel = torch.zeros(3)
        self.left_wrist_vel = torch.zeros(3)
        self.right_wrist_vel = torch.zeros(3)
        
        self.actual_head_pos = torch.zeros(3)
        self.head_pos = torch.zeros(3)
        self.left_wrist_pos = torch.zeros(3)
        self.right_wrist_pos = torch.zeros(3)

        self.vp_delta_pos = torch.zeros(3)

        self.invalid_hands = [False, False]

        self.left_hand = None
        self.right_hand = None
        ##########################################

        shm_send_shape = (3 + 4 + 4 + HW_DOF, )
        shm_recv_shape = (3 + 4 + (HW_DOF + 1 + 3) * 3, )
        shm_vprotate_quat = (4)
        self.loc_shm_send = shared_memory.SharedMemory(create=True, size=np.prod(shm_send_shape) * 4)
        self.loc_shm_recv = shared_memory.SharedMemory(create=True, size=np.prod(shm_recv_shape) * 4)
        self.loc_shm_vprot = shared_memory.SharedMemory(create=True, size=np.prod(shm_vprotate_quat) * 4)
        self.loc_shm_vprot = shared_memory.SharedMemory(create=True, size=np.prod(shm_vprotate_quat) * 4)
        self.loc_send_data = np.ndarray(shm_send_shape, dtype=np.float32, buffer=self.loc_shm_send.buf)

        self.last_loc_reset_time = time.monotonic()
        self.last_vp_reset_time = time.monotonic()

        self.loc_offset = self.loc_send_data[0:3]
        self.obs_quat = self.loc_send_data[3:7]
        self.obs_quat[:] = np.array([0., 0., 0., 1.], dtype=np.float32)
        self.loc_delta_rot = self.loc_send_data[7:11]
        self.loc_delta_rot[:] = np.array([0., 0., 0., 1.], dtype=np.float32)
        self.obs_joint_pos = self.loc_send_data[11:]

        self.loc_delta_angle = 0.

        self.joint_temperature = np.zeros((29))

        self.loc_recv_data = torch.frombuffer(self.loc_shm_recv.buf, dtype=torch.float32)
        self.location = self.loc_recv_data[0:3].numpy()
        self.obs_head_quat = self.loc_recv_data[3:7].numpy()
        self.body_pos_extend = self.loc_recv_data[7:].view((HW_DOF + 1 + 3), 3)

        if USE_DEX:
            self.dex_shm_send = shared_memory.SharedMemory(create=True, size=(75 * 2 + 7 * 2 + 1) * 4)
            self.dex_shm_send_data = np.ndarray(((75 * 2 + 7 * 2 + 1),), dtype=np.float32, buffer=self.dex_shm_send.buf)
            self.dex_shm_recv = shared_memory.SharedMemory(create=True, size=14 * 4)
            self.dex_shm_recv_data = np.ndarray((14, ), dtype=np.float32, buffer=self.dex_shm_recv.buf)

        self.loc_process = Process(target=localization.start_service, args=[self.loc_shm_send.name, self.loc_shm_recv.name, self.loc_shm_vprot.name])
        self.loc_process.start()
        
        self.head_rot = torch.zeros(4)
        self.left_hand_rot = torch.zeros(4)
        self.right_hand_rot = torch.zeros(4)
        self.vp_delta_rot = torch.zeros(4)
        self.vp_delta_rot_mat = np.zeros((4, 4), dtype=np.float32)
        self.vp_delta_rot_mat[0, 0] = 1.
        self.vp_delta_rot_mat[1, 1] = 1.
        self.vp_delta_rot_mat[2, 2] = 1.
        self.vp_delta_rot_mat[3, 3] = 1.

        self.head_rot[-1] = 1.
        self.left_hand_rot[-1] = 1.
        self.right_hand_rot[-1] = 1.
        self.vp_delta_rot[-1] = 1.

        # cmd and observation
        # self.xyyaw_command = np.array([0, 0., 0.], dtype= np.float32)
        self.xyyaw_command = np.array([0, 0., 0.], dtype= np.float32)
        self.commands_scale = np.array([self.env.scale_lin_vel, self.env.scale_lin_vel, self.env.scale_ang_vel])

        self.phase = torch.zeros(1, device=self.device, dtype=torch.float)

        self.Emergency_stop = False
        self.stop = False

        self.gamepad = Gamepad()

        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.server_address = ('127.0.0.1', 5701)

        self.sock_plot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address_plot = ('127.0.0.1', 5702)
        self.sock_plot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address_plot = ('127.0.0.1', 5702)

        time.sleep(1)

    def launch_dex_hands(self):
        self.dex_process = multiprocessing.get_context('spawn').Process(target=start_dex_server, args=[self.dex_shm_send.name, self.dex_shm_recv.name])
        self.dex_process.start()

    def prepare(self):
        self.launch_vision_pro()
        if USE_DEX:
            self.launch_dex_hands()  

    def reset_localization(self):
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        if time.monotonic() - self.last_loc_reset_time > 0.5:
            self.last_loc_reset_time = time.monotonic()
            # loc_offset = self.loc_offset.copy()
            rel_loc_offset = np.zeros((3))
            # rel_loc_offset[2:3] = self.head_pos[2] - self.body_pos_extend[..., 2].max(dim=0, keepdim=True).values.numpy()
            # rel_loc_offset[0:2] = self.head_pos[0:2] - self.location[0:2]
            rel_loc_offset[2:3] = -self.body_pos_extend[..., 2].min(dim=0, keepdim=True).values.numpy() + 0.015
            rel_loc_offset[0:2] = -self.location[0:2] + self.head_pos[0:2].numpy()

            # import pdb; pdb.set_trace()
            # self.loc_offset[:] = loc_offset[:]
            self.loc_offset[:] +=  rel_loc_offset

            cur_quat = torch.from_numpy(self.obs_quat.copy()).unsqueeze(0)
            head_quat = torch.from_numpy(self.obs_head_quat.copy()).unsqueeze(0)

            cur_quat = calc_heading_quat(cur_quat)
            head_quat = calc_heading_quat(head_quat)

            cur_heading = calc_heading(cur_quat)
            head_heading = calc_heading(head_quat)
            heading = cur_heading - (2 * np.pi - head_heading)

            axis = torch.zeros_like(cur_quat[..., 0:3])
            axis[..., 2] = 1

            # quat_delta_buf = quat_mul(quat_conjugate(cur_quat), head_quat)
            quat_delta_buf = quat_from_angle_axis(heading, axis)
            print('DEGREE', heading, 'Head DEG', head_heading, 'Cur DEG', cur_heading)
            self.loc_delta_rot[:] = quat_delta_buf[0].numpy()[:]
            self.loc_delta_angle = heading.item()

            # print("vp_HEAD:", self.head_rot)
            # print("rel_OFFSET:", rel_loc_offset)
            # print("min_BODY:", self.body_pos_extend[..., 2].min(dim=0, keepdim=True).values.numpy())
            # print("cur_LOCATION:", self.location)
            print('Location Offset Reset')

    def reset_vision_pro(self):
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        if time.monotonic() - self.last_vp_reset_time > 0.5:
            self.last_vp_reset_time = time.monotonic()

            cur_quat = torch.from_numpy(self.obs_quat.copy()).unsqueeze(0)
            head_quat = self.head_rot.unsqueeze(0)

            cur_quat = calc_heading_quat(cur_quat)
            head_quat = calc_heading_quat(head_quat)

            cur_heading = calc_heading(cur_quat)
            head_heading = calc_heading(head_quat)
            heading = cur_heading - head_heading

            axis = torch.zeros_like(cur_quat[..., 0:3])
            axis[..., 2] = 1

            # quat_delta_buf = quat_mul(quat_conjugate(cur_quat), head_quat)
            quat_delta_buf = quat_from_angle_axis(heading, axis)
            print('DEGREE', heading, 'Head DEG', head_heading, 'Cur DEG', cur_heading)
            self.vp_delta_rot[:] = quat_mul(quat_delta_buf, self.vp_delta_rot[None, :])[0]
            self.vp_delta_rot_mat[:3, :3] = quaternion_to_matrix(self.vp_delta_rot.roll(1, dims=0)).numpy()
            

            # print("vp_HEAD:", self.head_rot)
            # print("rel_OFFSET:", rel_loc_offset)
            # print("min_BODY:", self.body_pos_extend[..., 2].min(dim=0, keepdim=True).values.numpy())
            # print("cur_LOCATION:", self.location)
            print('Location Offset Reset')

    ##############################
    # subscriber callbacks
    ##############################
    def lowlevel_state_cb(self, msg: LowState):
        # wireless_remote btn
        joystick_data = msg.wireless_remote
        parsed_data = parse_remote_data(joystick_data)
        self.gamepad.update(parsed_data)
        
        if self.gamepad.L1.pressed:
            print(f'Policy start!')
            self.start_policy = True
        if self.gamepad.L2.pressed:
            self.start_policy = False
            self.Emergency_stop = False
            self.stop = True
            # print(f'Manual emergency stop!!!')
            # self.get_logger().info("Program exiting")
        if self.gamepad.R1.pressed:
            self.reset_vision_pro()
        if self.gamepad.R2.pressed:
            self.reset_localization()
        if self.gamepad.A.pressed and USE_DEX:
            if time.monotonic() - self.last_lock_time > 1.0:
                if self.dex_shm_send_data[164] < 1.5:
                    self.dex_shm_send_data[164] = 2.
                else:
                    self.dex_shm_send_data[164] = 1.
            self.last_lock_time = time.monotonic()
        if self.gamepad.B.pressed and COLLECT_MOTION_DATA:
            if time.monotonic() - self.last_record_time > 1.0:
                self.recording = not self.recording
                if self.recording:
                    self.rec_delta_pos = self.head_pos.clone()
                    self.rec_delta_pos[..., 2] = 0.
                if not self.recording:
                    torch.save(self.motion_data, f'./rec_motion/motion_current.pkl')
                    print('Motion Data Saved')
                self.last_record_time = time.monotonic()
                print(f'RECORDING {self.recording}')
        if self.gamepad.start.pressed and not PLAY_MOTION_DATA is None:
            if time.monotonic() - self.last_play_time > 1.0:
                if not self.playing:
                    self.play_initial_pos = self.head_pos.clone()
                    self.play_initial_pos[..., 2] = 0.
                    self.motion_buffer = torch.load(PLAY_MOTION_DATA)
                if self.playing:
                    vp_delta_pos = self.head_pos - self.actual_head_pos
                    vp_delta_pos[..., 2] = 0.
                    self.vp_delta_pos = vp_delta_pos

                self.playing = not self.playing
                if not self.playing:
                    self.motion_idx = 0
                self.last_play_time = time.monotonic()
                
        

        # print(self.body_pos_extend[-2], self.body_pos_extend[..., 2].min(dim=0, keepdim=True).values.numpy(), self.location)

        if self.move_by_wireless_remote:
            lx, ly = self.gamepad.lx, self.gamepad.ly
            if abs(lx) > 0.05 or abs(ly) > 0.05:
                self.loc_offset[0] += lx * 0.008 * np.cos(self.loc_delta_angle) + ly * 0.008 * np.sin(self.loc_delta_angle)
                self.loc_offset[1] += - lx * 0.008 * np.sin(self.loc_delta_angle) + ly * 0.008 * np.cos(self.loc_delta_angle)
        # print(self.loc_offset)
        # imu data
        imu_data = msg.imu_state
        self.msg_tick = msg.tick/1000
        self.roll, self.pitch, self.yaw = imu_data.rpy
        quat_temp = np.roll(np.array(imu_data.quaternion, dtype=np.float32), shift=-1, axis=-1)
        self.obs_quat[:] = quat_mul(torch.from_numpy(quat_temp)[None, :], imu_correction[None, :])[0].numpy()
        self.obs_acc = np.array([imu_data.accelerometer], dtype=np.float32)
        self.obs_ang_vel = np.array(imu_data.gyroscope, dtype=np.float32)
        self.obs_imu = np.array([self.roll, self.pitch, self.yaw], dtype=np.float32)

        # termination condition
        r_threshold = abs(self.roll) > 0.6
        p_threshold = abs(self.pitch) > 0.6
        if r_threshold or p_threshold:
            self.get_logger().warning("Roll or pitch threshold reached")

        # motor data
        self.joint_tau[:] = np.array([msg.motor_state[i].tau_est for i in range(HW_DOF)])
        self.joint_pos = [msg.motor_state[i].q for i in range(HW_DOF)]

        self.joint_temperature[:] = np.array([msg.motor_state[i].temperature[1] for i in range(HW_DOF)])
        # self.env.max_delta_angle_factor[(self.joint_temperature > 60) & (self.joint_temperature <= 80)] = 1.0
        # self.env.max_delta_angle_factor[(self.joint_temperature > 80) & (self.joint_temperature <= 100)] = 0.5
        # self.env.max_delta_angle_factor[(self.joint_temperature > 100) & (self.joint_temperature <= 115)] = 0.1
        # self.env.max_delta_angle_factor[(self.joint_temperature > 115)] = 0.0
        # self.env.max_delta_angle_factor[(self.joint_temperature <= 60)] = 1.0
        factor = 1 - ((self.joint_temperature - 80) / 35).clip(0., 1.)
        self.env.max_delta_angle_factor = factor.astype(np.float64)
        # idxs = np.arange(HW_DOF)[self.joint_temperature > 80]
        # if len(idxs) > 0:
        #     print('High Temperature Warning:', idxs)

        self.obs_joint_pos[:] = np.array(self.joint_pos, dtype=np.float32)

        self.over_limit = (self.obs_joint_pos <= (self.env.joint_limit_lo_ * self.env.soft_dof_pos_limit)).astype(int) + \
                         (self.obs_joint_pos >= (self.env.joint_limit_hi_ * self.env.soft_dof_pos_limit)).astype(int)
        self.over_limit = self.over_limit.astype(bool)
        
        joint_vel = [msg.motor_state[i].dq for i in range(HW_DOF)]
        self.obs_joint_vel = np.array(joint_vel, dtype=np.float32)
        # print(self.obs_joint_vel)

        if COLLECT_SID_DATA:
            self.sid_data['data_points'].append({'q': self.obs_joint_pos.copy(), 'dq': self.obs_joint_vel.copy(), 'q_tgt': self.real_q.copy(), 'tau': self.joint_tau.copy(), 'tau_est': (self.real_q - self.obs_joint_pos) * self.env.p_gains - self.obs_joint_vel * self.env.d_gains, 'temp': self.joint_temperature.copy()})


        quat_xyzw = torch.from_numpy(self.obs_quat.copy())[None, :]
        gravity_vector = torch.tensor([[0., 0., -1.]], dtype=torch.float32)
        self.projected_gravity = quat_rotate_inverse(quat_xyzw, gravity_vector)[0].cpu().numpy()

        # heading_quat_inv = torch_utils.calc_heading_quat_inv(quat_xyzw)
        # gravity = quat_rotate(quat_mul(quat_xyzw, heading_quat_inv), gravity_vector * 9.81)[0].cpu().numpy()
        # print(self.obs_acc + gravity)

        # print(self.joint_pos)

        # Joint limit check
        if self.start_policy and (((np.array(self.joint_pos)-np.array(self.env.joint_limit_lo))<0).sum() >0 or ((np.array(self.joint_pos)-np.array(self.env.joint_limit_hi))>0).sum() > 0):
            print("Joint limit reached")
            print('Low state', self.joint_pos)
            print('Low cmd', self.prev_action)
            print("Low limit Joint index: ", np.where((np.array(self.joint_pos)-np.array(self.env.joint_limit_lo))<0))
            print("High limit Joint index: ", np.where((np.array(self.joint_pos)-np.array(self.env.joint_limit_hi))>0))
            Warning("Emergency stop")
            self.stop = True
    ##############################
    # motor commands
    ##############################

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
        for i in range(HW_DOF):
            self.motor_cmd[i].kp = kp[i]  #*0.5
            self.motor_cmd[i].kd = kd[i]  #*3

    def set_motor_position(
        self,
        q: np.ndarray,
    ):
        max_delta_angles = self.env.max_delta_angles * self.env.max_delta_angle_factor
        q = np.clip(q, self.obs_joint_pos - max_delta_angles, self.obs_joint_pos + max_delta_angles)
        for i in range(HW_DOF):
            self.motor_cmd[i].q = q[i]
        self.real_q[:] = q
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        # self.cmd_msg.crc = get_crc(self.cmd_msg)
        self.cmd_msg.crc = crc.Crc(self.cmd_msg)
    ##############################
    # deploy policy
    ##############################
    def init_policy(self):
        self.get_logger().info("Preparing policy")
        faulthandler.enable()

        # prepare environment
        self.env = G1(task='self.task')

        # load policy
        file_pth = os.path.dirname(os.path.realpath(__file__))
        self.policy = torch.jit.load(os.path.join(file_pth, POLICY_PATH), map_location=self.env.device)  #0253 396
        self.policy.to(self.env.device)
        actions = self.policy(self.env.obs_buf.detach().reshape(1, -1))  # first inference takes longer time
        # self.policy = None
        # init p_gains, d_gains, torque_limits
        for i in range(HW_DOF):
            self.motor_cmd[i].q = self.env.default_dof_pos[0][i].item()
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # float(self.env.p_gains[i])  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.angles = self.env.default_dof_pos.cpu().clone().numpy()
    
    def get_walking_cmd_mask(self):
        walking_mask0 = np.abs(self.xyyaw_command[0]) > 0.1
        walking_mask1 = np.abs(self.xyyaw_command[1]) > 0.1
        walking_mask2 = np.abs(self.xyyaw_command[2]) > 0.2
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2

        walking_mask = walking_mask | (self.env.gait_indices.cpu() >= self.control_dt / self.env.cycle_time).numpy()[0]
        walking_mask |= np.logical_or(np.abs(self.obs_imu[1])>0.05, np.abs(self.obs_imu[0])>0.05)
        return walking_mask
    
    def  _get_phase(self):
        phase = self.env.gait_indices
        return phase
    
    def step_contact_targets(self):
        cycle_time = self.env.cycle_time
        standing_mask = ~self.get_walking_cmd_mask()
        self.env.gait_indices = torch.remainder(self.env.gait_indices + self.control_dt / cycle_time, 1.0)
        if standing_mask:
            self.env.gait_indices[:] = 0
    
    @torch.inference_mode()
    def get_vp_data(self):
        vel_factor = 1. if not NO_REF_VEL else 0.
        left_vel_factor = 1.
        right_vel_factor = 1.

        data = self.tv_wrapper.get_data_full()
        if not data is None and not FIX_POS and (not PLAY_MOTION_DATA or not self.playing):
            head_mat, left_wrist, right_wrist, left_hand, right_hand = data
            head_mat[2, 3] += VP_POS_DH
            left_wrist[2, 3] += VP_POS_DH
            right_wrist[2, 3] += VP_POS_DH

            if USE_DEX:
                self.dex_shm_send_data[0:75] = left_hand.astype(np.float32).flatten()
                self.dex_shm_send_data[75:150] = right_hand.astype(np.float32).flatten()

            head_mat = self.vp_delta_rot_mat @ head_mat
            left_wrist = self.vp_delta_rot_mat @ left_wrist
            right_wrist = self.vp_delta_rot_mat @ right_wrist

            # head_mat[:3, 3] = my_quat_rotate(self.vp_delta_rot[None, :], head_mat[None, :3, 3])[0]
            # left_wrist[:3, 3] = my_quat_rotate(self.vp_delta_rot[None, :], left_wrist[None, :3, 3])[0]
            # right_wrist[:3, 3] = my_quat_rotate(self.vp_delta_rot[None, :], right_wrist[None, :3, 3])[0]

            self.vp_head_seq.append(head_mat[:3, 3])
            self.vp_lhand_seq.append(left_wrist[:3, 3])
            self.vp_rhand_seq.append(right_wrist[:3, 3])
            if len(self.vp_head_seq) > self.vp_d_seqlen:
                self.vp_head_seq = self.vp_head_seq[-self.vp_d_seqlen:]
            if len(self.vp_lhand_seq) > self.vp_d_seqlen:
                self.vp_lhand_seq = self.vp_lhand_seq[-self.vp_d_seqlen:]
            if len(self.vp_rhand_seq) > self.vp_d_seqlen:
                self.vp_rhand_seq = self.vp_rhand_seq[-self.vp_d_seqlen:]

            head_seq = np.stack(self.vp_head_seq)
            lhand_seq = np.stack(self.vp_lhand_seq)
            rhand_seq = np.stack(self.vp_rhand_seq)

            head_d = np.gradient(head_seq, self.vp_dt, axis=0, edge_order=2)
            lhand_d = np.gradient(lhand_seq, self.vp_dt, axis=0, edge_order=2)
            rhand_d = np.gradient(rhand_seq, self.vp_dt, axis=0, edge_order=2)

            if ((left_wrist[:3, 3] - head_mat[:3, 3]) ** 2).sum() < 0.1:
                left_vel_factor = 0.
                self.invalid_hands[0] = True
            elif self.invalid_hands[0]:
                left_vel_factor = 0.
                self.invalid_hands[0] = False

            if ((right_wrist[:3, 3] - head_mat[:3, 3]) ** 2).sum() < 0.1:
                right_vel_factor = 0.
                self.invalid_hands[1] = True
            elif self.invalid_hands[1]:
                right_vel_factor = 0.
                self.invalid_hands[1] = False
            
            self.head_vel[:] = torch.from_numpy(head_d[-3]) * vel_factor
            self.left_wrist_vel[:] = torch.from_numpy(lhand_d[-3]) * vel_factor * left_vel_factor
            self.right_wrist_vel[:] = torch.from_numpy(rhand_d[-3]) * vel_factor * right_vel_factor

            head_mat = torch.from_numpy(head_mat).float()
            left_wrist = torch.from_numpy(left_wrist).float()
            right_wrist = torch.from_numpy(right_wrist).float()
            
            # self.head_rot[:] = quat_mul(matrix_to_quaternion(head_mat[:3, :3]).roll(-1, dims=-1)[None, :], self.vp_delta_rot[None, :])[0]
            # self.left_hand_rot[:] = quat_mul(matrix_to_quaternion(left_wrist[:3, :3]).roll(-1, dims=-1)[None, :], self.vp_delta_rot[None, :])[0]
            # self.right_hand_rot[:] = quat_mul(matrix_to_quaternion(right_wrist[:3, :3]).roll(-1, dims=-1)[None, :], self.vp_delta_rot[None, :])[0]
        
            self.head_rot[:] = matrix_to_quaternion(head_mat[:3, :3]).roll(-1, dims=-1)
            self.left_hand_rot[:] = matrix_to_quaternion(left_wrist[:3, :3]).roll(-1, dims=-1)
            self.right_hand_rot[:] = matrix_to_quaternion(right_wrist[:3, :3]).roll(-1, dims=-1)

            head_delta = my_quat_rotate(self.head_rot[None, :], torch.tensor([[1., 0., 0.]]))[0] * -0.06
            lhand_delta = my_quat_rotate(self.left_hand_rot[None, :], torch.tensor([[1., 0., 0.]]))[0] * 0.06
            rhand_delta = my_quat_rotate(self.right_hand_rot[None, :], torch.tensor([[1., 0., 0.]]))[0] * 0.06

            self.head_pos = head_mat[:3, 3] + head_delta + self.vp_delta_pos
            if not self.invalid_hands[0]:
                self.left_wrist_pos = left_wrist[:3, 3] + lhand_delta + self.vp_delta_pos
            if not self.invalid_hands[1]:
                self.right_wrist_pos = right_wrist[:3, 3] + rhand_delta + self.vp_delta_pos

        elif PLAY_MOTION_DATA and self.playing:
            if not data is None:
                head_mat, left_wrist, right_wrist, left_hand, right_hand = data
                head_mat = self.vp_delta_rot_mat @ head_mat
                self.actual_head_pos[:] = torch.from_numpy(head_mat[:3, 3]).float()

            cur_motion = self.motion_buffer['seq'][self.motion_idx]
            self.head_pos = self.play_initial_pos + cur_motion[0:3]
            self.left_wrist_pos = self.play_initial_pos + cur_motion[3:6]
            self.right_wrist_pos = self.play_initial_pos + cur_motion[6:9]

            self.head_vel = self.play_initial_pos + cur_motion[0+9:3+9]
            self.left_wrist_vel = self.play_initial_pos + cur_motion[3+9:6+9]
            self.right_wrist_vel = self.play_initial_pos + cur_motion[6+9:9+9]

            self.left_hand_rot = cur_motion[18:22]
            self.right_hand_rot = cur_motion[22:26]
            # head_mat = torch.cat([torch.diag(torch.ones(4)), torch.ones(4)[:, None]], dim=1)
            # left_wrist = torch.cat([torch.diag(torch.ones(4)), torch.ones(4)[:, None]], dim=1)
            # right_wrist = torch.cat([torch.diag(torch.ones(4)), torch.ones(4)[:, None]], dim=1)

            # head_mat[:3, 3] = cur_motion[0:3]
            # left_wrist[:3, 3] = cur_motion[3:6]
            # right_wrist[:3, 3] = cur_motion[6:9]
            # head_mat[:3, 4] = cur_motion[0+3:3+3]
            # left_wrist[:3, 4] = cur_motion[3+3:6+3]
            # right_wrist[:3, 4] = cur_motion[6+3:9+3]

            # self.head_mat[:]

        else:
            self.head_pos[:] = torch.tensor([ -0.0500,  0.0000,  1.2523])
            self.left_wrist_pos[:] = torch.tensor([0.05, 0.3457,  0.9])
            self.right_wrist_pos[:] = torch.tensor([0.05,  -0.3457,  0.9])

            self.head_vel[:] = 0.
            self.left_wrist_vel[:] = 0.
            self.right_wrist_vel[:] = 0.

            # self.head_rot[:] = torch.tensor([0., 0., 1., 0])
            # self.left_hand_rot[:] = torch.tensor([0., 0., 1., 0])
            # self.right_hand_rot[:] = torch.tensor([0., 0., 1., 0])

            self.head_rot[:] = torch.tensor([0., 0., 0., 1.])
            self.left_hand_rot[:] = torch.tensor([0., 0., 0., 1.])
            self.right_hand_rot[:] = torch.tensor([0., 0., 0., 1.])


    def launch_vision_pro(self, launch_image_client=True):
        # image and television
        if VISION_WRAPPER_BACKEND == 'vuer':
            img_shape = (480, 640, 3)
            self.img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.img_shm.buf)
            if launch_image_client:
                self.img_client = ImageClient(img_shape, self.img_shm.name, server_address='192.168.123.164')
                self.realsense_process = Process(target=self.img_client.receive_process)
                self.realsense_process.start()
            img_shm_name = self.img_shm.name
        else:
            img_shape = None
            img_shm_name = None

        self.tv_wrapper = VisionWrapper(backend=VISION_WRAPPER_BACKEND, avp_ip=VISION_PRO_IP,
                                        binocular=False,
                                        img_shape=img_shape,
                                        img_shm_name=img_shm_name)

    def compute_task_observation(self) -> np.ndarray:
        quat_xyzw = torch.from_numpy(self.obs_quat.copy())[None, :]
        # wxyz -> xyzw
        # w = quat[:, 0].clone()
        # quat[:, :3] = quat[:, 1:].clone()
        # quat[:, 3] = w
        
        ref_body_pos = torch.cat((self.left_wrist_pos, self.right_wrist_pos, self.head_pos)).reshape(1, 1, 3, 3).float()
        ref_body_vel = torch.cat((self.left_wrist_vel, self.right_wrist_vel, self.head_vel)).reshape(1, 1, 3, 3).float()
        
        body_pos_extend = self.body_pos_extend.clone()
        body_pos_extend[..., 2] = body_pos_extend[..., 2] - torch.min(body_pos_extend[..., 2], dim=0, keepdim=True).values
        body_pos = body_pos_extend[-3:]
        # ref_body_pos -= ref_body_pos[:, :, 2:3].clone()

        if CLIP_GOAL:
            ref_pos_rel = ref_body_pos - ref_body_pos[..., 2:3, :]
            head_delta = self.head_pos[None, :] - self.body_pos_extend[0:1, :]

            delta_len = torch.norm(head_delta[..., :2], dim=-1, keepdim=True)
            if delta_len[0, 0].item() > CLIP_GOAL_DIST:
                head_delta[..., :2] = head_delta[..., :2] / delta_len * CLIP_GOAL_DIST
                ref_body_pos = (self.body_pos_extend[0:1, :] + head_delta + ref_pos_rel[0, 0]).reshape(1, 1, 3, 3)

        time_steps = 1
        J = 3
        B = 1
        obs = []

        #############################
        heading_inv_rot = calc_heading_quat_inv(quat_xyzw)#xyzw
        heading_rot = calc_heading_quat(quat_xyzw)#xyzw
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, 3, 1)).repeat_interleave(
            time_steps, 0)

        # ##### Body position and rotation differences
        diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
        diff_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4),
                                                            diff_global_body_pos.view(-1, 3))  #
        # print('global', diff_local_body_pos_flat)
        
        wrist_rots = torch.cat((self.left_hand_rot, self.right_hand_rot))
        local_ref_body_rot = quat_mul(heading_inv_rot.view(B, 1, 4).repeat(1, 2, 1), wrist_rots.view(B, 2, 4))

        ##### body pos + Dof_pos This part will have proper futuers.
        # local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, time_steps, 1, 3)
        # local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - self.body_pos_extend[0].view(B, time_steps, 1, 3)
        ref_head_xy = ref_body_pos.view(B, time_steps, J, 1, 3)[:, :, -1].clone()
        ref_head_xy[..., 2] = 0.6
        local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - ref_head_xy  # locomotion
        local_ref_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
        # print('local', local_ref_body_pos)

        ##### body vel
        local_ref_body_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))

        # make some changes to how futures are appended.
        obs.append(diff_local_body_pos_flat.view(B, time_steps, -1) * ZERO_GLOBAL)  # 1 * timestep * J * 3
        # obs.append(torch.zeros(B, time_steps, 9))  # 1 * timestep * J * 3
        obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * J * 3
        obs.append(local_ref_body_vel.view(B, time_steps, -1))  # timestep  * J * 3
        obs.append(local_ref_body_rot.view(B, time_steps, -1))

        obs = torch.cat(obs, dim=-1).view(-1)
        return obs.numpy()
    
    def update_mujoco(self):
        ### mujoco viewer
        self.env.mj_data.qpos[0:3] = self.body_pos_extend[0, :]
        self.env.mj_data.qpos[3:7] = np.roll(self.obs_quat[:], shift=1, axis=-1)
        self.env.mj_data.qpos[7:] = self.obs_joint_pos     # (29, )
        # self.env.mj_data.qpos[7:] = self.angles
        # print(self.env.mj_data.qpos[3:7])
        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
        head_pos_rel = self.head_pos.numpy()
        lwrist_pos_rel = self.left_wrist_pos.numpy()
        rwrist_pos_rel = self.right_wrist_pos.numpy()

        for i, p in enumerate([lwrist_pos_rel, rwrist_pos_rel, head_pos_rel]):
            self.env.viewer.user_scn.geoms[i].pos = p
        self.env.viewer.user_scn.geoms[3].pos = self.location
        self.env.viewer.user_scn.geoms[3].pos[2] = self.body_pos_extend[-1, 2]
        # self.env.viewer.user_scn.geoms[0].mat = torch.from_numpy(left_wrist[:3, :3])
        # self.env.viewer.user_scn.geoms[1].mat = torch.from_numpy(right_wrist[:3, :3])
        self.env.viewer.sync()
        # print(f'Head:{self.head_pos}, LHand:{self.left_wrist_vel}, RHAND:{self.right_wrist_vel}')
            
    def compute_observations(self):
        """ Computes observations
        """
        task_obs = self.compute_task_observation()

        # quat_xyzw = torch.from_numpy(self.obs_quat.copy())[None, :]
        # ref_body_rot = torch.zeros(1, 33, 4)
        # ref_body_rot[0, 22] = self.left_hand_rot
        # ref_body_rot[0, 29] = self.right_hand_rot
        # task_obs_lr = compute_imitation_observations_teleop_max(None, quat_xyzw, self.body_pos_extend[None, -3:].clone(),
        #                                                              torch.cat((self.left_wrist_pos, self.right_wrist_pos, self.head_pos)).reshape(1, 1, 3, 3).float(),
        #                                                              torch.cat((self.left_wrist_vel, self.right_wrist_vel, self.head_vel)).reshape(1, 1, 3, 3).float(),
        #                                                              1,
        #                                                              ref_episodic_offset=None,
        #                                                              compute_diff=True, compute_rot=True,
        #                                                              ref_body_rot=ref_body_rot)
        # print(task_obs_lr - task_obs)

        history_to_be_append = self.env.trajectories[0: self.env.obs_context_len * LOG_STEP_LEN].copy()
        obs_buf = torch.tensor(np.concatenate((
                            self.obs_joint_pos, 
                            self.obs_joint_vel * OBS_VEL_SCALE, 
                            self.obs_ang_vel,
                            # self.obs_imu[:2],
                            self.projected_gravity,
                            task_obs,
                            self.prev_action, 
                            history_to_be_append
                            ), axis=-1).copy(), dtype=torch.float, device=self.device).unsqueeze(0)
        # add perceptive inputs if not blind
        self.env.obs_buf = obs_buf

    @torch.inference_mode()
    def main_loop(self):
        # keep stand up pose first
        _percent_1 = 0
        _duration_1 = 500
        firstRun = True
        init_success = False

        global COLLECT_SID_DATA
        temp_sid = COLLECT_SID_DATA
        COLLECT_SID_DATA = False
        while self.stand_up and not self.start_policy:
        # while True:
            if firstRun:
                firstRun = False
                rclpy.spin_once(self)
                start_pos = self.joint_pos
                self.reset_localization()
            else:
                self.set_gains(kp=self.env.p_gains, kd=self.env.d_gains)
                if _percent_1 < 1:
                    self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(self.env.default_dof_pos_np))
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                if _percent_1 == 1 and not init_success:
                    init_success = True
                    print("---Initialized---")
                if not NO_MOTOR:
                    self.motor_pub.publish(self.cmd_msg)
            rclpy.spin_once(self)
            self.update_mujoco()
            if USE_DEX:
                self.dex_shm_send_data[164] = -1.
            # print('############# Gravity #############')
            # print(self.projected_gravity)
        
        COLLECT_SID_DATA = temp_sid
        if USE_DEX:
            self.dex_shm_send_data[164] = 1.
        
        cnt = 0
        fps_ckt = time.monotonic()

        self.get_logger().info("start main loop")
        self.reset_localization()
        rclpy.spin_once(self)
        
        loop_start_time = time.monotonic()
        while rclpy.ok() or NO_ROS:
            mujoco_step_start = time.time()
            if self.Emergency_stop:
                self.stop = True
                # breakpoint()
            if self.stop:
                _percent_1 = 0
                _duration_1 = 1000
                start_pos = self.joint_pos
                while _percent_1 < 1:
                    self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(start_pos))
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                    if not NO_MOTOR:
                        self.motor_pub.publish(self.cmd_msg)
                if COLLECT_MOTION_DATA:
                    now = datetime.datetime.now()
                    torch.save(self.motion_data, f'./rec_motion/motion_data_{now.strftime("%m_%d_%H_%M_%S")}.pkl')
                    print('Motion Data Saved')
                self.get_logger().info("Program exit")
                if COLLECT_SID_DATA:
                    import pickle
                    now = datetime.datetime.now()
                    with open(f'./sys_id/sid_data_{now.strftime("%m_%d_%H_%M_%S")}.pkl', 'wb') as f:
                        pickle.dump(self.sid_data, f)
                    print('SID Data Saved')
                break
                    
            # print(self.head_pos)
            # spin stuff
            while self.control_dt > time.monotonic() - loop_start_time:  #0.012473  0.019963
                # rclpy.spin_once(self, timeout_sec=0.0001)
                time.sleep(max(0., self.control_dt - (time.monotonic() - loop_start_time) - 0.001))
                pass
            loop_start_time = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.001)

            # XXX special order, check legged gym
            self.compute_observations()
            raw_actions = self.policy(self.env.obs_buf) 
            # raw_actions = torch.clip(raw_actions, -6.28, 6.28)
        
            if torch.any(torch.isnan(raw_actions)):
                self.get_logger().info("Emergency stop due to NaN")
                self.set_motor_position(q=self.env.default_dof_pos_np)
                raise SystemExit
            
            raw_actions_np = raw_actions.clone().detach().cpu().numpy().squeeze(0)
            angles = raw_actions_np * self.env.scale_action + self.env.default_dof_pos_np
            # angles[self.over_limit] = np.clip(angles[self.over_limit], self.env.joint_limit_lo_[self.over_limit], self.env.joint_limit_hi_[self.over_limit])
            angles = np.clip(angles, self.env.joint_limit_lo_, self.env.joint_limit_hi_)

            # angles[:15] *= 0.

            self.angles = angles
            self.set_motor_position(self.angles)
            if not NO_MOTOR and not NO_ROS:
                self.motor_pub.publish(self.cmd_msg)

                cnt += 1
                if cnt == 10:
                    dt = (time.monotonic()-fps_ckt)/cnt
                    cnt = 0
                    fps_ckt = time.monotonic()
                    print(f"POL FREQ: {1/dt}")

                    if 1/dt < self.motor_pub_freq:
                        self.control_dt -= 2e-5
                    else:
                        self.control_dt += 2e-5
            
            obs_buf = self.env.obs_buf.clone().cpu()
            dof = obs_buf[0, :HW_DOF]
            dof_vel = obs_buf[0, HW_DOF:2 * HW_DOF]
            base_ang_vel = obs_buf[0, 2 * HW_DOF:2 * HW_DOF + 3]
            base_gravity = obs_buf[0, 2 * HW_DOF + 3:2 * HW_DOF + 6]
            task_obs = obs_buf[0, 2 * HW_DOF + 6:2 * HW_DOF + 6 + 3 * 3 * 3 + 2 * 4]
            self.env.record_trajectory(dof, dof_vel, base_ang_vel, base_gravity, torch.from_numpy(self.prev_action), task_obs)
            # self.prev_action = (self.real_q - self.env.default_dof_pos_np) / self.env.scale_action
            self.prev_action = raw_actions_np

            if self.control_dt > time.monotonic() - loop_start_time + 0.005:
               self.update_mujoco()

            if self.recording:
                cur_motion = torch.cat([
                        self.head_pos - self.rec_delta_pos,
                        self.left_wrist_pos - self.rec_delta_pos,
                        self.right_wrist_pos - self.rec_delta_pos,
                        self.head_vel,
                        self.left_wrist_vel,
                        self.right_wrist_vel,
                        self.left_hand_rot,
                        self.right_hand_rot
                    ]).float()
                if USE_DEX:
                    cur_motion = torch.cat([cur_motion, torch.from_numpy(self.dex_shm_recv_data[:14]).float().clone()])
                cur_motion = cur_motion.clone()
                self.motion_data['seq'].append(cur_motion)
            
            # if PLAY_MOTION_DATA:
            #     if self.playing:
            #         self.motion_idx = min(len(self.motion_buffer['seq']) - 1, self.motion_idx + 1)
            #         if USE_DEX:
            #             self.dex_shm_send_data[164] = -1.
            #             self.dex_shm_send_data[150:164] = self.motion_buffer['seq'][self.motion_idx][-14:].numpy()
            #     else:
            #         self.dex_shm_send_data[164] = 1.

            # self.send_plot()
    
    def send_plot(self):
        # action plot
        action_data_buffer = self.angles.tobytes()
        action_data_buffer = self.angles.tobytes()
        # self.sock.sendto(action_data_buffer, self.server_address)
        self.sock_plot.sendto(action_data_buffer, self.server_address_plot)
        self.sock_plot.sendto(action_data_buffer, self.server_address_plot)
            
    def vp_loop(self):
        last_time = time.monotonic()
        while True:
            try:
                self.get_vp_data()
                if time.monotonic() - last_time < self.vp_dt:
                    time.sleep(self.vp_dt - (time.monotonic() - last_time))
            except:
                import traceback; traceback.print_exc()
                return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name: stand, stand_w_waist, wb, squat', required=False, default='stand')
    args = parser.parse_args()
    
    rclpy.init(args=None)
    dp_node = DeployNode(args.task_name)
    dp_node.prepare()

    dp_node.get_logger().info("Deploy node started")
    
    thread = Thread(target=dp_node.vp_loop)
    thread.start()

    dp_node.main_loop()
    dp_node.destroy_node()
    rclpy.shutdown()
