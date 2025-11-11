from enum import IntEnum
import time
import numpy as np
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from multiprocessing import Process, shared_memory

import rclpy
import rclpy.executors
from rclpy.node import Node
from unitree_hg.msg import (
    MotorCmd,
    HandState,
    HandCmd,
)

unitree_tip_indices = [4, 9, 14]  # [thumb, index, middle] in OpenXR
Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "dex3/left/cmd"
kTopicDex3RightCommand = "dex3/right/cmd"
kTopicDex3LeftState = "dex3/left/state"
kTopicDex3RightState = "dex3/right/state"


class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6


class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


class _RIS_Mode:
    def __init__(self, id=0, status=0x01, timeout=0):
        self.motor_mode = 0
        self.id = id & 0x0F  # 4 bits for id
        self.status = status & 0x07  # 3 bits for status
        self.timeout = timeout & 0x01  # 1 bit for timeout

    def _mode_to_uint8(self):
        self.motor_mode |= self.id & 0x0F
        self.motor_mode |= (self.status & 0x07) << 4
        self.motor_mode |= (self.timeout & 0x01) << 7
        return self.motor_mode


class DexNode(Node):
    def __init__(self):
        super().__init__("dex_node")  # type: ignore
        self.left_hand_sub = self.create_subscription(
            HandState, "dex3/left/state", self.left_dex_hand_cb, 1
        )
        self.right_hand_sub = self.create_subscription(
            HandState, "dex3/right/state", self.right_dex_hand_cb, 1
        )

        self.left_hand_pos = np.zeros(7)
        self.right_hand_pos = np.zeros(7)
        self.left_hand_temp = np.zeros(7)
        self.right_hand_temp = np.zeros(7)

        self.left_hand_cmd = HandCmd(motor_cmd=[MotorCmd() for _ in range(7)])
        self.right_hand_cmd = HandCmd(motor_cmd=[MotorCmd() for _ in range(7)])

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        for id in Dex3_1_Left_JointIndex:
            ris_mode = _RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_hand_cmd.motor_cmd[id].mode = motor_mode
            self.left_hand_cmd.motor_cmd[id].q = q
            self.left_hand_cmd.motor_cmd[id].dq = dq
            self.left_hand_cmd.motor_cmd[id].tau = tau
            self.left_hand_cmd.motor_cmd[id].kp = kp
            self.left_hand_cmd.motor_cmd[id].kd = kd

        # initialize dex3-1's right hand cmd msg
        for id in Dex3_1_Right_JointIndex:
            ris_mode = _RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_hand_cmd.motor_cmd[id].mode = motor_mode
            self.right_hand_cmd.motor_cmd[id].q = q
            self.right_hand_cmd.motor_cmd[id].dq = dq
            self.right_hand_cmd.motor_cmd[id].tau = tau
            self.right_hand_cmd.motor_cmd[id].kp = kp
            self.right_hand_cmd.motor_cmd[id].kd = kd

        self.dex_limit_left_lo = np.array(
            [
                -1.04719755,
                -0.72431163,
                0.0,
                -1.57079632,
                -1.74532925,
                -1.57079632,
                -1.74532925,
            ]
        )
        self.dex_limit_left_hi = np.array(
            [1.04719755, 1.04719755, 1.74532925, 0, 0, 0, 0]
        )
        self.dex_limit_right_lo = np.array(
            [-1.04719755, -1.04719755, -1.74532925, 0, 0, 0, 0]
        )
        self.dex_limit_right_hi = np.array(
            [1.04719755, 0.72431163, 0, 1.57079632, 1.74532925, 1.57079632, 1.74532925]
        )

        self.dex_effort = np.array([2.45, 1.40, 1.40, 1.40, 1.40, 1.40, 1.40])
        self.max_dex_angles = self.dex_effort / kp
        self.max_dex_left_angle_factor = np.ones_like(self.max_dex_angles)
        self.max_dex_right_angle_factor = np.ones_like(self.max_dex_angles)

        self.left_dex_pub = self.create_publisher(HandCmd, "dex3/left/cmd", 1)
        self.right_dex_pub = self.create_publisher(HandCmd, "dex3/right/cmd", 1)

        self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)

    def left_dex_hand_cb(self, msg: HandState):
        self.left_hand_pos[:] = np.array([msg.motor_state[i].q for i in range(7)])
        self.left_hand_temp[:] = np.array(
            [msg.motor_state[i].temperature[1] for i in range(7)]
        )
        self.max_dex_left_angle_factor[
            (self.left_hand_temp > 80) & (self.left_hand_temp <= 100)
        ] = 0.5
        self.max_dex_left_angle_factor[(self.left_hand_temp > 100)] = 0.0
        self.max_dex_left_angle_factor[(self.left_hand_temp <= 80)] = 1.0

    def right_dex_hand_cb(self, msg: HandState):
        self.right_hand_pos[:] = np.array([msg.motor_state[i].q for i in range(7)])
        self.right_hand_temp[:] = np.array(
            [msg.motor_state[i].temperature[1] for i in range(7)]
        )
        self.max_dex_right_angle_factor[
            (self.right_hand_temp > 80) & (self.right_hand_temp <= 100)
        ] = 0.5
        self.max_dex_right_angle_factor[(self.right_hand_temp > 100)] = 0.0
        self.max_dex_right_angle_factor[(self.right_hand_temp <= 80)] = 1.0

    def control_dex_hands(self, target_q, factor=1):
        left_q = target_q[0:7] * factor
        right_q = target_q[7:14] * factor
        left_q = np.clip(
            left_q.astype(self.left_hand_pos.dtype),
            self.left_hand_pos - self.max_dex_angles * self.max_dex_left_angle_factor,
            self.left_hand_pos + self.max_dex_angles * self.max_dex_left_angle_factor,
        )
        right_q = np.clip(
            right_q.astype(self.right_hand_pos.dtype),
            self.right_hand_pos - self.max_dex_angles * self.max_dex_right_angle_factor,
            self.right_hand_pos + self.max_dex_angles * self.max_dex_right_angle_factor,
        )

        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_hand_cmd.motor_cmd[id].q = float(
                left_q[idx]
            )  # np.clip(left_q[idx], self.dex_limit_left_lo[idx], self.dex_limit_left_hi[idx])
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_hand_cmd.motor_cmd[id].q = float(
                right_q[idx]
            )  # np.clip(right_q[idx], self.dex_limit_right_lo[idx], self.dex_limit_right_hi[idx])
        self.left_dex_pub.publish(self.left_hand_cmd)
        self.right_dex_pub.publish(self.right_hand_cmd)
        # print('aaa')

    def dex_retarget(self, send_shm_name, recv_shm_name):
        send_shm = shared_memory.SharedMemory(name=send_shm_name, size=165 * 4)
        recv_shm = shared_memory.SharedMemory(name=recv_shm_name, size=14 * 4)
        send_shm_data = np.ndarray((165,), np.float32, send_shm.buf)
        recv_shm_data = np.ndarray((14,), np.float32, recv_shm.buf)
        fix_pos_buf = np.zeros((14,), dtype=np.float32)
        tgt_hand_pos = np.zeros((14,), dtype=np.float32)

        print("DEX SERVER INITIALIZED")
        try:
            while rclpy.ok():
                start_time = time.time()
                # get dual hand state
                rclpy.spin_once(self, timeout_sec=0.001)
                left_hand_mat = send_shm_data[0:75].reshape(25, 3).copy()
                right_hand_mat = send_shm_data[75:150].reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                manual_data = send_shm_data[150:164]
                if not np.all(
                    left_hand_mat == 0.0
                ):  # if hand data has been initialized.
                    ref_left_value = left_hand_mat[unitree_tip_indices]
                    ref_right_value = right_hand_mat[unitree_tip_indices]
                    ref_left_value[0] = ref_left_value[0] * 1.15
                    ref_left_value[1] = ref_left_value[1] * 1.05
                    ref_left_value[2] = ref_left_value[2] * 0.95
                    ref_right_value[0] = ref_right_value[0] * 1.15
                    ref_right_value[1] = ref_right_value[1] * 1.05
                    ref_right_value[2] = ref_right_value[2] * 0.95
                    left_q_target = self.hand_retargeting.left_retargeting.retarget(
                        ref_left_value
                    )[[0, 1, 2, 3, 4, 5, 6]]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(
                        ref_right_value
                    )[[0, 1, 2, 3, 4, 5, 6]]

                    # get dual hand action
                    tgt_hand_pos[:] = np.concatenate(
                        (left_q_target, right_q_target)
                    ).astype(np.float32)
                    if send_shm_data[164] < 1.5:
                        fix_pos_buf[:] = tgt_hand_pos[:]

                if send_shm_data[164] >= 0 and send_shm_data[164] < 1.5:
                    self.control_dex_hands(tgt_hand_pos[:14])
                elif send_shm_data[164] >= 1.5:
                    self.control_dex_hands(fix_pos_buf[:14])
                else:
                    self.control_dex_hands(manual_data)

                recv_shm_data[:7] = self.left_hand_pos[:]
                recv_shm_data[7:] = self.right_hand_pos[:]

                # if time.monotonic() - start_time < 0.02:
                #     time.sleep(max(0.02 - (time.monotonic() - start_time), 0.))
        except:
            print("DEX SERVER DOWN")
            import traceback

            traceback.print_exc()

        finally:
            print("Dex3_1_Controller has been closed.")


def start_dex_server(send_shm_name, recv_shm_name):
    rclpy.init(args=None)

    node = DexNode()
    node.dex_retarget(send_shm_name, recv_shm_name)
