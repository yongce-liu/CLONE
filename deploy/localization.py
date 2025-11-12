import os
import argparse
from teleop.crc import CRC

import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from unitree_hg.msg import (
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)
import time
from collections import deque
from multiprocessing import Process, shared_memory, Array

from teleop.local2word import fk_dof
from multiprocessing import Process, shared_memory
from teleop.image_server.image_client import ImageClient
from teleop.gamepad import Gamepad, parse_remote_data
import transforms3d as t3d
from teleop.utils.mat_tool import rotate_quaternion
from teleop.torch_utils import *
from pytorch_kinematics import matrix_to_quaternion
from threading import Thread

import pytorch_kinematics as pk


import zmq
import socket
import struct
import json
import pickle
from teleop.torch_utils import *


class Position_Client:
    def __init__(self, config: dict = None, Unit_Test=False):
        self.server_ip = config["server_ip"]
        self.port = config["port"]
        self.running = True

        self.ma_len = 10
        self.position_queue = []

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.position_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.position_factor = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.delta_quat = torch.from_numpy(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )

        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self.server_ip}:{self.port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def _close(self):
        self._socket.close()
        self._context.term()
        # self.ClientSocket.close()
        print("Position client has been closed.")

    def receive_process(self):
        print("\nPosition client has started, waiting to receive data...")
        try:
            while self.running:
                # import pdb; pdb.set_trace()
                # Receive message
                message = pickle.loads(self._socket.recv(1024))
                # message = self._socket.recv_string()
                # print("Received: {}".format(message))
                # receive_time = time.time()

                # if self._enable_performance_eval:
                #     header_size = struct.calcsize('dI')
                #     try:
                #         # Attempt to extract header and position data
                #         header = message[:header_size]
                #         pos_bytes = message[header_size:]
                #         timestamp, frame_id = struct.unpack('dI', header)
                #     except struct.error as e:
                #         print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                #         continue
                # else:
                #     # No header, entire message is image data
                #     pos_bytes = message
                # Decode image
                if message is None:
                    print("[Position Client] Failed to decode Position.")
                    continue
                else:
                    position, quat = message
                    self.position_queue.append(position)
                    if len(self.position_queue) > self.ma_len:
                        self.position_queue = self.position_queue[self.ma_len :]
                    position = sum(self.position_queue) / len(self.position_queue)

                    self.position[:] = (
                        quat_rotate(
                            self.delta_quat.unsqueeze(0),
                            torch.from_numpy(position * self.position_factor).unsqueeze(
                                0
                            ),
                        )
                        + self.position_offset
                    )
                    self.quat[:] = quat

                # if self._enable_performance_eval:
                #     self._update_performance_metrics(timestamp, frame_id, receive_time)
                #     self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Position client interrupted by user.")
        except Exception as e:
            print(f"[Position Client] An error occurred while receiving data: {e}")
        finally:
            self._close()


def compute_fk_body_pos(joint_pos: np.ndarray):
    extend_body_parent_ids = [22, 29, 15]
    extend_body_pos = torch.tensor([[0.06, 0, 0], [0.06, 0, 0], [0, 0, 0.4]])

    body_pos, body_quat = fk_dof(torch.from_numpy(joint_pos))
    body_quat = body_quat.roll(shifts=-1, dims=-1)

    extend_curr_pos = (
        my_quat_rotate(
            body_quat[extend_body_parent_ids].reshape(-1, 4),
            extend_body_pos.reshape(-1, 3),
        ).view(-1, 3)
        + body_pos[extend_body_parent_ids]
    )
    body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=0)
    body_quat_extend = torch.cat(
        [
            body_quat,
            body_quat[extend_body_parent_ids[0] : extend_body_parent_ids[0] + 1, :],
            body_quat[extend_body_parent_ids[1] : extend_body_parent_ids[1] + 1, :],
            body_quat[extend_body_parent_ids[2] : extend_body_parent_ids[2] + 1, :],
        ]
    )
    return body_pos_extend, body_quat_extend


def start_service(send_name, recv_name, vp_rotate_name):
    HW_DOF = 29
    shm_send_shape = (3 + 4 + 4 + HW_DOF,)
    shm_recv_shape = (3 + 4 + (HW_DOF + 1 + 3) * 3,)
    vp_shape = (4,)
    send_shm = shared_memory.SharedMemory(name=send_name, size=shm_send_shape[0] * 4)
    recv_shm = shared_memory.SharedMemory(name=recv_name, size=shm_recv_shape[0] * 4)
    vp_shm = shared_memory.SharedMemory(name=vp_rotate_name, size=vp_shape[0] * 4)
    vp_shm = shared_memory.SharedMemory(name=vp_rotate_name, size=vp_shape[0] * 4)
    send_shm_data = np.ndarray(shm_send_shape, dtype=np.float32, buffer=send_shm.buf)
    recv_shm_data = np.ndarray(shm_recv_shape, dtype=np.float32, buffer=recv_shm.buf)
    vp_data = np.ndarray(vp_shape, dtype=np.float32, buffer=vp_shm.buf)
    vp_data = np.ndarray(vp_shape, dtype=np.float32, buffer=vp_shm.buf)

    config = {
        "port": 60060,
        "server_ip": "192.168.123.164",
    }
    pos_client = Position_Client(config)
    pos_client.position = recv_shm_data[0:3]
    pos_client.position_offset = send_shm_data[0:3]
    pos_client.position_factor[2] *= -1
    pos_client.position_factor[1] *= -1

    send_shm_data[7:11] = pos_client.delta_quat[:].numpy()
    pos_client.delta_quat = torch.from_numpy(send_shm_data[7:11])

    recv_shm_data[3:7] = pos_client.quat[:]
    pos_client.quat = recv_shm_data[3:7]

    location = torch.from_numpy(pos_client.position).unsqueeze(0)

    thread = Thread(target=pos_client.receive_process)
    thread.start()

    quat = torch.from_numpy(send_shm_data[3:7])
    dof_pos = send_shm_data[11:]
    body_pos_extend_buf = torch.zeros(((HW_DOF + 1 + 3) * 3), dtype=torch.float32)
    body_pos_extend = torch.from_numpy(recv_shm_data[7:])

    while True:
        try:
            body_pos_extend_buf, body_quat_extend_buf = compute_fk_body_pos(
                dof_pos.copy()
            )
            # print(body_pos_extend_buf)
            num_bodies = body_pos_extend_buf.size(0)
            # vp_data = torch.from_numpy(np.array([vp_data[0], -vp_data[1], -vp_data[1], -vp_data[3]])) #wijk
            # quat_wijk = quat.clone()        # quat ijxw
            # quat_wijk = torch.roll(quat_wijk, 1, 0)
            # rot_quat = torch_utils.rotation_between_quaternions(vp_data, quat_wijk)
            # rot_quat[:] = torch.roll(rot_quat, -1, 0)
            # body_pos_extend_buf = torch_utils.my_quat_rotate(quat[None, :].clone().repeat(num_bodies, 1), body_pos_extend_buf)
            body_pos_extend_buf = my_quat_rotate(
                quat[None, :].clone().repeat(num_bodies, 1), body_pos_extend_buf
            )
            body_pos_extend_buf[:] = (
                body_pos_extend_buf[:]
                + location
                - (body_pos_extend_buf[-1, None] - body_pos_extend_buf[0:1])
            )
            body_pos_extend_buf[:, 2] = body_pos_extend_buf[:, 2] - torch.min(
                body_pos_extend_buf[:, 2]
            )
            body_pos_extend[:] = body_pos_extend_buf.view(-1)

        except KeyboardInterrupt:
            return
        except:
            print("ERRROOR")
            import traceback

            traceback.print_exc()
            return


def start_service_denoise(send_name, recv_name, vp_rotate_name):
    HW_DOF = 29
    shm_send_shape = (3 + 4 + 4 + HW_DOF,)
    shm_recv_shape = (3 + 4 + (HW_DOF + 1 + 3) * 3,)
    vp_shape = (4,)
    send_shm = shared_memory.SharedMemory(name=send_name, size=shm_send_shape[0])
    recv_shm = shared_memory.SharedMemory(name=recv_name, size=shm_recv_shape[0])
    vp_shm = shared_memory.SharedMemory(name=vp_rotate_name, size=vp_shape[0])
    vp_shm = shared_memory.SharedMemory(name=vp_rotate_name, size=vp_shape[0])
    send_shm_data = np.ndarray(shm_send_shape, dtype=np.float32, buffer=send_shm.buf)
    recv_shm_data = np.ndarray(shm_recv_shape, dtype=np.float32, buffer=recv_shm.buf)
    vp_data = np.ndarray(vp_shape, dtype=np.float32, buffer=vp_shm.buf)
    vp_data = np.ndarray(vp_shape, dtype=np.float32, buffer=vp_shm.buf)

    config = {
        "port": 60060,
        "server_ip": "192.168.123.164",
    }
    pos_client = Position_Client(config)
    pos_client.ma_len = 3
    pos_client.position = recv_shm_data[0:3]
    pos_client.position_offset = send_shm_data[0:3]
    pos_client.position_factor[2] *= -1
    pos_client.position_factor[1] *= -1

    send_shm_data[7:11] = pos_client.delta_quat[:].numpy()
    pos_client.delta_quat = torch.from_numpy(send_shm_data[7:11])

    recv_shm_data[3:7] = pos_client.quat[:]
    pos_client.quat = recv_shm_data[3:7]

    location = torch.from_numpy(pos_client.position).unsqueeze(0)

    thread = Thread(target=pos_client.receive_process)
    thread.start()

    quat = torch.from_numpy(send_shm_data[3:7])
    dof_pos = send_shm_data[11:]
    dof_pos_torch = torch.from_numpy(dof_pos)

    body_pos_extend_buf = torch.zeros(((HW_DOF + 1 + 3) * 3), dtype=torch.float32)
    body_pos_extend = torch.from_numpy(recv_shm_data[7:])

    denoise_step_size = 3 + HW_DOF + 4
    denoise_his_len = 50
    location_seq = torch.zeros(
        denoise_step_size * 128, device="cpu", dtype=torch.float32
    )
    denoise_model = torch.jit.load(
        "legged_gym/logs/g1:teleop/g1_slam_denoiser/slam_denoiser_jit.pt"
    )

    delta_t = 0.02
    target_fq = 50

    cnt = 0
    time_cnt = time.monotonic()
    time_start = time.monotonic()
    while True:
        try:
            body_pos_extend_buf, body_quat_extend_buf = compute_fk_body_pos(
                dof_pos.copy()
            )
            num_bodies = body_pos_extend_buf.size(0)
            body_pos_extend_buf = my_quat_rotate(
                quat[None, :].clone().repeat(num_bodies, 1), body_pos_extend_buf
            )

            location_seq[denoise_step_size:] = location_seq[:-denoise_step_size].clone()
            location_obs = location.clone()[0] + (
                body_pos_extend_buf[0] - body_pos_extend_buf[-1]
            )
            location_obs[..., 2] = 0.0
            location_obs = torch.cat(
                (location_obs, dof_pos_torch.clone(), quat.clone())
            )
            location_seq[:denoise_step_size] = location_obs
            location_seq_de = denoise_model(
                location_seq[None, : denoise_step_size * denoise_his_len].clone()
            ).cpu()[:, :3]

            body_pos_extend_buf[:] = body_pos_extend_buf[:] + location_seq_de
            body_pos_extend_buf[:, 2] = body_pos_extend_buf[:, 2] - torch.min(
                body_pos_extend_buf[:, 2]
            )
            body_pos_extend[:] = body_pos_extend_buf.view(-1)

            cnt += 1
            if cnt % 30 == 0:
                mean_freq = cnt / (time.monotonic() - time_cnt)
                if mean_freq > target_fq:
                    delta_t += 2e-4
                else:
                    delta_t -= 2e-4
                print("DEN FREQ:", mean_freq)
                cnt = 0
                time_cnt = time.monotonic()

            if time.monotonic() - time_start < delta_t:
                pass
            time_start = time.monotonic()

        except KeyboardInterrupt:
            return
        except:
            print("ERRROOR")
            import traceback

            traceback.print_exc()
            return


if __name__ == "__main__":
    config = {
        "port": 6006,
        "server_ip": "192.168.123.164",
    }
    pos_client = Position_Client(config)  # deployment test
    pos_client.receive_process()
