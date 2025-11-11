import zmq
import socket
import time
from collections import deque
import struct
import json
import pickle
import numpy as np
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


if __name__ == "__main__":
    config = {
        "port": 6006,
        "server_ip": "192.168.123.164",
    }
    pos_client = Position_Client(config)  # deployment test
    pos_client.receive_process()
