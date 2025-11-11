import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory


class ImageClient:
    def __init__(
        self,
        img_shape=None,
        img_shm_name=None,
        depth_img_shm_name=None,
        pad_shape=None,
        pad_shm_name=None,
        timestamp_shm_name=None,
        image_show=False,
        depth_show=False,
        server_address="192.168.123.164",
        port=5555,
        Unit_Test=False,
    ):
        """
        Initialize the image client with shared memory support for both color and depth images.

        Parameters:
        img_shape: Expected shape of color images (H, W, C)
        img_shm_name: Shared memory name for color images
        depth_img_shm_name: Shared memory name for depth images
        image_show: Whether to display received color images
        depth_show: Whether to display received depth images
        server_address: IP address of the image server
        port: Port number to connect to
        Unit_Test: Enable performance metrics collection
        """
        self.running = True
        self._image_show = image_show
        self._depth_show = depth_show
        self._server_address = server_address
        self._port = port
        # Initialize shared memory for color image
        self.color_shm = None
        self.color_array = None
        self.img_shape = img_shape
        self.pad_shape = pad_shape
        if img_shm_name is not None:
            try:
                self.color_shm = shared_memory.SharedMemory(name=img_shm_name)
                if img_shape is not None:
                    self.color_array = np.ndarray(
                        img_shape, dtype=np.uint8, buffer=self.color_shm.buf
                    )
                else:
                    print(
                        "[Image Client] Warning: img_shape not provided for color shared memory"
                    )
            except FileNotFoundError:
                print(
                    f"[Image Client] Shared memory {img_shm_name} not found for color image"
                )

        if pad_shm_name is not None:
            try:
                self.pad_shm = shared_memory.SharedMemory(name=pad_shm_name)
                if pad_shape is not None:
                    self.pad_array = np.ndarray(
                        pad_shape, dtype=np.uint8, buffer=self.pad_shm.buf
                    )
                else:
                    print(
                        "[Image Client] Warning: pad_shape not provided for pad shared memory"
                    )
            except FileNotFoundError:
                print(
                    f"[Image Client] Shared memory {pad_shm_name} not found for pad image"
                )

        # Initialize shared memory for depth image
        self.depth_shm = None
        self.depth_array = None
        if depth_img_shm_name is not None:
            try:
                self.depth_shm = shared_memory.SharedMemory(name=depth_img_shm_name)
                # Depth images are typically single channel (H, W)
                if img_shape is not None:
                    depth_shape = (
                        img_shape[0],
                        img_shape[1],
                    )  # Remove channel dimension
                    self.depth_array = np.ndarray(
                        depth_shape, dtype=np.uint16, buffer=self.depth_shm.buf
                    )
                else:
                    print(
                        "[Image Client] Warning: img_shape not provided for depth shared memory"
                    )
            except FileNotFoundError:
                print(
                    f"[Image Client] Shared memory {depth_img_shm_name} not found for depth image"
                )

        self.timestamp_shm = None
        self.timestamp_array = None
        if timestamp_shm_name is not None:
            try:
                self.timestamp_shm = shared_memory.SharedMemory(name=timestamp_shm_name)
                self.timestamp_array = np.ndarray(
                    (1,), dtype=np.float64, buffer=self.timestamp_shm.buf
                )
            except FileNotFoundError:
                print(
                    f"[Image Client] Shared memory {timestamp_shm_name} not found for timestamp"
                )
        # Performance evaluation setup
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

        # Depth visualization parameters
        self._min_depth = 0.1  # meters
        self._max_depth = 10.0  # meters

    def _init_performance_metrics(self):
        """Initialize performance tracking metrics"""
        self._frame_count = 0
        self._last_frame_id = -1
        self._time_window = 1.0
        self._frame_times = deque()
        self._latencies = deque()
        self._lost_frames = 0
        self._total_frames = 0

    def _process_depth_image(self, depth_image):
        """Convert depth image to colormap for visualization"""
        if depth_image is None:
            return None

        # Clip and normalize depth values
        depth_image = np.clip(depth_image, self._min_depth, self._max_depth)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        return depth_colormap

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        """Update performance metrics with new frame data"""
        latency = receive_time - timestamp
        self._latencies.append(latency)
        while (
            self._latencies
            and self._frame_times
            and self._latencies[0] < receive_time - self._time_window
        ):
            self._latencies.popleft()

        self._frame_times.append(receive_time)
        while (
            self._frame_times
            and self._frame_times[0] < receive_time - self._time_window
        ):
            self._frame_times.popleft()

        expected_frame_id = (
            self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        )
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(
                    f"[Image Client] Lost frames: {lost}, Expected: {expected_frame_id}, Received: {frame_id}"
                )

        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1
        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        """Print current performance metrics"""
        if self._frame_count % 30 == 0:
            real_time_fps = (
                len(self._frame_times) / self._time_window
                if self._time_window > 0
                else 0
            )

            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            lost_frame_rate = (
                (self._lost_frames / self._total_frames) * 100
                if self._total_frames > 0
                else 0
            )

            print(
                f"[Image Client] FPS: {real_time_fps:.2f}, Latency: {avg_latency*1000:.2f}ms, "
                f"Jitter: {jitter*1000:.2f}ms, Lost: {lost_frame_rate:.2f}%"
            )

    def _close(self):
        """Clean up resources"""
        if hasattr(self, "_socket"):
            self._socket.close()
        if hasattr(self, "_context"):
            self._context.term()
        if self._image_show or self._depth_show:
            cv2.destroyAllWindows()

        # Close shared memory handles
        if self.color_shm:
            self.color_shm.close()
        if self.depth_shm:
            self.depth_shm.close()

        print("Image client has been closed.")

    def receive_process(self):
        """Main receive loop for getting images from server"""
        self._context = zmq.Context()
        # subscribe the topic published
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print("\nImage client started, waiting for data...")
        try:
            while self.running:
                message = self._socket.recv()
                receive_time = time.time()

                # Parse message header
                pos = 0
                if self._enable_performance_eval:
                    header_size = struct.calcsize("dI")
                    header = message[:header_size]
                    timestamp, frame_id = struct.unpack("dI", header)
                    pos = header_size

                # # Parse length header
                # len_header_size = struct.calcsize('!II')
                # len_header = message[pos:pos+len_header_size]
                # color_len, depth_len = struct.unpack('!II', len_header)
                # pos += len_header_size

                len_header_size = struct.calcsize("!dII")
                len_header = message[pos : pos + len_header_size]
                timestamp, color_len, depth_len = struct.unpack("!dII", len_header)
                pos += len_header_size
                # print(f'timestamp: {timestamp}')
                # print(f'color_len: {color_len}')
                # print(f'depth_len: {depth_len}')
                # Extract and decode color image
                color_image = None
                if color_len > 0:
                    color_data = message[pos : pos + color_len]
                    pos += color_len
                    try:
                        color_image = cv2.imdecode(
                            np.frombuffer(color_data, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        # Update color shared memory
                        if self.color_array is not None and color_image is not None:
                            try:
                                np.copyto(self.color_array, color_image)
                                if self.pad_array is not None:
                                    self.pad_array.fill(0)
                                    # Calculate padding position (center)
                                    pad_top = (
                                        self.pad_shape[0] - self.img_shape[0]
                                    ) // 2
                                    pad_left = (
                                        self.pad_shape[1] - self.img_shape[1]
                                    ) // 2
                                    self.pad_array[
                                        pad_top : pad_top + self.img_shape[0],
                                        pad_left : pad_left + self.img_shape[1],
                                    ] = color_image
                            except Exception as e:
                                print(f"[Image Client] Color image copy error: {e}")
                    except Exception as e:
                        print(f"[Image Client] Color image decode error: {e}")

                # Extract and decode depth image
                depth_image = None
                depth_colormap = None
                if depth_len > 0:
                    depth_data = message[pos : pos + depth_len]
                    try:
                        # Decode as grayscale (actual depth data)
                        depth_image = cv2.imdecode(
                            np.frombuffer(depth_data, dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE,
                        )
                        # Update depth shared memory
                        if self.depth_array is not None and depth_image is not None:
                            try:
                                np.copyto(self.depth_array, depth_image)
                            except Exception as e:
                                print(f"[Image Client] Depth image copy error: {e}")
                        # Create colormap for visualization
                        if self._depth_show:
                            depth_colormap = self._process_depth_image(depth_image)
                    except Exception as e:
                        print(f"[Image Client] Depth image decode error: {e}")

                if self.timestamp_array is not None and timestamp is not None:
                    try:
                        np.copyto(self.timestamp_array, timestamp)
                    except Exception as e:
                        print(f"[Image Client] Timestamp copy error: {e}")
                # Display images if configured
                if self._image_show or self._depth_show:
                    display_width = 640  # Fixed display width

                    if self._image_show and color_image is not None:
                        color_display = cv2.resize(
                            color_image,
                            (
                                display_width,
                                int(
                                    display_width
                                    * color_image.shape[0]
                                    / color_image.shape[1]
                                ),
                            ),
                        )
                        cv2.imshow("Color Stream", color_display)

                    if self._depth_show and depth_colormap is not None:
                        depth_display = cv2.resize(
                            depth_colormap,
                            (
                                display_width,
                                int(
                                    display_width
                                    * depth_colormap.shape[0]
                                    / depth_colormap.shape[1]
                                ),
                            ),
                        )
                        cv2.imshow("Depth Stream", depth_display)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord("q"):
                        self.running = False
                    elif key & 0xFF == ord(" "):  # Space bar to pause
                        while True:
                            key = cv2.waitKey(0)
                            if key & 0xFF == ord(" "):  # Space bar to resume
                                break
                            elif key & 0xFF == ord("q"):
                                self.running = False
                                break

                # Update performance metrics if enabled
                if self._enable_performance_eval and color_image is not None:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] Error in receive_process: {e}")
        finally:
            self._close()


if __name__ == "__main__":
    # Example usage:
    # Create shared memory blocks first in your main script
    # color_shm = shared_memory.SharedMemory(create=True, size=480*640*3)
    # depth_shm = shared_memory.SharedMemory(create=True, size=480*640*2)  # uint16 for depth

    # Then create client
    client = ImageClient(
        img_shape=(480, 640, 3),  # Expected color image shape
        img_shm_name="color_shm_name",  # Replace with your color shm name
        depth_img_shm_name="depth_shm_name",  # Replace with your depth shm name
        image_show=True,
        depth_show=True,
        server_address="192.168.123.164",
        Unit_Test=False,
    )
    client.receive_process()
