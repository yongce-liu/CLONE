import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        img_shape: [height, width]
        serial_number: serial number
        enable_depth: whether to enable depth stream
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.init_realsense()

    def init_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(str(self.serial_number))
        config.enable_stream(
            rs.stream.color,
            self.img_shape[1],
            self.img_shape[0],
            rs.format.bgr8,
            self.fps,
        )

        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self.img_shape[1],
                self.img_shape[0],
                rs.format.z16,
                self.fps,
            )

        # Start pipeline without hardware reset (it was causing issues)
        profile = self.pipeline.start(config)

        self._device = profile.get_device()
        if self._device is None:
            print("[Image Server] pipe_profile.get_device() is None")
        if self.enable_depth and self._device is not None:
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames) if self.enable_depth else frames

            color_frame = aligned_frames.get_color_frame()
            depth_frame = (
                aligned_frames.get_depth_frame() if self.enable_depth else None
            )

            if not color_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            return color_image, depth_image
        except Exception as e:
            print(f"[RealSenseCamera] Error getting frame: {e}")
            return None, None

    def release(self):
        self.pipeline.stop()


class OpenCVCamera:
    def __init__(self, device_id, img_shape, fps):
        """
        device_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {device_id}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Test if the camera can read frames
        if not self._can_read_frame():
            print(
                f"[Image Server] Camera {self.id} Error: Failed to initialize the camera or read frames."
            )
            self.release()
            raise RuntimeError(f"Camera {device_id} initialization failed")

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def get_frame(self):
        ret, color_image = self.cap.read()
        if not ret:
            return None, None  # Return consistent format with RealSenseCamera
        return color_image, None


class ImageServer:
    def __init__(self, config, port=5555, Unit_Test=False):
        """
        config:
        {
            'fps':60,                                                       # frame per second
            'head_camera_type': 'realsense',                                 # opencv or realsense
            'head_camera_image_shape': [480, 640],                           # Head camera resolution [height, width]
            'head_camera_id_numbers': [0],                                   # '/dev/video0' (opencv)
            'wrist_camera_type': 'realsense',
            'wrist_camera_image_shape': [480, 640],                          # Wrist camera resolution [height, width]
            'wrist_camera_id_numbers': ["218622271789", "241222076627"],     # serial number (realsense)
        }
        """
        print(config)
        self.fps = config.get("fps", 60)
        self.head_camera_type = config.get("head_camera_type", "realsense")
        self.head_image_shape = config.get("head_camera_image_shape", [480, 640])
        self.head_camera_id_numbers = config.get("head_camera_id_numbers", [0])

        self.wrist_camera_type = config.get("wrist_camera_type", None)
        self.wrist_image_shape = config.get("wrist_camera_image_shape", [480, 640])
        self.wrist_camera_id_numbers = config.get("wrist_camera_id_numbers", None)

        self.port = port
        self.Unit_Test = Unit_Test

        # Initialize cameras
        self.head_cameras = self._initialize_cameras(
            self.head_camera_type,
            self.head_camera_id_numbers,
            self.head_image_shape,
            enable_depth=True,  # Enable depth for head cameras
        )

        self.wrist_cameras = self._initialize_cameras(
            self.wrist_camera_type,
            self.wrist_camera_id_numbers,
            self.wrist_image_shape,
            enable_depth=False,  # Disable depth for wrist cameras
        )

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        self._print_camera_info()

        print(
            "[Image Server] Image server has started, waiting for client connections..."
        )

    def _initialize_cameras(
        self, camera_type, id_numbers, img_shape, enable_depth=False
    ):
        cameras = []
        if not camera_type or not id_numbers:
            return cameras

        if camera_type == "opencv":
            for device_id in id_numbers:
                try:
                    camera = OpenCVCamera(
                        device_id=device_id, img_shape=img_shape, fps=self.fps
                    )
                    cameras.append(camera)
                except Exception as e:
                    print(
                        f"[Image Server] Failed to initialize OpenCV camera {device_id}: {e}"
                    )
        elif camera_type == "realsense":
            for serial_number in id_numbers:
                try:
                    camera = RealSenseCamera(
                        img_shape=img_shape,
                        fps=self.fps,
                        serial_number=serial_number,
                        enable_depth=enable_depth,
                    )
                    cameras.append(camera)
                except Exception as e:
                    print(
                        f"[Image Server] Failed to initialize RealSense camera {serial_number}: {e}"
                    )
        else:
            print(f"[Image Server] Unsupported camera_type: {camera_type}")

        return cameras

    def _print_camera_info(self):
        for i, cam in enumerate(self.head_cameras):
            if isinstance(cam, OpenCVCamera):
                print(
                    f"[Image Server] Head camera {i} (OpenCV) resolution: {cam.img_shape[0]}x{cam.img_shape[1]}"
                )
            elif isinstance(cam, RealSenseCamera):
                print(
                    f"[Image Server] Head camera {i} (RealSense {cam.serial_number}) resolution: {cam.img_shape[0]}x{cam.img_shape[1]}, Depth: {'Enabled' if cam.enable_depth else 'Disabled'}"
                )

        for i, cam in enumerate(self.wrist_cameras):
            if isinstance(cam, OpenCVCamera):
                print(
                    f"[Image Server] Wrist camera {i} (OpenCV) resolution: {cam.img_shape[0]}x{cam.img_shape[1]}"
                )
            elif isinstance(cam, RealSenseCamera):
                print(
                    f"[Image Server] Wrist camera {i} (RealSense {cam.serial_number}) resolution: {cam.img_shape[0]}x{cam.img_shape[1]}"
                )

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.time_window = 1.0
        self.frame_times = deque()
        self.start_time = time.time()

    def _update_performance_metrics(self, current_time):
        self.frame_times.append(current_time)
        while (
            self.frame_times and self.frame_times[0] < current_time - self.time_window
        ):
            self.frame_times.popleft()
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(
                f"[Image Server] FPS: {real_time_fps:.2f}, Frames: {self.frame_count}, Time: {elapsed_time:.2f}s"
            )

    def _close(self):
        for cam in self.head_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def _process_depth_image(self, depth_image):
        """Convert depth image to colormap for visualization"""
        if depth_image is None:
            return None

        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        return depth_colormap

    def send_process(self):
        try:
            while True:
                # Get frames from all cameras
                head_color_frames = []
                head_depth_frames = []
                wrist_color_frames = []

                # Process head cameras
                for cam in self.head_cameras:
                    color_image, depth_image = cam.get_frame()
                    if color_image is None:
                        print("[Image Server] Head camera frame read error")
                        break

                    head_color_frames.append(color_image)
                    if depth_image is not None:
                        head_depth_frames.append(self._process_depth_image(depth_image))

                if len(head_color_frames) != len(self.head_cameras):
                    print("[Image Server] Head image num mismatch")
                    time.sleep(0.1)
                    continue

                # Process wrist cameras if available
                if self.wrist_cameras:
                    for cam in self.wrist_cameras:
                        color_image, _ = cam.get_frame()
                        if color_image is None:
                            print("[Image Server] Wrist camera frame read error")
                            break
                        wrist_color_frames.append(color_image)

                    if len(wrist_color_frames) != len(self.wrist_cameras):
                        print("[Image Server] Wrist image num mismatch")
                        time.sleep(0.1)
                        continue

                # Concatenate frames
                head_color = (
                    cv2.hconcat(head_color_frames)
                    if len(head_color_frames) > 1
                    else head_color_frames[0]
                )

                if self.wrist_cameras:
                    wrist_color = (
                        cv2.hconcat(wrist_color_frames)
                        if len(wrist_color_frames) > 1
                        else wrist_color_frames[0]
                    )
                    full_color = cv2.hconcat([head_color, wrist_color])
                else:
                    full_color = head_color

                # Process depth images if available
                full_depth = None
                if head_depth_frames:
                    head_depth = (
                        cv2.hconcat(head_depth_frames)
                        if len(head_depth_frames) > 1
                        else head_depth_frames[0]
                    )
                    full_depth = head_depth

                # Encode and send frames
                ret, color_buffer = cv2.imencode(".jpg", full_color)
                if not ret:
                    print("[Image Server] Color frame encode failed")
                    continue

                color_bytes = color_buffer.tobytes()
                depth_bytes = b""

                if full_depth is not None:
                    ret, depth_buffer = cv2.imencode(".png", full_depth)
                    if ret:
                        depth_bytes = depth_buffer.tobytes()

                # Prepare message
                timestamp = time.time()

                if self.Unit_Test:
                    frame_id = self.frame_count
                    header = struct.pack(
                        "!dI", timestamp, frame_id
                    )  # 8-byte double, 4-byte unsigned int
                else:
                    header = b""

                # Send message: [header][color_len][depth_len][color_data][depth_data]
                color_len = len(color_bytes)
                depth_len = len(depth_bytes)
                len_header = struct.pack("!dII", timestamp, color_len, depth_len)
                message = header + len_header + color_bytes + depth_bytes

                self.socket.send(message)

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        except Exception as e:
            print(f"[Image Server] Error: {str(e)}")
        finally:
            self._close()


if __name__ == "__main__":
    config = {
        "fps": 60,
        "head_camera_type": "realsense",
        "head_camera_image_shape": [480, 640],
        "head_camera_id_numbers": [
            "213622077703"
        ],  # Replace with your actual serial number
    }

    server = ImageServer(config, Unit_Test=False)
    server.send_process()
