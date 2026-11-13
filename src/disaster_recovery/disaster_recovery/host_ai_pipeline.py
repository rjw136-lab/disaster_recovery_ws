#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: host_ai_pipeline.py
Author: Kyungpook National University / Jong Woo Ro

Description:
  Host PC High-Performance AI and Supervision Station Node.
  Runs on Windows 11 WSL2 Ubuntu 22.04 with NVIDIA GeForce GTX 1050 Ti (CUDA 12.x).

Key Functions:
  1. 10 Hz Watchdog Heartbeat Generator:
     - Regularly emits /heartbeat (std_msgs/Header) to assure the robot of active network connection.
  2. GPU-Accelerated Vision Pipeline (Ultralytics YOLO26n):
     - Ingests /camera/image_raw/compressed (default, JPEG) or /camera/image_raw from TurtleBot3 Burger.
     - Zero-lag "latest frame only" worker thread: stale frames are dropped instead of queued.
     - Rotates frames by `rotation_angle` (default 270 deg) because the Pi camera is mounted sideways.
     - Detects survivors ('person', COCO ID 0) with a person-specific threshold and an
       N-consecutive-frame debounce filter to suppress single-frame false positives.
     - Publishes visualized bounding box stream to /detection/image (+ optional OpenCV window).
     - Emits /disaster/victim_alert when a survivor is confirmed.
  3. Real-Time Audio STT Pipeline (Faster-Whisper):
     - Consumes raw PCM (/audio/raw, std_msgs/UInt8MultiArray, 48 kHz S16_LE mono) from the
       onboard USB microphone (audio_capture_node).
     - 2 s sliding window (50% overlap) -> 3:1 decimation to 16 kHz -> RMS voice-activity gate.
     - Executes Faster-Whisper (base, compute_type="int8") in a worker thread (Korean).
     - Performs keyword spotting for distress calls ("도와주세요", "살려주세요", "구조", "help", "here").
     - Emits /disaster/audio_alert upon distress detection.
=================================================================================================
"""

import threading
import queue
import time
import numpy as np
import cv2
try:
    from cv_bridge import CvBridge, CvBridgeError
    CV_BRIDGE_AVAILABLE = True
except (ImportError, AttributeError):
    CV_BRIDGE_AVAILABLE = False
    CvBridge = None
    CvBridgeError = Exception

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header, String, Bool, Float32, UInt8MultiArray
from sensor_msgs.msg import Image, CompressedImage

# Import AI Libraries with graceful CPU fallbacks
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class HostAIPipelineNode(Node):
    def __init__(self):
        super().__init__('host_ai_pipeline')

        # -------------------------------------------------------------------------
        # Parameters Configuration
        # -------------------------------------------------------------------------
        self.declare_parameter('heartbeat_rate_hz', 10.0)
        self.declare_parameter('enable_heartbeat', True)
        self.declare_parameter('yolo_model_path', 'yolo26n.pt')
        self.declare_parameter('whisper_model_size', 'base')
        self.declare_parameter('whisper_device', 'cpu')
        self.declare_parameter('whisper_compute_type', 'int8')
        self.declare_parameter('whisper_cpu_threads', 4)
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('person_conf_threshold', 0.55)
        self.declare_parameter('only_person_detection', True)
        self.declare_parameter('min_consecutive_detections', 2)
        self.declare_parameter('cuda_device', 0)
        self.declare_parameter('sample_rate_hz', 16000)
        self.declare_parameter('audio_raw_topic', '/audio/raw')
        self.declare_parameter('audio_sample_rate', 48000)
        self.declare_parameter('voice_activity_threshold', 0.012)
        self.declare_parameter('audio_buffer_sec', 2.0)
        self.declare_parameter('image_topic', '/camera/image_raw/compressed')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('show_window', True)
        self.declare_parameter('rotation_angle', 270)

        self.hb_rate = self.get_parameter('heartbeat_rate_hz').value
        self.enable_heartbeat = self.get_parameter('enable_heartbeat').value
        self.model_path = self.get_parameter('yolo_model_path').value
        self.whisper_size = self.get_parameter('whisper_model_size').value
        self.whisper_device = self.get_parameter('whisper_device').value
        self.whisper_compute_type = self.get_parameter('whisper_compute_type').value
        self.whisper_cpu_threads = self.get_parameter('whisper_cpu_threads').value
        self.conf_th = self.get_parameter('confidence_threshold').value
        self.person_conf_th = self.get_parameter('person_conf_threshold').value
        self.only_person = self.get_parameter('only_person_detection').value
        self.min_consecutive = self.get_parameter('min_consecutive_detections').value
        self.device_id = self.get_parameter('cuda_device').value
        self.sample_rate = self.get_parameter('sample_rate_hz').value
        self.audio_raw_topic = self.get_parameter('audio_raw_topic').value
        self.mic_sample_rate = self.get_parameter('audio_sample_rate').value
        self.vad_threshold = self.get_parameter('voice_activity_threshold').value
        self.audio_buffer_sec = self.get_parameter('audio_buffer_sec').value
        self.image_topic = self.get_parameter('image_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.show_window = self.get_parameter('show_window').value
        self.rotation_angle = self.get_parameter('rotation_angle').value

        self.bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        self.seq_id = 0

        # Debounce counter for person detection
        self.consecutive_person_count = 0
        self.fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0

        # Audio PCM streaming buffer & lock
        self.audio_pcm_buffer = bytearray()
        self.audio_lock = threading.Lock()

        # Latest-frame zero-lag buffer
        self.latest_frame = None
        self.latest_compressed_bytes = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.vision_thread_running = True

        # GUI display buffer & thread synchronization (ensures main-thread rendering for WSLg)
        self.display_frame = None
        self.display_lock = threading.Lock()

        # Distress keywords dictionary for emergency spotter
        self.distress_keywords = [
            "도와주세요", "살려주세요", "여기요", "사람있어요", "구조",
            "help", "save me", "here", "emergency"
        ]

        # -------------------------------------------------------------------------
        # Device & Model Initialization (GTX 1050 Ti VRAM Management)
        # -------------------------------------------------------------------------
        self.device_str = f'cuda:{self.device_id}' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        self.get_logger().info(f"Target Compute Device: {self.device_str}")

        # 1. Vision Model (YOLO)
        self.yolo_model = None
        if TORCH_AVAILABLE:
            try:
                self.yolo_model = YOLO(self.model_path)
                self.yolo_model.to(self.device_str)
                self.get_logger().info(f"Ultralytics YOLO model loaded on {self.device_str} successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")

        # 2. Audio Model (Faster-Whisper)
        self.whisper_model = None
        self.audio_queue = queue.Queue(maxsize=100)
        if WHISPER_AVAILABLE:
            # CPU INT8 acceleration (AVX2 on i7-8700) is prioritized to completely preserve 
            # the 4GB VRAM on GTX 1050 Ti for YOLO26n and RViz2.
            try:
                self.whisper_model = WhisperModel(
                    self.whisper_size,
                    device=self.whisper_device,
                    compute_type=self.whisper_compute_type,
                    cpu_threads=self.whisper_cpu_threads
                )
                self.get_logger().info(
                    f"Faster-Whisper ({self.whisper_size}, {self.whisper_compute_type}) initialized "
                    f"successfully on {self.whisper_device} ({self.whisper_cpu_threads} CPU threads, VRAM protected)."
                )
            except Exception as e:
                self.get_logger().warn(f"Faster-Whisper on {self.whisper_device} failed: {e}. Falling back to CPU int8.")
                try:
                    self.whisper_model = WhisperModel(self.whisper_size, device="cpu", compute_type="int8", cpu_threads=4)
                    self.get_logger().info("Faster-Whisper CPU default fallback initialized.")
                except Exception as ex:
                    self.get_logger().error(f"Whisper completely unavailable: {ex}")

        # -------------------------------------------------------------------------
        # Subscriptions & Publishers
        # -------------------------------------------------------------------------
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        if self.use_compressed or 'compressed' in self.image_topic:
            self.sub_image = self.create_subscription(
                CompressedImage, self.image_topic, self.cb_compressed_image, qos_best_effort
            )
            self.get_logger().info(f"Subscribing to CompressedImage (Zero-Lag Best Effort): {self.image_topic}")
        else:
            self.sub_image = self.create_subscription(
                Image, self.image_topic, self.cb_image, qos_best_effort
            )
            self.get_logger().info(f"Subscribing to Image (Zero-Lag Best Effort): {self.image_topic}")

        # Raw audio PCM byte stream from Raspberry Pi USB Microphone
        self.sub_audio_raw = self.create_subscription(
            UInt8MultiArray, self.audio_raw_topic, self.cb_audio_raw, 10
        )
        self.get_logger().info(f"Subscribing to Raw Audio: {self.audio_raw_topic}")

        # Legacy / fallback text transcription stream
        self.sub_audio = self.create_subscription(
            String, '/audio/transcription_raw', self.cb_audio_text, qos_reliable
        )

        self.pub_heartbeat = self.create_publisher(Header, '/heartbeat', 10)
        self.sub_enable_hb = self.create_subscription(Bool, '/host/enable_heartbeat', self._cb_enable_heartbeat, 10)
        self.pub_detection_img = self.create_publisher(Image, '/detection/image', 1)
        self.pub_victim_alert = self.create_publisher(String, '/disaster/victim_alert', 10)
        self.pub_audio_alert = self.create_publisher(String, '/disaster/audio_alert', 10)

        # Heartbeat Generation Timer (10 Hz = 0.10s)
        hb_period = 1.0 / self.hb_rate
        self.timer_heartbeat = self.create_timer(hb_period, self.publish_heartbeat)

        # Worker thread for real-time vision inference (zero queue lag)
        self.vision_thread = threading.Thread(target=self.vision_worker_loop, daemon=True)
        self.vision_thread.start()

        # Worker thread for Whisper STT
        self.audio_thread_running = True
        self.audio_thread = threading.Thread(target=self.whisper_worker_loop, daemon=True)
        self.audio_thread.start()

        # GUI Window Render Timer on the Main ROS 2 Executor Thread (Prevents WSLg black window)
        if self.show_window:
            self.timer_gui = self.create_timer(0.033, self.render_gui_window)

        self.get_logger().info("Host AI Pipeline Operational. Transmitting 10Hz Heartbeat.")

    # -----------------------------------------------------------------------------
    # Watchdog Heartbeat Generator
    # -----------------------------------------------------------------------------
    def _cb_enable_heartbeat(self, msg: Bool):
        self.enable_heartbeat = msg.data

    def publish_heartbeat(self):
        if not self.enable_heartbeat:
            return
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = f"host_station_seq_{self.seq_id}"
        self.seq_id += 1
        self.pub_heartbeat.publish(msg)

    # -----------------------------------------------------------------------------
    # Vision Inference Callbacks & Processing (Zero-Lag Worker)
    # -----------------------------------------------------------------------------
    def rotate_image(self, img):
        if self.rotation_angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def cb_image(self, msg: Image):
        cv_img = None
        if self.bridge is not None:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception:
                cv_img = None

        if cv_img is None:
            try:
                # Robust pure-NumPy fallback (immune to OpenCV/NumPy version conflicts)
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                if msg.encoding in ('rgb8', 'bgr8'):
                    cv_img = img_array.reshape((msg.height, msg.width, 3))
                    if msg.encoding == 'rgb8':
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                elif msg.encoding == 'mono8':
                    cv_img = img_array.reshape((msg.height, msg.width))
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                self.get_logger().error(f"Image conversion error: {e}")
                return

        with self.frame_lock:
            self.latest_frame = cv_img
        self.new_frame_event.set()

    def cb_compressed_image(self, msg: CompressedImage):
        with self.frame_lock:
            self.latest_compressed_bytes = msg.data
        self.new_frame_event.set()

    def vision_worker_loop(self):
        """Dedicated thread to process only the latest frame, preventing network/queue lag."""
        last_infer_t = 0.0
        infer_interval = 0.10  # 10 FPS inference throttle: reduces CPU/GPU utilization by ~65%

        while self.vision_thread_running:
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()

            with self.frame_lock:
                raw_bytes = self.latest_compressed_bytes
                self.latest_compressed_bytes = None
                frame = self.latest_frame
                self.latest_frame = None

            if raw_bytes is not None:
                try:
                    np_arr = np.frombuffer(raw_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    self.get_logger().error(f"Compressed image decode error: {e}")
                    continue

            if frame is None:
                continue

            # 1. Rotate to upright orientation (e.g. 270 degrees)
            frame = self.rotate_image(frame)

            # Throttle heavy YOLO inference rate to preserve CPU without overwriting HUD
            now = time.time()
            if now - last_infer_t < infer_interval:
                continue
            last_infer_t = now

            # 2. Process YOLO and publish
            self.process_frame(frame)

    def process_frame(self, cv_img):
        if self.yolo_model is None:
            return

        # Perform inference on GTX 1050 Ti
        # If only_person is enabled, restrict YOLO detection strictly to class 0 ('person')
        inference_kwargs = {
            'conf': self.conf_th,
            'device': self.device_str,
            'verbose': False
        }
        if self.only_person:
            inference_kwargs['classes'] = [0]

        results = self.yolo_model(cv_img, **inference_kwargs)

        annotated_frame = cv_img.copy()
        person_detected = False
        highest_conf = 0.0

        for r in results:
            annotated_frame = r.plot()
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.yolo_model.names.get(cls_id, '')

                # Only trigger if the detected class is strictly 'person' (COCO ID 0)
                # and confidence meets the human-specific threshold (default 0.55)
                if (cls_id == 0 or cls_name.lower() == 'person') and conf >= self.person_conf_th:
                    person_detected = True
                    if conf > highest_conf:
                        highest_conf = conf

        # Debounce: Consecutive frame filter to prevent single-frame flickering false positives
        if person_detected:
            self.consecutive_person_count += 1
        else:
            self.consecutive_person_count = 0

        # Only announce and publish SURVIVOR alert when confirmed over consecutive frames
        if self.consecutive_person_count >= self.min_consecutive:
            alert = String()
            alert.data = f"SURVIVOR_DETECTED|Confidence:{highest_conf:.2f}|Time:{time.time()}"
            self.pub_victim_alert.publish(alert)
            cv2.putText(
                annotated_frame, f"!!! SURVIVOR LOCATED ({highest_conf*100:.0f}%) !!!", (30, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2
            )

        # Real-time FPS Calculation & Status Overlay
        self.fps_counter += 1
        now = time.time()
        dt = now - self.fps_time
        if dt >= 1.0:
            self.current_fps = self.fps_counter / dt
            self.fps_counter = 0
            self.fps_time = now

        h_img, w_img = annotated_frame.shape[:2]
        font_scale = max(0.45, min(0.70, w_img / 820.0))
        bar_h = max(28, int(34 * (w_img / 640.0)))
        cv2.rectangle(annotated_frame, (0, 0), (w_img, bar_h), (15, 15, 15), -1)
        hb_str = "HB:ON" if self.enable_heartbeat else "HB:CUT"
        cv2.putText(
            annotated_frame,
            f"FPS:{self.current_fps:.1f} | YOLO26n | {hb_str} | Person:{self.consecutive_person_count}",
            (10, int(bar_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0) if self.enable_heartbeat else (0, 120, 255),
            2,
        )

        # Publish visualized inference stream for RViz2 / Monitoring UI
        try:
            if self.bridge is not None:
                out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            else:
                out_msg = Image()
                out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.frame_id = 'camera_link'
                out_msg.height = annotated_frame.shape[0]
                out_msg.width = annotated_frame.shape[1]
                out_msg.encoding = 'bgr8'
                out_msg.is_bigendian = 0
                out_msg.step = annotated_frame.shape[1] * 3
                out_msg.data = annotated_frame.tobytes()
            self.pub_detection_img.publish(out_msg)
        except Exception:
            pass

        # Dedicated Standalone AI Station Window buffer update
        if self.show_window:
            with self.display_lock:
                self.display_frame = annotated_frame

    def render_gui_window(self):
        """Render GUI frame from the main thread at ~30Hz, fully compatible with WSLg/X11."""
        with self.display_lock:
            frame = self.display_frame
        if frame is not None:
            win_name = "Disaster Recovery - YOLO26n Vision AI Station"
            if not hasattr(self, '_window_initialized'):
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 960, 720)
                self._window_initialized = True
            cv2.imshow(win_name, frame)
            cv2.waitKey(1)

    # -----------------------------------------------------------------------------
    # Audio STT Pipeline & Keyword Spotter
    # -----------------------------------------------------------------------------
    def cb_audio_raw(self, msg: UInt8MultiArray):
        """Receive 48kHz S16_LE PCM raw audio byte chunk from Raspberry Pi USB mic."""
        chunk_bytes = bytes(msg.data)
        if not chunk_bytes:
            return

        with self.audio_lock:
            self.audio_pcm_buffer.extend(chunk_bytes)
            # 48000 Hz, 16-bit Mono (2 bytes/sample) = 96,000 bytes/sec
            target_bytes = int(self.mic_sample_rate * 2 * self.audio_buffer_sec)

            if len(self.audio_pcm_buffer) >= target_bytes:
                pcm_window = bytes(self.audio_pcm_buffer[:target_bytes])
                # 50% overlap sliding window to prevent clipping words
                step = target_bytes // 2
                self.audio_pcm_buffer = self.audio_pcm_buffer[step:]

                # Parse signed 16-bit PCM
                audio_int16 = np.frombuffer(pcm_window, dtype=np.int16)

                # Resample: 48,000 Hz -> 16,000 Hz (exact 3:1 decimation)
                downsample_factor = max(1, self.mic_sample_rate // self.sample_rate)
                audio_16k = audio_int16[::downsample_factor]

                # Normalize to float32 [-1.0, 1.0] for Faster-Whisper
                audio_float = audio_16k.astype(np.float32) / 32768.0

                # Voice Activity Detection based on RMS amplitude
                rms = float(np.sqrt(np.mean(audio_float ** 2)))
                if rms >= self.vad_threshold:
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio_float)

    def cb_audio_text(self, msg: String):
        """Handle pre-transcribed text or direct audio text input."""
        text = msg.data.lower()
        self.check_distress_keywords(text)

    def whisper_worker_loop(self):
        """Worker thread for Faster-Whisper audio batch inference."""
        while self.audio_thread_running:
            try:
                audio_np = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.whisper_model is None:
                continue

            try:
                # Transcribe 16kHz float32 audio
                segments, info = self.whisper_model.transcribe(
                    audio_np,
                    beam_size=3,
                    language="ko",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                transcript = " ".join([segment.text for segment in segments]).strip()
                if transcript:
                    self.get_logger().info(f"[Audio STT] Detected Speech: '{transcript}'")
                    self.check_distress_keywords(transcript)
            except Exception as e:
                self.get_logger().error(f"Whisper inference error: {e}")

    def check_distress_keywords(self, transcript: str):
        transcript_lower = transcript.lower()
        matched = []
        for kw in self.distress_keywords:
            if kw in transcript_lower:
                matched.append(kw)

        if matched:
            alert_msg = String()
            alert_msg.data = f"DISTRESS_CALL_ALERT|Keywords:{','.join(matched)}|Transcript:{transcript}"
            self.pub_audio_alert.publish(alert_msg)
            self.get_logger().warn(f">>> [EMERGENCY AUDIO ALERT] Matched: {matched} in '{transcript}' <<<")

    def destroy_node(self):
        self.vision_thread_running = False
        self.audio_thread_running = False
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HostAIPipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Host AI Pipeline shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
