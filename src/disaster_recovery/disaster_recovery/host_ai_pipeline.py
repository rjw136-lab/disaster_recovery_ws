#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: host_ai_pipeline.py
Author: Autonomous Systems Laboratory / Modified for Jongwoo Noh

Description:
  Host PC High-Performance AI and Supervision Station Node.
  Runs on Windows 11 WSL2 Ubuntu 22.04 with NVIDIA GeForce GTX 1050 Ti (CUDA 12.x).

Key Functions:
  1. 10 Hz Watchdog Heartbeat Generator:
     - Regularly emits /heartbeat (std_msgs/Header) to assure the robot of active network connection.
  2. GPU-Accelerated Vision Pipeline (Ultralytics YOLO):
     - Ingests /camera/image_raw stream from TurtleBot3 Burger.
     - Runs YOLOv8n / YOLO26n (FP16 half-precision) on GTX 1050 Ti (device='cuda:0').
     - Detects survivors ('person', COCO ID 0) and hazardous elements.
     - Publishes visualized bounding box stream to /detection/image.
     - Emits /disaster/victim_alert when confidence exceeds threshold.
  3. Real-Time Audio STT Pipeline (Faster-Whisper):
     - Consumes audio stream (/audio/data) captured by the onboard USB microphone.
     - Executes Faster-Whisper (base / tiny with compute_type="float16" or "int8_float16") in a worker thread.
     - Performs automatic keyword spotting for distress calls ("도와주세요", "살려주세요", "구조", "help", "here").
     - Emits /disaster/audio_alert upon distress detection.
=================================================================================================
"""

import threading
import queue
import time
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header, String, Bool, Float32
from sensor_msgs.msg import Image

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
        self.declare_parameter('yolo_model_path', 'yolov8n.pt')
        self.declare_parameter('whisper_model_size', 'base')
        self.declare_parameter('confidence_threshold', 0.50)
        self.declare_parameter('cuda_device', 0)
        self.declare_parameter('sample_rate_hz', 16000)

        self.hb_rate = self.get_parameter('heartbeat_rate_hz').value
        self.model_path = self.get_parameter('yolo_model_path').value
        self.whisper_size = self.get_parameter('whisper_model_size').value
        self.conf_th = self.get_parameter('confidence_threshold').value
        self.device_id = self.get_parameter('cuda_device').value
        self.sample_rate = self.get_parameter('sample_rate_hz').value

        self.bridge = CvBridge()
        self.seq_id = 0

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
            try:
                compute_type = "float16" if "cuda" in self.device_str else "int8"
                self.whisper_model = WhisperModel(
                    self.whisper_size,
                    device="cuda" if "cuda" in self.device_str else "cpu",
                    compute_type=compute_type
                )
                self.get_logger().info(f"Faster-Whisper ({self.whisper_size}, {compute_type}) initialized.")
            except Exception as e:
                self.get_logger().warn(f"Faster-Whisper CUDA init issue: {e}. Falling back to CPU int8.")
                try:
                    self.whisper_model = WhisperModel(self.whisper_size, device="cpu", compute_type="int8")
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

        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.cb_image, qos_best_effort
        )
        # Raw mono audio byte stream
        self.sub_audio = self.create_subscription(
            String, '/audio/transcription_raw', self.cb_audio_text, qos_reliable
        )

        self.pub_heartbeat = self.create_publisher(Header, '/heartbeat', 10)
        self.pub_detection_img = self.create_publisher(Image, '/detection/image', 1)
        self.pub_victim_alert = self.create_publisher(String, '/disaster/victim_alert', 10)
        self.pub_audio_alert = self.create_publisher(String, '/disaster/audio_alert', 10)

        # Heartbeat Generation Timer (10 Hz = 0.10s)
        hb_period = 1.0 / self.hb_rate
        self.timer_heartbeat = self.create_timer(hb_period, self.publish_heartbeat)

        # Worker thread for Whisper STT
        self.audio_thread_running = True
        self.audio_thread = threading.Thread(target=self.whisper_worker_loop, daemon=True)
        self.audio_thread.start()

        self.get_logger().info("Host AI Pipeline Operational. Transmitting 10Hz Heartbeat.")

    # -----------------------------------------------------------------------------
    # Watchdog Heartbeat Generator
    # -----------------------------------------------------------------------------
    def publish_heartbeat(self):
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = f"host_station_seq_{self.seq_id}"
        self.seq_id += 1
        self.pub_heartbeat.publish(msg)

    # -----------------------------------------------------------------------------
    # Vision Inference Callback
    # -----------------------------------------------------------------------------
    def cb_image(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        if self.yolo_model is None:
            return

        # Perform half-precision inference
        # Classes: 0 is 'person' in standard COCO dataset
        results = self.yolo_model(
            cv_img,
            conf=self.conf_th,
            device=self.device_str,
            half=("cuda" in self.device_str),
            verbose=False
        )

        annotated_frame = cv_img.copy()
        person_detected = False
        highest_conf = 0.0

        for r in results:
            annotated_frame = r.plot()
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0:  # Person
                    person_detected = True
                    if conf > highest_conf:
                        highest_conf = conf

        if person_detected:
            alert = String()
            alert.data = f"SURVIVOR_DETECTED|Confidence:{highest_conf:.2f}|Time:{time.time()}"
            self.pub_victim_alert.publish(alert)
            cv2.putText(
                annotated_frame, "!!! SURVIVOR LOCATED !!!", (30, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2
            )

        # Publish visualized inference stream for RViz2 / Monitoring UI
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.pub_detection_img.publish(out_msg)
        except CvBridgeError:
            pass

    # -----------------------------------------------------------------------------
    # Audio STT Pipeline & Keyword Spotter
    # -----------------------------------------------------------------------------
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
        self.audio_thread_running = False
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
