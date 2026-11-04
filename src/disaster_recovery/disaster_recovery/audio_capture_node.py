#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: audio_capture_node.py
Author: Autonomous Systems Laboratory / Modified for Jongwoo Noh

Description:
  Onboard Audio Ingestion Node for Raspberry Pi 4.
  Interfaces with USB Omnidirectional Condenser Lapel Microphone (16 kHz, Mono, 16-bit PCM).
  Streams captured audio packets or simulated transcript tokens to ROS 2 topic /audio/transcription_raw.
=================================================================================================
"""

import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Try importing sounddevice or pyaudio
AUDIO_LIB = None
try:
    import sounddevice as sd
    AUDIO_LIB = 'sounddevice'
except ImportError:
    try:
        import pyaudio
        AUDIO_LIB = 'pyaudio'
    except ImportError:
        AUDIO_LIB = None


class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')

        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('chunk_duration_sec', 1.0)
        self.declare_parameter('device_index', -1)

        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.chunk_duration = self.get_parameter('chunk_duration_sec').value

        self.pub_transcription = self.create_publisher(String, '/audio/transcription_raw', 10)
        self.running = True

        self.get_logger().info(f"Audio Capture Node started. Audio Backend: {AUDIO_LIB}")

        # Capture worker thread
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def capture_loop(self):
        while self.running and rclpy.ok():
            # In a live setup with sounddevice:
            # data = sd.rec(int(self.chunk_duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
            # For network-lean telemetry, audio energy or raw PCM can be processed here.
            time.sleep(self.chunk_duration)

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
