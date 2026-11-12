#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: audio_capture_node.py
Author: Autonomous Systems Laboratory / Modified for Jong Woo Ro

Description:
  Onboard Audio Ingestion Node for Raspberry Pi 4.
  Interfaces with Generic AB13X USB Audio Microphone (Card 1, Device 0: plughw:1,0).
  Hardware Specs:
    - Format: S16_LE (Signed 16-bit PCM)
    - Channels: 1 (Mono)
    - Sample Rate: 48,000 Hz
  Streams captured raw PCM byte packets to ROS 2 topic /audio/raw (std_msgs/UInt8MultiArray).
=================================================================================================
"""

import sys
import time
import subprocess
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import UInt8MultiArray


class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')

        # -------------------------------------------------------------------------
        # Parameters Configuration
        # -------------------------------------------------------------------------
        self.declare_parameter('alsa_device', 'plughw:1,0')
        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('chunk_duration_sec', 0.5)
        self.declare_parameter('audio_topic', '/audio/raw')

        self.alsa_device = self.get_parameter('alsa_device').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.chunk_duration = self.get_parameter('chunk_duration_sec').value
        self.audio_topic = self.get_parameter('audio_topic').value

        # Bytes per chunk calculation: sample_rate * 2 bytes (16-bit) * channels * duration
        self.bytes_per_sample = 2
        self.chunk_bytes = int(self.sample_rate * self.bytes_per_sample * self.channels * self.chunk_duration)

        self.pub_audio_raw = self.create_publisher(UInt8MultiArray, self.audio_topic, 10)

        self.running = True
        self.process = None
        self.packet_count = 0

        self.get_logger().info(
            f"Starting Audio Capture: Device={self.alsa_device}, "
            f"Rate={self.sample_rate}Hz, Channels={self.channels}, Chunk={self.chunk_duration}s ({self.chunk_bytes} bytes)"
        )

        # Background capture thread
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def capture_loop(self):
        cmd = [
            'arecord',
            '-D', self.alsa_device,
            '-f', 'S16_LE',
            '-r', str(self.sample_rate),
            '-c', str(self.channels),
            '-t', 'raw',
            '-q'
        ]

        while self.running and rclpy.ok():
            try:
                self.get_logger().info(f"Opening ALSA stream: {' '.join(cmd)}")
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                while self.running and rclpy.ok():
                    raw_data = self.process.stdout.read(self.chunk_bytes)
                    if not raw_data or len(raw_data) < self.chunk_bytes:
                        if self.process.poll() is not None:
                            err = self.process.stderr.read().decode(errors='ignore')
                            self.get_logger().warn(f"arecord process terminated. Error: {err}")
                            break
                        continue

                    # Publish raw PCM chunk
                    msg = UInt8MultiArray()
                    msg.data = list(raw_data)
                    self.pub_audio_raw.publish(msg)
                    self.packet_count += 1

                    if self.packet_count % 10 == 0:
                        # Calculate RMS for diagnostic logging
                        int16_arr = np.frombuffer(raw_data, dtype=np.int16)
                        rms = np.sqrt(np.mean(int16_arr.astype(np.float32) ** 2))
                        self.get_logger().info(
                            f"[Mic Streaming] Transmitted {self.packet_count} chunks to {self.audio_topic} (RMS: {rms:.1f})"
                        )

            except Exception as e:
                self.get_logger().error(f"Audio capture error: {e}")
                time.sleep(1.0)
            finally:
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait()
                    self.process = None

            if self.running and rclpy.ok():
                self.get_logger().info("Re-attempting audio capture in 1.0 second...")
                time.sleep(1.0)

    def destroy_node(self):
        self.running = False
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
