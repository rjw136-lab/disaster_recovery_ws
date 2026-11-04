#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: gas_sensor_node.py

Description:
  MQ-2 gas sensor node for Raspberry Pi 4 via ADS1115 (I2C).
  Replaces the former OpenCR-firmware approach, which would have overwritten the stock
  TurtleBot3 OpenCR firmware (motors / odometry) and collided on /dev/ttyACM0.

Pipeline (20 Hz):
  ADS1115 AIN0 voltage -> divider inverse (x1.5) -> EMA -> Rs -> danger metric
  1. Warm-up phase (warmup_sec): no /gas/level is published (prevents false triggers).
  2. Baseline phase: averages Rs over baseline_samples in clean air -> Rs_air.
     (Or set rs_air_kohm > 0 to reuse a previously measured baseline.)
  3. Running phase: publishes
       /gas/level   (Float32) danger metric 0.0~1.0 = 1 - Rs/Rs_air
       /gas/voltage (Float32) reconstructed MQ-2 AOUT voltage
       /gas/rs_ro   (Float32) Rs/Ro (Ro = Rs_air / 9.83) for datasheet ppm curves
       /gas/sensor_ok (Bool)  False when I2C reads fail
=================================================================================================
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

from disaster_recovery.ads1115_mq2 import (
    ADS1115, EMAFilter, adc_to_sensor_volts, sensor_resistance_kohm,
    danger_metric, MQ2_CLEAN_AIR_RS_RO,
)

try:
    from smbus2 import SMBus
except ImportError:
    from smbus import SMBus  # apt: python3-smbus


class GasSensorNode(Node):
    def __init__(self):
        super().__init__('gas_sensor_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_address', 0x48)
        self.declare_parameter('adc_channel', 0)
        self.declare_parameter('adc_fsr_v', 4.096)
        self.declare_parameter('sample_rate_hz', 20.0)
        self.declare_parameter('ema_alpha', 0.15)
        self.declare_parameter('divider_r1_kohm', 10.0)
        self.declare_parameter('divider_r2_kohm', 20.0)
        self.declare_parameter('sensor_vcc', 5.0)
        self.declare_parameter('load_resistor_kohm', 10.0)
        self.declare_parameter('warmup_sec', 60.0)
        self.declare_parameter('baseline_samples', 100)
        self.declare_parameter('rs_air_kohm', 0.0)  # >0: skip baseline measurement
        self.declare_parameter('max_consecutive_errors', 10)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.r1 = gp('divider_r1_kohm')
        self.r2 = gp('divider_r2_kohm')
        self.vcc = gp('sensor_vcc')
        self.rl = gp('load_resistor_kohm')
        self.warmup_sec = gp('warmup_sec')
        self.baseline_n = int(gp('baseline_samples'))
        self.max_err = int(gp('max_consecutive_errors'))
        self.ema = EMAFilter(gp('ema_alpha'))

        self.bus_id = int(gp('i2c_bus'))
        self.adc_args = dict(address=int(gp('i2c_address')),
                             channel=int(gp('adc_channel')),
                             fsr=float(gp('adc_fsr_v')))
        self.bus = None
        self.adc = None
        self.err_count = 0
        self.open_adc()

        rs_air = float(gp('rs_air_kohm'))
        self.rs_air = rs_air if rs_air > 0.0 else None
        self.baseline_acc = []
        self.t_start = self.get_clock().now()

        self.pub_level = self.create_publisher(Float32, '/gas/level', 10)
        self.pub_volt = self.create_publisher(Float32, '/gas/voltage', 10)
        self.pub_rs_ro = self.create_publisher(Float32, '/gas/rs_ro', 10)
        self.pub_ok = self.create_publisher(Bool, '/gas/sensor_ok', 10)

        self.timer = self.create_timer(1.0 / gp('sample_rate_hz'), self.tick)
        self.get_logger().info(
            f"MQ-2 via ADS1115 @ i2c-{self.bus_id}/0x{self.adc_args['address']:02X}. "
            f"Warm-up {self.warmup_sec:.0f}s"
            + ("" if self.rs_air is None else f", fixed Rs_air={self.rs_air:.2f}kΩ"))

    def open_adc(self):
        try:
            if self.bus is not None:
                self.bus.close()
            self.bus = SMBus(self.bus_id)
            self.adc = ADS1115(self.bus, **self.adc_args)
            self.adc.start()
            return True
        except OSError as e:
            self.get_logger().error(f"ADS1115 open failed: {e}", throttle_duration_sec=5.0)
            self.adc = None
            return False

    def publish_ok(self, ok: bool):
        self.pub_ok.publish(Bool(data=ok))

    def tick(self):
        if self.adc is None and not self.open_adc():
            self.publish_ok(False)
            return

        try:
            v_adc = self.adc.read_volts()
            self.err_count = 0
        except OSError as e:
            self.err_count += 1
            self.get_logger().warn(f"I2C read error ({self.err_count}): {e}",
                                   throttle_duration_sec=2.0)
            if self.err_count >= self.max_err:
                self.adc = None  # force re-open next tick
            self.publish_ok(False)
            return

        self.publish_ok(True)
        v_out = adc_to_sensor_volts(self.ema.update(v_adc), self.r1, self.r2)
        rs = sensor_resistance_kohm(v_out, self.vcc, self.rl)
        self.pub_volt.publish(Float32(data=float(v_out)))

        elapsed = (self.get_clock().now() - self.t_start).nanoseconds / 1e9
        if elapsed < self.warmup_sec:
            return

        if self.rs_air is None:
            self.baseline_acc.append(rs)
            if len(self.baseline_acc) >= self.baseline_n:
                self.rs_air = sum(self.baseline_acc) / len(self.baseline_acc)
                self.get_logger().info(
                    f"Baseline done: Rs_air={self.rs_air:.2f}kΩ "
                    f"(set rs_air_kohm:={self.rs_air:.2f} to reuse)")
            return

        level = danger_metric(rs, self.rs_air)
        ro = self.rs_air / MQ2_CLEAN_AIR_RS_RO
        self.pub_level.publish(Float32(data=float(level)))
        self.pub_rs_ro.publish(Float32(data=float(rs / ro)))

    def destroy_node(self):
        if self.bus is not None:
            self.bus.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GasSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
