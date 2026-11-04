"""Unit tests for ADS1115 driver and MQ-2 math (no ROS / hardware required)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'disaster_recovery'))
from ads1115_mq2 import (  # noqa: E402
    ADS1115, EMAFilter, build_config, raw_to_volts,
    adc_to_sensor_volts, sensor_resistance_kohm, danger_metric,
)


class FakeBus:
    def __init__(self, conv_bytes):
        self.conv = conv_bytes
        self.writes = []

    def write_i2c_block_data(self, addr, reg, data):
        self.writes.append((addr, reg, list(data)))

    def read_i2c_block_data(self, addr, reg, n):
        assert reg == 0x00 and n == 2
        return list(self.conv)


def test_config_word_ain0_4v096_128sps():
    # OS=0 | MUX=100 | PGA=001 | MODE=0 | DR=100 | COMP_QUE=11
    assert build_config(0, 4.096, 128) == 0x4283


def test_raw_to_volts():
    assert raw_to_volts(0x7F, 0xFF) == pytest.approx(4.096 * 32767 / 32768)
    assert raw_to_volts(0x00, 0x00) == 0.0
    assert raw_to_volts(0xFF, 0xFF) == pytest.approx(-4.096 / 32768)  # two's complement


def test_driver_writes_config_and_reads():
    # 3.333 V -> raw = 3.333/4.096*32768 = 26664 = 0x6828
    bus = FakeBus([0x68, 0x28])
    adc = ADS1115(bus)
    adc.start()
    assert bus.writes == [(0x48, 0x01, [0x42, 0x83])]
    assert adc.read_volts() == pytest.approx(3.333, abs=1e-3)


def test_divider_inverse():
    assert adc_to_sensor_volts(3.3333) == pytest.approx(5.0, abs=1e-3)
    assert adc_to_sensor_volts(5.0 * 2 / 3) == pytest.approx(5.0)


def test_rs_and_metric_independent_of_rl():
    for rl in (1.0, 4.7, 10.0):
        rs_air = sensor_resistance_kohm(1.0, 5.0, rl)
        rs_gas = sensor_resistance_kohm(2.0, 5.0, rl)
        assert danger_metric(rs_gas, rs_air) == pytest.approx(1 - (3 / 2) / 4)  # 0.625
    assert sensor_resistance_kohm(2.5, 5.0, 10.0) == pytest.approx(10.0)


def test_metric_clamped():
    assert danger_metric(20.0, 10.0) == 0.0   # cleaner than baseline
    assert danger_metric(0.0, 10.0) == 1.0
    assert danger_metric(5.0, 0.0) == 0.0     # invalid baseline


def test_ema():
    f = EMAFilter(0.5)
    assert f.update(10.0) == 10.0
    assert f.update(0.0) == 5.0
    with pytest.raises(ValueError):
        EMAFilter(0.0)


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        build_config(4)
    with pytest.raises(ValueError):
        build_config(0, 3.3)
    assert not math.isnan(raw_to_volts(0x80, 0x00))
