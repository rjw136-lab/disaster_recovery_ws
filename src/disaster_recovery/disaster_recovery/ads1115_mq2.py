#!/usr/bin/env python3
"""
ADS1115 I2C ADC driver + MQ-2 signal processing (ROS-independent).

Hardware chain:
    MQ-2 module AOUT (0~5 V) -> 10k/20k divider (x 2/3, max 3.33 V)
    -> ADS1115 AIN0 (VDD = 3.3 V from RPi pin 1) -> I2C bus 1 (0x48) -> Raspberry Pi 4

Kept free of rclpy so that it can be unit-tested on any PC.
"""

# ---------------------------------------------------------------------------
# ADS1115 register map (TI ADS1115 datasheet, SBAS444)
# ---------------------------------------------------------------------------
REG_CONVERSION = 0x00
REG_CONFIG = 0x01

# MUX[14:12]: single-ended inputs vs GND
MUX_SINGLE = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}

# PGA[11:9]: full-scale range (V) -> bits
PGA_BITS = {
    6.144: 0x0000,
    4.096: 0x0200,
    2.048: 0x0400,
    1.024: 0x0600,
    0.512: 0x0800,
    0.256: 0x0A00,
}

MODE_CONTINUOUS = 0x0000  # bit 8 = 0
MODE_SINGLE_SHOT = 0x0100  # bit 8 = 1

# DR[7:5]: samples per second -> bits
DR_BITS = {8: 0x0000, 16: 0x0020, 32: 0x0040, 64: 0x0060,
           128: 0x0080, 250: 0x00A0, 475: 0x00C0, 860: 0x00E0}

COMP_QUE_DISABLE = 0x0003  # bits 1:0 = 11 -> comparator disabled


def build_config(channel: int = 0, fsr: float = 4.096, sps: int = 128) -> int:
    """Return 16-bit config word for continuous single-ended conversion."""
    if channel not in MUX_SINGLE:
        raise ValueError(f"channel must be 0..3, got {channel}")
    if fsr not in PGA_BITS:
        raise ValueError(f"fsr must be one of {sorted(PGA_BITS)}, got {fsr}")
    if sps not in DR_BITS:
        raise ValueError(f"sps must be one of {sorted(DR_BITS)}, got {sps}")
    return (MUX_SINGLE[channel] | PGA_BITS[fsr] | MODE_CONTINUOUS
            | DR_BITS[sps] | COMP_QUE_DISABLE)


def raw_to_volts(msb: int, lsb: int, fsr: float = 4.096) -> float:
    """Convert big-endian two's-complement conversion register to volts."""
    raw = (msb << 8) | lsb
    if raw & 0x8000:
        raw -= 1 << 16
    return raw * fsr / 32768.0


class ADS1115:
    """Minimal ADS1115 driver using an smbus/smbus2-compatible bus object."""

    def __init__(self, bus, address: int = 0x48, channel: int = 0,
                 fsr: float = 4.096, sps: int = 128):
        self.bus = bus
        self.address = address
        self.fsr = fsr
        self.config = build_config(channel, fsr, sps)

    def start(self):
        self.bus.write_i2c_block_data(
            self.address, REG_CONFIG,
            [(self.config >> 8) & 0xFF, self.config & 0xFF])

    def read_volts(self) -> float:
        msb, lsb = self.bus.read_i2c_block_data(self.address, REG_CONVERSION, 2)
        return raw_to_volts(msb, lsb, self.fsr)


# ---------------------------------------------------------------------------
# MQ-2 processing
# ---------------------------------------------------------------------------
MQ2_CLEAN_AIR_RS_RO = 9.83  # Rs/Ro in clean air (Hanwei MQ-2 datasheet curve)


class EMAFilter:
    """y[k] = a * x[k] + (1 - a) * y[k-1]"""

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else \
            self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


def adc_to_sensor_volts(v_adc: float, r1_kohm: float = 10.0,
                        r2_kohm: float = 20.0) -> float:
    """Undo the R1 (series) / R2 (to GND) divider."""
    return v_adc * (r1_kohm + r2_kohm) / r2_kohm


def sensor_resistance_kohm(v_out: float, vcc: float = 5.0,
                           rl_kohm: float = 10.0) -> float:
    """
    MQ-2 module: VCC -> Rs -> (AOUT) -> RL -> GND
    Rs = (VCC - Vout) * RL / Vout
    NOTE: RL cancels out in Rs/Rs_air, so the danger metric does not depend on RL.
    """
    v_out = min(max(v_out, 1e-3), vcc)
    return (vcc - v_out) * rl_kohm / v_out


def danger_metric(rs: float, rs_air: float) -> float:
    """
    Relative resistance drop vs. clean-air baseline, clamped to [0, 1].
    0.0 = clean-air level, 0.4 = Rs dropped by 40 %.
    This is NOT a ppm value; thresholds must be set empirically.
    """
    if rs_air <= 0.0:
        return 0.0
    return min(max(1.0 - rs / rs_air, 0.0), 1.0)
