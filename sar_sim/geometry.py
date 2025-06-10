from dataclasses import dataclass
import numpy as np
import math

@dataclass
class SensorParameters:
    carrier_frequency_hz: float      # e.g., 9.6e9 (X-band)
    bandwidth_hz: float              # e.g., 100e6
    pulse_width_s: float             # e.g., 10e-6
    PRF_hz: float                    # e.g., 1000
    sampling_frequency_hz: float     # e.g., 1e9
    sensor_speed_mps: float          # e.g., 7500 (LEO)
    sensor_height_m: float           # e.g., 100000
    grazing_angle_deg: float         # e.g., 40
    swath_m: float                   # Cross-range extent
    range_width_m: float             # Along-track extent
    azimuth_resolution_m: float      # Target resolution
    integration_angle_rad: float     # Derived
    max_capturable_height_m: float = 0.0

    @property
    def wavelength_m(self):
        return 3e8 / self.carrier_frequency_hz

    @property
    def grazing_angle_rad(self):
        return math.radians(self.grazing_angle_deg)



def compute_target_time_delays(sensor_params: SensorParameters, target_position: tuple[float, float, float]) -> np.ndarray:
    """
    Compute the round-trip time-of-flight (ToF) delay for each SAR pulse,
    from a moving sensor to a fixed target position.

    Parameters:
    - sensor_params: SensorParameters
        Radar system and trajectory configuration.
    - target_position: tuple
        (x, y, z) position of the target in metres.

    Returns:
    - np.ndarray: Time-of-flight values for each pulse [in seconds].
    """
    # Extract sensor parameters
    h = sensor_params.sensor_height_m
    v = sensor_params.sensor_speed_mps
    C = 3e8  # Speed of light
    PRI = 1.0 / sensor_params.PRF_hz
    integration_angle = sensor_params.integration_angle_rad
    grazing_angle_rad = sensor_params.grazing_angle_rad

    # Target coordinates
    x_t, y_t, z_t = target_position

    # Ground-projected distance from sensor to centre of swath
    y_c = h / math.tan(grazing_angle_rad)
    delta_max = y_c * math.tan(0.5 * integration_angle)

    num_of_samples = int((2 * abs(delta_max)) / (PRI * v))

    time_result = np.zeros(num_of_samples, dtype=np.float32)

    for i in range(num_of_samples):
        dist_deviation = (v * PRI) * i - delta_max
        slant_range = math.sqrt(
            (h - z_t) ** 2 +
            (dist_deviation - x_t) ** 2 +
            (y_c - y_t) ** 2
        )
        time_result[i] = (2 * slant_range) / C

    return time_result
