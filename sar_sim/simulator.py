import numpy as np
from sar_sim.chirp import apply_time_delay_frequency_domain, generate_chirp_time_domain, iq_demodulate_chirp, apply_phase_alignment, generate_baseband_chirp_time_domain, apply_sample_shift_alignment
from sar_sim.matched_filter import matched_filter_frequency_domain
from sar_sim.geometry import compute_target_time_delays, SensorParameters


def simulate_raw_data(sensor: SensorParameters, target_pos: tuple) -> np.ndarray:
    """
    Simulates SAR raw data for a point target using:
    - Sub-sample delay via frequency-domain phase shift
    - Delay buffer to position chirps properly in fast-time

    Returns:
    - np.ndarray: 2D complex array [azimuth × fast-time]
    """
    fs = sensor.sampling_frequency_hz
    chirp = generate_chirp_time_domain(
        sampling_frequency_mhz=fs / 1e6,
        carrier_frequency_mhz=sensor.carrier_frequency_hz / 1e6,
        bandwidth_mhz=sensor.bandwidth_hz / 1e6,
        pulse_width_us=sensor.pulse_width_s * 1e6
    )
    chirp_len = len(chirp)

    # Compute time delays (in seconds) and delay in samples
    time_result = compute_target_time_delays(sensor, target_pos)  # shape: (num_pulses,)
    delay_samples = (time_result * fs)

    # Allocate output buffer with delay margin
    min_delay = int(np.floor(delay_samples.min()))
    max_delay = int(np.ceil(delay_samples.max())) + chirp_len
    fast_time_span = max_delay - min_delay
    num_pulses = len(time_result)

    raw_data = np.zeros((num_pulses, fast_time_span), dtype=np.complex64)

    # Precompute FFT of demodulated chirp
    baseband_chirp = iq_demodulate_chirp(
        chirp,
        carrier_frequency_mhz=sensor.carrier_frequency_hz / 1e6,
        sampling_frequency_mhz=fs / 1e6
    )
    chirp_fft = np.fft.fft(baseband_chirp)
    freqs = np.fft.fftfreq(chirp_len, d=1 / fs)

    # Simulate raw data per pulse
    for i, delay in enumerate(delay_samples):
        fractional_delay = delay - np.floor(delay)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * fractional_delay)
        shifted_chirp = np.fft.ifft(chirp_fft * phase_shift).astype(np.complex64)

        # Place chirp in correct fast-time slot
        shift = int(np.floor(delay)) - min_delay
        raw_data[i, shift:shift + chirp_len] = shifted_chirp

    return raw_data, time_result, min_delay / fs  # include timing info if needed


def simulate_range_compression(
    raw_data: np.ndarray,
    sensor: SensorParameters
) -> np.ndarray:
    """
    Applies matched filtering (range compression) on SAR raw data.

    Parameters:
    - raw_data: 2D ndarray (azimuth × fast-time)
    - sensor: SensorParameters object

    Returns:
    - np.ndarray: Range-compressed SAR data with same fast-time length.
    """
    reference_chirp = generate_baseband_chirp_time_domain(
        sampling_frequency_mhz=sensor.sampling_frequency_hz / 1e6,
        bandwidth_mhz=sensor.bandwidth_hz / 1e6,
        pulse_width_us=sensor.pulse_width_s * 1e6
    )

    n_fast_time = raw_data.shape[1]
    compressed_data = np.zeros_like(raw_data, dtype=np.complex64)

    for i in range(raw_data.shape[0]):
        full_output = matched_filter_frequency_domain(raw_data[i, :], reference_chirp)
        compressed_data[i, :] = full_output[:n_fast_time]  # Truncate to match original length

    return compressed_data


def perform_range_migration(
    compressed_data: np.ndarray,
    sensor: SensorParameters,
    target_position: tuple,
    reference_signal_frequency_hz: float,
    sampling_frequency: float
) -> np.ndarray:
    """
    Applies range migration correction to the range-compressed SAR data
    by aligning a reference target’s time delay across pulses.

    Parameters:
    - compressed_data: 2D ndarray (azimuth × fast-time)
    - sensor: SensorParameters object
    - target_position: tuple (x, y, z) of actual target
    - sampling_frequency: float, in Hz

    Returns:
    - np.ndarray: Phase-aligned SAR data
    """
    # Time delays for actual target
    time_result_target = compute_target_time_delays(sensor, target_position)

    # Time delays for reference target at (0, 0, 0)
    time_result_ref = compute_target_time_delays(sensor, (0.0, 0.0, 0.0))

    # Apply phase alignment (range migration correction)
    aligned_data = apply_phase_alignment(
        data=compressed_data,
        time_result_target=time_result_target,
        time_result_ref=time_result_ref,
        sampling_frequency=sampling_frequency,
        start_time_s= time_result_target.min()
    )
    return aligned_data


def simulate_azimuth_compression(aligned_data: np.ndarray, reference_azimuth: np.ndarray) -> np.ndarray:
    """
    Perform azimuth compression by matched filtering along columns (azimuth lines).

    Parameters:
    - aligned_data: 2D ndarray (azimuth × fast-time), range-compressed and range-aligned
    - reference_azimuth: 1D ndarray, reference Doppler signature from a target (usually at (0, 0, 0))

    Returns:
    - np.ndarray: Fully compressed SAR image (focused in azimuth)
    """
    num_azimuth, num_range_bins = aligned_data.shape
    compressed_image = np.zeros_like(aligned_data, dtype=np.complex64)

    for j in range(num_range_bins):
        signal = aligned_data[:, j]
        compressed_column = matched_filter_frequency_domain(signal, reference_azimuth)
        compressed_image[:, j] = compressed_column[:num_azimuth]  # truncate to match original

    return compressed_image