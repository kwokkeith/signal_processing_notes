from matplotlib import pyplot as plt
import numpy as np
from sar_sim.chirp import apply_time_delay_frequency_domain, generate_chirp_time_domain, iq_demodulate_chirp, apply_phase_alignment, generate_baseband_chirp_time_domain, apply_sample_shift_alignment
from sar_sim.matched_filter import matched_filter_frequency_domain
from sar_sim.geometry import compute_target_time_delays, SensorParameters


def simulate_raw_data(sensor, target_pos, t_start: float = 0.0, margin_time_s: float = 1e-6):
    """
    Simulates SAR raw data using chirp generation and frequency-domain time shifting.

    Parameters:
    - sensor: SensorParameters with fs, Tp, B, fc.
    - target_pos: list of (x, y, z) tuples for each pulse.
    - t_start: time (s) when sampling begins.
    - margin_time_s: extra time to sample beyond last pulse.

    Returns:
    - raw_data: complex-valued 2D array [pulses x fast_time]
    - time_result: list of delays per pulse
    - t_start: fast-time start time (returned for reference)
    """
    fs = sensor.sampling_frequency_hz
    Tp = sensor.pulse_width_s
    B = sensor.bandwidth_hz
    fc = sensor.carrier_frequency_hz
    k = B / Tp

    # Compute time delays
    time_result = compute_target_time_delays(sensor, target_pos)
    num_pulses = len(time_result)
    max_delay = max(time_result)
    total_time = (max_delay + Tp + margin_time_s) - t_start
    num_fast_time = int(np.ceil(fs * total_time))

    # Generate undelayed baseband chirp from 0 to Tp
    t = np.linspace(0, Tp, int(np.floor(fs * Tp)), endpoint=False)
    t_fast = np.arange(num_fast_time) / fs + t_start
    base_chirp = np.exp(1j * 2 * np.pi * (fc * t + 0.5 * k * (t - Tp / 2)**2))
    plt.figure()
    plt.plot(t/1e-6, np.abs(base_chirp))


    # Precompute chirp FFT for efficient shifting
    chirp_fft = np.fft.fft(base_chirp, n=num_fast_time)
    freqs = np.fft.fftfreq(num_fast_time, d=1/fs)

    # Allocate full data buffer
    raw_data = np.zeros((num_pulses, num_fast_time), dtype=np.complex64)

    for i in range(num_pulses):
        delay = time_result[i]  # shift w.r.t sampling start

        # Phase shift in frequency domain
        # print(delay-t_start)
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * (delay - t_start))
        shifted_chirp_fft = chirp_fft * phase_shift
        shifted_chirp = np.fft.ifft(shifted_chirp_fft)
        shifted_chirp = shifted_chirp * np.exp(-1j * 2 * np.pi * fc * t_fast)  # demodulate to baseband

        insert_start = int(np.round((delay - t_start) * fs))
        insert_end = insert_start + len(base_chirp)
        raw_data[i,:] = shifted_chirp

        # if 0 <= insert_start < num_fast_time and insert_end <= num_fast_time:
        #     raw_data[i, insert_start:insert_end] = shifted_chirp[:insert_end - insert_start]
        # else:
        #     print(f"[WARN] Pulse {i} insertion out of bounds: {insert_start} to {insert_end}")


        if i == 0:
            # Debug: plot time-domain and frequency spectrum
            print(f"delay: {delay}")
            time_axis = np.arange(num_fast_time) / fs
            plt.figure(figsize=(10, 4))
            # plt.plot(time_axis, shifted_chirp.real, label='Real Part')
            # plt.plot(time_axis, shifted_chirp.imag, label='Imaginary Part', linestyle='--')
            plt.plot(time_axis, np.abs(shifted_chirp))
            plt.title("Time-Delayed Chirp (Pulse 0)")
            plt.xlim(4e-5, 5e-5)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            spectrum = np.fft.fftshift(np.abs(np.fft.fft(shifted_chirp)))
            plt.figure(figsize=(10, 4))
            plt.plot(np.fft.fftshift(freqs), 20 * np.log10(spectrum + 1e-6))
            plt.title("Frequency Spectrum of Time-Delayed Chirp (Pulse 0)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.xlim(-B, B)
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    return raw_data, time_result, t_start



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
    sampling_frequency: float,
    chirp_start_time: float = 0.0
) -> np.ndarray:
    """
    Applies range migration correction to the range-compressed SAR data
    by aligning a reference target’s time delay across pulses.

    Parameters:
    - compressed_data: 2D ndarray (azimuth × fast-time)
    - sensor: SensorParameters object
    - target_position: tuple (x, y, z) of actual target
    - sampling_frequency: float, in Hz
    - chirp_start_time: float, unused here but reserved for future extensions

    Returns:
    - np.ndarray: Phase-aligned SAR data (time domain)
    """

    # Compute time delays to actual and reference targets
    time_result_target = compute_target_time_delays(sensor, target_position)
    time_result_ref = compute_target_time_delays(sensor, (0.0, 0.0, 0.0))

    print(f"time_result_target (μs): {time_result_target[:10] * 1e6}")
    print(f"time_result_ref (μs): {time_result_ref[:10] * 1e6}")

    # Compute effective delay difference for alignment
    time_diff = time_result_target - 2 * time_result_ref
    print(f"time_diff (μs): {time_diff[:10] * 1e6}")

    # Frequency axis for FFT across fast-time dimension
    num_fast_time = compressed_data.shape[1]
    dt = 1 / sampling_frequency
    freqs = np.fft.fftfreq(num_fast_time, d=dt)  # Hz

    # Apply phase correction per pulse in frequency domain
    compressed_data_fft = np.fft.fft(compressed_data, axis=1)
    for i in range(compressed_data.shape[0]):
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * time_diff[i])
        compressed_data_fft[i, :] *= phase_shift

    # Convert back to time domain
    aligned_data = np.fft.ifft(compressed_data_fft, axis=1)
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