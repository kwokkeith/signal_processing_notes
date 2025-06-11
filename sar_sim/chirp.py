import numpy as np
import warnings

from scipy import signal



def generate_chirp_time_domain(
    sampling_frequency_mhz: float = 360*2.5,
    carrier_frequency_mhz: float = 360,
    bandwidth_mhz: float = 100,
    pulse_width_us: float = 10,
    dtype=np.complex64
) -> np.ndarray:
    """"
    Generates a baseband complex chirp signal for SAR in the time domain.

    Parameters:
    - sampling_frequency_mhz: float, in MHz
    - carrier_frequency_mhz: float, in MHz
    - bandwidth_mhz: float, in MHz
    - pulse_width_us: float, in microseconds
    - dtype: np.dtype, optional (default: complex64)

    Returns:
    - np.ndarray: complex chirp signal in time domain
    """
    # Check if sampling frequency meets Nyquist criterion (2.0 times for theoretical Nyquist, 2.5 times for practical applications)
    if sampling_frequency_mhz < 2.5 * (carrier_frequency_mhz + bandwidth_mhz/2):
        warnings.warn("Sampling frequency is not 2.5 times the maximum frequency of the chirp (below Nyquist Sampling Rate).")

    fs = sampling_frequency_mhz * 1e6  # Hz
    fc = carrier_frequency_mhz * 1e6   # Hz
    bw = bandwidth_mhz * 1e6           # Hz
    Tp = pulse_width_us * 1e-6         # s

    num_samples = int(np.floor(fs * Tp))
    t = np.linspace(-Tp / 2, Tp / 2, num_samples, endpoint=False)
    k = bw / Tp  # chirp rate in Hz/s

    phase = 2 * np.pi * (fc * t + 0.5 * k * t**2)
    chirp = np.exp(1j * phase).astype(dtype)

    return chirp


def apply_time_delay_frequency_domain(
    chirp: np.ndarray,
    time_delay_us: float,
    sampling_frequency_mhz: float = 360*2.5,
    dtype=np.complex64
) -> np.ndarray:
    """
    Applies a time delay (in microseconds) to a chirp using frequency domain phase shift.

    Parameters:
    - chirp: np.ndarray, complex chirp signal
    - time_delay_us: float, time delay in microseconds
    - sampling_frequency_mhz: float, sampling rate in MHz
    - dtype: np.dtype, optional (default: complex64)

    Returns:
    - np.ndarray: delayed chirp in time domain
    """
    time_delay = time_delay_us * 1e-6  # seconds
    fs = sampling_frequency_mhz * 1e6  # Hz
    n = chirp.shape[0]
    freqs = np.fft.fftfreq(n, d=1 / fs).astype(np.float32)

    chirp_fft = np.fft.fft(chirp)
    phase_shift = np.exp(-2j * np.pi * freqs * time_delay).astype(dtype)
    shifted_fft = chirp_fft * phase_shift

    return np.fft.ifft(shifted_fft).astype(dtype)


def iq_demodulate_chirp(
    chirp: np.ndarray,
    carrier_frequency_mhz: float = 360,
    sampling_frequency_mhz: float = 360*2.5,
    dtype=np.complex64
) -> np.ndarray:
    """
    Performs IQ demodulation on a passband chirp signal.

    Parameters:
    - chirp: np.ndarray
        Real or complex passband chirp signal (in time domain).
    - carrier_frequency_mhz: float
        Carrier frequency in MHz.
    - sampling_frequency_mhz: float
        Sampling frequency in MHz.
    - dtype: np.dtype, optional
        Output data type, default np.complex64

    Returns:
    - np.ndarray: Demodulated baseband chirp (complex)
    """
    fc = carrier_frequency_mhz * 1e6
    fs = sampling_frequency_mhz * 1e6
    n = chirp.shape[0]

    t = np.arange(n, dtype=np.float32) / fs
    demodulator = np.exp(-1j * 2 * np.pi * fc * t).astype(dtype)

    # If chirp is real, ensure it's cast to complex
    chirp_complex = chirp.astype(dtype, copy=False)
    
    baseband = chirp_complex * demodulator
    return baseband


def apply_phase_alignment(
    data: np.ndarray,
    time_result_target: np.ndarray,
    time_result_ref: np.ndarray,
    sampling_frequency: float,
    start_time_s: float = 0.0
) -> np.ndarray:
    """
    Applies range migration (phase-based) alignment to SAR data using time delay difference,
    corrected by the start time of the sampled window.

    Parameters:
    - data: np.ndarray
        Complex SAR data (azimuth × fast time).
    - time_result_target: np.ndarray
        1D array of time delays for the target (in seconds), one per azimuth line.
    - time_result_ref: np.ndarray
        1D array of reference time delays (in seconds), same length as target.
    - sampling_frequency: float
        Sampling frequency in Hz.
    - start_time_s: float
        Start time of the fast-time window (in seconds).

    Returns:
    - np.ndarray: Phase-aligned SAR data.
    """
    num_azimuth, num_fast_time = data.shape
    if len(time_result_target) != num_azimuth or len(time_result_ref) != num_azimuth:
        raise ValueError("Time delay arrays must match number of azimuth lines.")

    # Frequency axis
    freqs = np.fft.fftfreq(num_fast_time, d=1 / sampling_frequency).astype(np.float32)
    
    # Align delays to fast-time window
    delta_t = time_result_target - 2 * time_result_ref + start_time_s  # corrected difference

    # FFT along fast-time axis
    data_fft = np.fft.fft(data, axis=1)

    # Apply per-azimuth phase shift
    for i in range(num_azimuth):
        phase_shift = np.exp(-1j * 2 * np.pi * freqs * delta_t[i])
        data_fft[i, :] *= phase_shift.astype(data_fft.dtype)

    # IFFT back to time domain
    aligned_data = np.fft.ifft(data_fft, axis=1)
    return aligned_data



def apply_sample_shift_alignment(
    data: np.ndarray,
    time_result_target: np.ndarray,
    time_result_ref: np.ndarray,
    sampling_frequency: float
) -> np.ndarray:
    """
    Applies sample-shift-based range migration.

    Parameters:
    - data: np.ndarray [azimuth × fast time]
    - time_result_target: np.ndarray (s)
    - time_result_ref: np.ndarray (s)
    - sampling_frequency: float (Hz)

    Returns:
    - np.ndarray: Shifted SAR data
    """
    num_azimuth = data.shape[0]
    if len(time_result_target) != num_azimuth or len(time_result_ref) != num_azimuth:
        raise ValueError("Time delay arrays must match number of azimuth lines.")

    delta_samples = np.round((time_result_target - 2 * time_result_ref) * sampling_frequency).astype(int)
    aligned = np.zeros_like(data, dtype=data.dtype)

    for i in range(num_azimuth):
        aligned[i] = np.roll(data[i], -delta_samples[i])  # left shift = earlier arrival

    return aligned




def generate_baseband_chirp_time_domain(
    sampling_frequency_mhz: float,
    bandwidth_mhz: float,
    pulse_width_us: float,
    dtype=np.complex64
) -> np.ndarray:
    """
    Generate a complex baseband chirp signal in the time domain.

    This simulates an LFM chirp with a symmetric spectrum around 0 Hz,
    commonly used for SAR baseband simulation and matched filtering.

    Parameters:
    - sampling_frequency_mhz: float
        Sampling frequency in MHz.
    - bandwidth_mhz: float
        Chirp bandwidth in MHz.
    - pulse_width_us: float
        Chirp duration in microseconds.
    - dtype: np.dtype, optional
        Output data type, default is np.complex64.

    Returns:
    - np.ndarray: Complex-valued baseband chirp signal.
    """
    fs = sampling_frequency_mhz * 1e6     # Convert MHz → Hz
    bw = bandwidth_mhz * 1e6              # Convert MHz → Hz
    Tp = pulse_width_us * 1e-6            # Convert µs → s

    num_samples = int(np.floor(fs * Tp))
    t = np.linspace(-Tp / 2, Tp / 2, num_samples, endpoint=False)

    k = bw / Tp                           # Chirp rate in Hz/s
    phase = np.pi * k * t**2              # Symmetric baseband LFM chirp
    chirp = np.exp(1j * phase).astype(dtype)

    return chirp
